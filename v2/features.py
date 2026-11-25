"""Shared, order-invariant feature construction for training and serving."""

from difflib import SequenceMatcher
from functools import lru_cache

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

TOKEN_PATTERN = r"(?u)\b\w+\b"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
BASIC_COLUMNS = [
    "char_min", "char_max", "word_min", "word_max", "shared_words",
    "unique_word_total", "word_share", "word_count_difference", "word_count_mean",
]
ADVANCED_COLUMNS = [
    "content_overlap_min", "content_overlap_max", "stop_overlap_min",
    "stop_overlap_max", "token_overlap_min", "token_overlap_max", "first_word_equal",
    "last_word_equal", "longest_substring_ratio", "fuzz_ratio", "fuzz_partial_ratio",
    "token_sort_ratio", "token_set_ratio",
]


def overlap(common, n1, n2):
    return common / max(1, min(n1, n2)), common / max(1, max(n1, n2))


def handcrafted_row(a, b, advanced):
    t1, t2 = a.split(), b.split()
    s1, s2 = set(t1), set(t2)
    common = s1 & s2
    basic = [
        min(len(a), len(b)), max(len(a), len(b)), min(len(t1), len(t2)),
        max(len(t1), len(t2)), len(common), len(s1) + len(s2),
        len(common) / max(1, len(s1) + len(s2)), abs(len(t1) - len(t2)),
        (len(t1) + len(t2)) / 2,
    ]
    if not advanced:
        return basic
    stop1, stop2 = s1 & ENGLISH_STOP_WORDS, s2 & ENGLISH_STOP_WORDS
    stop_common = len(stop1 & stop2)
    longest = SequenceMatcher(None, a, b, autojunk=False).find_longest_match().size
    return basic + [
        *overlap(len(common) - stop_common, len(s1) - len(stop1), len(s2) - len(stop2)),
        *overlap(stop_common, len(stop1), len(stop2)),
        *overlap(len(common), len(t1), len(t2)),
        bool(t1 and t2 and t1[0] == t2[0]), bool(t1 and t2 and t1[-1] == t2[-1]),
        longest / max(1, min(len(a), len(b))), fuzz.QRatio(a, b),
        fuzz.partial_ratio(a, b), fuzz.token_sort_ratio(a, b), fuzz.token_set_ratio(a, b),
    ]


def fit_context(train, needed, embedding, seed):
    corpus = sorted(set(train.clean1) | set(train.clean2))
    context = {}
    if "bow" in needed:
        context["bow"] = CountVectorizer(max_features=3_000, token_pattern=TOKEN_PATTERN).fit(corpus)
    if "tfidf" in needed:
        context["tfidf"] = (
            TfidfVectorizer(ngram_range=(1, 2), token_pattern=TOKEN_PATTERN,
                            max_features=100_000, sublinear_tf=True, dtype=np.float32).fit(corpus),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), max_features=100_000,
                            sublinear_tf=True, dtype=np.float32).fit(corpus),
        )
    if "embedding" in needed:
        if embedding == "lsa":
            word = context["tfidf"][0]
            matrix = word.transform(corpus)
            components = min(64, matrix.shape[0] - 1, matrix.shape[1] - 1)
            if components < 1:
                raise ValueError("LSA requires at least two training questions and vocabulary terms.")
            context["embedding"] = ("lsa", word, TruncatedSVD(
                n_components=components, random_state=seed).fit(matrix))
        else:
            context["embedding"] = ("minilm", MINILM_MODEL, MINILM_REVISION)
            minilm_encoder(MINILM_MODEL, MINILM_REVISION)
    return context


def row_cosine(left, right):
    return np.asarray(left.multiply(right).sum(axis=1)).ravel()


@lru_cache(maxsize=1)
def minilm_encoder(name, revision):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install requirements-minilm.txt to use MiniLM.") from exc
    return SentenceTransformer(name, revision=revision, device="cpu", trust_remote_code=False)


def embedding_similarity(left, right, spec):
    kind, first, second = spec
    if kind == "lsa":
        a = normalize(second.transform(first.transform(left)))
        b = normalize(second.transform(first.transform(right)))
    else:
        encoder = minilm_encoder(first, second)
        codes, unique = pd.factorize(np.asarray(left + right, dtype=object))
        vectors = encoder.encode(unique.tolist(), batch_size=64, normalize_embeddings=True,
                                 show_progress_bar=False)
        a, b = np.split(vectors[codes], 2)
    return np.einsum("ij,ij->i", a, b).clip(-1, 1)


def build_features(clean1, clean2, context, needed):
    """Inputs are preprocessed. Canonical ordering also covers directional fuzzy scorers."""
    pairs = [(min(a, b), max(a, b)) for a, b in zip(clean1, clean2, strict=True)]
    left, right = map(list, zip(*pairs))
    blocks = {}
    if "basic" in needed or "hand" in needed:
        advanced = "hand" in needed
        columns = BASIC_COLUMNS + ADVANCED_COLUMNS if advanced else BASIC_COLUMNS
        hand = pd.DataFrame((handcrafted_row(a, b, advanced) for a, b in pairs),
                            columns=columns, dtype=np.float32)
        if advanced:
            blocks["hand"] = hand
        if "basic" in needed:
            blocks["basic"] = hand[BASIC_COLUMNS]
    if "bow" in needed:
        bow = context["bow"]
        similarity = row_cosine(normalize(bow.transform(left)), normalize(bow.transform(right)))
        blocks["bow"] = pd.DataFrame({"bow_cosine": similarity}, dtype=np.float32)
    if "tfidf" in needed:
        word, char = context["tfidf"]
        blocks["tfidf"] = pd.DataFrame({
            "word_tfidf_cosine": row_cosine(word.transform(left), word.transform(right)),
            "char_tfidf_cosine": row_cosine(char.transform(left), char.transform(right)),
        }, dtype=np.float32)
    if "embedding" in needed:
        blocks["embedding"] = pd.DataFrame({
            "sentence_cosine": embedding_similarity(left, right, context["embedding"]),
        }, dtype=np.float32)
    return blocks


def feature_matrix(blocks, selected):
    return pd.concat([blocks[name] for name in selected], axis=1)
