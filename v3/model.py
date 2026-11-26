"""One shared Keras question encoder for training and deployment."""

from tensorflow import keras

from v3.losses import TripletMarginLoss


def build_encoder(config, vocabulary_size):
    tokens = keras.Input(shape=(None,), dtype="int32", name="tokens")
    x = keras.layers.Embedding(vocabulary_size, config.embedding_dim, mask_zero=True)(tokens)
    x = keras.layers.LSTM(config.lstm_units)(x)
    x = keras.layers.Dropout(config.dropout)(x)
    x = keras.layers.Dense(config.projection_dim, use_bias=False)(x)
    embeddings = keras.layers.UnitNormalization(axis=-1, name="sentence_embedding")(x)
    return keras.Model(tokens, embeddings, name="question_encoder")


def build_trainer(encoder, config):
    inputs = tuple(
        keras.Input(shape=(None,), dtype="int32", name=name)
        for name in ("anchor", "positive", "negative")
    )
    anchor, positive, negative = (encoder(tokens) for tokens in inputs)
    similarity = keras.layers.Dot(axes=1)
    positive_similarity = similarity([anchor, positive])
    negative_similarity = similarity([anchor, negative])
    gap = keras.layers.Subtract()([negative_similarity, positive_similarity])
    model = keras.Model(inputs, gap, name="triplet_trainer")
    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        loss=TripletMarginLoss(config.margin),
    )
    return model
