import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
from sklearn.dummy import DummyClassifier
from streamlit.testing.v1 import AppTest

from v2.inference import ARTIFACT_VERSION


class StreamlitTests(unittest.TestCase):
    def test_missing_artifact_and_comparison_form(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            with patch.dict(os.environ, {"QUORA_MODEL_PATH": str(path)}):
                app = AppTest.from_file("v2/app_v2.py").run()
                self.assertIn("Train a model", app.warning[0].value)
                self.assertFalse(app.exception)

                model = DummyClassifier(strategy="prior").fit([[0], [1], [2]], [0, 1, 1])
                joblib.dump({
                    "version": ARTIFACT_VERSION, "experiment": "Fixture", "threshold": 0.5,
                    "model": model, "blocks": ("hand",), "context": {},
                }, path)

                app.run()
                app.button[0].click().run()
                self.assertIn("at least one word", app.warning[0].value)

                app.text_area[0].set_value("Where is the railway station?")
                app.text_area[1].set_value("How do I find the train station?")
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.subheader[0].value, "Duplicate")
                self.assertEqual(app.metric[0].value, "66.67%")


if __name__ == "__main__":
    unittest.main()
