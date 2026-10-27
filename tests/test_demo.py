"""Check the live demo uses the saved backend and produces a real prediction."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


@pytest.mark.skipif(not Path("artifacts/metadata.json").exists(), reason="Train artifacts first")
def test_demo_form_predicts_with_saved_models():
    app = AppTest.from_file(Path("app.py").resolve(), default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    values = [metric.value for metric in app.metric]
    assert values[0] == "89.64 / 100"
    assert len(values) == 3
