from pathlib import Path

from streamlit.testing.v1 import AppTest

from taxi.fixture import synthetic
from taxi.model import fit_bundle


def test_app_sections(tmp_path):
    models = tmp_path / "models"
    fit_bundle(synthetic(60), models, k=3, with_umap=False)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30)
    app.run()
    app.sidebar.text_input[0].set_value(str(models)).run()
    assert not app.exception
    for section in ["Discovered clusters", "Anomalies", "Predict uploaded trips"]:
        app.sidebar.radio[0].set_value(section).run()
        assert not app.exception
