from pathlib import Path
from io import BytesIO

import pytest
import streamlit as st

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


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_upload_ui_exports(tmp_path, monkeypatch, suffix):
    models = tmp_path / "models"
    fit_bundle(synthetic(60), models, k=3, with_umap=False)
    unseen = synthetic(12, 21)
    stream = BytesIO()
    if suffix == "csv":
        stream.write(unseen.write_csv().encode())
    else:
        unseen.write_parquet(stream)
    stream.name = f"unseen.{suffix}"
    stream.size = len(stream.getvalue())

    def upload(*args, **kwargs):
        stream.seek(0)
        return stream

    monkeypatch.setattr(st, "file_uploader", upload)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    app.sidebar.text_input[0].set_value(str(models)).run()
    app.sidebar.radio[0].set_value("Predict uploaded trips").run()
    assert not app.exception
    assert app.dataframe[0].value.shape[0] == 12
    assert any(
        element.label == "Download all accepted predictions"
        for element in app.get("download_button")
    )
