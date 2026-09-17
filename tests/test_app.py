from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_run_and_downloads():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=120)
    app.run()
    assert not app.exception
    app.slider[0].set_value(30)
    app.slider[1].set_value(15)
    app.button[0].click().run(timeout=120)
    assert not app.exception
    assert len(app.metric) == 4
    assert len(app.tabs) == 4
    assert len(app.get("download_button")) == 3
