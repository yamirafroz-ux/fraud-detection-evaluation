from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_prepares_replay_and_downloads():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=120)
    app.run()
    assert not app.exception
    assert len(app.get("iframe")) == 1
    assert len(app.get("download_button")) == 3
    app.selectbox[0].set_value(36)
    app.button[0].click().run(timeout=120)
    assert not app.exception
    assert app.session_state["replay_world"][0].customers == 36
