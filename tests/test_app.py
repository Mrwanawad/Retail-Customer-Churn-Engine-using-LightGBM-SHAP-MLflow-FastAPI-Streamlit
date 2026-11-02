import pytest

from churn.config import ROOT

APP = ROOT / "app" / "streamlit_app.py"
pytestmark = pytest.mark.skipif(not (ROOT / "app" / "data" / "churn_scores.parquet").exists(), reason="needs exported app data")


@pytest.mark.parametrize("page", ["overview", "scorer", "what-if", "campaign", "model-card"])
def test_page_renders_without_exceptions(page, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CHURN_APP_START_PAGE", page)
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.title, "page rendered no title"
