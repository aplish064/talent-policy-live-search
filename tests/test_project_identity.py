from pathlib import Path
import tomllib


def test_project_metadata_identifies_policy_search() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "talent-policy-live-search"
    assert "policy" in pyproject["project"]["description"]
    assert "activity" not in pyproject["project"]["description"]


def test_server_and_frontend_identify_policy_search() -> None:
    from talent_policy_search.server import app

    index_html = Path("src/talent_policy_search/static/index.html").read_text(encoding="utf-8")

    assert app.title == "Stateless Talent Policy Search"
    assert "人才政策实时搜索" in index_html
    assert "人才活动实时搜索" not in index_html
