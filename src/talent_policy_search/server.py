from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from talent_policy_search.models import SearchRequest, SearchResponse
from talent_policy_search.pipeline import SearchPipeline
from talent_policy_search.sources import SourceRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = Path(__file__).resolve().parent / "static"
SOURCE_CONFIG = PROJECT_ROOT / "config" / "official_sources.yaml"
PACKAGE_PROJECT_ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(title="Stateless Talent Policy Search")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def get_pipeline() -> SearchPipeline:
    """Cache the app dependency object only; search results are never cached."""
    registry = SourceRegistry.from_yaml(_source_config_path())
    return SearchPipeline(registry=registry)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    pipeline: SearchPipeline = Depends(get_pipeline),
) -> SearchResponse:
    return await pipeline.search(request.query)


def _source_config_path() -> Path:
    if SOURCE_CONFIG.exists():
        return SOURCE_CONFIG
    return PACKAGE_PROJECT_ROOT / "config" / "official_sources.yaml"
