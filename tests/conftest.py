from pathlib import Path

import pytest

from talent_policy_search.sources import SourceRegistry


@pytest.fixture
def registry() -> SourceRegistry:
    return SourceRegistry.from_yaml(Path("config/official_sources.yaml"))
