from __future__ import annotations

from pathlib import Path

import yaml

from talent_policy_search.models import OfficialEntity, SourceConfig


class SourceRegistry:
    def __init__(self, config: SourceConfig):
        self.config = config
        self.entities = config.entities
        self.jurisdictions = config.jurisdictions

    @classmethod
    def from_yaml(cls, path: Path) -> "SourceRegistry":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(SourceConfig.model_validate(data))

    def match_entity(self, query: str) -> list[OfficialEntity]:
        normalized_query = _normalize(query)
        if not normalized_query:
            return []
        matches: list[tuple[int, OfficialEntity]] = []
        for entity in self.entities:
            for name in entity.searchable_names:
                normalized_name = _normalize(name)
                if normalized_query == normalized_name:
                    matches.append((100, entity))
                elif normalized_query in normalized_name or normalized_name in normalized_query:
                    matches.append((80, entity))

        deduped: dict[str, tuple[int, OfficialEntity]] = {}
        for score, entity in matches:
            current = deduped.get(entity.id)
            if current is None or score > current[0]:
                deduped[entity.id] = (score, entity)
        return [entity for _, entity in sorted(deduped.values(), key=lambda item: item[0], reverse=True)]

    def all_allowed_suffixes(self) -> set[str]:
        suffixes: set[str] = set()
        for rule in self.jurisdictions.values():
            suffixes.update(rule.allow_suffixes)
        return suffixes

    def all_denied_domains(self) -> set[str]:
        domains: set[str] = set()
        for rule in self.jurisdictions.values():
            domains.update(rule.deny_domains)
        return domains


def _normalize(value: str) -> str:
    return "".join(value.lower().split())
