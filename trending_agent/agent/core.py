from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from ..sources.base import TrendItem, TrendSource
from .llm import HashtagLLM


@dataclass
class AgentConfig:
    """
    Configuration for the hashtag agent.
    """

    max_trends_per_source: int = 20
    max_hashtags: int = 20


class HashtagAgent:
    """
    Orchestrates multiple trend sources and the LLM to generate hashtag suggestions.
    """

    def __init__(
        self,
        *,
        sources: Sequence[TrendSource],
        llm: HashtagLLM,
        config: AgentConfig | None = None,
    ) -> None:
        self._sources = list(sources)
        self._llm = llm
        self._config = config or AgentConfig()

    def _collect_trends(self, *, region: str | None) -> List[TrendItem]:
        items: List[TrendItem] = []
        for src in self._sources:
            try:
                src_items = src.fetch_trends(region=region, limit=self._config.max_trends_per_source)
            except Exception:
                # In an educational setting, we fail soft here. In production,
                # you'd want structured logging/metrics.
                src_items = []
            items.extend(src_items)

        # Normalise scores across sources: simple min-max normalisation per source
        # would be better, but to keep this project approachable we rely on the
        # score semantics of each source and just sort descending.
        items.sort(key=lambda t: t.score, reverse=True)
        return items

    def suggest_hashtags(
        self,
        *,
        description: str,
        platform: str,
        region: str | None = None,
    ) -> List[str]:
        """
        Public entry point: collect trends, then delegate to LLM.
        """
        trends = self._collect_trends(region=region)
        return self._llm.suggest_hashtags(
            description=description,
            platform=platform,
            region=region,
            trends=trends,
            max_hashtags=self._config.max_hashtags,
        )


def build_default_agent() -> HashtagAgent:
    """
    Convenience constructor that wires up default components.
    """
    from ..sources.google_trends import default_google_trends_source

    source = default_google_trends_source()
    llm = HashtagLLM()
    return HashtagAgent(sources=[source], llm=llm)


def build_hybrid_agent(*, db_path: str | Path = "trends.db"):
    """
    Default "smarter" agent with:
    - SQLite trend history
    - velocity estimation
    - embeddings + clustering + retrieval
    - optional LLM refinement
    """
    from ..sources.google_trends import default_google_trends_source
    from ..storage.sqlite_store import TrendSQLiteStore
    from .hybrid import HybridHashtagAgent

    store = TrendSQLiteStore(db_path=db_path)
    source = default_google_trends_source()
    llm = HashtagLLM()
    return HybridHashtagAgent(sources=[source], llm=llm, store=store)

