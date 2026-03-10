from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ..ml.clustering import ClusterResult, cluster_embeddings
from ..ml.embeddings import embed_texts
from ..ml.vector_search import build_index, search
from ..processing.velocity import attach_velocity
from ..sources.base import TrendItem, TrendSource
from ..storage.sqlite_store import TrendSQLiteStore
from .llm import HashtagLLM


@dataclass
class HybridOutput:
    hashtags: List[str]
    backend: str
    top_trends: List[TrendItem]
    clusters: Dict[int, List[str]]  # cluster label -> keywords
    retrieved: List[Tuple[str, float]]  # keyword, sim


class HybridHashtagAgent:
    """
    Production-leaning pipeline:
    - collect trends
    - persist snapshots
    - attach velocity (growth proxy)
    - embed + cluster trends (topic grouping)
    - embed-search trends by similarity to the post description
    - generate hashtags:
        - retrieve-driven base candidates
        - optional LLM refinement (ollama/openai) OR rule fallback
    """

    def __init__(
        self,
        *,
        sources: Sequence[TrendSource],
        llm: HashtagLLM,
        store: TrendSQLiteStore,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_trends_per_source: int = 50,
        max_hashtags: int = 20,
        retrieval_k: int = 25,
    ) -> None:
        self._sources = list(sources)
        self._llm = llm
        self._store = store
        self._embedding_model = embedding_model
        self._max_trends_per_source = max_trends_per_source
        self._max_hashtags = max_hashtags
        self._retrieval_k = retrieval_k

    def _collect(self, *, region: str | None) -> List[TrendItem]:
        items: List[TrendItem] = []
        for src in self._sources:
            try:
                items.extend(src.fetch_trends(region=region, limit=self._max_trends_per_source))
            except Exception:
                continue
        # Persist and compute velocity against history
        self._store.insert_snapshot(items)
        items = attach_velocity(self._store, items)
        # Sort by score then velocity
        items.sort(key=lambda t: (t.score, t.velocity or 0.0), reverse=True)
        return items

    def run(self, *, description: str, platform: str, region: str | None) -> HybridOutput:
        trends = self._collect(region=region)
        keywords = [t.keyword for t in trends]

        # Embeddings + clustering over keywords (topic grouping)
        kw_vecs = embed_texts(keywords, model_name=self._embedding_model)
        clus: ClusterResult = cluster_embeddings(kw_vecs, min_cluster_size=3)
        clusters: Dict[int, List[str]] = {}
        for label, idxs in clus.clusters.items():
            clusters[int(label)] = [keywords[i] for i in idxs]

        # Retrieval: find trends semantically closest to the post description
        if kw_vecs.size == 0:
            retrieved: List[Tuple[str, float]] = []
        else:
            q = embed_texts([description], model_name=self._embedding_model)
            vi = build_index(kw_vecs, keywords)
            retrieved = search(vi, q[0], k=min(self._retrieval_k, len(keywords)))

        # Build a focused trend list for the LLM/rules: prioritize retrieved keywords
        retrieved_set = {k for (k, _s) in retrieved}
        focused: List[TrendItem] = [t for t in trends if t.keyword in retrieved_set]
        if len(focused) < min(10, len(trends)):
            focused.extend([t for t in trends if t.keyword not in retrieved_set][: (10 - len(focused))])

        hashtags = self._llm.suggest_hashtags(
            description=description,
            platform=platform,
            region=region,
            trends=focused,
            max_hashtags=self._max_hashtags,
        )

        # Best-effort backend name (internal detail for UI)
        backend = getattr(self._llm, "_select_backend", lambda: "unknown")()

        return HybridOutput(
            hashtags=hashtags,
            backend=str(backend),
            top_trends=trends[: min(20, len(trends))],
            clusters=clusters,
            retrieved=retrieved[: min(10, len(retrieved))],
        )

