from __future__ import annotations

import os
import re
from typing import Iterable, List

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
import requests

from ..sources.base import TrendItem


class HashtagLLM:
    """
    Thin wrapper around an OpenAI-compatible chat completion model.

    This class is intentionally minimal so you can swap out the underlying
    provider (OpenAI, local server, etc.) by changing environment variables.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        # Load .env if present (robust to running from different working dirs).
        load_dotenv(find_dotenv(usecwd=True))

        # Backend selection:
        # - "ollama": local/free model via http://localhost:11434
        # - "openai": OpenAI (or compatible) API (requires OPENAI_API_KEY)
        # - "rules": no-LLM fallback (always available)
        self._backend = (os.getenv("TRENDING_LLM_BACKEND") or "").strip().lower() or "auto"

        self._ollama_model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
        self._ollama_base_url = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")

        self._openai_model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self._openai_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._openai_api_key = api_key or os.getenv("OPENAI_API_KEY")

        if self._backend == "openai" and not self._openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot initialize OpenAI client.")

        # Lazily constructed clients.
        self._openai_client: OpenAI | None = None

    def suggest_hashtags(
        self,
        *,
        description: str,
        platform: str,
        region: str | None,
        trends: Iterable[TrendItem],
        max_hashtags: int = 20,
    ) -> List[str]:
        """
        Suggest hashtags leveraging trends + (optional) LLM.

        If no paid API key is configured, this will automatically try a free local
        Ollama model if available, and otherwise fall back to a rule-based generator.
        """
        backend = self._select_backend()
        if backend == "openai":
            return self._suggest_hashtags_openai(
                description=description,
                platform=platform,
                region=region,
                trends=trends,
                max_hashtags=max_hashtags,
            )
        if backend == "ollama":
            return self._suggest_hashtags_ollama(
                description=description,
                platform=platform,
                region=region,
                trends=trends,
                max_hashtags=max_hashtags,
            )
        return self._suggest_hashtags_rules(
            description=description,
            platform=platform,
            region=region,
            trends=trends,
            max_hashtags=max_hashtags,
        )

    def _select_backend(self) -> str:
        """
        Returns one of: "openai", "ollama", "rules".
        """
        if self._backend in {"openai", "ollama", "rules"}:
            return self._backend

        # auto mode: prefer free/local if present, otherwise paid API if configured.
        if self._ollama_is_available():
            return "ollama"
        if self._openai_api_key:
            return "openai"
        return "rules"

    def _ollama_is_available(self) -> bool:
        try:
            r = requests.get(f"{self._ollama_base_url}/api/tags", timeout=1.5)
            return r.status_code == 200
        except Exception:
            return False

    def _normalise_lines_to_hashtags(self, content: str, max_hashtags: int) -> List[str]:
        raw_lines = [line.strip() for line in (content or "").splitlines()]
        hashtags: List[str] = []
        for line in raw_lines:
            if not line:
                continue
            if line[0].isdigit() and "." in line:
                line = line.split(".", 1)[1].strip()
            if line.startswith("- "):
                line = line[2:].strip()
            if not line:
                continue
            if not line.startswith("#"):
                line = "#" + line.lstrip("#")
            # strip trailing punctuation/spaces
            line = line.strip().rstrip(",")
            hashtags.append(line)
        return hashtags[:max_hashtags]

    def _suggest_hashtags_openai(
        self,
        *,
        description: str,
        platform: str,
        region: str | None,
        trends: Iterable[TrendItem],
        max_hashtags: int,
    ) -> List[str]:
        trends_text_lines = [
            f"- {t.keyword} (score={t.score:.1f}, source={t.source}, region={t.region or 'GLOBAL'})"
            for t in trends
        ]
        trends_text = "\n".join(trends_text_lines) if trends_text_lines else "None (no trends available)."

        system_prompt = (
            "You are an expert social media growth consultant who designs high-performing hashtags.\n"
            "You receive:\n"
            "1) A description of the user's post or reel\n"
            "2) The target platform (e.g. Instagram, TikTok, X, YouTube Shorts)\n"
            "3) A list of trending topics/keywords from legal, public data sources\n\n"
            "Your job is to output ONLY a list of hashtags, one per line, with no explanations.\n"
            "Guidelines:\n"
            "- Prefer hashtags that combine the user's topic with relevant trending ideas.\n"
            "- Mix broad and niche hashtags.\n"
            "- Avoid banned, offensive or misleading tags.\n"
            "- Respect typical character limits for hashtags on the target platform.\n"
            "- Do not include the # symbol more than once per hashtag (standard format like #aestheticCoffee).\n"
        )

        user_prompt = (
            f"Platform: {platform}\n"
            f"Region: {region or 'GLOBAL'}\n"
            f"Post description:\n{description}\n\n"
            f"Trending topics:\n{trends_text}\n\n"
            f"Please return at most {max_hashtags} hashtags, one per line, nothing else."
        )

        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=self._openai_api_key, base_url=self._openai_base_url)

        resp = self._openai_client.chat.completions.create(
            model=self._openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
        )

        return self._normalise_lines_to_hashtags(resp.choices[0].message.content or "", max_hashtags)

    def _suggest_hashtags_ollama(
        self,
        *,
        description: str,
        platform: str,
        region: str | None,
        trends: Iterable[TrendItem],
        max_hashtags: int,
    ) -> List[str]:
        trends_text_lines = [
            f"- {t.keyword} (score={t.score:.1f}, source={t.source}, region={t.region or 'GLOBAL'})"
            for t in trends
        ]
        trends_text = "\n".join(trends_text_lines) if trends_text_lines else "None (no trends available)."

        prompt = (
            "You are an expert social media growth consultant who designs high-performing hashtags.\n"
            "Return ONLY hashtags, one per line, no explanations.\n\n"
            f"Platform: {platform}\n"
            f"Region: {region or 'GLOBAL'}\n"
            f"Post description:\n{description}\n\n"
            f"Trending topics:\n{trends_text}\n\n"
            f"Return at most {max_hashtags} hashtags, one per line."
        )

        r = requests.post(
            f"{self._ollama_base_url}/api/generate",
            json={
                "model": self._ollama_model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        return self._normalise_lines_to_hashtags(str(data.get("response", "")), max_hashtags)

    def _suggest_hashtags_rules(
        self,
        *,
        description: str,
        platform: str,
        region: str | None,
        trends: Iterable[TrendItem],
        max_hashtags: int,
    ) -> List[str]:
        """
        Free fallback: combine description keywords + top trends into hashtags.
        """

        def slug_to_hashtag(text: str) -> str:
            # Keep letters/numbers/spaces; collapse.
            cleaned = re.sub(r"[^0-9A-Za-z\s]", " ", text)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if not cleaned:
                return ""
            parts = cleaned.split(" ")
            # Make a simple camel-ish hashtag.
            out = "".join(p[:1].upper() + p[1:] for p in parts if p)
            # Cap to a reasonable length.
            return ("#" + out)[:40]

        # Extract a few keywords from description.
        desc_clean = re.sub(r"[^0-9A-Za-z\s]", " ", description.lower())
        tokens = [t for t in re.split(r"\s+", desc_clean) if len(t) >= 4]
        # Drop very common filler words (tiny stoplist).
        stop = {
            "this",
            "that",
            "with",
            "from",
            "your",
            "reel",
            "post",
            "video",
            "music",
            "more",
            "best",
            "today",
            "trending",
        }
        tokens = [t for t in tokens if t not in stop]

        candidates: List[str] = []

        # Platform staples (kept minimal).
        p = platform.strip().lower()
        if p in {"instagram", "ig"}:
            candidates += ["#Reels", "#InstaReels", "#ExplorePage"]
        elif p in {"tiktok"}:
            candidates += ["#TikTok", "#ForYou", "#FYP"]
        elif p in {"youtube", "shorts", "youtube_shorts"}:
            candidates += ["#YouTubeShorts", "#Shorts"]

        # Add description-derived tags.
        for t in tokens[:10]:
            ht = slug_to_hashtag(t)
            if ht:
                candidates.append(ht)

        # Add multi-word tags from trends + description.
        for ti in list(trends)[: max(0, max_hashtags)]:
            ht = slug_to_hashtag(ti.keyword)
            if ht:
                candidates.append(ht)

        # De-duplicate while preserving order.
        seen: set[str] = set()
        out: List[str] = []
        for c in candidates:
            if not c or c in seen:
                continue
            seen.add(c)
            out.append(c)
            if len(out) >= max_hashtags:
                break

        # Ensure we return something even with empty description/trends.
        if not out:
            out = ["#Trending", "#Viral", "#ContentCreator"][:max_hashtags]

        return out


