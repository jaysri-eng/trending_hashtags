Trending Hashtag AI Agent
==========================

This project is a small Python-based AI agent that helps you discover **trending topics** and turn them into **effective hashtags** for your social posts and reels.

At a high level, it:

- Fetches trending topics from **legal, public sources** (e.g. Google Trends via `pytrends`)
- Normalizes and aggregates these topics into a unified internal format
- Uses a configurable **LLM layer**:
  - **Free option**: local Ollama model (no credits needed)
  - **Optional paid option**: OpenAI-compatible API (if you have credits)
  - Understand your **content description** and **target platform** (Instagram, TikTok, X, etc.)
  - Propose **platform-appropriate hashtags** aimed at maximizing reach and engagement
- Exposes a simple **CLI** so you can run it from the terminal

> For a deep-dive into the architecture, design decisions, and model details, see `study.md` (generated as part of this project).

## Quick start

1. **Create and activate a virtualenv** (optional but recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate  # on macOS / Linux
# .venv\Scripts\activate   # on Windows PowerShell
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Set your LLM API key** (example with OpenAI-compatible API):

```bash
export OPENAI_API_KEY="your_api_key_here"
```

If you don’t have API credits, you can use the **free local Ollama backend** instead:

```bash
# 1) Install Ollama (one-time) from https://ollama.com
# 2) Pull a model (one-time)
ollama pull llama3.1

# 3) Tell the agent to use Ollama
export TRENDING_LLM_BACKEND="ollama"
export OLLAMA_MODEL="llama3.1"
```

4. **Run the CLI**:

```bash
python -m trending_agent.cli \
  suggest \
  --platform instagram \
  --region IN \
  --description "Aesthetic coffee shop reel with lo-fi music"
```

By default it uses the **hybrid pipeline** (trend history + embeddings + clustering + retrieval + optional LLM refinement).

You can force the old simple pipeline:

```bash
python -m trending_agent.cli suggest --mode simple --platform instagram --region IN --description "..."
```

The agent will:

- Fetch trending topics (e.g. via Google Trends) for your region
- Combine them with your content description
- Ask the LLM (Ollama/OpenAI) to propose a curated list of hashtags (or fall back to rule-based tags)

## Project layout

- `trending_agent/`
  - `__init__.py`: package marker
  - `sources/`: code for various data sources
    - `base.py`: common interfaces for trend sources
    - `google_trends.py`: implementation using `pytrends`
  - `agent/`: logic that orchestrates sources and the LLM
    - `core.py`: aggregator + agent orchestration
    - `llm.py`: thin wrapper around an OpenAI-compatible chat completion API
  - `cli.py`: command-line interface entry point
- `requirements.txt`: Python dependencies
- `study.md`: in-depth explanation of architecture and implementation (generated later)

Refer to `study.md` for a thorough conceptual explanation of how each piece works and how to extend the agent with new data sources or different LLMs.

