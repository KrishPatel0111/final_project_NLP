# Copilot / AI Agent Instructions — final_project_NLP

This file gives concise, actionable guidance for an AI coding agent working in this repository.

Overview
- This project prepares and evaluates summarization and LLM-extraction pipelines over Guardian articles.
- Major directories:
  - `data/` : raw and processed inputs; `data/guardian_articles_raw.csv` is the main article source. Outputs are in `data/outputs/` (per-model JSONL files).
  - `extraction/` : extraction prompt builders and LLM extraction code (e.g., `deepseek_extractor.py`, `qwen_extraction.py`).
  - `summarization/` : model runners and utilities for producing summaries (scripts like `run_bart.py`, `run_gemma.py`, `run_mistral.py`, `run_pegasus.py`, `run_t5.py`, `run_textrank.py`).
  - `data/prompt_templates/` and `data/prompt_previews/` : canonical prompt templates and example prompts used by extraction and labeling.

Key patterns and conventions
- Scripts are designed to be runnable directly: each `run_*.py` exposes a `run(in_csv, out_jsonl, limit=0)` function and uses `if __name__ == "__main__": run()`.
- Paths are resolved relative to repository root using `get_root_dir()` from `summarization/utils.py`. When editing code, prefer relative paths and these helpers.
- Outputs are JSONL records written with helper `write_record` (see `summarization/utils.py`). Each record includes `doc_id`, `domain`, `url`, `model`, `style`, `title`, `summary` (or other fields for extraction outputs).
- Tokenization and model config: many summarization runners use Hugging Face `AutoTokenizer`/`AutoModelForSeq2SeqLM` and set `tokenizer.model_max_length` to the model encoder limit (see `run_bart.py`). Follow existing sizing/clamping helpers: `clamp_chars`, `target_word_bounds`, `approx_tokens_from_words` in `summarization/sum_config.py` and `utils.py`.

Developer workflows (discoverable commands)
- Install dependencies: `pip install -r requirements.txt` (project uses Hugging Face, torch, tqdm, etc.).
- Run a summarization pipeline (example):
  - `python summarization/run_bart.py` — reads `data/guardian_articles_raw.csv`, writes `data/outputs/bart.jsonl`.
  - All `run_*.py` files follow the same `run()` pattern; to limit processing use a small CSV or edit the `limit` parameter in the `run()` call.
- Run extraction/labeling helpers: run individual modules under `extraction/` directly (for example `python extraction/deepseek_extractor.py`), but inspect their top-level `run` or test functions before executing.

Integration and data flow
- Single source of truth: `data/guardian_articles_raw.csv` → `summarization/*` produce model-specific outputs in `data/outputs/` → downstream analysis uses those JSONL outputs (see `summarization/merge_all.py` and `random_analysis.py`).
- Prompt templates from `data/prompt_templates/` feed `extraction/` scripts. When updating prompts, maintain the JSON structure in those files and add preview examples into `data/prompt_previews/` for reproducibility.

Project-specific notes for AI edits
- Preserve the `run(in_csv, out_jsonl, limit=0)` signature in runner scripts to keep tooling consistent.
- Use helpers in `summarization/utils.py` for file iteration (`iter_articles`), writing JSONL (`write_record`), and path resolution (`get_root_dir`, `ensure_dir`). Avoid ad-hoc file I/O patterns.
- Many modules assume CUDA if available: check `torch.cuda.is_available()` and prefer non-destructive defaults (do not force GPU-only changes).
- Keep model strings explicit (e.g., `facebook/bart-large-cnn`) and don't change model hyperparameters without adding matching config constants in `sum_config.py`.

Examples to reference while coding
- Reading articles: `summarization/utils.py::iter_articles` — follows (i, title, body, domain, url) tuple pattern.
- Writing output: `summarization/run_bart.py` — how `summarize_one` and `run()` produce JSONL records.
- Prompt templates: `data/prompt_templates/*.json` — follow existing keys and fields when adding new templates.

When to ask the user
- If a change affects dataset layout (moving or renaming `data/guardian_articles_raw.csv` or output JSONL names). 
- When adding new model runners or changing model sizes/hyperparameters that will alter output formats.

Next steps after edits
- Run a small smoke test: `python summarization/run_bart.py` and open the first 3 lines of `data/outputs/bart.jsonl`.

If anything above is unclear or you want more detail (examples, CLI flags, or tests), tell me which area to expand.
