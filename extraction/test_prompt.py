# test_prompt_hardcoded.py

import json
import pandas as pd

from prompt_builder import (
    load_labeled_examples,
    build_few_shot_prompt,
    build_chat_messages,
)

def save_pretty_prompt(prompt: str, chat_messages: list, filename: str = "logs/prompt_ex.log"):
    """
    Save both the raw few-shot prompt string and chat messages
    into a clean, readable log file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        
        f.write("=" * 100 + "\n")
        f.write("FULL FEW-SHOT PROMPT (STRING)\n")
        f.write("=" * 100 + "\n\n")
        f.write(prompt)
        f.write("\n\n\n")

        f.write("=" * 100 + "\n")
        f.write("CHAT MESSAGES (RAW JSON)\n")
        f.write("=" * 100 + "\n\n")
        f.write(json.dumps(chat_messages, indent=2, ensure_ascii=False))
        f.write("\n")

    print(f"\nSaved nicely formatted prompt file → {filename}\n")


def main():
    # ------------------------------------------------------------------
    # HARD-CODED FILE PATHS (change here ONLY if your structure changes)
    # ------------------------------------------------------------------
    ARTICLES_PATH = "data/guardian_articles_raw.csv"
    SUMMARIES_PATH = "data/outputs/textrank.jsonl"
    LABELED_PATH = "data/labeled_examples/labeled_data.json"

    # Which labeled example to test?
    DOC_ID = 0          # <-- CHANGE THIS to test another labeled example
    SUMMARY_STYLE = "bullets"
    N_EXAMPLES = 3      # how many few-shot examples to include

    print("\nLoading labeled examples…")
    labeled_examples = load_labeled_examples(LABELED_PATH)
    if not labeled_examples:
        raise RuntimeError("Could not load labeled examples")

    # Pick target example
    target_example = None
    for ex in labeled_examples:
        if ex.get("doc_id") == DOC_ID:
            target_example = ex
            break

    if target_example is None:
        raise RuntimeError(f"No labeled example found with doc_id={DOC_ID}")

    title = target_example["title"]
    domain = target_example["domain"]

    print(f"Using example: doc_id={DOC_ID}, title={title!r}, domain={domain!r}")

    # ------------------------------------------------------------------
    # Load raw article + summary
    # ------------------------------------------------------------------
    print("\nLoading article CSV…")
    articles_df = pd.read_csv(ARTICLES_PATH)

    print("Loading summaries JSONL…")
    summaries_df = pd.read_json(SUMMARIES_PATH, lines=True)

    # --- Article text lookup ---
    article_rows = articles_df[articles_df["title"] == title]
    if article_rows.empty:
        raise RuntimeError(f"No article found with title {title!r}")

    # HARD-CODED COLUMN NAME: change "text" if your CSV uses another name
    article_text = article_rows.iloc[0]["text"]

    # --- Summary lookup ---
    summary_rows = summaries_df[
        (summaries_df["title"] == title) &
        (summaries_df["style"] == SUMMARY_STYLE)
    ]

    if summary_rows.empty:
        raise RuntimeError(
            f"No summary found with title {title!r} and style {SUMMARY_STYLE!r}"
        )

    # HARD-CODED COLUMN NAME: change "summary" if JSONL uses "summary_text"
    summary_text = summary_rows.iloc[0]["summary"]

    # ------------------------------------------------------------------
    # Build prompt and chat messages
    # ------------------------------------------------------------------
    print("\nBuilding few-shot prompt…")
    prompt = build_few_shot_prompt(
        article_title=title,
        article_text=article_text,
        summary_text=summary_text,
        domain=domain,
        examples_file=LABELED_PATH,
        n_examples=N_EXAMPLES,
        include_system=True,
    )

    print("Building chat messages…")
    chat_messages = build_chat_messages(
        article_title=title,
        article_text=article_text,
        summary_text=summary_text,
        domain=domain,
        examples_file=LABELED_PATH,
        n_examples=N_EXAMPLES,
    )

    # ------------------------------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------------------------------

    print("\n" + "="*80)
    print("TARGET LABELED EXAMPLE STRUCTURE")
    print("="*80)
    print(json.dumps(target_example, indent=2, ensure_ascii=False))

    print("\n" + "="*80)
    print("FEW-SHOT PROMPT (FIRST 2000 CHARACTERS)")
    print("="*80)
    print(prompt[:2000])

    print("\n" + "="*80)
    print("CHAT MESSAGE PAYLOAD")
    print("="*80)
    print(json.dumps(chat_messages, indent=2, ensure_ascii=False))
    save_pretty_prompt(prompt, chat_messages)


if __name__ == "__main__":
    main()
