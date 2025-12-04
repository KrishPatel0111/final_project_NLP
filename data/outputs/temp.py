import json
from collections import defaultdict

# ================== CONFIG ==================

# Hard-coded path to your data file
FILE_PATH = r"/users/PAS3150/patel4933/final/final_project_NLP/data/outputs/textrank.jsonl"  # <- change this

# Field names in your JSON
TITLE_KEY = "title"
STYLE_KEY = "style"
SUMMARY_KEY = "summary"

# Style names in your data
BULLET_STYLES = {"bullets", "bullet"}
PARA_STYLES = {"one_paragraph", "one paragraph"}

# ============================================


def load_json_file(path):
    """
    Load data from either:
    - JSON list: [ {...}, {...} ]
    - JSON Lines (one JSON object per line)
    Returns: list of dicts
    """
    with open(path, "r", encoding="utf-8") as f:
        # Peek first non-space character
        start = f.read(1)
        while start and start.isspace():
            start = f.read(1)

        f.seek(0)  # go back to beginning

        if start == "[":
            # Regular JSON array
            return json.load(f)
        else:
            # JSONL / newline-delimited JSON
            data = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
            return data


def count_dots(text: str) -> int:
    """Return the number of '.' characters in the text."""
    return text.count(".")


def main():
    data = load_json_file(FILE_PATH)

    if not isinstance(data, list):
        raise ValueError("Expected the file to contain a list or JSONL of objects.")

    # Group summaries by title and style
    grouped = defaultdict(lambda: {"bullets": [], "one_paragraph": []})

    for row in data:
        title = row.get(TITLE_KEY)
        if not title:
            continue  # skip rows without title

        style_raw = str(row.get(STYLE_KEY, "")).strip().lower()
        summary = row.get(SUMMARY_KEY, "")

        if style_raw in BULLET_STYLES:
            grouped[title]["bullets"].append(summary)
        elif style_raw in PARA_STYLES:
            grouped[title]["one_paragraph"].append(summary)
        # ignore other styles like "three_sentence"
    same_count = 0
    different_count = 0
    # Compare for each title
    for title, styles in grouped.items():
        bullet_summaries = styles["bullets"]
        para_summaries = styles["one_paragraph"]

        if not bullet_summaries and not para_summaries:
            continue

        # Precompute dot-counts for one_paragraph summaries
        para_dot_counts = [count_dots(s) for s in para_summaries]

       

        # For each bullet summary, check if ANY one_paragraph summary
        # has the same number of '.' characters
        for b_sum in bullet_summaries:
            b_dots = count_dots(b_sum)
            if b_dots in para_dot_counts:
                same_count += 1
            else:
                different_count += 1

        print("=" * 80)
        print(f"Title: {title}")
        print(f"  Bullet summaries:        {len(bullet_summaries)}")
        print(f"  One-paragraph summaries: {len(para_summaries)}")
        print(f"  SAME (same '.' count):   {same_count}")
        print(f"  DIFFERENT:               {different_count}")

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
