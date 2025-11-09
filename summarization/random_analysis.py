from pathlib import Path
import pandas as pd

from utils import get_root_dir

root = Path(get_root_dir())

textrank_path = root / "data" / "outputs" / "textrank.jsonl"
articles_path = root / "data" / "guardian_articles_raw.csv"

# Sanity checks so pandas doesn't misinterpret a bad path as literal JSON
if not textrank_path.exists():
    raise FileNotFoundError(f"Textrank file not found: {textrank_path.resolve()}")
if textrank_path.stat().st_size == 0:
    raise ValueError(f"Textrank file is empty: {textrank_path.resolve()}")

# Read JSONL correctly
output = pd.read_json(textrank_path, lines=True)  # critical: lines=True for .jsonl
# If your JSONL lines are not plain UTF-8, add: encoding="utf-8"

# Read CSV
if not articles_path.exists():
    raise FileNotFoundError(f"Guardian CSV not found: {articles_path.resolve()}")
input_df = pd.read_csv(articles_path)

# Ensure required columns exist
for df_name, df, needed in [
    ("output (textrank.jsonl)", output, "url"),
    ("input (guardian_articles_raw.csv)", input_df, "url"),
]:
    if needed not in df.columns:
        raise KeyError(f"Column '{needed}' missing from {df_name}. Columns: {list(df.columns)}")

# Compute missing
set_vals = set(input_df["url"])
missing = output[~output["url"].isin(set_vals)]

print(f"Missing {len(missing)} articles from TextRank output.")
