# summarization/merge_outputs.py
import glob

IN_PATHS = [
    "outputs/textrank.jsonl",
    "outputs/bart.jsonl",
    "outputs/pegasus.jsonl",
    "outputs/flan.jsonl",
    "outputs/mt5.jsonl",
    "outputs/mistral.jsonl",  # optional
]

def run(out_path="outputs/all_summaries.jsonl"):
    with open(out_path, "w", encoding="utf-8") as out:
        for p in IN_PATHS:
            try:
                for line in open(p, "r", encoding="utf-8"):
                    out.write(line)
            except FileNotFoundError:
                # skip missing models (e.g., mistral)
                continue
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    run()
