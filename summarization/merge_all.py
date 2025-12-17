# summarization/merge_outputs.py
import glob

IN_PATHS = [
    "data/outputs/textrank.jsonl",
    "data/outputs/bart.jsonl",
    "data/outputs/pegasus.jsonl",
    "data/outputs/gemma.jsonl",
    "data/outputs/mt5.jsonl",
    "data/outputs/mistral.jsonl",
]

def run(out_path="data/outputs/all_summaries.jsonl"):
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
