import json

def pretty_print_textrank(file_path, output_path="textrank_readable.txt"):
    """
    Reads a JSONL file of TextRank summaries and writes them
    to a text file in a clean, readable format.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Group entries by doc_id
    grouped = {}
    for entry in data:
        doc_id = entry.get("doc_id")
        grouped.setdefault(doc_id, []).append(entry)

    # Write formatted output to file
    with open(output_path, "w", encoding="utf-8") as out:
        for doc_id, entries in grouped.items():
            base = entries[0]
            out.write("=" * 100 + "\n")
            out.write(f"📰  TITLE: {base['title']}\n")
            out.write(f"🌐  URL: {base['url']}\n")
            out.write(f"🏷️  DOMAIN: {base['domain']}\n")
            out.write(f"🤖  MODEL: {base['model']}\n")
            out.write("-" * 100 + "\n")

            for e in entries:
                style = e.get("style", "unknown")
                summary = e.get("summary", "").strip()
                out.write(f"📝 STYLE: {style.upper()}\n")
                out.write(summary + "\n")
                out.write("-" * 100 + "\n")

            out.write("\n")

    print(f"✅ Readable summaries have been saved to: {output_path}")

if __name__ == "__main__":
    # Change this to your actual file path if needed
    file_path = "textrank.jsonl"
    pretty_print_textrank(file_path)
