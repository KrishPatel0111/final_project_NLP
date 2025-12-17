# summarization/bart_runner.py
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sum_config import target_word_bounds, approx_tokens_from_words
from utils import iter_articles, write_record, ensure_dir, word_count, clamp_chars, get_root_dir
import torch, os

MODEL = "facebook/bart-large-cnn"
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mdl.to(device)

# Ensure the tokenizer enforces the encoder window
MAX_INPUT_TOKENS = getattr(mdl.config, "max_position_embeddings", 1024) or 1024
tok.model_max_length = min(MAX_INPUT_TOKENS, 1024)

def summarize_one(body, src_words):
    lo, hi = target_word_bounds(src_words)
    min_new = max(approx_tokens_from_words(lo), 32)  # ← Add this
    max_new = approx_tokens_from_words(hi)

    enc = tok(
        clamp_chars(body),
        return_tensors="pt",
        truncation=True,
        max_length=tok.model_max_length,
        padding=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out_ids = mdl.generate(
            **enc,
            min_new_tokens=min_new,  # ← Add this
            max_new_tokens=max_new,
            num_beams=4,
            early_stopping=True,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
        )

    out = tok.batch_decode(out_ids, skip_special_tokens=True)[0]
    return out

def run(in_csv="data/guardian_articles_raw.csv", out_jsonl="data/outputs/bart.jsonl", limit=0):
    root = get_root_dir()
    in_csv = os.path.join(root, in_csv)
    out_jsonl = os.path.join(root, out_jsonl)
    print("[bart] IN:", in_csv)
    print("[bart] OUT:", out_jsonl)

    ensure_dir(out_jsonl)
    wrote = 0
    
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for i, title, body, domain, url in tqdm(iter_articles(in_csv, limit=limit)):
            src_words = word_count(body)
            s = summarize_one(body, src_words)
            
            write_record(out, {
                "doc_id": i,
                "domain": domain,
                "url": url,
                "model": "BART-large-cnn",
                "style": "one_paragraph",  # Only one style
                "title": title,
                "summary": s
            })
            out.flush()
            wrote += 1
            
        print(f"[bart] wrote {wrote} records")

if __name__ == "__main__":
    run()