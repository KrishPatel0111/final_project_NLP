# summarization/mistral_runner.py
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from sum_config import STYLES, target_word_bounds, approx_tokens_from_words
from utils import iter_articles, write_record, ensure_dir, word_count, clamp_chars, get_root_dir

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
mdl = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype="auto")

def prompt(style, title, body):
    base = (
        "You are a careful, faithful summarizer. Preserve named entities, numbers, dates, and culturally relevant context "
        "(identity, heritage, religion, community, local references). Do NOT add facts.\n\n"
        f"TITLE: {title}\n\nARTICLE:\n{body}\n\n"
    )
    ask = {"bullets": "Produce 5-7 bullet points.",
           "one_paragraph": "Produce one paragraph of 5-7 sentences.",
           "three_sentence": "Produce exactly three sentences."}[style]
    return base + ask

def summarize_one(title, body, style, src_words):
    if style == "one_paragraph":
        lo, hi = target_word_bounds(src_words)
        max_new = approx_tokens_from_words(hi)
    elif style == "bullets":
        max_new = 220
    else:
        max_new = 120
    
    p = prompt(style, title, clamp_chars(body))
    inputs = tok(p, return_tensors="pt", truncation=True, max_length=4096).to(mdl.device)
    
    with torch.no_grad():
        out = mdl.generate(
            **inputs, 
            max_new_tokens=max_new, 
            do_sample=False,          # Greedy decoding (deterministic)
            # temperature=0.2,        # ← REMOVE THIS LINE
            repetition_penalty=1.05, 
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id if tok.pad_token_id else tok.eos_token_id
        )
    
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(p):].strip()

def run(in_csv="data/guardian_articles_raw.csv", out_jsonl="data/outputs/mistral.jsonl", limit=0):
    root = get_root_dir()
    in_csv = os.path.join(root, in_csv)
    out_jsonl = os.path.join(root, out_jsonl)
    ensure_dir(out_jsonl)
    print("[bart] IN:", in_csv)
    print("[bart] OUT:", out_jsonl)
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for i, title, body, domain, url in tqdm(iter_articles(in_csv, limit=limit)):
            src_words = word_count(body)
            for style in STYLES:
                s = summarize_one(title, body, style, src_words)
                write_record(out, {
                    "doc_id": i, "domain": domain, "url": url,
                    "model": "Mistral-7B-Instruct", "style": style, "title": title, "summary": s
                })
                out.flush()

if __name__ == "__main__":
    run()
