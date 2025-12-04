# summarization/mt5_runner.py
import math, re, os
from typing import List, Tuple
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from sum_config import target_word_bounds, approx_tokens_from_words
from utils import (
    iter_articles, write_record, ensure_dir,
    word_count, clamp_chars, get_root_dir
)

MODEL = "csebuetnlp/mT5_multilingual_XLSum"

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
mdl = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL,
    dtype=torch.float16 if torch.cuda.is_available() else None,  # ← Fixed
    device_map="auto"
)
MAX_SRC_TOK = min(getattr(mdl.config, "n_positions", 1024) or 1024, 1024)
tok.model_max_length = MAX_SRC_TOK

summ = pipeline(
    "summarization",
    model=mdl,
    tokenizer=tok,
    device_map="auto"
)

_ws = re.compile(r"\s+")
def clean_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"\u00ad", "", s)
    s = re.sub(r"\s*—\s*", " — ", s)
    s = re.sub(_ws, " ", s).strip()
    return s

def post_clean(s: str) -> str:
    s = re.sub(r"\s+([,.!?;:])", r"\1", s)
    s = re.sub(r"\.(\s*\.){1,}", ".", s)
    s = re.sub(_ws, " ", s).strip()
    return s

def chunk_token_ids(input_ids, max_len: int, overlap: int = 64) -> List[List[int]]:
    chunks = []
    i = 0
    L = len(input_ids)
    stride = max_len - overlap
    if stride <= 0:
        stride = max_len
    while i < L:
        chunks.append(input_ids[i:i + max_len])
        i += stride
    return chunks

def summarize_chunk_ids(input_ids, max_new_tokens: int, min_new_tokens: int = 32,
                        num_beams=4, length_penalty=1.5, no_repeat_ngram_size=3,  # ← Increased from 1.0
                        repetition_penalty=1.05) -> str:
    input_ids = torch.tensor([input_ids], device=mdl.device)
    with torch.no_grad():
        out = mdl.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=False,  # ← Changed from True - let it finish sentences
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
        )
    return tok.batch_decode(out, skip_special_tokens=True)[0].strip()

def map_reduce_summarize(body: str, target_tokens: int) -> str:
    body = clean_text(body)
    enc = tok(body, truncation=True, max_length=tok.model_max_length, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()

    if len(ids) <= MAX_SRC_TOK:
        # Single pass
        return summarize_chunk_ids(
            ids, 
            max_new_tokens=target_tokens, 
            min_new_tokens=100,  # ← Increased from 80
            num_beams=4, 
            length_penalty=1.5  # ← Increased from 1.0
        )

    # Map: summarize chunks
    windows = chunk_token_ids(ids, MAX_SRC_TOK, overlap=96)
    chunk_summaries = []
    per_chunk = max(64, target_tokens // max(1, len(windows)))
    for w in windows:
        cs = summarize_chunk_ids(
            w, 
            max_new_tokens=per_chunk, 
            min_new_tokens=50,  # ← Increased from 32
            num_beams=4, 
            length_penalty=1.5  # ← Increased from 1.0
        )
        chunk_summaries.append(cs)

    combined = clean_text(" ".join(chunk_summaries))
    final = summarize_chunk_ids(
        tok(combined, truncation=True, max_length=MAX_SRC_TOK)["input_ids"],
        max_new_tokens=target_tokens,
        min_new_tokens=100,  # ← Increased from 80
        num_beams=5,
        length_penalty=1.5,  # ← Increased from 1.05
    )
    return final
def summarize_one(body: str, src_words: int) -> str:
    lo, hi = target_word_bounds(src_words)
    max_new = approx_tokens_from_words(hi)
    
    base = map_reduce_summarize(clamp_chars(body), target_tokens=max_new)
    return post_clean(base)

def run(in_csv="data/guardian_articles_raw.csv", out_jsonl="data/outputs/mt5.jsonl", limit=0):
    root = get_root_dir()
    in_csv = os.path.join(root, in_csv)
    out_jsonl = os.path.join(root, out_jsonl)
    ensure_dir(out_jsonl)
    
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for i, title, body, domain, url in tqdm(iter_articles(in_csv, limit=limit)):
            src_words = word_count(body)
            s = summarize_one(body, src_words)
            
            write_record(out, {
                "doc_id": i,
                "domain": domain,
                "url": url,
                "model": "mT5-XLSum",
                "style": "one_paragraph",
                "title": title,
                "summary": s
            })
            out.flush()

if __name__ == "__main__":
    run()