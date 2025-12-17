# summarization/gemma_runner.py
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from sum_config import STYLES, target_word_bounds, approx_tokens_from_words
from utils import iter_articles, write_record, ensure_dir, word_count, clamp_chars, get_root_dir

MODEL = "google/gemma-2-9b-it"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

mdl = AutoModelForCausalLM.from_pretrained(
    MODEL, 
    device_map="auto", 
    torch_dtype=torch.bfloat16
)

def prompt(style, title, body):
    """Create style-specific prompts using Gemma's chat format."""
    system_msg = (
        "You are a careful, faithful summarizer. Preserve named entities, numbers, dates, and culturally relevant context "
        "(identity, heritage, religion, community, local references). Do NOT add facts."
    )
    
    style_instruction = {
        "bullets": "Produce exactly 5-7 bullet points. Start each with a dash (-).",
        "one_paragraph": "Produce one cohesive paragraph of 5-7 sentences.",
        "three_sentence": "Produce exactly three sentences.",
    }[style]
    
    user_msg = f"{style_instruction}\n\nTITLE: {title}\n\nARTICLE:\n{body}"
    
    messages = [
        {"role": "user", "content": f"{system_msg}\n\n{user_msg}"}
    ]
    
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def summarize_one(title, body, style, src_words):
    """Generate summary for one article in specified style."""
    if style == "one_paragraph":
        lo, hi = target_word_bounds(src_words)
        max_new = approx_tokens_from_words(hi)
    elif style == "bullets":
        max_new = 250
    else:  # three_sentence
        max_new = 120
    
    p = prompt(style, title, clamp_chars(body))
    inputs = tok(p, return_tensors="pt", truncation=True, max_length=8192).to(mdl.device)
    
    with torch.no_grad():
        outputs = mdl.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,          # Deterministic greedy decoding
            repetition_penalty=1.05,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    
    # Decode and extract only the assistant's response
    full_text = tok.decode(outputs[0], skip_special_tokens=True)
    prompt_text = tok.decode(inputs['input_ids'][0], skip_special_tokens=True)
    
    # Remove the prompt
    if full_text.startswith(prompt_text):
        summary = full_text[len(prompt_text):].strip()
    else:
        if "<start_of_turn>model" in full_text:
            summary = full_text.split("<start_of_turn>model")[-1].replace("<end_of_turn>", "").strip()
        else:
            summary = full_text.strip()
    
    return summary

def run(in_csv="data/guardian_articles_raw.csv", out_jsonl="data/outputs/gemma.jsonl", limit=0):
    root = get_root_dir()
    in_csv = os.path.join(root, in_csv)
    out_jsonl = os.path.join(root, out_jsonl)
    ensure_dir(out_jsonl)
    
    print("[gemma] IN:", in_csv)
    print("[gemma] OUT:", out_jsonl)
    
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for i, title, body, domain, url in tqdm(iter_articles(in_csv, limit=limit)):
            src_words = word_count(body)
            for style in STYLES:
                try:
                    s = summarize_one(title, body, style, src_words)
                except Exception as e:
                    print(f"Error on doc {i}, style {style}: {e}")
                    s = "[GENERATION_FAILED]"
                
                write_record(out, {
                    "doc_id": i,
                    "domain": domain,
                    "url": url,
                    "model": "Gemma-2-9B-IT",
                    "style": style,
                    "title": title,
                    "summary": s
                })
                out.flush()

if __name__ == "__main__":
    run()