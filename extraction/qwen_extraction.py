# extraction/qwen_extractor.py
import json
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict
import gc

from prompt_builder import build_fast_chat_messages

MODEL = "Qwen/Qwen2.5-7B-Instruct"

print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    max_memory={0: "28GB"}
)

torch.cuda.empty_cache()
gc.collect()


def extract_cues(
    article_title: str,
    article_text: str,
    summary_text: str,
    domain: str,
    templates_dir: str = "data/prompt_templates",
    max_retries: int = 2
) -> Dict:
    """Extract cues using pre-processed prompt templates (FAST)."""
    
    # Build messages using FAST method
    messages = build_fast_chat_messages(
        article_title=article_title,
        article_text=article_text,
        summary_text=summary_text,
        domain=domain,
        templates_dir=templates_dir,
        n_examples=1  # Start with 1 to avoid OOM
    )
    
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    prompt_tokens = tok(prompt, return_tensors="pt")['input_ids'].shape[1]
    print(f"  Prompt tokens: {prompt_tokens}")
    
    if prompt_tokens > 24000:
        print(f"  ⚠️ Prompt too long, truncating article...")
        article_text = article_text[:8000]
        messages = build_fast_chat_messages(
            article_title, article_text, summary_text, domain, templates_dir, n_examples=1
        )
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    for attempt in range(max_retries):
        try:
            torch.cuda.empty_cache()
            gc.collect()
            
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=20000).to(mdl.device)
            
            with torch.no_grad():
                outputs = mdl.generate(
                    **inputs,
                    max_new_tokens=3000,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id if tok.pad_token_id else tok.eos_token_id,
                )
            
            full_text = tok.decode(outputs[0], skip_special_tokens=True)
            prompt_text = tok.decode(inputs['input_ids'][0], skip_special_tokens=True)
            
            if full_text.startswith(prompt_text):
                response = full_text[len(prompt_text):].strip()
            else:
                if "<|im_start|>assistant" in full_text:
                    response = full_text.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
                else:
                    response = full_text.strip()
            
            response = response.replace("```json", "").replace("```", "").strip()
            
            del outputs, inputs
            torch.cuda.empty_cache()
            gc.collect()
            
            result = json.loads(response)
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"  Attempt {attempt + 1}/{max_retries} failed: CUDA OOM")
            torch.cuda.empty_cache()
            gc.collect()
            if attempt < max_retries - 1:
                continue
            else:
                return {"error": "CUDA out of memory"}
        
        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}/{max_retries} failed: Invalid JSON")
            if attempt < max_retries - 1:
                continue
            else:
                return {"error": "JSON parse failed", "raw_output": response[:500]}
        
        except Exception as e:
            print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            if attempt < max_retries - 1:
                continue
            else:
                return {"error": str(e)}
    
    return {"error": "Extraction failed after all retries"}


def run_extraction(
    summaries_jsonl: str = "data/outputs/all_summaries.jsonl",
    articles_csv: str = "data/guardian_articles_raw.csv",
    templates_dir: str = "data/prompt_templates",
    output_dir: str = "data/extractions",
    limit: int = 0
):
    """Run extraction using pre-processed templates with pandas for efficiency."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load summaries with pandas
    print("Loading summaries...")
    summaries_df = pd.read_json(summaries_jsonl, lines=True)
    print(f"Loaded {len(summaries_df)} summaries")
    print(f"Summaries columns: {summaries_df.columns.tolist()}")
    
    # Load articles with pandas
    print("\nLoading articles...")
    articles_df = pd.read_csv(articles_csv)
    print(f"Loaded {len(articles_df)} articles")
    print(f"Articles columns: {articles_df.columns.tolist()}")
    
    # Merge summaries with articles on title
    print("\nMerging summaries with articles...")
    
    # Keep only necessary columns from articles to avoid conflicts
    articles_subset = articles_df[['title', 'text', 'domain']].copy()
    
    # Drop domain from summaries if it exists (we'll use the one from articles)
    if 'domain' in summaries_df.columns:
        summaries_df = summaries_df.drop(columns=['domain'])
    
    # Merge without suffixes since we've removed conflicts
    merged_df = summaries_df.merge(
        articles_subset,
        on='title',
        how='left'
    )
    
    # Check for missing articles
    missing = merged_df['text'].isna().sum()
    if missing > 0:
        print(f"⚠️  Warning: {missing} summaries have no matching article")
        merged_df = merged_df.dropna(subset=['text'])
    
    print(f"Successfully merged {len(merged_df)} summary-article pairs")
    print(f"Merged columns: {merged_df.columns.tolist()}")
    
    # Apply limit if specified
    if limit > 0:
        merged_df = merged_df.head(limit)
        print(f"Processing first {limit} pairs (limit applied)")
    
    # Process each row
    output_file = os.path.join(output_dir, "qwen_extractions.jsonl")
    count = 0
    errors = 0
    
    with open(output_file, 'w', encoding='utf-8') as out:
        for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Extracting cues"):
            # Truncate title for display
            title_display = row['title'][:50] + "..." if len(row['title']) > 50 else row['title']
            print(f"\n📄 Row {idx}: title={title_display}, model={row['model']}, style={row['style']}, domain={row['domain']}")
            
            # Extract cues
            result = extract_cues(
                article_title=row['title'],
                article_text=row['text'],
                summary_text=row['summary'],
                domain=row['domain'],
                templates_dir=templates_dir
            )
            
            # Check for errors
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                errors += 1
            else:
                print(f"✅ Success")
            
            # Add metadata
            result.update({
                'title': row['title'],
                'model': row['model'],
                'style': row['style'],
                'domain': row['domain'],
                'extractor': 'Qwen2.5-7B-Instruct'
            })
            
            # Write to output
            out.write(json.dumps(result, ensure_ascii=False) + '\n')
            out.flush()
            
            count += 1
            
            # Clean up memory
            torch.cuda.empty_cache()
            gc.collect()
    
    print(f"\n{'='*80}")
    print(f"✅ Extraction Complete!")
    print(f"   Processed: {count} pairs")
    print(f"   Errors: {errors}")
    if count > 0:
        print(f"   Success rate: {((count - errors) / count * 100):.1f}%")
    print(f"   Output: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Test on small subset first
    run_extraction(limit=5)