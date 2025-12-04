# extraction/gemini_extractor.py
import google.generativeai as genai
import json
import pandas as pd
from tqdm import tqdm
import os
from prompt_builder_fast import build_fast_prompt
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)  # Get free key from https://aistudio.google.com/app/apikey

def extract_with_gemini(article_title, article_text, summary_text, domain, templates_dir):
    """
    Extract cues using Gemini 1.5 Flash with JSON mode.
    """
    # Build prompt
    prompt = build_fast_prompt(
        article_title=article_title,
        article_text=article_text,
        summary_text=summary_text,
        domain=domain,
        templates_dir=templates_dir,
        n_examples=3  # Can use 3 examples - Gemini has huge context
    )
    
    # Create model with JSON response mode
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config={
            "response_mime_type": "application/json",  # FORCES JSON OUTPUT
            "temperature": 0.1,
        }
    )
    
    try:
        response = model.generate_content(prompt)
        result = json.loads(response.text)
        return result
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error for '{article_title}': {e}")
        return None
    except Exception as e:
        print(f"❌ Error processing '{article_title}': {e}")
        return None

def main():
    """Run extraction on all summaries using Gemini."""
    
    print("="*80)
    print("GEMINI CUE EXTRACTION")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    summaries_df = pd.read_json("data/outputs/all_summaries.jsonl", lines=True)
    articles_df = pd.read_csv("data/guardian_articles_raw.csv")
    
    # Merge
    print("\n2. Merging articles and summaries...")
    if 'domain' in summaries_df.columns:
        summaries_df = summaries_df.drop(columns=['domain'])
    
    merged_df = summaries_df.merge(
        articles_df[['title', 'text', 'domain']], 
        on='title', 
        how='inner'
    )
    print(f"   Merged: {len(merged_df)} article-summary pairs")
    
    # Test mode: limit to first 10
    limit = 10  # SET TO 0 FOR FULL RUN
    if limit > 0:
        merged_df = merged_df.head(limit)
        print(f"\n⚠️  TEST MODE: Processing only {limit} samples")
    
    # Extract
    print("\n3. Extracting cues with Gemini...")
    results = []
    
    for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
        result = extract_with_gemini(
            article_title=row['title'],
            article_text=row['text'],
            summary_text=row['summary'],
            domain=row['domain'],
            templates_dir="data/prompt_templates"
        )
        
        if result:
            result['model'] = row['model']
            result['style'] = row['style']
            results.append(result)
        
        # Save incrementally every 10
        if len(results) % 10 == 0 and len(results) > 0:
            output_file = "data/extractions/gemini_extractions.jsonl"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(f"\n💾 Saved {len(results)} extractions")
    
    # Final save
    print(f"\n✅ Completed: {len(results)}/{len(merged_df)} successful extractions")
    output_file = "data/extractions/gemini_extractions.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"✅ Saved to: {output_file}")

if __name__ == "__main__":
    main()