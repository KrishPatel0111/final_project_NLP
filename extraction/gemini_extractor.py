# extraction/gemini_extractor_production.py
import google.generativeai as genai
import json
import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv
import re
from datetime import datetime
from pathlib import Path

"""
PRODUCTION CUE EXTRACTION SCRIPT

Extracts cultural and contextual cues from all summaries in all_summaries.jsonl
using Gemini 2.5 Flash with location tracking to be added later.

Input: 
  - data/outputs/all_summaries.jsonl (summaries from different models/styles)
  - data/guardian_articles_raw.csv (original articles)
  
Output:
  - data/extractions/gemini_extractions_production.jsonl (one extraction per line)
  - data/extractions/extraction_metadata.json (statistics and progress info)
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'templates_dir': 'data/prompt_previews',
    'n_examples': 2,  # Number of few-shot examples per domain
    'output_file': 'data/extractions/gemini_extractions_production.jsonl',
    'metadata_file': 'data/extractions/extraction_metadata.json',
    'model_name': 'models/gemini-2.5-flash',
    'temperature': 0.1,
    'save_every': 1,  # Save after EVERY extraction (safer)
}

# Global template cache
TEMPLATES = {}

# ============================================================================
# TEMPLATE LOADING
# ============================================================================

def load_templates():
    """
    Load all prompt templates at startup (only once).
    This is much faster than rebuilding prompts each time.
    """
    
    global TEMPLATES
    
    template_dir = CONFIG['templates_dir']
    domains = ["educational", "politics", "cultural", "sport", "technology", "social"]
    n_examples = CONFIG['n_examples']
    
    print(f"\n📋 Loading prompt templates from: {template_dir}/")
    
    for domain in domains:
        filename = f"{domain.lower()}_ex_no_{n_examples}.txt"
        filepath = os.path.join(template_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                TEMPLATES[domain] = f.read()
            print(f"   ✅ Loaded: {domain} ({len(TEMPLATES[domain]):,} chars)")
        else:
            print(f"   ⚠️  Missing: {filename}")
    
    if not TEMPLATES:
        print("\n❌ No templates loaded!")
        print("   Please run: python extraction/generate_prompt_templates.py")
        return False
    
    print(f"\n✅ Loaded {len(TEMPLATES)} prompt templates")
    return True

# ============================================================================
# PROMPT BUILDING
# ============================================================================

def format_target(article_title, article_text, summary_text, domain):
    """
    Format target article/summary to append to template.
    This is called for each extraction.
    """
    
    return f"""ARTICLE TITLE: {article_title}
DOMAIN: {domain}

ORIGINAL ARTICLE (identify cues from here):
{article_text}

SUMMARY (check if cues are preserved here):
{summary_text}

OUTPUT YOUR JSON ANALYSIS:
"""

# ============================================================================
# JSON FIXING
# ============================================================================

def fix_json_errors(json_text):
    """
    Fix common JSON errors in Gemini responses.
    """
    
    # Fix common typos
    json_text = json_text.replace('"Contexttype"', '"Contextual"')
    json_text = json_text.replace('"Cue type"', '"Cue Type"')
    json_text = json_text.replace('"cue type"', '"Cue Type"')
    
    # Remove trailing commas before closing braces/brackets
    json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
    
    # Remove markdown code blocks if present
    json_text = json_text.replace('```json', '').replace('```', '')
    
    return json_text.strip()

# ============================================================================
# EXTRACTION
# ============================================================================

def extract_with_gemini(article_title, article_text, summary_text, domain, model):
    """
    Extract cues using Gemini with pre-loaded template.
    
    Returns:
        dict: Extraction result or None if failed
    """
    
    # Get template for this domain
    template = TEMPLATES.get(domain)
    
    if not template:
        print(f"\n⚠️  No template for domain: {domain}, using Politics template")
        template = TEMPLATES.get("Politics", "")
        
        if not template:
            print(f"\n❌ No templates available!")
            return None
    
    # Format target article (fast string concatenation)
    target = format_target(article_title, article_text, summary_text, domain)
    
    # Combine template + target
    prompt = template + target
    print(prompt)
    
    try:
        # Call Gemini API
        response = model.generate_content(prompt)
        
        # Get response text
        json_text = response.text
        
        # Try parsing directly
        try:
            result = json.loads(json_text)
            return result
        except json.JSONDecodeError:
            # Try fixing common errors
            fixed_text = fix_json_errors(json_text)
            try:
                result = json.loads(fixed_text)
                return result
            except json.JSONDecodeError as e:
                print(f"\n❌ JSON parse error: {e}")
                print(f"   First 200 chars: {json_text[:200]}")
                return None
        
    except Exception as e:
        print(f"\n❌ API Error: {e}")
        return None

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """
    Load summaries and articles, merge them by title.
    
    Returns:
        DataFrame: Merged data with all necessary columns
    """
    
    print("\n📊 Loading data...")
    
    # Load summaries
    summaries_file = "data/outputs/all_summaries.jsonl"
    if not os.path.exists(summaries_file):
        print(f"❌ File not found: {summaries_file}")
        return None
    
    summaries_df = pd.read_json(summaries_file, lines=True)
    print(f"   ✅ Loaded {len(summaries_df)} summaries")
    print(f"   Columns: {list(summaries_df.columns)}")
    
    # Load articles
    articles_file = "data/guardian_articles_raw.csv"
    if not os.path.exists(articles_file):
        print(f"❌ File not found: {articles_file}")
        return None
    
    articles_df = pd.read_csv(articles_file)
    print(f"   ✅ Loaded {len(articles_df)} articles")
    
    # Merge on title
    merged_df = summaries_df.merge(
        articles_df[['title', 'text', 'domain']], 
        on='title', 
        how='inner',
        suffixes=('_summary', '_article')
    )
    # Add this at the top of run_extraction() after loading data:
    merged_df = merged_df.head(5)  # Test with 5 first
    
    # Handle domain column conflict if it exists
    if 'domain_summary' in merged_df.columns and 'domain_article' in merged_df.columns:
        merged_df = merged_df.drop(columns=['domain_summary'])
        merged_df = merged_df.rename(columns={'domain_article': 'domain'})
    
    print(f"   ✅ Merged: {len(merged_df)} article-summary pairs")
    
    return merged_df

# ============================================================================
# PROGRESS TRACKING
# ============================================================================

def load_existing_extractions():
    """
    Load existing extractions to resume from where we left off.
    
    Returns:
        set: Set of (title, model, style) tuples already processed
    """
    
    output_file = CONFIG['output_file']
    
    if not os.path.exists(output_file):
        return set()
    
    completed = set()
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    extraction = json.loads(line)
                    key = (
                        extraction.get('article_title'),
                        extraction.get('summary_model'),
                        extraction.get('summary_style')
                    )
                    completed.add(key)
        
        print(f"\n📂 Found {len(completed)} existing extractions")
        print(f"   Will resume from where we left off")
        
    except Exception as e:
        print(f"\n⚠️  Error loading existing extractions: {e}")
        return set()
    
    return completed

def save_extraction(extraction, mode='a'):
    """
    Save a single extraction to the output file.
    
    Args:
        extraction: Dict to save
        mode: 'a' for append, 'w' for write (overwrite)
    """
    
    output_file = CONFIG['output_file']
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, mode, encoding='utf-8') as f:
        f.write(json.dumps(extraction, ensure_ascii=False) + '\n')

def save_metadata(stats):
    """
    Save extraction metadata (statistics, progress, etc.)
    """
    
    metadata_file = CONFIG['metadata_file']
    
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

# ============================================================================
# MAIN EXTRACTION PIPELINE
# ============================================================================

def run_extraction():
    """
    Main extraction pipeline.
    Process all summaries and extract cues using Gemini.
    """
    
    print("="*80)
    print("PRODUCTION CUE EXTRACTION WITH GEMINI 2.5 FLASH")
    print("="*80)
    
    # Configure Gemini API
    print("\n🔑 Configuring Gemini API...")
    api_key = input("Enter your Gemini API key (or press Enter to use GOOGLE_API_KEY env var): ").strip()
    
    if not api_key:
        api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ No API key provided!")
        print("   Get a key from: https://aistudio.google.com/app/apikey")
        return
    
    genai.configure(api_key=api_key)
    print("✅ API key configured")
    
    # Create Gemini model
    print(f"\n🤖 Creating model: {CONFIG['model_name']}")
    try:
        model = genai.GenerativeModel(
            CONFIG['model_name'],
            generation_config={
                "response_mime_type": "application/json",
                "temperature": CONFIG['temperature'],
            }
        )
        print("✅ Model created")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return
    
    # Load templates
    if not load_templates():
        return
    
    # Load data
    merged_df = load_data()
    if merged_df is None:
        return
    
    # Load existing extractions (for resuming)
    completed = load_existing_extractions()
    
    # Filter out already completed
    if completed:
        merged_df['key'] = merged_df.apply(
            lambda row: (row['title'], row['model'], row['style']), 
            axis=1
        )
        merged_df = merged_df[~merged_df['key'].isin(completed)]
        merged_df = merged_df.drop(columns=['key'])
        print(f"\n📋 Remaining to process: {len(merged_df)} summaries")
    
    if len(merged_df) == 0:
        print("\n✅ All extractions already completed!")
        return
    
    # Statistics
    stats = {
        'start_time': datetime.now().isoformat(),
        'total_summaries': len(merged_df),
        'completed': len(completed),
        'successful': 0,
        'failed': 0,
        'config': CONFIG.copy()
    }
    
    print("\n" + "="*80)
    print("STARTING EXTRACTION")
    print("="*80)
    print(f"\nTotal to process: {len(merged_df)}")
    print(f"Already completed: {len(completed)}")
    print(f"Saving to: {CONFIG['output_file']}")
    print(f"\n⚠️  Note: Saving after EVERY extraction (safe mode)")
    print("="*80)
    
    # Process each summary
    print("\n🔄 Extracting cues...")
    
    for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Processing"):
        
        # Extract cues
        try:
            result = extract_with_gemini(
                article_title=row['title'],
                article_text=row['text'],
                summary_text=row['summary'],
                domain=row['domain'],
                model=model
            )
            
            if result:
                # Add metadata
                extraction = {
                    'article_title': row['title'],
                    'summary_model': row['model'],  # Model that generated the summary
                    'summary_style': row['style'],  # Style of the summary
                    'extraction_model': CONFIG['model_name'],  # Model used for extraction
                    'domain': row['domain'],
                    'url': row.get('url', ''),
                    'doc_id': row.get('doc_id', -1),
                    'extraction_timestamp': datetime.now().isoformat(),
                    'extraction_result': result
                }
                
                # Save immediately
                save_extraction(extraction, mode='a')
                
                stats['successful'] += 1
                
            else:
                # Extraction failed
                stats['failed'] += 1
                
                # Save error info
                error_entry = {
                    'article_title': row['title'],
                    'summary_model': row['model'],
                    'summary_style': row['style'],
                    'domain': row['domain'],
                    'extraction_model': CONFIG['model_name'],
                    'extraction_timestamp': datetime.now().isoformat(),
                    'status': 'FAILED',
                    'error': 'Extraction returned None'
                }
                
                # Save to separate error log
                error_file = CONFIG['output_file'].replace('.jsonl', '_errors.jsonl')
                os.makedirs(os.path.dirname(error_file), exist_ok=True)
                with open(error_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(error_entry, ensure_ascii=False) + '\n')
                
                # Stop on error as requested
                print(f"\n❌ Extraction failed for: {row['title']}")
                print(f"   Model: {row['model']}, Style: {row['style']}")
                print(f"\n💾 Saved {stats['successful']} successful extractions before stopping")
                print(f"   Output: {CONFIG['output_file']}")
                print(f"   Errors: {error_file}")
                
                # Save final metadata
                stats['end_time'] = datetime.now().isoformat()
                stats['status'] = 'STOPPED_ON_ERROR'
                save_metadata(stats)
                
                return
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print(f"   Article: {row['title']}")
            print(f"   Model: {row['model']}, Style: {row['style']}")
            
            stats['failed'] += 1
            
            # Save metadata before stopping
            stats['end_time'] = datetime.now().isoformat()
            stats['status'] = 'STOPPED_ON_ERROR'
            save_metadata(stats)
            
            print(f"\n💾 Saved {stats['successful']} successful extractions before stopping")
            
            return
    
    # All completed successfully
    stats['end_time'] = datetime.now().isoformat()
    stats['status'] = 'COMPLETED'
    save_metadata(stats)
    
    print("\n" + "="*80)
    print("✅ EXTRACTION COMPLETE!")
    print("="*80)
    print(f"\n📊 Statistics:")
    print(f"   Total processed: {stats['successful'] + stats['failed']}")
    print(f"   Successful: {stats['successful']}")
    print(f"   Failed: {stats['failed']}")
    print(f"   Success rate: {stats['successful']/(stats['successful']+stats['failed'])*100:.1f}%")
    print(f"\n📁 Output files:")
    print(f"   Extractions: {CONFIG['output_file']}")
    print(f"   Metadata: {CONFIG['metadata_file']}")
    if stats['failed'] > 0:
        error_file = CONFIG['output_file'].replace('.jsonl', '_errors.jsonl')
        print(f"   Errors: {error_file}")
    print("="*80)

# ============================================================================
# CLI
# ============================================================================

def main():
    """
    Command-line interface.
    """
    
    import sys
    load_dotenv()
    
    # Check for test mode
    if '--test' in sys.argv:
        print("\n🧪 TEST MODE: Will process only 5 summaries")
        # Modify merged_df in run_extraction to limit to 5
        CONFIG['test_mode'] = True
    
    run_extraction()

if __name__ == "__main__":
    main()