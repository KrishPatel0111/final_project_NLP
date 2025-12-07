# extraction/extract_article_cues.py
import google.generativeai as genai
import json
import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv
import re
from datetime import datetime

"""
STAGE 1: EXTRACT ALL CUES FROM ARTICLES

This script extracts ALL cultural and contextual cues from each unique article.
Run this ONCE to generate the canonical set of cues for each article.

Input:
  - data/guardian_articles_raw.csv (original articles)
  
Output:
  - data/article_cues/article_cues.jsonl (one line per article with all cues)
  
Usage:
  python extraction/extract_article_cues.py
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'templates_dir': 'data/prompt_previews',
    'n_examples': 2,
    'output_dir': 'data/article_cues',
    'output_file': 'data/article_cues/gemini_cues.jsonl',
    'model_name': 'models/gemini-2.5-flash',
    'temperature': 0.1,
}

# ============================================================================
# PROMPT BUILDING
# ============================================================================

def load_template(domain, templates_dir):
    """Load prompt template for a domain."""
    
    filename = f"{domain.lower()}_ex_no_{CONFIG['n_examples']}.txt"
    filepath = os.path.join(templates_dir, filename)
    
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Fallback to politics
    fallback = os.path.join(templates_dir, f"politics_{CONFIG['n_examples']}ex.txt")
    if os.path.exists(fallback):
        with open(fallback, 'r', encoding='utf-8') as f:
            return f.read()
    
    return None

def build_article_extraction_prompt(article_title, article_text, domain, templates_dir):
    """
    Build prompt to extract meaningful cues from an article.
    Updated to be more selective and avoid noise.
    """
    
    template = load_template(domain, templates_dir)
    
    if not template:
        raise ValueError(f"No template found for domain: {domain}")
    
    # More selective extraction prompt
    target = f"""ARTICLE TITLE: {article_title}
DOMAIN: {domain}

ORIGINAL ARTICLE:
{article_text}

TASK: Extract cultural and contextual cues from the article above.

CULTURAL CUES TO EXTRACT:
✓ Idioms and fixed expressions (e.g., "double-edged sword", "history repeating itself")
✓ Named people (e.g., "Tommy Robinson", "Oswald Mosley")
✓ Named places (e.g., "Tower Hamlets", "Whitechapel", "Cable Street")
✓ Organizations (e.g., "BNP", "English Defence League", "United East End")
✓ Cultural events (e.g., "Battle of Cable Street", "curry festival")
✓ Cultural artifacts (e.g., "flags", "Pearly kings and queens")
✓ Slang or dialect terms (e.g., "cockneys")

CONTEXTUAL CUES TO EXTRACT:
✓ Discourse markers (e.g., "however", "therefore", "rather than")
✓ Time/sequence markers (e.g., "89 years ago", "In the decades since", "This week")
✓ Specific pronouns with clear antecedents (e.g., "his march" referring to Robinson)
✓ Causal phrases explaining why/how (e.g., "as a result", "to prevent disorder")
✓ Framing phrases that set context (e.g., "double-edged sword", "prime target")
✓ Event ordering cues (e.g., "weeks after", "Since then", "before the rally")

DO NOT EXTRACT:
✗ Generic pronouns without context (it, they, he, she used generally)
✗ Simple connectors (and, but, or, so, as)
✗ Common verbs (is, was, has, have)
✗ Articles (the, a, an)
✗ Very long sentences or phrases (keep cues concise, under 10 words ideally)

QUALITY OVER QUANTITY: Extract only meaningful, distinctive cues.
For now, mark "preserved" as null (we will check preservation later).

OUTPUT YOUR JSON ANALYSIS:
"""
    return template + '\n'+ target

# ============================================================================
# JSON FIXING
# ============================================================================

def fix_json_errors(json_text):
    """Fix common JSON errors."""
    
    json_text = json_text.replace('"Contexttype"', '"Contextual"')
    json_text = json_text.replace('"Cue type"', '"Cue Type"')
    json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
    json_text = json_text.replace('```json', '').replace('```', '')
    
    return json_text.strip()

# ============================================================================
# EXTRACTION
# ============================================================================

def flatten_cues(extraction_result):
    """
    Flatten the nested cue structure into a simple list.
    Handles different possible key names from Gemini.
    """
    
    cues = []
    
    if 'tables' in extraction_result:
        # Old structure
        for table in extraction_result.get('tables', []):
            for row in table.get('rows', []):
                # Try multiple possible key names for text
                text = (
                    row.get('Text Span (from original)') or  # ← NEW! Gemini uses this
                    row.get('Text Span (Exact)') or
                    row.get('Text Span (Exact Quote from Original)') or
                    row.get('text') or
                    ''
                )
                
                # Skip if text is empty
                if not text or text.strip() == '':
                    continue
                
                cue = {
                    'cue_type': row.get('Cue Type', row.get('cue_type', 'unknown')),
                    'subtype': row.get('Subtype', row.get('subtype', 'unknown')),
                    'text': text.strip()
                }
                cues.append(cue)
    
    elif 'cues' in extraction_result:
        # New structure
        for cue in extraction_result.get('cues', []):
            text = cue.get('text', '').strip()
            
            # Skip empty
            if not text:
                continue
            
            cues.append({
                'cue_type': cue.get('cue_type', 'unknown'),
                'subtype': cue.get('subtype', 'unknown'),
                'text': text
            })
    
    return cues

def filter_noisy_cues(cues):
    """
    Remove low-value generic cues.
    """
    
    # Generic single words to skip
    SKIP_WORDS = {
        'while', 'since', 'but', 'yet', 'however', 'although',
    }
    
    filtered = []
    
    for cue in cues:
        text = cue['text'].lower().strip()
        subtype = cue.get('subtype', '').lower()
        
        # Skip single generic words
        if text in SKIP_WORDS:
            continue
        
        # Skip pronouns entirely (too generic)
        if subtype == 'pronouns':
            continue
        
        # Skip very short connectors
        if subtype == 'connector' and len(text) < 4:
            continue
        
        # Skip generic connectors with "said" or "wrote"
        if any(word in text for word in ['said', 'wrote', 'added', 'replied']):
            continue
        
        # Keep everything else
        filtered.append(cue)
    
    return filtered

# Update extract_cues_from_article to use the filter
def extract_cues_from_article(article_title, article_text, domain, model, templates_dir):
    """
    Extract ALL cues from an article.
    Returns the Gemini extraction result.
    """
    
    try:
        # Build prompt
        prompt = build_article_extraction_prompt(
            article_title, 
            article_text, 
            domain, 
            templates_dir
        )
        
        # Call API
        response = model.generate_content(prompt)
        
        # Parse JSON
        try:
            result = json.loads(response.text)
            return result  # ← Return ONLY result
        except json.JSONDecodeError:
            # Try fixing
            fixed = fix_json_errors(response.text)
            result = json.loads(fixed)
            return result  # ← Return ONLY result
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None  # ← Return ONLY None
# ============================================================================
# MAIN PIPELINE
# ============================================================================

def extract_all_articles():
    """
    Main pipeline: Extract cues from all unique articles.
    """
    
    print("="*80)
    print("STAGE 1: EXTRACT CUES FROM ARTICLES")
    print("="*80)
    
    # Setup API
    print("\n🔑 Configuring Gemini API...")
    api_key = input("Enter Gemini API key (or press Enter for env var): ").strip()
    
    if not api_key:
        api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ No API key!")
        return
    
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(
        CONFIG['model_name'],
        generation_config={
            "response_mime_type": "application/json",
            "temperature": CONFIG['temperature'],
        }
    )
    
    print("✅ API configured")
    
    # Load articles
    print("\n📊 Loading articles...")
    articles_file = "data/guardian_articles_raw.csv"
    
    if not os.path.exists(articles_file):
        print(f"❌ File not found: {articles_file}")
        return
    
    articles_df = pd.read_csv(articles_file)
    print(f"   ✅ Loaded {len(articles_df)} articles")
    
    # Check for existing extractions
    output_file = CONFIG['output_file']
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    completed = set()
    
    if os.path.exists(output_file):
        print(f"\n📂 Found existing extractions: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    completed.add(data['title'])
        print(f"   ✅ Already completed: {len(completed)} articles")
    
    # Filter out completed
    remaining = articles_df[~articles_df['title'].isin(completed)]
    
    if len(remaining) == 0:
        print("\n✅ All articles already processed!")
        return
    
    print(f"\n📋 Articles to process: {len(remaining)}")
    print(f"   Total: {len(articles_df)}")
    print(f"   Completed: {len(completed)}")
    print(f"   Remaining: {len(remaining)}")
    
    # Extract cues from each article
    print("\n" + "="*80)
    print("EXTRACTING CUES")
    print("="*80)
    print(f"\nOutput: {output_file}")
    print("Saving after EACH article (safe mode)")
    print("="*80 + "\n")
    
    successful = 0
    failed = 0

    for idx, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Processing"):
        
        # Extract cues
        extraction_result = extract_cues_from_article(
            article_title=row['title'],
            article_text=row['text'],
            domain=row['domain'],
            model=model,
            templates_dir=CONFIG['templates_dir']
        )
        
        if extraction_result:
            # Flatten to simple cue list
            cues = flatten_cues(extraction_result)
            
            # Filter noise (remove generic cues)
            cues = filter_noisy_cues(cues)
            
            # Build output
            article_cues = {
                'title': row['title'],
                'domain': row['domain'],
                'url': row.get('url', ''),
                'extraction_model': CONFIG['model_name'],
                'extraction_timestamp': datetime.now().isoformat(),
                'total_cues': len(cues),
                'cues': cues,
                'raw_extraction': extraction_result  # Just the Gemini response
            }
            
            # Save immediately
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(article_cues, ensure_ascii=False) + '\n')
            
            successful += 1
            
        else:
            # Failed
            failed += 1
            
            print(f"\n❌ Failed: {row['title']}")
            print(f"   Domain: {row['domain']}")
            
            # Save error
            error_file = output_file.replace('.jsonl', '_errors.jsonl')
            with open(error_file, 'a', encoding='utf-8') as f:
                error_entry = {
                    'title': row['title'],
                    'domain': row['domain'],
                    'timestamp': datetime.now().isoformat(),
                    'status': 'FAILED'
                }
                f.write(json.dumps(error_entry, ensure_ascii=False) + '\n')
            
            # Stop on error
            print(f"\n💾 Saved {successful} articles before stopping")
            return
    
    # Summary
    print("\n" + "="*80)
    print("✅ EXTRACTION COMPLETE!")
    print("="*80)
    print(f"\n📊 Statistics:")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Total processed: {successful + failed}")
    print(f"\n📁 Output: {output_file}")
    
    # Show sample statistics
    print(f"\n📈 Sample Statistics:")
    all_cues = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                all_cues.append(data['total_cues'])
    
    if all_cues:
        print(f"   Avg cues per article: {sum(all_cues)/len(all_cues):.1f}")
        print(f"   Min cues: {min(all_cues)}")
        print(f"   Max cues: {max(all_cues)}")
    
    print("="*80)

# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    load_dotenv()
    extract_all_articles()