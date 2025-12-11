# extraction/extract_article_cues.py
import openai
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
  python extraction/deepseek_cue_extraction.py
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'templates_dir': 'data/prompt_previews',
    'n_examples': 2,
    'output_dir': 'data/article_cues',
    'output_file': 'data/article_cues/gpt4o_mini_cues.jsonl',
    'model_name': 'gpt-4o-mini',
    'temperature': 0.3,  # Higher for more creative/comprehensive extraction
    'max_tokens': 16384,  # Maximum for GPT-4o-mini to allow 150+ cues
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
    OPTIMIZED prompt to extract 100-150+ cues (matching Gemini performance).
    """
    
    template = load_template(domain, templates_dir)
    
    if not template:
        raise ValueError(f"No template found for domain: {domain}")
    
    # COMPREHENSIVE extraction prompt (optimized for high cue count)
    target = f"""ARTICLE TITLE: {article_title}
DOMAIN: {domain}

ORIGINAL ARTICLE:
{article_text}

TASK: Extract EVERY cultural and contextual cue from the article above. Be EXTREMELY COMPREHENSIVE and extract 100-150+ cues.

EXTRACT ALL OF THESE (with examples):

1. CULTURAL CUES:
✓ Idioms and expressions: "double-edged sword", "history repeating itself", "turning point"
✓ Named people: "Tommy Robinson", "Oswald Mosley" - extract EVERY person mentioned
✓ Named places: "Tower Hamlets", "Whitechapel", "Cable Street", "London" - extract ALL locations
✓ Organizations: "BNP", "English Defence League", "United East End", "Parliament"
✓ Cultural events: "Battle of Cable Street", "curry festival", "protests"
✓ Cultural artifacts: "flags", "Pearly kings and queens", "banners"
✓ Slang/dialect: "cockneys", regional terms
✓ Titles and roles: "Mayor", "MP", "activist", "leader"
✓ Political terms: "far-right", "anti-fascist", "nationalist"

2. CONTEXTUAL CUES:
✓ Discourse markers: "however", "therefore", "rather than", "meanwhile", "nevertheless"
✓ Time markers: "89 years ago", "In the decades since", "This week", "recently", "yesterday"
✓ Specific dates: "1936", "October", "Monday", any date mentioned
✓ Pronouns with context: "his march", "their rally", "its members"
✓ Causal phrases: "as a result", "to prevent disorder", "because of", "due to"
✓ Framing phrases: "double-edged sword", "prime target", "key figure"
✓ Event ordering: "weeks after", "Since then", "before the rally", "following"
✓ Comparison terms: "more than", "less than", "similar to", "unlike"

3. EXTRACT EVERYTHING:
✓ Every person's name AND their title/role separately
✓ Every location (countries, cities, streets, buildings)
✓ Every organization, party, group
✓ Every time reference (dates, periods, durations)
✓ Every number with context: "thousands", "89 years", "500 people"
✓ Every quote or key phrase from officials/experts
✓ Every political/ideological term
✓ Every institution (government bodies, courts, media)
✓ Every social group (demographics, communities)

IMPORTANT INSTRUCTIONS:
- Extract AT LEAST 100-150 cues per article
- Be EXHAUSTIVE - extract EVERY relevant phrase
- Include obvious references too (don't skip common terms)
- For "Democratic Senator Warren" extract: "Democratic", "Senator", "Warren" as separate cues
- Extract both full phrases AND meaningful components
- Use exact text from article
- Keep cues under 10 words (but extract more cues rather than longer ones)

DO NOT EXTRACT:
✗ Generic pronouns alone (it, they, he, she without context)
✗ Simple connectors alone (and, but, or, so, as)
✗ Common verbs alone (is, was, has, have)
✗ Articles alone (the, a, an)

QUALITY AND QUANTITY: Extract as many meaningful cues as possible. Aim for 150+ cues.
Mark "preserved" as null (we will check preservation later).

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
    Handles different possible key names from Gemini AND OpenAI.
    """
    
    cues = []
    
    if 'tables' in extraction_result:
        # Old structure (Gemini tables format)
        for table in extraction_result.get('tables', []):
            for row in table.get('rows', []):
                # Try multiple possible key names for text
                text = (
                    row.get('Text Span (from original)') or
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
        # New structure (standard JSON format)
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

# Update extract_cues_from_article to use OpenAI
def extract_cues_from_article(article_title, article_text, domain, client, templates_dir):
    """
    Extract ALL cues from an article.
    Returns the extraction result.
    """
    
    try:
        # Build prompt
        prompt = build_article_extraction_prompt(
            article_title, 
            article_text, 
            domain, 
            templates_dir
        )
        
        # Call OpenAI API (matching Gemini's JSON response format)
        response = client.chat.completions.create(
            model=CONFIG['model_name'],
            messages=[
                {
                    "role": "system", 
                    "content": "You are an EXPERT at extracting cultural and contextual cues from text. Your goal is to be EXTREMELY COMPREHENSIVE and extract 100-150+ cues per article. Extract EVERY person, place, organization, date, time reference, discourse marker, political term, and cultural reference. Be thorough - extract MORE rather than less. Return only valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=CONFIG['temperature'],
            max_tokens=CONFIG['max_tokens'],
            response_format={"type": "json_object"}
        )
        
        # Parse JSON
        try:
            print("="*80)
            result = json.loads(response.choices[0].message.content)
            return result  # ← Return ONLY result
        except json.JSONDecodeError:
            # Try fixing
            fixed = fix_json_errors(response.choices[0].message.content)
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
    print("STAGE 1: EXTRACT CUES FROM ARTICLES (OpenAI GPT-4o-mini)")
    print("="*80)
    
    # Load environment variables
    load_dotenv()
    
    # Setup OpenAI API
    print("\n🔑 Configuring OpenAI API...")
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ No OPENAI_API_KEY found in .env file!")
        return
    
    client = openai.OpenAI(api_key=api_key)
    
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
    
    # LIMIT TO 5 ARTICLES FOR TESTING
    remaining = remaining.head(5)
    
    print(f"\n📋 Articles to process: {len(remaining)} (LIMITED TO 5 FOR TESTING)")
    print(f"   Total: {len(articles_df)}")
    print(f"   Completed: {len(completed)}")
    print(f"   Remaining (full): {len(articles_df) - len(completed)}")
    print(f"   🎯 Target: ~150 cues per article (matching Gemini)")
    
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
            client=client,
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
                'raw_extraction': extraction_result  # Just the OpenAI response
            }
            
            # Save immediately
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(article_cues, ensure_ascii=False) + '\n')
            
            successful += 1
            
            # Show comparison with Gemini
            print(f"\n✅ Extracted {len(cues)} cues (Gemini avg: ~150)")
            
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
        avg_cues = sum(all_cues)/len(all_cues)
        print(f"   Avg cues per article: {avg_cues:.1f}")
        print(f"   Min cues: {min(all_cues)}")
        print(f"   Max cues: {max(all_cues)}")
        print(f"\n   🎯 Gemini avg: ~150 cues")
        print(f"   📊 GPT-4o-mini avg: {avg_cues:.1f} cues")
        
        # Show performance comparison
        if avg_cues >= 120:
            print(f"   🎉 EXCELLENT! Matching Gemini performance!")
        elif avg_cues >= 80:
            print(f"   👍 GOOD! Close to Gemini performance")
        elif avg_cues >= 50:
            print(f"   ✓ Decent extraction")
        else:
            print(f"   ⚠️  Lower than Gemini - may need prompt optimization")
    
    print("="*80)

# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    load_dotenv()
    extract_all_articles()