"""
Stage 2: Check Cue Preservation in Summaries
Reads pre-extracted article cues and checks if they're preserved in each summary.
"""

import pandas as pd
import json
import os
from datetime import datetime
from tqdm import tqdm
import google.generativeai as genai
from pathlib import Path
import time
from dotenv import load_dotenv
load_dotenv() 

# Configure Gemini
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

# Paths
ARTICLE_CUES_FILE = 'data/article_cues/gemini_cues_fixed.jsonl'
SUMMARIES_FILE = 'data/outputs/all_summaries.jsonl'
OUTPUT_FILE = 'data/extractions/preservation_results.jsonl'
ERROR_LOG_FILE = 'data/extractions/preservation_errors.log'

# Create output directory
Path('data/extractions').mkdir(parents=True, exist_ok=True)

# Initialize error log
def log_error(error_msg):
    """Log errors to file and continue processing."""
    timestamp = datetime.now().isoformat()
    with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {error_msg}\n")
    print(f"❌ ERROR LOGGED: {error_msg}")


def load_article_cues():
    """Load all pre-extracted article cues into memory."""
    print("📂 Loading article cues...")
    article_cues = {}
    
    with open(ARTICLE_CUES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            title = data['title']
            article_cues[title] = data
    
    print(f"✅ Loaded cues for {len(article_cues)} articles")
    return article_cues


def load_summaries():
    """Load all summaries to process."""
    print("📂 Loading summaries...")
    summaries = []
    
    with open(SUMMARIES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            summaries.append(json.loads(line))
    
    print(f"✅ Loaded {len(summaries)} summaries")
    return summaries


def get_processed_summaries():
    """Get set of already processed summaries for resume capability."""
    if not os.path.exists(OUTPUT_FILE):
        return set()
    
    processed = set()
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Create unique key: title + model + format
            key = f"{data['title']}|{data['summary_model']}|{data['summary_format']}"
            processed.add(key)
    
    print(f"✅ Found {len(processed)} already processed summaries (resuming)")
    return processed


def create_preservation_prompt(cues, summary_text):
    """Create Gemini prompt to check if cues are preserved in summary."""
    
    # Convert cues to JSON string
    cues_json = json.dumps(cues, indent=2, ensure_ascii=False)
    
    prompt = f"""You are a JSON output machine. You ONLY output valid JSON arrays. No markdown, no explanation, no preamble.

TASK: For each cue below, add a "preserved" field with "Yes" or "No" indicating if it appears in the summary.

CUES:
{cues_json}

SUMMARY:
{summary_text}

RULES:
- "Yes" = cue appears exactly, with minor variation, or paraphrased with same meaning
- "No" = cue is missing or meaning changed

CRITICAL REQUIREMENTS:
1. Return ONLY a JSON array (starts with [ and ends with ])
2. NO markdown formatting, NO ```json blocks, NO explanatory text
3. NO trailing commas (commas before ] or }} are INVALID)
4. Use double quotes for all strings
5. Return the EXACT same structure as input, just add "preserved": "Yes" or "preserved": "No"

EXAMPLE OUTPUT FORMAT:
[
  {{"cue_type": "Cultural", "subtype": "Idiom", "text": "example text", "preserved": "Yes"}},
  {{"cue_type": "Cultural", "subtype": "Named person", "text": "John Doe", "preserved": "No"}}
]

NOW OUTPUT THE JSON ARRAY:"""
    
    return prompt


import openai

def check_preservation_with_gemini(cues, summary_text, batch_size=20):
    """Use Gemini to check which cues are preserved (processes in batches)."""
    
    all_results = []
    
    # Process cues in batches
    for i in range(0, len(cues), batch_size):
        batch = cues[i:i+batch_size]
        
        try:
            prompt = f"""Check if each cue appears in the summary.

CUES (batch {i//batch_size + 1}):
{json.dumps(batch, ensure_ascii=False)}

SUMMARY:
{summary_text}

Add "preserved": "Yes" if cue appears (exactly, paraphrased, or minor variation).
Add "preserved": "No" if missing or meaning changed.

Return the same array with "preserved" field added."""

            model = genai.GenerativeModel("gemini-2.5-flash")  # Try 1.5 instead
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=8000,
                    response_mime_type="application/json"
                )
            )
            
            # Parse result
            result = json.loads(response.text)
            
            # Extract array if wrapped
            if isinstance(result, dict):
                for key in ['cues', 'result', 'data', 'items', 'batch']:
                    if key in result and isinstance(result[key], list):
                        result = result[key]
                        break
            
            if isinstance(result, list):
                all_results.extend(result)
            else:
                log_error(f"Batch {i//batch_size + 1}: Expected list, got {type(result)}")
                # Add all as "No" to continue
                for cue in batch:
                    cue['preserved'] = "No"
                    all_results.append(cue)
            
            # Small delay between batches
            time.sleep(0.5)
            
        except json.JSONDecodeError as e:
            log_error(f"Batch {i//batch_size + 1} JSON error: {str(e)}")
            # Add all as "No" to continue
            for cue in batch:
                cue['preserved'] = "No"
                all_results.append(cue)
                
        except Exception as e:
            log_error(f"Batch {i//batch_size + 1} error: {str(e)}")
            # Add all as "No" to continue
            for cue in batch:
                cue['preserved'] = "No"
                all_results.append(cue)
    
    return all_results if len(all_results) > 0 else None

def process_summary(summary_data, article_cues_dict):
    """Process a single summary and check cue preservation."""
    
    title = summary_data['title']
    summary_model = summary_data['model']
    summary_format = summary_data['style']
    summary_text = summary_data['summary']
    
    # Get article cues
    if title not in article_cues_dict:
        log_error(f"No article cues found for: {title}")
        return None
    
    article_data = article_cues_dict[title]
    cues = article_data['cues']
    
    if len(cues) == 0:
        log_error(f"Article has 0 cues: {title}")
        return None
    
    # Check preservation with Gemini (returns cues with "preserved" field added)
    preserved_cues = check_preservation_with_gemini(cues, summary_text)
    
    if preserved_cues is None:
        return None
    
    # Convert "Yes"/"No" to boolean and count preserved
    cues_preserved = 0
    for cue in preserved_cues:
        # Convert Yes/No to boolean
        preserved_value = cue.get('preserved', 'No')
        cue['preserved'] = (preserved_value.lower() == 'yes')
        if cue['preserved']:
            cues_preserved += 1
    
    # Calculate preservation rate
    preservation_rate = cues_preserved / len(cues) if len(cues) > 0 else 0.0
    
    # Create result
    result = {
        'title': title,
        'url': summary_data.get('url', ''),
        'domain': summary_data.get('domain', ''),
        'summary_model': summary_model,
        'summary_format': summary_format,
        'extraction_model': article_data['extraction_model'],
        'extraction_timestamp': article_data['extraction_timestamp'],
        'preservation_timestamp': datetime.now().isoformat(),
        'article_total_cues': len(cues),
        'cues_preserved': cues_preserved,
        'preservation_rate': round(preservation_rate, 4),
        'preservation_details': preserved_cues  # Same structure, just with preserved: true/false
    }
    
    return result

def main():
    """Main processing loop."""
    
    # ========== TEST MODE: Process only 1 summary ==========
    TEST_MODE = False
    TEST_LIMIT = 5
    # =======================================================
    
    print("=" * 80)
    print("STAGE 2: CHECKING CUE PRESERVATION IN SUMMARIES")
    if TEST_MODE:
        print(f"⚠️  TEST MODE: Processing only {TEST_LIMIT} summary")
    print("=" * 80)
    
    # Load data
    article_cues_dict = load_article_cues()
    summaries = load_summaries()
    processed = get_processed_summaries()
    
    # Filter summaries to process
    to_process = []
    for summary in summaries:
        key = f"{summary['title']}|{summary['model']}|{summary['style']}"
        if key not in processed:
            to_process.append(summary)
    
    # ========== LIMIT TO TEST_LIMIT SUMMARIES ==========
    if TEST_MODE:
        to_process = to_process[:TEST_LIMIT]
    # ==================================================
    
    print(f"\n📊 Summary:")
    print(f"   Total summaries: {len(summaries)}")
    print(f"   Already processed: {len(processed)}")
    print(f"   To process: {len(to_process)}")
    
    if len(to_process) == 0:
        print("\n✅ All summaries already processed!")
        return
    
    # Process summaries
    print(f"\n🔄 Processing {len(to_process)} summaries...")
    print(f"   Output: {OUTPUT_FILE}")
    print(f"   Error log: {ERROR_LOG_FILE}")
    print()
    
    success_count = 0
    error_count = 0
    
    for summary_data in tqdm(to_process, desc="Processing summaries"):
        try:
            result = process_summary(summary_data, article_cues_dict)
            
            if result is not None:
                # Save result
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                success_count += 1
            else:
                error_count += 1
            
            # Rate limiting: small delay between API calls
            time.sleep(0.5)
            
        except Exception as e:
            error_count += 1
            log_error(f"Failed to process summary: {summary_data.get('title', 'Unknown')} - {str(e)}")
            continue
    
    # Final summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"✅ Successfully processed: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📁 Results saved to: {OUTPUT_FILE}")
    
    if error_count > 0:
        print(f"⚠️  Check error log: {ERROR_LOG_FILE}")

if __name__ == "__main__":
    main()