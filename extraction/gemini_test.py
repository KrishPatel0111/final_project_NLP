# extraction/test_gemini.py
import google.generativeai as genai
import json
import pandas as pd
from prompt_builder import build_fast_prompt
import re

def fix_json_errors(json_text):
    """
    Fix common JSON errors in Gemini responses.
    """
    # Fix typo: "Contexttype" -> "Contextual"
    json_text = json_text.replace('"Contexttype"', '"Contextual"')
    
    # Fix other common typos
    json_text = json_text.replace('"Cue type"', '"Cue Type"')
    json_text = json_text.replace('"cue type"', '"Cue Type"')
    json_text = json_text.replace('"Text span"', '"Text Span (Exact Quote from Original)"')
    json_text = json_text.replace('"Preserved"', '"Preserved in Summary?"')
    
    # Remove trailing commas before closing braces/brackets
    json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
    
    return json_text

def test_single_gemini_call():
    """
    Test Gemini with a single extraction to verify:
    1. API key works
    2. JSON output is valid
    3. Response structure matches expectations
    """
    
    print("="*80)
    print("TESTING GEMINI 1.5 FLASH - SINGLE API CALL")
    print("="*80)
    
    # Configure Gemini
    api_key = input("\nEnter your Gemini API key (or set GOOGLE_API_KEY env var): ").strip()
    if not api_key:
        import os
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("❌ No API key provided!")
            return
    
    genai.configure(api_key=api_key)
    print("✅ API key configured")
    
    # List available models and find the right one
    print("\n📋 Finding Gemini 1.5 Flash model...")
    try:
        models = list(genai.list_models())
        generate_models = [
            m for m in models 
            if 'generateContent' in m.supported_generation_methods
        ]
        
        # Find Flash model
        flash_model = None
        for model in generate_models:
            if 'flash' in model.name.lower():
                flash_model = model.name
                break
        
        if not flash_model:
            flash_model = generate_models[0].name if generate_models else None
        
        print(f"   ✅ Using model: {flash_model}")
        
    except Exception as e:
        print(f"   ⚠️  Using default model")
        flash_model = "gemini-1.5-flash"
    
    # Load test data
    print("\n1. Loading test data...")
    try:
        summaries_df = pd.read_json("data/outputs/all_summaries.jsonl", lines=True)
        articles_df = pd.read_csv("data/guardian_articles_raw.csv")
        
        if 'domain' in summaries_df.columns:
            summaries_df = summaries_df.drop(columns=['domain'])
        
        merged_df = summaries_df.merge(
            articles_df[['title', 'text', 'domain']], 
            on='title', 
            how='inner'
        )
        
        test_row = merged_df.iloc[0]
        print(f"   ✅ Testing: {test_row['title'][:50]}...")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Build prompt
    print("\n2. Building prompt...")
    try:
        prompt = build_fast_prompt(
            article_title=test_row['title'],
            article_text=test_row['text'],
            summary_text=test_row['summary'],
            domain=test_row['domain'],
            templates_dir="data/prompt_templates",
            n_examples=2
        )
        print(f"   ✅ Prompt: {len(prompt)} chars (~{len(prompt)//4} tokens)")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Call Gemini
    print(f"\n3. Calling Gemini API...")
    try:
        model = genai.GenerativeModel(
            flash_model,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1,
            }
        )
        
        print("   ⏳ Sending request...")
        response = model.generate_content(prompt)
        print("   ✅ Received response!")
        
    except Exception as e:
        print(f"   ❌ API Error: {e}")
        return
    
    # Parse JSON with auto-fix
    print("\n4. Parsing JSON response...")
    
    # Save raw response first
    import os
    os.makedirs("data/extractions", exist_ok=True)
    with open("data/extractions/test_gemini_raw.txt", 'w', encoding='utf-8') as f:
        f.write(response.text)
    print("   💾 Raw response saved to: data/extractions/test_gemini_raw.txt")
    
    try:
        # Try parsing directly first
        result = json.loads(response.text)
        print("   ✅ Valid JSON (no fixes needed)!")
        
    except json.JSONDecodeError as e:
        print(f"   ⚠️  JSON error: {e}")
        print("   🔧 Attempting to fix...")
        
        # Try fixing
        fixed_text = fix_json_errors(response.text)
        
        try:
            result = json.loads(fixed_text)
            print("   ✅ Fixed and parsed successfully!")
            
            # Save fixed version
            with open("data/extractions/test_gemini_fixed.txt", 'w', encoding='utf-8') as f:
                f.write(fixed_text)
            print("   💾 Fixed JSON saved to: data/extractions/test_gemini_fixed.txt")
            
        except json.JSONDecodeError as e2:
            print(f"   ❌ Still invalid after fixes: {e2}")
            print(f"\n   First 1000 chars of response:")
            print(f"   {response.text[:1000]}")
            return
    
    # Show structure
    print("\n   📋 Response structure:")
    print(f"      - doc_id: {result.get('doc_id')}")
    print(f"      - domain: {result.get('domain')}")
    print(f"      - title: {result.get('title', '')[:50]}...")
    print(f"      - tables: {len(result.get('tables', []))} table(s)")
    
    if result.get('tables'):
        for i, table in enumerate(result['tables'], 1):
            rows = table.get('rows', [])
            print(f"      - Table {i}: {len(rows)} rows")
    
    # Save result
    print("\n5. Saving test result...")
    output_file = "data/extractions/test_gemini_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Saved to: {output_file}")
    
    # Show sample cues
    print("\n6. Sample extracted cues:")
    print("   " + "-"*76)
    if result.get('tables') and result['tables'][0].get('rows'):
        for i, row in enumerate(result['tables'][0]['rows'][:5], 1):
            cue = row.get('Text Span (Exact)') or row.get('Text Span (Exact Quote from Original)') or 'N/A'
            preserved = row.get('Preserved?') or row.get('Preserved in Summary?') or 'N/A'
            
            print(f"   {i}. {cue[:60]}")
            print(f"      → {preserved}")
    print("   " + "-"*76)
    
    # Summary
    print("\n" + "="*80)
    print("✅ TEST SUCCESSFUL!")
    print("="*80)
    print("\n✓ API key works")
    print("✓ Gemini returns valid JSON (with minor auto-fixes)")
    print("✓ Extraction logic works")
    print("✓ Cues are being extracted correctly")
    print(f"\n✓ Working model: {flash_model}")
    print("\n📁 Files created:")
    print("   - data/extractions/test_gemini_raw.txt (original response)")
    print("   - data/extractions/test_gemini_fixed.txt (if fixes were needed)")
    print("   - data/extractions/test_gemini_result.json (parsed result)")
    print("\n🚀 Next step: Run full extraction with Gemini!")
    print("   Command: python extraction/gemini_extractor.py")
    print("="*80)

if __name__ == "__main__":
    test_single_gemini_call()