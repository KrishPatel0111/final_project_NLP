# extraction/test_deepseek.py

import json
import pandas as pd
import requests
import re
import os
from prompt_builder import build_fast_prompt

# NEW: Auto-load .env file
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# JSON FIXING (same as Gemini logic)
# =============================================================================

def fix_json_errors(json_text):
    """
    Fix common JSON errors in model responses.
    """

    json_text = json_text.replace('"Contexttype"', '"Contextual"')
    json_text = json_text.replace('"Cue type"', '"Cue Type"')
    json_text = json_text.replace('"cue type"', '"Cue Type"')
    json_text = json_text.replace('"Text span"', '"Text Span (Exact Quote from Original)"')
    json_text = json_text.replace('"Preserved"', '"Preserved in Summary?"')

    # Remove trailing commas
    json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)

    # Remove accidental markdown fences
    json_text = json_text.replace("```json", "").replace("```", "")

    return json_text.strip()


# =============================================================================
# DeepSeek API CALL
# =============================================================================

def deepseek_call(prompt, model="deepseek-chat", temperature=0.1):
    """
    Sends a request to DeepSeek Chat Completions API.
    """

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "❌ Missing DEEPSEEK_API_KEY! "
            "Make sure it is in your .env file as: DEEPSEEK_API_KEY=your_key_here"
        )

    url = "https://api.deepseek.com/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        print("\n❌ DeepSeek API error:")
        print(response.text)
        return None

    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("\n❌ Unexpected DeepSeek format:", response.text)
        return None


# =============================================================================
# MAIN TEST FUNCTION
# =============================================================================

def test_single_deepseek_call():
    """
    Tests whether DeepSeek API returns valid JSON extraction output.
    """

    print("="*80)
    print("TESTING DEEPSEEK v3.2 - SINGLE API CALL")
    print("="*80)

    # NEW: Check for API key loaded from .env
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("\n❌ DEEPSEEK_API_KEY missing!")
        print("➡ Add it to .env as: DEEPSEEK_API_KEY=your_key_here")
        return
    else:
        print("✅ DeepSeek API key detected")

    # Load test data
    print("\n1. Loading test data...")

    try:
        summaries_df = pd.read_json("data/outputs/all_summaries.jsonl", lines=True)
        articles_df = pd.read_csv("data/guardian_articles_raw.csv")

        if "domain" in summaries_df.columns:
            summaries_df = summaries_df.drop(columns=["domain"])

        merged_df = summaries_df.merge(
            articles_df[["title", "text", "domain"]],
            on="title",
            how="inner"
        )

        test_row = merged_df.iloc[0]
        print(f"   ✅ Testing article: {test_row['title'][:60]}...")

    except Exception as e:
        print("❌ Error loading test data:", e)
        return

    # Build prompt
    print("\n2. Building prompt...")

    try:
        prompt = build_fast_prompt(
            article_title=test_row["title"],
            article_text=test_row["text"],
            summary_text=test_row["summary"],
            domain=test_row["domain"],
            templates_dir="data/prompt_templates",
            n_examples=2
        )

        print(f"   ✅ Prompt built ({len(prompt)} chars)")

    except Exception as e:
        print("❌ Error building prompt:", e)
        return

    # Call DeepSeek
    print("\n3. Calling DeepSeek API...")
    print("   ⏳ Sending request...")

    response_text = deepseek_call(prompt)

    if response_text is None:
        print("❌ DeepSeek call failed — no response")
        return

    print("   ✅ Response received!")

    # Save raw output
    os.makedirs("data/extractions", exist_ok=True)
    raw_path = "data/extractions/test_deepseek_raw.txt"

    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(response_text)

    print(f"   💾 Raw response saved to: {raw_path}")

    # Attempt JSON parsing
    print("\n4. Parsing JSON response...")

    try:
        result = json.loads(response_text)
        print("   ✅ Valid JSON (no fixes needed!)")

    except json.JSONDecodeError:
        print("   ⚠️ JSON parsing failed — applying fixer...")
        fixed = fix_json_errors(response_text)

        try:
            result = json.loads(fixed)
            print("   ✅ JSON fixed and parsed successfully")

            fixed_path = "data/extractions/test_deepseek_fixed.txt"
            with open(fixed_path, "w", encoding="utf-8") as f:
                f.write(fixed)

            print(f"   💾 Fixed JSON saved to: {fixed_path}")

        except json.JSONDecodeError as e2:
            print("❌ Still invalid after fixes:", e2)
            print("\nFirst 500 chars of response:")
            print(response_text[:500])
            return

    # Preview structure
    print("\n📋 Response structure preview:")
    print(f"   - domain: {result.get('domain')}")
    print(f"   - title: {result.get('title', '')[:60]}")
    print(f"   - tables: {len(result.get('tables', []))}")

    # Display sample cues
    print("\n5. Sample extracted cues:")
    print("-" * 70)

    if result.get("tables") and result["tables"][0].get("rows"):
        for row in result["tables"][0]["rows"][:5]:
            cue = row.get("Text Span (Exact Quote from Original)", "N/A")
            preserved = row.get("Preserved in Summary?", "N/A")
            print(f"• {cue[:80]}")
            print(f"    → {preserved}")
    else:
        print("(No cue rows found.)")

    # Save final JSON
    result_path = "data/extractions/test_deepseek_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n6. Saving parsed result...")
    print(f"   💾 Saved to: {result_path}")

    # Summary
    print("\n" + "="*80)
    print("✅ TEST SUCCESSFUL — DEEPSEEK IS WORKING!")
    print("="*80)
    print("✓ API key valid (loaded from .env)")
    print("✓ DeepSeek returned structured output")
    print("✓ JSON parsed (with auto-fixes)")
    print("✓ Cue extraction previewed")
    print("\n🚀 Ready to run full extraction:")
    print("   python extraction/deepseek_extractor_production.py")
    print("="*80)


if __name__ == "__main__":
    test_single_deepseek_call()
