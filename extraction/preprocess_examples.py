# extraction/preprocess_examples.py
import json
import os
from typing import Dict, List
from collections import defaultdict

def load_labeled_data(filepath: str) -> List[Dict]:
    """Load labeled data (now with article_text and summary_text included)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def group_examples_by_domain(examples: List[Dict]) -> Dict[str, List[Dict]]:
    """Group examples by their domain."""
    domain_map = defaultdict(list)
    for example in examples:
        domain = example.get('domain', 'Unknown')
        domain_map[domain].append(example)
    return dict(domain_map)

def create_example_prompt_text(example: Dict) -> str:
    """
    Convert a single labeled example into a formatted prompt string.
    Handles complex nested JSON structures properly.
    """
    
    # Get article text (truncate if too long to save tokens)
    article_text = example.get('article_text', '[Article text not available]')
    if len(article_text) > 3000:  # Truncate very long articles
        article_text = article_text[:3000] + "\n[...article truncated for brevity...]"
    
    # Get summary text  
    summary_text = example.get('summary_text', '[Summary text not available]')
    
    # Create a clean copy of the example for the expected output
    # Remove the article_text and summary_text from the expected output
    # since they're already shown above
    expected_output = {k: v for k, v in example.items() 
                      if k not in ['article_text', 'summary_text']}
    
    # Convert to JSON string with proper formatting
    # Use ensure_ascii=False to keep special characters readable
    expected_json = json.dumps(expected_output, indent=2, ensure_ascii=False)
    
    prompt = f"""ARTICLE TITLE: {example.get('title', 'Untitled')}
DOMAIN: {example.get('domain', 'Unknown')}

ORIGINAL ARTICLE (identify cues from here):
{article_text}

SUMMARY (check if cues are preserved here):
{summary_text}

EXPECTED OUTPUT (showing how cues are identified and preservation determined):
{expected_json}
"""
    return prompt

def save_domain_prompts(domain_map: Dict[str, List[Dict]], output_dir: str):
    """
    Save pre-formatted prompts for each domain.
    Each file contains a list of formatted example objects.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for domain, examples in domain_map.items():
        # Format each example
        formatted_examples = []
        for idx, example in enumerate(examples, 1):
            # Add separator and example number
            example_header = f"\n{'='*80}\nEXAMPLE {idx}\n{'='*80}\n\n"
            
            try:
                example_text = create_example_prompt_text(example)
                
                formatted_examples.append({
                    'doc_id': example.get('doc_id'),
                    'title': example.get('title', 'Untitled'),
                    'formatted_text': example_header + example_text,
                    'has_article': 'article_text' in example and len(example.get('article_text', '')) > 0,
                    'has_summary': 'summary_text' in example and len(example.get('summary_text', '')) > 0
                })
                
            except Exception as e:
                print(f"⚠️  Error formatting example {idx} for domain '{domain}': {e}")
                print(f"   Title: {example.get('title', 'Unknown')}")
                continue
        
        # Save to JSON file with proper encoding
        filename = f"{domain.lower().replace(' ', '_')}_prompts.json"
        output_file = os.path.join(output_dir, filename)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_examples, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(formatted_examples)} examples for '{domain}' to {filename}")

def save_system_instruction(output_dir: str):
    """Save the system instruction to a separate text file."""
    
    system_instruction = """You are an expert at analyzing news articles and their summaries to identify cultural and contextual cues.

Your task is to:
1. Read the ORIGINAL ARTICLE and identify cultural/contextual cues (quotes, cultural references, named entities, etc.)
2. Read the SUMMARY and determine if each cue is preserved or lost
3. Output a JSON object with the exact structure shown in the examples

CRITICAL INSTRUCTIONS:
- START YOUR RESPONSE WITH { AND END WITH }
- DO NOT add markdown code blocks (no ```json)
- DO NOT add explanations before or after the JSON
- DO NOT add extra fields not shown in the examples
- Follow the exact JSON structure from the examples
- Use the exact same field names: "doc_id", "domain", "title", "tables"
- Each table should have "header" and "rows" arrays
- Each row should be an object with keys matching the headers

IMPORTANT: Your output should match the structure shown in the EXPECTED OUTPUT sections of the examples exactly.
"""
    
    output_file = os.path.join(output_dir, "system_instruction.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(system_instruction)
    
    print(f"✅ Saved system instruction to system_instruction.txt")

def preprocess_all():
    """
    Main preprocessing function.
    Loads labeled data (WITH article/summary text), groups by domain,
    and saves formatted prompts.
    """
    print("="*80)
    print("PREPROCESSING LABELED EXAMPLES WITH ARTICLE/SUMMARY TEXT")
    print("="*80)
    
    # Load labeled data WITH text (created by add_text_to_labels.py)
    print("\n1. Loading labeled data with article/summary text...")
    labeled_file = "data/labeled_examples/labeled_data_with_text.json"
    
    if not os.path.exists(labeled_file):
        print(f"\n❌ ERROR: {labeled_file} not found!")
        print("   Please run: python extraction/add_text_to_labels.py first")
        return
    
    examples = load_labeled_data(labeled_file)
    print(f"   ✅ Loaded {len(examples)} examples")
    
    # Verify article_text and summary_text are present
    examples_with_article = sum(1 for ex in examples if 'article_text' in ex and len(ex.get('article_text', '')) > 0)
    examples_with_summary = sum(1 for ex in examples if 'summary_text' in ex and len(ex.get('summary_text', '')) > 0)
    
    print(f"   📊 Examples with article text: {examples_with_article}/{len(examples)}")
    print(f"   📊 Examples with summary text: {examples_with_summary}/{len(examples)}")
    
    if examples_with_article < len(examples) or examples_with_summary < len(examples):
        print(f"\n⚠️  WARNING: Some examples missing text!")
        print(f"   This may affect the quality of few-shot learning.")
    
    # Group by domain
    print("\n2. Grouping examples by domain...")
    domain_map = group_examples_by_domain(examples)
    
    for domain, domain_examples in domain_map.items():
        print(f"   - {domain}: {len(domain_examples)} examples")
    
    # Save formatted prompts
    print("\n3. Saving formatted prompts (with article/summary text)...")
    output_dir = "data/prompt_templates"
    save_domain_prompts(domain_map, output_dir)
    
    # Save system instruction
    print("\n4. Saving system instruction...")
    save_system_instruction(output_dir)
    
    # Validation: Load one example and verify it's valid JSON
    print("\n5. Validating generated templates...")
    try:
        test_file = os.path.join(output_dir, f"{list(domain_map.keys())[0].lower().replace(' ', '_')}_prompts.json")
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            print(f"   ✅ Successfully loaded and parsed: {os.path.basename(test_file)}")
            print(f"   ✅ Contains {len(test_data)} formatted examples")
            if test_data:
                print(f"   ✅ First example has {len(test_data[0]['formatted_text'])} characters")
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"\nFormatted prompts saved to: {output_dir}/")
    print("Files created:")
    print("  - system_instruction.txt")
    for domain in domain_map.keys():
        print(f"  - {domain.lower().replace(' ', '_')}_prompts.json")
    print("\nNext steps:")
    print("  1. Verify templates: cat data/prompt_templates/cultural_prompts.json | head -100")
    print("  2. Run extraction: python extraction/gpt4_extractor.py")
    print("="*80)

if __name__ == "__main__":
    preprocess_all()