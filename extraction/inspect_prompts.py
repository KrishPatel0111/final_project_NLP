# extraction/inspect_prompts.py
import os
import json
from prompt_builder import load_system_instruction, load_domain_examples

def generate_prompt_for_inspection(domain, n_examples=2):
    """
    Generate a prompt for a given domain WITHOUT the actual article/summary to analyze.
    This shows you exactly what the few-shot examples look like.
    """
    
    templates_dir = "data/prompt_templates"
    
    # Load system instruction
    system_instruction = load_system_instruction(templates_dir)
    
    # Load domain examples
    examples = load_domain_examples(domain, templates_dir,n_examples)
    
    # Build the prompt (without target article/summary)
    prompt_parts = []
    
    # Add system instruction
    prompt_parts.append("="*80)
    prompt_parts.append("SYSTEM INSTRUCTION")
    prompt_parts.append("="*80)
    prompt_parts.append(system_instruction)
    prompt_parts.append("\n")
    
    # Add examples
    prompt_parts.append("="*80)
    prompt_parts.append(f"FEW-SHOT EXAMPLES FOR DOMAIN: {domain.upper()}")
    prompt_parts.append("="*80)
    prompt_parts.append("\n".join(examples))
    prompt_parts.append("\n")
    
    # Add placeholder for target article/summary
    prompt_parts.append("="*80)
    prompt_parts.append("TARGET ARTICLE/SUMMARY TO ANALYZE (NOT SHOWN IN THIS PREVIEW)")
    prompt_parts.append("="*80)
    prompt_parts.append("""
ARTICLE TITLE: [Target article title will be inserted here]
DOMAIN: [Target domain will be inserted here]

ORIGINAL ARTICLE (identify cues from here):
[Full target article text will be inserted here]

SUMMARY (check if cues are preserved here):
[Target summary text will be inserted here]

NOW OUTPUT YOUR ANALYSIS AS JSON:
""")
    
    return "\n".join(prompt_parts)

def save_domain_prompts():
    """
    Generate and save prompt previews for all domains.
    """
    
    print("="*80)
    print("GENERATING DOMAIN PROMPT PREVIEWS")
    print("="*80)
    
    # Define domains (based on your labeled data)
    domains = ["Educational", "Politics", "Cultural", "Sport", "Technology"]
    
    # Create output directory
    output_dir = "data/prompt_previews"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}/")
    print()
    
    # Generate for different numbers of examples
    example_counts = [1, 2, 3]
    
    total_files = 0
    
    for domain in domains:
        print(f"📋 Domain: {domain}")
        
        for n_examples in example_counts:
            try:
                # Generate prompt
                prompt = generate_prompt_for_inspection(domain, n_examples)
                
                # Create filename
                filename = f"{domain.lower()}_ex_no_{n_examples}.txt"
                filepath = os.path.join(output_dir, filename)
                
                # Save to file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(prompt)
                
                # Get stats
                char_count = len(prompt)
                token_estimate = char_count // 4  # Rough estimate
                
                print(f"   ✅ {filename}")
                print(f"      - {n_examples} example(s)")
                print(f"      - {char_count:,} characters")
                print(f"      - ~{token_estimate:,} tokens (estimated)")
                
                total_files += 1
                
            except Exception as e:
                print(f"   ❌ Failed for {domain} with {n_examples} examples: {e}")
        
        print()
    
    print("="*80)
    print(f"✅ COMPLETE! Generated {total_files} prompt preview files")
    print("="*80)
    print(f"\n📁 Files saved to: {output_dir}/")
    print("\nYou can now review:")
    print("  - System instructions")
    print("  - Few-shot examples for each domain")
    print("  - Prompt structure and format")
    print("\nTo view a prompt:")
    print(f"  cat {output_dir}/politics_ex_no_2.txt | less")
    print("="*80)

def generate_single_domain_prompt(domain, n_examples=2):
    """
    Generate and display a single domain prompt (for quick inspection).
    """
    
    print("="*80)
    print(f"PROMPT PREVIEW: {domain.upper()} with {n_examples} example(s)")
    print("="*80)
    print()
    
    prompt = generate_prompt_for_inspection(domain, n_examples)
    
    print(prompt)
    
    print()
    print("="*80)
    print("END OF PROMPT PREVIEW")
    print("="*80)
    print(f"\nCharacters: {len(prompt):,}")
    print(f"Estimated tokens: ~{len(prompt)//4:,}")

def main():
    """
    Main function with options.
    """
    
    import sys
    
    if len(sys.argv) > 1:
        # Single domain mode
        domain = sys.argv[1]
        n_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        
        # Capitalize first letter
        domain = domain.capitalize()
        
        print(f"\n🔍 Generating preview for: {domain} (with {n_examples} examples)\n")
        generate_single_domain_prompt(domain, n_examples)
        
    else:
        # Generate all
        save_domain_prompts()
        
        print("\n" + "="*80)
        print("USAGE FOR SINGLE DOMAIN:")
        print("="*80)
        print("python extraction/inspect_prompts.py <domain> [n_examples]")
        print("\nExamples:")
        print("  python extraction/inspect_prompts.py politics")
        print("  python extraction/inspect_prompts.py cultural 3")
        print("  python extraction/inspect_prompts.py educational 1")
        print("="*80)

if __name__ == "__main__":
    main()