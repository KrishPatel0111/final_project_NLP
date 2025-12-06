# extraction/prompt_builder_fast.py
import json
import os
from typing import List, Dict

def load_system_instruction(templates_dir: str = "data/prompt_templates") -> str:
    """Load the system instruction from file."""
    system_file = os.path.join(templates_dir, "system_instruction.txt")
    if not os.path.exists(system_file):
        raise FileNotFoundError(f"System instruction file not found: {system_file}")
    
    with open(system_file, 'r', encoding='utf-8') as f:
        return f.read()


def load_domain_examples(domain: str, templates_dir: str = "data/prompt_templates", n: int = 3) -> List[str]:
    """
    Load pre-formatted example prompts for a specific domain.
    
    Args:
        domain: Domain name (case-insensitive)
        templates_dir: Directory containing prompt templates
        n: Number of examples to load
    
    Returns:
        List of formatted example strings
    """
    # Normalize domain name to match file naming
    domain_filename = f"{domain.lower().replace(' ', '_')}_prompts.json"
    domain_file = os.path.join(templates_dir, domain_filename)
    
    if not os.path.exists(domain_file):
        print(f"⚠️  No examples found for domain '{domain}' at {domain_file}")
        return []
    
    with open(domain_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    # Keep only examples that have has_article == True
    examples_with_article = [ex for ex in examples if ex.get("has_article")]

    if not examples_with_article:
        print(f"⚠️  No examples with has_article=True for domain '{domain}'")
        return []

    # Return up to n formatted example texts from filtered list
    return [ex["formatted_text"] for ex in examples_with_article[:n]]


def build_fast_prompt(
    article_title: str,
    article_text: str,
    summary_text: str,
    domain: str,
    templates_dir: str = "data/prompt_templates",
    n_examples: int = 3
) -> str:
    """
    Build prompt using pre-processed templates (FAST).
    
    Args:
        article_title: Title of article to analyze
        article_text: Full article text
        summary_text: Summary to analyze
        domain: Domain of the article
        templates_dir: Directory with prompt templates
        n_examples: Number of examples to include
    
    Returns:
        Complete prompt string
    """
    # Load system instruction
    system_instruction = load_system_instruction(templates_dir)
    
    # Load pre-formatted examples
    example_texts = load_domain_examples(domain, templates_dir, n_examples)
    
    # Build prompt
    prompt_parts = [
        system_instruction,
        "\n" + "="*80,
        "FEW-SHOT EXAMPLES",
        "="*80
    ]
    
    # Add examples (already formatted!)
    prompt_parts.extend(example_texts)
    
    # Add target article/summary
    prompt_parts.extend([
        "\n" + "="*80,
        "NOW ANALYZE THIS NEW ARTICLE",
        "="*80 + "\n",
        f"ARTICLE TITLE: {article_title}",
        f"DOMAIN: {domain}\n",
        f"ARTICLE:\n{article_text}\n",
        f"SUMMARY:\n{summary_text}\n",
        "OUTPUT (valid JSON only, no markdown):\n"
    ])
    
    return "\n".join(prompt_parts)


def build_fast_chat_messages(
    article_title: str,
    article_text: str,
    summary_text: str,
    domain: str,
    templates_dir: str = "data/prompt_templates",
    n_examples: int = 3
) -> List[Dict[str, str]]:
    """
    Build chat messages using pre-processed templates (FAST).
    
    Returns:
        List of message dictionaries for chat template
    """
    # Load system instruction
    system_instruction = load_system_instruction(templates_dir)
    
    # Load pre-formatted examples
    example_texts = load_domain_examples(domain, templates_dir, n_examples)
    
    # Build user message
    user_parts = example_texts + [
        "\n" + "="*80,
        "NOW ANALYZE THIS NEW ARTICLE",
        "="*80 + "\n",
        f"ARTICLE TITLE: {article_title}",
        f"DOMAIN: {domain}\n",
        f"ARTICLE:\n{article_text}\n",
        f"SUMMARY:\n{summary_text}\n",
        "OUTPUT (valid JSON only, no markdown):"
    ]
    
    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": "\n".join(user_parts)}
    ]