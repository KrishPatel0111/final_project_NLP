# extraction/add_text_to_labels.py
import json
import pandas as pd
from pathlib import Path

def add_text_to_labeled_data():
    """
    Merge labeled data with article text from CSV by matching titles.
    Also adds summary text from the summaries JSONL file.
    """
    print("="*80)
    print("ADDING ARTICLE/SUMMARY TEXT TO LABELED DATA")
    print("="*80)
    
    # Load labeled data
    print("\n1. Loading labeled data...")
    labeled_path = "data/labeled_examples/labeled_data.json"
    with open(labeled_path, 'r', encoding='utf-8') as f:
        labeled_data = json.load(f)
    print(f"   ✅ Loaded {len(labeled_data)} labeled examples")
    
    # Load articles from CSV
    print("\n2. Loading articles from CSV...")
    articles_df = pd.read_csv("data/guardian_articles_raw.csv")
    print(f"   ✅ Loaded {len(articles_df)} articles")
    print(f"   Columns: {list(articles_df.columns)}")
    
    # Create title -> article text mapping
    articles_dict = dict(zip(articles_df['title'], articles_df['text']))
    
    # Load summaries from JSONL
    print("\n3. Loading summaries from JSONL...")
    summaries_df = pd.read_json("data/outputs/textrank.jsonl", lines=True)
    print(f"   ✅ Loaded {len(summaries_df)} summaries")
    
    # Create title -> summary mapping (take first summary for each title)
    summaries_dict = {}
    for _, row in summaries_df.iterrows():
        title = row['title']
        if title not in summaries_dict and row['style']=='one_paragraph':  # Keep first summary
            summaries_dict[title] = row['summary']
    
    # Merge article and summary text into labeled data
    print("\n4. Merging article/summary text by title...")
    updated_data = []
    not_found_articles = []
    not_found_summaries = []
    
    for example in labeled_data:
        title = example.get('title')
        
        # Add article text
        if title in articles_dict:
            example['article_text'] = articles_dict[title]
        else:
            example['article_text'] = '[Article not found]'
            not_found_articles.append(title)
        
        # Add summary text
        if title in summaries_dict:
            example['summary_text'] = summaries_dict[title]
        else:
            example['summary_text'] = '[Summary not found]'
            not_found_summaries.append(title)
        
        updated_data.append(example)
    
    # Report any missing matches
    if not_found_articles:
        print(f"\n   ⚠️  Could not find articles for {len(not_found_articles)} titles:")
        for title in not_found_articles[:5]:  # Show first 5
            print(f"      - {title}")
    
    if not_found_summaries:
        print(f"\n   ⚠️  Could not find summaries for {len(not_found_summaries)} titles:")
        for title in not_found_summaries[:5]:  # Show first 5
            print(f"      - {title}")
    
    # Save updated labeled data
    print("\n5. Saving updated labeled data...")
    output_path = "data/labeled_examples/labeled_data_with_text.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Saved to: {output_path}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Total examples: {len(updated_data)}")
    print(f"✅ Articles matched: {len(updated_data) - len(not_found_articles)}")
    print(f"✅ Summaries matched: {len(updated_data) - len(not_found_summaries)}")
    if not_found_articles:
        print(f"⚠️  Articles not found: {len(not_found_articles)}")
    if not_found_summaries:
        print(f"⚠️  Summaries not found: {len(not_found_summaries)}")
    print("="*80)
    
    return updated_data

if __name__ == "__main__":
    add_text_to_labeled_data()