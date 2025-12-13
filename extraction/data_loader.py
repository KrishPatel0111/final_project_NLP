"""
Step 1: Data Loader
Loads articles (CSV) and summaries (JSONL) and matches them by title.
"""

import pandas as pd
import json


def load_articles(csv_path):
    """
    Load articles from CSV file.
    
    Args:
        csv_path: Path to guardian_articles_raw.csv
    
    Returns:
        DataFrame with articles
    """
    print("📖 Loading articles from CSV...")
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Rename 'text' column to 'article_text' for clarity
    df = df.rename(columns={'text': 'article_text'})
    
    print(f"   ✓ Loaded {len(df)} articles")
    print(f"   ✓ Columns: {list(df.columns)}")
    
    return df


def load_summaries(jsonl_path):
    """
    Load summaries from JSONL file.
    
    Args:
        jsonl_path: Path to all_summaries.jsonl
    
    Returns:
        DataFrame with summaries
    """
    print("\n📝 Loading summaries from JSONL...")
    
    # Read JSONL - each line is a separate JSON object
    summaries = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse each line as JSON
            summary = json.loads(line)
            summaries.append(summary)
    
    # Convert to DataFrame for easy manipulation
    df = pd.DataFrame(summaries)
    
    print(f"   ✓ Loaded {len(df)} summaries")
    print(f"   ✓ Columns: {list(df.columns)}")
    
    # Show breakdown by model and style
    if 'model' in df.columns and 'style' in df.columns:
        print(f"\n   📊 Breakdown:")
        for model in df['model'].unique():
            count = len(df[df['model'] == model])
            print(f"      {model}: {count} summaries")
    
    return df


def match_articles_and_summaries(articles_df, summaries_df):
    """
    Match articles with their summaries by title.
    
    Args:
        articles_df: DataFrame with articles
        summaries_df: DataFrame with summaries
    
    Returns:
        Merged DataFrame with both article text and summaries
    """
    print("\n🔗 Matching articles with summaries by title...")
    
    # Merge on 'title' column
    # This combines rows where titles match
    merged = summaries_df.merge(
        articles_df[['title', 'article_text', 'domain']],  # Only keep columns we need
        on='title',                                         # Match by title
        how='left',                                         # Keep all summaries
        suffixes=('_summary', '_article')                   # If duplicate columns, add suffix
    )
    
    # Check for any summaries that didn't match
    unmatched = merged[merged['article_text'].isna()]
    if len(unmatched) > 0:
        print(f"\n   ⚠️  Warning: {len(unmatched)} summaries didn't match any article!")
        print(f"   (This might be okay if you have more summaries than articles)")
    
    # Remove unmatched summaries
    merged = merged[merged['article_text'].notna()]
    
    print(f"   ✓ Successfully matched {len(merged)} article-summary pairs")
    
    return merged


def test_on_doc_0(merged_df):
    """
    Show what doc_id 0 looks like to verify everything worked.
    
    Args:
        merged_df: Merged DataFrame
    """
    print("\n" + "="*80)
    print("🔍 TESTING: Looking at doc_id 0")
    print("="*80)
    
    # Get all summaries for doc_id 0
    doc_0 = merged_df[merged_df['doc_id'] == 0]
    
    if len(doc_0) == 0:
        print("❌ No data found for doc_id 0!")
        return
    
    # Show basic info
    print(f"\n📄 Article Title:")
    print(f"   {doc_0.iloc[0]['title']}")
    
    print(f"\n📊 Found {len(doc_0)} summaries for this article:")
    for idx, row in doc_0.iterrows():
        print(f"   - {row['model']} ({row['style']})")
    
    # Show article text preview
    article_text = doc_0.iloc[0]['article_text']
    print(f"\n📖 Article Text Preview (first 500 chars):")
    print(f"   {article_text[:500]}...")
    print(f"\n   Total article length: {len(article_text)} characters")
    
    # Show one summary as example
    print(f"\n📝 Example Summary (TextRank, bullets):")
    bullets_summary = doc_0[doc_0['style'] == 'bullets'].iloc[0]['summary']
    print(f"   {bullets_summary[:500]}...")
    print(f"\n   Summary length: {len(bullets_summary)} characters")
    
    print("\n" + "="*80)
    print("✅ Data loading looks good!")
    print("="*80)


def main():
    """
    Main function - runs all the loading and testing.
    """
    # File paths - UPDATE THESE TO YOUR ACTUAL PATHS
    csv_path = "../data/guardian_articles_raw.csv"
    jsonl_path = "../data/outputs/textrank.jsonl"
    
    # Step 1: Load articles
    articles = load_articles(csv_path)
    
    # Step 2: Load summaries
    summaries = load_summaries(jsonl_path)
    
    # Step 3: Match them together
    merged = match_articles_and_summaries(articles, summaries)
    
    # Step 4: Test on doc_id 0
    test_on_doc_0(merged)
    
    # Return the merged data for next steps
    return merged


if __name__ == "__main__":
    # Run the loader
    merged_data = main()
    
    print("\n💾 Merged data is ready!")
    print(f"   Shape: {merged_data.shape} (rows × columns)")
    print(f"   Ready for extraction pipeline!")