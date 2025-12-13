"""
Test entity extraction on FULL doc_id 0 article
Uses the actual data from data_loader.py
"""

from data_loader import load_articles, load_summaries, match_articles_and_summaries
from entity_extractor import EntityExtractor


def test_full_article():
    """
    Test entity extraction on the FULL article (not just a sample).
    """
    print("="*80)
    print("🧪 TESTING: Entity Extraction on FULL Doc 0 Article")
    print("="*80)
    
    # Load actual data
    print("\n📚 Loading data...")
    articles = load_articles("../data/guardian_articles_raw.csv")
    summaries = load_summaries("../data/outputs/textrank.jsonl")
    merged = match_articles_and_summaries(articles, summaries)
    
    # Get doc_id 0 (full article text)
    doc_0 = merged[merged['doc_id'] == 0].iloc[0]
    article_text = doc_0['article_text']
    
    print(f"\n📖 Article length: {len(article_text)} characters")
    print(f"   Title: {doc_0['title'][:60]}...")
    
    # Initialize extractor
    print("\n🤖 Initializing entity extractor...")
    extractor = EntityExtractor(model_name="en_core_web_lg")
    
    # Extract from FULL article
    print("\n📝 Extracting entities from FULL article...")
    result = extractor.extract_and_summarize(article_text)
    
    # Show results
    print(f"\n✅ Found {result['total_count']} unique entities!")
    print(f"\n📊 Breakdown by type:")
    for entity_type, count in sorted(result['by_type_count'].items(), 
                                     key=lambda x: x[1], reverse=True):
        print(f"   {entity_type}: {count}")
    
    # Show ALL entities by type
    print(f"\n🔍 ALL entities found:")
    for entity_type, entities in sorted(result['grouped'].items()):
        print(f"\n   {entity_type} ({len(entities)}):")
        for entity in entities:
            print(f"      - {entity}")
    
    # Check for specific people we expect
    print("\n\n🔎 Checking for specific people mentioned in article:")
    expected_people = [
        "Elon Musk",
        "Tommy Robinson", 
        "Glyn Robbins",
        "Lutfur Rahman",
        "Oswald Mosley"
    ]
    
    all_people = result['grouped'].get('PERSON', [])
    for person in expected_people:
        found = any(person.lower() in p.lower() for p in all_people)
        status = "✅" if found else "❌"
        print(f"   {status} {person}")
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    test_full_article()