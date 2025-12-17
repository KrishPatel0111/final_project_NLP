"""
Step 2: Entity Extractor
Uses spaCy to extract cultural cues (named entities) from text.
"""

import spacy
from typing import List, Dict


class EntityExtractor:
    """
    Extracts named entities (cultural cues) using spaCy.
    
    Entity types we extract:
    - PERSON: People (Glyn Robbins, Tommy Robinson)
    - ORG: Organizations (United East End, BNP)
    - GPE: Geopolitical entities (Tower Hamlets, London)
    - LOC: Locations (East End)
    - NORP: Nationalities/religious/political groups (British Jews, Muslims)
    - EVENT: Named events (Battle of Cable Street)
    - WORK_OF_ART: Creative works
    """
    
    def __init__(self, model_name="en_core_web_lg"):
        """
        Initialize spaCy model.
        
        Args:
            model_name: Which spaCy model to use
                - en_core_web_sm: Small, fast (we'll use this for now)
                - en_core_web_lg: Large, more accurate (can upgrade later)
        """
        print(f"🤖 Loading spaCy model: {model_name}...")
        self.nlp = spacy.load(model_name)
        print("   ✓ spaCy model loaded!")
        
        # Entity types we care about for cultural cues
        self.cultural_entity_types = {
            "PERSON",      # People
            "ORG",         # Organizations
            "GPE",         # Countries, cities, states
            "LOC",         # Non-GPE locations
            "NORP",        # Nationalities, religious/political groups
            "EVENT",       # Named events
            "WORK_OF_ART"  # Titles of books, songs, etc.
        }
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract all cultural entities from text.
        
        Args:
            text: Input text (article or summary)
        
        Returns:
            List of dicts with entity info:
            [
              {"text": "Glyn Robbins", "type": "PERSON"},
              {"text": "Tower Hamlets", "type": "GPE"},
              ...
            ]
        """
        # Run spaCy NLP pipeline on the text
        doc = self.nlp(text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            # Only keep entity types we care about
            if ent.label_ in self.cultural_entity_types:
                entities.append({
                    "text": ent.text,           # The actual text
                    "type": ent.label_,         # Entity type (PERSON, ORG, etc.)
                    "start": ent.start_char,    # Where it starts in text
                    "end": ent.end_char         # Where it ends in text
                })
        
        return entities
    
    def deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Remove duplicate entities (same text + type).
        
        Why: spaCy might find "Tower Hamlets" multiple times.
        We only want to count it once.
        
        Args:
            entities: List of entity dicts
        
        Returns:
            Deduplicated list
        """
        seen = set()
        unique = []
        
        for entity in entities:
            # Create unique key: text + type
            key = (entity["text"].lower(), entity["type"])
            
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        
        return unique
    
    def group_by_type(self, entities: List[Dict]) -> Dict[str, List[str]]:
        """
        Group entities by their type.
        
        Args:
            entities: List of entity dicts
        
        Returns:
            Dict mapping type to list of entity texts:
            {
              "PERSON": ["Glyn Robbins", "Tommy Robinson"],
              "GPE": ["Tower Hamlets", "London"],
              ...
            }
        """
        grouped = {}
        
        for entity in entities:
            entity_type = entity["type"]
            entity_text = entity["text"]
            
            if entity_type not in grouped:
                grouped[entity_type] = []
            
            grouped[entity_type].append(entity_text)
        
        return grouped
    
    def extract_and_summarize(self, text: str) -> Dict:
        """
        Extract entities and return nice summary.
        
        Args:
            text: Input text
        
        Returns:
            Dict with entities and summary stats
        """
        # Extract all entities
        entities = self.extract_entities(text)
        
        # Remove duplicates
        unique_entities = self.deduplicate_entities(entities)
        
        # Group by type
        grouped = self.group_by_type(unique_entities)
        
        return {
            "entities": unique_entities,       # Full list with all info
            "grouped": grouped,                # Grouped by type
            "total_count": len(unique_entities),
            "by_type_count": {k: len(v) for k, v in grouped.items()}
        }


def test_on_doc_0():
    """
    Test the entity extractor on doc_id 0 article.
    """
    print("\n" + "="*80)
    print("🧪 TESTING: Entity Extraction on Doc 0")
    print("="*80)
    
    # Sample text from doc 0 (first part of article)
    sample_text = """
    "The East End of London is the far right's prime target – the essence of 
    everything they don't like. They feel if they can march through our borough 
    with impunity, they can go anywhere. For them, it's like Wembley (stadium), 
    it's the ultimate goal," said Glyn Robbins, co-founder of United East End, 
    an anti-far right coalition of community organisations. In the East End, the 
    historically working-class neighbourhoods in the shadow of the City of London, 
    there's a feeling that history is repeating itself. It was 89 years ago this 
    month that local people, many of them British Jews, drove out Oswald Mosley's 
    Blackshirt militia from Whitechapel in the East End, in what has become known 
    as the Battle of Cable Street. In the decades since, the National Front, the 
    BNP and the English Defence League have all tried to carve out a foothold in 
    the East End. This week, rather than attempting to escort protesters, as they 
    had in Mosley's day, London's police stopped the hard right UK Independence 
    Party (Ukip) from staging a "crusade on Whitechapel" in the borough of Tower 
    Hamlets, where 40% of the population is Muslim.
    """
    
    # Initialize extractor
    extractor = EntityExtractor()
    
    # Extract entities
    print("\n📝 Extracting entities from sample text...")
    result = extractor.extract_and_summarize(sample_text)
    
    # Show results
    print(f"\n✅ Found {result['total_count']} unique entities!")
    print(f"\n📊 Breakdown by type:")
    for entity_type, count in result['by_type_count'].items():
        print(f"   {entity_type}: {count}")
    
    # Show detailed entities
    print(f"\n🔍 Detailed entities:")
    for entity_type, entities in result['grouped'].items():
        print(f"\n   {entity_type}:")
        for entity in entities[:5]:  # Show first 5 of each type
            print(f"      - {entity}")
        if len(entities) > 5:
            print(f"      ... and {len(entities) - 5} more")
    
    print("\n" + "="*80)
    print("✅ Entity extraction working!")
    print("="*80)
    
    return result


if __name__ == "__main__":
    # Run the test
    test_on_doc_0()
    
    print("\n💡 Next step: Check if these entities appear in summaries!")