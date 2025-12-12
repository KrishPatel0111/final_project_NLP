"""
Step 3: Contextual Cue Extractor - FINAL VERSION
Uses curated academic marker lists from multiple teaching sources.
"""

import re
from typing import List, Dict, Set
from collections import Counter


class ContextualExtractor:
    """
    Extracts contextual cues using curated academic marker lists.
    
    Sources: Spellzone, EnglishBix, GrammarBank, DCL, Reading Rockets, 
             Language On, Academic Phrasebank, + additional markers
    """
    
    def __init__(self):
        """
        Initialize with curated word lists for each cue type.
        """
        
        # TEMPORAL MARKERS
        # Core (from Spellzone, EnglishBix, GrammarBank, DCL)
        temporal_core = {
            "after", "afterwards", "before", "earlier", "previously",
            "first", "second", "next", "then", "later", "soon", "shortly",
            "straightaway", "suddenly", "finally", "lastly", "eventually",
            "during", "while", "when", "whenever", "once", "until", "now",
            "meanwhile"
        }
        
        # Additional temporal markers
        temporal_additional = {
            "immediately", "over"
        }
        
        # Combine all temporal single words
        self.temporal_markers = temporal_core | temporal_additional
        
        # Temporal phrases (multi-word)
        self.temporal_phrases = [
            # Core
            "by the time", "at first", "in the beginning", "in the end",
            "now that", "in the meantime",
            # Additional
            "earlier on", "later on", "from then on", "back then",
            "these days", "nowadays", "up to now", "so far",
            "from time to time", "every now and then", "all of a sudden",
            "at that moment", "by then", "for a long time", "for a while",
            "over time", "sooner or later"
        ]
        
        # CAUSAL MARKERS
        # Core (from Reading Rockets, Language On, DCL)
        causal_core = {
            "because", "as", "so", "therefore", "thus", "hence",
            "consequently", "accordingly", "produce"
        }
        
        # Additional causal markers
        causal_additional = {
            "causing", "thanks"
        }
        
        # Combine all causal single words
        self.causal_markers = causal_core | causal_additional
        
        # Causal phrases (multi-word)
        self.causal_phrases = [
            # Core
            "since", "due to", "owing to", "because of", "on account of",
            "in explanation", "the reason why", "led to", "bring about",
            "result from", "as a result", "as a consequence", "this is why",
            "in order to", "in order that", "for the purpose of",
            # Additional
            "given that", "seeing that", "now that", "in view of",
            "in light of", "thanks to", "as a result of", "as a consequence of",
            "for this reason", "for that reason", "thereby",
            "which means that", "which implies that", "which leads to",
            "which results in", "leading to", "resulting in",
            "giving rise to", "bringing about", "with the aim of",
            "with the intention of"
        ]
        
        # DISCOURSE MARKERS (keeping these for now - can update if needed)
        self.discourse_markers = {
            "however", "therefore", "moreover", "furthermore",
            "nevertheless", "consequently", "thus", "hence",
            "meanwhile", "additionally", "similarly", "conversely",
            "indeed", "nonetheless", "likewise", "otherwise",
            "besides", "instead", "rather", "though", "although",
            "yet", "still"
        }
        
        # FRAMING PHRASES - Based on Academic Phrasebank
        self.framing_patterns = [
            # Introducing topic/aim (core)
            r"this paper examines", r"this paper argues that",
            r"this study investigates", r"the present study aims to",
            r"the purpose of this paper is to", r"this article focuses on",
            # Outlining structure (core)
            r"this paper is organized as follows",
            r"the discussion is divided into", r"in the first section",
            r"the next section examines", r"the final section discusses",
            # Limiting scope (core)
            r"this study is limited to", r"the discussion focuses on",
            r"for the purposes of this paper",
            r"it is beyond the scope of this paper",
            # Stance/evaluation (core)
            r"I argue that", r"it is suggested that",
            r"the evidence indicates that", r"these results suggest that",
            # Referencing sections (core)
            r"as discussed above", r"as shown in the previous section",
            r"so far, this paper has considered",
            r"the following section turns to",
            # Summarizing/concluding (core)
            r"in conclusion, this paper has shown that",
            r"to sum up, the findings suggest",
            r"overall, the results indicate that",
            r"taken together, these findings imply that",
            # Additional - introducing/situating
            r"this paper sets out to explore",
            r"this study addresses the question of",
            r"the central issue examined here is",
            r"the present article contributes to",
            # Additional - scope/delimitation
            r"the analysis concentrates on",
            r"the present discussion is restricted to",
            r"the data considered in this paper",
            # Additional - stance/cautious claims
            r"I suggest that", r"I propose that",
            r"the findings seem to indicate that",
            r"it appears that", r"it may be argued that",
            # Additional - structuring/transitions
            r"having outlined .*, the paper now turns to",
            r"the next section builds on",
            r"following this, I examine",
            r"the discussion now moves to",
            # Additional - concluding/implications
            r"in summary, the study demonstrates that",
            r"overall, the analysis supports",
            r"these findings have implications for",
            r"these results point to",
            # Journalistic framing (common in news articles)
            r"according to", r"in a statement", r"in an interview",
            r"told reporters", r"said that", r"announced that",
            r"revealed that", r"claimed that", r"argued that",
            r"suggested that", r"officials said"
        ]
        
        # PRONOUNS
        self.pronouns = {
            "he", "she", "they", "it", "this", "that",
            "these", "those", "them", "their", "his", "her",
            "him", "hers", "theirs", "its", "himself", "herself",
            "themselves", "itself"
        }
    
    def find_word_markers(self, text: str, markers: Set[str]) -> List[Dict]:
        """Find single-word markers in text."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        found = []
        for i, word in enumerate(words):
            if word in markers:
                found.append({
                    "text": word,
                    "position": i,
                    "type": "word_marker"
                })
        
        return found
    
    def find_phrase_markers(self, text: str, phrases: List[str]) -> List[Dict]:
        """Find multi-word phrases in text."""
        text_lower = text.lower()
        found = []
        
        for phrase in phrases:
            # Escape special regex characters in the phrase
            pattern = re.compile(r'\b' + re.escape(phrase) + r'\b')
            matches = pattern.finditer(text_lower)
            
            for match in matches:
                found.append({
                    "text": phrase,
                    "position": match.start(),
                    "type": "phrase_marker"
                })
        
        return found
    
    def find_framing_phrases(self, text: str) -> List[Dict]:
        """Find framing phrases using regex patterns."""
        text_lower = text.lower()
        found = []
        
        for pattern in self.framing_patterns:
            try:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    found.append({
                        "text": match.group(),
                        "position": match.start(),
                        "type": "framing"
                    })
            except re.error:
                # Skip patterns that cause regex errors
                continue
        
        return found
    
    def extract_all_cues(self, text: str) -> Dict:
        """Extract all contextual cues from text."""
        # Discourse markers
        discourse = self.find_word_markers(text, self.discourse_markers)
        
        # Temporal: combine words and phrases
        temporal_words = self.find_word_markers(text, self.temporal_markers)
        temporal_phrases = self.find_phrase_markers(text, self.temporal_phrases)
        temporal = temporal_words + temporal_phrases
        
        # Causal: combine words and phrases
        causal_words = self.find_word_markers(text, self.causal_markers)
        causal_phrases = self.find_phrase_markers(text, self.causal_phrases)
        causal = causal_words + causal_phrases
        
        # Framing
        framing = self.find_framing_phrases(text)
        
        # Pronouns
        pronouns = self.find_word_markers(text, self.pronouns)
        
        return {
            "discourse": discourse,
            "temporal": temporal,
            "causal": causal,
            "framing": framing,
            "pronouns": pronouns
        }
    
    def deduplicate_and_count(self, cues: List[Dict]) -> Dict:
        """Count unique cues."""
        cue_counts = Counter(cue["text"] for cue in cues)
        unique_cues = list(set(cue["text"] for cue in cues))
        
        return {
            "unique_cues": sorted(unique_cues),
            "total_count": len(unique_cues),
            "total_occurrences": len(cues),
            "counts": dict(cue_counts)
        }
    
    def extract_and_summarize(self, text: str) -> Dict:
        """Extract all contextual cues and provide summary."""
        # Extract all cues
        all_cues = self.extract_all_cues(text)
        
        # Summarize each type
        summary = {}
        for cue_type, cues in all_cues.items():
            summary[cue_type] = self.deduplicate_and_count(cues)
        
        # Calculate totals
        total_unique = sum(s["total_count"] for s in summary.values())
        total_occurrences = sum(s["total_occurrences"] for s in summary.values())
        
        return {
            "by_type": summary,
            "total_unique_cues": total_unique,
            "total_occurrences": total_occurrences
        }


def test_contextual_extraction():
    """Test contextual cue extraction."""
    print("="*80)
    print("🧪 TESTING: Contextual Cue Extraction (CURATED ACADEMIC LISTS)")
    print("="*80)
    
    sample_text = """
    However, the situation has changed significantly. Previously, officials 
    said that the policy would remain in place. According to recent reports, 
    this is no longer the case. Because of mounting pressure, the government 
    decided to reverse course. Therefore, new measures will be implemented.
    Meanwhile, critics argue that this is too little, too late. Earlier this 
    week, protesters gathered outside the building. Due to safety concerns,
    the event was cancelled. As a result, many people were disappointed.
    They expressed their frustration at that moment. He said the decision 
    was inevitable given that circumstances had changed. This paper examines
    the implications. In conclusion, these findings suggest that reform is needed.
    """
    
    print(f"\n📝 Sample text length: {len(sample_text)} characters")
    
    extractor = ContextualExtractor()
    
    print(f"\n🔍 Extracting contextual cues...")
    result = extractor.extract_and_summarize(sample_text)
    
    print(f"\n✅ Found {result['total_unique_cues']} unique contextual cues")
    print(f"   (with {result['total_occurrences']} total occurrences)")
    
    print(f"\n📊 Breakdown by type:")
    for cue_type, data in result['by_type'].items():
        print(f"\n   {cue_type.upper()}: {data['total_count']} unique, {data['total_occurrences']} total")
        for cue in data['unique_cues'][:20]:  # Show first 20
            count = data['counts'][cue]
            print(f"      - '{cue}' ({count}x)")
        if data['total_count'] > 20:
            print(f"      ... and {data['total_count'] - 20} more")
    
    print("\n" + "="*80)
    print("✅ Contextual extraction working with curated lists!")
    print("="*80)
    
    # Show counts
    print(f"\n📈 List sizes:")
    print(f"   Temporal markers: {len(extractor.temporal_markers)} words + {len(extractor.temporal_phrases)} phrases")
    print(f"   Causal markers: {len(extractor.causal_markers)} words + {len(extractor.causal_phrases)} phrases")
    print(f"   Discourse markers: {len(extractor.discourse_markers)} words")
    print(f"   Framing patterns: {len(extractor.framing_patterns)} patterns")
    print(f"   Pronouns: {len(extractor.pronouns)} pronouns")


if __name__ == "__main__":
    test_contextual_extraction()
    
    print("\n💡 Next: Combine with entity extraction and check preservation!")