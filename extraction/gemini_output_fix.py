"""
Fix articles with 0 cues by re-processing their raw_extraction with updated flatten_cues()
"""

import json
from pathlib import Path

INPUT_FILE = 'data/article_cues/gemini_cues.jsonl'
OUTPUT_FILE = 'data/article_cues/gemini_cues_fixed.jsonl'
BACKUP_FILE = 'data/article_cues/gemini_cues_backup.jsonl'


def flatten_cues(extraction_result):
    """Flatten Gemini's table format - handles ALL column name formats (case-insensitive)."""
    
    flattened = []
    
    if not extraction_result or 'tables' not in extraction_result:
        return flattened
    
    for table in extraction_result['tables']:
        if 'rows' not in table:
            continue
        
        for row in table['rows']:
            # Create lowercase key mapping for case-insensitive lookup
            row_lower = {k.lower(): v for k, v in row.items()}
            
            # ===== EXTRACT TEXT (try ALL possible names) =====
            text = (
                row_lower.get('text span (from original)') or
                row_lower.get('text span (exact)') or
                row_lower.get('cue_original_text') or  # NEW!
                row_lower.get('cue_phrase') or
                row_lower.get('text') or
                row_lower.get('cue') or
                ''
            )
            
            if not text or text.strip() == '':
                continue
            
            text = text.strip()
            
            # ===== EXTRACT CUE TYPE =====
            cue_type_raw = (
                row_lower.get('cue type') or
                row_lower.get('cue_type') or
                row_lower.get('cue_category') or  # NEW! Some articles use this for type
                row_lower.get('type') or
                ''
            )
            
            # ===== EXTRACT SUBTYPE/CATEGORY =====
            subtype_raw = (
                row_lower.get('cue_subcategory') or  # NEW! Prioritize this
                row_lower.get('subtype') or
                row_lower.get('category') or
                row_lower.get('cue_category') or
                ''
            )
            
            # ===== DETERMINE FINAL CUE TYPE & SUBTYPE =====
            if cue_type_raw and subtype_raw:
                # If cue_type_raw is like "Cultural Cues", extract just "Cultural"
                if 'cultural' in cue_type_raw.lower():
                    cue_type = 'Cultural'
                elif 'contextual' in cue_type_raw.lower():
                    cue_type = 'Contextual'
                else:
                    cue_type = cue_type_raw
                subtype = subtype_raw
            elif subtype_raw:
                subtype = subtype_raw
                category_lower = subtype_raw.lower()
                
                # Map category to cue_type
                if any(word in category_lower for word in [
                    'idiom', 'fixed expression',
                    'named people', 'named person', 'people',
                    'named place', 'place',
                    'named organization', 'organization',
                    'cultural event', 'event', 'holiday', 'tradition',
                    'cultural artifact', 'artifact',
                    'dialect', 'slang'
                ]):
                    cue_type = 'Cultural'
                elif any(word in category_lower for word in [
                    'connector', 'discourse marker',
                    'time', 'sequence',
                    'pronoun', 'reference',
                    'causal', 'logical',
                    'framing',
                    'event ordering'
                ]):
                    cue_type = 'Contextual'
                else:
                    cue_type = 'Cultural'
            else:
                continue
            
            # Clean up subtype names
            subtype_mapping = {
                'Idioms / Fixed Expressions': 'Idiom',
                'Named People': 'Named person',
                'Named Places': 'Place',
                'Named Organizations': 'Organization',
                'Cultural Events / Holidays / Traditions': 'Event',
                'Cultural Artifacts': 'Artifact',
                'Dialect or Slang Terms': 'Slang',
                'Connectors / Discourse Markers': 'Connector',
                'Time or Sequence Markers': 'Time',
                'Pronouns / References': 'Pronouns',
                'Causal or Logical Phrases': 'Causal',
                'Framing Phrases': 'Framing',
                'Event Ordering Cues': 'Event ordering'
            }
            
            for old, new in subtype_mapping.items():
                if old in subtype:
                    subtype = new
                    break
            
            flattened.append({
                'cue_type': cue_type,
                'subtype': subtype,
                'text': text
            })
    
    return flattened


def filter_noisy_cues(cues):
    """Remove generic/noisy cues."""
    
    SKIP_WORDS = {'while', 'since', 'but', 'yet', 'however', 'although', 'and', 'or', 'so', 'as'}
    
    filtered = []
    for cue in cues:
        text = cue['text'].lower().strip()
        subtype = cue.get('subtype', '').lower()
        
        # Skip single generic words
        if text in SKIP_WORDS:
            continue
        
        # Skip all pronouns
        if 'pronoun' in subtype:
            continue
        
        # Skip short connectors
        if 'connector' in subtype and len(text) < 4:
            continue
        
        # Skip "he said" type phrases
        if any(word in text for word in ['said', 'wrote', 'added', 'noted']):
            continue
        
        filtered.append(cue)
    
    return filtered


def main():
    print("=" * 80)
    print("FIXING ARTICLES WITH 0 CUES")
    print("=" * 80)
    
    # Backup original file
    import shutil
    shutil.copy(INPUT_FILE, BACKUP_FILE)
    print(f"✅ Backed up original to: {BACKUP_FILE}")
    
    # Process articles
    fixed_count = 0
    total_count = 0
    zero_cue_count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            article = json.loads(line)
            total_count += 1
            
            # Check if article has 0 cues
            if article['total_cues'] == 0:
                zero_cue_count += 1
                
                # Re-process raw_extraction
                if 'raw_extraction' in article:
                    print(f"\n🔧 Fixing: {article['title'][:60]}...")
                    
                    # Re-flatten cues
                    cues = flatten_cues(article['raw_extraction'])
                    cues = filter_noisy_cues(cues)
                    
                    # Update article
                    article['cues'] = cues
                    article['total_cues'] = len(cues)
                    
                    print(f"   ✅ Extracted {len(cues)} cues (was 0)")
                    fixed_count += 1
                else:
                    print(f"   ⚠️  No raw_extraction found: {article['title'][:60]}")
            
            # Write updated article
            f_out.write(json.dumps(article, ensure_ascii=False) + '\n')
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total articles: {total_count}")
    print(f"Articles with 0 cues: {zero_cue_count}")
    print(f"Successfully fixed: {fixed_count}")
    print(f"\n✅ Fixed file saved to: {OUTPUT_FILE}")
    print(f"📁 Backup saved to: {BACKUP_FILE}")
    print("\nNext step: Replace original file with fixed version:")
    print(f"  mv {OUTPUT_FILE} {INPUT_FILE}")


if __name__ == "__main__":
    main()