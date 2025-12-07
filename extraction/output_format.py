"""
Output format handler for cultural and contextual cue extraction.
Ensures all automated extractions match the manual annotation format.
"""

from datetime import datetime
from typing import List, Dict, Optional


def create_cue_entry(
    subtype: str,
    text_span: str,
    preserved: str,
    detection_method: Optional[str] = None,
    confidence: Optional[float] = None
) -> Dict:
    """
    Create a single cue entry.
    
    Args:
        subtype: e.g., "Named people", "Discourse marker"
        text_span: The actual text from the article
        preserved: "Yes", "No", or "Partially"
        detection_method: How it was detected (e.g., "spacy_ner", "word_list")
        confidence: Optional confidence score (0-1)
    
    Returns:
        Dict formatted as a table row
    """
    entry = {
        "Subtype": subtype,
        "Text Span (from original)": text_span,
        "Preserved in Summary?": preserved
    }
    
    # Add optional metadata
    if detection_method:
        entry["detection_method"] = detection_method
    if confidence is not None:
        entry["confidence"] = confidence
    
    return entry


def create_extraction_output(
    doc_id: int,
    domain: str,
    title: str,
    cultural_cues: List[Dict],
    contextual_cues: List[Dict],
    method: str = "automated",
    article_text: Optional[str] = None,
    summary_text: Optional[str] = None,
    summary_model: Optional[str] = None,
    summary_style: Optional[str] = None
) -> Dict:
    """
    Create standardized extraction output matching manual annotation format.
    
    Args:
        doc_id: Document ID
        domain: Article domain (e.g., "politics", "technology")
        title: Article title
        cultural_cues: List of cultural cue dicts with keys: subtype, text_span, preserved
        contextual_cues: List of contextual cue dicts with same structure
        method: "automated", "manual", or "hybrid"
        article_text: Optional full article text
        summary_text: Optional summary text
        summary_model: Which model generated the summary
        summary_style: bullets, paragraph, or sentences
    
    Returns:
        Dict formatted exactly like manual annotations
    """
    output = {
        "doc_id": doc_id,
        "domain": domain,
        "title": title,
        "extraction_method": method,
        "extraction_timestamp": datetime.now().isoformat(),
        "tables": []
    }
    
    # Add optional metadata
    if summary_model:
        output["summary_model"] = summary_model
    if summary_style:
        output["summary_style"] = summary_style
    
    # Cultural cues table
    cultural_rows = []
    for cue in cultural_cues:
        row = {
            "Cue Type": "Cultural",
            "Subtype": cue["subtype"],
            "Text Span (from original)": cue["text_span"],
            "Preserved in Summary?": cue["preserved"]
        }
        
        # Add detection metadata if present
        if "detection_method" in cue:
            row["detection_method"] = cue["detection_method"]
        if "confidence" in cue:
            row["confidence"] = cue["confidence"]
        
        cultural_rows.append(row)
    
    output["tables"].append({
        "header": ["Cue Type", "Subtype", "Text Span (from original)", "Preserved in Summary?"],
        "rows": cultural_rows
    })
    
    # Contextual cues table
    contextual_rows = []
    for cue in contextual_cues:
        row = {
            "Cue Type": "Contextual",
            "Subtype": cue["subtype"],
            "Text Span": cue["text_span"],
            "Preserved in Summary?": cue["preserved"]
        }
        
        # Add detection metadata if present
        if "detection_method" in cue:
            row["detection_method"] = cue["detection_method"]
        if "confidence" in cue:
            row["confidence"] = cue["confidence"]
        
        contextual_rows.append(row)
    
    output["tables"].append({
        "header": ["Cue Type", "Subtype", "Text Span", "Preserved in Summary?"],
        "rows": contextual_rows
    })
    
    return output


def calculate_preservation_status(article_text: str, summary_text: str, cue_text: str) -> str:
    """
    Determine if a cue is preserved in the summary.
    
    Args:
        article_text: Full article text
        summary_text: Full summary text
        cue_text: The specific cue to check
    
    Returns:
        "Yes", "No", or "Partially"
    """
    cue_lower = cue_text.lower().strip()
    summary_lower = summary_text.lower()
    
    # Exact match
    if cue_lower in summary_lower:
        return "Yes"
    
    # Check for partial matches (for multi-word expressions)
    if " " in cue_lower:
        words = cue_lower.split()
        matches = sum(1 for word in words if word in summary_lower)
        if matches == len(words):
            return "Yes"
        elif matches > 0:
            return "Partially"
    
    return "No"


# Subtype mappings for consistent naming
CULTURAL_SUBTYPES = {
    "PERSON": "Named people",
    "ORG": "Named organizations", 
    "GPE": "Named places",
    "LOC": "Named places",
    "NORP": "Cultural groups/identities",
    "EVENT": "Cultural events/artifacts",
    "WORK_OF_ART": "Cultural events/artifacts",
    "idiom": "Idioms/fixed expressions",
    "slang": "Dialect/slang terms"
}

CONTEXTUAL_SUBTYPES = {
    "discourse": "Connector/discourse marker",
    "temporal": "Time/sequence marker",
    "pronoun": "Pronouns/references",
    "causal": "Causal/logical phrases",
    "framing": "Framing phrase"
}


def normalize_subtype(cue_type: str, raw_subtype: str) -> str:
    """
    Normalize subtypes to match manual annotation taxonomy.
    
    Args:
        cue_type: "Cultural" or "Contextual"
        raw_subtype: Raw subtype from detection method
    
    Returns:
        Normalized subtype string
    """
    if cue_type == "Cultural":
        return CULTURAL_SUBTYPES.get(raw_subtype, raw_subtype)
    elif cue_type == "Contextual":
        return CONTEXTUAL_SUBTYPES.get(raw_subtype, raw_subtype)
    return raw_subtype