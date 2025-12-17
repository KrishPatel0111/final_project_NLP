"""
Main pipeline: compare articles to summaries and compute cue preservation.

Steps per article-summary pair:
1. Extract cultural cues (spaCy entities) from the article.
2. Extract contextual cues (discourse, temporal, causal, framing, pronouns) from the article.
3. For each cue, check whether it is preserved in the summary.
4. Format the result with `create_extraction_output` and save to JSONL.
"""

import json
from pathlib import Path

from data_loader import load_articles, load_summaries, match_articles_and_summaries
from entity_extractor import EntityExtractor
from contextual_extractor import ContextualExtractor
from output_format import (
    create_extraction_output,
    calculate_preservation_status,
    normalize_subtype,
)


def build_cultural_cues(article_text: str, summary_text: str, entity_result: dict):
    """
    Convert EntityExtractor output into standardized cultural cue entries.
    """
    cultural_cues = []

    # entity_result["entities"] is a list of dicts:
    #   {"text": ..., "type": "PERSON" | "ORG" | ...}
    for ent in entity_result["entities"]:
        raw_type = ent["type"]          # spaCy label
        text_span = ent["text"]

        # Normalize subtype to match manual taxonomy
        subtype = normalize_subtype("Cultural", raw_type)

        # Check if preserved in summary
        preserved = calculate_preservation_status(
            article_text=article_text,
            summary_text=summary_text,
            cue_text=text_span,
        )

        cultural_cues.append({
            "subtype": subtype,
            "text_span": text_span,
            "preserved": preserved,
            "detection_method": "spacy_ner"
        })

    return cultural_cues


def build_contextual_cues(article_text: str, summary_text: str, contextual_result: dict):
    """
    Convert ContextualExtractor output into standardized contextual cue entries.
    """
    contextual_cues = []

    # contextual_result has keys: "discourse", "temporal", "causal", "framing", "pronouns"
    # Each is a list of dicts: {"text": ..., "position": ..., "type": "word_marker"/"phrase_marker"/"framing"}
    for category, cues in contextual_result.items():
        # Map internal key "pronouns" -> raw subtype "pronoun" to match CONTEXTUAL_SUBTYPES
        if category == "pronouns":
            raw_subtype = "pronoun"
        else:
            raw_subtype = category

        subtype = normalize_subtype("Contextual", raw_subtype)

        for cue in cues:
            text_span = cue["text"]

            preserved = calculate_preservation_status(
                article_text=article_text,
                summary_text=summary_text,
                cue_text=text_span,
            )

            detection_method = cue.get("type", "curated_list")

            contextual_cues.append({
                "subtype": subtype,
                "text_span": text_span,
                "preserved": preserved,
                "detection_method": detection_method
            })

    return contextual_cues


def run_main_pipeline(
    articles_csv_path: str,
    summaries_jsonl_path: str,
    output_jsonl_path: str,
    spacy_model_name: str = "en_core_web_sm",
):
    """
    Run full pipeline over all article-summary pairs and write JSONL output.

    Each line in the output file is the standardized extraction output
    for a single article-summary pair.
    """
    articles_csv_path = Path(articles_csv_path)
    summaries_jsonl_path = Path(summaries_jsonl_path)
    output_jsonl_path = Path(output_jsonl_path)

    print("📚 Loading articles and summaries...")
    articles_df = load_articles(str(articles_csv_path))
    summaries_df = load_summaries(str(summaries_jsonl_path))
    merged_df = match_articles_and_summaries(articles_df, summaries_df)

    print(f"\n✅ Ready to process {len(merged_df)} article-summary pairs")

    print("\n🤖 Initializing extractors...")
    entity_extractor = EntityExtractor(model_name=spacy_model_name)
    contextual_extractor = ContextualExtractor()

    # Make sure output directory exists
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    num_pairs = len(merged_df)
    processed = 0

    with output_jsonl_path.open("w", encoding="utf-8") as f_out:
        for idx, row in merged_df.iterrows():
            processed += 1

            doc_id = int(row["doc_id"])
            title = str(row["title"])
            domain = str(row.get("domain", "unknown"))

            article_text = str(row["article_text"])
            summary_text = str(row["summary"])

            summary_model = str(row.get("model", "unknown"))
            summary_style = str(row.get("style", "unknown"))

            print(f"\n[{processed}/{num_pairs}] Processing doc_id={doc_id}, model={summary_model}, style={summary_style}")

            # --- 1) Cultural cues (entities) from article ---
            entity_result = entity_extractor.extract_and_summarize(article_text)
            cultural_cues = build_cultural_cues(
                article_text=article_text,
                summary_text=summary_text,
                entity_result=entity_result,
            )

            # --- 2) Contextual cues from article ---
            contextual_raw = contextual_extractor.extract_all_cues(article_text)
            contextual_cues = build_contextual_cues(
                article_text=article_text,
                summary_text=summary_text,
                contextual_result=contextual_raw,
            )

            # --- 3) Build standardized output for this pair ---
            extraction_output = create_extraction_output(
                doc_id=doc_id,
                domain=domain,
                title=title,
                cultural_cues=cultural_cues,
                contextual_cues=contextual_cues,
                method="automated",
                article_text=article_text,
                summary_text=summary_text,
                summary_model=summary_model,
                summary_style=summary_style,
            )

            # --- 4) Write as one JSON object per line ---
            f_out.write(json.dumps(extraction_output) + "\n")

    print(f"\n🎉 Done! Wrote outputs to: {output_jsonl_path}")


if __name__ == "__main__":
    # 🔧 Update these paths to match your actual data layout.
    #
    # Given what you uploaded, something like:
    #   data/
    #       "Gathered Data - guardian_articles_raw (1).csv"
    #       "all_summaries.jsonl"
    #
    # And we'll write to: outputs/auto_extractions.jsonl

    run_main_pipeline(
        articles_csv_path="../data/guardian_articles_raw.csv",
        summaries_jsonl_path="../data/outputs/all_summaries.jsonl",
        output_jsonl_path="../data/outputs/auto_extractions.jsonl",
        spacy_model_name="en_core_web_lg",   # or "en_core_web_lg" if you have it
    )
