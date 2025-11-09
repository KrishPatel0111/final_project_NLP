# summarization/config.py
STYLES = ("bullets", "one_paragraph", "three_sentence")

def target_word_bounds(src_words: int) -> tuple[int, int]:
    if src_words < 400:   return (90, 140)
    if src_words < 1000:  return (120, 200)
    return (150, 240)

def approx_tokens_from_words(n_words: int) -> int:
    return int(n_words * 1.3)  # rough heuristic
