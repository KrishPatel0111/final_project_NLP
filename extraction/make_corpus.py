import csv

input_csv = "../data/guardian_articles_raw.csv"
output_txt = "guardian_corpus.txt"

ARTICLE_COLUMN = "text"   # <-- Correct column name from your CSV

with open(input_csv, newline='', encoding='utf-8') as f_in, \
     open(output_txt, 'w', encoding='utf-8') as f_out:
    
    reader = csv.DictReader(f_in)
    
    for row in reader:
        text = row.get(ARTICLE_COLUMN, "")
        if not text:
            continue
        
        # Clean newlines inside the article
        text = text.replace("\n", " ").strip()
        
        if text:
            f_out.write(text + "\n")
