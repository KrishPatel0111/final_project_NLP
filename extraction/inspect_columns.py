import csv

path = "../data/guardian_articles_raw.csv"
with open(path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    print(reader.fieldnames)
