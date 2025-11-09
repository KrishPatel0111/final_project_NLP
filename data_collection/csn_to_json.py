import pandas as pd

def csn_to_json(input_csv_path, output_json_path):
    df = pd.read_csv(input_csv_path)
    df.to_json(output_json_path)

if __name__ == "__main__":
    csn_to_json("guardian_articles_raw.csv", "guardian_articles.json")
    