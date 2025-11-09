import os
from dotenv import load_dotenv
import httpx

# Load environment variables from .env
load_dotenv("/home/krish/uni/cse5525/final/final_project_NLP/keys.env")

# Access your key
API_KEY = os.getenv("GUARDIAN_API_KEY")

if not API_KEY:
    raise ValueError("Guardian API key not found. Check your .env file!")

# Example request
url = "https://content.guardianapis.com/search"
params = {"q": "artificial intelligence", "api-key": API_KEY, "page-size": 5}

resp = httpx.get(url, params=params, timeout=30)
resp.raise_for_status()

data = resp.json()["response"]["results"]
for article in data:
    print(article["webTitle"], "=>", article["webUrl"])
