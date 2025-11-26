import pandas as pd
import requests
import os
import time
import json
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
TMDB_API_KEY = os.getenv('TMDB_API')
INPUT_FILE = 'data/interim/standarized_movies_details.csv'
OUTPUT_FILE = 'data/interim/standarized_movies_keywords.csv'
BASE_URL = "https://api.themoviedb.org/3/movie/{}/keywords"

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {TMDB_API_KEY}"
}

def get_movie_keywords(tmdb_id, max_retries=3):
    """
    Fetches keywords for a specific movie ID.
    Returns a list of dictionaries [{'id': 1, 'name': 'keyword'}, ...]
    """
    url = BASE_URL.format(tmdb_id)

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('keywords', [])
            elif response.status_code == 404:
                print(f"Keywords for ID {tmdb_id} not found (404).")
                return []
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 1))
                time.sleep(retry_after + 1)
                continue
            else:
                print(f"Error {response.status_code} for ID {tmdb_id}")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"Network error for ID {tmdb_id}: {e}")
            time.sleep(1)
    
    return []

def process_keywords():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Input file not found at {INPUT_FILE}. Please run the previous step first.")
        return

    print(f"Processing {len(df)} movies...")

    # We'll store the results in a new column
    keywords_data = []
    errors = []

    # Iterate through the dataframe
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        tmdb_id = row['tmdb_id']
        
        # Ensure valid ID before requesting
        if pd.isna(tmdb_id):
            keywords_data.append("[]")
            continue

        keywords_list = get_movie_keywords(int(tmdb_id))
        
        if keywords_list is not None:
            # We dump to JSON string to preserve the structure (ID + Name)
            # This is ideal for Graph DB ingestion later.
            keywords_str = json.dumps(keywords_list)
            keywords_data.append(keywords_str)
        else:
            # In case of total failure, append empty list string
            keywords_data.append("[]")
            errors.append(tmdb_id)

        # Rate limiting
        time.sleep(0.25)

    # Add the new column to the dataframe
    df['keywords'] = keywords_data

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # Summary
    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Movies Processed: {len(df)}")
    print(f"Errors: {len(errors)}")
    if len(errors) > 0:
        print(f"IDs with errors: {errors[:10]} ...")
    print(f"File saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_keywords()