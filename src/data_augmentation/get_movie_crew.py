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
INPUT_FILE = 'data/interim/standarized_movies_keywords.csv'
OUTPUT_FILE = 'data/processed/movies_final.csv'
BASE_URL = "https://api.themoviedb.org/3/movie/{}/credits"

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {TMDB_API_KEY}"
}

def get_movie_credits(tmdb_id, max_retries=3):
    """
    Fetches credits for a specific movie ID.
    Returns a tuple of (directors_json_string, writers_json_string)
    """
    url = BASE_URL.format(tmdb_id)
    params = {
        "language": "en-US"
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                crew = data.get('crew', [])
                
                # Filter Directors
                # We strictly look for job == 'Director'
                directors = [
                    {
                        "id": member['id'],
                        "name": member['name'],
                        "job": member['job'],
                        "profile_path": member.get('profile_path')
                    }
                    for member in crew if member.get('job') == 'Director'
                ]

                # Filter Writers
                # We look for the department 'Writing' to catch Screenplay, Novel, Story, etc.
                writers = [
                    {
                        "id": member['id'],
                        "name": member['name'],
                        "job": member['job'], # Important to distinguish Novel vs Screenplay
                        "profile_path": member.get('profile_path')
                    }
                    for member in crew if member.get('department') == 'Writing'
                ]

                return json.dumps(directors), json.dumps(writers)

            elif response.status_code == 404:
                print(f"Credits endpoint for ID {tmdb_id} not found (404).")
                return "[]", "[]"
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 1))
                time.sleep(retry_after + 1)
                continue
            else:
                print(f"Error {response.status_code} for ID {tmdb_id}")
                return "[]", "[]"
                
        except requests.exceptions.RequestException as e:
            print(f"Network error for ID {tmdb_id}: {e}")
            time.sleep(1)
    
    return "[]", "[]"

def process_credits():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Input file not found at {INPUT_FILE}. Please run the keywords enrichment step first.")
        return

    print(f"Processing credits for {len(df)} movies...")

    directors_data = []
    writers_data = []
    errors = []

    # Iterate through the dataframe
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        tmdb_id = row['tmdb_id']
        
        # Ensure valid ID before requesting
        if pd.isna(tmdb_id):
            directors_data.append("[]")
            writers_data.append("[]")
            continue

        directors_str, writers_str = get_movie_credits(int(tmdb_id))
        
        directors_data.append(directors_str)
        writers_data.append(writers_str)

        # Rate limiting
        time.sleep(0.25)

    # Add the new columns to the dataframe
    df['directors'] = directors_data
    df['writers'] = writers_data

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # Summary
    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Movies Processed: {len(df)}")
    
    # Basic check
    with_directors = df[df['directors'] != "[]"]
    print(f"Movies with Directors found: {len(with_directors)}")
    
    print(f"Final dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_credits()