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
INPUT_FILE = 'data/interim/standarized_movies.csv'
LINKS_FILE = 'data/interim/links.csv'
OUTPUT_FILE = 'data/interim/standarized_movies_details.csv'
BASE_URL = "https://api.themoviedb.org/3/movie/{}"

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {TMDB_API_KEY}"
}

def get_movie_details(tmdb_id, max_retries=3):
    """
    Fetches detailed info for a specific movie ID.
    """
    url = BASE_URL.format(tmdb_id)
    params = {
        "language": "en-US"
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None # Movie ID might be dead or invalid
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 1))
                time.sleep(retry_after + 1)
                continue
            else:
                print(f"Error {response.status_code} for ID {tmdb_id}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Network error for ID {tmdb_id}: {e}")
            time.sleep(1)
    
    return None

def load_links_mapping():
    """
    Loads the links.csv file and returns a dictionary mapping
    MovieLens ID (movieId) -> TMDB ID (tmdbId).
    """
    if not os.path.exists(LINKS_FILE):
        print(f"Warning: {LINKS_FILE} not found. Fallback logic will be disabled.")
        return {}

    try:
        # links.csv columns: movieId, imdbId, tmdbId
        links_df = pd.read_csv(LINKS_FILE, dtype={'movieId': int, 'tmdbId': 'Int64'})
        # Drop rows where tmdbId is NaN
        links_df = links_df.dropna(subset=['tmdbId'])
        
        # Create dictionary
        return dict(zip(links_df['movieId'], links_df['tmdbId']))
    except Exception as e:
        print(f"Error loading links file: {e}")
        return {}

def process_details():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Input file not found at {INPUT_FILE}")
        return

    print(f"Loading fallback links from {LINKS_FILE}...")
    ml_to_tmdb_map = load_links_mapping()

    print(f"Processing {len(df)} movies...")

    detailed_records = []
    errors = []
    recovered_count = 0

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        ml_id = row['ml_movie_id']
        tmdb_id = row['tmdb_id']

        # 1. Resolve TMDB ID
        # If the previous step failed to find an ID (NaN), try the links map
        is_recovered = False
        if pd.isna(tmdb_id):
            # Lookup in links file
            found_id = ml_to_tmdb_map.get(ml_id)
            if found_id:
                tmdb_id = int(found_id)
                is_recovered = True
                recovered_count += 1
            else:
                # Still no ID found, cannot fetch details
                errors.append(ml_id)
                continue
        else:
            tmdb_id = int(tmdb_id)
        
        # 2. Fetch Details
        details = get_movie_details(tmdb_id)
        
        if details:
            # Parse Genres
            genre_names = [g['name'] for g in details.get('genres', [])]
            genre_str = "|".join(genre_names)

            # Extract release year from details if possible, fallback to ML year
            api_date = details.get('release_date', '')
            if api_date and len(api_date) >= 4:
                final_year = api_date[:4]
            else:
                final_year = row['release_year']

            record = {
                # ID fields
                'ml_movie_id': ml_id,
                'tmdb_id': tmdb_id,
                'imdb_id': details.get('imdb_id'),
                
                # Basic Info (Prioritize API data for consistency)
                'title': details.get('title'), 
                'release_year': final_year,
                
                # Metrics
                'popularity': details.get('popularity'),
                'vote_average': details.get('vote_average'),
                'vote_count': details.get('vote_count'),
                
                # Enriched Details
                'budget': details.get('budget'),
                'revenue': details.get('revenue'),
                'runtime': details.get('runtime'),
                
                # Content & Assets
                'overview': details.get('overview'),
                'genres': genre_str,
                'poster_path': details.get('poster_path'),
                'backdrop_path': details.get('backdrop_path')
            }
            detailed_records.append(record)
        else:
            # ID existed (either from search or links) but API call failed (404 or network)
            print(f"Failed to fetch details for TMDB ID {tmdb_id} (ML ID {ml_id})")
            errors.append(ml_id)

        # Rate limiting
        time.sleep(0.25)

    # Create DataFrame
    details_df = pd.DataFrame(detailed_records)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    details_df.to_csv(OUTPUT_FILE, index=False)

    # Summary
    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Total Processed: {len(df)}")
    print(f"Successfully Enriched: {len(details_df)}")
    print(f"Recovered via Links File: {recovered_count}")
    print(f"Still Missing/Failed: {len(errors)}")
    if len(errors) > 0:
        print(f"First 10 Missing ML IDs: {errors[:10]}")
    print(f"File saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_details()