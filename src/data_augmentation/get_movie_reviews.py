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
OUTPUT_FILE = 'data/interim/movies_with_reviews.csv'
BASE_URL = "https://api.themoviedb.org/3/movie/{}/reviews"

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {TMDB_API_KEY}"
}

def get_movie_reviews(tmdb_id, max_retries=3):
    """
    Fetches reviews for a specific movie ID.
    Returns a list of strings (review content only).
    """
    url = BASE_URL.format(tmdb_id)
    params = {
        "language": "en-US",
        "page": 1 # We stick to page 1 to keep data size manageable, usually contains top reviews
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                # Extract only the content text
                reviews_content = [r.get('content', '') for r in results]
                return reviews_content
            elif response.status_code == 404:
                print(f"Reviews endpoint for ID {tmdb_id} not found (404).")
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

def process_reviews():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        # We only need the ID columns to link back
        df = pd.read_csv(INPUT_FILE, usecols=['ml_movie_id', 'tmdb_id'])
    except FileNotFoundError:
        print(f"Input file not found at {INPUT_FILE}. Please run the details enrichment step first.")
        return

    print(f"Processing reviews for {len(df)} movies...")

    reviews_data = []
    errors = []
    
    # Iterate through the dataframe
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        ml_id = row['ml_movie_id']
        tmdb_id = row['tmdb_id']
        
        # Ensure valid ID before requesting
        if pd.isna(tmdb_id):
            # If we don't have a TMDB ID, we can't get reviews. 
            # We append empty list to keep rows aligned if merging later, 
            # though this file is separate.
            reviews_data.append({
                'ml_movie_id': ml_id,
                'tmdb_id': None,
                'reviews': "[]"
            })
            continue

        tmdb_id = int(tmdb_id)
        review_list = get_movie_reviews(tmdb_id)
        
        if review_list is not None:
            # Dump to JSON string to preserve structure and handle escaping quotes/newlines
            reviews_str = json.dumps(review_list)
            
            reviews_data.append({
                'ml_movie_id': ml_id,
                'tmdb_id': tmdb_id,
                'reviews': reviews_str
            })
        else:
            # Fallback for failures
            reviews_data.append({
                'ml_movie_id': ml_id,
                'tmdb_id': tmdb_id,
                'reviews': "[]"
            })
            errors.append(ml_id)

        # Rate limiting
        time.sleep(0.25)

    # Create DataFrame
    result_df = pd.DataFrame(reviews_data)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)

    # Summary
    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Total Processed: {len(result_df)}")
    print(f"Errors/Failed Requests: {len(errors)}")
    
    # metrics
    non_empty = result_df[result_df['reviews'] != "[]"]
    print(f"Movies with at least 1 review: {len(non_empty)}")
    
    print(f"File saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_reviews()