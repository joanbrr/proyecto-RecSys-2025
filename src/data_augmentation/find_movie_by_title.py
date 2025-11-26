import pandas as pd
import requests
import os
import time
import re
import json
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
TMDB_API_KEY = os.getenv('TMDB_API')
INPUT_FILE = 'data/raw/ml-100k/u.item'
OUTPUT_FILE = 'data/interim/standarized_movies.csv'
BASE_URL = "https://api.themoviedb.org/3/search/movie"

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {TMDB_API_KEY}"
}

def parse_movielens_item_file(filepath):
    """
    Parses the u.item file from MovieLens 100k.
    Encoding is usually ISO-8859-1 (Latin-1) for this specific dataset.
    """
    # MovieLens 100k columns (we only strictly need ID, Title, Release Date)
    cols = [
        "movie_id", "movie_title", "release_date", "video_release_date",
        "imdb_url", "unknown", "Action", "Adventure", "Animation",
        "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western"
    ]
    
    df = pd.read_csv(
        filepath, 
        sep='|', 
        names=cols, 
        encoding='latin-1', 
        index_col=False
    )
    return df

def extract_title_and_year(raw_title):
    """
    Extracts the clean title and year from strings like "Toy Story (1995)".
    Returns (title, year). Year can be None if not found.
    """
    # Regex to find (YYYY) at the end of the string
    match = re.search(r'(.*)\s\((\d{4})\)$', raw_title.strip())
    if match:
        return match.group(1), match.group(2)
    else:
        return raw_title.strip(), None

def fetch_movie_details(title, year=None, max_retries=3):
    """
    Queries TMDB for the movie. 
    Returns a dictionary with status and data.
    """
    params = {
        "query": title,
        "language": "en-US",
        "page": 1
    }
    
    # Adding year drastically improves accuracy and reduces false "multiple results" flags
    if year:
        params['primary_release_year'] = year

    for attempt in range(max_retries):
        try:
            response = requests.get(BASE_URL, params=params, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limit hit
                retry_after = int(response.headers.get("Retry-After", 1))
                time.sleep(retry_after + 1)
                continue
            else:
                # Server error or other 4xx
                print(f"Error {response.status_code} for {title}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            time.sleep(1)
    
    return None

def process_movies():
    # Load Data
    print(f"Loading data from {INPUT_FILE}...")
    try:
        movies_df = parse_movielens_item_file(INPUT_FILE)
    except FileNotFoundError:
        print(f"File not found at {INPUT_FILE}. Please ensure the path is correct.")
        return

    # Prepare lists to store enriched data
    enriched_data = []

    print("Starting API queries...")
    
    # Iterate and Query
    # Using tqdm for a progress bar
    for _, row in tqdm(movies_df.iterrows(), total=movies_df.shape[0]):
        ml_id = row['movie_id']
        raw_title = row['movie_title']
        
        clean_title, year = extract_title_and_year(raw_title)
        
        # Fallback: if regex failed to find year in title, try using the release_date column
        if not year and pd.notna(row['release_date']):
            try:
                year = row['release_date'].split('-')[-1]
            except:
                pass

        api_response = fetch_movie_details(clean_title, year)
        
        # Default values
        tmdb_id = None
        tmdb_title = None
        overview = None
        genres = []
        popularity = 0.0
        vote_avg = 0.0
        vote_count = 0
        poster_path = None
        backdrop_path = None
        match_status = "failed" # found, conflict, not_found, failed
        result_count = 0

        if api_response:
            results = api_response.get('results', [])
            result_count = len(results)
            
            if result_count == 1:
                # Perfect match case
                match_status = "exact"
                data = results[0]
            elif result_count > 1:
                # Conflict case as requested
                match_status = "conflict"
                # We still take the first one as a candidate, but flag it
                # Usually the first result in TMDB is the 'best match' by popularity
                data = results[0] 
            else:
                match_status = "not_found"
                data = None

            if data:
                tmdb_id = data.get('id')
                tmdb_title = data.get('title')
                overview = data.get('overview')
                genres = data.get('genre_ids')
                popularity = data.get('popularity')
                vote_avg = data.get('vote_average')
                vote_count = data.get('vote_count')
                poster_path = data.get('poster_path')
                backdrop_path = data.get('backdrop_path')

        enriched_data.append({
            'ml_movie_id': ml_id,
            'ml_title': raw_title,
            'tmdb_id': tmdb_id,
            'tmdb_title': tmdb_title,
            'release_year': year,
            'match_status': match_status,
            'result_count': result_count,
            'overview': overview,
            'genres': genres, # Stores as list
            'popularity': popularity,
            'vote_average': vote_avg,
            'vote_count': vote_count,
            'poster_path': poster_path,
            'backdrop_path': backdrop_path
        })

        # Rate limiting: Be gentle (approx 4 req/sec max)
        time.sleep(0.25) 

    # Save Data
    print("Saving enriched data...")
    result_df = pd.DataFrame(enriched_data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    result_df.to_csv(OUTPUT_FILE, index=False)
    
    # Summary
    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Total Movies: {len(result_df)}")
    print(f"Exact Matches: {len(result_df[result_df['match_status'] == 'exact'])}")
    print(f"Conflicts (>1 result): {len(result_df[result_df['match_status'] == 'conflict'])}")
    print(f"Not Found: {len(result_df[result_df['match_status'] == 'not_found'])}")
    print(f"File saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_movies()