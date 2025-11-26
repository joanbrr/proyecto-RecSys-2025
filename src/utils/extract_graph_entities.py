import pandas as pd
import json
import os
from tqdm import tqdm
import ast

# Configuration
INPUT_FILE = 'data/processed/movies_final.csv'
NODES_DIR = 'data/processed/nodes'
OUTPUT_FILE = 'data/processed/movies_graph_ready.csv'

def load_json_safe(json_str):
    """Safely loads JSON, handling empty strings or nans."""
    if pd.isna(json_str) or json_str == "":
        return []
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return []

def extract_entities():
    print(f"Loading final dataset from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("Input file not found. Please run the credit enrichment step first.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Ensure output directory exists
    os.makedirs(NODES_DIR, exist_ok=True)

    # --- STORAGE DICTIONARIES ---
    # Genres: We only have names, so we generate new sequential IDs
    # Format: { "Action": 1, "Comedy": 2 }
    genre_map = {} 
    
    # Keywords: We have TMDB IDs. 
    # Format: { 123: "fight" }
    keyword_map = {}

    # Directors: We have TMDB IDs.
    # Format: { 123: {"name": "David Fincher", "profile_path": "..."} }
    director_map = {}

    # Writers: We have TMDB IDs.
    # Format: { 123: {"name": "Chuck Palahniuk", "profile_path": "..."} }
    writer_map = {}

    # --- NEW COLUMNS FOR THE MAIN DATASET ---
    # We will replace the complex objects with simple lists of IDs
    movie_genre_ids = []
    movie_keyword_ids = []
    movie_director_ids = []
    movie_writer_ids = []

    print("Extracting nodes and normalizing dataset...")

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        
        # 1. PROCESS GENRES (String: "Drama|Thriller")
        g_ids = []
        if pd.notna(row['genres']):
            genres_list = row['genres'].split('|')
            for g_name in genres_list:
                g_name = g_name.strip()
                if not g_name: continue
                
                if g_name not in genre_map:
                    # Assign new ID (starting at 1)
                    genre_map[g_name] = len(genre_map) + 1
                
                g_ids.append(genre_map[g_name])
        movie_genre_ids.append(g_ids)

        # 2. PROCESS KEYWORDS (JSON)
        k_ids = []
        keywords_list = load_json_safe(row['keywords'])
        for k in keywords_list:
            kid = k['id']
            kname = k['name']
            
            # Store in map if not exists
            if kid not in keyword_map:
                keyword_map[kid] = kname
            
            k_ids.append(kid)
        movie_keyword_ids.append(k_ids)

        # 3. PROCESS DIRECTORS (JSON)
        d_ids = []
        directors_list = load_json_safe(row['directors'])
        for d in directors_list:
            did = d['id']
            
            if did not in director_map:
                director_map[did] = {
                    'name': d['name'],
                    'profile_path': d.get('profile_path')
                }
            
            d_ids.append(did)
        movie_director_ids.append(d_ids)

        # 4. PROCESS WRITERS (JSON)
        w_ids = []
        writers_list = load_json_safe(row['writers'])
        for w in writers_list:
            wid = w['id']
            
            if wid not in writer_map:
                writer_map[wid] = {
                    'name': w['name'],
                    'profile_path': w.get('profile_path')
                }
            
            w_ids.append(wid)
        movie_writer_ids.append(w_ids)

    # --- SAVE NODE FILES ---
    
    print("Saving node CSVs...")

    # Genres
    pd.DataFrame(list(genre_map.items()), columns=['id', 'name']).to_csv(f'{NODES_DIR}/genres.csv', index=False)
    
    # Keywords
    pd.DataFrame(list(keyword_map.items()), columns=['id', 'name']).to_csv(f'{NODES_DIR}/keywords.csv', index=False)

    # Directors
    d_rows = [{'id': k, 'name': v['name'], 'profile_path': v['profile_path']} for k, v in director_map.items()]
    pd.DataFrame(d_rows).to_csv(f'{NODES_DIR}/directors.csv', index=False)

    # Writers
    w_rows = [{'id': k, 'name': v['name'], 'profile_path': v['profile_path']} for k, v in writer_map.items()]
    pd.DataFrame(w_rows).to_csv(f'{NODES_DIR}/writers.csv', index=False)

    # --- SAVE NORMALIZED DATASET ---
    
    print("Saving normalized movies dataset...")
    
    # Create a copy of the original DF
    final_df = df.copy()
    
    # Replace columns with ID lists
    final_df['genres'] = movie_genre_ids
    final_df['keywords'] = movie_keyword_ids
    final_df['directors'] = movie_director_ids
    final_df['writers'] = movie_writer_ids
    
    # Ensure lists are stringified for CSV storage (e.g., "[1, 2, 3]")
    # This makes it loadable as a list later
    final_df['genres'] = final_df['genres'].apply(json.dumps)
    final_df['keywords'] = final_df['keywords'].apply(json.dumps)
    final_df['directors'] = final_df['directors'].apply(json.dumps)
    final_df['writers'] = final_df['writers'].apply(json.dumps)

    final_df.to_csv(OUTPUT_FILE, index=False)

    # --- REPORT ---
    print("-" * 30)
    print("Extraction Complete.")
    print(f"Unique Genres: {len(genre_map)}")
    print(f"Unique Keywords: {len(keyword_map)}")
    print(f"Unique Directors: {len(director_map)}")
    print(f"Unique Writers: {len(writer_map)}")
    print(f"Node files saved to: {NODES_DIR}/")
    print(f"Normalized dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_entities()