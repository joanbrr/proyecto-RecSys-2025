import pandas as pd

DATA_PATH ='data/processed'

movies_df = pd.read_csv(f'{DATA_PATH}/movies_graph_ready.csv')

movies_overviews_df = movies_df[['ml_movie_id','tmdb_id','imdb_id','overview']]

movies_overviews_df.to_csv(f'{DATA_PATH}/movies_with_overviews.csv')
