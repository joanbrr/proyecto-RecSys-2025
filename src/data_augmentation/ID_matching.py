import pandas as pd

data_folder = './data/raw/ml-100k/'

r_cols = ['user_id', 'item_id', 'rating', 'timestamp']

train_df = pd.read_csv(f'{data_folder}u1.base', sep='\t', names=r_cols, encoding='latin-1')
train_df['rating'] = train_df['rating'].astype(int)

movies_processed = pd.read_csv("./data/processed/processed_movies.csv")
movies_processed = movies_processed[['movieId', 'item_id', 'imdbId', 'tmdbId']]

merged_df = movies_processed.merge(train_df, how='inner', on='item_id')

merged_df.to_csv('./data/processed/processed_train.csv', index=False)
