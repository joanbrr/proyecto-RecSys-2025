import pandas as pd
import numpy as np
from collections import Counter
import ast # Para convertir strings de listas a listas de forma segura
from typing import Dict, List, Tuple, Union
from numpy.linalg import norm

def create_mood_profiles(train_df: pd.DataFrame, movies_info: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], Dict[str, set]]:
    """
    Pre-calcula los perfiles de mood de los usuarios y un mapa de moods por película.

    Args:
        train_df: DataFrame con el historial de ratings (user_id, movieId, ...).
        movies_info: DataFrame con la información aumentada de las películas, incluyendo moods.

    Returns:
        Una tupla conteniendo:
        - user_mood_profiles: Dict {user_id: {mood: normalized_score, ...}}
        - movie_mood_map: Dict {movieId: set_of_moods}
    """
    
    # Usamos ast.literal_eval para convertir de forma segura el string "mood1, mood2" a una lista
    movies_info['mood_list'] = movies_info['mood'].apply(lambda x: [m.strip() for m in x.split(',')] if isinstance(x, str) and x.strip() else ['neutral'])
    # Aseguramos que los IDs son del mismo tipo para el merge
    movie_mood_map = movies_info.set_index('movieId')['mood_list'].apply(set).to_dict()

    train_df['movieId_str'] = train_df['movieId'].astype(str)
    movies_info['movieId_str'] = movies_info['movieId'].astype(str)
    
    # Unimos el historial del usuario con los moods de las películas que ha visto
    user_history_moods = train_df.merge(
        movies_info[['movieId_str', 'mood_list']],
        on="movieId_str",
        how='left'
    )
    
    user_mood_profiles = {}
    # Iteramos sobre cada usuario
    for user_id, group in user_history_moods.groupby('user_id'):
        # Juntamos todas las listas de moods de todas las películas que ha visto
        all_moods = [mood for sublist in group['mood_list'].dropna() for mood in sublist]
        
        if not all_moods:
            continue
            
        # Contamos la frecuencia de cada mood
        mood_counts = Counter(all_moods)
        total_moods = sum(mood_counts.values())
        
        # Normalizamos para tener un vector de probabilidad (perfil)
        user_mood_profiles[str(user_id)] = {mood: count / total_moods for mood, count in mood_counts.items()}
        
    return user_mood_profiles, {str(k): v for k, v in movie_mood_map.items()}

def augment_users_with_moods(users_df: pd.DataFrame, user_mood_profiles: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Añade columnas de distribución de moods al DataFrame de usuarios.

    Args:
        users_df: DataFrame original de usuarios.
        user_mood_profiles: Diccionario {user_id: {mood: score, ...}}

    Returns:
        DataFrame de usuarios con columnas extra para cada mood.
    """
    # Obtener todos los moods presentes
    all_moods = set()
    for mood_dist in user_mood_profiles.values():
        all_moods.update(mood_dist.keys())
    all_moods = sorted(all_moods)

    # Inicializar las columnas de mood en el DataFrame de usuarios
    for mood in all_moods:
        users_df[f'mood_{mood}'] = 0.0

    # Asignar los valores de mood a cada usuario
    for idx, row in users_df.iterrows():
        uid = str(row['user_id'])
        mood_dist = user_mood_profiles.get(uid, {})
        for mood in all_moods:
            users_df.at[idx, f'mood_{mood}'] = mood_dist.get(mood, 0.0)

    return users_df

def main():
    data_folder = './data/raw/ml-100k/'

    train_df = pd.read_csv('./data/processed/processed_train.csv')

    u_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']

    users_df = pd.read_csv(f'{data_folder}u.user', sep='|', names=u_cols, encoding='latin-1')

    movies_info = pd.read_csv("./data/processed/ml-100k-augmented.csv")

    user_mood_profiles, movie_mood_map = create_mood_profiles(train_df, movies_info)
    users_df_augmented = augment_users_with_moods(users_df, user_mood_profiles)
    users_df_augmented.to_csv('./data/processed/users_with_moods.csv', index=False)

if __name__ == "__main__":
    main()