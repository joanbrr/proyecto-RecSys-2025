import requests
import pandas as pd
import time
from tqdm import tqdm
from dotenv import load_dotenv
import os

# Cargar la clave de la API
load_dotenv()
MY_API_KEY = os.getenv('TMDB_API')

# Configurar los headers para autenticación con Bearer token
headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {MY_API_KEY}"
}

def get_movie_details_by_tmdb_id(tmdb_id):
    """
    Obtiene los detalles de una película usando su TMDB ID
    """
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?language=en-US"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            print(f"Película {tmdb_id} no encontrada")
            return None
        else:
            print(f"Error {response.status_code} para la película {tmdb_id}")
            return None
    except Exception as e:
        print(f"Excepción al obtener la película {tmdb_id}: {e}")
        return None

def get_movie_keywords(tmdb_id):
    """
    Obtiene las palabras clave de una película usando su TMDB ID
    """
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/keywords"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('keywords', [])
        return []
    except:
        return []

def infer_mood(genres_list, keywords_list):
    """
    Infiera el estado de ánimo (mood) a partir de los géneros y palabras clave
    """
    genres_str = ' '.join([g['name'] for g in genres_list]).lower() if genres_list else ''
    keywords_str = ' '.join([kw['name'] for kw in keywords_list]).lower() if keywords_list else ''
    moods = []

    # Inferencia basada en géneros
    if any(g in genres_str for g in ['comedy', 'animation']):
        moods.append('ligero')
    if 'romance' in genres_str:
        moods.append('romántico')
    if any(g in genres_str for g in ['horror', 'thriller']):
        moods.append('intenso')
    if any(g in genres_str for g in ['action', 'adventure']):
        moods.append('emocionante')
    if 'drama' in genres_str and 'crime' not in genres_str:
        moods.append('emocional')
    if any(g in genres_str for g in ['documentary', 'history']):
        moods.append('reflexivo')
    if 'family' in genres_str or 'children' in genres_str:
        moods.append('familiar')

    # Refinamiento usando palabras clave
    if any(kw in keywords_str for kw in ['peaceful', 'calm', 'meditation', 'nature']):
        moods.append('relajante')
    if any(kw in keywords_str for kw in ['dark', 'disturbing', 'violent']):
        moods.append('oscuro')
    if any(kw in keywords_str for kw in ['inspiring', 'uplifting', 'heartwarming']):
        moods.append('inspirador')
    if any(kw in keywords_str for kw in ['suspense', 'mystery', 'twist']):
        moods.append('suspenso')

    return moods if moods else ['neutral']

def augment_single_movie(row):
    """
    Añade información de TMDB a una sola película
    """
    result = row.to_dict()
    if pd.isna(row['tmdbId']):
        result.update({
            'runtime': None,
            'tmdb_rating': None,
            'vote_count': None,
            'popularity': None,
            'budget': None,
            'revenue': None,
            'tagline': None,
            'overview': None,
            'keywords': None,
            'mood': None,
            'release_date': None,
            'original_language': None
        })
        return result

    tmdb_id = int(row['tmdbId'])
    movie_data = get_movie_details_by_tmdb_id(tmdb_id)
    if not movie_data:
        result.update({
            'runtime': None,
            'tmdb_rating': None,
            'vote_count': None,
            'popularity': None,
            'budget': None,
            'revenue': None,
            'tagline': None,
            'overview': None,
            'keywords': None,
            'mood': None,
            'release_date': None,
            'original_language': None
        })
        return result

    keywords = get_movie_keywords(tmdb_id)
    mood = infer_mood(movie_data.get('genres', []), keywords)
    result.update({
        'runtime': movie_data.get('runtime'),
        'tmdb_rating': movie_data.get('vote_average'),
        'vote_count': movie_data.get('vote_count'),
        'popularity': movie_data.get('popularity'),
        'budget': movie_data.get('budget'),
        'revenue': movie_data.get('revenue'),
        'tagline': movie_data.get('tagline'),
        'overview': movie_data.get('overview'),
        'keywords': ', '.join([kw['name'] for kw in keywords[:15]]),  # Máximo 15 palabras clave
        'mood': ', '.join(mood),
        'release_date': movie_data.get('release_date'),
        'original_language': movie_data.get('original_language')
    })
    return result

def augment_movielens_dataset(limit=None, save_every=50):
    """
    Función principal para enriquecer todo el dataset de MovieLens

    Args:
        limit: Procesar solo las primeras N películas (útil para pruebas)
        save_every: Guardar el progreso cada N películas
    """
    print("Cargando datos de MovieLens...")
    data = pd.read_csv("./data/processed/processed_movies.csv")

    if limit:
        data = data.head(limit)
        print(f"\nProcesando solo las primeras {limit} películas")

    # Filtrar películas que tienen TMDB ID
    movies_with_tmdb = data[data['tmdbId'].notna()].copy()
    print(f"\nSe van a enriquecer {len(movies_with_tmdb)} películas con TMDB ID")

    # Procesar películas
    augmented_movies = []
    request_count = 0
    start_time = time.time()

    print("\nDescargando datos de TMDB...")
    print("Límite de velocidad: 50 peticiones por segundo para autenticación Bearer token")

    for idx, row in tqdm(movies_with_tmdb.iterrows(), total=len(movies_with_tmdb)):
        # Enriquecer la película
        augmented_movie = augment_single_movie(row)
        augmented_movies.append(augmented_movie)

        # Control de velocidad: TMDB permite ~50 peticiones/segundo con Bearer token
        # Para estar seguros, usamos 40 peticiones por segundo
        request_count += 2  # Hacemos 2 peticiones por película (detalles + keywords)
        if request_count >= 40:
            elapsed = time.time() - start_time
            if elapsed < 1:
                time.sleep(1 - elapsed)
            request_count = 0
            start_time = time.time()

        # Guardar progreso cada cierto número de películas
        if len(augmented_movies) % save_every == 0:
            temp_df = pd.DataFrame(augmented_movies)
            temp_df.to_csv('./data/processed/ml-100k-augmented-temp.csv', index=False)
            print(f"\n✓ Progreso guardado: {len(augmented_movies)} películas procesadas")

    # Crear DataFrame final
    result_df = pd.DataFrame(augmented_movies)

    # Guardar resultado final
    output_file = './data/processed/ml-100k-augmented.csv'
    result_df.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print("✓ ENRIQUECIMIENTO COMPLETO")
    print(f"{'='*60}")
    print(f"Guardado en: {output_file}")
    print(f"\nEstadísticas:")
    print(f"  Total de películas procesadas: {len(result_df)}")
    print(f"  Películas con duración: {result_df['runtime'].notna().sum()}")
    print(f"  Películas con mood: {result_df['mood'].notna().sum()}")
    print(f"  Películas con keywords: {result_df['keywords'].notna().sum()}")
    print(f"  Duración media: {result_df['runtime'].mean():.1f} minutos")
    print(f"  Nota media TMDB: {result_df['tmdb_rating'].mean():.2f}/10")

    return result_df

def analyze_moods(df):
    """
    Analiza la distribución de moods en el dataset
    """
    print("\n" + "="*60)
    print("ANÁLISIS DE MOODS")
    print("="*60)

    # Extraer todos los moods
    all_moods = []
    for mood_str in df['mood'].dropna():
        moods = [m.strip() for m in mood_str.split(',')]
        all_moods.extend(moods)

    # Contar moods
    mood_counts = pd.Series(all_moods).value_counts()

    print("\nDistribución de moods:")
    for mood, count in mood_counts.items():
        print(f"  {mood}: {count} películas ({count/len(df)*100:.1f}%)")

    return mood_counts

# Ejecución principal
if __name__ == "__main__":
    print("="*60)
    print("Enriquecimiento TMDB para MovieLens 100K")
    print("="*60)

    # Preguntar al usuario si quiere limitar el número de películas (opcional)
    respuesta = input("\n¿Quieres procesar solo una parte del dataset para pruebas? (s/n): ")
    if respuesta.lower() == 's':
        limite = input("¿Cuántas películas quieres procesar?: ")
        try:
            limite = int(limite)
        except:
            print("Valor no válido, se procesarán todas las películas.")
            limite = None
    else:
        limite = None

    df_resultado = augment_movielens_dataset(limit=limite)

    print("\nMuestra de resultados:")
    print(df_resultado[['title', 'runtime', 'tmdb_rating', 'mood']].to_string())

    # Analizar moods
    analyze_moods(df_resultado)

    print("\n✓ Proceso finalizado. Consulta 'ml-100k-augmented.csv' para ver los resultados.")