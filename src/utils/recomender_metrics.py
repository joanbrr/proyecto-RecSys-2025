import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
from typing import Dict, List, Set, Tuple, Optional, Union
import pandas as pd

# ============================================================================
# Data Preparation Functions
# ============================================================================

def prepare_ground_truth(testset: List[Tuple[str, str, float]], 
                        rating_threshold: float = 2.0) -> Dict[str, Set[str]]:
    """
    Creates a dictionary of relevant items for each user from the test set.

    Args:
        testset: List of tuples (user_id, item_id, rating)
        rating_threshold: Minimum rating for an item to be considered relevant

    Returns:
        Dictionary {user_id: set of relevant item_ids}
    """
    ground_truth = defaultdict(set)
    for uid, iid, rating in testset:
        if rating >= rating_threshold:
            ground_truth[uid].add(iid)
    return dict(ground_truth)


def prepare_recommendations(raw_recommendations: Dict[str, List]) -> Dict[str, List[str]]:
    """
    Standardizes recommendations format to {user_id: [item_id, ...]}.
    Handles both list of item IDs and list of (item_id, score) tuples.

    Args:
        raw_recommendations: Dict with user_id as keys and recommendations as values

    Returns:
        Dictionary {user_id: [item_id, ...]}
    """
    standardized = {}
    for uid, recs in raw_recommendations.items():
        if not recs:
            standardized[uid] = []
        elif isinstance(recs[0], (tuple, list)):
            # Extract item_id from (item_id, score) tuples
            standardized[uid] = [item for item, _ in recs]
        else:
            standardized[uid] = list(recs)
    return standardized


def get_all_items(trainset: List[Tuple[str, str, float]], 
                  testset: Optional[List[Tuple[str, str, float]]] = None) -> Set[str]:
    """
    Extracts all unique items from training and optionally test set.

    Args:
        trainset: List of tuples (user_id, item_id, rating)
        testset: Optional list of tuples (user_id, item_id, rating)

    Returns:
        Set of all unique item_ids
    """
    all_items = {iid for _, iid, _ in trainset}
    if testset:
        all_items.update(iid for _, iid, _ in testset)
    return all_items

# ============================================================================
# Rating prediction Metrics (require ratings predictions and ground truth)
# ============================================================================

def calculate_rmse(predictions: List[Tuple[str, str, float, float]]) -> float:
    """Calculate Root Mean Squared Error."""
    errors = [(true - pred)**2 for _, _, true, pred in predictions]
    return np.sqrt(np.mean(errors))


def calculate_mae(predictions: List[Tuple[str, str, float, float]]) -> float:
    """Calculate Mean Absolute Error."""
    errors = [abs(true - pred) for _, _, true, pred in predictions]
    return np.mean(errors)

# ============================================================================
# Ranking Metrics (require only recommendations and ground truth)
# ============================================================================

def precision_at_k(recommendations: Dict[str, List[str]], 
                   ground_truth: Union[Dict[str, Set[str]], Dict[str, Dict[str, float]]], 
                   k: int = 10,
                   relevance_threshold: float = 0) -> float:
    """
    Computes Precision@k averaged over all users.
    
    Args:
        recommendations: {user_id: [item_id, ...]}
        ground_truth: {user_id: set(item_id)} or {user_id: {item_id: rating}}
        k: Number of top recommendations to consider
        relevance_threshold: Minimum rating to consider item relevant (for graded format)
    """
    precisions = []
    
    for uid, recs in recommendations.items():
        uid_str = str(uid)
        
        if uid_str not in ground_truth or not ground_truth[uid_str]:
            continue
        
        top_k = [str(item) for item in recs[:k]]
        
        # Handle both formats
        if isinstance(ground_truth[uid_str], dict):
            # Graded relevance: filter by threshold
            relevant = {str(item) for item, rating in ground_truth[uid_str].items() 
                       if rating > relevance_threshold}
        else:
            # Binary relevance: convert to strings
            relevant = {str(item) for item in ground_truth[uid_str]}
        
        if len(top_k) == 0 or len(relevant) == 0:
            continue
        
        hits = len(set(top_k) & relevant)
        precision = hits / len(top_k)
        precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0.0


def recall_at_k(recommendations: Dict[str, List[str]], 
                ground_truth: Union[Dict[str, Set[str]], Dict[str, Dict[str, float]]], 
                k: int = 10,
                relevance_threshold: float = 0) -> float:
    """
    Computes Recall@k averaged over all users.
    
    Args:
        recommendations: {user_id: [item_id, ...]}
        ground_truth: {user_id: set(item_id)} or {user_id: {item_id: rating}}
        k: Number of top recommendations to consider
        relevance_threshold: Minimum rating to consider item relevant (for graded format)
    """
    recalls = []
    
    for uid, recs in recommendations.items():
        uid_str = str(uid)
        
        if uid_str not in ground_truth or not ground_truth[uid_str]:
            continue
        
        top_k = [str(item) for item in recs[:k]]
        
        # Handle both formats
        if isinstance(ground_truth[uid_str], dict):
            # Graded relevance: filter by threshold
            relevant = {str(item) for item, rating in ground_truth[uid_str].items() 
                       if rating > relevance_threshold}
        else:
            # Binary relevance: convert to strings
            relevant = {str(item) for item in ground_truth[uid_str]}
        
        if len(relevant) == 0:
            continue
        
        hits = len(set(top_k) & relevant)
        recall = hits / len(relevant)
        recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0


def f1_at_k(recommendations: Dict[str, List[str]], 
            ground_truth: Union[Dict[str, Set[str]], Dict[str, Dict[str, float]]], 
            k: int = 10,
            relevance_threshold: float = 0) -> float:
    """
    Computes F1-score@k (harmonic mean of precision and recall).
    
    More robust implementation that computes F1 per-user then averages,
    rather than computing F1 from averaged precision/recall.
    """
    f1_scores = []
    
    for uid, recs in recommendations.items():
        uid_str = str(uid)
        
        if uid_str not in ground_truth or not ground_truth[uid_str]:
            continue
        
        top_k = [str(item) for item in recs[:k]]
        
        # Handle both formats
        if isinstance(ground_truth[uid_str], dict):
            relevant = {str(item) for item, rating in ground_truth[uid_str].items() 
                       if rating > relevance_threshold}
        else:
            relevant = {str(item) for item in ground_truth[uid_str]}
        
        if len(relevant) == 0 or len(top_k) == 0:
            continue
        
        hits = len(set(top_k) & relevant)
        precision = hits / len(top_k)
        recall = hits / len(relevant)
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)
    
    return np.mean(f1_scores) if f1_scores else 0.0


def map_at_k(recommendations: Dict[str, List[str]], 
             ground_truth: Union[Dict[str, Set[str]], Dict[str, Dict[str, float]]], 
             k: int = 10,
             relevance_threshold: float = 0) -> float:
    """
    Computes Mean Average Precision@k.
    
    Formula (as implemented in pyreclab):
    AP@k = (1/tp) * sum(P@i) for all relevant items found in top-k
    where tp = number of relevant items found (true positives)
    
    Args:
        recommendations: {user_id: [item_id, ...]}
        ground_truth: {user_id: set(item_id)} or {user_id: {item_id: rating}}
        k: Number of top recommendations to consider
        relevance_threshold: Minimum rating to consider item relevant (for graded format)
    """
    ap_scores = []
    
    for uid, recs in recommendations.items():
        uid_str = str(uid)
        
        if uid_str not in ground_truth or not ground_truth[uid_str]:
            continue
        
        # Handle both formats
        if isinstance(ground_truth[uid_str], dict):
            relevant = {str(item) for item, rating in ground_truth[uid_str].items() 
                       if rating > relevance_threshold}
        else:
            relevant = {str(item) for item in ground_truth[uid_str]}
        
        if len(relevant) == 0:
            continue
        
        top_k = [str(item) for item in recs[:k]]
        
        tp = 0  # True positives (hits)
        sum_precisions = 0.0
        
        # Iterate through recommendations
        for i, item_id in enumerate(top_k):
            if item_id in relevant:
                tp += 1
                # Precision at position i+1
                sum_precisions += tp / (i + 1)
        
        # Divide by number of relevant items FOUND (tp), not total relevant
        avg_precision = sum_precisions / tp if tp > 0 else 0.0
        ap_scores.append(avg_precision)
    
    return np.mean(ap_scores) if ap_scores else 0.0

def ndcg_at_k(recommendations: Dict[str, List[str]], 
              ground_truth: Union[Dict[str, Set[str]], Dict[str, Dict[str, float]]], 
              k: int = 10,
              relevance_threshold: float = 0) -> float:
    """
    Computes Normalized Discounted Cumulative Gain@k.
    
    Implementation matches pyreclab's NDCG calculation:
    - DCG: sum of (1 / log2(i+1)) for each relevant item at position i
    - IDCG: sum of (1 / log2(i+1)) for positions 1 to min(k, |relevant|)
    
    Args:
        recommendations: {user_id: [item_id, ...]}
        ground_truth: {user_id: set(item_id)} or {user_id: {item_id: rating}}
        k: Number of top recommendations to consider
        relevance_threshold: Minimum rating to consider item relevant (for graded format)
    """
    ndcg_scores = []
    
    for uid, recs in recommendations.items():
        uid_str = str(uid)
        
        if uid_str not in ground_truth or not ground_truth[uid_str]:
            continue
        
        # Handle both formats
        if isinstance(ground_truth[uid_str], dict):
            relevant = {str(item) for item, rating in ground_truth[uid_str].items() 
                       if rating > relevance_threshold}
        else:
            relevant = {str(item) for item in ground_truth[uid_str]}
        
        if len(relevant) == 0:
            continue
        
        top_k = [str(item) for item in recs[:k]]
        
        dcg = 0.0
        idcg = 0.0
        
        # Compute DCG and IDCG
        for i, item_id in enumerate(top_k):
            position = i + 1  # 1-indexed position
            log2_i_plus_1 = np.log2(position + 1)
            
            # DCG: add discount for relevant items found
            if item_id in relevant:
                dcg += 1.0 / log2_i_plus_1
            
            # IDCG: add discount for positions up to min(k, |relevant|)
            if position <= len(relevant):
                idcg += 1.0 / log2_i_plus_1
        
        # Normalize DCG by IDCG
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def mrr_at_k(recommendations: Dict[str, List[str]], 
             ground_truth: Union[Dict[str, Set[str]], Dict[str, Dict[str, float]]], 
             k: int = 10,
             relevance_threshold: float = 0) -> float:
    """
    Computes Mean Reciprocal Rank@k.
    
    Returns the average of the reciprocal ranks of the first relevant item.
    MRR is particularly useful for tasks where finding at least one relevant
    item quickly is important (e.g., search, question answering).
    
    Args:
        recommendations: {user_id: [item_id, ...]}
        ground_truth: {user_id: set(item_id)} or {user_id: {item_id: rating}}
        k: Number of top recommendations to consider
        relevance_threshold: Minimum rating to consider item relevant (for graded format)
    
    Returns:
        Mean reciprocal rank across all users
    """
    reciprocal_ranks = []
    
    for uid, recs in recommendations.items():
        uid_str = str(uid)
        
        if uid_str not in ground_truth or not ground_truth[uid_str]:
            continue
        
        # Handle both formats
        if isinstance(ground_truth[uid_str], dict):
            # Graded relevance: filter by threshold
            relevant = {str(item) for item, rating in ground_truth[uid_str].items() 
                       if rating > relevance_threshold}
        else:
            # Binary relevance: convert to strings
            relevant = {str(item) for item in ground_truth[uid_str]}
        
        if len(relevant) == 0:
            continue
        
        top_k = [str(item) for item in recs[:k]]
        
        # Find the first relevant item
        for i, item in enumerate(top_k):
            if item in relevant:
                reciprocal_ranks.append(1.0 / (i + 1))
                break
        else:
            # No relevant item found in top-k
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


# ============================================================================
# Diversity & Coverage Metrics
# ============================================================================

def novelty_at_k(recommendations: Dict[str, List[str]], 
                 item_popularity: Dict[str, int], 
                 k: int = 10) -> float:
    """
    Calculates average novelty based on item popularity.
    Novelty = -log2(popularity / total_interactions)
    Higher values indicate more novel (less popular) recommendations.
    """
    total_interactions = sum(item_popularity.values())
    novelty_scores = []
    
    for uid, recs in recommendations.items():
        top_k = recs[:k]
        user_novelty = []
        
        for item in top_k:
            if item in item_popularity:
                pop = item_popularity[item]
                novelty = -np.log2(pop / total_interactions)
                user_novelty.append(novelty)
        
        if user_novelty:
            novelty_scores.append(np.mean(user_novelty))
    
    return np.mean(novelty_scores) if novelty_scores else 0.0


def catalog_coverage(recommendations: Dict[str, List[str]], 
                     all_items: Set[str], 
                     k: int = 10) -> float:
    """
    Calculates what proportion of the catalog is recommended to at least one user.
    Returns a value between 0 and 1.
    """
    recommended_items = set()
    
    for recs in recommendations.values():
        recommended_items.update(recs[:k])
    
    return len(recommended_items) / len(all_items) if all_items else 0.0


def intra_list_similarity(recommendations: Dict[str, List[str]], 
                         item_features: Dict[str, Set[str]], 
                         k: int = 10) -> float:
    """
    Calculates average Intra-List Similarity using Jaccard similarity.
    Lower values indicate more diverse recommendations.
    """
    ils_scores = []
    
    for uid, recs in recommendations.items():
        top_k = recs[:k]
        
        if len(top_k) < 2:
            continue
        
        similarities = []
        for item1, item2 in combinations(top_k, 2):
            features1 = item_features.get(item1, set())
            features2 = item_features.get(item2, set())
            
            intersection = len(features1 & features2)
            union = len(features1 | features2)
            
            if union > 0:
                similarities.append(intersection / union)
        
        if similarities:
            ils_scores.append(np.mean(similarities))
    
    return np.mean(ils_scores) if ils_scores else 0.0

# ============================================================================
# Custom metrics
# ============================================================================


def calculate_cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Calcula la similitud de coseno entre dos vectores representados como diccionarios."""
    intersection = set(vec1.keys()) & set(vec2.keys())
    
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = np.sqrt(sum1) * np.sqrt(sum2)

    if not denominator:
        return 0.0
    return float(numerator) / denominator


def mood_alignment(
    recommendations: Dict[str, List[Union[str, Tuple[str, float]]]],
    user_mood_profiles: Dict[str, Dict[str, float]],
    movie_mood_map: Dict[str, set]
) -> float:
    """
    Calcula el alineamiento de mood promedio usando similitud de coseno.

    Args:
        recommendations: Dict {user_id: [movieId, ...]}.
        user_mood_profiles: Perfil de mood pre-calculado para cada usuario.
        movie_mood_map: Mapa pre-calculado de movieId a sus moods.
    """
    alignment_scores = []

    # Estandarizar el formato de las recomendaciones por si vienen con puntaje
    recs_clean = {
        str(uid): [item[0] if isinstance(item, tuple) else item for item in items]
        for uid, items in recommendations.items()
    }

    for user_id, rec_items in recs_clean.items():
        if user_id not in user_mood_profiles or not rec_items:
            continue

        # Perfil histórico del usuario
        user_profile = user_mood_profiles[user_id]

        # Crear el perfil de mood para la lista de recomendaciones
        rec_moods_list = [mood for item_id in rec_items for mood in movie_mood_map.get(str(item_id), set())]
        
        if not rec_moods_list:
            continue
        
        rec_mood_counts = Counter(rec_moods_list)
        total_rec_moods = sum(rec_mood_counts.values())
        
        # Normalizar para crear el vector de perfil de la recomendación
        rec_profile = {mood: count / total_rec_moods for mood, count in rec_mood_counts.items()}
        
        # Calcular la similitud de coseno y guardarla
        similarity = calculate_cosine_similarity(user_profile, rec_profile)
        alignment_scores.append(similarity)

    # El resultado final es el promedio de alineamiento entre todos los usuarios
    return np.mean(alignment_scores) if alignment_scores else 0.0


# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluate_recommendations(
    recommendations: Dict[str, List[Union[str, Tuple[str, float]]]],
    ground_truth: Dict[str, Set[str]],
    k_values: List[int] = [5, 10, 20],
    item_popularity: Optional[Dict[str, int]] = None,
    item_features: Optional[Dict[str, Set[str]]] = None,
    all_items: Optional[Set[str]] = None
) -> Dict[str, Dict[int, float]]:
    """
    Comprehensive evaluation of recommendation quality.
    
    Args:
        recommendations: Dict {user_id: [item_id, ...] or [(item_id, score), ...]}
        ground_truth: Dict {user_id: set of relevant item_ids}
        k_values: List of k values to evaluate at
        item_popularity: Dict {item_id: interaction_count} for novelty metric
        item_features: Dict {item_id: set of features} for diversity metric
        all_items: Set of all item_ids for coverage metric
    
    Returns:
        Dictionary of metrics with structure {metric_name: {k: score}}

    """
    # Standardize recommendations format
    recs = prepare_recommendations(recommendations)
    
    results = defaultdict(dict)
    
    for k in k_values:
        # Core ranking metrics (always available)
        results['precision'][k] = precision_at_k(recs, ground_truth, k)
        results['recall'][k] = recall_at_k(recs, ground_truth, k)
        results['f1'][k] = f1_at_k(recs, ground_truth, k)
        results['map'][k] = map_at_k(recs, ground_truth, k)
        results['ndcg'][k] = ndcg_at_k(recs, ground_truth, k)
        results['mrr'][k] = mrr_at_k(recs, ground_truth, k)
        
        # Diversity & coverage metrics (if data available)
        if item_popularity is not None:
            results['novelty'][k] = novelty_at_k(recs, item_popularity, k)
        
        if all_items is not None:
            results['catalog_coverage'][k] = catalog_coverage(recs, all_items, k)
    
    return dict(results)


def print_evaluation_results(results: Dict[str, Dict[int, float]]):
    """Pretty print evaluation results."""
    print("\n" + "="*60)
    print("RECOMMENDATION EVALUATION RESULTS")
    print("="*60)
    
    for metric_name, k_scores in sorted(results.items()):
        print(f"\n{metric_name.upper().replace('_', ' ')}:")
        for k, score in sorted(k_scores.items()):
            print(f"  @{k:2d}: {score:.4f}")
    print("\n" + "="*60)
