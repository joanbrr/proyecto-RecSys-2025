import numpy as np
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Set, Tuple, Optional, Union

# ============================================================================
# Data Preparation Functions
# ============================================================================

def prepare_ground_truth(testset: List[Tuple], rating_threshold: float = 3.0) -> Dict[str, Set[str]]:
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


def prepare_item_popularity(trainset) -> Dict[str, int]:
    """
    Calculates item popularity from training set.

    Args:
        trainset: Training set from surprise Dataset

    Returns:
        Dictionary {item_id: interaction_count}
    """
    item_counts = defaultdict(int)
    for _, iiid, _ in trainset.all_ratings():
        iid = trainset.to_raw_iid(iiid)
        item_counts[iid] += 1
    return dict(item_counts)


# ============================================================================
# Ranking Metrics (require only recommendations and ground truth)
# ============================================================================

def precision_at_k(recommendations: Dict[str, List[str]], 
                   ground_truth: Dict[str, Set[str]], 
                   k: int = 10) -> float:
    """Computes Precision@k averaged over all users."""
    precisions = []
    
    for uid, recs in recommendations.items():
        if uid not in ground_truth or not ground_truth[uid]:
            continue
        
        top_k = set(recs[:k])
        relevant = ground_truth[uid]
        
        if len(top_k) == 0:
            continue
            
        precision = len(top_k & relevant) / len(top_k)
        precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0.0


def recall_at_k(recommendations: Dict[str, List[str]], 
                ground_truth: Dict[str, Set[str]], 
                k: int = 10) -> float:
    """Computes Recall@k averaged over all users."""
    recalls = []
    
    for uid, recs in recommendations.items():
        if uid not in ground_truth or not ground_truth[uid]:
            continue
        
        top_k = set(recs[:k])
        relevant = ground_truth[uid]
        
        recall = len(top_k & relevant) / len(relevant)
        recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0


def f1_at_k(recommendations: Dict[str, List[str]], 
            ground_truth: Dict[str, Set[str]], 
            k: int = 10) -> float:
    """Computes F1-score@k (harmonic mean of precision and recall)."""
    precision = precision_at_k(recommendations, ground_truth, k)
    recall = recall_at_k(recommendations, ground_truth, k)
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def map_at_k(recommendations: Dict[str, List[str]], 
             ground_truth: Dict[str, Set[str]], 
             k: int = 10) -> float:
    """Computes Mean Average Precision@k."""
    ap_scores = []
    
    for uid, recs in recommendations.items():
        if uid not in ground_truth or not ground_truth[uid]:
            continue
        
        relevant = ground_truth[uid]
        top_k = recs[:k]
        
        hits = 0
        sum_precisions = 0.0
        
        for i, item_id in enumerate(top_k):
            if item_id in relevant:
                hits += 1
                sum_precisions += hits / (i + 1)
        
        num_relevant = min(k, len(relevant))
        ap = sum_precisions / num_relevant if num_relevant > 0 else 0.0
        ap_scores.append(ap)
    
    return np.mean(ap_scores) if ap_scores else 0.0


def ndcg_at_k(recommendations: Dict[str, List[str]], 
              ground_truth: Dict[str, Set[str]], 
              k: int = 10) -> float:
    """Computes Normalized Discounted Cumulative Gain@k."""
    
    def dcg(relevances):
        """Calculate DCG for a list of binary relevances."""
        relevances = np.asarray(relevances)
        if relevances.size:
            return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))
        return 0.0
    
    ndcg_scores = []
    
    for uid, recs in recommendations.items():
        if uid not in ground_truth or not ground_truth[uid]:
            continue
        
        relevant = ground_truth[uid]
        top_k = recs[:k]
        
        # Binary relevance: 1 if relevant, 0 otherwise
        relevances = [1 if item in relevant else 0 for item in top_k]
        ideal_relevances = sorted(relevances, reverse=True)
        
        dcg_score = dcg(relevances)
        idcg_score = dcg(ideal_relevances)
        
        if idcg_score > 0:
            ndcg_scores.append(dcg_score / idcg_score)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def mrr_at_k(recommendations: Dict[str, List[str]], 
             ground_truth: Dict[str, Set[str]], 
             k: int = 10) -> float:
    """
    Computes Mean Reciprocal Rank@k.
    Returns the average of the reciprocal ranks of the first relevant item.
    """
    reciprocal_ranks = []
    
    for uid, recs in recommendations.items():
        if uid not in ground_truth or not ground_truth[uid]:
            continue
        
        relevant = ground_truth[uid]
        top_k = recs[:k]
        
        for i, item in enumerate(top_k):
            if item in relevant:
                reciprocal_ranks.append(1.0 / (i + 1))
                break
        else:
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
        item_popularity: Dict {item_id: interaction_count} for novelty
        item_features: Dict {item_id: set of features} for diversity
        all_items: Set of all item_ids for coverage
    
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
        results['hit_rate'][k] = hit_rate_at_k(recs, ground_truth, k)
        results['mrr'][k] = mrr_at_k(recs, ground_truth, k)
        
        # Diversity & coverage metrics (if data available)
        if item_popularity is not None:
            results['novelty'][k] = novelty_at_k(recs, item_popularity, k)
        
        if item_features is not None:
            results['intra_list_similarity'][k] = intra_list_similarity(recs, item_features, k)
        
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
