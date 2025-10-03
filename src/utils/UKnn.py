import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class UserKNN:

    def __init__(self, k: int = 7, similarity: str = 'pearson', min_support: int = 1):
        self.k = k
        self.similarity = similarity
        self.min_support = min_support
        
        # Los iremos llenando con el entrenamiento
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.global_mean = 0.0
        self.user_means = {}
        
    def fit(self, trainset: List[Tuple[str, str, float]]):
        
        users = sorted(set(uid for uid, _, _ in trainset))
        items = sorted(set(iid for _, iid, _ in trainset))
        
        # Vamos llenando los diccionarios
        # Son todos mapeos de id que sería por ejemplo 'user1' a índices 0, 1, 2...
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.item_to_idx = {item: i for i, item in enumerate(items)}
        self.idx_to_item = {i: item for item, i in self.item_to_idx.items()}
        
        n_users = len(users)
        n_items = len(items)
        
        # Construimos la matriz usuario-item y la llenamos de NANs
        self.user_item_matrix = np.full((n_users, n_items), np.nan)
        
        # Llenamos la matriz con las valoraciones del trainset
        for uid, iid, rating in trainset:
            user_idx = self.user_to_idx[uid]
            item_idx = self.item_to_idx[iid]
            self.user_item_matrix[user_idx, item_idx] = rating
        
        # Calculamos global mean y user means
        all_ratings = [r for _, _, r in trainset]
        self.global_mean = np.mean(all_ratings)
        
        for user_idx in range(n_users):
            user_ratings = self.user_item_matrix[user_idx]
            user_ratings = user_ratings[~np.isnan(user_ratings)]
            if len(user_ratings) > 0:
                self.user_means[user_idx] = np.mean(user_ratings)
            else:
                self.user_means[user_idx] = self.global_mean
        
        # Finalmente, calculamos la matriz de similitudes
        self._compute_similarities()
        
    def _compute_similarities(self):
        """
        Compute user-user similarity matrix
        """

        n_users = len(self.user_to_idx)
        self.similarity_matrix = np.zeros((n_users, n_users))
        
        if self.similarity == 'pearson':
            self._compute_pearson_similarity()
        elif self.similarity == 'cosine':
            self._compute_cosine_similarity()
        else:
            raise ValueError(f"Unknown similarity: {self.similarity}")
    
    def _compute_pearson_similarity(self):
        """
        Compute Pearson correlation coefficient between users
        """

        # Por si queremos implementarlo en el futuro...
    
    def _compute_cosine_similarity(self):
        """
        Compute cosine similarity between users
        """

        # Sustituimos NaNs por 0 para el cálculo de similitud
        matrix_filled = np.nan_to_num(self.user_item_matrix, nan=0.0)
        self.similarity_matrix = cosine_similarity(matrix_filled)
    
    def predict(self, user_id: str, item_id: str) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        
        # Elegante por si nos columpiamos
        if user_id not in self.user_to_idx:
            print(f"User {user_id} unknown, returning global mean.")
            return self.global_mean
        
        if item_id not in self.item_to_idx:
            user_idx = self.user_to_idx[user_id]
            print(f"Item {item_id} unknown, returning user mean.")
            return self.user_means.get(user_idx, self.global_mean)
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        # Nos guardamos las similitudes del usuario
        user_similarities = self.similarity_matrix[user_idx]
        
        # Buscamos usuarios que hayan valorado el ítem
        item_ratings = self.user_item_matrix[:, item_idx]
        rated_mask = ~np.isnan(item_ratings)
        
        if not np.any(rated_mask):
            return self.user_means[user_idx]
        
        # Buscamos los k usuarios más similares que hayan valorado el ítem
        similar_users = []
        for other_idx in np.where(rated_mask)[0]:
            if other_idx != user_idx:
                sim = user_similarities[other_idx]
                similar_users.append((other_idx, sim))
        
        # Ordenamos por similitud y nos quedamos con los k mejores
        similar_users.sort(key=lambda x: abs(x[1]), reverse=True)
        similar_users = similar_users[:self.k]
        
        if not similar_users:
            print(f"User with too much nitched tastes: {user_id}, returning user mean.")
            return self.user_means[user_idx]

        # Inicializamos el numerador y el denominador
        numerator = 0.0
        denominator = 0.0
        
        for other_idx, sim in similar_users:
            other_rating = item_ratings[other_idx]
            other_mean = self.user_means[other_idx]
            
            # El rating estabilizado con la media
            numerator += sim * (other_rating - other_mean)
            denominator += abs(sim)
        
        if denominator > 0:
            prediction = self.user_means[user_idx] + (numerator / denominator)
        else:
            prediction = self.user_means[user_idx]
        
        # Lo mantenemos el rango de valores posibles
        return np.clip(prediction, 1.0, 5.0)
    
    def predict_all(self, testset: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float, float]]:
        """
        Generate predictions for all user-item pairs in testset.
        
        Args:
            testset: List of (user_id, item_id, true_rating) tuples
            
        Returns:
            List of (user_id, item_id, true_rating, predicted_rating) tuples
        """
        predictions = []
        for uid, iid, true_rating in testset:
            pred = self.predict(uid, iid)
            predictions.append((uid, iid, true_rating, pred))
        return predictions
    
    def get_top_n(self, user_ids: List[str] = None, n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top-N recommendations for each user.
        
        Args:
            user_ids: List of user IDs. If None, uses all users.
            n: Number of recommendations per user
            
        Returns:
            Dictionary {user_id: [(item_id, predicted_rating), ...]}
        """
        # Básicamente agarramos todas las predicciones, las ordenamos y nos quedamos con las n mejores
        all_predictions = self.get_all_predictions(user_ids)
        
        top_n = {}
        for uid, predictions in all_predictions.items():
            
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = predictions[:n]
        
        return top_n


def calculate_rmse(predictions: List[Tuple[str, str, float, float]]) -> float:
    """Calculate Root Mean Squared Error."""
    errors = [(true - pred)**2 for _, _, true, pred in predictions]
    return np.sqrt(np.mean(errors))


def calculate_mae(predictions: List[Tuple[str, str, float, float]]) -> float:
    """Calculate Mean Absolute Error."""
    errors = [abs(true - pred) for _, _, true, pred in predictions]
    return np.mean(errors)
