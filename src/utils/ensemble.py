from typing import Dict, List, Tuple
import numpy as np


class HybridEnsemble:
    """
    Ensemble híbrido que combina recomendaciones de múltiples modelos.
    """
    
    def __init__(self, strategy: str = 'weighted', weights: List[float] = None):
        """
        Args:
            strategy: 'weighted', 'rank_fusion', o 'cascade'
            weights: Pesos para cada modelo (debe sumar 1.0)
        """
        self.strategy = strategy
        self.weights = weights or [0.5, 0.5]
    
    def weighted_hybrid(self, 
                       recs1: Dict[int, List[Tuple[int, float]]], 
                       recs2: Dict[int, List[Tuple[int, float]]],
                       n: int = 50) -> Dict[int, List[int]]:
        """
        Combina scores ponderados de dos modelos.
        """
        ensemble_recs = {}
        all_users = set(recs1.keys()) | set(recs2.keys())
        
        for user in all_users:
            scores = {}
            
            # Scores del modelo 1
            if user in recs1:
                for item, score in recs1[user]:
                    scores[item] = scores.get(item, 0) + self.weights[0] * score
            
            # Scores del modelo 2
            if user in recs2:
                for item, score in recs2[user]:
                    scores[item] = scores.get(item, 0) + self.weights[1] * score
            
            # Top-N por score combinado
            top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
            ensemble_recs[user] = [item for item, _ in top_items]
        
        return ensemble_recs
    
    def rank_fusion(self,
                   recs1: Dict[int, List[int]],
                   recs2: Dict[int, List[int]],
                   n: int = 50) -> Dict[int, List[int]]:
        """
        Reciprocal Rank Fusion: combina rankings sin necesidad de scores.
        """
        ensemble_recs = {}
        all_users = set(recs1.keys()) | set(recs2.keys())
        k = 60  # Constante para RRF
        
        for user in all_users:
            scores = {}
            
            # RRF del modelo 1
            if user in recs1:
                for rank, item in enumerate(recs1[user], 1):
                    scores[item] = scores.get(item, 0) + 1 / (k + rank)
            
            # RRF del modelo 2
            if user in recs2:
                for rank, item in enumerate(recs2[user], 1):
                    scores[item] = scores.get(item, 0) + 1 / (k + rank)
            
            # Top-N por RRF score
            top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
            ensemble_recs[user] = [item for item, _ in top_items]
        
        return ensemble_recs
    
    def cascade(self,
               recs1: Dict[int, List[int]],
               recs2: Dict[int, List[int]],
               n: int = 50,
               split_ratio: float = 0.7) -> Dict[int, List[int]]:
        """
        Cascade: toma primeros N*split_ratio del modelo 1, resto del modelo 2.
        """
        ensemble_recs = {}
        all_users = set(recs1.keys()) | set(recs2.keys())
        n1 = int(n * split_ratio)
        n2 = n - n1
        
        for user in all_users:
            combined = []
            
            # Primeros n1 del modelo 1
            if user in recs1:
                combined.extend(recs1[user][:n1])
            
            # Completar con modelo 2 (evitando duplicados)
            if user in recs2:
                seen = set(combined)
                for item in recs2[user]:
                    if item not in seen and len(combined) < n:
                        combined.append(item)
                        seen.add(item)
            
            ensemble_recs[user] = combined[:n]
        
        return ensemble_recs
    
    def combine(self,
               recs1: Dict,
               recs2: Dict,
               n: int = 50,
               **kwargs) -> Dict[int, List[int]]:
        """
        Método principal que ejecuta la estrategia seleccionada.
        """
        if self.strategy == 'weighted':
            return self.weighted_hybrid(recs1, recs2, n)
        elif self.strategy == 'rank_fusion':
            return self.rank_fusion(recs1, recs2, n)
        elif self.strategy == 'cascade':
            split_ratio = kwargs.get('split_ratio', 0.7)
            return self.cascade(recs1, recs2, n, split_ratio)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
