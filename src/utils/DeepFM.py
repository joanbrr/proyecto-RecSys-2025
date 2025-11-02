import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional
import pickle
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM as DeepFMBase


class DeepFM:
    def __init__(self, 
                embedding_dim: int = 16,
                dnn_hidden_units: Tuple[int] = (128, 64),
                dnn_dropout: float = 0.2,
                learning_rate: float = 0.001,
                epochs: int = 10,
                batch_size: int = 2048,
                device: str = 'cpu'):
        """
        DeepFM wrapper using deepctr-torch.
        
        Args:
            embedding_dim: Dimension of embeddings
            dnn_hidden_units: Tuple of hidden layer sizes for deep component
            dnn_dropout: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: 'cpu' or 'cuda'
        """
        self.embedding_dim = embedding_dim
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_dropout = dnn_dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        
        self.model = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.n_users = 0
        self.n_items = 0
        self.linear_feature_columns = []
        self.dnn_feature_columns = []
        self.feature_names = []
        
    def _prepare_features(self, df: pd.DataFrame, fit_encoders: bool = True) -> Dict:
        """Prepare and encode features for the model."""
        df = df.copy()
        
        if fit_encoders:
            df['user_id'] = self.user_encoder.fit_transform(df['user_id'])
            df['item_id'] = self.item_encoder.fit_transform(df['item_id'])
            self.n_users = len(self.user_encoder.classes_)
            self.n_items = len(self.item_encoder.classes_)
        else:
            # Handle unknown values by filtering or mapping to known values
            user_mask = df['user_id'].isin(self.user_encoder.classes_)
            item_mask = df['item_id'].isin(self.item_encoder.classes_)
            df = df[user_mask & item_mask].copy()
            
            df['user_id'] = self.user_encoder.transform(df['user_id'])
            df['item_id'] = self.item_encoder.transform(df['item_id'])
        
        X = {name: df[name].values for name in self.feature_names}
        return X
    
    def _build_model(self):
        """Build the DeepFM model using deepctr-torch."""
        user_feat = SparseFeat("user_id", vocabulary_size=self.n_users, embedding_dim=self.embedding_dim)
        item_feat = SparseFeat("item_id", vocabulary_size=self.n_items, embedding_dim=self.embedding_dim)
        
        self.linear_feature_columns = [user_feat, item_feat]
        self.dnn_feature_columns = [user_feat, item_feat]
        self.feature_names = get_feature_names(self.linear_feature_columns + self.dnn_feature_columns)
        
        model = DeepFMBase(
            self.linear_feature_columns,
            self.dnn_feature_columns,
            task='binary',
            dnn_hidden_units=self.dnn_hidden_units,
            dnn_dropout=self.dnn_dropout,
            device=self.device
        )
        
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["auc"])
        return model
    
    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Train the DeepFM model."""
        train_df_copy = train_df.copy()
        train_df_copy['user_id'] = self.user_encoder.fit_transform(train_df_copy['user_id'])
        train_df_copy['item_id'] = self.item_encoder.fit_transform(train_df_copy['item_id'])
        self.n_users = len(self.user_encoder.classes_)
        self.n_items = len(self.item_encoder.classes_)
        
        self.model = self._build_model()
        
        X_train = {name: train_df_copy[name].values for name in self.feature_names}
        y_train = (train_df['rating'] >= 4).astype(int).values
        
        validation_data = None
        if val_df is not None:
            val_df_copy = val_df.copy()
            user_mask = val_df_copy['user_id'].isin(self.user_encoder.classes_)
            item_mask = val_df_copy['item_id'].isin(self.item_encoder.classes_)
            val_df_copy = val_df_copy[user_mask & item_mask].copy()
            
            val_df_copy['user_id'] = self.user_encoder.transform(val_df_copy['user_id'])
            val_df_copy['item_id'] = self.item_encoder.transform(val_df_copy['item_id'])
            
            X_val = {name: val_df_copy[name].values for name in self.feature_names}
            y_val = (val_df_copy['rating'] >= 4).astype(int).values
            validation_data = (X_val, y_val)
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=2,
            validation_data=validation_data
        )
        
        return history
    
    def predict(self, X_test: Dict) -> np.ndarray:
        """Predict ratings for test data."""
        return self.model.predict(X_test, batch_size=self.batch_size)
    
    def get_top_n(self, test_df: pd.DataFrame, n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Get top-N recommendations for users in test set."""
        test_df_filtered = test_df.copy()
        user_mask = test_df_filtered['user_id'].isin(self.user_encoder.classes_)
        item_mask = test_df_filtered['item_id'].isin(self.item_encoder.classes_)
        test_df_filtered = test_df_filtered[user_mask & item_mask]
        
        test_df_filtered['user_id_enc'] = self.user_encoder.transform(test_df_filtered['user_id'])
        test_df_filtered['item_id_enc'] = self.item_encoder.transform(test_df_filtered['item_id'])
        
        X_test = {'user_id': test_df_filtered['user_id_enc'].values, 'item_id': test_df_filtered['item_id_enc'].values}
        y_scores = self.model.predict(X_test, batch_size=self.batch_size).reshape(-1)
        
        test_df_filtered['score'] = y_scores
        
        recommendations = {}
        for user_id, group in test_df_filtered.groupby('user_id'):
            top_items = group.nlargest(n, 'score')[['item_id', 'score']].values.tolist()
            recommendations[str(user_id)] = [(str(item), score) for item, score in top_items]
        
        return recommendations
    
    def save_model(self, filepath: str):
        """Save the trained model and encoders."""
        torch.save(self.model.state_dict(), f"{filepath}_model.pth")
        
        model_data = {
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'embedding_dim': self.embedding_dim,
            'dnn_hidden_units': self.dnn_hidden_units,
            'dnn_dropout': self.dnn_dropout,
            'linear_feature_columns': self.linear_feature_columns,
            'dnn_feature_columns': self.dnn_feature_columns,
            'feature_names': self.feature_names
        }
        
        with open(f"{filepath}_config.pkl", 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load a trained model and encoders."""
        with open(f"{filepath}_config.pkl", 'rb') as f:
            model_data = pickle.load(f)
        
        self.user_encoder = model_data['user_encoder']
        self.item_encoder = model_data['item_encoder']
        self.n_users = model_data['n_users']
        self.n_items = model_data['n_items']
        self.embedding_dim = model_data['embedding_dim']
        self.dnn_hidden_units = model_data['dnn_hidden_units']
        self.dnn_dropout = model_data['dnn_dropout']
        self.linear_feature_columns = model_data['linear_feature_columns']
        self.dnn_feature_columns = model_data['dnn_feature_columns']
        self.feature_names = model_data['feature_names']
        
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(f"{filepath}_model.pth"))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))
