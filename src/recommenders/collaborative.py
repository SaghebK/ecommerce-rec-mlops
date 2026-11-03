import numpy as np
from sklearn.decomposition import TruncatedSVD

class CollaborativeFilter:
    def __init__(self, n_components: int = 20):
        self.model = TruncatedSVD(n_components=n_components, random_state=42)
    
    def fit(self, user_item_matrix: np.ndarray):
        self.model.fit(user_item_matrix)
        
    def predict_user_embeddings(self, user_item_matrix: np.ndarray) -> np.ndarray:
        return self.model.transform(user_item_matrix)
