from sentence_transformers import SentenceTransformer
import numpy as np

class ContentBasedRecommender:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def extract_features(self, text_descriptions: list[str]) -> np.ndarray:
        embeddings = self.model.encode(text_descriptions, show_progress_bar=False)
        return embeddings
