import numpy as np

class HybridRanker:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha  # Weight for content-based score
        
    def rank(self, content_scores: np.ndarray, collab_scores: np.ndarray) -> np.ndarray:
        final_scores = (self.alpha * content_scores) + ((1 - self.alpha) * collab_scores)
        return np.argsort(final_scores)[::-1]
