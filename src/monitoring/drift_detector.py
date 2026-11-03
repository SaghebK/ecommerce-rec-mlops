from scipy.stats import ks_2samp
import numpy as np

class DriftDetector:
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        
    def detect_feature_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> dict:
        drift_results = {}
        n_features = reference_data.shape[1] if len(reference_data.shape) > 1 else 1
        
        for i in range(n_features):
            ref_col = reference_data[:, i] if n_features > 1 else reference_data
            curr_col = current_data[:, i] if n_features > 1 else current_data
            stat, p_value = ks_2samp(ref_col, curr_col)
            drift_results[f"feature_{i}"] = {
                "p_value": float(p_value),
                "drift_detected": bool(p_value < self.threshold)
            }
        return drift_results
