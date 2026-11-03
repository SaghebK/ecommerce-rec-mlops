import mlflow
import mlflow.pyfunc
from sklearn.metrics import ndcg_score
import numpy as np

def validate_and_promote_model(challenger_model_uri: str, champion_model_uri: str, X_val: np.ndarray, y_val: np.ndarray, threshold_improvement: float = 0.0) -> bool:
    print(f"Loading Champion model from: {champion_model_uri}")
    champion_model = mlflow.pyfunc.load_model(champion_model_uri)
    
    print(f"Loading Challenger model from: {challenger_model_uri}")
    challenger_model = mlflow.pyfunc.load_model(challenger_model_uri)
    
    challenger_preds = challenger_model.predict(X_val)
    champion_preds = champion_model.predict(X_val)
    
    challenger_ndcg = ndcg_score([y_val], [challenger_preds], k=10)
    champion_ndcg = ndcg_score([y_val], [champion_preds], k=10)
    
    print(f"Champion NDCG@10: {champion_ndcg:.4f}")
    print(f"Challenger NDCG@10: {challenger_ndcg:.4f}")
    
    if challenger_ndcg >= champion_ndcg + threshold_improvement:
        print("Challenger model outperformed Champion. Promoting to Production stage.")
        return True
    else:
        print("Challenger model failed to improve performance. Retaining current Champion.")
        return False
