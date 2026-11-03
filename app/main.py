from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Hybrid E-commerce Recommendation API",
    version="1.0.0",
    description="Production API for recommendation serving with integrated MLOps monitoring"
)

class RecommendationRequest(BaseModel):
    user_id: str
    top_k: int = 10

class RecommendationResponse(BaseModel):
    user_id: str
    recommended_item_ids: list[str]

@app.get("/")
def read_root():
    return {"status": "online", "service": "Recommendation Engine API"}

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    dummy_recommendations = [f"item_{i}" for i in range(1, request.top_k + 1)]
    return RecommendationResponse(
        user_id=request.user_id,
        recommended_item_ids=dummy_recommendations
    )
