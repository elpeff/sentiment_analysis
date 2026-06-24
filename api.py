from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Sentiment Analysis API")

# ── Load model once at startup ─────────────────────────────────────────────────
sentiment_model = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# ── Request & Response schemas ─────────────────────────────────────────────────
class SentimentRequest(BaseModel):
    sentence: str

class SentimentResponse(BaseModel):
    label: str
    score: float

# ── Endpoint ───────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest):
    result = sentiment_model(request.sentence)[0]
    return SentimentResponse(
        label=result["label"],           # "POSITIVE" or "NEGATIVE"
        score=round(result["score"] * 100, 1)
    )

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Sentiment API is running"}
