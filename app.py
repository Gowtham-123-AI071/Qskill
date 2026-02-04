# backend/src/api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backend.src.models.predict import get_model

app = FastAPI(title="Spam Mail Detector", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for production replace with frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    message: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
def health():
    return {"status":"ok", "service":"spam-mail-detector"}

@app.post("/predict-spam", response_model=PredictResponse)
def predict_spam(req: MessageRequest):
    try:
        model = get_model()
        res = model.predict(req.message)
        # res is dict for single
        return {"prediction": res["label"], "confidence": res["confidence"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
