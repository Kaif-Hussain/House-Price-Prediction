# backend/main.py — very top of the file
from dotenv import load_dotenv
from pathlib import Path

from backend.chatbot import ask_chatbot

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI
from fastapi import FastAPI
from pydantic import BaseModel

from backend.model_utils import meta, predict

app = FastAPI()


class FeatureData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


class ChatRequest(BaseModel):
    message: str


@app.post("/predict")
def predict_price(features: FeatureData):
    price = predict(features.model_dump())
    return {"predicted_price": price, "unit": "$100,000"}


@app.get("/meta")
def get_meta():
    return meta

@app.post("/chat")
def chat(request: ChatRequest):
    return {"response": ask_chatbot(request.message)}