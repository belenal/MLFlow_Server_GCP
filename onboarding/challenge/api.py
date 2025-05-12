import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from onboarding.challenge.config import settings
from onboarding.challenge.model import DelayModel


# Request and Response Models
class DataFrameRequest(BaseModel):
    data: list[dict[str, str]]


class PredictionResponse(BaseModel):
    predictions: list[int]


# Initialize FastAPI app
app = FastAPI()

model = DelayModel(settings["categorical_features"], settings["top_10_features"])
model.load_model(
    "onboarding/challenge/delay_model.pkl"
)  # Ensure the model file is in the same directory


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(request: DataFrameRequest) -> dict:
    input_data = pd.DataFrame(request.data)
    predictions = model.predict(input_data)
    return PredictionResponse(predictions=predictions).model_dump()
