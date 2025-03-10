from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import numpy as np

# Use absolute imports
from services.weather_service import WeatherService
from services.astra_db_service import AstraDBService
from services.groq_slm_service import GroqSLMService
from models.lstm_model import LSTMModel
from utils.exceptions import ModelError, DataProcessingError

app = FastAPI()

class TemperaturePredictionRequest(BaseModel):
    data: List[float]

class TemperaturePredictionResponse(BaseModel):
    predictions: List[float]

@app.post("/temperature/predict", response_model=TemperaturePredictionResponse)
async def predict_temperature(request: TemperaturePredictionRequest):
    try:
        model = LSTMModel(input_shape=(len(request.data), 1))
        model.load_model('path_to_saved_model.h5')
        data = np.array(request.data).reshape((1, len(request.data), 1))
        predictions = model.predict(data)
        return TemperaturePredictionResponse(predictions=predictions.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/weather/current")
async def get_current_weather(location: str):
    try:
        weather_service = WeatherService()
        weather_data = await weather_service.get_current_weather(location)
        await weather_service.close()
        return weather_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/groq/context/{context_id}")
async def get_groq_context(context_id: str):
    try:
        groq_service = GroqSLMService()
        context_data = await groq_service.get_context(context_id)
        await groq_service.close()
        return context_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/astra/create_table")
async def create_astra_table(table_name: str, schema: str):
    try:
        astra_service = AstraDBService()
        astra_service.create_table(table_name, schema)
        astra_service.close()
        return {"message": "Table created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
