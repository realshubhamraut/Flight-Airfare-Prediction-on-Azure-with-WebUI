from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

app = FastAPI()


model = joblib.load("models/best_model.pkl")
encoders = joblib.load("models/encoders.pkl")

class FlightData(BaseModel):
    Airline: str
    Source: str
    Destination: str
    Date_of_Journey: str  # Format: "YYYY-MM-DD"
    Dep_Time: str         # Format: "HH:MM"
    Arrival_Time: str     # Format: "HH:MM"
    Duration: str         # Format: "2h 30m"
    Total_Stops: int
    Additional_Info: str


@app.get("/")
async def root():
    return {"message": "Welcome to the Flight Price Prediction API"}

@app.post("/predict")
async def predict(flight_data: FlightData):
    # Parse date
    journey_date = datetime.strptime(flight_data.Date_of_Journey, "%Y-%m-%d")
    
    # Parse times
    dep_hour, dep_min = map(int, flight_data.Dep_Time.split(":"))
    arrival_hour, arrival_min = map(int, flight_data.Arrival_Time.split(":"))
    
    # Parse duration
    duration_hour = 0
    duration_min = 0
    if 'h' in flight_data.Duration:
        duration_hour = int(flight_data.Duration.split('h')[0].strip())
        if 'm' in flight_data.Duration:
            duration_min = int(flight_data.Duration.split('h')[1].split('m')[0].strip())
    elif 'm' in flight_data.Duration:
        duration_min = int(flight_data.Duration.split('m')[0].strip())
    
    # Encode categorical features
    input_data = {
        "Airline": encoders["Airline"].transform([flight_data.Airline])[0],
        "Source": encoders["Source"].transform([flight_data.Source])[0],
        "Destination": encoders["Destination"].transform([flight_data.Destination])[0],
        "Additional_Info": encoders["Additional_Info"].transform([flight_data.Additional_Info])[0],
        "Date": journey_date.day,
        "Month": journey_date.month,
        "Stops": flight_data.Total_Stops,
        "Dep_hour": dep_hour,
        "Dep_min": dep_min,
        "Arrival_hour": arrival_hour,
        "Arrival_min": arrival_min,
        "Duration_hour": duration_hour,
        "Duration_min": duration_min
    }
    
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    
    return {"predicted_price": round(prediction, 2)}