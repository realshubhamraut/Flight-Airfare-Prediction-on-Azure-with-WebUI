from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

app = FastAPI()

model = joblib.load("models/best_ml_model.pkl")
encoders = joblib.load("models/saved_encoders.pkl")

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
    return {"message": "Flight Price Prediction API is running"}

@app.post("/predict")
async def predict(flight_data: FlightData):
    try:
        # Parse the journey date and times
        journey_date = datetime.strptime(flight_data.Date_of_Journey, "%Y-%m-%d")
        # We still parse Dep_Time here (if needed elsewhere) but won't use it in input_data.
        dep_hour, dep_min = map(int, flight_data.Dep_Time.split(":"))
        arrival_hour, arrival_min = map(int, flight_data.Arrival_Time.split(":"))
        
        # Parse duration (e.g., "2h 30m")
        duration_hour = 0
        duration_min = 0
        duration = flight_data.Duration.lower().replace(" ", "")
        if "h" in duration:
            parts = duration.split("h")
            duration_hour = int(parts[0])
            if len(parts) > 1 and "m" in parts[1]:
                duration_min = int(parts[1].replace("m", ""))
        elif "m" in duration:
            duration_min = int(duration.replace("m", ""))
        
        # Prepare input data (note: removing Dep_hour and Dep_min to match training)
        input_data = {
            "Airline": flight_data.Airline.strip(),
            "Source": flight_data.Source.strip(),
            "Destination": flight_data.Destination.strip(),
            "Additional_Info": flight_data.Additional_Info.strip(),
            "Date": journey_date.day,
            "Month": journey_date.month,
            "Total_Stops": flight_data.Total_Stops,
            "Arrival_hour": arrival_hour,
            "Arrival_min": arrival_min,
            "Duration_hour": duration_hour,
            "Duration_min": duration_min
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical features using the loaded encoders.
        categorical_cols = ["Airline", "Source", "Destination", "Additional_Info"]
        for col in categorical_cols:
            if col in input_df.columns:
                try:
                    input_df[col] = encoders[col].transform(input_df[col])
                except Exception as e:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Error encoding {col}: {str(e)}"
                    )
        
        print("Input DataFrame for prediction:\n", input_df)
        
        # Predict and return result
        prediction = model.predict(input_df)[0]
        return {"predicted_price": round(prediction, 2)}
    
    except Exception as e:
        print("Error during prediction:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))