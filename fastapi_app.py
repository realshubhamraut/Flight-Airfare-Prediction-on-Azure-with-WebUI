from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import monitoring module
try:
    from monitoring.data_collector import ModelDataCollector, DataDriftDetector
    MONITORING_ENABLED = True
    logger.info("Monitoring module loaded successfully")
except ImportError:
    MONITORING_ENABLED = False
    logger.warning("Monitoring module not available - running without monitoring")

app = FastAPI(
    title="Flight Price Prediction API",
    description="Production ML API with monitoring capabilities",
    version="2.0.0"
)

model = joblib.load("models/best_ml_model.pkl")
encoders = joblib.load("models/saved_encoders.pkl")

# Initialize monitoring
if MONITORING_ENABLED:
    input_collector = ModelDataCollector("flight_price", identifier="inputs")
    output_collector = ModelDataCollector("flight_price", identifier="outputs")
    drift_detector = DataDriftDetector()
    
    # Set baseline statistics for drift detection (from training data)
    drift_detector.set_baseline("Duration_hour", mean=8.5, std=5.2, min_val=0, max_val=48)
    drift_detector.set_baseline("Total_Stops", mean=1.0, std=0.8, min_val=0, max_val=4)
    drift_detector.set_baseline("Arrival_hour", mean=12, std=6, min_val=0, max_val=23)

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
    return {
        "message": "Flight Price Prediction API is running",
        "version": "2.0.0",
        "monitoring_enabled": MONITORING_ENABLED
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "encoders_loaded": encoders is not None
    }


@app.get("/metrics")
async def get_metrics():
    """Get monitoring metrics."""
    if not MONITORING_ENABLED:
        return {"error": "Monitoring not enabled"}
    
    input_metrics = input_collector.get_metrics()
    output_metrics = output_collector.get_metrics()
    
    return {
        "input_metrics": input_metrics,
        "output_metrics": output_metrics,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/drift")
async def check_drift():
    """Check for data drift on monitored features."""
    if not MONITORING_ENABLED:
        return {"error": "Monitoring not enabled"}
    
    drift_results = {}
    for feature in ["Duration_hour", "Total_Stops", "Arrival_hour"]:
        drift_results[feature] = drift_detector.check_drift(feature)
    
    return {
        "drift_analysis": drift_results,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
async def predict(flight_data: FlightData):
    start_time = time.time()
    request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    
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
        
        logger.info(f"[{request_id}] Processing prediction request")
        
        # Collect input data for monitoring
        if MONITORING_ENABLED:
            input_collector.collect(input_data, request_id)
            # Track features for drift detection
            drift_detector.add_observation("Duration_hour", duration_hour)
            drift_detector.add_observation("Total_Stops", flight_data.Total_Stops)
            drift_detector.add_observation("Arrival_hour", arrival_hour)
        
        # Predict
        prediction = model.predict(input_df)[0]
        predicted_price = round(float(prediction), 2)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Collect output for monitoring
        if MONITORING_ENABLED:
            output_collector.collect_prediction(
                input_data=input_data,
                prediction=predicted_price,
                latency_ms=latency_ms,
                request_id=request_id
            )
        
        logger.info(f"[{request_id}] Prediction: ₹{predicted_price} (latency: {latency_ms:.2f}ms)")
        
        return {
            "predicted_price": predicted_price,
            "request_id": request_id,
            "latency_ms": round(latency_ms, 2)
        }
    
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"[{request_id}] Error during prediction: {repr(e)}")
        
        # Log error in monitoring
        if MONITORING_ENABLED:
            output_collector.collect_prediction(
                input_data={},
                prediction=None,
                latency_ms=latency_ms,
                request_id=request_id
            )
        
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recent_predictions")
async def get_recent_predictions(n: int = 10):
    """Get recent predictions for debugging/monitoring."""
    if not MONITORING_ENABLED:
        return {"error": "Monitoring not enabled"}
    
    return {
        "recent_predictions": output_collector._collector.get_recent_predictions(n),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/flush_logs")
async def flush_logs():
    """Flush monitoring data to log files."""
    if not MONITORING_ENABLED:
        return {"error": "Monitoring not enabled"}
    
    input_file = input_collector.flush()
    output_file = output_collector.flush()
    
    return {
        "message": "Logs flushed successfully",
        "input_log": input_file,
        "output_log": output_file
    }