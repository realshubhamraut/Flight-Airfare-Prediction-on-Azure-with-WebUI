"""
Flask API for Flight Price Prediction
=====================================
RESTful prediction API with /predict endpoint providing sub-500ms latency
with LabelEncoder transformations.

Author: Flight Airfare Prediction Team
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import datetime
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load model and encoders at startup
try:
    model = joblib.load("models/best_ml_model.pkl")
    encoders = joblib.load("models/saved_encoders.pkl")
    logger.info("Model and encoders loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or encoders: {e}")
    model = None
    encoders = None


def parse_duration(duration_str):
    """Parse duration string (e.g., '2h 30m') into hours and minutes."""
    duration_str = duration_str.lower().replace(" ", "")
    duration_hour = 0
    duration_min = 0
    
    if "h" in duration_str:
        parts = duration_str.split("h")
        duration_hour = int(parts[0])
        if len(parts) > 1 and "m" in parts[1]:
            duration_min = int(parts[1].replace("m", ""))
    elif "m" in duration_str:
        duration_min = int(duration_str.replace("m", ""))
    
    return duration_hour, duration_min


def validate_request_data(data):
    """Validate incoming request data."""
    required_fields = [
        "Airline", "Source", "Destination", "Date_of_Journey",
        "Dep_Time", "Arrival_Time", "Duration", "Total_Stops", "Additional_Info"
    ]
    
    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    return True, None


@app.route("/", methods=["GET"])
def home():
    """Health check endpoint."""
    return jsonify({
        "message": "Flight Price Prediction API (Flask)",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)"
        }
    })


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "encoders_loaded": encoders is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict flight price based on input features.
    
    Expected JSON payload:
    {
        "Airline": "IndiGo",
        "Source": "Delhi",
        "Destination": "Kolkata",
        "Date_of_Journey": "2025-03-15",
        "Dep_Time": "08:30",
        "Arrival_Time": "11:00",
        "Duration": "2h 30m",
        "Total_Stops": 0,
        "Additional_Info": "No Info"
    }
    """
    start_time = time.time()
    
    # Check if model is loaded
    if model is None or encoders is None:
        return jsonify({
            "error": "Model not loaded. Please check server logs."
        }), 500
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate request data
        is_valid, error_msg = validate_request_data(data)
        if not is_valid:
            return jsonify({"error": error_msg}), 400
        
        # Parse journey date
        journey_date = datetime.strptime(data["Date_of_Journey"], "%Y-%m-%d")
        
        # Parse times
        arrival_hour, arrival_min = map(int, data["Arrival_Time"].split(":"))
        
        # Parse duration
        duration_hour, duration_min = parse_duration(data["Duration"])
        
        # Prepare input data for model
        input_data = {
            "Airline": data["Airline"].strip(),
            "Source": data["Source"].strip(),
            "Destination": data["Destination"].strip(),
            "Additional_Info": data["Additional_Info"].strip(),
            "Date": journey_date.day,
            "Month": journey_date.month,
            "Total_Stops": int(data["Total_Stops"]),
            "Arrival_hour": arrival_hour,
            "Arrival_min": arrival_min,
            "Duration_hour": duration_hour,
            "Duration_min": duration_min
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Apply LabelEncoder transformations
        categorical_cols = ["Airline", "Source", "Destination", "Additional_Info"]
        for col in categorical_cols:
            if col in input_df.columns and col in encoders:
                try:
                    input_df[col] = encoders[col].transform(input_df[col])
                except ValueError as e:
                    return jsonify({
                        "error": f"Invalid value for {col}: {input_data[col]}. "
                                f"Valid values: {list(encoders[col].classes_)}"
                    }), 400
        
        logger.info(f"Input DataFrame: {input_df.to_dict()}")
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        return jsonify({
            "predicted_price": round(float(prediction), 2),
            "currency": "INR",
            "latency_ms": round(latency_ms, 2),
            "input_summary": {
                "airline": data["Airline"],
                "route": f"{data['Source']} → {data['Destination']}",
                "date": data["Date_of_Journey"],
                "stops": data["Total_Stops"]
            }
        })
    
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        return jsonify({"error": f"Invalid data format: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Batch prediction endpoint for multiple flights.
    
    Expected JSON payload:
    {
        "flights": [
            {...flight_data_1...},
            {...flight_data_2...}
        ]
    }
    """
    start_time = time.time()
    
    if model is None or encoders is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if not data or "flights" not in data:
            return jsonify({"error": "No flights data provided"}), 400
        
        predictions = []
        for idx, flight in enumerate(data["flights"]):
            try:
                # Reuse single prediction logic
                journey_date = datetime.strptime(flight["Date_of_Journey"], "%Y-%m-%d")
                arrival_hour, arrival_min = map(int, flight["Arrival_Time"].split(":"))
                duration_hour, duration_min = parse_duration(flight["Duration"])
                
                input_data = {
                    "Airline": flight["Airline"].strip(),
                    "Source": flight["Source"].strip(),
                    "Destination": flight["Destination"].strip(),
                    "Additional_Info": flight["Additional_Info"].strip(),
                    "Date": journey_date.day,
                    "Month": journey_date.month,
                    "Total_Stops": int(flight["Total_Stops"]),
                    "Arrival_hour": arrival_hour,
                    "Arrival_min": arrival_min,
                    "Duration_hour": duration_hour,
                    "Duration_min": duration_min
                }
                
                input_df = pd.DataFrame([input_data])
                
                for col in ["Airline", "Source", "Destination", "Additional_Info"]:
                    if col in encoders:
                        input_df[col] = encoders[col].transform(input_df[col])
                
                pred = model.predict(input_df)[0]
                predictions.append({
                    "index": idx,
                    "predicted_price": round(float(pred), 2),
                    "route": f"{flight['Source']} → {flight['Destination']}"
                })
            except Exception as e:
                predictions.append({
                    "index": idx,
                    "error": str(e)
                })
        
        latency_ms = (time.time() - start_time) * 1000
        
        return jsonify({
            "predictions": predictions,
            "total_processed": len(predictions),
            "latency_ms": round(latency_ms, 2)
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/model_info", methods=["GET"])
def model_info():
    """Return information about the loaded model."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_type": type(model).__name__,
        "features_expected": [
            "Airline", "Source", "Destination", "Additional_Info",
            "Date", "Month", "Total_Stops", "Arrival_hour",
            "Arrival_min", "Duration_hour", "Duration_min"
        ],
        "encoders_available": list(encoders.keys()) if encoders else [],
        "airlines_supported": list(encoders["Airline"].classes_) if encoders and "Airline" in encoders else [],
        "sources_supported": list(encoders["Source"].classes_) if encoders and "Source" in encoders else [],
        "destinations_supported": list(encoders["Destination"].classes_) if encoders and "Destination" in encoders else []
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
