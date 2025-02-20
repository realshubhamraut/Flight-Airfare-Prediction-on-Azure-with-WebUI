import os
import joblib
import pandas as pd
from azureml.monitoring import ModelDataCollector

def init():
    global model, encoders, input_dc, output_dc
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best_ml_model.pkl')
    encoders_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'saved_encoders.pkl')
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    input_dc = ModelDataCollector("flight_service", identifier="inputs")
    output_dc = ModelDataCollector("flight_service", identifier="outputs")

def run(raw_data):
    data = pd.read_json(raw_data)
    prediction = model.predict(data)[0]
    output_dc.collect(data)
    return {"predicted_price": float(prediction)}