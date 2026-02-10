### Flight Airfare Prediction on Azure with WebUI

<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://flask.palletsprojects.com/">
    <img src="https://img.shields.io/badge/-Flask-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask">
  </a>
  <a href="https://fastapi.tiangolo.com/">
    <img src="https://img.shields.io/badge/-FastAPI-009485?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  </a>
  <a href="https://spark.apache.org/docs/latest/api/python/">
    <img src="https://img.shields.io/badge/-PySpark-E25A1C?style=flat-square&logo=apachespark&logoColor=white" alt="PySpark">
  </a>
  <a href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  </a>
  <a href="https://pandas.pydata.org/">
    <img src="https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas">
  </a>
  <a href="https://numpy.org/">
    <img src="https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
  </a>
  <a href="https://www.docker.com/">
    <img src="https://img.shields.io/badge/-Docker-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker">
  </a>
  <a href="https://azure.microsoft.com/en-us/products/machine-learning">
    <img src="https://img.shields.io/badge/-Azure%20ML-0078D4?style=flat-square&logo=microsoft-azure&logoColor=white" alt="Azure ML">
  </a>
  <a href="https://azure.microsoft.com/">
    <img src="https://img.shields.io/badge/-Azure%20Web%20Apps-0078D4?style=flat-square&logo=microsoft-azure&logoColor=white" alt="Azure Web Apps">
  </a>
  <a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  </a>
  <a href="https://joblib.readthedocs.io/">
    <img src="https://img.shields.io/badge/-Joblib-FF9900?style=flat-square&logo=python&logoColor=white" alt="Joblib">
  </a>
  <a href="http://flight-ui-8339.eastus.azurecontainer.io:8501/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App">
  </a>
</div>

---

A production-grade flight airfare prediction platform built with PySpark ETL pipelines for distributed data processing, ensemble machine learning models with hyperparameter tuning, and RESTful APIs using Flask and FastAPI. The entire solution is containerized with Docker and deployed on Azure using Container Registry, Azure ML, and Container Instances.

<div style="display: flex; flex-wrap: wrap; gap: 8px; margin: 20px 0;">
  <a href="http://flight-ui-8339.eastus.azurecontainer.io:8501/">
    <img src="https://img.shields.io/badge/View_Live-WEB--UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
  <a href="http://flight-api-8256.eastus.azurecontainer.io:8000/docs">
    <img src="https://img.shields.io/badge/View_Live-Swagger_Docs-85EA2D?style=for-the-badge&logo=swagger&logoColor=black" alt="API Documentation">
  </a>
  <a href="http://flight-api-8256.eastus.azurecontainer.io:8000/">
    <img src="https://img.shields.io/badge/View_Live-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  </a>
  <a href="https://portal.azure.com/">
    <img src="https://img.shields.io/badge/View_Live-Azure_Portal-0078D4?style=for-the-badge&logo=microsoft-azure&logoColor=white" alt="Azure">
  </a>
</div>

| Service | URL | Description |
|:--------|:----|:------------|
|  WEB-UI | [flight-ui-8339.eastus.azurecontainer.io:8501](http://flight-ui-8339.eastus.azurecontainer.io:8501/) | Interactive web interface for flight price predictions with calendar view and 3D route maps |
| FastAPI Swagger | [flight-api-8256.eastus.azurecontainer.io:8000/docs](http://flight-api-8256.eastus.azurecontainer.io:8000/docs) | Interactive API documentation with request/response schemas and test endpoints |
| FastAPI Root | [flight-api-8256.eastus.azurecontainer.io:8000](http://flight-api-8256.eastus.azurecontainer.io:8000/) | REST API root endpoint with health check and service info `(ALSO YOU CAN USE THIS API endpoint IN YOUR APPS)`|

---

### Azure Deployment Screenshots

| Azure Container Registry | Azure Container Instances |
|:------------------------:|:-------------------------:|
| ![ACR Screenshot](docs/screenshots/1.png) | ![ACI Screenshot](docs/screenshots/2.png) |

| Azure Resource Group | Streamlit Web App |
|:--------------------:|:-----------------:|
| ![Resource Group](docs/screenshots/3.png) | ![Streamlit App](docs/screenshots/4.png) |

| UI |  Dockering |
|:--------------------:|:-----------------:|
| ![UI Screenshot](docs/screenshots/6.png) | ![Docker Screenshot](docs/screenshots/5.png) |


---

### Architecture Overview

```
+-------------------------------------------------------------------------+
|                    Flight Airfare Prediction Platform                   |
+-------------------------------------------------------------------------+
|                                                                         |
|  +-------------+    +------------------+    +-----------------------+   |
|  |  Raw Data   | -> |  PySpark ETL     | -> |  Feature Engineering  |   |
|  |  (CSV)      |    |  Pipeline        |    |  (Temporal, Route,    |   |
|  |             |    |                  |    |   Categorical)        |   |
|  +-------------+    +------------------+    +-----------+-----------+   |
|                                                         |               |
|  +------------------------------------------------------+----------+    |
|  |                   ML Training Pipeline               |          |    |
|  |  +-----------------+  +------------------------------v-------+  |    |
|  |  | Lasso Feature   |  | Ensemble Models (RandomForest,      |  |    |
|  |  | Selection       |->| GradientBoosting, ExtraTreesRegr.)  |  |    |
|  |  +-----------------+  +------------------------------+-------+  |    |
|  |                                                      |          |    |
|  |  +-----------------+  +------------------------------v-------+  |    |
|  |  | Hyperparameter  |->| Best Model Selection (Joblib)       |  |    |
|  |  | Tuning (RSCV)   |  |                                     |  |    |
|  |  +-----------------+  +----------------------------------+--+  |    |
|  +--------------------------------------------------------------+      |
|                                                                         |
|  +-------------------------------------------------------------------+  |
|  |                        REST API Layer                             |  |
|  |  +-------------------+    +-----------------------------------+   |  |
|  |  | Flask API (:5000) |    | FastAPI (:8000)                   |   |  |
|  |  | /predict          |    | /predict (Pydantic validation)   |   |  |
|  |  | /batch_predict    |    | /docs (Swagger UI)               |   |  |
|  |  +-------------------+    +-----------------------------------+   |  |
|  +-------------------------------------------------------------------+  |
|                                                                         |
|  +-------------------------------------------------------------------+  |
|  |           Streamlit Frontend (Interactive Web UI)                 |  |
|  |  - Price Prediction  - Calendar View  - 3D Route Maps             |  |
|  +-------------------------------------------------------------------+  |
|                                                                         |
|  +-------------------------------------------------------------------+  |
|  |                    Azure Deployment                               |  |
|  |  Docker -> ACR -> Azure Web Apps/ACI -> Azure ML                  |  |
|  |           (Production Monitoring via ModelDataCollector)          |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
```

---

### Technology Stack

| Category | Technologies |
|----------|-------------|
| Data Processing | PySpark, Pandas, NumPy |
| Machine Learning | scikit-learn (RandomForest, GradientBoosting, ExtraTreesRegressor, Lasso), Joblib |
| Feature Engineering | LabelEncoder, Temporal Pattern Extraction, Route Analytics |
| REST APIs | Flask, FastAPI, Pydantic |
| Frontend | Streamlit, Altair, PyDeck, Folium |
| Containerization | Docker |
| Cloud Deployment | Azure ML, Azure Container Registry, Azure Web Apps, Azure Container Instances |
| Monitoring | ModelDataCollector (Azure ML) |

---

### Key Features

#### PySpark ETL Pipeline
- Distributed Processing: PySpark-based ETL pipelines for scalable processing of large booking datasets
- Feature Extraction Workflows:
  - Temporal patterns (journey dates, departure/arrival hours)
  - Route analytics and stop counting
  - Categorical encoding with StringIndexer

#### Ensemble Machine Learning
- Models Trained: RandomForest, GradientBoosting, ExtraTreesRegressor
- Feature Selection: Lasso-based feature selection for optimal feature subset
- Hyperparameter Tuning: Systematic optimization using RandomizedSearchCV
- Model Serialization: Joblib for efficient model persistence

#### RESTful Prediction APIs
- Dual API Framework:
  - Flask API (`flask_app.py`): `/predict`, `/batch_predict`, `/model_info`, `/health`
  - FastAPI (`fastapi_app.py`): `/predict` with Swagger UI at `/docs`
- Performance: Sub-500ms prediction latency
- Validation: Pydantic models for request/response validation
- Transformations: LabelEncoder for categorical feature encoding

#### Interactive Streamlit Frontend
- Real-time flight price predictions
- Multi-airline comparison
- Calendar-based price forecasting
- 3D route visualization with PyDeck
- Interactive charts with Altair

#### Azure Cloud Deployment
- Containerization: Docker images for consistent deployment
- Registry: Azure Container Registry for image management
- Hosting: Azure Web Apps and Azure Container Instances
- Model Management: Azure ML for model lifecycle orchestration
- Monitoring: ModelDataCollector for production inference tracking

---

### Project Structure

```
Flight-Airfare-Prediction-on-Azure-with-WebUI/
├── etl/
│   ├── config/                    # ETL configuration
│   ├── extract/                   # Data extraction modules
│   ├── transform/                 # Data transformation modules
│   ├── load/                      # Data loading modules
│   ├── quality/                   # Data validation
│   └── run_pipeline.py            # Pipeline runner
├── models/
│   ├── best_ml_model.pkl          # Trained ensemble model
│   └── saved_encoders.pkl         # LabelEncoders for categorical features
├── data/
│   ├── train.csv                  # Training dataset
│   ├── test.csv                   # Test dataset
│   ├── bronze/                    # Raw data layer
│   ├── silver/                    # Cleaned data layer
│   └── gold/                      # Feature-engineered data layer
├── monitoring/
│   └── data_collector.py          # Production monitoring module
├── app.py                         # Streamlit frontend application
├── flask_app.py                   # Flask REST API
├── fastapi_app.py                 # FastAPI REST API
├── inference.py                   # Azure ML inference script
├── deploy_model.py                # Azure ML deployment configuration
├── register_model.py              # Azure ML model registration
├── EDA-model-building.ipynb       # Jupyter notebook for EDA and model training
├── Dockerfile.api                 # API Docker configuration
├── Dockerfile.streamlit           # Streamlit Docker configuration
├── requirements.txt               # Python dependencies
├── conda_dependencies.yml         # Conda environment specification
└── README.md
```

---

### Quick Start

#### Clone the Repository
```bash
git clone https://github.com/realshubhamraut/Flight-Airfare-Prediction-on-Azure-with-WebUI
cd Flight-Airfare-Prediction-on-Azure-with-WebUI
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Run PySpark ETL Pipeline (Optional)
```bash
python etl/run_pipeline.py
```

#### Train Models (Optional)
Run the Jupyter notebook `EDA-model-building.ipynb` to train models with hyperparameter tuning.

#### Start the APIs

Flask API:
```bash
python flask_app.py
# API available at http://localhost:5000
```

FastAPI:
```bash
uvicorn fastapi_app:app --reload
# API available at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

#### Launch Streamlit Frontend
```bash
streamlit run app.py
```

---

### Docker Deployment

#### Build and Run Locally
```bash
# Build API image
docker build -f Dockerfile.api -t flight-api .

# Build Streamlit image
docker build -f Dockerfile.streamlit -t flight-streamlit .

# Run containers
docker run -p 8000:8000 flight-api
docker run -p 8501:8501 flight-streamlit
```

#### Deploy to Azure Container Registry
```bash
# Login to Azure
az login

# Create ACR (if not exists)
az acr create --resource-group <RESOURCE_GROUP> --name <ACR_NAME> --sku Basic

# Login to ACR
az acr login --name <ACR_NAME>

# Tag and push images
docker tag flight-api <ACR_LOGIN_SERVER>/flight-api:latest
docker push <ACR_LOGIN_SERVER>/flight-api:latest

docker tag flight-streamlit <ACR_LOGIN_SERVER>/flight-streamlit:latest
docker push <ACR_LOGIN_SERVER>/flight-streamlit:latest
```

#### Deploy to Azure Container Instances
```bash
az container create --resource-group <RESOURCE_GROUP> \
  --name flight-api \
  --image <ACR_LOGIN_SERVER>/flight-api:latest \
  --cpu 1 --memory 1.5 \
  --ports 8000
```

---

### API Usage Examples

#### Flask API - Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Airline": "IndiGo",
    "Source": "Delhi",
    "Destination": "Kolkata",
    "Date_of_Journey": "2025-03-15",
    "Dep_Time": "08:30",
    "Arrival_Time": "11:00",
    "Duration": "2h 30m",
    "Total_Stops": 0,
    "Additional_Info": "No Info"
  }'
```

#### FastAPI - Prediction with Swagger UI
Visit `http://localhost:8000/docs` for interactive API documentation.

---

### Model Performance

| Model | R2 Score | MAE | RMSE |
|-------|----------|-----|------|
| RandomForest (Tuned) | 0.82+ | ~1200 | ~1800 |
| GradientBoosting (Tuned) | 0.80+ | ~1300 | ~1900 |
| ExtraTreesRegressor (Tuned) | 0.81+ | ~1250 | ~1850 |

Performance varies based on hyperparameter tuning results.

---

### Configuration

#### Azure ML Configuration
Place your `config.json` in the project root:
```json
{
    "subscription_id": "<SUBSCRIPTION_ID>",
    "resource_group": "<RESOURCE_GROUP>",
    "workspace_name": "<WORKSPACE_NAME>"
}
```

---

### License

This project is licensed under the MIT License.

---

### Author

Shubham Raut  
[GitHub](https://github.com/realshubhamraut)
