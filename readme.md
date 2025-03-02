# Flight-Airfare-Prediction-on-Azure-with-WebUI ✈️

<div style="display: flex; gap: 10px;">
  
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://fastapi.tiangolo.com/">
    <img src="https://img.shields.io/badge/-FastAPI-009485?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  </a>
    <a href="https://joblib.readthedocs.io/">
    <img src="https://img.shields.io/badge/-joblib-FF9900?style=flat-square&logo=python&logoColor=white" alt="Joblib">
  </a>
  <a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  </a>
  <a href="https://www.docker.com/">
    <img src="https://img.shields.io/badge/-Docker-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker">
  </a>
  <a href="https://azure.microsoft.com/">
    <img src="https://img.shields.io/badge/-Azure-0078D4?style=flat-square&logo=microsoft-azure&logoColor=white" alt="Azure">
  </a>
  <a href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit‑Learn">
  </a>

  <a href="https://flight-airfare-prediction-on-azure-with-webui.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App">
  </a>
</div>

---

**Flight-Airfare-Prediction-on-Azure-with-WebUI** is an end‑to‑end solution that predicts flight airfares using machine learning while offering an interactive web interface. The system includes advanced data exploration and feature engineering, multiple model building, and REST API endpoints via FastAPI. An intuitive Streamlit application allows users to input flight details and view predictions in real time. The entire solution is containerized with Docker and deployed on Azure for scalable, production‑grade operations.

[VIEW LIVE](https://flight-airfare-prediction-on-azure-with-webui.streamlit.app/)

[Access API](https://flight-webapp.azurewebsites.net/docs)

## Technology Stack

- **Frontend & Visualization:**  
  Streamlit, Altair, PyDeck, Folium

- **API & Backend:**  
  FastAPI, Python (joblib, Pandas)

- **Machine Learning:**  
  Feature engineering and modeling in Jupyter Notebooks, model serialization with joblib

- **Containerization & Deployment:**  
  Docker, Azure Container Registry, Azure Machine Learning, Azure Web Apps

## 🌟 Features

- **End-to-End ML Pipeline:**  
  - Performed comprehensive EDA and feature engineering to prepare flight data.  
  - Developed, validated, and selected multiple predictive models with high accuracy.

- **Real-Time Predictions via REST API:**  
  - Deployed a FastAPI service exposing endpoints (e.g., `/predict`) for sub‑500 ms response times.
  - Implements robust data pre‑processing and encoding for dynamic flight predictions.

- **Interactive User Interface:**  
  - Streamlit web app enables users to input flight details, view predictions, and explore dynamic visualizations such as calendar charts and 3D route maps.
  - Integrated interactive charts using Altair and spatial visualizations with Folium.

- **Cloud-Ready Deployment:**  
  - Containerized the application using Docker for consistency and scalability.  
  - Deployed on Azure using Azure Container Registry, Azure ML for model management, and Azure Web Apps for global availability 24/7.

## 🛠️ Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/realshubhamraut/Flight-Airfare-Prediction-on-Azure-with-WebUI
   cd Flight-Airfare-Prediction-on-Azure-with-WebUI

2. **Install Dependencies:**

- For local environment using pip:
  ```sh
  pip install -r requirements.txt
(Alternatively) Create a Conda environment using conda_dependencies.yml.


**Configure Azure Resources:**

- Download your Azure ML workspace configuration file (e.g., config.json) and place it in the project root.

- Ensure you have access to an Azure Subscription with resources like Container Registries, ML Workspaces, and Web Apps.


####  **Build & Push the Docker Image:**
```
docker build -t flight-fastapi-app .
docker tag flight-fastapi-app <ACR_LOGIN_SERVER>/flight-fastapi-app:latest
docker push <ACR_LOGIN_SERVER>/flight-fastapi-app:latest
```

Run the Local Web App:

_To run the FastAPI service:_

```
uvicorn fastapi_app:app --reload
```
Finally to launch the streamlit interface


```
streamlit run app.py
```