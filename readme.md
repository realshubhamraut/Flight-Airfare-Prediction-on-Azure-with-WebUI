# Flight-Airfare-Prediction-on-Azure-with-WebUI

## Overview

This project is an end‑to‑end solution that predicts flight airfares using machine learning and serves the predictions via a modern web interface. The system includes data exploration, feature engineering, multiple model building, and deployment. A REST API built with FastAPI exposes prediction endpoints, while a Streamlit web application provides an interactive user interface. The entire solution is containerized using Docker and deployed on Azure (using Azure ML, Container Registry, and Web Apps).

## Features

- **Data Exploration & Model Building:**  
  - Extensive exploratory data analysis and feature engineering performed in Jupyter notebooks.
  - Multiple machine learning models developed and compared with the best model saved as `best_model.pkl` along with associated encoders in `encoders.pkl`.

- **REST API for Predictions:**  
  - FastAPI exposes endpoints (e.g., `/predict`) for real-time flight fare predictions.
  - Uses a robust data pre‑processing pipeline and model inference to return predictions in sub‑500 ms.

- **Interactive Web Interface:**  
  - Streamlit-based UI (`app.py`) allows users to enter flight details, view prediction results, and visualize flight routes on maps and interactive charts.
  - Calendar view for flight fare trends over a period with interactive Altair charts.

- **Containerization & Cloud Deployment:**  
  - Dockerfile defines the container image to run FastAPI (and optionally the Streamlit app).
  - Deployed to Azure using Azure Container Registry, Azure Machine Learning for model management, and Azure Web Apps for serving the Docker container 24/7.

## Prerequisites

- Python 3.9+
- Docker
- Azure CLI
- An active Azure Subscription with access to Azure Machine Learning, Container Registries, and Web Apps.
- Required Python packages as listed in `requirements.txt` & `conda_dependencies.yml`.

## Installation & Setup

1. **Clone the Repository**

   ```bash
   git clone <repository_url>
   cd Flight-Airfare-Prediction-on-Azure-with-WebUI