# Azure ML Integration Guide

This directory contains all Azure ML integration components for the Flight Airfare Prediction project.

## 📋 Prerequisites

1. **Azure Subscription** - Free tier works for testing
2. **Azure CLI** installed: `brew install azure-cli`
3. **Python 3.9+** with Azure ML SDK v2

## 🚀 Quick Start

### Step 1: Install Azure ML SDK v2

```bash
pip install azure-ai-ml azure-identity mlflow azureml-mlflow
```

### Step 2: Login to Azure

```bash
az login
```

### Step 3: Create Azure ML Workspace (one-time setup)

```bash
python azureml/setup_workspace.py
```

### Step 4: Train with MLflow Tracking

Run the training notebook or:

```bash
python azureml/train_with_mlflow.py
```

### Step 5: Register Model

```bash
python azureml/register_model.py
```

### Step 6: Deploy to Managed Endpoint

```bash
python azureml/deploy_endpoint.py
```

## 📁 Directory Structure

```
azureml/
├── README.md                 # This file
├── setup_workspace.py        # Create/connect to Azure ML workspace
├── train_with_mlflow.py      # Training script with experiment tracking
├── register_model.py         # Register model to Azure ML registry
├── deploy_endpoint.py        # Deploy managed online endpoint
├── score.py                  # Scoring script for inference
└── environment.yml           # Conda environment for Azure ML
```

## 🔧 Configuration

Create `config.json` in project root (gitignored):

```json
{
    "subscription_id": "your-subscription-id",
    "resource_group": "flight-prediction-rg",
    "workspace_name": "flight-ml-workspace"
}
```

Or set environment variables:
```bash
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="flight-prediction-rg"
export AZURE_ML_WORKSPACE="flight-ml-workspace"
```

## 🌟 Features

| Feature | Description |
|---------|-------------|
| **Experiment Tracking** | MLflow integration for metrics, parameters, artifacts |
| **Model Registry** | Versioned model storage with lineage |
| **Managed Endpoints** | Auto-scaling, blue-green deployments |
| **Data Drift Detection** | Built-in monitoring for production |
| **CI/CD Ready** | GitHub Actions integration available |

## 💰 Cost Estimation

| Resource | Est. Monthly Cost |
|----------|------------------|
| Azure ML Workspace | Free (storage only) |
| Managed Endpoint (Standard_DS2_v2) | ~$0.20/hour when running |
| Model Registry | Free (included) |

## 📊 MLflow Tracking

View experiments in Azure ML Studio:
```
https://ml.azure.com
```

Or locally:
```bash
mlflow ui --port 5000
```
