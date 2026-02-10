
## Deployment Status: ✅ Complete

### Deployed Resources

| Resource | Type | Location | Status |
|----------|------|----------|--------|
| `flight-prediction-rg` | Resource Group | eastus | Active |
| `flightpredacr7382` | Container Registry | eastus | Active |
| `flight-api` | Container Instance | eastus | Running |
| `flight-streamlit` | Container Instance | eastus | Running |

---

## Access URLs

### API (FastAPI)
- **URL**: http://flight-api-8256.eastus.azurecontainer.io:8000
- **Swagger UI**: http://flight-api-8256.eastus.azurecontainer.io:8000/docs
- **Health Check**: http://flight-api-8256.eastus.azurecontainer.io:8000/

### Streamlit UI
- **URL**: http://flight-ui-8339.eastus.azurecontainer.io:8501

---

## Resource Specifications

| Container | CPU | Memory | Image |
|-----------|-----|--------|-------|
| flight-api | 1 core | 1.5 GB | flightpredacr7382.azurecr.io/flight-api:latest |
| flight-streamlit | 1 core | 1.5 GB | flightpredacr7382.azurecr.io/flight-streamlit:latest |

---

## Cost Estimation (Azure Free Account)

| Resource | Est. Monthly Cost |
|----------|------------------|
| Container Registry (Basic) | ~$5/month |
| Container Instance (API) | ~$35/month (1 vCPU, 1.5GB) |
| Container Instance (UI) | ~$35/month (1 vCPU, 1.5GB) |
| **Total** | **~$75/month** |

---

## Useful Commands

### Check Container Status
```bash
az container list --resource-group flight-prediction-rg --output table
```

### View Container Logs
```bash
# API logs
az container logs --resource-group flight-prediction-rg --name flight-api

# Streamlit logs
az container logs --resource-group flight-prediction-rg --name flight-streamlit
```

### Restart Containers
```bash
az container restart --resource-group flight-prediction-rg --name flight-api
az container restart --resource-group flight-prediction-rg --name flight-streamlit
```

### Delete All Resources (to stop billing)
```bash
az group delete --name flight-prediction-rg --yes --no-wait
```

---

## Test the API

### Health Check
```bash
curl http://flight-api-8256.eastus.azurecontainer.io:8000/
```

### Predict Flight Price
```bash
curl -X POST http://flight-api-8256.eastus.azurecontainer.io:8000/predict \
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

---

## Files Created

| File | Purpose |
|------|---------|
| `Dockerfile.api` | API-specific Docker image |
| `Dockerfile.streamlit` | Streamlit-specific Docker image |
| `requirements-api.txt` | API dependencies |
| `requirements-streamlit.txt` | Streamlit dependencies |
| `.dockerignore` | Exclude files from Docker build |
| `deploy_azure.sh` | Automated deployment script |

---

## Redeployment

To redeploy after making changes:

```bash
# 1. Rebuild images for correct platform
docker buildx build --platform linux/amd64 -f Dockerfile.api -t flight-api:latest --load .
docker buildx build --platform linux/amd64 -f Dockerfile.streamlit -t flight-streamlit:latest --load .

# 2. Tag and push to ACR
ACR_NAME="flightpredacr7382"
az acr login --name $ACR_NAME
docker tag flight-api:latest $ACR_NAME.azurecr.io/flight-api:latest
docker tag flight-streamlit:latest $ACR_NAME.azurecr.io/flight-streamlit:latest
docker push $ACR_NAME.azurecr.io/flight-api:latest
docker push $ACR_NAME.azurecr.io/flight-streamlit:latest

# 3. Restart containers to pull new images
az container restart --resource-group flight-prediction-rg --name flight-api
az container restart --resource-group flight-prediction-rg --name flight-streamlit
```

---

*Deployed on: February 9, 2026*
