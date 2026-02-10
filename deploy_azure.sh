#!/bin/bash
# =============================================================================
# Azure Deployment Script for Flight Airfare Prediction
# =============================================================================
# Prerequisites:
# - Azure CLI installed (brew install azure-cli)
# - Docker running
# - Logged into Azure (az login)
#
# Usage:
#   chmod +x deploy_azure.sh
#   ./deploy_azure.sh
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Configuration Variables (Modify these as needed)
# -----------------------------------------------------------------------------
RESOURCE_GROUP="flight-prediction-rg"
LOCATION="eastus"  # Use a location with free tier availability
ACR_NAME="flightpredictionacr$(date +%s | tail -c 5)"  # Must be globally unique
API_APP_NAME="flight-price-api"
STREAMLIT_APP_NAME="flight-price-ui"
APP_SERVICE_PLAN="flight-prediction-plan"
SKU="B1"  # Basic tier - cheapest paid option (~$13/month), or use F1 for free

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=================================================${NC}"
echo -e "${GREEN}  Flight Airfare Prediction - Azure Deployment   ${NC}"
echo -e "${GREEN}=================================================${NC}"

# -----------------------------------------------------------------------------
# Step 1: Check Prerequisites
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Step 1: Checking prerequisites...${NC}"

if ! command -v az &> /dev/null; then
    echo -e "${RED}Azure CLI is not installed. Install with: brew install azure-cli${NC}"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed or not running.${NC}"
    exit 1
fi

echo -e "${GREEN}OK: Prerequisites check passed${NC}"

# -----------------------------------------------------------------------------
# Step 2: Azure Login Check
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Step 2: Checking Azure login status...${NC}"

if ! az account show &> /dev/null; then
    echo "Not logged in. Starting Azure login..."
    az login
fi

SUBSCRIPTION=$(az account show --query name -o tsv)
echo -e "${GREEN}OK: Logged into Azure subscription: ${SUBSCRIPTION}${NC}"

# -----------------------------------------------------------------------------
# Step 3: Create Resource Group
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Step 3: Creating Resource Group...${NC}"

az group create \
    --name $RESOURCE_GROUP \
    --location $LOCATION \
    --output none

echo -e "${GREEN}OK: Resource Group '$RESOURCE_GROUP' created in $LOCATION${NC}"

# -----------------------------------------------------------------------------
# Step 4: Create Azure Container Registry
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Step 4: Creating Azure Container Registry...${NC}"

az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic \
    --admin-enabled true \
    --output none

ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer -o tsv)
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)

echo -e "${GREEN}OK: ACR '$ACR_NAME' created${NC}"
echo -e "  Login Server: $ACR_LOGIN_SERVER"

# -----------------------------------------------------------------------------
# Step 5: Build Docker Images
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Step 5: Building Docker images...${NC}"

# Build API image
echo "Building API image..."
docker build -f Dockerfile.api -t flight-api:latest .

# Build Streamlit image
echo "Building Streamlit image..."
docker build -f Dockerfile.streamlit -t flight-streamlit:latest .

echo -e "${GREEN}OK: Docker images built successfully${NC}"

# -----------------------------------------------------------------------------
# Step 6: Tag and Push Images to ACR
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Step 6: Pushing images to Azure Container Registry...${NC}"

# Login to ACR
az acr login --name $ACR_NAME

# Tag images
docker tag flight-api:latest $ACR_LOGIN_SERVER/flight-api:latest
docker tag flight-streamlit:latest $ACR_LOGIN_SERVER/flight-streamlit:latest

# Push images
docker push $ACR_LOGIN_SERVER/flight-api:latest
docker push $ACR_LOGIN_SERVER/flight-streamlit:latest

echo -e "${GREEN}OK: Images pushed to ACR${NC}"

# -----------------------------------------------------------------------------
# Step 7: Create App Service Plan
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Step 7: Creating App Service Plan...${NC}"

az appservice plan create \
    --name $APP_SERVICE_PLAN \
    --resource-group $RESOURCE_GROUP \
    --is-linux \
    --sku $SKU \
    --output none

echo -e "${GREEN}OK: App Service Plan created (SKU: $SKU)${NC}"

# -----------------------------------------------------------------------------
# Step 8: Create and Deploy API Web App
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Step 8: Deploying API Web App...${NC}"

az webapp create \
    --resource-group $RESOURCE_GROUP \
    --plan $APP_SERVICE_PLAN \
    --name $API_APP_NAME \
    --deployment-container-image-name $ACR_LOGIN_SERVER/flight-api:latest \
    --output none

# Configure container settings
az webapp config appsettings set \
    --resource-group $RESOURCE_GROUP \
    --name $API_APP_NAME \
    --settings WEBSITES_PORT=8000 \
    --output none

# Set ACR credentials
az webapp config container set \
    --resource-group $RESOURCE_GROUP \
    --name $API_APP_NAME \
    --docker-custom-image-name $ACR_LOGIN_SERVER/flight-api:latest \
    --docker-registry-server-url https://$ACR_LOGIN_SERVER \
    --docker-registry-server-user $ACR_USERNAME \
    --docker-registry-server-password $ACR_PASSWORD \
    --output none

API_URL="https://${API_APP_NAME}.azurewebsites.net"
echo -e "${GREEN}OK: API deployed at: $API_URL${NC}"

# -----------------------------------------------------------------------------
# Step 9: Create and Deploy Streamlit Web App
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Step 9: Deploying Streamlit Web App...${NC}"

az webapp create \
    --resource-group $RESOURCE_GROUP \
    --plan $APP_SERVICE_PLAN \
    --name $STREAMLIT_APP_NAME \
    --deployment-container-image-name $ACR_LOGIN_SERVER/flight-streamlit:latest \
    --output none

# Configure container settings
az webapp config appsettings set \
    --resource-group $RESOURCE_GROUP \
    --name $STREAMLIT_APP_NAME \
    --settings WEBSITES_PORT=8501 \
    --output none

# Set ACR credentials
az webapp config container set \
    --resource-group $RESOURCE_GROUP \
    --name $STREAMLIT_APP_NAME \
    --docker-custom-image-name $ACR_LOGIN_SERVER/flight-streamlit:latest \
    --docker-registry-server-url https://$ACR_LOGIN_SERVER \
    --docker-registry-server-user $ACR_USERNAME \
    --docker-registry-server-password $ACR_PASSWORD \
    --output none

STREAMLIT_URL="https://${STREAMLIT_APP_NAME}.azurewebsites.net"
echo -e "${GREEN}OK: Streamlit UI deployed at: $STREAMLIT_URL${NC}"

# -----------------------------------------------------------------------------
# Step 10: Enable Continuous Deployment (Optional)
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Step 10: Enabling continuous deployment...${NC}"

az webapp deployment container config \
    --resource-group $RESOURCE_GROUP \
    --name $API_APP_NAME \
    --enable-cd true \
    --output none

az webapp deployment container config \
    --resource-group $RESOURCE_GROUP \
    --name $STREAMLIT_APP_NAME \
    --enable-cd true \
    --output none

echo -e "${GREEN}OK: Continuous deployment enabled${NC}"

# -----------------------------------------------------------------------------
# Deployment Summary
# -----------------------------------------------------------------------------
echo -e "\n${GREEN}=================================================${NC}"
echo -e "${GREEN}  Deployment Complete! 🎉                        ${NC}"
echo -e "${GREEN}=================================================${NC}"
echo ""
echo -e "Resource Group:    ${YELLOW}$RESOURCE_GROUP${NC}"
echo -e "Container Registry: ${YELLOW}$ACR_NAME${NC}"
echo ""
echo -e "${GREEN} API Endpoint:${NC}"
echo -e "   $API_URL"
echo -e "   $API_URL/docs (Swagger UI)"
echo ""
echo -e "${GREEN}🖥️  Streamlit UI:${NC}"
echo -e "   $STREAMLIT_URL"
echo ""
echo -e "${YELLOW}Note: It may take 2-3 minutes for the apps to start.${NC}"
echo ""
echo -e "To check logs:"
echo -e "  az webapp log tail --name $API_APP_NAME --resource-group $RESOURCE_GROUP"
echo -e "  az webapp log tail --name $STREAMLIT_APP_NAME --resource-group $RESOURCE_GROUP"
echo ""
echo -e "To delete all resources:"
echo -e "  az group delete --name $RESOURCE_GROUP --yes --no-wait"
