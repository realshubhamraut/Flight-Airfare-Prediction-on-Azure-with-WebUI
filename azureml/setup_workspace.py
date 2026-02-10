"""
Azure ML Workspace Setup
========================
Creates or connects to an Azure ML workspace for the Flight Price Prediction project.
Uses Azure ML SDK v2 (azure-ai-ml).

Usage:
    python azureml/setup_workspace.py
"""

import os
import json
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

# Configuration - override with environment variables or config.json
CONFIG = {
    "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
    "resource_group": os.getenv("AZURE_RESOURCE_GROUP", "flight-prediction-rg"),
    "workspace_name": os.getenv("AZURE_ML_WORKSPACE", "flight-ml-workspace"),
    "location": os.getenv("AZURE_LOCATION", "eastus")
}


def load_config():
    """Load configuration from config.json if exists."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            file_config = json.load(f)
            CONFIG.update({k: v for k, v in file_config.items() if v})
    return CONFIG


def get_credential():
    """Get Azure credential with fallback to interactive browser."""
    try:
        credential = DefaultAzureCredential()
        # Test the credential
        credential.get_token("https://management.azure.com/.default")
        print("✅ Using DefaultAzureCredential")
        return credential
    except Exception:
        print("⚠️  DefaultAzureCredential failed, using interactive browser login...")
        return InteractiveBrowserCredential()


def get_or_create_workspace(config: dict) -> MLClient:
    """Get existing workspace or create a new one."""
    
    credential = get_credential()
    
    # Check if subscription_id is set
    if not config.get("subscription_id"):
        print("\n❌ Error: subscription_id not set!")
        print("\nSet it via environment variable:")
        print("  export AZURE_SUBSCRIPTION_ID='your-subscription-id'")
        print("\nOr create config.json in project root:")
        print('  {"subscription_id": "your-subscription-id", "resource_group": "flight-prediction-rg", "workspace_name": "flight-ml-workspace"}')
        print("\nTo find your subscription ID:")
        print("  az account show --query id -o tsv")
        raise ValueError("subscription_id is required")
    
    # Try to connect to existing workspace
    try:
        ml_client = MLClient(
            credential=credential,
            subscription_id=config["subscription_id"],
            resource_group_name=config["resource_group"],
            workspace_name=config["workspace_name"]
        )
        ws = ml_client.workspaces.get(config["workspace_name"])
        print(f"✅ Connected to existing workspace: {ws.name}")
        return ml_client
        
    except Exception as e:
        print(f"⚠️  Workspace not found: {e}")
        print(f"🔧 Creating new workspace: {config['workspace_name']}...")
        
        # Create subscription-level client first
        ml_client = MLClient(
            credential=credential,
            subscription_id=config["subscription_id"],
            resource_group_name=config["resource_group"]
        )
        
        # Create workspace
        ws = Workspace(
            name=config["workspace_name"],
            location=config["location"],
            display_name="Flight Price Prediction ML",
            description="Azure ML workspace for flight airfare prediction model",
            tags={"project": "flight-prediction", "environment": "production"}
        )
        
        ws = ml_client.workspaces.begin_create(ws).result()
        print(f"✅ Created workspace: {ws.name}")
        
        # Now create a client connected to the workspace
        ml_client = MLClient(
            credential=credential,
            subscription_id=config["subscription_id"],
            resource_group_name=config["resource_group"],
            workspace_name=config["workspace_name"]
        )
        
        return ml_client


def save_workspace_config(config: dict):
    """Save workspace configuration for later use."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    
    workspace_config = {
        "subscription_id": config["subscription_id"],
        "resource_group": config["resource_group"],
        "workspace_name": config["workspace_name"]
    }
    
    with open(config_path, "w") as f:
        json.dump(workspace_config, f, indent=4)
    
    print(f"✅ Workspace config saved to: config.json")


def main():
    """Main setup function."""
    print("=" * 60)
    print("  Azure ML Workspace Setup - Flight Price Prediction")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    print(f"\n📋 Configuration:")
    print(f"   Subscription: {config.get('subscription_id', 'NOT SET')[:8]}..." if config.get('subscription_id') else "   Subscription: NOT SET")
    print(f"   Resource Group: {config['resource_group']}")
    print(f"   Workspace: {config['workspace_name']}")
    print(f"   Location: {config['location']}")
    
    # Get or create workspace
    try:
        ml_client = get_or_create_workspace(config)
        
        # Save config for future use
        if config.get("subscription_id"):
            save_workspace_config(config)
        
        # Print workspace details
        ws = ml_client.workspaces.get(config["workspace_name"])
        print(f"\n🎉 Workspace Ready!")
        print(f"   Name: {ws.name}")
        print(f"   Location: {ws.location}")
        print(f"   Studio URL: https://ml.azure.com/home?wsid={ws.id}")
        
        return ml_client
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        raise


if __name__ == "__main__":
    main()
