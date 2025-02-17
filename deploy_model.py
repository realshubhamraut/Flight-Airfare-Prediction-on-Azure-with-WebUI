from azureml.core import Workspace, Environment, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

# Load workspace from the configuration file
ws = Workspace.from_config(path="config.json")

# Create an Azure ML environment from your conda_dependencies.yml.
myenv = Environment.from_conda_specification(
    name="flight_env", 
    file_path="conda_dependencies.yml"
)

# Retrieve your registered model (ensure you have already run register_model.py)
model = Model(ws, name="flight_price_model")


# Create the inference configuration with your scoring script
inference_config = InferenceConfig(
    entry_script="inference.py",  # Make sure inference.py is in the project root
    environment=myenv
)

# Define deployment configuration for ACI (Azure Container Instance)
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=2)

# Deploy the model as a web service in Azure ML
service = Model.deploy(
    workspace=ws,
    name="flight-price-service",
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config
)
service.wait_for_deployment(show_output=True)
print("Scoring URI:", service.scoring_uri)