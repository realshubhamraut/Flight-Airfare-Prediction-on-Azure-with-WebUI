from azureml.core import Workspace, Environment, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

ws = Workspace.from_config(path="config.json")

myenv = Environment.from_conda_specification(
    name="flight_env", 
    file_path="conda_dependencies.yml"
)

model = Model(ws, name="flight_prices_model")


inference_config = InferenceConfig(
    entry_script="inference.py", 
    environment=myenv
)

deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=2)

service = Model.deploy(
    workspace=ws,
    name="flight-price-service",
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config
)
service.wait_for_deployment(show_output=True)
print("Scoring URI:", service.scoring_uri)