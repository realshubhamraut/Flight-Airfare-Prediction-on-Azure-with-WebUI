from azureml.core import Workspace, Model

ws = Workspace.from_config()

model_blob_url = "https://<your-storage-account>.blob.core.windows.net/<container>/best_model.pkl"

model = Model.register(
    workspace=ws,
    model_path=model_blob_url,
    model_name="flight_price_model"
)

print("Model registered: ", model.name, model.id, model.version)