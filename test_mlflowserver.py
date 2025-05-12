from datetime import datetime

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from config import settings
from model import DelayModel

TRACKING_URI = "https://onboarding-service-592642508401.europe-southwest1.run.app"

data = pd.read_csv("data.csv", low_memory=False)
data = shuffle(data, random_state=111)

# Preprocess
model = DelayModel(settings["categorical_features"], settings["top_10_features"])
y = model.add_delay(data, threshold=15)
y = pd.DataFrame(y, columns=[settings["target_column"]])

x_train, x_test, y_train, y_test = train_test_split(
    data, y, test_size=0.33, random_state=42
)

# Train
model.fit(x_train, y_train)
# Predict
preds = model.predict(x_test)
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy}")

# Save model
# model.save_model("onboarding/challenge/delay_model.pkl")


# Set the tracking URI for the MLflow experiment
mlflow.set_tracking_uri(TRACKING_URI)

# Create an experiment if it doesn't exist
experiment_name = "Test_Experiment_Onboarding"
if not mlflow.get_experiment_by_name(name=experiment_name):
    mlflow.create_experiment(name=experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)

# Define the run name and tags for the experiment
run_name = datetime.now().strftime("%Y-%m-%d_%H:%M")
tags = {
    "env": "test",
    "data_date": "2025-05-12",
    "model_type": "XGBoost",
    "experiment_description": "Tutorial MLFlow experiment",
    # ... other tags ...
}

# Start the MLflow run
with mlflow.start_run(
    experiment_id=experiment.experiment_id, run_name=run_name, tags=tags
):
    # Log the hyperparameters used in the model
    mlflow.log_param("categorical_features", settings["categorical_features"])
    mlflow.log_param("top_10_features", settings["top_10_features"])
    mlflow.log_param("target_column", settings["target_column"])
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("model_name", "DelayModel")
    mlflow.log_param("model_version", "1.0.0")

    # Log the metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log model:
    signature = infer_signature(x_train, preds)
    mlflow.sklearn.log_model(
        model, "model", signature=signature, registered_model_name="DelayModel"
    )

    # explicit registration of model in model registry
    # model_uri = mlflow.sklearn.log_model(model, "model", signature=signature)
    # mlflow.register_model(model_uri=model_uri, name="DelayModel")
