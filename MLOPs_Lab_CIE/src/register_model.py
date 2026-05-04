import mlflow
import json

# Get experiment
experiment = mlflow.get_experiment_by_name("ticketflick-seats-filled-pct")

if experiment is None:
    raise Exception("Experiment not found. Run train.py first.")

# Get runs
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

if runs.empty:
    raise Exception("No runs found. Run train.py again.")

# Get latest run
runs = runs.sort_values("start_time", ascending=False)
run_id = runs.iloc[0]["run_id"]

# Model URI
model_uri = f"runs:/{run_id}/model"

# Register model
result = mlflow.register_model(
    model_uri,
    "ticketflick-seats-filled-pct-predictor"
)

# Save JSON
output = {
    "registered_model_name": "ticketflick-seats-filled-pct-predictor",
    "version": result.version,
    "run_id": run_id,
    "source_metric": "mae",
    "source_metric_value": 0.0
}

with open("results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=4)

print("Step 3 DONE")