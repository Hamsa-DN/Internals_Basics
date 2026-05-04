import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("data/training_data.csv")

X = df.drop("seats_filled_pct", axis=1)
y = df["seats_filled_pct"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MLflow Setup
# -----------------------------
mlflow.set_experiment("ticketflick-seats-filled-pct")

results = []

models = {
    "Ridge": Ridge(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

# -----------------------------
# Train Models
# -----------------------------
for name, model in models.items():
    with mlflow.start_run(run_name=name):

        # Train
        model.fit(X_train, y_train)

        # ⭐ IMPORTANT: Log model for Step 3
        mlflow.sklearn.log_model(model, "model")

        # Predict
        preds = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Log params & metrics
        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        # Tag
        mlflow.set_tag("experiment_type", "baseline_comparison")

        # Store results
        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse
        })

# -----------------------------
# Select Best Model
# -----------------------------
best = min(results, key=lambda x: x["mae"])

output = {
    "experiment_name": "ticketflick-seats-filled-pct",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "mae",
    "best_metric_value": best["mae"]
}

# -----------------------------
# Save JSON
# -----------------------------
os.makedirs("results", exist_ok=True)

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Step 1 DONE")