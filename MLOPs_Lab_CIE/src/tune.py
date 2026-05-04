import pandas as pd
import mlflow
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/training_data.csv")

X = df.drop("seats_filled_pct", axis=1)
y = df["seats_filled_pct"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [50, 150, 250],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 3, 5]
}

mlflow.set_experiment("ticketflick-seats-filled-pct")

with mlflow.start_run(run_name="tuning-ticketflick") as parent:

    model = RandomForestRegressor(random_state=42)

    grid = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring="neg_mean_absolute_error"
    )

    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    best_mae = -grid.best_score_

output = {
    "search_type": "grid",
    "n_folds": 3,
    "total_trials": len(grid.cv_results_["params"]),
    "best_params": best_params,
    "best_mae": best_mae,
    "best_cv_mae": best_mae,
    "parent_run_name": "tuning-ticketflick"
}

with open("results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=4)

print("Step 2 DONE")