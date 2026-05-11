"""
MLOps Certification Course — Capstone Project
=============================================
Salary Prediction using Years of Experience

Pipeline:
  1. Data loading & preprocessing
  2. Train 3 regression models (Linear Regression, Random Forest, Decision Tree)
  3. Track experiments with MLflow
  4. Deploy best model via MLflow Model Registry
  5. Monitor production data with WhyLogs (WhyLabs open-source SDK)
"""

import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import whylogs as why
from whylogs.core.schema import DatasetSchema

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────
BASE_DIR   = r"E:\2026\mlops\Capstone - Copy"
DATA_PATH  = os.path.join(BASE_DIR, "2687_capstone_project_dataset_v1_vv6_ahjq7xz.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MLFLOW_DIR = os.path.join(BASE_DIR, "mlruns")
os.makedirs(MLFLOW_DIR, exist_ok=True)
mlflow.set_tracking_uri(Path(MLFLOW_DIR).as_uri())  # → file:///E:/2026/mlops/... (cross-platform safe)

# ─────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Data Loading & Preprocessing")
print("=" * 60)

df = pd.read_csv(DATA_PATH, index_col=0)
print(f"\nDataset shape : {df.shape}")
print(f"Columns       : {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nDescriptive statistics:\n{df.describe()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Features and target
X = df[["YearsExperience"]].values
y = df["Salary"].values

# Train / test split  (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size : {len(X_train)} | Test size : {len(X_test)}")

# Feature scaling (used by Linear Regression; kept identical for all)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────
# 2. HELPER: METRICS + MLFLOW RUN
# ─────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def train_and_track(name, model, X_tr, X_te, y_tr, y_te, params, experiment_name):
    """Train model, log to MLflow, return metrics and run_id."""
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        metrics = compute_metrics(y_te, y_pred)

        # Log params & metrics
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Infer signature and log model
        signature = infer_signature(X_tr, model.predict(X_tr))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=f"salary_{name.replace(' ', '_').lower()}"
        )

        run_id = run.info.run_id
        print(f"\n  [{name}]  R2={metrics['R2']:.4f}  RMSE={metrics['RMSE']:,.0f}  MAE={metrics['MAE']:,.0f}")
        print(f"  MLflow run_id: {run_id}")
        return metrics, y_pred, run_id, model


# ─────────────────────────────────────────────────────────
# 3. MODEL TRAINING + EXPERIMENT TRACKING
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Model Training & Experiment Tracking (MLflow)")
print("=" * 60)

EXPERIMENT = "Salary_Prediction_Capstone"
results = {}

# ── 3a. Linear Regression ──────────────────────────────
lr_params = {"model_type": "LinearRegression", "fit_intercept": True}
lr_model   = LinearRegression(fit_intercept=True)
lr_metrics, lr_pred, lr_rid, lr_m = train_and_track(
    "Linear Regression", lr_model,
    X_train_sc, X_test_sc, y_train, y_test,
    lr_params, EXPERIMENT
)
results["Linear Regression"] = {"metrics": lr_metrics, "pred": lr_pred, "run_id": lr_rid}

# ── 3b. Random Forest Regressor ───────────────────────
rf_params = {"model_type": "RandomForestRegressor", "n_estimators": 100, "max_depth": 5, "random_state": 42}
rf_model   = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_metrics, rf_pred, rf_rid, rf_m = train_and_track(
    "Random Forest", rf_model,
    X_train, X_test, y_train, y_test,   # tree-based: no scaling needed
    rf_params, EXPERIMENT
)
results["Random Forest"] = {"metrics": rf_metrics, "pred": rf_pred, "run_id": rf_rid}

# ── 3c. Decision Tree Regressor ───────────────────────
dt_params = {"model_type": "DecisionTreeRegressor", "max_depth": 5, "random_state": 42}
dt_model   = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_metrics, dt_pred, dt_rid, dt_m = train_and_track(
    "Decision Tree", dt_model,
    X_train, X_test, y_train, y_test,
    dt_params, EXPERIMENT
)
results["Decision Tree"] = {"metrics": dt_metrics, "pred": dt_pred, "run_id": dt_rid}

# ─────────────────────────────────────────────────────────
# 4. MODEL COMPARISON
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Model Comparison")
print("=" * 60)

comparison_df = pd.DataFrame(
    {name: v["metrics"] for name, v in results.items()}
).T.reset_index().rename(columns={"index": "Model"})
comparison_df = comparison_df[["Model", "R2", "RMSE", "MAE", "MSE"]]
comparison_df["R2"]   = comparison_df["R2"].round(4)
comparison_df["RMSE"] = comparison_df["RMSE"].round(2)
comparison_df["MAE"]  = comparison_df["MAE"].round(2)
comparison_df["MSE"]  = comparison_df["MSE"].round(2)

print(f"\n{comparison_df.to_string(index=False)}")

# Best model by R2
best_name = comparison_df.loc[comparison_df["R2"].idxmax(), "Model"]
best_r2   = comparison_df.loc[comparison_df["R2"].idxmax(), "R2"]
print(f"\n>>> Best model: {best_name}  (R2 = {best_r2})")

# Save comparison CSV
comp_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
comparison_df.to_csv(comp_path, index=False)
print(f"    Comparison saved → {comp_path}")

# ─────────────────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────────────────
X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
X_range_sc = scaler.transform(X_range)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Salary Prediction — Model Comparison", fontsize=14, fontweight="bold")

model_configs = [
    ("Linear Regression", lr_m,  X_range_sc, lr_pred,  "steelblue"),
    ("Random Forest",     rf_m,  X_range,    rf_pred,  "forestgreen"),
    ("Decision Tree",     dt_m,  X_range,    dt_pred,  "darkorange"),
]

for ax, (name, mdl, X_rng, y_p, color) in zip(axes, model_configs):
    ax.scatter(X_test, y_test, color="gray", label="Actual", zorder=3, s=60)
    ax.scatter(X_test, y_p,    color=color,  label="Predicted", marker="x", zorder=4, s=80)
    ax.plot(X_range, mdl.predict(X_rng), color=color, linewidth=2, label="Fit line")
    r2 = results[name]["metrics"]["R2"]
    ax.set_title(f"{name}\nR² = {r2:.4f}")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary ($)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "model_comparison_plot.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Plot saved       → {plot_path}")

# Bar chart: R2 comparison
fig2, ax2 = plt.subplots(figsize=(8, 5))
colors = ["steelblue", "forestgreen", "darkorange"]
bars = ax2.bar(comparison_df["Model"], comparison_df["R2"], color=colors, width=0.5)
ax2.set_ylim(0, 1.05)
ax2.set_ylabel("R² Score")
ax2.set_title("R² Score Comparison Across Models")
for bar, val in zip(bars, comparison_df["R2"]):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.01, f"{val:.4f}",
             ha="center", va="bottom", fontweight="bold")
ax2.grid(axis="y", alpha=0.3)
plt.tight_layout()
bar_path = os.path.join(OUTPUT_DIR, "r2_comparison.png")
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    R2 bar chart     → {bar_path}")

# ─────────────────────────────────────────────────────────
# 6. DEPLOY BEST MODEL (MLflow Model Registry)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Model Deployment via MLflow")
print("=" * 60)

best_run_id      = results[best_name]["run_id"]
registered_name  = f"salary_{best_name.replace(' ', '_').lower()}"
model_uri        = f"runs:/{best_run_id}/model"

# Transition to Production stage
client = mlflow.tracking.MlflowClient()
try:
    versions = client.search_model_versions(f"name='{registered_name}'")
    if versions:
        latest_version = sorted(versions, key=lambda v: int(v.version))[-1]
        client.transition_model_version_stage(
            name=registered_name,
            version=latest_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"\n  Model '{registered_name}' v{latest_version.version} → Production")
    else:
        print(f"\n  No registered versions found for '{registered_name}'.")
except Exception as e:
    print(f"\n  Model registry note: {e}")

# Load the production model and run sample predictions
print("\n  Loading model from MLflow for inference...")
loaded_model = mlflow.sklearn.load_model(model_uri)
sample_years = np.array([[2.0], [5.5], [10.0]])

# Apply scaling only for Linear Regression
if best_name == "Linear Regression":
    sample_input = scaler.transform(sample_years)
else:
    sample_input = sample_years

sample_preds = loaded_model.predict(sample_input)
print("\n  Sample predictions (deployed model):")
print(f"  {'YearsExp':>10} | {'Predicted Salary':>18}")
print(f"  {'-'*30}")
for yr, sal in zip(sample_years.flatten(), sample_preds):
    print(f"  {yr:>10.1f} | ${sal:>17,.0f}")

# ─────────────────────────────────────────────────────────
# 7. MODEL MONITORING WITH WHYLOGS (WhyLabs SDK)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Model Monitoring with WhyLogs (WhyLabs)")
print("=" * 60)

# Simulate production data (slight distribution shift)
np.random.seed(0)
prod_years   = np.random.uniform(1, 12, size=50)        # slightly wider range
prod_salaries = loaded_model.predict(
    scaler.transform(prod_years.reshape(-1, 1))
    if best_name == "Linear Regression"
    else prod_years.reshape(-1, 1)
) + np.random.normal(0, 5000, size=50)                  # add noise

prod_df = pd.DataFrame({
    "YearsExperience": prod_years,
    "PredictedSalary": prod_salaries
})

# Profile training data (reference)
train_df = pd.DataFrame({
    "YearsExperience": X_train.flatten(),
    "Salary":          y_train
})

print("\n  Profiling training (reference) data...")
ref_profile = why.log(train_df)
ref_view    = ref_profile.view()
ref_stats   = ref_view.to_pandas()
print(f"  Reference profile columns: {list(ref_stats.index.get_level_values(0).unique())}")

print("\n  Profiling production data...")
prod_profile = why.log(prod_df)
prod_view    = prod_profile.view()
prod_stats   = prod_view.to_pandas()
print(f"  Production profile columns: {list(prod_stats.index.get_level_values(0).unique())}")

# Save profiles — write via view to avoid Windows-illegal timestamp filenames
ref_path  = os.path.join(OUTPUT_DIR, "whylogs_reference_profile.bin")
prod_path = os.path.join(OUTPUT_DIR, "whylogs_production_profile.bin")
ref_view.write(path=ref_path)
prod_view.write(path=prod_path)
print(f"\n  WhyLogs reference profile → {ref_path}")
print(f"  WhyLogs production profile → {prod_path}")

# Drift summary
print("\n  Drift Detection Summary:")
print(f"  {'Feature':<20} {'Ref Mean':>12} {'Prod Mean':>12} {'Drift':>12}")
print(f"  {'-'*60}")

drift_records = {}
for feat, ref_col, prod_col in [
    ("YearsExperience", "YearsExperience", "YearsExperience")
]:
    try:
        ref_mean  = ref_stats.loc[(feat, "mean"), "value"]  if (feat, "mean")  in ref_stats.index  else np.mean(X_train)
        prod_mean = prod_stats.loc[(feat, "mean"), "value"] if (feat, "mean") in prod_stats.index  else np.mean(prod_years)
        drift     = abs(prod_mean - ref_mean) / (abs(ref_mean) + 1e-9) * 100
        drift_records[feat] = {"ref_mean": ref_mean, "prod_mean": prod_mean, "drift_pct": drift}
        flag = " ⚠ DRIFT" if drift > 10 else ""
        print(f"  {feat:<20} {ref_mean:>12.2f} {prod_mean:>12.2f} {drift:>10.1f}%{flag}")
    except Exception as e:
        print(f"  {feat:<20} (stats not available: {e})")

# ─────────────────────────────────────────────────────────
# 8. FINAL SUMMARY REPORT
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"""
  Dataset     : Salary Dataset (30 samples)
  Features    : YearsExperience → Salary
  Split       : 80% train / 20% test

  Model Performance:
  {comparison_df.to_string(index=False)}

  Best Model  : {best_name}  (R² = {best_r2})
  Deployment  : Registered as '{registered_name}' in MLflow (Production stage)
  Monitoring  : WhyLogs profiles logged for reference and production data
  MLflow UI   : Run `mlflow ui --backend-store-uri file://{MLFLOW_DIR}` to explore
""")

print("=" * 60)
print("Capstone pipeline completed successfully!")
print("=" * 60)
