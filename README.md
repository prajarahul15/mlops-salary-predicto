# MLOps Certification Course — Capstone Project

## Salary Prediction using Years of Experience

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Objectives](#2-objectives)
3. [Dataset](#3-dataset)
4. [Project Structure](#4-project-structure)
5. [Step 1 — Environment Setup](#5-step-1--environment-setup)
6. [Step 2 — Data Collection & Preprocessing](#6-step-2--data-collection--preprocessing)
7. [Step 3 — Model Training & Experiment Tracking (MLflow)](#7-step-3--model-training--experiment-tracking-mlflow)
8. [Step 4 — Model Deployment (MLflow)](#8-step-4--model-deployment-mlflow)
9. [Step 5 — Model Monitoring (WhyLabs / WhyLogs)](#9-step-5--model-monitoring-whylabs--whylogs)
10. [Results & Model Comparison](#10-results--model-comparison)
11. [Expected Outcomes](#11-expected-outcomes)
12. [How to Run](#12-how-to-run)

---

## 1. Problem Statement

In many industries, salary structures are influenced significantly by the level of experience an individual possesses. Accurate salary predictions based on such a variable can help organizations in budget planning and ensuring equitable salary distributions.

This project develops a **predictive model that estimates salaries using years of experience** as the primary predictor. The challenge is to build and compare **multiple regression models** to identify which model provides the most accurate and reliable predictions.

To enhance the development process and ensure model reliability post-deployment, the project integrates:

- **MLflow** — for managing the machine learning lifecycle (experiment tracking, model versioning, and deployment).
- **WhyLabs (WhyLogs)** — for monitoring the model in production to track performance metrics and detect deviations or anomalies in real-time.

---

## 2. Objectives

| # | Objective | Description |
|---|-----------|-------------|
| 1 | **Model Development** | Develop three types of regression models |
| 2 | **Linear Regression** | Establish a baseline for performance comparison |
| 3 | **Random Forest Regressor** | Assess performance improvements using an ensemble method |
| 4 | **Decision Tree Regressor** | Evaluate a non-linear regression approach |
| 5 | **Experiment Tracking** | Use MLflow to track parameters, metrics, models, and artifacts |
| 6 | **Model Deployment** | Deploy the best-performing model into a production-like environment using MLflow |
| 7 | **Model Monitoring** | Implement WhyLabs to monitor the deployed model, detecting operational drifts or anomalies |

---

## 3. Dataset

- **Source:** Kaggle — *Salary Dataset*
- **File:** `2687_capstone_project_dataset_v1_vv6_ahjq7xz.csv`
- **Records:** 30 samples
- **Features:**

| Column | Type | Description |
|--------|------|-------------|
| `YearsExperience` | float | Number of years of professional experience (1.2 – 10.6) |
| `Salary` | float | Annual salary in USD (37,732 – 122,392) |

**Descriptive Statistics:**

| Statistic | YearsExperience | Salary |
|-----------|----------------|--------|
| Count | 30 | 30 |
| Mean | 5.41 | $76,004 |
| Std Dev | 2.84 | $27,414 |
| Min | 1.20 | $37,732 |
| Max | 10.60 | $122,392 |

---

## 4. Project Structure

```
Capstone - Copy/
│
├── 2687_capstone_project_1_kzi_y15cb7q.pdf          # Assignment instructions
├── 2687_capstone_project_dataset_v1_vv6_ahjq7xz.csv # Salary dataset
├── capstone_mlops.py                                  # Main pipeline script
├── README.md                                          # This file
│
├── mlruns/                                            # MLflow experiment store
│   ├── 0/                                             # Default experiment
│   ├── <experiment_id>/                               # Salary_Prediction_Capstone
│   └── models/                                        # Registered model versions
│       ├── salary_linear_regression/
│       ├── salary_random_forest/
│       └── salary_decision_tree/
│
└── outputs/
    ├── model_comparison.csv                           # Metrics table (all models)
    ├── model_comparison_plot.png                      # Fit-line plots (3 models)
    ├── r2_comparison.png                              # R² bar chart
    ├── reference_profile*.bin                         # WhyLogs training data profile
    └── production_profile*.bin                        # WhyLogs production data profile
```

---

## 5. Step 1 — Environment Setup

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install mlflow scikit-learn pandas numpy matplotlib whylogs
```

### Key Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | ≥1.0 | Model training and evaluation |
| `mlflow` | ≥2.0 | Experiment tracking, model registry, deployment |
| `whylogs` | ≥1.0 | Data profiling and drift monitoring (WhyLabs SDK) |
| `pandas` | ≥1.3 | Data loading and manipulation |
| `numpy` | ≥1.21 | Numerical computations |
| `matplotlib` | ≥3.4 | Visualizations |

### MLflow Tracking URI

All experiments are stored locally in the `mlruns/` directory. The tracking URI is set programmatically:

```python
import mlflow
mlflow.set_tracking_uri("file:///path/to/Capstone - Copy/mlruns")
```

To launch the MLflow UI after running the pipeline:

```bash
mlflow ui --backend-store-uri file:///path/to/Capstone - Copy/mlruns
# Then open http://127.0.0.1:5000 in your browser
```

---

## 6. Step 2 — Data Collection & Preprocessing

### Loading the Data

```python
import pandas as pd

df = pd.read_csv("2687_capstone_project_dataset_v1_vv6_ahjq7xz.csv", index_col=0)
```

### Data Quality Checks

- **Missing values:** None detected across all columns.
- **Outliers:** No anomalous values; salary range is consistent with years-of-experience trend.
- **Data types:** Both columns are numeric floats — no encoding required.

### Train / Test Split

The dataset is split 80/20 using a fixed random seed for reproducibility:

```python
from sklearn.model_selection import train_test_split

X = df[["YearsExperience"]].values
y = df["Salary"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train: 24 samples | Test: 6 samples
```

### Feature Scaling

`StandardScaler` is applied for the **Linear Regression** model (tree-based models do not require scaling):

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
```

---

## 7. Step 3 — Model Training & Experiment Tracking (MLflow)

All three models are trained within MLflow runs under the experiment **`Salary_Prediction_Capstone`**. For each run, the following are logged:

- **Parameters** — model type, hyperparameters
- **Metrics** — MSE, RMSE, MAE, R²
- **Model artifact** — serialised model with input/output signature
- **Registered model** — auto-registered to the MLflow Model Registry

### 7.1 Linear Regression (Baseline)

```python
from sklearn.linear_model import LinearRegression
import mlflow, mlflow.sklearn

mlflow.set_experiment("Salary_Prediction_Capstone")

with mlflow.start_run(run_name="Linear Regression"):
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    mlflow.log_params({"model_type": "LinearRegression", "fit_intercept": True})
    mlflow.log_metrics({"R2": 0.9024, "RMSE": 7059.04, "MAE": 6286.45, "MSE": 49830096.86})
    mlflow.sklearn.log_model(model, name="model",
                             registered_model_name="salary_linear_regression")
```

**Purpose:** Establishes the performance baseline. Given the strong linear relationship in the data, this model is expected to perform well.

### 7.2 Random Forest Regressor (Ensemble)

```python
from sklearn.ensemble import RandomForestRegressor

with mlflow.start_run(run_name="Random Forest"):
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    mlflow.log_params({"model_type": "RandomForestRegressor",
                       "n_estimators": 100, "max_depth": 5})
    mlflow.log_metrics({"R2": 0.8793, "RMSE": 7853.30, "MAE": 6697.81})
    mlflow.sklearn.log_model(model, name="model",
                             registered_model_name="salary_random_forest")
```

**Purpose:** Tests whether an ensemble of decision trees can improve upon the baseline through variance reduction.

### 7.3 Decision Tree Regressor (Non-linear)

```python
from sklearn.tree import DecisionTreeRegressor

with mlflow.start_run(run_name="Decision Tree"):
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    mlflow.log_params({"model_type": "DecisionTreeRegressor", "max_depth": 5})
    mlflow.log_metrics({"R2": 0.8349, "RMSE": 9183.05, "MAE": 7178.00})
    mlflow.sklearn.log_model(model, name="model",
                             registered_model_name="salary_decision_tree")
```

**Purpose:** Evaluates a non-linear, rule-based approach that partitions the feature space into rectangular regions.

---

## 8. Step 4 — Model Deployment (MLflow)

### Selecting the Best Model

The best model is selected by highest R² score on the test set:

```
Best Model: Linear Regression  (R² = 0.9024)
Registered as: salary_linear_regression
```

### Transitioning to Production

Using the MLflow Python client, the best model is promoted to the **Production** stage in the Model Registry:

```python
client = mlflow.tracking.MlflowClient()

client.transition_model_version_stage(
    name="salary_linear_regression",
    version="1",
    stage="Production",
    archive_existing_versions=True
)
```

### Loading & Serving the Production Model

```python
loaded_model = mlflow.sklearn.load_model("runs:/<run_id>/model")
predictions = loaded_model.predict(scaler.transform([[5.5]]))
# → $76,211
```

### Sample Inference Results

| Years of Experience | Predicted Salary |
|--------------------|-----------------|
| 2.0 | $43,228 |
| 5.5 | $76,211 |
| 10.0 | $118,618 |

### Serving via REST API (optional)

The registered Production model can be served as a REST endpoint using MLflow's built-in serving:

```bash
mlflow models serve \
  -m "models:/salary_linear_regression/Production" \
  --port 5001 \
  --no-conda

# Send a prediction request:
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"instances": [[5.5]]}'
```

---

## 9. Step 5 — Model Monitoring (WhyLabs / WhyLogs)

WhyLogs (the open-source SDK powering WhyLabs) is used to profile and monitor data distributions in training vs. production.

### 9.1 Profiling Training Data (Reference Baseline)

```python
import whylogs as why
import pandas as pd

train_df = pd.DataFrame({
    "YearsExperience": X_train.flatten(),
    "Salary": y_train
})

ref_profile = why.log(train_df)
ref_profile.writer("local").option(base_dir="outputs", filename="reference_profile").write()
```

### 9.2 Profiling Production Data

Simulated production data (50 samples with a slightly wider years-of-experience range and noise added to predictions) is profiled in the same way:

```python
prod_df = pd.DataFrame({
    "YearsExperience": prod_years,
    "PredictedSalary": prod_salaries
})

prod_profile = why.log(prod_df)
prod_profile.writer("local").option(base_dir="outputs", filename="production_profile").write()
```

### 9.3 Drift Detection Results

| Feature | Reference Mean | Production Mean | Drift (%) | Status |
|---------|---------------|----------------|-----------|--------|
| YearsExperience | 5.29 | 6.92 | **30.8%** | ⚠ DRIFT DETECTED |

A drift threshold of 10% is used. The 30.8% shift in mean `YearsExperience` between training and the simulated production batch triggers a **drift alert** — this would warrant model retraining or recalibration in a real production environment.

### 9.4 Connecting to WhyLabs Cloud (Production Setup)

To send profiles to the WhyLabs cloud dashboard, set the following environment variables with your WhyLabs API credentials:

```python
import os
import whylogs as why
from whylogs.api.writer.whylabs import WhyLabsWriter

os.environ["WHYLABS_DEFAULT_ORG_ID"]   = "<your-org-id>"
os.environ["WHYLABS_API_KEY"]          = "<your-api-key>"
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "<your-dataset-id>"

profile = why.log(prod_df)
profile.writer("whylabs").write()
```

This streams the profiles to the WhyLabs dashboard for real-time monitoring, anomaly detection, and alerting.

---

## 10. Results & Model Comparison

### Metrics Summary (Test Set)

| Model | R² | RMSE ($) | MAE ($) | MSE |
|-------|----|----------|---------|-----|
| **Linear Regression** | **0.9024** | **7,059** | **6,286** | 49,830,097 |
| Random Forest | 0.8793 | 7,853 | 6,698 | 61,674,290 |
| Decision Tree | 0.8349 | 9,183 | 7,178 | 84,328,468 |

### Key Findings

**Linear Regression wins** across all metrics. This is expected because:

- The underlying relationship between years of experience and salary is strongly **linear** (the data follows an approximately straight-line trend from ~$37K at 1.2 years to ~$122K at 10.6 years).
- Tree-based models (Random Forest, Decision Tree) are powerful for complex, non-linear relationships but can **overfit or underfit** on small, linearly-distributed datasets like this one (30 samples).
- Random Forest improves over Decision Tree by reducing variance through bagging, but still falls short of the simpler linear model.

### Visual Outputs

The following plots are saved in the `outputs/` directory:

- **`model_comparison_plot.png`** — Scatter plots with fit lines for all 3 models side by side.
- **`r2_comparison.png`** — Bar chart comparing R² scores across models.

---

## 11. Expected Outcomes

As per the project objectives, the following deliverables have been produced:

| # | Expected Outcome | Status |
|---|-----------------|--------|
| 1 | Comparison of model performances to identify the most effective regression technique | ✅ Complete — see Section 10 |
| 2 | Fully tracked and documented ML project via MLflow, with models ready for deployment | ✅ Complete — 3 runs logged, models registered in MLflow Model Registry |
| 3 | Operational model in production with ongoing performance monitoring through WhyLabs | ✅ Complete — best model promoted to Production stage; WhyLogs profiles generated with drift detection |

---

## 12. How to Run

### 1. Clone / navigate to the project folder

```bash
cd "Capstone - Copy"
```

### 2. Install dependencies

```bash
pip install mlflow scikit-learn pandas numpy matplotlib whylogs
```

### 3. Run the pipeline

```bash
python capstone_mlops.py
```

The script will:
- Load and preprocess the dataset
- Train all 3 models with MLflow experiment tracking
- Print a model comparison table
- Deploy the best model to the MLflow Model Registry (Production stage)
- Profile training and production data with WhyLogs
- Print drift detection results
- Save plots and CSVs to `outputs/`

### 4. Explore experiments in MLflow UI

```bash
mlflow ui --backend-store-uri file://./mlruns
# Open http://127.0.0.1:5000
```

---

## Author

**Rahul** | MLOps Certification Course — Capstone Project  
Dataset: Salary Dataset (Kaggle)  
Tools: Python · scikit-learn · MLflow · WhyLogs (WhyLabs)
