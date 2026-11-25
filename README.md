# ML Sentinel

> An autonomous reliability layer for production machine learning systems.

ML Sentinel is an experimental MLOps platform designed to observe, evaluate, and automatically respond to problems in production ML systems.

The goal is not simply to deploy models.

The goal is to build a system that can answer:

- Is this model healthy?
- Is the production data changing?
- Is model performance degrading?
- Should a new model version be promoted?
- Should a deployment be blocked?
- Should the system retrain?
- Should we roll back?

## Vision

ML systems are different from traditional software systems.

A service can be:

- running
- responsive
- within CPU/memory limits
- returning HTTP 200

and still be producing bad predictions.

ML Sentinel aims to detect these failures and turn observations into automated decisions.

## Core Loop

```text
Observe
   ↓
Detect
   ↓
Diagnose
   ↓
Decide
   ↓
Act
   ↓
Verify
   ↺
```

## Planned Capabilities

### Model Lifecycle
- Experiment tracking
- Model versioning
- Model evaluation
- Model promotion
- Model rollback

### ML Observability
- Data drift detection
- Prediction drift detection
- Model quality monitoring
- Latency monitoring
- Feature monitoring

### Reliability
- Deployment policies
- Canary evaluation
- Automatic rollback
- Model health scoring
- Failure detection

### Automation
- Automated retraining
- Model promotion
- Deployment blocking
- Recovery workflows

### Chaos Testing

The platform will intentionally introduce failures such as:

- Data drift
- Concept drift
- Training-serving skew
- Bad model versions
- Latency spikes
- Stale models
- Canary failures

The objective is to test whether the platform can detect and respond to these situations.

## Architecture

```text
                 ML SENTINEL

                    Data
                     │
                     ▼
              ┌──────────────┐
              │ Feature Layer│
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │ Model Gateway│
              └──────┬───────┘
                     │
              ┌──────┴──────┐
              │             │
              ▼             ▼
           Model v1      Model v2
              │             │
              └──────┬──────┘
                     ▼
              ┌──────────────┐
              │Observability │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │Policy Engine │
              └──────┬───────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Retrain    Promote    Rollback
```

## Technology

The project is designed to run locally using free/open-source tools.

Planned technologies include:

- Python
- scikit-learn
- MLflow
- FastAPI
- DuckDB
- Evidently
- Prometheus
- Grafana
- Docker
- GitHub Actions
- pytest

Additional technologies will only be introduced when they solve a clearly defined engineering problem.

## Model Evaluation

Training now runs stratified five-fold cross-validation on the training split and logs each metric's mean and standard deviation to MLflow. The existing holdout validation metrics remain separate and unchanged.

An optional benchmark compares the logistic-regression baseline with a CPU-only XGBoost candidate using the same folds. Logistic regression remains the default; benchmarking does not register or promote the candidate.

An optional logistic-regression tuning helper searches regularization strength and class weighting using stratified cross-validation. It returns a best estimator for evaluation without changing the default training workflow.

SHAP explanations are available for the fitted logistic-regression pipeline. Attributions are additive on the model's log-odds scale and do not alter predictions or model registry state.

## Local Data Versioning

DVC tracks the reference dataset and generated production datasets using local cache storage. Dataset contents remain local and ignored by Git; DVC metadata files are versioned. No DVC remote is configured, so data is not uploaded or shared. Install `requirements-dev.txt`, then use `dvc status` and `dvc checkout` to inspect or restore tracked data.

## Project Philosophy

This project follows one principle:

> Don't add infrastructure because it is popular.  
> Add it because the system needs it.

The platform will evolve incrementally, with architectural decisions documented throughout development.

## Status

🚧 Currently under development.

**Day 01 — Platform foundation**

## Local CI and Container Image

The GitHub Actions workflow runs the test suite and builds the container image on pushes to `main`, pull requests targeting `main`, and manual dispatches. To run the same checks locally:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
docker build --tag ml-sentinel:local .
```

`requirements.txt` contains runtime dependencies; `requirements-dev.txt` adds test dependencies. The serving container expects a reachable MLflow tracking store with a model assigned to the `production` alias. Configure `MLFLOW_TRACKING_URI` for that store when running the image; building the image does not provision or modify a model registry.
