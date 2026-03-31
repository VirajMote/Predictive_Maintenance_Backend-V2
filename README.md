# Predictive Maintenance System — Backend API

A production ML system for industrial machinery failure prediction, RUL estimation, and health scoring. Built with Random Forest, XGBoost, and LSTM models, served via a FastAPI REST API deployed on Railway.

**Live API:** [`https://predictivemaintenance-v2-production.up.railway.app` ](https://predictivemaintenancefrontend-v2-production.up.railway.app/) 
**Interactive Docs:** `https://predictivemaintenance-v2-production.up.railway.app/docs`  
**Frontend Dashboard:** `https://predictivemaintenancefrontend-v2-production.up.railway.app`

---

## What it does

Given sensor readings from an industrial machine (temperature, RPM, torque, tool wear), the system returns:

- **Failure probability** across 6 failure modes (Machine failure, TWF, HDF, PWF, OSF, RNF)
- **Remaining Useful Life (RUL)** in cycles
- **Health score** from 0 (critical) to 100 (healthy)
- **Human-readable alerts** for operators

---

## API Reference

### Endpoints

| Method | Endpoint | Access | Description |
|---|---|---|---|
| POST | `/predict` | Public | Single machine prediction from sensor readings |
| POST | `/predict/batch` | Public | Batch predictions — up to 100 machines per call |
| GET | `/health` | Public | Server status + model load confirmation |
| GET | `/model/info` | Public | Loaded model type, features, and thresholds |
| GET | `/docs` | Public | Interactive Swagger UI |

### Request Schema — `/predict`

```json
{
  "readings": [
    {
      "type_code": "M",
      "air_temperature_K": 298.1,
      "process_temperature_K": 308.6,
      "rotational_speed_rpm": 1551,
      "torque_Nm": 42.8,
      "tool_wear_min": 0
    }
  ],
  "cycle_duration_seconds": null
}
```

| Field | Type | Range | Description |
|---|---|---|---|
| `type_code` | string | L, M, H | Machine quality type |
| `air_temperature_K` | float | 290–320 | Ambient air temperature in Kelvin |
| `process_temperature_K` | float | 300–320 | Process temperature in Kelvin |
| `rotational_speed_rpm` | float | 1000–3000 | Spindle speed in RPM |
| `torque_Nm` | float | 0–80 | Torque in Newton-metres |
| `tool_wear_min` | float | 0–300 | Cumulative tool wear in minutes |
| `cycle_duration_seconds` | float | optional | If provided, converts RUL cycles → hours |

### Response Schema

```json
{
  "failure_probabilities": {
    "Machine failure": 0.03,
    "TWF": 0.01,
    "HDF": 0.02,
    "PWF": 0.01,
    "OSF": 0.00,
    "RNF": 0.00
  },
  "failure_predictions": {
    "Machine failure": 0,
    "TWF": 0,
    "HDF": 0,
    "PWF": 0,
    "OSF": 0,
    "RNF": 0
  },
  "rul_cycles": 187.4,
  "rul_hours": null,
  "health_score": 94.2,
  "health_label": "Healthy",
  "alerts": []
}
```

| Field | Description |
|---|---|
| `failure_probabilities` | Per-label probability from the RF classifier |
| `failure_predictions` | Binary prediction using tuned threshold per label |
| `rul_cycles` | Predicted remaining useful life in cycles (0–200) |
| `rul_hours` | RUL converted to hours if `cycle_duration_seconds` was provided |
| `health_score` | Weighted aggregate score 0–100 (100 = fully healthy) |
| `health_label` | `Healthy` / `Warning` / `Critical` |
| `alerts` | Human-readable alert strings for operators |

### Failure Mode Labels

| Label | Full Name | Description |
|---|---|---|
| `TWF` | Tool Wear Failure | Tool reaches wear limit |
| `HDF` | Heat Dissipation Failure | Temperature delta too low at low RPM |
| `PWF` | Power Failure | Power outside [3500–9000] W range |
| `OSF` | Overstrain Failure | Tool wear × torque exceeds material limit |
| `RNF` | Random Failure | Stochastic failure (0.1% base rate) |

---

## System Architecture

```
Raw Sensor Data (CSV)
        │
        ▼
Feature Engineering
  ├── Physics features (Power = Torque × ω, Strain, Temp Diff)
  ├── Rolling stats (mean/std over 5/10/20 windows)
  └── RUL labels (backward from failure timestamps, capped at 200)
        │
        ▼
Training Pipeline (train.py)
  ├── 70/15/15 stratified split
  ├── RF / XGBoost — 5-fold CV, threshold tuned on val set
  ├── LSTM — BiLSTM classifier + Huber-loss RUL regressor
  └── Best model saved as model_package.pkl
        │
        ▼
FastAPI Inference Server (api.py)
  ├── POST /predict        → single machine prediction
  ├── POST /predict/batch  → up to 100 machines
  ├── GET  /health         → server status
  └── GET  /model/info     → loaded model metadata
```

---

## ML Design Decisions

| Decision | Reason |
|---|---|
| Scaler fit inside CV folds | Prevents data leakage from val/test into normalization |
| Per-label threshold tuning on val set | Default 0.5 is wrong for ~3% positive rate |
| `scale_pos_weight` per XGBoost label | Computed dynamically from actual class counts |
| RUL capped at 200 cycles | Healthy machines don't have a meaningful "true" RUL |
| Huber loss for RUL regression | Robust to outliers vs MSE |
| BiLSTM (bidirectional) | Captures both forward and backward temporal context |
| Sample weights for LSTM | SMOTE doesn't work on time-series sequences |

---

## Model Performance (AI4I 2020 Dataset)

| Model | CV F1 | Test F1 | RUL MAE | RUL RMSE |
|---|---|---|---|---|
| Random Forest | 74.95% | 85.08% | 9.73 cycles | 16.25 cycles |

---

## Project Structure

```
├── config.py               # All hyperparameters and paths
├── feature_engineering.py  # Physics features, rolling stats, RUL labeling
├── baseline_model.py       # RF + XGBoost training and evaluation
├── lstm_model.py           # Bidirectional LSTM classifier + regressor
├── inference.py            # Health score, alerts, prediction schema
├── api.py                  # FastAPI serving layer
├── train.py                # Training orchestrator (entry point)
├── requirements.txt
├── Dockerfile
├── railway.toml
├── data/
│   └── ai4i2020.csv        # UCI AI4I 2020 Predictive Maintenance Dataset
└── models/
    └── model_package.pkl   # Generated after training
```

---

## Running Locally

### 1. Setup

```bash
# Python 3.11 required
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### 2. Train

```bash
# Train RF only (fastest, ~30 seconds)
python train.py --model rf --data data/ai4i2020.csv --save rf

# Train all models, save best
python train.py --model all --data data/ai4i2020.csv --save best
```

### 3. Serve

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### 4. Test

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "readings": [{
      "type_code": "M",
      "air_temperature_K": 298.1,
      "process_temperature_K": 308.6,
      "rotational_speed_rpm": 1551,
      "torque_Nm": 42.8,
      "tool_wear_min": 0
    }]
  }'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"readings": [{"type_code": "M", "air_temperature_K": 298.1, "process_temperature_K": 308.6, "rotational_speed_rpm": 1551, "torque_Nm": 42.8, "tool_wear_min": 0}]},
      {"readings": [{"type_code": "H", "air_temperature_K": 300.2, "process_temperature_K": 311.3, "rotational_speed_rpm": 1285, "torque_Nm": 65.4, "tool_wear_min": 240}]}
    ]
  }'
```

---

## Deployment (Railway)

The model trains automatically on first startup — no pre-built model file needed:

```
Docker build → pip install → python train.py → uvicorn api:app
```

Redeployment triggers automatically on every `git push` to `main`.

To retrain with new data:
```bash
# Replace data/ai4i2020.csv, then:
git add data/
git commit -m "chore: update dataset"
git push
```

---

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — UCI Machine Learning Repository.

10,000 data points · 5 sensor features · 6 failure mode labels · 3.39% failure rate

---

## Tech Stack

`Python 3.11` · `scikit-learn` · `XGBoost` · `TensorFlow/Keras` · `FastAPI` · `Pydantic` · `joblib` · `Docker` · `Railway`
