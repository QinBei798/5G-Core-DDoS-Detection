# 5G Core Network DDoS/Signaling Storm Detection System — Design Spec

**Date:** 2026-06-01
**Scope:** Single-project, greenfield
**Approach:** Config-driven modular Python package

## Overview

A template-based simulation project that generates synthetic 5G Core Network signaling data, trains two ML classifiers (Random Forest, SVM) to detect DDoS/Signaling Storm attacks, and produces evaluation plots for a research paper.

All tunable parameters live in `config.json`. Each component is an independent module in `src/`.

---

## File Architecture

```
5G-Core-Network/
├── config.json                    # All tunable parameters (single source of truth)
├── main.py                        # Entry point — orchestrates the pipeline end-to-end
├── src/
│   ├── __init__.py
│   ├── data_generator.py          # Synthetic 5G signaling data generator (numpy)
│   ├── ml_pipeline.py             # Random Forest + SVM training & evaluation
│   └── plotter.py                 # ROC curve comparison + confusion matrix charts
├── roc_curve.png                  # Generated artifact
├── confusion_matrix.png           # Generated artifact
└── requirements.txt               # numpy, scikit-learn, matplotlib
```

---

## Component Specifications

### 1. `config.json` — Configuration Profile

Controls all data generation parameters and ML hyperparameters. Users swap experiments by editing this file — no code changes needed.

```json
{
  "data": {
    "n_samples": 5000,
    "attack_ratio": 0.35,
    "random_seed": 42,
    "test_split": 0.2,
    "features": {
      "http2_sbi_request_rate": {
        "normal": { "mean": 100, "std": 30 },
        "attack": { "mean": 800, "std": 200 }
      },
      "pfcp_session_msg_density": {
        "normal": { "mean": 50, "std": 15 },
        "attack": { "mean": 400, "std": 120 }
      },
      "ngap_auth_anomaly_score": {
        "normal": { "mean": 0.05, "std": 0.03 },
        "attack": { "mean": 0.75, "std": 0.20 }
      },
      "malicious_ip_ratio": {
        "normal": { "mean": 0.01, "std": 0.01 },
        "attack": { "mean": 0.60, "std": 0.25 }
      }
    }
  },
  "models": {
    "random_forest": {
      "n_estimators": 100,
      "max_depth": 10,
      "random_state": 42
    },
    "svm": {
      "C": 1.0,
      "gamma": "scale",
      "kernel": "rbf",
      "probability": true,
      "random_state": 42
    }
  },
  "output": {
    "roc_curve_path": "roc_curve.png",
    "confusion_matrix_path": "confusion_matrix.png",
    "figure_dpi": 150,
    "figsize": [8, 6]
  }
}
```

All four features use **5G Core Network terminology**:

| Feature | 5G Domain | Description |
|---|---|---|
| `http2_sbi_request_rate` | SBI (Service-Based Interface) | HTTP/2 request rate across NFs |
| `pfcp_session_msg_density` | UPF/SMF (PFCP — N4 interface) | PFCP session message density |
| `ngap_auth_anomaly_score` | AMF/gNB (NGAP — N2 interface) | Authentication anomaly score on NGAP |
| `malicious_ip_ratio` | NRF/SEPP | Ratio of connections from known-malicious IPs |

### 2. `src/data_generator.py` — Synthetic Data Generator

**Interface:**
```python
def generate_data(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) — feature matrix and label vector."""
```

**Behavior:**
- Uses `np.random.default_rng(seed=config["data"]["random_seed"])` for reproducibility.
- Generates `n_samples` rows. `attack_ratio` determines label split.
- Normal samples draw from Normal distributions per feature (configurable mean/std per class).
- Attack samples use elevated means and wider variance to simulate signaling storms.
- Label: `0` = Normal, `1` = Attack (DDoS/Signaling Storm).
- Feature values are clipped to `>= 0` after generation (rates and ratios must be non-negative).
- Returns a shape check assertion: `X.shape == (n_samples, 4)` and `y.shape == (n_samples,)`.

**Dependencies:** numpy only.

### 3. `src/ml_pipeline.py` — ML Training & Evaluation

**Interface:**
```python
def train_and_evaluate(
    X: np.ndarray, y: np.ndarray, config: dict
) -> dict:
    """Return trained models, predictions, and probabilities."""
```

**Return schema:**
```python
{
    "rf_model": RandomForestClassifier,   # fitted
    "svm_model": SVC,                     # fitted (probability=True)
    "y_test": np.ndarray,
    "y_pred_rf": np.ndarray,
    "y_pred_svm": np.ndarray,
    "y_proba_rf": np.ndarray,             # shape (n_test, 2), column 1 = attack probability
    "y_proba_svm": np.ndarray,            # shape (n_test, 2)
    "scaler": StandardScaler,             # fitted
}
```

**Behavior:**
1. `train_test_split` with `test_size=config["data"]["test_split"]`, `stratify=y`, `random_state=config["data"]["random_seed"]`.
2. `StandardScaler.fit_transform` on training features; `.transform` on test features.
3. Train `RandomForestClassifier` using params from `config["models"]["random_forest"]`.
4. Train `SVC` using params from `config["models"]["svm"]` (must include `probability=True` for ROC).
5. Print classification reports (precision, recall, f1) to stdout for both models.

**Dependencies:** scikit-learn (`train_test_split`, `StandardScaler`, `RandomForestClassifier`, `SVC`, `classification_report`).

### 4. `src/plotter.py` — Chart Generation

**Critical:** `matplotlib.use('Agg')` is set **at module level** (before any pyplot import) to ensure headless WSL compatibility.

**Interface:**
```python
def plot_roc_curve(y_true: np.ndarray, proba_dict: dict, config: dict) -> None:
    """Save ROC curve comparison of RF vs SVM to config path."""

def plot_confusion_matrix(y_true: np.ndarray, y_pred_rf: np.ndarray, config: dict) -> None:
    """Save Random Forest confusion matrix heatmap to config path."""
```

**ROC Curve (`plot_roc_curve`):**
- `proba_dict`: `{"Random Forest": y_proba_rf[:, 1], "SVM": y_proba_svm[:, 1]}`
- Overlays both models on a single chart.
- Includes diagonal dashed baseline (random classifier).
- Legend shows AUC for each model.
- Title: "ROC Curve — 5G Core DDoS Detection"
- Saves to `config["output"]["roc_curve_path"]` (default: `roc_curve.png` in workspace root).

**Confusion Matrix (`plot_confusion_matrix`):**
- Random Forest only.
- Annotated heatmap with counts and percentages.
- Labels: "Normal" and "Attack".
- Color map: Blues.
- Title: "Confusion Matrix — Random Forest"
- Saves to `config["output"]["confusion_matrix_path"]` (default: `confusion_matrix.png` in workspace root).

**Figure settings:** `dpi` and `figsize` from config.

**Dependencies:** matplotlib, numpy.

### 5. `main.py` — Orchestrator

Minimal entry point. Loads config, calls each module, reports results.

```python
import json
from src.data_generator import generate_data
from src.ml_pipeline import train_and_evaluate
from src.plotter import plot_roc_curve, plot_confusion_matrix

def main():
    with open("config.json") as f:
        config = json.load(f)

    X, y = generate_data(config)
    results = train_and_evaluate(X, y, config)

    proba_dict = {
        "Random Forest": results["y_proba_rf"][:, 1],
        "SVM": results["y_proba_svm"][:, 1],
    }
    plot_roc_curve(results["y_test"], proba_dict, config)
    plot_confusion_matrix(results["y_test"], results["y_pred_rf"], config)
    print("Done. Charts saved.")

if __name__ == "__main__":
    main()
```

---

## Data Flow

```
config.json
    │
    ▼
data_generator.py ──► (X: 5000×4, y: 5000,)
    │
    ▼
ml_pipeline.py ──► {rf_model, svm_model, y_test, y_pred_rf, y_pred_svm, y_proba_rf, y_proba_svm, scaler}
    │
    ├──► plotter.plot_roc_curve()      ──► roc_curve.png
    └──► plotter.plot_confusion_matrix() ──► confusion_matrix.png
```

---

## Error Handling

| Scenario | Behavior |
|---|---|
| `config.json` missing or unparseable | Print "Error: config.json not found or invalid JSON." and `sys.exit(1)` |
| Invalid config key (e.g., missing `data.random_seed`) | Assert with descriptive message at function entry |
| Feature value < 0 after generation | Clip to 0 (valid for rates/ratios) |
| PNG save fails (disk full, permissions) | Catch `OSError`, print the failing path, re-raise |

---

## Dependency List (`requirements.txt`)

```
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
```

No other dependencies.

---

## Acceptance Criteria

1. `python main.py` runs end-to-end without errors in a headless WSL terminal.
2. `roc_curve.png` and `confusion_matrix.png` exist in workspace root after execution.
3. Both models achieve >90% accuracy on the synthetic dataset (reasonably separable classes).
4. Changing `config.json` values (e.g., `n_samples`, `attack_ratio`, RF `n_estimators`) produces different outputs without code changes.
5. Running twice with the same seed produces identical outputs (determinism).
