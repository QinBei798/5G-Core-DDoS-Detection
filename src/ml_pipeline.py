"""Machine-learning pipeline for 5G Core DDoS detection.

Trains Random Forest and SVM classifiers on pre-generated 5G signaling
feature vectors and returns models along with predictions for downstream
evaluation and plotting.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_and_evaluate_models(
    X: np.ndarray, y: np.ndarray, config: dict
) -> dict:
    """Train Random Forest and SVM classifiers on 5G signaling data.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 4)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Label vector (0 = Normal, 1 = Attack).
    config : dict
        Configuration dictionary loaded from config.json.  Must contain
        ``"data"`` (with keys ``test_split`` and ``random_seed``) and
        ``"models"`` (with sub-dicts ``"random_forest"`` and ``"svm"``).

    Returns
    -------
    dict
        Keys:
        - rf_model : trained RandomForestClassifier
        - svm_model : trained SVC
        - y_test : true labels for the test set
        - y_proba_rf : predicted probabilities from Random Forest
        - y_proba_svm : predicted probabilities from SVM
        - y_pred_rf : hard predictions from Random Forest
    """
    print("[ML Pipeline] Starting training pipeline...")

    if "data" not in config:
        raise KeyError("Missing 'data' section in config")
    if "models" not in config:
        raise KeyError("Missing 'models' section in config")
    if "random_forest" not in config["models"]:
        raise KeyError("Missing 'models.random_forest' section in config")
    if "svm" not in config["models"]:
        raise KeyError("Missing 'models.svm' section in config")

    # ---- 1. Train / test split ------------------------------------------------
    data_cfg = config["data"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_cfg["test_split"],
        random_state=data_cfg["random_seed"],
        stratify=y,
    )

    # ---- 2. Feature scaling ---------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[ML Pipeline] Feature scaling complete.")

    # ---- 3. Train Random Forest -----------------------------------------------
    rf_cfg = config["models"]["random_forest"]
    rf_model = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        random_state=rf_cfg["random_state"],
    )
    rf_model.fit(X_train_scaled, y_train)
    print("[ML Pipeline] Random Forest training complete.")

    # ---- 4. Train SVM ---------------------------------------------------------
    svm_cfg = config["models"]["svm"]
    svm_model = SVC(
        C=svm_cfg["C"],
        gamma=svm_cfg["gamma"],
        kernel=svm_cfg["kernel"],
        probability=svm_cfg["probability"],
        random_state=svm_cfg["random_state"],
    )
    svm_model.fit(X_train_scaled, y_train)
    print("[ML Pipeline] SVM training complete.")

    # ---- 5. Generate predictions ----------------------------------------------
    y_pred_rf = rf_model.predict(X_test_scaled)
    y_proba_rf = rf_model.predict_proba(X_test_scaled)
    y_proba_svm = svm_model.predict_proba(X_test_scaled)

    # ---- 6. Classification reports --------------------------------------------
    print(
        classification_report(
            y_test,
            y_pred_rf,
            target_names=["Normal", "Attack"],
        )
    )
    print(
        classification_report(
            y_test,
            svm_model.predict(X_test_scaled),
            target_names=["Normal", "Attack"],
        )
    )

    # ---- 7. Return ------------------------------------------------------------
    return {
        "rf_model": rf_model,
        "svm_model": svm_model,
        "y_test": y_test,
        "y_proba_rf": y_proba_rf,
        "y_proba_svm": y_proba_svm,
        "y_pred_rf": y_pred_rf,
    }
