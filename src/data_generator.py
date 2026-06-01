"""Synthetic 5G Core Network signaling data generator.

Generates realistic feature vectors mimicking both normal 5G control-plane
traffic and DDoS attack patterns targeting the Service-Based Interface (SBI),
N2/N3/N4 protocol layers.
"""

import json
import sys

import numpy as np


def load_config(config_path: str = "config.json") -> dict:
    """Load and return the configuration from a JSON file.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    SystemExit
        If the file is not found or contains invalid JSON.
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file '{config_path}': {e}")
        sys.exit(1)
    if "data" not in config:
        print(f"Error: Missing 'data' section in {config_path}")
        sys.exit(1)
    if "features" not in config["data"]:
        print(f"Error: Missing 'data.features' section in {config_path}")
        sys.exit(1)
    return config


def generate_5g_signaling_data(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic 5G Core Network signaling data.

    Draws independent normal-distribution samples for each of the four
    5G-specific features, using per-class (Normal / Attack) means and
    standard deviations sourced from the configuration.

    Feature overview
    ----------------
    * HTTP2_SBI_request_rate       – HTTP/2 request rate on SBI (req/s)
    * PFCP_session_msg_density     – PFCP N4 session message density (msg/s)
    * NGAP_auth_anomaly_score      – NGAP N2 authentication anomaly score [0-1]
    * GTPU_tunnel_throughput_variance – GTP-U N3 tunnel throughput variance [0-1]

    Attack samples exhibit significantly elevated means (e.g. ~8-15x normal)
    to simulate a signalling-storm / service-exhaustion DDoS scenario.

    Parameters
    ----------
    config : dict
        Top-level configuration dictionary.  Must contain the ``"data"`` key
        with sub-keys ``n_samples``, ``attack_ratio``, ``random_seed``, and
        ``features``.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 4)
        Feature matrix.  Column order follows the key order in
        ``config["data"]["features"]``.
    y : np.ndarray of shape (n_samples,)
        Integer labels: 0 for Normal traffic, 1 for Attack traffic.
    """
    data_cfg = config["data"]
    n_samples: int = data_cfg["n_samples"]
    attack_ratio: float = data_cfg["attack_ratio"]
    random_seed: int = data_cfg["random_seed"]
    features_cfg: dict = data_cfg["features"]

    rng = np.random.default_rng(seed=random_seed)

    n_attack = int(n_samples * attack_ratio)
    n_normal = n_samples - n_attack

    feature_names = list(features_cfg.keys())
    n_features = len(feature_names)

    # ---- Draw per-class samples for every feature ---------------------------
    X_normal = np.zeros((n_normal, n_features))
    X_attack = np.zeros((n_attack, n_features))

    for i, name in enumerate(feature_names):
        fc = features_cfg[name]
        X_normal[:, i] = rng.normal(
            loc=fc["normal"]["mean"],
            scale=fc["normal"]["std"],
            size=n_normal,
        )
        X_attack[:, i] = rng.normal(
            loc=fc["attack"]["mean"],
            scale=fc["attack"]["std"],
            size=n_attack,
        )

    # ---- Combine, label, shuffle, clip -------------------------------------
    X = np.vstack([X_normal, X_attack])
    y = np.hstack(
        [np.zeros(n_normal, dtype=np.int64), np.ones(n_attack, dtype=np.int64)]
    )

    perm = rng.permutation(n_samples)
    X = X[perm]
    y = y[perm]

    # Physical constraint: rates / scores cannot be negative
    X = np.clip(X, 0.0, None)
    X[:, 2] = np.clip(X[:, 2], 0.0, 1.0)  # NGAP_auth_anomaly_score in [0,1]
    X[:, 3] = np.clip(X[:, 3], 0.0, 1.0)  # GTPU_tunnel_throughput_variance in [0,1]

    # ---- Sanity checks ------------------------------------------------------
    assert X.shape == (n_samples, n_features), (
        f"Unexpected X.shape: expected {(n_samples, n_features)}, got {X.shape}"
    )
    assert y.shape == (n_samples,), (
        f"Unexpected y.shape: expected {(n_samples,)}, got {y.shape}"
    )

    return X, y
