#!/usr/bin/env python3
"""Multi-scenario experiment orchestrator for 5G Core DDoS Detection.

Runs three isolated experiment scenarios (baseline, stealth, high_noise),
each with tailored config modifications, and collects output artifacts into
dedicated subdirectories under experiments/.
"""

import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
EXPERIMENTS_DIR = ROOT / "experiments"
ARTIFACTS = ["roc_curve.png", "confusion_matrix.png"]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(config: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def run_pipeline() -> bool:
    result = subprocess.run(
        [sys.executable, str(ROOT / "main.py")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr.strip()}")
        return False
    return True


def collect_artifacts(scenario_dir: Path, config: dict) -> None:
    scenario_dir.mkdir(parents=True, exist_ok=True)
    for artifact in ARTIFACTS:
        src = ROOT / artifact
        if src.exists():
            shutil.move(str(src), str(scenario_dir / artifact))
    save_config_snapshot = scenario_dir / "config.json"
    with open(save_config_snapshot, "w") as f:
        json.dump(config, f, indent=2)


# ── Scenario Definitions ──────────────────────────────────────────────

BASE_CONFIG = load_config()

SCENARIOS = {
    "baseline": {
        "name": "Standard Baseline",
        "mods": {},  # Run with current defaults
    },
    "stealth": {
        "name": "Stealthy & Micro Attack",
        "mods": {
            "data.attack_ratio": 0.05,
            "data.features.NGAP_auth_anomaly_score.attack.mean": 0.25,
        },
    },
    "high_noise": {
        "name": "High Background Noise",
        "mods": {
            "data.features.HTTP2_SBI_request_rate.normal.std": 180,   # 3×60
            "data.features.PFCP_session_msg_density.normal.std": 75,   # 3×25
            "data.features.NGAP_auth_anomaly_score.normal.std": 0.21,  # 3×0.07
            "data.features.GTPU_tunnel_throughput_variance.normal.std": 0.15,  # 3×0.05
        },
    },
}


def set_nested(d: dict, key_path: str, value) -> None:
    """Set a nested dict value by dot-separated key path."""
    keys = key_path.split(".")
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("5G Core DDoS Detection — Multi-Scenario Experiment Suite")
    print("=" * 60)

    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    results = []

    for slug, scenario in SCENARIOS.items():
        print(f"\n>>> Scenario: {scenario['name']} ({slug})")

        config = deepcopy(BASE_CONFIG)
        for key_path, value in scenario["mods"].items():
            set_nested(config, key_path, value)

        scenario_dir = EXPERIMENTS_DIR / slug

        # Clean prior root artifacts
        for artifact in ARTIFACTS:
            p = ROOT / artifact
            if p.exists():
                p.unlink()

        save_config(config, CONFIG_PATH)
        print("  Config applied. Running pipeline...")

        if not run_pipeline():
            results.append((slug, "FAILED"))
            continue

        collect_artifacts(scenario_dir, config)
        results.append((slug, "OK"))
        print(f"  Artifacts collected → {scenario_dir}")

    # Restore original config
    save_config(BASE_CONFIG, CONFIG_PATH)

    # ── Verification ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    all_ok = True
    for slug, _ in SCENARIOS.items():
        scenario_dir = EXPERIMENTS_DIR / slug
        missing = []
        for artifact in ARTIFACTS:
            if not (scenario_dir / artifact).exists():
                missing.append(artifact)
        config_snapshot = scenario_dir / "config.json"
        if not config_snapshot.exists():
            missing.append("config.json")
        if missing:
            print(f"  {slug}: MISSING {missing}")
            all_ok = False
        else:
            sizes = []
            for a in ARTIFACTS:
                p = scenario_dir / a
                sizes.append(f"{a} ({p.stat().st_size:,} bytes)")
            print(f"  {slug}: OK — {', '.join(sizes)} + config.json")

    print()
    if all_ok:
        print("All scenarios passed. Ready for commit.")
    else:
        print("Some scenarios failed — check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
