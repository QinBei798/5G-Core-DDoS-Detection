"""5G Core Network DDoS/Signaling Storm Detection System.

Orchestrates synthetic data generation, ML training, and evaluation plotting.
"""

import sys

from src.data_generator import load_config, generate_5g_signaling_data
from src.ml_pipeline import train_and_evaluate_models
from src.plotter import plot_roc_curve, plot_confusion_matrix


def main() -> None:
    """Run the full 5G Core DDoS detection pipeline."""
    try:
        # 1. Load configuration
        config = load_config()

        # 2. Generate synthetic 5G signaling data
        X, y = generate_5g_signaling_data(config)

        # 3. Train and evaluate ML models
        results = train_and_evaluate_models(X, y, config)

        # 4. Build probability dict for ROC curve (attack-class probabilities)
        proba_dict = {
            "Random Forest": results["y_proba_rf"][:, 1],
            "SVM": results["y_proba_svm"][:, 1],
        }

        # 5. Generate evaluation charts
        plot_roc_curve(results["y_test"], proba_dict, config)
        plot_confusion_matrix(results["y_test"], results["y_pred_rf"], config)

        print("Done. Charts saved.")

    except Exception as e:
        print(f"Error: Pipeline failed — {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
