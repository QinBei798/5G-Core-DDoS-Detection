import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix


def plot_roc_curve(y_true: np.ndarray, proba_dict: dict, config: dict) -> None:
    """Plot and save ROC curve comparison of classifiers.

    Args:
        y_true: Ground truth labels (0=Normal, 1=Attack)
        proba_dict: {"Random Forest": attack_proba_rf, "SVM": attack_proba_svm}
        config: Configuration dictionary from config.json
    """
    for model_name, proba in proba_dict.items():
        if proba.ndim != 1:
            raise ValueError(
                f"proba_dict['{model_name}'] must be a 1D array of attack-class "
                f"probabilities ([:, 1] sliced), got shape {proba.shape}"
            )

    figsize = tuple(config["output"]["figsize"])
    dpi = config["output"]["figure_dpi"]
    save_path = config["output"]["roc_curve_path"]

    plt.figure(figsize=figsize, dpi=dpi)

    for model_name, proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve — 5G Core DDoS Detection")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.tight_layout()

    try:
        plt.savefig(save_path)
        print(f"[Plotter] ROC curve saved to {save_path}")
    except OSError as e:
        print(f"Error: Failed to save ROC curve to {save_path}: {e}")
        raise
    finally:
        plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred_rf: np.ndarray, config: dict) -> None:
    """Plot and save confusion matrix for Random Forest classifier.

    Args:
        y_true: Ground truth labels (0=Normal, 1=Attack)
        y_pred_rf: Random Forest predictions
        config: Configuration dictionary from config.json
    """
    figsize = tuple(config["output"]["figsize"])
    dpi = config["output"]["figure_dpi"]
    save_path = config["output"]["confusion_matrix_path"]

    cm = confusion_matrix(y_true, y_pred_rf)

    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix — Random Forest")
    plt.colorbar()

    classes = ["Normal", "Attack"]
    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Annotate cells with count and row-percentage
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            row_sum = cm[i].sum()
            count_str = f"{cm[i, j]}"
            pct_str = f"({cm[i, j] / row_sum:.1%})"
            text_color = 'white' if cm[i, j] > threshold else 'black'
            # Count in bold, centered
            plt.text(j, i - 0.15, count_str, ha='center', va='center',
                     color=text_color, fontweight='bold')
            # Percentage below
            plt.text(j, i + 0.15, pct_str, ha='center', va='center',
                     color=text_color)

    plt.tight_layout()

    try:
        plt.savefig(save_path)
        print(f"[Plotter] Confusion matrix saved to {save_path}")
    except OSError as e:
        print(f"Error: Failed to save confusion matrix to {save_path}: {e}")
        raise
    finally:
        plt.close()
