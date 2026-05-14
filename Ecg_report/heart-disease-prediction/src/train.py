"""
src/train.py
------------
Train five classifiers, compare via cross-validation, persist the best model.
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics         import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, f1_score, accuracy_score
)

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM":                 SVC(probability=True, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=7),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
}

PALETTE = {
    "Logistic Regression": "#3b82f6",
    "Random Forest":       "#10b981",
    "SVM":                 "#f59e0b",
    "KNN":                 "#8b5cf6",
    "Decision Tree":       "#ef4444",
}


def cross_validate_all(X_train, y_train, cv_folds: int = 5):
    """Return dict of {model_name: mean_cv_accuracy}."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}
    for name, model in MODELS.items():
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
        results[name] = {"mean": scores.mean(), "std": scores.std(), "scores": scores}
        print(f"  {name:<22} CV Acc: {scores.mean():.4f} ± {scores.std():.4f}")
    return results


def train_best(X_train, y_train, X_test, y_test, cv_results: dict):
    """Fit the best model on full training set, return (model_name, model, metrics)."""
    best_name = max(cv_results, key=lambda k: cv_results[k]["mean"])
    best_model = MODELS[best_name]
    best_model.fit(X_train, y_train)

    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "f1":          round(f1_score(y_test, y_pred), 4),
        "roc_auc":     round(roc_auc_score(y_test, y_proba), 4),
        "cm":          confusion_matrix(y_test, y_pred).tolist(),
        "report":      classification_report(y_test, y_pred, output_dict=True),
        "best_model":  best_name,
        "cv_results":  {k: {"mean": float(v["mean"]), "std": float(v["std"])}
                        for k, v in cv_results.items()},
    }
    return best_name, best_model, metrics, y_pred, y_proba


def save_artifacts(model, scaler, metrics, model_dir: str = "models"):
    """Pickle model, scaler, and metrics dict."""
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "best_model.pkl"),  "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(model_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
    print(f"Artifacts saved to '{model_dir}/'")


def load_artifacts(model_dir: str = "models"):
    with open(os.path.join(model_dir, "best_model.pkl"),  "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_dir, "metrics.pkl"), "rb") as f:
        metrics = pickle.load(f)
    return model, scaler, metrics


# ─── Plotting helpers ────────────────────────────────────────────────────────

def plot_cv_comparison(cv_results: dict, save_path: str = "static/plots/cv_comparison.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    names  = list(cv_results.keys())
    means  = [cv_results[n]["mean"] for n in names]
    stds   = [cv_results[n]["std"]  for n in names]
    colors = [PALETTE[n] for n in names]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    bars = ax.barh(names, means, xerr=stds, color=colors, height=0.55,
                   error_kw=dict(ecolor="#94a3b8", capsize=4, lw=1.5), zorder=2)
    for bar, val in zip(bars, means):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=9,
                color="white", fontweight="bold")

    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Accuracy", color="#94a3b8", fontsize=10)
    ax.set_title("5-Fold CV Accuracy Comparison", color="white", fontsize=13, pad=12)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.xaxis.grid(True, color="#334155", linestyle="--", lw=0.7, zorder=1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def plot_confusion_matrix(cm: list, save_path: str = "static/plots/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    im = ax.imshow(cm_arr, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    fontsize=20, fontweight="bold",
                    color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "#1e3a5f")

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Disease", "Disease"], color="#94a3b8")
    ax.set_yticklabels(["No Disease", "Disease"], color="#94a3b8")
    ax.set_xlabel("Predicted", color="#94a3b8"); ax.set_ylabel("Actual", color="#94a3b8")
    ax.set_title("Confusion Matrix", color="white", fontsize=12, pad=10)
    for spine in ax.spines.values(): spine.set_edgecolor("#334155")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def plot_roc_curve(y_test, y_proba, auc_score: float,
                   save_path: str = "static/plots/roc_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    ax.plot(fpr, tpr, color="#3b82f6", lw=2.5, label=f"AUC = {auc_score:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#3b82f6")
    ax.plot([0, 1], [0, 1], "--", color="#475569", lw=1.2)

    ax.set_xlabel("False Positive Rate", color="#94a3b8")
    ax.set_ylabel("True Positive Rate",  color="#94a3b8")
    ax.set_title("ROC-AUC Curve", color="white", fontsize=12, pad=10)
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor="#334155", edgecolor="none", labelcolor="white", fontsize=10)
    for spine in ax.spines.values(): spine.set_edgecolor("#334155")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def plot_feature_importance(model, feature_names: list,
                            save_path: str = "static/plots/feature_importance.png"):
    """Only for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    importances = model.feature_importances_
    idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    colors = plt.cm.Blues(np.linspace(0.4, 0.95, len(idx)))
    ax.barh([feature_names[i] for i in idx], importances[idx], color=colors, height=0.6)
    ax.set_xlabel("Importance", color="#94a3b8")
    ax.set_title("Feature Importances", color="white", fontsize=12, pad=10)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.xaxis.grid(True, color="#334155", linestyle="--", lw=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
