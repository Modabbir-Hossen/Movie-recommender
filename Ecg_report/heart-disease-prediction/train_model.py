"""
train_model.py
--------------
Run this script to train all models, evaluate them, save the best one,
and generate evaluation plots.

Usage:
    python train_model.py                       # uses data/heart_disease.csv
    python train_model.py --data path/to/data.csv
"""

import argparse
import sys
import os

# Ensure project root is on the path (works on Windows + Mac + Linux)
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.preprocess import load_data, clean_data, get_splits, FEATURE_COLS
from src.train import (
    cross_validate_all, train_best, save_artifacts,
    plot_cv_comparison, plot_confusion_matrix,
    plot_roc_curve, plot_feature_importance
)


def main():
    parser = argparse.ArgumentParser(description="Train heart disease classifiers")
    parser.add_argument("--data",   default="data/heart_disease.csv", help="Path to CSV dataset")
    parser.add_argument("--models", default="models",                 help="Directory to save model artifacts")
    args = parser.parse_args()

    print("\n🫀  Heart Disease Prediction — Training Pipeline")
    print("=" * 52)

    # 1. Load & clean
    print("\n[1/5] Loading and cleaning data …")
    df = load_data(args.data)
    df = clean_data(df)
    print(f"      {len(df)} samples | {df['target'].mean()*100:.1f}% positive class")

    # 2. Split
    print("\n[2/5] Splitting and scaling features …")
    X_train, X_test, y_train, y_test, scaler = get_splits(df)
    print(f"      Train: {len(X_train)}  |  Test: {len(X_test)}")

    # 3. Cross-validate all models
    print("\n[3/5] Cross-validating all classifiers …")
    cv_results = cross_validate_all(X_train, y_train)

    # 4. Train best model
    print("\n[4/5] Training best model on full train set …")
    best_name, best_model, metrics, y_pred, y_proba = train_best(
        X_train, y_train, X_test, y_test, cv_results
    )
    print(f"\n  ✅ Best Model  : {best_name}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  F1-Score       : {metrics['f1']:.4f}")
    print(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")

    # 5. Generate plots
    print("\n[5/5] Generating evaluation plots …")
    plot_cv_comparison(cv_results)
    plot_confusion_matrix(metrics["cm"])
    plot_roc_curve(y_test, y_proba, metrics["roc_auc"])
    plot_feature_importance(best_model, FEATURE_COLS)
    print("      Plots saved to static/plots/")

    # Save artifacts
    save_artifacts(best_model, scaler, metrics, model_dir=args.models)

    print("\n✅  Pipeline complete. Run `python app.py` to start the web app.\n")


if __name__ == "__main__":
    main()
