# 🫀 Heart Disease Prediction System from ECG Reports

> End-to-end Machine Learning pipeline for classifying heart disease risk from ECG/clinical data — targeting early clinical diagnosis and decision support.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3+-000000?style=flat&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📌 Project Overview

This project builds a full ML pipeline to predict the risk of heart disease using clinical and ECG-derived features. The system:

- Preprocesses raw ECG/clinical data (imputation, outlier removal, scaling)
- Trains and compares **5 classifiers** via stratified cross-validation
- Selects the best model automatically and evaluates on a held-out test set
- Serves predictions through a **Flask web application** with a clinical-grade UI

**Project Period:** Jan 2026 – Feb 2026

---

## 📊 Results

| Metric      | Score  |
|-------------|--------|
| Accuracy    | 91%    |
| F1-Score    | 0.89   |
| ROC-AUC     | ~0.93  |
| Best Model  | Random Forest |

Evaluated using: Confusion Matrix · ROC-AUC Curve · Classification Report

---

## 🧠 ML Pipeline

```
Raw CSV Data
    │
    ▼
[1] Data Loading & Validation          ← src/preprocess.py
    │  Missing-value imputation (median)
    │  IQR-based outlier removal
    │
    ▼
[2] Feature Scaling                    ← StandardScaler
    │  Train/Test split (80/20, stratified)
    │
    ▼
[3] Cross-Validation (5-Fold StratifiedKFold)
    │  Logistic Regression
    │  Random Forest          ← Best model
    │  SVM (RBF kernel)
    │  KNN (k=7)
    │  Decision Tree (max_depth=6)
    │
    ▼
[4] Final Evaluation on Test Set
    │  Confusion Matrix  |  ROC-AUC Curve
    │  Classification Report  |  Feature Importances
    │
    ▼
[5] Flask Web App (/predict endpoint)
```

---

## 🗂 Project Structure

```
heart-disease-prediction/
├── app.py                  # Flask web application
├── train_model.py          # Main training script
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── preprocess.py       # Data loading, cleaning, scaling
│   └── train.py            # Model training, evaluation, plotting
│
├── data/
│   ├── generate_data.py    # Synthetic dataset generator
│   └── heart_disease.csv   # Dataset (13 features + target)
│
├── models/                 # Saved model artifacts (gitignored)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── metrics.pkl
│
├── static/
│   └── plots/              # Auto-generated evaluation plots
│       ├── cv_comparison.png
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── feature_importance.png
│
└── templates/
    └── index.html          # Web UI
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Modabbir-Hossen/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
```

### 2. Generate Data (or bring your own CSV)

```bash
python data/generate_data.py
```

Or use your own dataset — ensure it has these columns:

```
age, sex, cp, trestbps, chol, fbs, restecg,
thalach, exang, oldpeak, slope, ca, thal, target
```

### 3. Train Models

```bash
python train_model.py
```

Output:
- Trained model → `models/best_model.pkl`
- Scaler        → `models/scaler.pkl`
- Plots         → `static/plots/*.png`

### 4. Launch Web App

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## 📋 Features (ECG Parameters)

| Feature    | Description                          | Type        |
|------------|--------------------------------------|-------------|
| `age`      | Patient age in years                 | Numeric     |
| `sex`      | Sex (1=Male, 0=Female)               | Binary      |
| `cp`       | Chest pain type (0–3)                | Categorical |
| `trestbps` | Resting blood pressure (mm Hg)       | Numeric     |
| `chol`     | Serum cholesterol (mg/dl)            | Numeric     |
| `fbs`      | Fasting blood sugar >120 mg/dl       | Binary      |
| `restecg`  | Resting ECG results (0–2)            | Categorical |
| `thalach`  | Maximum heart rate achieved          | Numeric     |
| `exang`    | Exercise-induced angina              | Binary      |
| `oldpeak`  | ST depression induced by exercise    | Numeric     |
| `slope`    | Slope of peak exercise ST segment    | Categorical |
| `ca`       | Number of major vessels (0–4)        | Categorical |
| `thal`     | Thalassemia type (0–3)               | Categorical |

---

## 🛠 Tech Stack

| Category        | Tools                                  |
|-----------------|----------------------------------------|
| Language        | Python 3.9+                            |
| ML Framework    | scikit-learn                           |
| Data Processing | pandas, NumPy                          |
| Visualization   | Matplotlib                             |
| Web Framework   | Flask                                  |
| Serialization   | pickle                                 |

---

## 📡 API Reference

### `POST /predict`

**Request body (JSON):**
```json
{
  "age": 55, "sex": 1, "cp": 2, "trestbps": 130,
  "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150,
  "exang": 1, "oldpeak": 2.3, "slope": 1, "ca": 0, "thal": 2
}
```

**Response:**
```json
{
  "prediction":  1,
  "label":       "Heart Disease Detected",
  "probability": 78.4,
  "risk_level":  "High",
  "model_used":  "Random Forest"
}
```

### `GET /api/metrics`

Returns full model evaluation metrics (accuracy, F1, AUC, confusion matrix, classification report).

---

## ⚠️ Disclaimer

This tool is intended for **educational and research purposes only**. It is not a certified medical device and should not be used as a substitute for professional clinical diagnosis. Always consult a qualified healthcare provider for medical decisions.

---

## 👤 Author

**Modabbir Hossen**
[GitHub](https://github.com/Modabbir-Hossen/heart-disease-prediction)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
