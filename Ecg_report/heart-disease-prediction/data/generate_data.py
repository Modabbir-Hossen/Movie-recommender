"""
Generate synthetic heart disease dataset based on UCI Cleveland Heart Disease features.
Run this once to create sample data for training.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

np.random.seed(42)

N = 1025  # samples

age         = np.random.randint(29, 77, N)
sex         = np.random.randint(0, 2, N)          # 0=female, 1=male
cp          = np.random.randint(0, 4, N)           # chest pain type 0-3
trestbps    = np.random.randint(90, 200, N)        # resting BP
chol        = np.random.randint(126, 565, N)       # cholesterol
fbs         = np.random.randint(0, 2, N)           # fasting blood sugar >120 mg/dl
restecg     = np.random.randint(0, 3, N)           # resting ECG 0-2
thalach     = np.random.randint(71, 202, N)        # max heart rate
exang       = np.random.randint(0, 2, N)           # exercise induced angina
oldpeak     = np.round(np.random.uniform(0, 6.2, N), 1)  # ST depression
slope       = np.random.randint(0, 3, N)           # slope of peak exercise ST
ca          = np.random.randint(0, 5, N)           # major vessels colored by fluoroscopy
thal        = np.random.randint(0, 4, N)           # thalassemia type

# Create correlated target: older age + high cp + high chol → more likely disease
target_prob = (
    0.3 * (age > 55).astype(int)
    + 0.25 * (cp >= 2).astype(int)
    + 0.15 * (chol > 240).astype(int)
    + 0.15 * (exang == 1).astype(int)
    + 0.15 * (oldpeak > 2).astype(int)
)
target_prob = np.clip(target_prob / target_prob.max(), 0.05, 0.95)
target = (np.random.rand(N) < target_prob).astype(int)

df = pd.DataFrame({
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
    "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
    "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca,
    "thal": thal, "target": target
})

df.to_csv("heart_disease.csv", index=False)
print(f"Dataset saved → heart_disease.csv  ({N} rows, {df['target'].mean()*100:.1f}% positive)")
