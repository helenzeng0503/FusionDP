import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNRegressor, TabPFNClassifier


import sys
import os

# Add src/ to path if needed
# sys.path.append(os.path.abspath("./tabpfn-extensions/src"))
# from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor

# --------------------------------
# 1. Load and prepare data
# --------------------------------
sens_cols = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
acols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BaseExcess',
       'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
       'Calcium', 'Chloride', 'Creatinine', 'Glucose', 'Lactate', 'Magnesium',
       'Phosphate', 'Potassium', 'Bilirubin_total', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS',
        'SepsisLabel']

# Load dataset
support = pd.read_csv("../sepsis_all_support_10%.csv")[acols]
query = pd.read_csv("../sepsis_all_query_10%.csv")[acols]


# Features (drop Age and In-hospital_death from features)
feature_cols = [col for col in support.columns if col not in sens_cols+["SepsisLabel"]]

# Drop rows with missing features
support = support.dropna(subset=['Gender', 'SepsisLabel'])


# Select support and query sets
x_support = support[feature_cols]
y_support = support['Gender']

x_query = query[feature_cols]
y_query = query['Gender']

print(f"Support samples: {len(x_support)}, Query samples: {len(x_query)}")

# --------------------------------
# 3. Run TabPFNRegressor
# --------------------------------
model = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
# model = AutoTabPFNRegressor(max_time=120, device="cuda")  # Uncomment if using AutoTabPFN extensions

# Fit on support set
model.fit(x_support, y_support)

# Predict Age for query set
y_pred = model.predict(x_query)

# --------------------------------
# 4. Save predictions
# --------------------------------
output = pd.DataFrame({
    'Index': x_query.index,
    'Predicted_Gender': y_pred
})
output.to_csv("y_pred_Gender10%.csv", index=False)

# --------------------------------
# 5. Evaluate (Mean Squared Error)
# --------------------------------
acc = accuracy_score(y_query, y_pred)
print(f"Accuracy on Gender query set: {acc:.4f}")

