import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNRegressor, TabPFNClassifier
from sklearn.preprocessing import LabelEncoder


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

# For support set
support['Unit'] = support[['Unit1', 'Unit2']].idxmax(axis=1)

# For query set
query['Unit'] = query[['Unit1', 'Unit2']].idxmax(axis=1)

support = support.drop(columns=['Unit1', 'Unit2'])
query = query.drop(columns=['Unit1', 'Unit2'])

# Features (drop Age and In-hospital_death from features)
feature_cols = [col for col in support.columns if col not in sens_cols+["SepsisLabel", "Unit"]]

label_encoder = LabelEncoder()

# Fit the encoder on the training data (support)
support['Unit'] = label_encoder.fit_transform(support['Unit'])

# Transform the query set
query['Unit'] = label_encoder.transform(query['Unit'])

# Drop rows with missing features
support = support.dropna(subset=['Unit', 'SepsisLabel'])


# Select support and query sets
x_support = support[feature_cols]
y_support = support['Unit']

x_query = query[feature_cols]
y_query = query['Unit']

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
predicted_unit_df = pd.DataFrame(
    pd.get_dummies(label_encoder.inverse_transform(y_pred), prefix='Unit'),
    index=query.index  # Align the index with the query set
)

# Step 3: Save the predictions to a CSV
output = pd.DataFrame({
    'Index': query.index
}).join(predicted_unit_df)

output.to_csv("y_pred_Unit10%.csv", index=False)

# --------------------------------
# 5. Evaluate (Mean Squared Error)
# --------------------------------
acc = accuracy_score(y_query, y_pred)
print(f"Accuracy on Unit query set: {acc:.4f}")
