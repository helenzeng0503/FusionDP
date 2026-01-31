import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from tabpfn import TabPFNClassifier, TabPFNRegressor

# ---------------------------
# 1. Load data
# ---------------------------
sens_cols = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
acols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BaseExcess',
       'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
       'Calcium', 'Chloride', 'Creatinine', 'Glucose', 'Lactate', 'Magnesium',
       'Phosphate', 'Potassium', 'Bilirubin_total', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS',
        'SepsisLabel']

support = pd.read_csv("../sepsis_all_support_10%.csv")[acols]
query = pd.read_csv("../sepsis_all_query_10%.csv")[acols]

support = support.dropna(subset=['Age', 'SepsisLabel'])
query = query.dropna(subset=['Age'])

feature_cols = [c for c in support.columns if c not in sens_cols + ['SepsisLabel']]

# ---------------------------
# 2. Equal-width binning (10 bins)
# ---------------------------
n_bins = 10
support['Age_bin'], bins = pd.qcut(
    support['Age'], q=10, labels=False, retbins=True, duplicates='drop'
)
query['Age_bin'] = pd.cut(
    query['Age'], bins=bins, labels=False, include_lowest=True
)


# Bin centers
bin_centers = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(n_bins)])

# ---------------------------
# 3. TabPFNClassifier
# ---------------------------
clf = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
clf.fit(support[feature_cols].to_numpy(), support['Age_bin'].astype(int).to_numpy())

probs = clf.predict_proba(query[feature_cols].to_numpy())  # shape: (N, n_bins)
y_bin = np.dot(probs, bin_centers)

# ---------------------------
# 4. TabPFNRegressor
# ---------------------------
reg = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu')
reg.fit(support[feature_cols], support['Age'])
y_reg = reg.predict(query[feature_cols])

# ---------------------------
# 5. Blend outputs
# ---------------------------
alpha = 0.5
y_blend = alpha * y_reg + (1 - alpha) * y_bin
y_true = query['Age'].values

# ---------------------------
# 6. Save & Evaluate
# ---------------------------
output = pd.DataFrame({
    'Index': query.index,
    'y_true': y_true,
    'y_bin': y_bin,
    'y_reg': y_reg,
    'y_blend': y_blend
})
output.to_csv("y_pred_Age_10%.csv", index=False)

print(f"MSE (bin-proba weighted): {mean_squared_error(y_true, y_bin):.4f}")
print(f"MSE (reg only):           {mean_squared_error(y_true, y_reg):.4f}")
print(f"MSE (blend α={alpha}):     {mean_squared_error(y_true, y_blend):.4f}")
