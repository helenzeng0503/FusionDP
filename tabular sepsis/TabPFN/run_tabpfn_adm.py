import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
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

support = support.dropna(subset=['HospAdmTime', 'SepsisLabel'])
query = query.dropna(subset=['HospAdmTime'])

feature_cols = [c for c in support.columns if c not in sens_cols + ['SepsisLabel']]

# ---------------------------
# 2. Clip and standardize HospAdmTime
# ---------------------------
support['HospAdmTime_clipped'] = support['HospAdmTime'].clip(-500, 25)
query['HospAdmTime_clipped'] = query['HospAdmTime'].clip(-500, 25)

scaler = StandardScaler()
support['HospAdmTime_z'] = scaler.fit_transform(support[['HospAdmTime_clipped']])
query['HospAdmTime_z'] = scaler.transform(query[['HospAdmTime_clipped']])

# ---------------------------
# 3. Equal-width binning on z-scores
# ---------------------------
n_bins = 10
support['HospAdmTime_bin'], bins = pd.cut(
    support['HospAdmTime_z'], bins=n_bins, labels=False, retbins=True, include_lowest=True
)
query['HospAdmTime_bin'] = pd.cut(
    query['HospAdmTime_z'], bins=bins, labels=False, include_lowest=True
)

bin_centers = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(n_bins)])

# ---------------------------
# 4. TabPFNClassifier with softmax output
# ---------------------------
clf = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
clf.fit(support[feature_cols].to_numpy(), support['HospAdmTime_bin'].astype(int).to_numpy())

# Use softmax output (probability per bin)
probs = clf.predict_proba(query[feature_cols].to_numpy())  # shape: (N, n_bins)

# Soft prediction in z-score space
y_bin_z = np.dot(probs, bin_centers)  # shape: (N,)
y_bin = scaler.inverse_transform(y_bin_z.reshape(-1, 1)).flatten()

# ---------------------------
# 5. TabPFNRegressor (optional)
# ---------------------------
reg = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu')
reg.fit(support[feature_cols], support['HospAdmTime_z'])
y_reg_z = reg.predict(query[feature_cols])
y_reg = scaler.inverse_transform(y_reg_z.reshape(-1, 1)).flatten()

# ---------------------------
# 6. Blending
# ---------------------------
alpha = 0.7
y_blend = alpha * y_reg + (1 - alpha) * y_bin
y_true = query['HospAdmTime'].values

# ---------------------------
# 7. Save & Evaluate
# ---------------------------
output = pd.DataFrame({
    'Index': query.index,
    'y_true_clipped': y_true,
    'y_bin': y_bin,
    'y_reg': y_reg,
    'HospAdmTime': y_blend
})
output.to_csv("y_pred_HospAdmTime10%.csv", index=False)

print(f"MSE (bin-proba weighted): {mean_squared_error(y_true, y_bin):.4f}")
print(f"MSE (reg only):           {mean_squared_error(y_true, y_reg):.4f}")
print(f"MSE (blend α={alpha}):     {mean_squared_error(y_true, y_blend):.4f}")
