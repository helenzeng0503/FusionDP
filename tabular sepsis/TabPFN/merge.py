import pandas as pd
import numpy as np


# Imputed predictions
y_pred_age = pd.read_csv("y_pred_Age_10%.csv")  # should have columns: Index, Predicted_Age
y_pred_gender = pd.read_csv("y_pred_Gender10%.csv")  # columns: Index, Predicted_Gender
y_pred_unit = pd.read_csv("y_pred_Unit10%.csv")  # columns: Index, unit (0-3)
y_pred_adm = pd.read_csv("y_pred_HospAdmTime10%.csv")
y_pred_iculos = pd.read_csv("y_pred_ICULOS_10%.csv")

# 2. Split support and query sets
df_query = pd.read_csv("../sepsis_all_query_10%.csv")

# 3. Merge imputations into query set

# Merge Age
df_query['Age'] = df_query['Age'].astype(float)
df_query.loc[y_pred_age['Index'].values, 'Age'] = y_pred_age['y_blend'].values

# Merge Gender
df_query.loc[y_pred_gender['Index'].values, 'Gender'] = y_pred_gender['Predicted_Gender'].values

# Merge HospAdmTime
df_query.loc[y_pred_adm['Index'].values, 'HospAdmTime'] = y_pred_adm['HospAdmTime'].values

# # Merge ICULOS
df_query.loc[y_pred_iculos['Index'].values, 'ICULOS'] = y_pred_iculos['ICULOS'].values


# # Drop old ICUType one-hot columns
df_query = df_query.drop(columns=['Unit1', 'Unit2'], errors='ignore')

# Concatenate imputed one-hot ICUType
df_query[['Unit1', 'Unit2']] = y_pred_unit[['Unit_Unit1', 'Unit_Unit2']].astype(int)

# 4. Save everything

df_query.reset_index().to_csv("sepsis_train_tabpfn_imputed.csv", index=False)

print("✅ Saved:")
print("- icuab_train_tabpfn_imputed.csv (with imputed Age, Gender, Unit, HospAdmTime, ICULOS)")
