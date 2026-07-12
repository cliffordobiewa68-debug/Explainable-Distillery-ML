#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 11:05:18 2026

@author: cliffordobiewa
"""

import pandas as pd
import numpy as np

###############################################################
# IMPORT LIBRARIES
###############################################################

import warnings
warnings.filterwarnings("ignore")

import os
import joblib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler

from sklearn.multioutput import MultiOutputRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Seed for absolute data consistency
np.random.seed(42)
records = 1000
dates = pd.date_range(start="2024-01-01", periods=records, freq="D")

# 1. Raw Spent Wash (Highly Acidic and Concentrated Organic Loading)
raw_ph = np.random.uniform(3.8, 4.5, records)
raw_cod = np.random.uniform(90000, 140000, records)
raw_bod = raw_cod * np.random.uniform(0.45, 0.55, records)
raw_tds = np.random.uniform(70000, 110000, records)
raw_tss = np.random.uniform(12000, 25000, records)

# 2. Cleanup Inputs & Resource Recovery (Lime for pH, Nutrients for Bioreactors)
lime_dosage = np.random.uniform(2.5, 4.0, records)  # kg needed per m3 
nutrient_urea = np.random.uniform(0.2, 0.5, records)  # kg/m3 for UASB bacterial health
nutrient_dap = np.random.uniform(0.05, 0.15, records) # Diammonium Phosphate
biogas_yield = raw_bod * np.random.uniform(0.005, 0.007, records) # m3 generated

# 3. Final Polished Effluent (After Anaerobic Digestion + Multi-Effect Evaporator + CPU)
final_ph = np.random.uniform(6.8, 7.8, records)
final_cod = raw_cod * np.random.uniform(0.0005, 0.001, records)  # 99.9% organic destruction
final_bod = final_cod * np.random.uniform(0.1, 0.2, records)
final_tds = np.random.uniform(50, 150, records)  # Clean condensed steam distillation water
final_tss = np.random.uniform(5, 15, records)

# Compiling into a unified engineering datasheet
df = pd.DataFrame({
    'Record_Date': dates.strftime('%Y-%m-%d'),
    'Raw_pH': np.round(raw_ph, 2),
    'Raw_COD_mgL': np.round(raw_cod, 0),
    'Raw_BOD_mgL': np.round(raw_bod, 0),
    'Raw_TDS_mgL': np.round(raw_tds, 0),
    'Raw_TSS_mgL': np.round(raw_tss, 0),
    'Input_Lime_kg_m3': np.round(lime_dosage, 2),
    'Input_Urea_kg_m3': np.round(nutrient_urea, 2),
    'Input_DAP_kg_m3': np.round(nutrient_dap, 2),
    'Output_Biogas_m3_m3': np.round(biogas_yield, 2),
    'Final_pH': np.round(final_ph, 2),
    'Final_COD_mgL': np.round(final_cod, 0),
    'Final_BOD_mgL': np.round(final_bod, 0),
    'Final_TDS_mgL': np.round(final_tds, 0),
    'Final_TSS_mgL': np.round(final_tss, 0)
})

# Save output to spreadsheet file
df.to_csv('distillery_spent_wash_1000_records.csv', index=False)
print("Success! 'distillery_spent_wash_1000_records.csv' has been generated with 1000 records.")

# Convert Record_Date to datetime
df['Record_Date'] = pd.to_datetime(df['Record_Date'])

# Month name
df['Month'] = df['Record_Date'].dt.month_name()

# Quarter
df['Quarter'] = df['Record_Date'].dt.quarter

# Season
df['Season'] = np.select(
    [
        df['Record_Date'].dt.month.isin([12,1,2]),
        df['Record_Date'].dt.month.isin([3,4,5]),
        df['Record_Date'].dt.month.isin([6,7,8]),
        df['Record_Date'].dt.month.isin([9,10,11])
    ],
    [
        'Dry Season',
        'Long Rains',
        'Cool Season',
        'Short Rains'
    ],
    default='Unknown'
)
np.random.seed(42)

df['Operating_Shift'] = np.random.choice(
['Morning', 'Afternoon', 'Night'],
size=len(df),
p=[0.40, 0.35, 0.25]
)
df['Production_Level'] = np.random.choice(
['Low', 'Medium', 'High'],
size=len(df),
p=[0.25, 0.50, 0.25]
)
df['Plant_Status'] = np.random.choice(
['Normal Operation', 'Maintenance'],
size=len(df),
p=[0.92, 0.08]
)
df['Month'] = df['Record_Date'].dt.month_name()
df['Quarter'] = df['Record_Date'].dt.quarter

print(df.head())
print("\nOperating Shift")
print(df['Operating_Shift'].value_counts())

print("\nProduction Level")
print(df['Production_Level'].value_counts())

print("\nPlant Status")
print(df['Plant_Status'].value_counts())

print("\nQuarter")
print(df['Quarter'].value_counts())

print("\nSeason")
print(df['Season'].value_counts())
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
df['Operating_Shift'].value_counts().plot(
    kind='bar',
    color=['steelblue','orange','green']
)
plt.title('Distribution of Operating Shift')
plt.xlabel('Operating Shift')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
plt.figure(figsize=(6,4))
df['Production_Level'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=90
)
plt.ylabel("")
plt.title('Production Capacity Level')
plt.tight_layout()
plt.show()
plt.figure(figsize=(6,4))
df['Plant_Status'].value_counts().plot(
    kind='bar',
    color=['green','red']
)

plt.title("Treatment Plant Status")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
plt.figure(figsize=(6,4))
df['Quarter'].value_counts().sort_index().plot(
    kind='bar',
    color='purple'
)

plt.title("Distribution by Quarter")
plt.xlabel("Quarter")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()
plt.figure(figsize=(6,4))
df['Season'].value_counts().plot(
    kind='bar',
    color='teal'
)

plt.title("Season Distribution")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
from scipy import stats

print("="*60)
print("DATA CLEANING AND PREPROCESSING")
print("="*60)
# ----------------------------------------------------------
# 1. Check Dataset Structure
# ----------------------------------------------------------
print("\nDataset Shape Before Cleaning")
print(df.shape)

print("\nData Types")
print(df.dtypes)
# ----------------------------------------------------------
# 2. Missing Values
# ----------------------------------------------------------

print("\nMissing Values Before Imputation")
print(df.isnull().sum())

# Median Imputation for Numeric Variables
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column].fillna(df[column].median(), inplace=True)

print("\nMissing Values After Imputation")
print(df.isnull().sum())
# ----------------------------------------------------------
# 3. Duplicate Records
# ----------------------------------------------------------

duplicates = df.duplicated().sum()

print("\nDuplicate Records Found:", duplicates)

df = df.drop_duplicates()

print("Dataset Shape After Removing Duplicates")
print(df.shape)

# ----------------------------------------------------------
# 4. Consistency Checks
# ----------------------------------------------------------

df['Record_Date'] = pd.to_datetime(df['Record_Date'])

print("\nDate Format Verified")

print("\nChecking Numeric Columns")

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

print("Numeric Formatting Completed")

# ----------------------------------------------------------
# 5. Outlier Detection using IQR
# ----------------------------------------------------------

print("\nIQR Outlier Detection")

Q1 = df[numeric_columns].quantile(0.25)

Q3 = df[numeric_columns].quantile(0.75)

IQR = Q3 - Q1

outliers = ((df[numeric_columns] < (Q1 - 1.5 * IQR)) |
            (df[numeric_columns] > (Q3 + 1.5 * IQR)))

print(outliers.sum())

# ----------------------------------------------------------
# 6. Outlier Treatment (Winsorization)
# ----------------------------------------------------------

for column in numeric_columns:

    lower = Q1[column] - 1.5 * IQR[column]
    upper = Q3[column] + 1.5 * IQR[column]

    df[column] = np.where(df[column] < lower,
                          lower,
                          df[column])

    df[column] = np.where(df[column] > upper,
                          upper,
                          df[column])

print("\nOutliers Treated Using IQR Winsorization")

# ----------------------------------------------------------
# 7. Z-score Verification
# ----------------------------------------------------------

z_scores = np.abs(stats.zscore(df[numeric_columns]))

remaining_outliers = (z_scores > 3).sum()

print("\nRemaining Z-score Outliers")
print(pd.Series(remaining_outliers,
                index=numeric_columns))

# ----------------------------------------------------------
# 8. Final Dataset Validation
# ----------------------------------------------------------

print("\nFinal Dataset Shape")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum().sum())

print("\nDuplicate Records")
print(df.duplicated().sum())

print("\nData Cleaning Completed Successfully")
# ==========================================================
# 4.2.3 DESCRIPTIVE STATISTICS
# ==========================================================

import pandas as pd
import numpy as np

print("="*70)
print("DESCRIPTIVE STATISTICS OF THE STUDY VARIABLES")
print("="*70)

# Select numerical variables
numeric_columns = [
    'Raw_pH',
    'Raw_COD_mgL',
    'Raw_BOD_mgL',
    'Raw_TDS_mgL',
    'Raw_TSS_mgL',
    'Input_Lime_kg_m3',
    'Input_Urea_kg_m3',
    'Input_DAP_kg_m3',
    'Output_Biogas_m3_m3',
    'Final_pH',
    'Final_COD_mgL',
    'Final_BOD_mgL',
    'Final_TDS_mgL',
    'Final_TSS_mgL'
]

# Generate descriptive statistics
descriptive_statistics = df[numeric_columns].describe().T

# Add additional statistics
descriptive_statistics['Median'] = df[numeric_columns].median()
descriptive_statistics['Variance'] = df[numeric_columns].var()
descriptive_statistics['Skewness'] = df[numeric_columns].skew()
descriptive_statistics['Kurtosis'] = df[numeric_columns].kurt()

# Arrange columns
descriptive_statistics = descriptive_statistics[
    ['count',
     'mean',
     'Median',
     'std',
     'Variance',
     'min',
     '25%',
     '50%',
     '75%',
     'max',
     'Skewness',
     'Kurtosis']
]

print(descriptive_statistics)

# Save to Excel
descriptive_statistics.to_excel(
    "Table_4_Descriptive_Statistics.xlsx"
)

print("\nDescriptive statistics saved successfully.")
# ==========================================================
# CORRELATION MATRIX
# ==========================================================

import matplotlib.pyplot as plt
import seaborn as sns

correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(14,10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f"
)
plt.title("Correlation Matrix of Study Variables")

plt.tight_layout()

plt.show()
# ==========================================================
# HISTOGRAMS
# ==========================================================

df[numeric_columns].hist(
    figsize=(18,14),
    bins=20
)

plt.tight_layout()

plt.show()
# ==========================================================
# BOXPLOTS
# ==========================================================

for variable in numeric_columns:

    plt.figure(figsize=(6,4))

    plt.boxplot(df[variable])

    plt.title(variable)

    plt.ylabel(variable)

    plt.show()
    # ==========================================================
# COEFFICIENT OF VARIATION
# ==========================================================

cv = (df[numeric_columns].std()/df[numeric_columns].mean())*100

cv = cv.round(2)

print("\nCoefficient of Variation (%)")

print(cv)
print(df.head())
print(df.shape)
# ==========================================================
# FEATURE SELECTION
# ==========================================================

features = [

'Raw_pH',

'Raw_COD_mgL',

'Raw_BOD_mgL',

'Raw_TDS_mgL',

'Raw_TSS_mgL',

'Input_Lime_kg_m3',

'Input_Urea_kg_m3',

'Input_DAP_kg_m3'

]
X = df[features]

###############################################################
# OUTPUT VARIABLES
###############################################################

Y=df[[

'Final_pH',

'Final_COD_mgL',

'Final_BOD_mgL',

'Final_TDS_mgL',

'Final_TSS_mgL'

]]
print(Y.head())
###############################################################
# COMPLIANCE CLASS
###############################################################

df['Compliance']=np.where(

(

(df['Final_pH']>=6.5)&

(df['Final_pH']<=8.5)&

(df['Final_COD_mgL']<=100)&

(df['Final_BOD_mgL']<=30)&

(df['Final_TSS_mgL']<=30)

),

1,

0

)

print(df['Compliance'].value_counts())
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
    cross_val_score
)
(
    X_train,
    X_test,
    Y_train,
    Y_test
) = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42
)
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)
import warnings
warnings.filterwarnings('ignore')

import os
###############################################################
# COMPLIANCE CLASS
###############################################################

df['Compliance']=np.where(

(

(df['Final_pH']>=6.5)&

(df['Final_pH']<=8.5)&

(df['Final_COD_mgL']<=100)&

(df['Final_BOD_mgL']<=30)&

(df['Final_TSS_mgL']<=30)

),

1,

0

)

print(df['Compliance'].value_counts())
###############################################################
# TRAIN TEST SPLIT
###############################################################

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

print(type(X_train))
print(type(X_test))
print(type(Y_train))
print(type(Y_test))

print(X_train.shape)
print(Y_test.shape)
###############################################################
# STANDARDIZATION
###############################################################

from sklearn.preprocessing import StandardScaler
import joblib
scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(

X_train

)

X_test_scaled=scaler.transform(

X_test

)
import os

os.makedirs("Models", exist_ok=True)
joblib.dump(

scaler,

'Models/Scaler.pkl'

)
###############################################################
# CROSS VALIDATION
###############################################################

cv=KFold(

n_splits=10,

shuffle=True,

random_state=42

)
###############################################################
# DATA SUMMARY
###############################################################

print("="*60)

print("INPUT VARIABLES")

print("="*60)

print(X.columns)

print("="*60)

print("OUTPUT VARIABLES")

print("="*60)

print(Y.columns)

print("="*60)

print("TRAINING OBSERVATIONS")

print(len(X_train))

print("="*60)

print("TEST OBSERVATIONS")

print(len(X_test))
###############################################################
# RANDOM FOREST MULTI-OUTPUT REGRESSION
###############################################################

print("="*70)
print("TRAINING RANDOM FOREST MODEL")
print("="*70)

rf_model = MultiOutputRegressor(

    RandomForestRegressor(

        n_estimators=300,

        random_state=42,

        max_depth=20,

        min_samples_split=2,

        min_samples_leaf=1,

        n_jobs=-1

    )

)

rf_model.fit(X_train, Y_train)

print("Random Forest Model Successfully Trained.")
###############################################################
# PREDICTIONS
###############################################################

rf_predictions = rf_model.predict(X_test)

rf_predictions = pd.DataFrame(

    rf_predictions,

    columns=Y.columns

)

print(rf_predictions.head())
###############################################################
# SAVE PREDICTIONS
###############################################################
import os

# Create output folders if they don't exist
os.makedirs("Results", exist_ok=True)
os.makedirs("Figures", exist_ok=True)
os.makedirs("Models", exist_ok=True)

print("Folders created successfully.")
rf_predictions.to_excel(

    "Results/RandomForest_Predictions.xlsx",

    index=False

)

print("Predictions Saved.")
print(type(Y_test))
print(type(rf_predictions))

print(Y_test)
Y_test = pd.DataFrame(
    Y_test,
    columns=Y.columns
)

rf_predictions = pd.DataFrame(
    rf_predictions,
    columns=Y.columns
)

print(type(Y_test))
print(type(rf_predictions))
###############################################################
# MODEL EVALUATION
###############################################################
###############################################################
# MODEL EVALUATION
###############################################################




    