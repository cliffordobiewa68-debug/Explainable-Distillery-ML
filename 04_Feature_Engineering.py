#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 11:34:06 2026

@author: cliffordobiewa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================
PROJECT:
Explainable Machine Learning Model for Distillery Effluent
Treatment and Management

MODULE:
04_Feature_Engineering.py
==============================================================
"""

###############################################################
# IMPORT LIBRARIES
###############################################################

import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

###############################################################
# CREATE PROJECT FOLDERS
###############################################################

os.makedirs("Dataset", exist_ok=True)
os.makedirs("Models", exist_ok=True)
os.makedirs("Results", exist_ok=True)

###############################################################
# LOAD CLEAN DATASET
###############################################################

df = pd.read_csv(
    "Dataset/distillery_clean.csv"
)

print("="*70)
print("FEATURE ENGINEERING")
print("="*70)

print(df.head())

###############################################################
# INPUT FEATURES
###############################################################

numerical_features = [

'Raw_pH',

'Raw_COD_mgL',

'Raw_BOD_mgL',

'Raw_TDS_mgL',

'Raw_TSS_mgL',

'Input_Lime_kg_m3',

'Input_Urea_kg_m3',

'Input_DAP_kg_m3',

'Output_Biogas_m3_m3'

]

###############################################################
# CATEGORICAL FEATURES
###############################################################

categorical_features = [

'Operating_Shift',

'Production_Level',

'Plant_Status',

'Season',

'Quarter'

]

###############################################################
# TARGET VARIABLES
###############################################################

target_variables = [

'Final_pH',

'Final_COD_mgL',

'Final_BOD_mgL',

'Final_TDS_mgL',

'Final_TSS_mgL'

]

###############################################################
# CREATE FEATURE MATRIX
###############################################################

X_numeric = df[numerical_features]

X_categorical = pd.get_dummies(

    df[categorical_features],

    drop_first=True

)

X = pd.concat(

    [

        X_numeric,

        X_categorical

    ],

    axis=1

)

###############################################################
# TARGET MATRIX
###############################################################

Y = df[target_variables]

###############################################################
# COMPLIANCE TARGET
###############################################################

y_class = df["Compliance"]

###############################################################
# DISPLAY FEATURES
###############################################################

print("="*70)
print("INPUT FEATURES")
print("="*70)

print(X.columns)

print("\nTotal Features:", len(X.columns))

print("="*70)
print("OUTPUT VARIABLES")
print("="*70)

print(Y.columns)

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

###############################################################
# CLASSIFICATION SPLIT
###############################################################

_, _, y_train_cls, y_test_cls = train_test_split(

    X,

    y_class,

    test_size=0.20,

    random_state=42,

    shuffle=True

)

###############################################################
# VERIFY DATA TYPES
###############################################################

print("\n")

print(type(X_train))
print(type(X_test))
print(type(Y_train))
print(type(Y_test))

###############################################################
# VERIFY SHAPES
###############################################################

print("\nTraining Shape")

print(X_train.shape)

print(Y_train.shape)

print("\nTesting Shape")

print(X_test.shape)

print(Y_test.shape)

###############################################################
# STANDARDIZATION
###############################################################

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(

    X_train

)

X_test_scaled = scaler.transform(

    X_test

)

###############################################################
# SAVE SCALER
###############################################################

joblib.dump(

    scaler,

    "Models/Scaler.pkl"

)

###############################################################
# SAVE ENGINEERED DATASETS
###############################################################

joblib.dump(

    X_train,

    "Dataset/X_train.pkl"

)

joblib.dump(

    X_test,

    "Dataset/X_test.pkl"

)

joblib.dump(

    Y_train,

    "Dataset/Y_train.pkl"

)

joblib.dump(

    Y_test,

    "Dataset/Y_test.pkl"

)

joblib.dump(

    X_train_scaled,

    "Dataset/X_train_scaled.pkl"

)

joblib.dump(

    X_test_scaled,

    "Dataset/X_test_scaled.pkl"

)

joblib.dump(

    y_train_cls,

    "Dataset/y_train_cls.pkl"

)

joblib.dump(

    y_test_cls,

    "Dataset/y_test_cls.pkl"

)

###############################################################
# FEATURE LIST
###############################################################

feature_list = pd.DataFrame({

    "Feature": X.columns

})

feature_list.to_excel(

    "Results/Feature_List.xlsx",

    index=False

)

###############################################################
# SUMMARY
###############################################################

print("="*70)
print("FEATURE ENGINEERING COMPLETED")
print("="*70)

print("Feature Matrix :", X.shape)

print("Training Set   :", X_train.shape)

print("Testing Set    :", X_test.shape)

print("Outputs        :", Y.shape)

print("Scaler Saved")

print("Feature List Saved")

print("="*70)
print("Overall Compliance")
print(df["Compliance"].value_counts())

print("\nTraining")
print(y_train_cls.value_counts())

print("\nTesting")
print(y_test_cls.value_counts())