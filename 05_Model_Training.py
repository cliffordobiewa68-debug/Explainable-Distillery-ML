#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 11:36:15 2026

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
05_Model_Training.py
==============================================================
"""

###############################################################
# IMPORT LIBRARIES
###############################################################

import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.multioutput import MultiOutputRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import (
    KFold,
    cross_val_score
)

###############################################################
# CREATE FOLDERS
###############################################################

os.makedirs("Models", exist_ok=True)
os.makedirs("Results", exist_ok=True)

###############################################################
# LOAD DATA
###############################################################

X_train = joblib.load("Dataset/X_train.pkl")
X_test = joblib.load("Dataset/X_test.pkl")

Y_train = joblib.load("Dataset/Y_train.pkl")
Y_test = joblib.load("Dataset/Y_test.pkl")

X_train_scaled = joblib.load("Dataset/X_train_scaled.pkl")
X_test_scaled = joblib.load("Dataset/X_test_scaled.pkl")

print("="*70)
print("MODEL TRAINING")
print("="*70)

###############################################################
# CROSS VALIDATION
###############################################################

cv = KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

###############################################################
# RANDOM FOREST
###############################################################

print("\nTraining Random Forest...")

rf_model = MultiOutputRegressor(

    RandomForestRegressor(

        n_estimators=300,

        max_depth=20,

        min_samples_split=2,

        min_samples_leaf=1,

        random_state=42,

        n_jobs=-1

    )

)

rf_model.fit(
    X_train,
    Y_train
)

rf_predictions = pd.DataFrame(

    rf_model.predict(X_test),

    columns=Y_train.columns,

    index=Y_test.index

)

joblib.dump(

    rf_model,

    "Models/RandomForest_Model.pkl"

)

rf_predictions.to_excel(

    "Results/RandomForest_Predictions.xlsx",

    index=False

)

print("Random Forest Completed.")

###############################################################
# RANDOM FOREST CROSS VALIDATION
###############################################################

rf_cv = cross_val_score(

    RandomForestRegressor(

        n_estimators=300,

        random_state=42

    ),

    X_train,

    Y_train.iloc[:,0],

    cv=cv,

    scoring="r2"

)

###############################################################
# GRADIENT BOOSTING
###############################################################

print("\nTraining Gradient Boosting...")

gb_model = MultiOutputRegressor(

    GradientBoostingRegressor(

        random_state=42,

        n_estimators=300,

        learning_rate=0.05,

        max_depth=4

    )

)

gb_model.fit(

    X_train,

    Y_train

)

gb_predictions = pd.DataFrame(

    gb_model.predict(X_test),

    columns=Y_train.columns,

    index=Y_test.index

)

joblib.dump(

    gb_model,

    "Models/GradientBoosting_Model.pkl"

)

gb_predictions.to_excel(

    "Results/GradientBoosting_Predictions.xlsx",

    index=False

)

print("Gradient Boosting Completed.")

###############################################################
# GRADIENT BOOSTING CROSS VALIDATION
###############################################################

gb_cv = cross_val_score(

    GradientBoostingRegressor(

        random_state=42

    ),

    X_train,

    Y_train.iloc[:,0],

    cv=cv,

    scoring="r2"

)

###############################################################
# LINEAR REGRESSION
###############################################################


print("\nTraining Linear Regression...")

lr_model = MultiOutputRegressor(

    LinearRegression()

)

lr_model.fit(

    X_train,

    Y_train

)

lr_predictions = pd.DataFrame(

    lr_model.predict(X_test),

    columns=Y_train.columns,

    index=Y_test.index

)

joblib.dump(

    lr_model,

    "Models/LinearRegression_Model.pkl"

)

lr_predictions.to_excel(

    "Results/LinearRegression_Predictions.xlsx",

    index=False

)

print("Linear Regression Completed.")

###############################################################
# LINEAR REGRESSION CROSS VALIDATION
###############################################################

lr_cv = cross_val_score(

    LinearRegression(),

    X_train,

    Y_train.iloc[:, 0],

    cv=cv,

    scoring="r2"

)

###############################################################
# CROSS VALIDATION SUMMARY
###############################################################

cv_results = pd.DataFrame({

    "Model": [

        "Linear Regression",

        "Random Forest",

        "Gradient Boosting"

    ],

    "Mean_R2": [

        lr_cv.mean(),

        rf_cv.mean(),

        gb_cv.mean()

    ],

    "Std": [

        lr_cv.std(),

        rf_cv.std(),

        gb_cv.std()

    ]

})

cv_results.to_excel(

    "Results/CrossValidation_Results.xlsx",

    index=False

)
###############################################################
# TRAINING SUMMARY
###############################################################

print("="*70)

print("ALL MODELS TRAINED SUCCESSFULLY")

print("="*70)

print(cv_results)

print("="*70)