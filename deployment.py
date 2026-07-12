#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 10:42:04 2026

@author: cliffordobiewa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================
PROJECT:
Explainable Machine Learning Model for Distillery
Effluent Treatment and Management

MODULE:
08_Model_Deployment.py

DESCRIPTION
--------------------------------------------------------------
Deployment backend.

This module loads the trained model and scaler,
accepts new observations,
generates predictions,
and returns results to the Streamlit application.

Author:
Clifford Otieno Obiewa
Kabarak University
==============================================================
"""

###############################################################
# IMPORT LIBRARIES
###############################################################

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

###############################################################
# PROJECT PATHS
###############################################################

BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "Dataset"

MODEL_DIR = BASE_DIR / "Models"

###############################################################
# LOAD MODEL
###############################################################

MODEL_FILE = MODEL_DIR / "Best_Model.pkl"

SCALER_FILE = MODEL_DIR / "Scaler.pkl"

print("=" * 60)
print("Loading Deployment Resources")
print("=" * 60)

model = joblib.load(MODEL_FILE)

scaler = joblib.load(SCALER_FILE)

print("✓ Best model loaded")

print("✓ Scaler loaded")
###############################################################
# LOAD TRAINING FEATURES
###############################################################

X_train = joblib.load(DATA_DIR / "X_train.pkl")

if isinstance(X_train, pd.DataFrame):

    FEATURE_NAMES = X_train.columns.tolist()

else:

    FEATURE_NAMES = [

        f"Feature_{i+1}"

        for i in range(X_train.shape[1])

    ]

print(f"\nNumber of Input Features : {len(FEATURE_NAMES)}")

print("\nFeature List")

for feature in FEATURE_NAMES:

    print(" -", feature)
    
    ###############################################################
# TARGET VARIABLES
###############################################################

Y_train = joblib.load(DATA_DIR / "Y_train.pkl")

TARGET_NAMES = list(Y_train.columns)

print("\nPredicted Variables")

for target in TARGET_NAMES:

    print(" -", target)
    
    ###############################################################
# PREDICTION FUNCTION
###############################################################

def predict_effluent(input_data):
    """
    Predict treated effluent quality.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    dict
    """

    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURE_NAMES
    )

    scaled = scaler.transform(input_df)

    prediction = model.predict(scaled)[0]

    results = {}

    for target, value in zip(TARGET_NAMES, prediction):

        results[target] = round(float(value), 3)

    return results

###############################################################
# TEST
###############################################################

if __name__ == "__main__":

    print("\nRunning Deployment Test")

    sample = {}

    for feature in FEATURE_NAMES:

        sample[feature] = float(X_train.iloc[0][feature])

    predictions = predict_effluent(sample)

    print("\nPredictions")

    for target, value in predictions.items():

        print(f"{target:20s}: {value}")