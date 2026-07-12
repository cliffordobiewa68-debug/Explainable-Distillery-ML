#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 12:14:06 2026

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
06_Model_Evaluation.py
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
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score
)

###############################################################
# CREATE OUTPUT FOLDERS
###############################################################

os.makedirs("Results", exist_ok=True)
os.makedirs("Figures", exist_ok=True)

###############################################################
# LOAD TEST DATA
###############################################################

X_test = joblib.load("Dataset/X_test.pkl")
Y_test = joblib.load("Dataset/Y_test.pkl")

###############################################################
# LOAD TRAINED MODELS
###############################################################

models = {

    "Linear Regression":
        joblib.load("Models/LinearRegression_Model.pkl"),

    "Random Forest":
        joblib.load("Models/RandomForest_Model.pkl"),

    "Gradient Boosting":
        joblib.load("Models/GradientBoosting_Model.pkl")

}

###############################################################
# MODEL EVALUATION
###############################################################

results = []

print("="*70)
print("MODEL EVALUATION")
print("="*70)

for model_name, model in models.items():

    predictions = model.predict(X_test)

    predictions = pd.DataFrame(
        predictions,
        columns=Y_test.columns,
        index=Y_test.index
    )

    print(f"\nEvaluating {model_name}")

    ###########################################################
    # EACH TARGET VARIABLE
    ###########################################################

    for variable in Y_test.columns:

        actual = Y_test[variable]
        pred = predictions[variable]

        mae = mean_absolute_error(actual, pred)

        rmse = np.sqrt(
            mean_squared_error(actual, pred)
        )

        r2 = r2_score(actual, pred)

        mape = mean_absolute_percentage_error(
            actual,
            pred
        ) * 100

        evs = explained_variance_score(
            actual,
            pred
        )

        results.append({

            "Model": model_name,
            "Variable": variable,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE (%)": mape,
            "Explained Variance": evs

        })

###############################################################
# SAVE RESULTS
###############################################################

evaluation_results = pd.DataFrame(results)

evaluation_results.to_excel(

    "Results/Model_Evaluation.xlsx",

    index=False

)

###############################################################
# OVERALL MODEL PERFORMANCE
###############################################################

summary = evaluation_results.groupby(

    "Model"

).agg({

    "MAE":"mean",

    "RMSE":"mean",

    "R2":"mean",

    "MAPE (%)":"mean",

    "Explained Variance":"mean"

}).reset_index()

summary.to_excel(

    "Results/Overall_Model_Performance.xlsx",

    index=False

)

###############################################################
# BEST MODEL
###############################################################

best_model = summary.loc[
    summary["R2"].idxmax()
]

best_model_name = best_model["Model"]

print("\n")
print("="*70)
print("BEST MODEL")
print("="*70)
print(best_model)
print("="*70)

###############################################################
# SAVE BEST MODEL
###############################################################

best_model_object = models[best_model_name]

joblib.dump(
    best_model_object,
    "Models/Best_Model.pkl"
)

print(f"\nBest model ({best_model_name}) saved to Models/Best_Model.pkl")

###############################################################
# BAR CHART OF R2
###############################################################

plt.figure(figsize=(8,5))

plt.bar(

    summary["Model"],

    summary["R2"]

)

plt.ylabel("Average R²")

plt.title("Average R² of Machine Learning Models")

plt.tight_layout()

plt.savefig(

    "Figures/Model_R2_Comparison.png",

    dpi=300

)

plt.close()

###############################################################
# BAR CHART OF RMSE
###############################################################

plt.figure(figsize=(8,5))

plt.bar(

    summary["Model"],

    summary["RMSE"]

)

plt.ylabel("Average RMSE")

plt.title("Average RMSE of Machine Learning Models")

plt.tight_layout()

plt.savefig(

    "Figures/Model_RMSE_Comparison.png",

    dpi=300

)

plt.close()

###############################################################
# BAR CHART OF MAE
###############################################################

plt.figure(figsize=(8,5))

plt.bar(

    summary["Model"],

    summary["MAE"]

)

plt.ylabel("Average MAE")

plt.title("Average MAE of Machine Learning Models")

plt.tight_layout()

plt.savefig(

    "Figures/Model_MAE_Comparison.png",

    dpi=300

)

plt.close()

###############################################################
# PRINT RESULTS
###############################################################

print("\nEvaluation Results")

print(summary)

print("\nResults saved to Results folder")

print("Figures saved to Figures folder")

print("="*70)