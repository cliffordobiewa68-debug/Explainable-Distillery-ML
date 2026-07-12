#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================
PROJECT:
Explainable Machine Learning Model for Distillery Effluent
Treatment and Management

MODULE:
02_Data_Cleaning.py

AUTHOR:
Clifford Otieno Obiewa

==============================================================
"""

###############################################################
# IMPORT LIBRARIES
###############################################################

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

###############################################################
# CREATE FOLDERS
###############################################################

os.makedirs("Dataset", exist_ok=True)
os.makedirs("Results", exist_ok=True)

###############################################################
# LOAD DATASET
###############################################################

df = pd.read_csv("Dataset/distillery_spent_wash.csv")

print("="*70)
print("DATASET LOADED")
print("="*70)

print(df.head())

###############################################################
# CONVERT DATE
###############################################################

df["Record_Date"] = pd.to_datetime(df["Record_Date"])

###############################################################
# DATA INFORMATION
###############################################################

print("\nShape")
print(df.shape)

print("\nData Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

###############################################################
# REMOVE DUPLICATES
###############################################################

duplicates = df.duplicated().sum()

print("\nDuplicate Records:", duplicates)

df.drop_duplicates(inplace=True)

###############################################################
# DEFINE VARIABLES
###############################################################

target_columns = [

    "Final_pH",
    "Final_COD_mgL",
    "Final_BOD_mgL",
    "Final_TDS_mgL",
    "Final_TSS_mgL",
    "Compliance"

]

predictor_columns = [

    "Raw_pH",
    "Raw_COD_mgL",
    "Raw_BOD_mgL",
    "Raw_TDS_mgL",
    "Raw_TSS_mgL",
    "Input_Lime_kg_m3",
    "Input_Urea_kg_m3",
    "Input_DAP_kg_m3",
    "Output_Biogas_m3_m3"

]

categorical_columns = [

    "Operating_Shift",
    "Production_Level",
    "Plant_Status",
    "Month",
    "Quarter",
    "Season"

]

###############################################################
# HANDLE MISSING VALUES
###############################################################

for col in predictor_columns:

    df[col].fillna(

        df[col].median(),

        inplace=True

    )

for col in categorical_columns:

    df[col].fillna(

        df[col].mode()[0],

        inplace=True

    )

###############################################################
# REMOVE NEGATIVE VALUES
###############################################################

for col in predictor_columns:

    df.loc[df[col] < 0, col] = np.nan

    df[col].fillna(

        df[col].median(),

        inplace=True

    )

###############################################################
# OUTLIER TREATMENT
###############################################################

for col in predictor_columns:

    Q1 = df[col].quantile(0.25)

    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    df[col] = np.clip(

        df[col],

        lower,

        upper

    )

###############################################################
# VERIFY TARGET VARIABLES
###############################################################

print("\n")

print("="*70)

print("TARGET VARIABLE DISTRIBUTION")

print("="*70)

print(df["Compliance"].value_counts())

###############################################################
# SUMMARY STATISTICS
###############################################################

summary = df.describe(include="all")

summary.to_excel(

    "Results/Data_Summary.xlsx"

)

###############################################################
# DATA QUALITY REPORT
###############################################################

quality = pd.DataFrame({

    "Variable": df.columns,

    "Data_Type": df.dtypes.values,

    "Missing": df.isnull().sum().values,

    "Unique_Values": df.nunique().values

})

quality.to_excel(

    "Results/Data_Quality_Report.xlsx",

    index=False

)

###############################################################
# SAVE CLEAN DATASET
###############################################################

df.to_csv(

    "Dataset/distillery_clean.csv",

    index=False

)

###############################################################
# FINAL OUTPUT
###############################################################

print("\n")

print("="*70)

print("DATA CLEANING COMPLETED")

print("="*70)

print("Final Shape:", df.shape)

print("\nCompliance Distribution")

print(df["Compliance"].value_counts())

print("\nMissing Values Remaining")

print(df.isnull().sum().sum())

print("\nClean Dataset Saved")

print("Dataset/distillery_clean.csv")

print("\nQuality Report Saved")

print("Results/Data_Quality_Report.xlsx")

print("\nSummary Report Saved")

print("Results/Data_Summary.xlsx")

print("="*70)