#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 11:19:32 2026

@author: cliffordobiewa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================
PROJECT:
Explainable Machine Learning Model for Distillery Effluent
Treatment and Management

AUTHOR:
Clifford Otieno Obiewa

MSc Information Technology
Kabarak University

MODULE:
01_Data_Generation.py
==============================================================
"""

###############################################################
# IMPORT LIBRARIES
###############################################################

import os
import numpy as np
import pandas as pd

np.random.seed(42)

###############################################################
# CREATE PROJECT FOLDERS
###############################################################

folders = [
    "Dataset",
    "Models",
    "Results",
    "Figures"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

###############################################################
# NUMBER OF RECORDS
###############################################################

records = 1000

###############################################################
# DATE
###############################################################

dates = pd.date_range(
    start="2024-01-01",
    periods=records,
    freq="D"
)

###############################################################
# RAW WASTEWATER CHARACTERISTICS
###############################################################

Raw_pH = np.random.normal(
    4.20,
    0.18,
    records
)

Raw_pH = np.clip(
    Raw_pH,
    3.8,
    4.6
)

Raw_COD = np.random.normal(
    115000,
    12000,
    records
)

Raw_COD = np.clip(
    Raw_COD,
    90000,
    140000
)

Raw_BOD = (
    Raw_COD *
    np.random.uniform(
        0.45,
        0.55,
        records
    )
)

Raw_TDS = np.random.normal(
    90000,
    9000,
    records
)

Raw_TDS = np.clip(
    Raw_TDS,
    70000,
    110000
)

Raw_TSS = np.random.normal(
    18000,
    3500,
    records
)

Raw_TSS = np.clip(
    Raw_TSS,
    12000,
    25000
)

###############################################################
# TREATMENT INPUTS
###############################################################

Input_Lime = np.random.normal(
    3.2,
    0.4,
    records
)

Input_Lime = np.clip(
    Input_Lime,
    2.5,
    4.5
)

Input_Urea = np.random.normal(
    0.35,
    0.06,
    records
)

Input_Urea = np.clip(
    Input_Urea,
    0.20,
    0.50
)

Input_DAP = np.random.normal(
    0.10,
    0.02,
    records
)

Input_DAP = np.clip(
    Input_DAP,
    0.05,
    0.15
)

###############################################################
# CATEGORICAL VARIABLES
###############################################################

Operating_Shift = np.random.choice(

    ["Morning","Afternoon","Night"],

    records,

    p=[0.40,0.35,0.25]

)

Production_Level = np.random.choice(

    ["Low","Medium","High"],

    records,

    p=[0.25,0.50,0.25]

)

Plant_Status = np.random.choice(

    ["Normal Operation","Maintenance"],

    records,

    p=[0.92,0.08]

)

###############################################################
# TREATMENT EFFICIENCY
###############################################################


# Seasonal treatment efficiency
season_factor = np.random.choice(
    [1.00, 0.995, 0.990],
    size=records,
    p=[0.60, 0.25, 0.15]
)

# Plant status
plant_factor = np.where(
    Plant_Status == "Maintenance",
    0.992,
    1.000
)

# Production loading
production_factor = np.select(
    [
        Production_Level == "Low",
        Production_Level == "Medium",
        Production_Level == "High"
    ],
    [
        1.002,
        1.000,
        0.995
    ]
)

# Operating shift influence
shift_factor = np.select(
    [
        Operating_Shift == "Morning",
        Operating_Shift == "Afternoon",
        Operating_Shift == "Night"
    ],
    [
        1.000,
        0.998,
        0.995
    ]
)

# Overall treatment efficiency
efficiency = (
    np.random.uniform(0.9988, 0.9995, records)
    * season_factor
    * plant_factor
    * production_factor
    * shift_factor
)

efficiency = np.clip(
    efficiency,
    0.995,
    0.9997
)

###############################################################
# BIOGAS
###############################################################

Output_Biogas = (
    Raw_BOD
    * np.random.uniform(0.0055, 0.0070, records)
)

###############################################################
# SENSOR NOISE
###############################################################

noise_ph = np.random.normal(0, 0.12, records)
noise_cod = np.random.normal(0, 10, records)
noise_bod = np.random.normal(0, 3, records)
noise_tds = np.random.normal(0, 6, records)
noise_tss = np.random.normal(0, 2, records)

###############################################################
# FINAL TREATED EFFLUENT
###############################################################

Final_pH = (
    7.2
    + noise_ph
    + (Input_Lime - 3.2) * 0.15
)

Final_pH = np.clip(Final_pH, 6.5, 8.5)

Final_COD = (
    Raw_COD * (1 - efficiency)
    + noise_cod
)

Final_COD = np.clip(Final_COD, 40, 180)

Final_BOD = (
    Final_COD
    * np.random.uniform(0.12, 0.22, records)
    + noise_bod
)

Final_BOD = np.clip(Final_BOD, 5, 45)

Final_TDS = (
    np.random.normal(95, 18, records)
    + noise_tds
)

Final_TDS = np.clip(Final_TDS, 50, 180)

Final_TSS = (
    np.random.normal(12, 4, records)
    + noise_tss
)

Final_TSS = np.clip(Final_TSS, 2, 45)

###############################################################
# CREATE DATAFRAME
###############################################################

df = pd.DataFrame({

    "Record_Date":dates,

    "Raw_pH":Raw_pH.round(2),

    "Raw_COD_mgL":Raw_COD.round(0),

    "Raw_BOD_mgL":Raw_BOD.round(0),

    "Raw_TDS_mgL":Raw_TDS.round(0),

    "Raw_TSS_mgL":Raw_TSS.round(0),

    "Input_Lime_kg_m3":Input_Lime.round(2),

    "Input_Urea_kg_m3":Input_Urea.round(2),

    "Input_DAP_kg_m3":Input_DAP.round(2),

    "Output_Biogas_m3_m3":Output_Biogas.round(2),

    "Operating_Shift":Operating_Shift,

    "Production_Level":Production_Level,

    "Plant_Status":Plant_Status,

    "Final_pH":Final_pH.round(2),

    "Final_COD_mgL":Final_COD.round(0),

    "Final_BOD_mgL":Final_BOD.round(0),

    "Final_TDS_mgL":Final_TDS.round(0),

    "Final_TSS_mgL":Final_TSS.round(0)

})

###############################################################
# DATE FEATURES
###############################################################

df["Month"] = df["Record_Date"].dt.month_name()

df["Quarter"] = df["Record_Date"].dt.quarter

df["Season"] = np.select(

[
df["Record_Date"].dt.month.isin([12,1,2]),
df["Record_Date"].dt.month.isin([3,4,5]),
df["Record_Date"].dt.month.isin([6,7,8]),
df["Record_Date"].dt.month.isin([9,10,11])
],

[
"Dry Season",
"Long Rains",
"Cool Season",
"Short Rains"
]

)

###############################################################
# COMPLIANCE
###############################################################

###############################################################
# COMPLIANCE STATUS
###############################################################

df["Compliance"] = np.where(

    (
        (df["Final_pH"] >= 6.5) &
        (df["Final_pH"] <= 8.5) &
        (df["Final_COD_mgL"] <= 100) &
        (df["Final_BOD_mgL"] <= 30) &
        (df["Final_TSS_mgL"] <= 30)
    ),

    1,

    0

)

print("\nCompliance Distribution")
print(df["Compliance"].value_counts())
print(df["Compliance"].value_counts(normalize=True) * 100)

###############################################################
# SAVE DATASET
###############################################################

df.to_csv(

"Dataset/distillery_spent_wash.csv",

index=False

)

print("="*70)

print("DATASET CREATED SUCCESSFULLY")

print("="*70)

print(df.head())

print(df.shape)

print(df["Compliance"].value_counts())

print("="*70)
print("="*70)
print("COMPLIANCE DISTRIBUTION")
print(df["Compliance"].value_counts())
print(df["Compliance"].value_counts(normalize=True))
print("="*70)

print(df[
    [
        "Final_pH",
        "Final_COD_mgL",
        "Final_BOD_mgL",
        "Final_TSS_mgL",
        "Compliance"
    ]
].head(20))
print(df["Compliance"].value_counts())