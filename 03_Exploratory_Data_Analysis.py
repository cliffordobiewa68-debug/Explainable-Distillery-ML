#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 11:25:42 2026

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
03_Exploratory_Data_Analysis.py
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
import matplotlib.pyplot as plt

###############################################################
# CREATE FOLDERS
###############################################################

os.makedirs("Figures", exist_ok=True)
os.makedirs("Results", exist_ok=True)

###############################################################
# LOAD CLEAN DATA
###############################################################

df = pd.read_csv(
    "Dataset/distillery_clean.csv"
)

df["Record_Date"] = pd.to_datetime(df["Record_Date"])

###############################################################
# NUMERICAL VARIABLES
###############################################################

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

###############################################################
# DESCRIPTIVE STATISTICS
###############################################################

stats_table = df[numeric_columns].describe().T

stats_table["Median"] = df[numeric_columns].median()
stats_table["Variance"] = df[numeric_columns].var()
stats_table["Skewness"] = df[numeric_columns].skew()
stats_table["Kurtosis"] = df[numeric_columns].kurt()

stats_table = stats_table[

[
'count',
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
'Kurtosis'

]

]

print(stats_table)

stats_table.to_excel(

"Results/Table_4_Descriptive_Statistics.xlsx"

)

###############################################################
# COEFFICIENT OF VARIATION
###############################################################

cv = (

df[numeric_columns].std()

/

df[numeric_columns].mean()

) * 100

cv = cv.round(2)

cv.to_excel(

"Results/Table_4_Coefficient_of_Variation.xlsx"

)

###############################################################
# FREQUENCY TABLES
###############################################################

categorical = [

"Operating_Shift",

"Production_Level",

"Plant_Status",

"Season",

"Quarter",

"Compliance"

]

for variable in categorical:

    freq = df[variable].value_counts()

    freq.to_excel(

        f"Results/Frequency_{variable}.xlsx"

    )

###############################################################
# HISTOGRAMS
###############################################################

for variable in numeric_columns:

    plt.figure(figsize=(6,4))

    plt.hist(

        df[variable],

        bins=20

    )

    plt.title(variable)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(

        f"Figures/{variable}_Histogram.png",

        dpi=300

    )

    plt.close()

###############################################################
# BOXPLOTS
###############################################################

for variable in numeric_columns:

    plt.figure(figsize=(6,4))

    plt.boxplot(

        df[variable]

    )

    plt.title(variable)

    plt.tight_layout()

    plt.savefig(

        f"Figures/{variable}_Boxplot.png",

        dpi=300

    )

    plt.close()

###############################################################
# CORRELATION MATRIX
###############################################################

corr = df[numeric_columns].corr()

corr.to_excel(

"Results/Table_4_Correlation_Matrix.xlsx"

)

plt.figure(figsize=(12,10))

plt.imshow(

corr,

aspect="auto"

)

plt.colorbar()

plt.xticks(

range(len(corr.columns)),

corr.columns,

rotation=90

)

plt.yticks(

range(len(corr.columns)),

corr.columns

)

plt.tight_layout()

plt.savefig(

"Figures/Correlation_Matrix.png",

dpi=300

)

plt.close()

###############################################################
# MONTHLY AVERAGES
###############################################################

monthly = df.groupby(

"Month"

)[

numeric_columns

].mean()

monthly.to_excel(

"Results/Monthly_Averages.xlsx"

)

###############################################################
# SEASONAL AVERAGES
###############################################################

seasonal = df.groupby(

"Season"

)[

numeric_columns

].mean()

seasonal.to_excel(

"Results/Seasonal_Averages.xlsx"

)

###############################################################
# TIME SERIES
###############################################################

targets = [

"Final_COD_mgL",

"Final_BOD_mgL",

"Final_TSS_mgL",

"Final_pH"

]

for variable in targets:

    plt.figure(figsize=(12,4))

    plt.plot(

        df["Record_Date"],

        df[variable]

    )

    plt.title(variable)

    plt.tight_layout()

    plt.savefig(

        f"Figures/{variable}_TimeSeries.png",

        dpi=300

    )

    plt.close()

###############################################################
# BAR CHARTS
###############################################################

plots = [

"Operating_Shift",

"Production_Level",

"Plant_Status",

"Season"

]

for variable in plots:

    plt.figure(figsize=(6,4))

    df[variable].value_counts().plot(

        kind="bar"

    )

    plt.title(variable)

    plt.tight_layout()

    plt.savefig(

        f"Figures/{variable}.png",

        dpi=300

    )

    plt.close()

###############################################################
# COMPLIANCE PIE CHART
###############################################################

plt.figure(figsize=(6,6))

df["Compliance"].value_counts().plot(

kind="pie",

autopct="%1.1f%%"

)

plt.ylabel("")

plt.tight_layout()

plt.savefig(

"Figures/Compliance_PieChart.png",

dpi=300

)

plt.close()

###############################################################
# SUMMARY
###############################################################

print("="*70)

print("EDA COMPLETED SUCCESSFULLY")

print("="*70)

print("Descriptive Statistics Saved")

print("Correlation Matrix Saved")

print("Histograms Saved")

print("Boxplots Saved")

print("Time Series Saved")

print("Monthly Summary Saved")

print("Seasonal Summary Saved")

print("Frequency Tables Saved")

print("Figures Generated")

print("="*70)