#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 10:48:55 2026

@author: cliffordobiewa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================
Explainable Machine Learning System
for
Distillery Effluent Treatment and Management

Streamlit User Interface

Author:
Clifford Otieno Obiewa
==============================================================
"""

###############################################################
# IMPORT LIBRARIES
###############################################################

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

###############################################################
# IMPORT DEPLOYMENT BACKEND
###############################################################

from deployment import (
    FEATURE_NAMES,
    TARGET_NAMES,
    predict_effluent
)

###############################################################
# PAGE CONFIGURATION
###############################################################

st.set_page_config(

    page_title="Distillery Effluent ML",

    page_icon="🧪",

    layout="wide",

    initial_sidebar_state="expanded"

)

###############################################################
# CUSTOM HEADER
###############################################################

st.title("🧪 Explainable Machine Learning System")

st.subheader("Distillery Effluent Treatment and Management")

st.markdown("---")

###############################################################
# SIDEBAR
###############################################################

st.sidebar.image(
    "https://img.icons8.com/color/96/artificial-intelligence.png",
    width=90
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Page",

    [

        "🏠 Home",

        "📈 Prediction",

        "📊 Explainability",

        "📉 Model Performance",

        "📁 Reports",

        "ℹ About"

    ]

)

###############################################################
# HOME PAGE
###############################################################

if page == "🏠 Home":

    st.header("Welcome")

    st.write("""

This application demonstrates an Explainable Machine Learning
model for predicting treated distillery effluent quality.

The system predicts:

- Final pH
- Final COD
- Final BOD
- Final TDS
- Final TSS

It also provides explainability using:

- Feature Importance
- Permutation Importance
- SHAP
- LIME
- PDP
- ICE

""")

    st.success("Select **Prediction** from the left menu to begin.")