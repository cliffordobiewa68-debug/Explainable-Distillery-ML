# ============================================================
# AIKABARAK PROJECT
#
# Streamlit Application
#
# Explainable Machine Learning Model for
# Distillery Effluent Treatment and Management
#
# PART A:
# Imports
# Configuration
# Utilities
# Caching
# Model Loading
# Automatic Feature Alignment
#
# Author:
# Clifford Otieno Obiewa
#
# MSc Information Technology
# Kabarak University
# ============================================================


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import streamlit as st

import pandas as pd
import numpy as np
import datetime

from pathlib import Path

import joblib


# Visualisation

import matplotlib.pyplot as plt
import seaborn as sns



# Machine Learning

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)



# Explainability

import shap



# PDF Reporting

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib.styles import getSampleStyleSheet






# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================


st.set_page_config(

    page_title=
    "AI Distillery Effluent Treatment System",

    page_icon="💧",

    layout="wide"

)





# ============================================================
# PROJECT DIRECTORIES
# ============================================================


BASE_DIR = Path(__file__).parent


MODEL_DIR = BASE_DIR / "Models"

DATA_DIR = BASE_DIR / "Data"

REPORT_DIR = BASE_DIR / "Reports"

EXPLAIN_DIR = BASE_DIR / "Explainability"



for folder in [

    MODEL_DIR,
    DATA_DIR,
    REPORT_DIR,
    EXPLAIN_DIR

]:

    folder.mkdir(
        exist_ok=True
    )






# ============================================================
# MODEL FILES
# ============================================================


MODEL_FILE = (

    MODEL_DIR /

    "Best_Model.pkl"

)



SCALER_FILE = (

    MODEL_DIR /

    "Scaler.pkl"

)





# ============================================================
# USER INPUT VARIABLES
# ============================================================


INPUT_FEATURES = [

    "Raw_pH",

    "Raw_COD_mgL",

    "Raw_BOD_mgL",

    "Raw_TDS_mgL",

    "Raw_TSS_mgL",

    "Input_Lime_kg_m3",

    "Input_Urea_kg_m3",

    "Input_DAP_kg_m3"

]





# ============================================================
# EXPECTED OUTPUT VARIABLES
# ============================================================


OUTPUT_FEATURES = [

    "Final_pH",

    "Final_COD_mgL",

    "Final_BOD_mgL",

    "Final_TDS_mgL",

    "Final_TSS_mgL"

]






# ============================================================
# ENVIRONMENTAL LIMITS
# ============================================================


EFFLUENT_LIMITS = {


    "Final_pH":
    {
        "min":6,
        "max":9
    },


    "Final_COD_mgL":
    {
        "max":50
    },


    "Final_BOD_mgL":
    {
        "max":30
    },


    "Final_TSS_mgL":
    {
        "max":50
    },


    "Final_TDS_mgL":
    {
        "max":1000
    }


}






# ============================================================
# LOAD DATASET
# ============================================================


@st.cache_data

def load_dataset():


    file = (

        DATA_DIR /

        "cleaned_distillery_effluent.csv"

    )


    if file.exists():


        return pd.read_csv(file)



    return None







# ============================================================
# LOAD MODEL
# ============================================================


@st.cache_resource

def load_models():


    models={}



    if MODEL_FILE.exists():


        model = joblib.load(

            MODEL_FILE

        )


        models["Best_Model"]=model



    else:


        models["Best_Model"]=None



    return models






# ============================================================
# LOAD SCALER
# ============================================================


@st.cache_resource

def load_scaler():


    if SCALER_FILE.exists():


        return joblib.load(

            SCALER_FILE

        )


    return None






# ============================================================
# PREPARE INPUT DATA
# ============================================================


def prepare_input(data):


    df=data.copy()



    df=df[

        INPUT_FEATURES

    ]



    return df






# ============================================================
# AUTOMATIC FEATURE ENGINEERING
#
# Matches training features
# ============================================================


def align_features(

        input_data,

        model

):


    try:


        expected_features = (

            model.feature_names_in_

        )


    except:


        expected_features=(

            input_data.columns

        )



    aligned=pd.DataFrame()



    for feature in expected_features:



        if feature in input_data.columns:


            aligned[feature]=(

                input_data[feature]

            )



        else:


            # Automatically create
            # missing engineered variables


            aligned[feature]=0




    aligned=aligned.fillna(0)



    return aligned







# ============================================================
# PREDICTION FUNCTION
# ============================================================


def make_prediction(

        input_data,

        models

):


    model=models.get(

        "Best_Model"

    )



    if model is None:


        raise Exception(

            "Best_Model.pkl missing"

        )



    aligned_data = align_features(

        input_data,

        model

    )



    prediction = model.predict(

        aligned_data

    )



    prediction=np.array(

        prediction

    )



    # Multi-output model


    if prediction.ndim == 2:


        if prediction.shape[1]==5:


            result=pd.DataFrame(

                prediction,

                columns=
                OUTPUT_FEATURES

            )


        else:


            result=pd.DataFrame(

                prediction

            )


    else:


        result=pd.DataFrame(

            {

            "Final_COD_mgL":

            prediction

            }

        )



    return result







# ============================================================
# COMPLIANCE FUNCTION
# ============================================================


def check_compliance(results):


    status={}



    for parameter,value in results.items():



        limits=EFFLUENT_LIMITS.get(

            parameter,

            {}

        )


        passed=True



        if "min" in limits:


            if value < limits["min"]:

                passed=False



        if "max" in limits:


            if value > limits["max"]:

                passed=False



        status[parameter]=(

            "PASS"

            if passed

            else

            "FAIL"

        )



    return status






# ============================================================
# SESSION STORAGE
# ============================================================


if "prediction_result" not in st.session_state:


    st.session_state.prediction_result=None




if "batch_result" not in st.session_state:


    st.session_state.batch_result=None







# ============================================================
# INITIALIZE RESOURCES
# ============================================================


models=load_models()


scaler=load_scaler()


dataset=load_dataset()






# ============================================================
# SIDEBAR STATUS
# ============================================================


st.sidebar.title(

    "💧 AI Effluent Treatment System"

)



if models["Best_Model"] is not None:


    st.sidebar.success(

        "✓ Best_Model.pkl Loaded"

    )


    try:

        st.sidebar.info(

            f"""
Model Features:

{len(models['Best_Model'].feature_names_in_)}

features detected

"""

        )


    except:

        pass



else:


    st.sidebar.error(

        """
No trained model found.

Expected:

Models/Best_Model.pkl

"""

    )





# ============================================================
# END OF PART A
# ============================================================
# ============================================================
# PART B:
# REDESIGNED PREDICTION INTERFACE
#
# - Manual Prediction
# - Batch Prediction
# - Automatic Feature Handling
# - Result Display
# ============================================================



# ============================================================
# APPLICATION TITLE
# ============================================================


st.title(
    "💧 Explainable ML-Based Distillery Effluent Treatment System"
)


st.markdown(
"""
This application predicts treated effluent quality parameters
using trained Machine Learning models.

The system predicts:

- Final pH
- Final COD
- Final BOD
- Final TDS
- Final TSS

"""
)



# ============================================================
# CHECK MODEL AVAILABILITY
# ============================================================


available_models = [

    name
    for name,model in models.items()
    if model is not None

]


if len(available_models)==0:


    st.error(
        """
        No trained models found.

        Please ensure the Models folder contains:

        Final_pH_model.pkl
        Final_COD_model.pkl
        Final_BOD_model.pkl
        Final_TDS_model.pkl
        Final_TSS_model.pkl

        """
    )





# ============================================================
# CREATE TABS
# ============================================================


tab1, tab2, tab3 = st.tabs(

[
"🧪 Single Prediction",
"📂 Batch Prediction",
"📋 Historical Data"
]

)





# ============================================================
# TAB 1:
# SINGLE PREDICTION
# ============================================================


with tab1:


    st.subheader(
        "Enter Raw Effluent and Treatment Inputs"
    )



    col1,col2,col3 = st.columns(3)



    with col1:


        raw_ph = st.number_input(

            "Raw pH",

            min_value=0.0,

            max_value=14.0,

            value=4.5,

            step=0.1

        )



        raw_cod = st.number_input(

            "Raw COD (mg/L)",

            min_value=0.0,

            value=5000.0,

            step=50.0

        )



        raw_bod = st.number_input(

            "Raw BOD (mg/L)",

            min_value=0.0,

            value=2500.0,

            step=50.0

        )



    with col2:


        raw_tds = st.number_input(

            "Raw TDS (mg/L)",

            min_value=0.0,

            value=1500.0,

            step=50.0

        )


        raw_tss = st.number_input(

            "Raw TSS (mg/L)",

            min_value=0.0,

            value=500.0,

            step=10.0

        )



        lime = st.number_input(

            "Lime Dosage (kg/m³)",

            min_value=0.0,

            value=1.0,

            step=0.1

        )



    with col3:


        urea = st.number_input(

            "Urea Dosage (kg/m³)",

            min_value=0.0,

            value=0.2,

            step=0.05

        )



        dap = st.number_input(

            "DAP Dosage (kg/m³)",

            min_value=0.0,

            value=0.2,

            step=0.05

        )






    # Create dataframe


    input_record = pd.DataFrame(

        {

        "Raw_pH":[raw_ph],

        "Raw_COD_mgL":[raw_cod],

        "Raw_BOD_mgL":[raw_bod],

        "Raw_TDS_mgL":[raw_tds],

        "Raw_TSS_mgL":[raw_tss],

        "Input_Lime_kg_m3":[lime],

        "Input_Urea_kg_m3":[urea],

        "Input_DAP_kg_m3":[dap]

        }

    )



    st.write(
        "Input Data Preview"
    )


    st.dataframe(
        input_record
    )





    # Prediction button


    if st.button(
        "🚀 Predict Effluent Quality",
        key="single_prediction"
    ):


        try:


            prepared = prepare_input(
                input_record
            )


            result = make_prediction(

                prepared,

                models

            )


            st.session_state.prediction_result=result



            st.success(
                "Prediction completed successfully"
            )



        except Exception as e:


            st.error(
                f"Prediction error: {e}"
            )





    # Display result


    if st.session_state.prediction_result is not None:


        st.subheader(
            "Predicted Final Effluent Quality"
        )


        result = (

            st.session_state.prediction_result

        )


        st.dataframe(

            result.style.format(
                "{:.3f}"
            )

        )



        # Compliance


        compliance = check_compliance(

            result.iloc[0].to_dict()

        )


        st.subheader(
            "Environmental Compliance Status"
        )


        for parameter,status in compliance.items():


            if status=="PASS":

                st.success(
                    f"{parameter}: {status}"
                )

            else:

                st.error(
                    f"{parameter}: {status}"
                )






# ============================================================
# TAB 2:
# BATCH PREDICTION
# ============================================================


with tab2:


    st.subheader(
        "Upload Multiple Effluent Samples"
    )



    st.info(

"""
Upload CSV containing:

Raw_pH,
Raw_COD_mgL,
Raw_BOD_mgL,
Raw_TDS_mgL,
Raw_TSS_mgL,
Input_Lime_kg_m3,
Input_Urea_kg_m3,
Input_DAP_kg_m3

"""
    )




    uploaded_file = st.file_uploader(

        "Choose CSV file",

        type=["csv"]

    )





    if uploaded_file:


        try:


            batch_data = pd.read_csv(

                uploaded_file

            )


            st.write(

                "Uploaded Data"

            )


            st.dataframe(

                batch_data.head()

            )



            missing = [

                col

                for col in INPUT_FEATURES

                if col not in batch_data.columns

            ]



            if missing:


                st.error(

                    f"Missing columns: {missing}"

                )


            else:


                if st.button(

                    "Run Batch Prediction",

                    key="batch_prediction"

                ):



                    prepared = prepare_input(

                        batch_data

                    )


                    predictions = make_prediction(

                        prepared,

                        models

                    )



                    final_results = pd.concat(

                        [

                        batch_data.reset_index(drop=True),

                        predictions.reset_index(drop=True)

                        ],

                        axis=1

                    )



                    st.session_state.batch_result = (

                        final_results

                    )



                    st.success(

                        "Batch prediction completed"

                    )





        except Exception as e:


            st.error(

                f"File processing error: {e}"

            )





    if st.session_state.batch_result is not None:


        st.subheader(

            "Batch Prediction Results"

        )


        st.dataframe(

            st.session_state.batch_result

        )



        csv = (

            st.session_state.batch_result

            .to_csv(index=False)

            .encode("utf-8")

        )



        st.download_button(

            label="⬇ Download Predictions CSV",

            data=csv,

            file_name=
            "effluent_predictions.csv",

            mime=
            "text/csv"

        )






# ============================================================
# TAB 3:
# HISTORICAL DATA VIEW
# ============================================================


with tab3:


    st.subheader(

        "Historical Dataset Explorer"

    )



    if dataset is not None:


        st.write(

            f"Dataset size: {dataset.shape}"

        )


        st.dataframe(

            dataset.head(100)

        )



        st.download_button(

            "Download Historical Dataset",

            dataset.to_csv(
                index=False
            ),

            "historical_effluent_data.csv"

        )


    else:


        st.warning(

            """
            Historical dataset not found.

            Expected location:

            Data/
            cleaned_distillery_effluent.csv

            """

        )





# ============================================================
# END PART B
# ============================================================

# ============================================================
# PART C:
# DASHBOARDS, COMPLIANCE MONITORING,
# VISUALIZATIONS AND PERFORMANCE METRICS
#
# ============================================================



st.divider()

st.header(
    "📊 Environmental Monitoring Dashboard"
)




# ============================================================
# CHECK AVAILABLE PREDICTIONS
# ============================================================


dashboard_result = None


raw_input_result = None



if st.session_state.prediction_result is not None:


    dashboard_result = (

        st.session_state.prediction_result

    )



    raw_input_result = input_record





elif st.session_state.batch_result is not None:


    dashboard_result = (

        st.session_state.batch_result[
            OUTPUT_FEATURES
        ]

    )






# ============================================================
# EXECUTIVE KPI DASHBOARD
# ============================================================


if dashboard_result is not None:



    st.subheader(
        "Treatment Plant Performance Indicators"
    )



    latest = dashboard_result.iloc[0]



    col1,col2,col3,col4,col5 = st.columns(5)



    with col1:


        st.metric(

            label="Final pH",

            value=round(

                latest["Final_pH"],

                2

            )

        )



    with col2:


        st.metric(

            label="Final COD",

            value=

            f"{latest['Final_COD_mgL']:.1f} mg/L"

        )



    with col3:


        st.metric(

            label="Final BOD",

            value=

            f"{latest['Final_BOD_mgL']:.1f} mg/L"

        )



    with col4:


        st.metric(

            label="Final TDS",

            value=

            f"{latest['Final_TDS_mgL']:.1f} mg/L"

        )



    with col5:


        st.metric(

            label="Final TSS",

            value=

            f"{latest['Final_TSS_mgL']:.1f} mg/L"

        )







# ============================================================
# COMPLIANCE MONITORING
# ============================================================


st.subheader(
    "Environmental Compliance Assessment"
)



if dashboard_result is not None:


    compliance_records=[]



    for parameter in OUTPUT_FEATURES:



        value=float(

            latest[parameter]

        )


        limit = EFFLUENT_LIMITS.get(

            parameter,

            {}

        )



        status="PASS"



        if "min" in limit:


            if value < limit["min"]:

                status="FAIL"



        if "max" in limit:


            if value > limit["max"]:

                status="FAIL"




        compliance_records.append(

            {

            "Parameter":

            parameter,


            "Predicted Value":

            round(value,3),


            "Compliance":

            status


            }

        )



    compliance_df=pd.DataFrame(

        compliance_records

    )



    st.dataframe(

        compliance_df,

        use_container_width=True

    )





# ============================================================
# REMOVAL EFFICIENCY CALCULATION
# ============================================================


st.subheader(

    "Treatment Removal Efficiency"

)



if (

    dashboard_result is not None

    and raw_input_result is not None

):



    efficiency={}



    parameters={


        "COD":

        (

        raw_input_result["Raw_COD_mgL"][0],

        latest["Final_COD_mgL"]

        ),



        "BOD":

        (

        raw_input_result["Raw_BOD_mgL"][0],

        latest["Final_BOD_mgL"]

        ),



        "TDS":

        (

        raw_input_result["Raw_TDS_mgL"][0],

        latest["Final_TDS_mgL"]

        ),



        "TSS":

        (

        raw_input_result["Raw_TSS_mgL"][0],

        latest["Final_TSS_mgL"]

        )


    }



    for name,(raw,final) in parameters.items():


        if raw > 0:


            efficiency[name]=round(

                ((raw-final)/raw)*100,

                2

            )


        else:


            efficiency[name]=0




    efficiency_df=pd.DataFrame(

        {

        "Parameter":

        list(efficiency.keys()),


        "Removal Efficiency (%)":

        list(efficiency.values())

        }

    )



    st.dataframe(

        efficiency_df

    )



    st.bar_chart(

        efficiency_df.set_index(

            "Parameter"

        )

    )






# ============================================================
# RAW VS TREATED VISUALIZATION
# ============================================================


st.subheader(

    "Raw Effluent vs Treated Effluent"

)



if (

    dashboard_result is not None

    and raw_input_result is not None

):



    comparison=pd.DataFrame(

        {

        "COD":

        [

        raw_input_result["Raw_COD_mgL"][0],

        latest["Final_COD_mgL"]

        ],


        "BOD":

        [

        raw_input_result["Raw_BOD_mgL"][0],

        latest["Final_BOD_mgL"]

        ],


        "TDS":

        [

        raw_input_result["Raw_TDS_mgL"][0],

        latest["Final_TDS_mgL"]

        ],


        "TSS":

        [

        raw_input_result["Raw_TSS_mgL"][0],

        latest["Final_TSS_mgL"]

        ]

        },

        index=[

        "Raw",

        "Treated"

        ]

    )



    st.bar_chart(

        comparison

    )








# ============================================================
# HISTORICAL DATA ANALYSIS
# ============================================================


if dataset is not None:



    st.divider()


    st.header(

        "📈 Historical Dataset Analytics"

    )



    tab1,tab2,tab3 = st.tabs(

        [

        "Trends",

        "Correlation",

        "Statistics"

        ]

    )



    # -------------------------------
    # Trends
    # -------------------------------


    with tab1:



        parameter=st.selectbox(

            "Select parameter",

            dataset.columns

        )


        st.line_chart(

            dataset[parameter]

        )




    # -------------------------------
    # Correlation
    # -------------------------------


    with tab2:


        numeric = dataset.select_dtypes(

            include=np.number

        )



        fig,ax=plt.subplots(

            figsize=(10,6)

        )


        sns.heatmap(

            numeric.corr(),

            annot=True,

            ax=ax

        )


        st.pyplot(fig)






    # -------------------------------
    # Statistics
    # -------------------------------


    with tab3:


        st.dataframe(

            dataset.describe()

        )








# ============================================================
# MODEL PERFORMANCE
# ============================================================


st.divider()


st.header(

    "🤖 Model Performance"

)



# ============================================================
# MODEL PERFORMANCE
# LOAD EXISTING EVALUATION RESULTS
# ============================================================


st.divider()


st.header(
    "🤖 Model Performance Monitoring"
)



performance_sources = [

    BASE_DIR /
    "final_predictions_summary.csv",

    BASE_DIR /
    "Results" /
    "Explainability" /
    "Explainability_Summary.csv"

]



performance_loaded=False



for file in performance_sources:


    if file.exists():


        try:


            performance = pd.read_csv(file)



            st.subheader(

                f"Source: {file.name}"

            )


            st.dataframe(

                performance,

                use_container_width=True

            )


            performance_loaded=True


            break



        except Exception as e:


            st.warning(

                f"Could not load {file}: {e}"

            )





if not performance_loaded:


    st.info(

"""
No model evaluation summary detected.

Expected files:

final_predictions_summary.csv

or

Results/Explainability/Explainability_Summary.csv

"""

    )

# ============================================================
# END PART C
# ============================================================

# ============================================================
# PART D:
# EXPLAINABLE AI MODULE
#
# SHAP
# LIME
# Feature Importance
# Permutation Importance
#
# ============================================================



st.divider()

st.header(
    "🧠 Explainable Artificial Intelligence (XAI) Dashboard"
)



# ============================================================
# EXPLAINABILITY PATHS
# ============================================================


SHAP_DIR = (

    BASE_DIR /

    "Results" /

    "Explainability" /

    "SHAP"

)



LIME_DIR = (

    BASE_DIR /

    "Results" /

    "Explainability" /

    "LIME"

)



FEATURE_DIR = (

    BASE_DIR /

    "Results" /

    "Explainability" /

    "FeatureImportance"

)



PERM_DIR = (

    BASE_DIR /

    "Results" /

    "Explainability" /

    "Permutation"

)





# ============================================================
# OUTPUT PARAMETER SELECTOR
# ============================================================


selected_output = st.selectbox(

    "Select Prediction Output",

    OUTPUT_FEATURES

)





# ============================================================
# CREATE EXPLAINABILITY TABS
# ============================================================


xai_tab1, xai_tab2, xai_tab3, xai_tab4 = st.tabs(

    [

    "🔍 SHAP Explanation",

    "💡 LIME Explanation",

    "📊 Feature Importance",

    "🔄 Permutation Importance"

    ]

)





# ============================================================
# SHAP EXPLANATION
# ============================================================


with xai_tab1:



    st.subheader(

        "SHAP Feature Contribution Analysis"

    )



    shap_file = (

        SHAP_DIR /

        f"{selected_output}_SHAP_Values.csv"

    )



    if shap_file.exists():


        shap_data = pd.read_csv(

            shap_file

        )



        st.success(

            f"Loaded {shap_file.name}"

        )



        st.dataframe(

            shap_data.head(20),

            use_container_width=True

        )



        # Identify numeric columns


        numeric_cols = (

            shap_data

            .select_dtypes(

                include=np.number

            )

            .columns

        )



        if len(numeric_cols)>0:



            shap_values = (

                shap_data[numeric_cols]

                .abs()

                .mean()

                .sort_values(

                    ascending=False

                )

                .head(15)

            )



            st.subheader(

                "Top SHAP Influencing Features"

            )



            st.bar_chart(

                shap_values

            )



    else:


        st.warning(

            f"No SHAP file found for {selected_output}"

        )







# ============================================================
# LIME EXPLANATION
# ============================================================


with xai_tab2:



    st.subheader(

        "Local Interpretable Model-Agnostic Explanations (LIME)"

    )



    lime_file=(

        LIME_DIR /

        f"{selected_output}_LIME.csv"

    )



    if lime_file.exists():


        lime_data=pd.read_csv(

            lime_file

        )


        st.success(

            f"Loaded {lime_file.name}"

        )



        st.dataframe(

            lime_data,

            use_container_width=True

        )



        # Plot contribution values


        numeric_columns=(

            lime_data

            .select_dtypes(

                include=np.number

            )

            .columns

        )



        if len(numeric_columns)>0:


            lime_plot=(

                lime_data

                [

                numeric_columns

                ]

                .iloc[0]

            )



            st.subheader(

                "Local Feature Contributions"

            )


            st.bar_chart(

                lime_plot

            )



    else:


        st.warning(

            f"No LIME file found for {selected_output}"

        )






# ============================================================
# FEATURE IMPORTANCE
# ============================================================


with xai_tab3:



    st.subheader(

        "Model Feature Importance Ranking"

    )



    feature_file=(

        FEATURE_DIR /

        f"{selected_output}_FeatureImportance.csv"

    )



    if feature_file.exists():


        feature_data=pd.read_csv(

            feature_file

        )


        st.success(

            f"Loaded {feature_file.name}"

        )



        st.dataframe(

            feature_data,

            use_container_width=True

        )



        numeric=(

            feature_data

            .select_dtypes(

                include=np.number

            )

        )



        if not numeric.empty:


            importance_values=(

                numeric.iloc[:,0]

                .head(15)

            )



            st.bar_chart(

                importance_values

            )



    else:


        st.warning(

            f"No feature importance file found for {selected_output}"

        )







# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================


with xai_tab4:



    st.subheader(

        "Permutation Importance Analysis"

    )



    perm_file_options=[

        PERM_DIR /

        f"{selected_output}_PermutationImportance.csv",


        PERM_DIR /

        f"{selected_output}_Top20Permutation.csv"

    ]



    permutation_loaded=False



    for file in perm_file_options:



        if file.exists():



            perm_data=pd.read_csv(

                file

            )



            st.success(

                f"Loaded {file.name}"

            )



            st.dataframe(

                perm_data,

                use_container_width=True

            )



            numeric=(

                perm_data

                .select_dtypes(

                    include=np.number

                )

            )



            if not numeric.empty:


                st.subheader(

                    "Permutation Ranking"

                )


                st.bar_chart(

                    numeric.iloc[:,0]

                )



            permutation_loaded=True


            break





    if not permutation_loaded:


        st.warning(

            f"No permutation importance file found for {selected_output}"

        )







# ============================================================
# XAI SUMMARY
# ============================================================


st.divider()


summary_file=(

    BASE_DIR /

    "Results" /

    "Explainability" /

    "Explainability_Summary.csv"

)



if summary_file.exists():


    st.subheader(

        "Explainability Summary"

    )


    summary=pd.read_csv(

        summary_file

    )


    st.dataframe(

        summary,

        use_container_width=True

    )





# ============================================================
# END PART D
# ============================================================

# ============================================================
# PART E:
# PDF REPORTS, DOWNLOADS,
# DEPLOYMENT POLISH AND FINAL INTEGRATION
#
# ============================================================


import io

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib.styles import getSampleStyleSheet




# ============================================================
# REPORT GENERATION FUNCTIONS
# ============================================================


def generate_pdf_report(

        prediction_data,

        compliance_data,

        efficiency_data=None

):


    buffer = io.BytesIO()



    doc = SimpleDocTemplate(

        buffer,

        title=
        "AI Distillery Effluent Treatment Report"

    )



    styles = getSampleStyleSheet()



    content=[]



    # Title

    content.append(

        Paragraph(

            "AI-Based Distillery Effluent Treatment Report",

            styles["Title"]

        )

    )


    content.append(

        Spacer(1,20)

    )



    # Project details

    content.append(

        Paragraph(

"""
Project:
Explainable Machine Learning Model for
Distillery Effluent Treatment and Management

Institution:
Kabarak University

Application:
AI Effluent Treatment Monitoring System

""",

            styles["Normal"]

        )

    )



    content.append(

        Spacer(1,20)

    )




    # Prediction table


    content.append(

        Paragraph(

            "Predicted Effluent Quality",

            styles["Heading2"]

        )

    )



    prediction_table=[

        ["Parameter","Predicted Value"]

    ]



    for col in prediction_data.columns:


        prediction_table.append(

            [

            col,

            str(

                round(

                    prediction_data[col].iloc[0],

                    3

                )

            )

            ]

        )



    table=Table(

        prediction_table

    )



    table.setStyle(

        TableStyle(

            [

            ("GRID",(0,0),(-1,-1),0.5,None)

            ]

        )

    )


    content.append(table)



    content.append(

        Spacer(1,20)

    )




    # Compliance table


    content.append(

        Paragraph(

            "Compliance Assessment",

            styles["Heading2"]

        )

    )



    compliance_table=[

        ["Parameter","Status"]

    ]



    for key,value in compliance_data.items():


        compliance_table.append(

            [

            key,

            value

            ]

        )



    table2=Table(

        compliance_table

    )


    table2.setStyle(

        TableStyle(

            [

            ("GRID",(0,0),(-1,-1),0.5,None)

            ]

        )

    )


    content.append(table2)




    # Efficiency section


    if efficiency_data is not None:


        content.append(

            Spacer(1,20)

        )


        content.append(

            Paragraph(

                "Treatment Efficiency",

                styles["Heading2"]

            )

        )


        efficiency_table=[

            [

            "Parameter",

            "Removal Efficiency (%)"

            ]

        ]



        for key,value in efficiency_data.items():


            efficiency_table.append(

                [

                key,

                str(value)

                ]

            )



        table3=Table(

            efficiency_table

        )


        table3.setStyle(

            TableStyle(

                [

                ("GRID",(0,0),(-1,-1),0.5,None)

                ]

            )

        )


        content.append(table3)





    # Timestamp


    content.append(

        Spacer(1,20)

    )


    content.append(

        Paragraph(

            f"Generated on: {datetime.datetime.now()}",

            styles["Normal"]

        )

    )



    doc.build(

        content

    )



    buffer.seek(0)


    return buffer







# ============================================================
# REPORT SECTION
# ============================================================


st.divider()


st.header(

    "📄 AI Treatment Reports and Downloads"

)





# ============================================================
# GENERATE REPORT
# ============================================================


if (

    st.session_state.prediction_result

    is not None

):


    prediction_data = (

        st.session_state.prediction_result

    )



    compliance_data = check_compliance(

        prediction_data.iloc[0].to_dict()

    )



    pdf = generate_pdf_report(

        prediction_data,

        compliance_data

    )



    st.download_button(

        label=
        "📄 Download PDF Treatment Report",

        data=pdf,

        file_name=
        "AI_Distillery_Effluent_Report.pdf",

        mime=
        "application/pdf"

    )



else:


    st.info(

"""
Run a prediction first
to generate a PDF report.

"""

    )






# ============================================================
# DOWNLOAD PREDICTION DATA
# ============================================================


if st.session_state.prediction_result is not None:


    prediction_csv=(

        st.session_state.prediction_result

        .to_csv(index=False)

        .encode("utf-8")

    )



    st.download_button(

        label=
        "⬇ Download Prediction CSV",

        data=
        prediction_csv,

        file_name=
        "effluent_prediction.csv",

        mime=
        "text/csv"

    )






# ============================================================
# DOWNLOAD EXPLAINABILITY RESULTS
# ============================================================


st.subheader(

    "Download Explainability Results"

)



xai_download_files=[

    (

    "SHAP Summary",

    BASE_DIR /

    "Results" /

    "Explainability" /

    "Explainability_Summary.csv"

    ),


    (

    "Feature Importance",

    BASE_DIR /

    "final_feature_importances.csv"

    )


]





for name,file in xai_download_files:



    if file.exists():


        with open(

            file,

            "rb"

        ) as f:



            st.download_button(

                label=
                f"⬇ Download {name}",

                data=f,

                file_name=file.name,

                mime=
                "text/csv"

            )






# ============================================================
# APPLICATION INFORMATION
# ============================================================


st.divider()


st.markdown(

"""
## About This Application

**AI-Based Explainable Machine Learning Model
for Distillery Effluent Treatment and Management**

Developed for:

**MSc Information Technology**

**Kabarak University**

---

### Machine Learning Framework

✓ Multi-output Machine Learning Regression

✓ Automated Feature Alignment

✓ Environmental Compliance Monitoring

✓ SHAP Explainability

✓ LIME Explainability

✓ Feature Importance Analysis


Application Version:

**1.0**

"""

)




# ============================================================
# FOOTER
# ============================================================


st.markdown(

"""
---

Developed by:

**Clifford Otieno Obiewa**

MSc Information Technology Research Project

Kabarak University

"""

)



# ============================================================
# END PART E
# ============================================================