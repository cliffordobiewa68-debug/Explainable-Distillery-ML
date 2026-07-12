# -*- coding: utf-8 -*-
import pandas as pd
#Path to your CSV file
file_path = "/Users/cliffordobiewa/Desktop/Kabarak Project/creditmodelling.csv"
#Read the csv file into a DataFrame
df = pd.read_csv(file_path)
#Display the first 5 rows
print(df.head())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# 1. LOAD DATA
df = pd.read_csv("/Users/cliffordobiewa/Desktop/Kabarak Project/creditmodelling.csv")
print("Initial Data:\n", df.head())
print("\nSummary:\n", df.info())

# 2. REMOVE DUPLICATES
df.drop_duplicates(inplace=True)
#count duplicates
duplicates = df.duplicated().sum()
print(duplicates) 

# 3. HANDLE MISSING VALUES
for col in df.columns:
    if df[col].dtype in ['float64','int64']:
        df[col].fillna(df[col].median(),inplace=True)
    else:
        df[col].fillna(df[col].mode()[0],inplace=True)
        print("\nNew missing values per column are:\n",df.isnull().sum())
        
# 4. HANDLE OUTLIERS (simple IQR cap)
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    df[col] = np.clip(df[col], lower, upper)
    print(df.shape)

# 5. CONVERT TARGET (object with 3 categories → binary)


target_col = "Credit_Score"

df[target_col] = df[target_col].astype(str)

df["binary_target"] = df[target_col].apply(lambda x: 0 if x == "Poor" else 1)

# 6. FEATURE/TARGET SPLIT
X = df.drop([target_col, "binary_target"], axis=1)
y = df["binary_target"]

# 7. IDENTIFY COLUMN TYPES AND DROP STRINGS
categorical = X.select_dtypes(include=["object"]).columns.tolist()
numeric = X.select_dtypes(include=np.number).columns.tolist()
cols_to_drop = df.select_dtypes(include=['float64','object']).columns
df_filtered = df.drop(columns=cols_to_drop)

for col in categorical:
    X[col] = X[col].astype(str)
    
print(df.columns.tolist())

# 8. PREPROCESSING PIPELINE
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric),
        ("cat", categorical_pipeline, categorical)
    ]
)

# 9. FULL MODEL PIPELINE

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=5000))
])

# 10. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

#EXPLORATORY DATA ANALYSIS

# Target distribution
df["binary_target"].value_counts().plot(kind="bar")
plt.title("Binary Target Distribution")
plt.show()

# Correlation heatmap (only numeric)
plt.imshow(df[numeric_cols].corr(), cmap="viridis", interpolation="none")
plt.colorbar()
plt.title("Numeric Feature Correlations")
plt.show()



#TRAIN MODEL
model.fit(X_train, y_train)

#MODEL EVALUATION

y_pred = model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# SAVE MODEL (for Flask use)
import joblib
joblib.dump(model, "credit_model.pkl")
print("Model saved as credit_model.pkl")

#ALLOW USERS TO USE

from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("credit_model.pkl")

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = pd.DataFrame([request.form])
        prediction = model.predict(data)[0]
        return f"Predicted Credit Status: {prediction}"
    return '''
        <form method="post">
            <label>Age</label>
            <input name="Age" />
            <label>Occupation</label>
            <input name="Occupation" />
            <label>Annual_Income</label>
            <input name="Annual_Income" />
            <label>Monthly_Inhand_Salary</label>
            <input name="Monthly_Inhand_Salary" />
            <label>Num_Bank_Accounts</label>
            <input name="Num_Bank_Accounts" />
            <label>Num_Credit_Card</label>
            <input name="Num_Credit_Card" />
            <label>Num_of_Laon</label>
            <input name="Num_of_Loan" />
            <label>Delay_from_due_date</label>
            <input name="Delay_from_due_date" />
            <label>Num_of_Delayed_Payment</label>
            <input name="Num_of_Delayed_Payment" />
            <label>Num_Credit_Inquiries</label>
            <input name="Num_Credit_Inquiries" />
            <label>Outstanding_Debt</label>
            <input name="Outstanding_Debt" />
            <label>Credit_Utilization_Ratio</label>
            <input name="Credit_Utilization_Ratio" />
            <label>Credit_History_Age</label>
            <input name="Credit_History_Age" />
            <label>Interest_Rate</label>
            <input name="Interest_Rate" />
            <label>Total_EMI_per_month</label>
            <input name="Total_EMI_per_month" />
            <label>ID</label>
            <input name="ID" />
            <label>Customer_ID</label>
            <input name="Customer_ID" />
            <label>Month</label>
            <input name="Month" />
            <label>Name</label>
            <input name="Name" />
            <label>SSN</label>
            <input name="SSN" />
            <label>Type_of_Loan</label>
            <input name="Type_of_Loan" />
            <label>Changed_Credit_Limit</label>
            <input name="Changed_Credit_Limit" />
            <label>Credit_Mix</label>
            <input name="Credit_Mix" />
            <label>Payment_of_Min_Amount</label>
            <input name="Payment_of_Min_Amount" />
            <label>Amount_invested_monthly</label>
            <input name="Amount_invested_monthly" />
            <label>Payment_Behaviour</label>
            <input name="Payment_Behaviour" />
            <label>Monthly_Balance</label>
            <input name="Monthly_Balance" />
            <button type="submit">Predict</button>
        </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)
print(df.columns.tolist())

