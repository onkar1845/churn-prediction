import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import pickle
import json
df = pd.read_csv('../data/churn.csv')


# print("First 5 rows:")
# print(df.head())


# # check missing data and memory
# print("check missing data ")
# print(df.info())

# #show basic info for data distribution and outliers
# print("Dataset information")
# print(df.describe())

# #last 5 rows
# print(df.tail())

# #count rows and columns total
# print(df.shape)

# #summarry missin valuse per column
# print(df.isnull().sum())

# convert TotalCharges to numeric 
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

#handle missing values
df=df.dropna()

#Drop unnecessary column
df=df.drop('customerID', axis=1)

print("Cleaned data", df.shape)
print(df.info())

# #check missing values or blank or not num
# print(df['TotalCharges'].unique())
# print((df['TotalCharges'] == " ").sum())

# Step 1: Convert binary Yes/No columns
yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

for col in yes_no_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Step 2: Convert gender separately
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# Step 3: One-hot encoding for remaining categorical columns
df = pd.get_dummies(df, drop_first=True)

# Step 4: Convert boolean to int (IMPORTANT for ML)
df = df.astype({col: 'int' for col in df.select_dtypes('bool').columns})

print("\nFinal Data Shape:", df.shape)
print("\nAny Missing Values:", df.isnull().sum().sum())
print("\nData Types:\n", df.dtypes)

#Features (input) and Target(Output)
X= df.drop('Churn', axis=1)

#Target (output)
y=df['Churn']

print("X shape:", X.shape)
print("y shape:", y.shape)

#Train and Test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42 ) #random state= help get same result

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

#initialize the StandardScaler
scaler = StandardScaler()

#fit on training data
X_train =scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

#Step 3 EDA

df_eda = pd.read_csv('../data/churn.csv')

df_eda['TotalCharges'] = pd.to_numeric(df_eda['TotalCharges'], errors='coerce')
df_eda = df_eda.dropna()
df_eda = df_eda.drop('customerID', axis=1)

#Churn Distribution
sns.countplot(x='Churn', data=df_eda)
plt.title("Churn Distribution")
plt.show()

#box plot Tenure vs Churn
sns.boxplot(x='Churn', y='tenure', data=df_eda)
plt.title("Tenure vs Churn")
plt.show()  #low tenure more churn means lavkar service sodnar

#MonthlyCharges vs Churn
sns.boxplot(x='Churn', y='MonthlyCharges', data=df_eda)
plt.title("Monthly Charges vs Churn")
plt.show() #High charges - Higher churn probability

#Contract vs Churn
sns.countplot(x='Contract', hue='Churn', data=df_eda)
plt.title("Contract vs Churn")
plt.xticks(rotation=30)
plt.show()

#Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df_eda.corr(numeric_only=True), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Step 4 : Model Building
# 1.Logistic Regression

# Initialize model
model = LogisticRegression(class_weight='balanced', max_iter=1000)

#train model
model.fit(X_train, y_train)

#Prediction
y_pred_lr= model.predict(X_test)

print("Balanced Logistic Regression")

#Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred_lr))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

# 2. Random Forest Model 

#Initialize model
rf_model = RandomForestClassifier(n_estimators=100,class_weight='balanced', random_state=42)

#Train
rf_model.fit(X_train, y_train)

#predict
y_pred_rf_bal = rf_model.predict(X_test)

print("Balanced Random Forest")

print("Accuracy:", accuracy_score(y_test, y_pred_rf_bal))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf_bal))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf_bal))

#3. xgboost model

xgb_model = XGBClassifier(
    n_estimators=100,
    scale_pos_weight=3, #handle karnar imblance je ahet te
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

#train model

xgb_model.fit(X_train, y_train)

#predict
y_pred_xgb = xgb_model.predict(X_test)

print("XGBoost Model")

print("Accuracy:", accuracy_score(y_test, y_pred_xgb))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

#Step 6 Model Saving

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model_path = os.path.join(BASE_DIR, 'models', 'churn_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

# Create folder if not exists
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)

# Save model
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# Save scaler
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print("Model Saved Successfully")   

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

feature_path = os.path.join(BASE_DIR, 'models', 'features.json')

with open(feature_path, 'w') as f:
    json.dump(list(X.columns), f)

print("Feature names saved")