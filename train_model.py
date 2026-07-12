import pandas as pd
import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("dataset/loan_prediction.csv")

# Remove Loan_ID
df.drop("Loan_ID", axis=1, inplace=True)

# -------------------------
# Handle Missing Values
# -------------------------

cat_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

num_cols = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

cat_imputer = SimpleImputer(strategy="most_frequent")
num_imputer = SimpleImputer(strategy="median")

df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# -------------------------
# Encode Categorical Data
# -------------------------

encoders = {}

for column in cat_cols:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder

target_encoder = LabelEncoder()
df["Loan_Status"] = target_encoder.fit_transform(df["Loan_Status"])

encoders["Loan_Status"] = target_encoder

# -------------------------
# Features & Target
# -------------------------

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# -------------------------
# Split Dataset
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# Train Model
# -------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------
# Accuracy
# -------------------------

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print(f"Model Accuracy : {accuracy*100:.2f}%")

# -------------------------
# Save Model
# -------------------------

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/model.pkl")
joblib.dump(encoders, "model/encoders.pkl")

print("model.pkl created successfully")
print("encoders.pkl created successfully")