import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("cleaned_data.csv")  # Change to your CSV file name

# Columns to encode
categorical_columns = [
    'Gender',
    'Married',
    'Education',
    'Self_Employed',
    'Property_Area'
]

encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Save encoders
joblib.dump(encoders, "encoders.pkl")

print("encoders.pkl created successfully!")
