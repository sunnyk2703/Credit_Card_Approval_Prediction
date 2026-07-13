import pandas as pd
import numpy as np

# Load datasets
application = pd.read_csv("application_record.csv")
credit = pd.read_csv("credit_record.csv")

# Display first 5 rows
print("Application Dataset:")
print(application.head())

print("\nCredit Dataset:")
print(credit.head())

# Dataset information
print("\nApplication Info:")
print(application.info())

print("\nCredit Info:")
print(credit.info())

# Missing values
print("\nMissing Values in Application Dataset:")
print(application.isnull().sum())

print("\nMissing Values in Credit Dataset:")
print(credit.isnull().sum())
# Convert negative values to positive
application['DAYS_BIRTH'] = application['DAYS_BIRTH'].abs()
application['DAYS_EMPLOYED'] = application['DAYS_EMPLOYED'].abs()

# Create a new feature
application['TOTAL_FAMILY_MEMBERS'] = (
    application['CNT_FAM_MEMBERS'] + application['CNT_CHILDREN']
)

# Merge application and credit datasets
merged_data = pd.merge(application, credit, on='ID', how='inner')

print("\nMerged Dataset Shape:")
print(merged_data.shape)

print("\nFirst 5 Rows of Merged Dataset:")
print(merged_data.head())
# Convert categorical columns to numeric values
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for column in merged_data.select_dtypes(include='object').columns:
    merged_data[column] = le.fit_transform(merged_data[column].astype(str))

print("\nCategorical columns converted successfully.")
print("\nFinal Dataset Information:")
print(merged_data.info())

print("\nFirst 5 Rows:")
print(merged_data.head())
merged_data.to_csv("cleaned_data.csv", index=False)

print("\n✅ Cleaned dataset saved successfully as cleaned_data.csv")
