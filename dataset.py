import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_loan_data(n_records=1000):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate base data
    countries = ['India', 'Nigeria', 'USA']
    loan_types = ['Personal', 'Business', 'Education', 'Emergency']
    loan_status = ['Active', 'Completed', 'Defaulted', 'Late']
    
    data = {
        'Customer_ID': [f'CUST{i:06d}' for i in range(1, n_records + 1)],
        'Country': np.random.choice(countries, n_records),
        'Loan_Type': np.random.choice(loan_types, n_records),
        'Loan_Amount': np.random.uniform(100, 10000, n_records).round(2),
        'Interest_Rate': np.random.uniform(8, 36, n_records).round(2),
        'Loan_Term_Months': np.random.choice([3, 6, 12, 24], n_records),
        'Credit_Score': np.random.normal(650, 100, n_records).round().astype(int),
        'Monthly_Income': np.random.lognormal(8, 0.5, n_records).round(2),
        'Debt_to_Income_Ratio': np.random.uniform(0.1, 0.6, n_records).round(3),
        'Employment_Length_Years': np.random.uniform(0, 20, n_records).round(1),
        'Current_Employment_Status': np.random.choice(['Employed', 'Self-Employed', 'Business Owner'], n_records),
        'Number_of_Dependents': np.random.choice(range(0, 6), n_records),
        'Loan_Status': np.random.choice(loan_status, n_records),
        'Days_Past_Due': np.zeros(n_records),
        'Previous_Loans': np.random.choice(range(0, 6), n_records),
        'Previous_Defaults': np.random.choice(range(0, 3), n_records, p=[0.8, 0.15, 0.05])
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Adjust Days_Past_Due based on Loan_Status
    df.loc[df['Loan_Status'] == 'Late', 'Days_Past_Due'] = np.random.randint(1, 90, size=len(df[df['Loan_Status'] == 'Late']))
    df.loc[df['Loan_Status'] == 'Defaulted', 'Days_Past_Due'] = np.random.randint(90, 180, size=len(df[df['Loan_Status'] == 'Defaulted']))
    
    # Adjust metrics by country
    country_adjustments = {
        'India': {'Income_Multiplier': 0.3, 'Interest_Addition': 5},
        'Nigeria': {'Income_Multiplier': 0.4, 'Interest_Addition': 8},
        'USA': {'Income_Multiplier': 1, 'Interest_Addition': 0}
    }
    
    for country, adjustments in country_adjustments.items():
        mask = df['Country'] == country
        df.loc[mask, 'Monthly_Income'] *= adjustments['Income_Multiplier']
        df.loc[mask, 'Interest_Rate'] += adjustments['Interest_Addition']
    
    # Calculate additional metrics
    df['Monthly_Payment'] = (df['Loan_Amount'] * (1 + df['Interest_Rate']/100)) / df['Loan_Term_Months']
    df['Risk_Score'] = (
        df['Credit_Score'] / 10 -
        df['Debt_to_Income_Ratio'] * 100 +
        df['Employment_Length_Years'] * 5 -
        df['Previous_Defaults'] * 50 +
        df['Previous_Loans'] * 10
    ).round(2)
    
    return df

# Generate and save test data
test_data = generate_loan_data(1000)
test_data.to_csv('branch_loan_data.csv', index=False)

print("Test dataset shape:", test_data.shape)
print("\nSample of generated data:")
print(test_data.head())
print("\nSummary statistics:")
print(test_data.describe())