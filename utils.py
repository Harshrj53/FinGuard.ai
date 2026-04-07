import pandas as pd
import numpy as np
import os

def generate_synthetic_fraud_data(num_samples=10000, output_path="data/fraud_data.csv"):
    """
    Generates a synthetic dataset for fraud detection and saves it to a CSV file.
    """
    np.random.seed(42)
    
    # Generate features
    transaction_amount = np.abs(np.random.normal(loc=150, scale=300, size=num_samples))
    transaction_amount = np.clip(transaction_amount, 1.0, 15000.0) # Clip between $1 and $15000
    
    # Time of day (0 to 23)
    transaction_time = np.random.randint(0, 24, size=num_samples)
    
    merchant_categories = ['Retail', 'Travel', 'Food', 'Entertainment', 'Online', 'Other']
    merchant_category = np.random.choice(merchant_categories, size=num_samples, p=[0.3, 0.1, 0.2, 0.1, 0.25, 0.05])
    
    payment_methods = ['Credit Card', 'Debit Card', 'Bank Transfer', 'Digital Wallet']
    payment_method = np.random.choice(payment_methods, size=num_samples, p=[0.5, 0.3, 0.05, 0.15])
    
    device_types = ['Mobile', 'Desktop', 'Tablet', 'Unknown']
    device_type = np.random.choice(device_types, size=num_samples, p=[0.6, 0.3, 0.05, 0.05])
    
    location_mismatch = np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1])
    failed_login_attempts = np.random.poisson(lam=0.2, size=num_samples)
    
    # Clip failed attempts to reasonable max
    failed_login_attempts = np.clip(failed_login_attempts, 0, 10)
    
    unusual_spending_score = np.random.uniform(0, 100, size=num_samples)
    transaction_frequency = np.random.poisson(lam=3, size=num_samples)
    account_age_days = np.random.randint(1, 3650, size=num_samples)
    is_international = np.random.choice([0, 1], size=num_samples, p=[0.95, 0.05])
    
    df = pd.DataFrame({
        'transaction_amount': transaction_amount,
        'transaction_time': transaction_time,
        'merchant_category': merchant_category,
        'payment_method': payment_method,
        'device_type': device_type,
        'location_mismatch': location_mismatch,
        'failed_login_attempts': failed_login_attempts,
        'unusual_spending_score': unusual_spending_score,
        'transaction_frequency': transaction_frequency,
        'account_age_days': account_age_days,
        'is_international': is_international
    })
    
    # Create logic for target variable (risk_flag)
    # Fraud is more likely if certain conditions are met
    fraud_prob = zeros = np.zeros(num_samples)
    
    # Base probability
    fraud_prob += 0.01 
    
    # Increase probability based on specific risky factors
    fraud_prob += np.where(df['transaction_amount'] > 3000, 0.15, 0)
    fraud_prob += np.where(df['location_mismatch'] == 1, 0.20, 0)
    fraud_prob += np.where(df['failed_login_attempts'] > 3, 0.15, 0)
    fraud_prob += np.where(df['unusual_spending_score'] > 80, 0.20, 0)
    fraud_prob += np.where(df['is_international'] == 1, 0.10, 0)
    fraud_prob += np.where((df['transaction_time'] < 5) & (df['transaction_amount'] > 500), 0.20, 0)
    fraud_prob += np.where(df['account_age_days'] < 30, 0.10, 0)
    
    # Normalize and clip probabilities
    fraud_prob = np.clip(fraud_prob, 0, 0.95)
    
    # Simulate final binary outcome
    risk_flag = np.random.binomial(n=1, p=fraud_prob)
    df['risk_flag'] = risk_flag
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Dataset generated and saved to {output_path} with {num_samples} samples. Class distribution:\n{df['risk_flag'].value_counts()}")
    return df

if __name__ == "__main__":
    generate_synthetic_fraud_data(num_samples=15000, output_path="data/fraud_data.csv")
