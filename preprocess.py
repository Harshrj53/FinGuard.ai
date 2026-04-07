import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(filepath)

def get_preprocessor():
    """
    Creates and returns a Scikit-Learn ColumnTransformer for preprocessing.
    Numeric features are scaled, and categorical features are one-hot encoded.
    """
    numeric_features = [
        'transaction_amount', 'transaction_time', 'failed_login_attempts',
        'unusual_spending_score', 'transaction_frequency', 'account_age_days'
    ]
    
    categorical_features = ['merchant_category', 'payment_method', 'device_type']
    
    binary_features = ['location_mismatch', 'is_international'] # Already 0/1, pass through but good to be explicit
    
    # 1. Pipeline for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', 'passthrough', binary_features) # Pass binary as is
        ])
    
    return preprocessor

def preprocess_and_split(df, target_col='risk_flag', test_size=0.2, random_state=42):
    """
    Splits data into train and test sets, ignoring the target column.
    We don't fit the preprocessor here, we just split. 
    The preprocessor is fitted inside a pipeline in train_model.py.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    return X_train, X_test, y_train, y_test
