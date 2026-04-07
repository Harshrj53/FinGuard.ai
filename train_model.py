import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
from utils import generate_synthetic_fraud_data
from preprocess import load_data, get_preprocessor, preprocess_and_split

def ensure_data():
    """Generates synthetic data if it doesn't already exist."""
    data_path = 'data/fraud_data.csv'
    if not os.path.exists(data_path):
        print("Data not found. Generating synthetic dataset...")
        generate_synthetic_fraud_data(num_samples=15000, output_path=data_path)
    return data_path

def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test data and returns a dictionary of metrics."""
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_proba) if hasattr(model, "predict_proba") else 0,
        'Confusion Matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics

def train_and_compare_models():
    """Main training routine."""
    # 1. Ensure Data
    data_path = ensure_data()
    df = load_data(data_path)
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = preprocess_and_split(df)
    print(f"Data shape - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class distribution - Train:\n{y_train.value_counts(normalize=True)}")
    print(f"Class distribution - Test:\n{y_test.value_counts(normalize=True)}")
    
    # 3. Define Models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    preprocessor = get_preprocessor()
    
    results = {}
    best_f1 = -1
    best_model_name = ""
    best_pipeline = None
    
    print("\n--- Training and Evaluating Models ---")
    
    # 4. Train, evaluate, and compare
    for name, clf in models.items():
        print(f"\nTraining {name}...")
        
        # Build pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # Fit pipeline
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics
        
        print(f"Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall:    {metrics['Recall']:.4f}")
        print(f"F1 Score:  {metrics['F1 Score']:.4f}")
        print(f"ROC-AUC:   {metrics['ROC-AUC']:.4f}")
        
        # Track the best model based on F1 Score (since it's a fraud/imbalanced problem)
        if metrics['F1 Score'] > best_f1:
            best_f1 = metrics['F1 Score']
            best_model_name = name
            best_pipeline = pipeline

    print("\n--- Model Comparison Summary ---")
    summary_df = pd.DataFrame(results).T.drop(columns=['Confusion Matrix'])
    print(summary_df.sort_values(by="F1 Score", ascending=False))
    
    print(f"\nBest Model elected: **{best_model_name}** with F1: {best_f1:.4f}")
    
    # 5. Save the best pipeline (includes preprocessor + model)
    os.makedirs('models', exist_ok=True)
    model_path = 'models/fraud_model.pkl'
    
    # We save a dictionary containing the pipeline and its stats so we can load it in Streamlit
    save_data = {
        'model_name': best_model_name,
        'pipeline': best_pipeline,
        'metrics': results, # Save all metrics for the dashboard
        'features': list(X_train.columns)
    }
    
    joblib.dump(save_data, model_path)
    print(f"\nModel bundle saved to '{model_path}'")

if __name__ == "__main__":
    train_and_compare_models()
