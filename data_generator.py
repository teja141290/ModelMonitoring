import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta

# Generate reference data
def generate_reference_data(num_samples=1000, num_features=5, num_categorical=2, random_state=42):
    # Generate numerical features
    X_num, _ = make_classification(n_samples=num_samples, n_features=num_features - num_categorical,
                                   n_informative=3, n_redundant=0, n_clusters_per_class=1,
                                   random_state=random_state)
    
    # Convert numerical features to DataFrame
    num_cols = [f'num_feature_{i}' for i in range(X_num.shape[1])]
    ref_data = pd.DataFrame(X_num, columns=num_cols)
    
    # Generate categorical features (as random categories)
    cat_cols = [f'cat_feature_{i}' for i in range(num_categorical)]
    for cat_col in cat_cols:
        ref_data[cat_col] = np.random.choice(['A', 'B', 'C'], size=num_samples)
    
    # Generate categorical target
    ref_data['target'] = np.random.choice(['class1', 'class2'], size=num_samples)
    
    # Generate dates for reference data
    ref_data['date'] = pd.date_range(start='2022-01-01', periods=num_samples, freq='D')
    
    return ref_data

reference_data = generate_reference_data()
reference_data.to_csv('data/reference_data.csv')

# Generate live data with data drift over time
def generate_live_data(reference_data, drift_rate=0.1):
    live_data = reference_data.copy()
    
    # Introduce data drift progressively over quarterly intervals
    start_date = reference_data['date'].min()
    end_date = reference_data['date'].max()
    
    current_date = start_date
    while current_date <= end_date:
        quarterly_data = reference_data[reference_data['date'] <= current_date]
        
        # Introduce drift to numerical features
        numerical_features = [col for col in quarterly_data.columns if col.startswith('num_feature_')]
        for feature in numerical_features:
            drift_amount = np.random.uniform(-drift_rate, drift_rate)
            live_data.loc[live_data['date'] == current_date, feature] += drift_amount
        
        # Introduce drift to categorical features
        categorical_features = [col for col in quarterly_data.columns if col.startswith('cat_feature_')]
        for feature in categorical_features:
            for idx, value in live_data[live_data['date'] == current_date][feature].items():
                if np.random.rand() < drift_rate:
                    live_data.at[idx, feature] = np.random.choice(['A', 'B', 'C'])
        
        # Introduce drift to categorical target
        categorical_target = 'target'
        if np.random.rand() < drift_rate:
            live_data.loc[live_data['date'] == current_date, categorical_target] = np.random.choice(['class1', 'class2'])
        
        current_date += timedelta(days=1)  # Move to the next day
    
    return live_data

live_data = generate_live_data(reference_data)
live_data.to_csv('data/live_data.csv')