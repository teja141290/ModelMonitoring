import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
from dython.nominal import associations

# Generate synthetic data
def generate_synthetic_data(start_date, num_days, drift_start_date=None):
    np.random.seed(42)
    dates = pd.date_range(start=start_date, periods=num_days, freq='D').tolist()
    num_features = 5
    cat_features = 3
    
    data = {
        'date': np.random.choice(dates, 1000),
        'target': np.random.choice([0, 1], 1000)
    }
    
    for i in range(num_features):
        data[f'num_feature_{i}'] = np.random.randn(1000)
        
    for i in range(cat_features):
        data[f'cat_feature_{i}'] = np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2])
        
    df = pd.DataFrame(data)
    
    if drift_start_date:
        drift_start_date = pd.to_datetime(drift_start_date)
        drift_indices = df['date'] >= drift_start_date
        for i in range(num_features):
            df.loc[drift_indices, f'num_feature_{i}'] += np.random.rand(drift_indices.sum()) * 0.5
            
        for i in range(cat_features):
            df.loc[drift_indices, f'cat_feature_{i}'] = np.random.choice(['A', 'B', 'C'], drift_indices.sum(), p=[0.2, 0.5, 0.3])
    
    return df

# Encode data
def encode_data(reference_data, live_data):
    encoder = LabelEncoder()
    for column in reference_data.columns:
        if reference_data[column].dtype == 'object':
            reference_data[column] = encoder.fit_transform(reference_data[column])
            live_data[column] = encoder.transform(live_data[column])
    return reference_data, live_data

# Calculate model performance metrics
def calculate_metrics(model, X, y, metric):
    if metric == 'Accuracy':
        y_pred = model.predict(X)
        return accuracy_score(y, y_pred)
    elif metric == 'Precision':
        y_pred = model.predict(X)
        return precision_score(y, y_pred)
    elif metric == 'Recall':
        y_pred = model.predict(X)
        return recall_score(y, y_pred)
    else:
        return None

# # Calculate numerical data drift
# def calculate_numerical_statistics(reference_data, live_data, feature, metric):
#     ref_data = reference_data[feature]
#     live_data = live_data[feature]
#     if metric == 'ks_test':
#         stat, p_value = ks_2samp(ref_data, live_data)
#         drifted = p_value < 0.05
#         return stat, p_value, drifted
#     else:
#         raise ValueError("Unsupported metric")

# # Calculate categorical data drift
# def calculate_categorical_statistics(reference_data, live_data, feature, metric):
#     ref_data = reference_data[feature].value_counts().values
#     live_data = live_data[feature].value_counts().values
#     if metric == 'chi_square':
#         stat, p_value, _, _ = chi2_contingency([ref_data, live_data])
#         drifted = p_value < 0.05
#         return stat, p_value, drifted
#     else:
#         raise ValueError("Unsupported metric")

def calculate_drift(reference_data, live_data, num_metric, cat_metric):
    """
    Calculate the drift between reference and live data using specified metrics for numerical and categorical features.

    Parameters:
    - reference_data: DataFrame containing the reference data.
    - live_data: DataFrame containing the live data.
    - num_metric: The numerical drift metric to use ('KS Test' or 'Wasserstein Distance').
    - cat_metric: The categorical drift metric to use ('Chi-Square' or "Cramér's V").

    Returns:
    - drift_metrics: Dictionary containing drift metrics for each feature.
    """
    drift_metrics = {}
    
    for column in reference_data.columns:
        if column == 'target' or column == 'date':
            continue

        if reference_data[column].dtype in [np.float64, np.int64]:
            if num_metric == 'KS Test':
                stat, p_value = ks_2samp(reference_data[column], live_data[column])
            elif num_metric == 'Wasserstein Distance':
                stat = wasserstein_distance(reference_data[column], live_data[column])
                p_value = np.nan  # Wasserstein distance does not have a p-value
            drifted = p_value < 0.05 if not np.isnan(p_value) else stat > 0.1  # Example threshold for Wasserstein distance
            drift_metrics[column] = (num_metric, stat, p_value, drifted)

        else:
            if cat_metric == 'Chi-Square':
                contingency_table = pd.crosstab(reference_data[column], live_data[column])
                stat, p_value, _, _ = chi2_contingency(contingency_table)
            elif cat_metric == "Cramér's V":
                combined_data = pd.concat([reference_data[[column]], live_data[[column]]], axis=0)
                assoc = associations(combined_data, nominal_columns=[column], compute_only=True)
                print(assoc)
                stat = assoc['corr'][column][column]
                p_value = np.nan  # Cramér's V does not have a p-value
            drifted = p_value < 0.05 if not np.isnan(p_value) else stat > 0.1  # Example threshold for Cramér's V
            drift_metrics[column] = (cat_metric, stat, p_value, drifted)
            
    return drift_metrics


# Split data into quarters dynamically
def split_into_quarters(data):
    data['date'] = pd.to_datetime(data['date'])
    min_date = data['date'].min()
    max_date = data['date'].max()
    date_ranges = pd.date_range(start=min_date, end=max_date, freq='Q')
    quarters = []
    if len(date_ranges) >= 2:
        for start, end in zip(date_ranges[:-1], date_ranges[1:]):
            quarters.append(data[(data['date'] >= start) & (data['date'] < end)])
        quarters.append(data[data['date'] >= date_ranges[-2]])
    else:
        quarters.append(data)
    return quarters


# Get quarter label
def get_quarter_label(start_date, quarter):
    year = start_date.year + (quarter - 1) // 4
    return f'Q{quarter % 4 or 4} {year}'

# Main function to display the model monitoring page
# def display_model_monitoring_page(reference_data, live_data):
#     st.header('Model Monitoring')

#     # Train model on reference data
#     model = RandomForestClassifier(random_state=42)
#     X_ref = reference_data.drop(['target', 'date'], axis=1)
#     y_ref = reference_data['target']
#     model.fit(X_ref, y_ref)

#     # Calculate metrics for reference and live data
#     ref_metrics = calculate_metrics(model, X_ref, y_ref)
#     X_live = live_data.drop(['target', 'date'], axis=1)
#     y_live = live_data['target']
#     live_metrics = calculate_metrics(model, X_live, y_live)

#     # Display model performance comparison
#     st.subheader('Model Performance Comparison')
#     metrics_df = pd.DataFrame([ref_metrics, live_metrics], index=['Reference Data', 'Live Data'])
#     st.write(metrics_df)

#     # Data drift summary
#     st.subheader('Data Drift Summary')
#     drift_metrics = calculate_drift(reference_data, live_data)
#     drifted_features = {feature: p_value for feature, p_value in drift_metrics.items() if p_value < 0.05}
#     drift_summary = {
#         'Total Features': len(X_ref.columns),
#         'Drifted Features': len(drifted_features),
#         'Percentage Drifted': len(drifted_features) / len(X_ref.columns) * 100
#     }
#     st.write(pd.DataFrame([drift_summary]))

#     # Important features drift
#     feature_importances = model.feature_importances_
#     important_features = np.argsort(feature_importances)[-5:]  # Top 5 important features
#     important_feature_names = [X_ref.columns[i] for i in important_features]
#     important_drifted_features = {f: drift_metrics[f] for f in important_feature_names if f in drifted_features}
#     important_drift_summary = {
#         'Total Important Features': len(important_features),
#         'Drifted Important Features': len(important_drifted_features),
#         'Percentage Drifted Important Features': len(important_drifted_features) / len(important_features) * 100
#     }
#     st.write(pd.DataFrame([important_drift_summary]))

#     # Select key metric
#     key_metric = st.selectbox('Select Key Metric', ['Accuracy', 'Precision', 'Recall'])
#     st.write(f'Key Metric Selected: {key_metric}')
#     performance_change = live_metrics[key_metric] - ref_metrics[key_metric]
#     st.write(f'Performance Change in {key_metric}: {performance_change}')

#     # Threshold for model retraining
#     performance_threshold = st.slider('Select Performance Threshold', 0.0, 1.0, 0.1)
#     drift_threshold = st.slider('Select Drift Threshold', 0.0, 1.0, 0.05)
#     retraining_required = (performance_change < -performance_threshold) or (drift_summary['Percentage Drifted'] > drift_threshold * 100)
#     st.write(f'Retraining Required: {retraining_required}')

#     # Quarterly data split
#     quarters = split_into_quarters(live_data)
#     quarter_labels = [get_quarter_label(quarters[0]['date'].min(), i + 1) for i in range(len(quarters))]
    
#     # Cumulative metrics
#     cumulative_metrics = []
#     cumulative_drifts = []
#     for i in range(len(quarters)):
#         cumulative_quarter_live = live_data[live_data['date'] < quarters[i]['date'].max()]
#         X_cumulative = cumulative_quarter_live.drop(['target', 'date'], axis=1)
#         y_cumulative = cumulative_quarter_live['target']
#         cumulative_metric = calculate_metrics(model, X_cumulative, y_cumulative)[key_metric]
#         cumulative_metrics.append(cumulative_metric)
        
#         cumulative_drift = calculate_drift(reference_data, cumulative_quarter_live)
#         cumulative_drift_percentage = sum(p < 0.05 for p in cumulative_drift.values()) / len(cumulative_drift) * 100
#         cumulative_drifts.append(cumulative_drift_percentage)
    
#     fig_metrics, ax1 = plt.subplots()
#     ax1.plot(quarter_labels, cumulative_metrics, marker='o', label='Cumulative Metrics')
#     ax1.set_xlabel('Quarter')
#     ax1.set_ylabel(f'Cumulative {key_metric}')
#     ax1.legend()

#     fig_drift, ax2 = plt.subplots()
#     ax2.plot(quarter_labels, cumulative_drifts, marker='o', label='Cumulative Data Drift', color='orange')
#     ax2.set_xlabel('Quarter')
#     ax2.set_ylabel('Cumulative Data Drift (%)')
#     ax2.legend()

#     st.pyplot(fig_metrics)
#     st.pyplot(fig_drift)

def display_model_monitoring_page(reference_data, live_data):
    st.header('Model Monitoring')

    # Train model on reference data
    model = RandomForestClassifier(random_state=42)
    X_ref = reference_data.drop(['target', 'date'], axis=1)
    y_ref = reference_data['target']
    model.fit(X_ref, y_ref)

    # Calculate metrics for reference and live data
    ref_metrics = {
        'Accuracy': accuracy_score(y_ref, model.predict(X_ref)),
        'Precision': precision_score(y_ref, model.predict(X_ref)),
        'Recall': recall_score(y_ref, model.predict(X_ref))
    }
    X_live = live_data.drop(['target', 'date'], axis=1)
    y_live = live_data['target']
    live_metrics = {
        'Accuracy': accuracy_score(y_live, model.predict(X_live)),
        'Precision': precision_score(y_live, model.predict(X_live)),
        'Recall': recall_score(y_live, model.predict(X_live))
    }

    # Display model performance comparison
    st.subheader('Model Performance Comparison')
    metrics_df = pd.DataFrame([ref_metrics, live_metrics], index=['Reference Data', 'Live Data'])
    st.write(metrics_df)

    # Select statistical tests for numerical and categorical data
    num_metric = st.selectbox('Select Numerical Drift Metric', ['KS Test', 'Wasserstein Distance'])
    cat_metric = st.selectbox('Select Categorical Drift Metric', ['Chi-Square', "Cramér's V"])

    # Data drift summary
    st.subheader('Data Drift Summary')
    drift_metrics = calculate_drift(reference_data, live_data, num_metric, cat_metric)
    drifted_features = {feature: values for feature, values in drift_metrics.items() if values[3]}
    drift_summary = {
        'Total Features': len(X_ref.columns),
        'Drifted Features': len(drifted_features),
        'Percentage Drifted': len(drifted_features) / len(X_ref.columns) * 100
    }
    st.write(pd.DataFrame([drift_summary]))

    # Detailed drift table
    st.subheader('Detailed Drift Table')
    drift_table = pd.DataFrame.from_dict(drift_metrics, orient='index', columns=['Metric Type', 'Statistic', 'P-Value', 'Drifted'])
    st.write(drift_table)

    # Model retraining rules based on selected thresholds
    st.header('Model Retraining Rules')

    # Data drift threshold
    drift_threshold = st.slider('Select Threshold for Data Drift (%)', min_value=0.0, max_value=100.0, value=10.0, step=1.0)

    # Model performance threshold
    model_performance_metric = st.selectbox('Select Model Performance Metric', ['Accuracy', 'Precision', 'Recall'])
    model_performance_threshold = st.slider(f'Select Threshold for {model_performance_metric}', min_value=0.0, max_value=1.0, value=0.8, step=0.01)

    # Compare current data drift and model performance with thresholds
    if len(drifted_features) / len(X_ref.columns) * 100 > drift_threshold or live_metrics[model_performance_metric] < model_performance_threshold:
        st.warning('Model retraining recommended!')

    # Cumulative Model Metrics and Data Drift
    st.subheader('Cumulative Model Metrics and Data Drift')
    quarters = split_into_quarters(live_data)

    cumulative_metrics = []
    cumulative_drift = []
    quarter_labels = [get_quarter_label(quarters[0]['date'].min(), i + 1) for i in range(len(quarters))]

    for i in range(len(quarters)):
        cumulative_quarter_live = live_data[live_data['date'] < quarters[i]['date'].max()]
        X_quarter = cumulative_quarter_live.drop(['target', 'date'], axis=1)
        y_quarter = cumulative_quarter_live['target']
        quarter_metrics = calculate_metrics(model, X_quarter, y_quarter, model_performance_metric)
        cumulative_metrics.append(quarter_metrics)

        quarter_drift = calculate_drift(reference_data, cumulative_quarter_live, num_metric, cat_metric)
        drifted_features = {feature for feature, values in quarter_drift.items() if values[3]}
        cumulative_drift.append(len(drifted_features) / len(X_ref.columns) * 100)

    # Plot cumulative model metrics
    fig_metrics = plt.figure(figsize=(10, 5))
    plt.plot(quarter_labels, [metrics if isinstance(metrics, float) else metrics[model_performance_metric.lower()] for metrics in cumulative_metrics], marker='o')
    plt.title(f'Cumulative {model_performance_metric}')
    plt.xlabel('Quarter')
    plt.ylabel(model_performance_metric)
    plt.xticks(rotation=45)
    st.pyplot(fig_metrics)



    # Plot cumulative data drift
    fig_drift = plt.figure(figsize=(10, 5))
    plt.plot(quarter_labels, cumulative_drift, marker='o')
    plt.title('Cumulative Data Drift')
    plt.xlabel('Quarter')
    plt.ylabel('Percentage Drifted Features')
    plt.xticks(rotation=45)
    st.pyplot(fig_drift)




# # Calculate data drift
# def calculate_drift(reference_data, live_data):
#     drift_metrics = {}
#     for column in reference_data.columns:
#         if column == 'date' or column == 'target':
#             continue
#         if reference_data[column].dtype == 'float64':
#             stat, p_value, _ = calculate_numerical_statistics(reference_data, live_data, column, 'ks_test')
#             drift_metrics[column] = p_value
#         else:
#             stat, p_value, _ = calculate_categorical_statistics(reference_data, live_data, column, 'chi_square')
#             drift_metrics[column] = p_value
#     return drift_metrics

# Main function
def main():
    st.title('Data Quality Dashboard')
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Model Monitoring', 'Feature Level Analysis'])

    # Generate synthetic data
    reference_data = generate_synthetic_data(start_date='2023-01-01', num_days=730)
    live_data = generate_synthetic_data(start_date='2025-01-15', num_days=365, drift_start_date='2025-8-01')

    # Encode data
    reference_data, live_data = encode_data(reference_data, live_data)

    if page == 'Model Monitoring':
        display_model_monitoring_page(reference_data, live_data)
    else:
        st.write('Feature Level Analysis page is under construction.')

if __name__ == '__main__':
    main()



