import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import numpy as np
# Load your CSV (update path if needed)
def load_data():
    # Assuming the CSV file is in the same directory as this script
    df = pd.read_csv("wine.csv")
    df2 = df.drop_duplicates()
    X = df2.drop("quality", axis=1)
    y = df2["quality"]
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X, y, scaler

# Apply Isolation Forest for anomaly detection with feature scaling
def detect_anomalies_isolation_forest(X, y_, contamination=0.05, random_state=42):
    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    anomalies = iso_forest.fit_predict(X)
    
    # Return non-anomalous data (where anomalies == 1)
    return X[anomalies == 1], y_[anomalies == 1]

def detect_anomalies_iqr(X, y_, factor=2.5):
    """
    Detect anomalies in dataset X using the IQR method.
    Compatible with Pandas DataFrames and Series.
    """
    # Convert to numpy arrays for calculation
    X_values = X.values
    Q1 = np.percentile(X_values, 25, axis=0)
    Q3 = np.percentile(X_values, 75, axis=0)
    IQR = Q3 - Q1

    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    # Find the mask of non-anomalous rows
    mask = np.all((X_values >= lower_bound) & (X_values <= upper_bound), axis=1)

    # Apply mask by position using iloc to avoid index misalignment
    filtered_X = X.iloc[mask]
    filtered_y = y_.iloc[mask]

    return filtered_X, filtered_y
