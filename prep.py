import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
# Load your CSV (update path if needed)
def load_data():
    # Assuming the CSV file is in the same directory as this script
    df = pd.read_csv("wine.csv")
    df2 = df.drop_duplicates()
    X = df2.drop("quality", axis=1)
    y = df2["quality"]
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X,y

# Apply Isolation Forest for anomaly detection with feature scaling
def detect_anomalies_isolation_forest(X, y_, contamination=0.05, random_state=42):
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    anomalies = iso_forest.fit_predict(X)
    
    # Return non-anomalous data (where anomalies == 1)
    return X[anomalies == 1], y_[anomalies == 1]
