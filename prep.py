import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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
