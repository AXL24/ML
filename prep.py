import pandas as pd

# Load your CSV (update path if needed)
def load_data():
    # Assuming the CSV file is in the same directory as this script
    df = pd.read_csv("wine.csv")
    X = df.drop("quality", axis=1)
    y = df["quality"]
    return X,y
