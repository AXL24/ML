import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prep import load_data


st.title("  ༄༂χâγ--δựηɠ--ɱô--ɧìηɧ--δự--βáσ--ςɧấτ--ɭượηɠ--ɾượμ--ναηɠ--đỏ༂࿐ ")

# Upload CSV
X,y_ = load_data()  
st.subheader("Preview of Data")
st.dataframe(X.head(), use_container_width=True)

# Apply Z-score anomaly detection
def detect_anomalies_zscore(X, threshold=3):
    z_scores = (X - X.mean()) / X.std()
    anomalies = (z_scores.abs() > threshold).any(axis=1)
    return X[~anomalies], y_[~anomalies]

# Checkbox to enable anomaly detection
apply_anomaly = st.checkbox("Remove anomalies (Z-score threshold = 3)")

if apply_anomaly:
    filtered_X, filtered_y = detect_anomalies_zscore(X)
    st.info(f"Anomalies removed: {len(X) - len(filtered_X)} rows")
else:
    filtered_X, filtered_y = X, y_


st.subheader("Preview of Data")
st.dataframe(filtered_X.head(), use_container_width=True)

st.subheader("Summary Statistics")
st.write(filtered_X.describe(), width=500, use_container_width=True)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(filtered_X.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("Feature Distribution")
feature = st.selectbox("Select a feature to visualize distribution", filtered_X.columns)
fig, ax = plt.subplots()
sns.histplot(filtered_X[feature], kde=True, ax=ax)
st.pyplot(fig)

st.subheader("Scatter Plot")
x_axis = st.selectbox("X-axis", filtered_X.columns, index=0)
y_axis = st.selectbox("Y-axis", filtered_X.columns, index=1)
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_X, x=x_axis, y=y_axis, hue=filtered_y, palette="viridis", ax=ax)

st.pyplot(fig)

