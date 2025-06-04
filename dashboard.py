import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prep import load_data, detect_anomalies_isolation_forest
from lr import train_logistic_regression


st.title("  ༄༂χâγ--δựηɠ--ɱô--ɧìηɧ--δự--βáσ--ςɧấτ--ɭượηɠ--ɾượμ--ναηɠ--đỏ༂࿐ ")

# Upload CSV
X, y_, scaler = load_data()  
st.subheader("Preview of Data")
st.dataframe(X.head(), use_container_width=True)

# Checkbox to enable anomaly detection
apply_anomaly = st.checkbox("Remove anomalies (Isolation Forest)")

if apply_anomaly:
    filtered_X, filtered_y = detect_anomalies_isolation_forest(X,y_)
    st.info(f"Anomalies removed: {len(X) - len(filtered_X)} rows")
else:
    filtered_X, filtered_y = X, y_



# Display the filtered data
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

# Train Logistic Regression Model
clf, report = train_logistic_regression(filtered_X=filtered_X, filtered_y=filtered_y)
st.write("Classification Report:")
st.json(report)

# Add form for wine prediction
st.subheader("Predict Wine Quality")
st.write("Enter wine attributes to predict its quality:")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=15.0, value=7.0, step=0.001, format="%.3f")
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.5, step=0.001, format="%.3f")
    citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.3, step=0.001, format="%.3f")
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=15.0, value=2.0, step=0.001, format="%.3f")
    chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.1, step=0.001, format="%.3f")
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0, max_value=100, value=30, step=1)

with col2:
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0, max_value=300, value=100, step=1)
    density = st.number_input("Density", min_value=0.9, max_value=1.1, value=0.99, step=0.0001, format="%.3f")
    pH = st.number_input("pH", min_value=2.0, max_value=4.0, value=3.2, step=0.001, format="%.3f")
    sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.6, step=0.001, format="%.3f")
    alcohol = st.number_input("Alcohol", min_value=8.0, max_value=15.0, value=10.0, step=0.001, format="%.3f")

if st.button("Predict Quality"):
    # Create input array
    input_data = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    })
    
    # Scale the input data using the same scaler from training
    scaled_input = scaler.transform(input_data)
    
    # Make prediction
    prediction = clf.predict(scaled_input)[0]
    probability = clf.predict_proba(scaled_input)[0]
    
    # Display results
    st.success(f"Predicted Quality: {'High Quality' if prediction == 1 else 'Low Quality'}")
    st.write(f"Confidence: {probability[prediction]*100:.2f}%")
