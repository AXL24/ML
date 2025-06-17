import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prep import load_data, detect_anomalies_isolation_forest, detect_anomalies_iqr
from lr import train_logistic_regression


st.title("  Xây dựng mô hình dự đoán chất lượng rượu 🍷")

st.image("https://i.pinimg.com/736x/d1/b1/e2/d1b1e27b74679147f55b0c4b8e0d3602.jpg")# Upload CSV
X, y_, scaler = load_data()  
st.subheader("Preview of Data")
st.dataframe(X.head(), use_container_width=True)

# Checkbox to enable anomaly detection
apply_anomaly1 = st.checkbox("Remove anomalies (Isolation Forest)")
apply_anomaly2 = st.checkbox("Remove anomalies (IQR)")

if apply_anomaly1 and apply_anomaly2:
    # If both are selected, apply Isolation Forest first, then IQR on the result
    filtered_X, filtered_y = detect_anomalies_isolation_forest(X, y_)
    st.info(f"Anomalies removed by Isolation Forest: {len(X) - len(filtered_X)} rows")
    filtered_X, filtered_y = detect_anomalies_iqr(filtered_X, filtered_y)
    st.info(f"Anomalies removed by IQR: {len(X) - len(filtered_X)} rows (after Isolation Forest)")
elif apply_anomaly1:
    filtered_X, filtered_y = detect_anomalies_isolation_forest(X, y_)
    st.info(f"Anomalies removed: {len(X) - len(filtered_X)} rows")
elif apply_anomaly2:
    filtered_X, filtered_y = detect_anomalies_iqr(X, y_)
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
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=15.0, value=st.session_state.get('fixed_acidity', 7.0), step=0.001, format="%.3f")
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=st.session_state.get('volatile_acidity', 0.5), step=0.001, format="%.3f")
    citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=st.session_state.get('citric_acid', 0.3), step=0.001, format="%.3f")
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=15.0, value=st.session_state.get('residual_sugar', 2.0), step=0.001, format="%.3f")
    chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=st.session_state.get('chlorides', 0.1), step=0.001, format="%.3f")
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0, max_value=100, value=st.session_state.get('free_sulfur_dioxide', 30), step=1)

# Add randomize button between columns
def randomize_values():
    import random
    st.session_state.fixed_acidity = random.uniform(0.0, 15.0)
    st.session_state.volatile_acidity = random.uniform(0.0, 2.0)
    st.session_state.citric_acid = random.uniform(0.0, 1.0)
    st.session_state.residual_sugar = random.uniform(0.0, 15.0)
    st.session_state.chlorides = random.uniform(0.0, 1.0)
    st.session_state.free_sulfur_dioxide = random.randint(0, 100)
    st.session_state.total_sulfur_dioxide = random.randint(0, 300)
    st.session_state.density = random.uniform(0.9, 1.1)
    st.session_state.pH = random.uniform(2.0, 4.0)
    st.session_state.sulphates = random.uniform(0.0, 2.0)
    st.session_state.alcohol = random.uniform(8.0, 15.0)

st.button("🎲 Random ", on_click=randomize_values)

with col2:
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0, max_value=300, value=st.session_state.get('total_sulfur_dioxide', 100), step=1)
    density = st.number_input("Density", min_value=0.9, max_value=1.1, value=st.session_state.get('density', 0.99), step=0.0001, format="%.3f")
    pH = st.number_input("pH", min_value=2.0, max_value=4.0, value=st.session_state.get('pH', 3.2), step=0.001, format="%.3f")
    sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=st.session_state.get('sulphates', 0.6), step=0.001, format="%.3f")
    alcohol = st.number_input("Alcohol", min_value=8.0, max_value=15.0, value=st.session_state.get('alcohol', 10.0), step=0.001, format="%.3f")

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

