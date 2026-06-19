import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

st.markdown("<h1 style='text-align: center; color: #6a0dad;'>🌸 Iris Flower Classification</h1>", unsafe_allow_html=True)
st.write("This app predicts the species of an Iris flower based on its measurements!")

@st.cache_resource
def load_model():
    return joblib.load('models/iris_model.joblib')

@st.cache_resource
def load_info():
    with open('models/model_info.json', 'r') as f:
        return json.load(f)

@st.cache_resource
def load_ranges():
    with open('models/feature_ranges.json', 'r') as f:
        return json.load(f)

model = load_model()
model_info = load_info()
feature_ranges = load_ranges()

st.header("📝 Input Features")

sepal_length = st.slider("Sepal Length (cm)", float(feature_ranges['sepal_length']['min']), float(feature_ranges['sepal_length']['max']), float(feature_ranges['sepal_length']['default']), 0.1)
sepal_width = st.slider("Sepal Width (cm)", float(feature_ranges['sepal_width']['min']), float(feature_ranges['sepal_width']['max']), float(feature_ranges['sepal_width']['default']), 0.1)
petal_length = st.slider("Petal Length (cm)", float(feature_ranges['petal_length']['min']), float(feature_ranges['petal_length']['max']), float(feature_ranges['petal_length']['default']), 0.1)
petal_width = st.slider("Petal Width (cm)", float(feature_ranges['petal_width']['min']), float(feature_ranges['petal_width']['max']), float(feature_ranges['petal_width']['default']), 0.1)

if st.button("🎯 Predict Species", type="primary"):
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_features)
    prediction_proba = model.predict_proba(input_features)[0]
    predicted_class = model_info['target_names'][prediction[0]]
    
    st.success(f"### Predicted Species: **{predicted_class}** 🌸")
    
    st.subheader("Confidence Scores:")
    for i, prob in enumerate(prediction_proba):
        st.progress(float(prob), text=f"{model_info['target_names'][i]}: {prob*100:.1f}%")
