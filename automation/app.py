import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load a dummy model (for now, using a random function)
def load_model():
    return lambda x: np.random.rand(len(x))

model = load_model()

st.title("Automation Showcase: Streamlit, Docker & Tableau")

st.write("This app demonstrates automation using Streamlit, Docker, and Tableau.")

# User input
num_samples = st.slider("Select number of samples", min_value=1, max_value=100, value=10)
data = pd.DataFrame({"Feature_1": np.random.randn(num_samples), "Feature_2": np.random.randn(num_samples)})
st.write("### Input Data", data)

# Predictions
data["Prediction"] = model(data[["Feature_1", "Feature_2"]].values)
st.write("### Predictions", data)

# Save for Tableau
csv_path = "output_data.csv"
data.to_csv(csv_path, index=False)
st.write(f"Download the data for Tableau: [Download CSV](./{csv_path})")

