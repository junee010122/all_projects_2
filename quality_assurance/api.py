# api.py - Streamlit API
import streamlit as st
import pickle

def load_model():
    return pickle.load(open("best_model.pkl", "rb"))

def main():
    st.title("SECOM Pass/Fail Prediction")
    user_input = st.text_area("Enter feature values (comma-separated)")
    if st.button("Predict"):
        model = load_model()
        prediction = model.predict([list(map(float, user_input.split(',')))])
        st.write(f"Prediction: {'Pass' if prediction[0] == 1 else 'Fail'}")

if __name__ == "__main__":
    main()