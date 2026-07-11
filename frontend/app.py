# frontend/app.py
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="🏠 California Housing Predictor", layout="wide")
st.title("🏠 California Housing Price Predictor")

tab1, tab2 = st.tabs(["📊 Predict", "🤖 Ask the Model"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        MedInc = st.slider("Median Income (10k USD)", 0.5, 15.0, 5.0)
        HouseAge = st.slider("House Age", 1, 52, 20)
        AveRooms = st.slider("Avg Rooms", 1.0, 15.0, 5.0)
        AveBedrms = st.slider("Avg Bedrooms", 0.5, 5.0, 1.0)
    with col2:
        Population = st.slider("Population", 3, 5000, 1000)
        AveOccup = st.slider("Avg Occupancy", 0.5, 10.0, 3.0)
        Latitude = st.slider("Latitude", 32.0, 42.0, 34.0)
        Longitude = st.slider("Longitude", -124.0, -114.0, -118.0)

    if st.button("Predict Price", type="primary"):
        payload = {
            "MedInc": MedInc, "HouseAge": HouseAge, "AveRooms": AveRooms,
            "AveBedrms": AveBedrms, "Population": Population,
            "AveOccup": AveOccup, "Latitude": Latitude, "Longitude": Longitude
        }
        try:
            res = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            res.raise_for_status()
            data = res.json()
            predicted_price = data.get("predicted_price")
            unit = data.get("unit", "$100,000")
            if predicted_price is None:
                st.error(f"Unexpected response from backend: {data}")
            else:
                st.success(f"Predicted Value: ${predicted_price:,.0f} {unit}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the backend: {e}")

with tab2:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, msg in st.session_state.chat_history:
        st.chat_message(role).write(msg)

    user_msg = st.chat_input("Ask about the model, features, or dataset...")
    if user_msg:
        st.session_state.chat_history.append(("user", user_msg))
        st.chat_message("user").write(user_msg)
        try:
            res = requests.post(f"{API_URL}/chat", json={"message": user_msg}, timeout=60)
            res.raise_for_status()
            reply = res.json()["response"]
        except requests.exceptions.RequestException as e:
            reply = f"Error contacting backend: {e}"
        st.session_state.chat_history.append(("assistant", reply))
        st.chat_message("assistant").write(reply)