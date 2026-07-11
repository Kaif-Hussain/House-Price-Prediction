# frontend/app.py
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container { padding-top: 2rem; max-width: 1100px; }

    .app-header { text-align: center; padding: 1rem 0 1.5rem 0; }
    .app-header h1 {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .app-header p { color: #9ca3af; font-size: 1rem; margin-top: 0; }

    div[data-testid="stNumberInput"] label { font-weight: 600; color: #e5e7eb !important; font-size: 0.9rem; }

    div.stButton > button {
        background: linear-gradient(135deg, #2563eb, #1e40af);
        color: white; border: none; border-radius: 12px;
        padding: 0.7rem 2rem; font-weight: 600; font-size: 1rem; width: 100%;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
    }
    div.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4); }

    .result-card {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white; border-radius: 16px; padding: 1.8rem; text-align: center; margin-top: 1rem;
    }
    .result-card .amount { font-size: 2.4rem; font-weight: 700; margin: 0.3rem 0; }
    .result-card .label { font-size: 0.85rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.05em; }

    .chat-container {
        background: #0b141a;
        border: 1px solid #2a2f32;
        border-radius: 16px;
        padding: 1.2rem;
        min-height: 420px;
        max-height: 520px;
        overflow-y: auto;
        margin-bottom: 1rem;
    }
    .bubble-row { display: flex; margin-bottom: 0.55rem; }
    .bubble-row.user { justify-content: flex-end; }
    .bubble-row.assistant { justify-content: flex-start; }
    .bubble {
        max-width: 75%; padding: 0.55rem 0.9rem; border-radius: 12px;
        font-size: 0.92rem; line-height: 1.45;
    }
    .bubble.user { background: #005c4b; color: #e9edef; border-top-right-radius: 2px; }
    .bubble.assistant { background: #202c33; color: #e9edef; border-top-left-radius: 2px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <h1>🏠 California Housing Price Predictor</h1>
    <p>Get instant price estimates and ask questions about the underlying ML model</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊  Predict Price", "💬  Ask the Model"])

# ---------------------------------------------------------------------------
# TAB 1 — Predict
# ---------------------------------------------------------------------------
with tab1:
    with st.container(border=True):
        st.markdown("##### Enter property details")
        col1, col2 = st.columns(2)
        with col1:
            MedInc = st.number_input("Median Income (in $10,000s)", min_value=0.5, max_value=15.0,
                                      value=5.0, step=0.1, help="Typical range: 0.5 – 15.0 (e.g. 5.0 = $50,000)")
            HouseAge = st.number_input("House Age (years)", min_value=1, max_value=52,
                                        value=20, step=1, help="Typical range: 1 – 52 years")
            AveRooms = st.number_input("Average Rooms per Household", min_value=1.0, max_value=20.0,
                                        value=5.0, step=0.1, help="Typical range: 1.0 – 20.0")
            AveBedrms = st.number_input("Average Bedrooms per Household", min_value=0.5, max_value=6.0,
                                         value=1.0, step=0.1, help="Typical range: 0.5 – 6.0")
        with col2:
            Population = st.number_input("Block Population", min_value=3, max_value=40000,
                                          value=1000, step=10, help="Typical range: 3 – 40,000")
            AveOccup = st.number_input("Average Occupancy per Household", min_value=0.5, max_value=20.0,
                                        value=3.0, step=0.1, help="Typical range: 0.5 – 20.0")
            Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0,
                                        value=34.0, step=0.01, help="California range: 32.0 – 42.0")
            Longitude = st.number_input("Longitude", min_value=-124.5, max_value=-114.0,
                                         value=-118.0, step=0.01, help="California range: -124.5 – -114.0")

        predict_clicked = st.button("Predict Price →", type="primary")

    if predict_clicked:
        payload = {
            "MedInc": MedInc, "HouseAge": HouseAge, "AveRooms": AveRooms,
            "AveBedrms": AveBedrms, "Population": Population,
            "AveOccup": AveOccup, "Latitude": Latitude, "Longitude": Longitude
        }
        with st.spinner("Estimating price..."):
            try:
                res = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
                res.raise_for_status()
                data = res.json()
                predicted_price = data.get("predicted_price", data.get("predicted_value"))
                if predicted_price is None:
                    raise KeyError("predicted_price")
                price = predicted_price * 100000
                st.markdown(
                    f'<div class="result-card"><div class="label">Estimated Value</div>'
                    f'<div class="amount">${price:,.0f}</div></div>',
                    unsafe_allow_html=True
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the backend: {e}")

# ---------------------------------------------------------------------------
# TAB 2 — Chat (WhatsApp style)
# ---------------------------------------------------------------------------
with tab2:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hi! Ask me anything about the model or the dataset it was trained on 🏡"}
        ]

    # Build HTML with NO leading whitespace per line (avoids markdown code-block bug)
    parts = ['<div class="chat-container">']
    for msg in st.session_state.chat_history:
        role_class = "user" if msg["role"] == "user" else "assistant"
        content = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
        parts.append(f'<div class="bubble-row {role_class}"><div class="bubble {role_class}">{content}</div></div>')
    parts.append('</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)

    user_msg = st.chat_input("Type a message...")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.spinner("Typing..."):
            try:
                res = requests.post(f"{API_URL}/chat", json={"message": user_msg}, timeout=60)
                res.raise_for_status()
                reply = res.json()["response"]
            except requests.exceptions.RequestException as e:
                reply = f"⚠️ Error contacting backend: {e}"
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()