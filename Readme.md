<div align="center">

# 🏠 House Price Prediction — End-to-End ML System

**A complete Machine Learning project taken from raw data to a live, deployed product** — training → REST API → interactive frontend → an LLM chatbot that can *reason about the model itself*.

[![Live App](https://img.shields.io/badge/🚀_Live_App-Streamlit-FF4B4B?style=for-the-badge)](https://house-price-prediction-lhe52kur6appcxo3cpn9dzr.streamlit.app/)
[![API](https://img.shields.io/badge/⚡_API-Render-46E3B7?style=for-the-badge)](https://house-price-prediction-lyk5.onrender.com/)

**🔗 [Try the App](https://house-price-prediction-lhe52kur6appcxo3cpn9dzr.streamlit.app/) &nbsp;|&nbsp; 🔗 [API Base URL](https://house-price-prediction-lyk5.onrender.com/)**

</div>

---

## 🎯 Project Goal

Most ML tutorials stop at a `.ipynb` notebook with a good R² score. **This project doesn't.**

The real goal here was to practice building an ML system the way it would exist in production:

- ✅ **Train** a regression model on the California Housing dataset with a clean, reproducible pipeline
- ✅ **Serve** predictions through a **FastAPI** backend, containerized and deployed on **Render**
- ✅ **Consume** that API from an interactive **Streamlit** frontend
- ✅ **Go further than a plain predictor** by wiring up a **LangChain + LangGraph agent** (powered by **Gemini 2.5 Flash**) that can answer open-ended questions about the model and the dataset by *writing and executing its own pandas code*

In short: this is a training-to-deployment ML project first, and a housing price predictor second.

---

## 🌐 Live Demo

| Layer | Link | Description |
|---|---|---|
| 🖥️ Frontend | **[Streamlit App](https://house-price-prediction-lhe52kur6appcxo3cpn9dzr.streamlit.app/)** | Predict prices + chat with the model |
| ⚙️ Backend API | **[Render Deployment](https://house-price-prediction-lyk5.onrender.com/)** | FastAPI service exposing `/predict`, `/meta`, `/chat` |

> ⚠️ The backend is hosted on Render's free tier, so the first request after inactivity may take **30–60 seconds** to spin up ("cold start"). Subsequent requests are fast.

---

## 🏗️ Architecture

```
                        ┌──────────────────────────┐
                        │   Streamlit Frontend      │
                        │   (Tab 1: Predict)        │
                        │   (Tab 2: Chat)           │
                        └────────────┬──────────────┘
                                     │ HTTPS
                                     ▼
                        ┌──────────────────────────┐
                        │      FastAPI Backend      │
                        │  /predict   /meta  /chat  │
                        └──────┬─────────────┬──────┘
                               │             │
                 ┌─────────────┘             └─────────────┐
                 ▼                                          ▼
     ┌───────────────────────┐                 ┌────────────────────────────┐
     │  RandomForestRegressor │                 │  LangGraph Agent            │
     │  (trained on           │                 │  + Gemini 2.5 Flash          │
     │   California Housing)  │                 │  + run_python tool          │
     └───────────────────────┘                 └────────────────────────────┘
```

**Flow:**
1. The **frontend** collects property details and calls the backend's `/predict` endpoint, or sends a chat message to `/chat`.
2. The **backend** loads a pre-trained, pickled `RandomForestRegressor` and returns a price estimate.
3. For chat, a **LangGraph state graph** routes the message to a **Gemini 2.5 Flash** model, which is bound to a custom `run_python` tool — letting the agent inspect the real dataset and model metadata to answer questions accurately instead of hallucinating.
4. If the LLM quota is exhausted, the chatbot gracefully **falls back to a local, metadata-driven responder** so the app never breaks for the user.

---

## ✨ Key Features

### 📊 Price Prediction
- Trained on the classic **California Housing** dataset (20,640 records, 8 features)
- Model: **Random Forest Regressor** (40 estimators, max depth 8)
- **R² ≈ 0.74**, MSE ≈ 0.34
- Clean `/predict` REST endpoint with strict `pydantic` input validation

### 💬 AI Chatbot (the core highlight)
- Built with **LangChain** + **LangGraph** (`StateGraph`, tool-calling, conditional edges)
- Powered by **`gemini-2.5-flash`** via `langchain-google-genai`
- Equipped with a **sandboxed `run_python` tool** — the agent writes and executes real pandas/numpy code against the dataset and model to answer questions like:
  - *"Which feature matters most for price?"*
  - *"What's the average house age in this dataset?"*
  - *"How accurate is the model?"*
- **Graceful degradation**: automatically detects Gemini quota/rate-limit errors and switches to a deterministic, metadata-based fallback responder — so the chat tab never shows a broken experience

### 🖥️ Frontend (Streamlit)
- Two-tab UI: **Predict Price** and **Ask the Model**
- Custom CSS for a polished, WhatsApp-style chat bubble interface
- Fully decoupled from the backend via an `API_URL` environment variable

### ⚙️ Backend (FastAPI)
- Lightweight, typed REST API with three endpoints: `/predict`, `/meta`, `/chat`
- Dockerized for reproducible deployment
- Deployed live on **Render**

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| **Modeling** | scikit-learn, pandas, numpy |
| **Backend API** | FastAPI, uvicorn, Pydantic |
| **AI Agent** | LangChain, LangGraph, `langchain-google-genai` (Gemini 2.5 Flash) |
| **Frontend** | Streamlit |
| **Deployment** | Docker, Render (backend), Streamlit Community Cloud (frontend) |
| **Language** | Python 3.10 |

---

## 📁 Project Structure

```
House-Price-Prediction/
├── backend/
│   ├── main.py              # FastAPI app: /predict, /meta, /chat routes
│   ├── model_utils.py       # Model loading + prediction logic
│   ├── chatbot.py           # LangGraph agent + Gemini + run_python tool
│   ├── artifacts/
│   │   ├── model.pkl            # Trained RandomForestRegressor
│   │   └── model_meta.json      # Metrics, feature importance, dataset stats
│   └── Dockerfile
├── frontend/
│   └── app.py                # Streamlit UI (Predict + Chat tabs)
├── notebook/                 # EDA & model training notebooks
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## 🚀 Getting Started Locally

### 1. Clone the repository
```bash
git clone https://github.com/Kaif-Hussain/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2. Set up a virtual environment & install dependencies
```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

### 4. Run the backend (FastAPI)
```bash
uvicorn backend.main:app --reload --port 8000
```
API will be live at `http://localhost:8000`

### 5. Run the frontend (Streamlit)
In a separate terminal:
```bash
API_URL=http://localhost:8000 streamlit run frontend/app.py
```

---

## 📡 API Reference

**Base URL (production):** `https://house-price-prediction-lyk5.onrender.com`

### `POST /predict`
Predict the median house value for a district.

```json
{
  "MedInc": 5.0,
  "HouseAge": 20,
  "AveRooms": 5.0,
  "AveBedrms": 1.0,
  "Population": 1000,
  "AveOccup": 3.0,
  "Latitude": 34.0,
  "Longitude": -118.0
}
```
**Response**
```json
{ "predicted_price": 2.71, "unit": "$100,000" }
```

### `GET /meta`
Returns model metadata — model type, hyperparameters, metrics, feature importances, and dataset summary statistics.

### `POST /chat`
Chat with the LangGraph-powered agent about the model or dataset.
```json
{ "message": "Which feature has the biggest impact on price?" }
```
**Response**
```json
{ "response": "MedInc (median income) is by far the strongest driver, followed by AveOccup and location." }
```

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Model | Random Forest Regressor |
| Estimators | 40 |
| Max Depth | 8 |
| R² Score | **0.739** |
| MSE | 0.342 |

**Top predictive features:** Median Income (`MedInc`) dominates by a wide margin, followed by average occupancy and geographic location (latitude/longitude).

---

## 🗺️ Roadmap

- [ ] Add CI/CD pipeline for automated testing and deployment
- [ ] Model versioning / experiment tracking (MLflow)
- [ ] Add authentication & rate limiting on the API
- [ ] Expand chatbot tool set (e.g. visualization generation)
- [ ] Add unit tests for backend endpoints

---

## 🙋 Author

**Kaif Hussain**
B.Tech ECE, NIT Patna &nbsp;|&nbsp; Aspiring MLOps Engineer

- GitHub: [@Kaif-Hussain](https://github.com/Kaif-Hussain)
- LeetCode: [Kaif__Hussain](https://leetcode.com/Kaif__Hussain)

---

<div align="center">

⭐ If you found this project interesting, consider giving it a star!

</div>