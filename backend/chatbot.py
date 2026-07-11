# backend/chatbot.py
import pickle
import pandas as pd
import numpy as np
import io
import contextlib
import os
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from backend.model_utils import meta
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

with open(ARTIFACTS_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv(ARTIFACTS_DIR / "california_housing.csv")


def _top_feature_importances(limit: int = 3) -> str:
    items = sorted(
        meta["feature_importance"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]
    return ", ".join(f"{name} ({value:.3f})" for name, value in items)


def _local_fallback_response(message: str) -> str:
    text = message.lower()

    if any(keyword in text for keyword in ["which model", "what model", "machine learning model", "ml model", "trained model"]):
        return (
            f"The prediction model is a {meta['model_type']} with {meta['n_estimators']} trees and max_depth={meta['max_depth']}. "
            f"It predicts {meta['target_description']}."
        )

    if any(keyword in text for keyword in ["metric", "r2", "mse", "score", "accuracy"]):
        return (
            f"The model is a {meta['model_type']} with R2 = {meta['metrics']['r2']:.3f} "
            f"and MSE = {meta['metrics']['mse']:.3f}."
        )

    if any(keyword in text for keyword in ["important", "importance", "feature"]):
        return (
            f"The most influential features are { _top_feature_importances() }. "
            f"MedInc is the strongest driver of the prediction."
        )

    if any(keyword in text for keyword in ["dataset", "summary", "data"]):
        return (
            f"This dataset has {len(df):,} rows and predicts {meta['target_description']}. "
            f"Key features are: {', '.join(meta['features'])}."
        )

    if any(keyword in text for keyword in ["price", "predict", "prediction"]):
        return (
            f"Predictions are returned in units of $100,000 for {meta['target_description']}. "
            f"The strongest feature is MedInc, followed by {', '.join(name for name, _ in sorted(meta['feature_importance'].items(), key=lambda item: item[1], reverse=True)[1:3])}."
        )

    return (
        f"I’m temporarily using local model metadata because the hosted chatbot is rate-limited. "
        f"The model is a {meta['model_type']} predicting {meta['target_description']}. "
        f"Top features: { _top_feature_importances() }."
    )


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "resource_exhausted" in text or "quota" in text or "rate limit" in text or "429" in text

SAFE_GLOBALS = {"pd": pd, "np": np, "df": df, "model": model}

@tool
def run_python(code: str) -> str:
    """
    Execute Python/pandas code to answer questions about the dataset or the trained model.
    Available in scope: `df` (the housing dataset), `model` (trained sklearn pipeline,
    access via model.named_steps['model']), `pd`, `np`.
    Always end with a print(...) statement — only printed output is returned.
    """
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {"__builtins__": {}}, dict(SAFE_GLOBALS))
    except Exception as e:
        return f"Error running code: {e}"
    result = output.getvalue().strip()
    return result if result else "Code ran but produced no printed output."

tools = [run_python]
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    temperature=0,
).bind_tools(tools, tool_choice="any")

SYSTEM_PROMPT = (
    f"You are an assistant explaining a California housing price prediction model. "
    f"The trained model is a {meta['model_type']} that predicts {meta['target_description']}. "
    "You can answer ANY question about the dataset or the trained model by writing "
    "Python/pandas code and running it with the run_python tool. Always compute real "
    "answers rather than guessing. Keep answers concise and non-technical unless asked."
)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def call_model(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

agent = graph.compile()

def ask_chatbot(message: str) -> str:
    try:
        result = agent.invoke({"messages": [("user", message)]})
        return result["messages"][-1].content
    except Exception as exc:
        if _is_quota_error(exc):
            return _local_fallback_response(message)
        return "Sorry, the chatbot is temporarily unavailable right now. Please try again later."