# backend/chatbot.py
import pickle
import pandas as pd
import numpy as np
import io
import contextlib
import json
import os
from pathlib import Path
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage
from dotenv import load_dotenv
from backend.model_utils import meta

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ROOT_DIR = BASE_DIR.parent

load_dotenv(dotenv_path=ROOT_DIR / ".env")

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


def _run_python_code(code: str) -> str:
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {"__builtins__": {}}, dict(SAFE_GLOBALS))
    except Exception as e:
        return f"Error running code: {e}"
    result = output.getvalue().strip()
    return result if result else "Code ran but produced no printed output."


def _quota_fallback_response(message: str) -> str:
    text = message.lower()

    if any(keyword in text for keyword in ["who develops", "who developed", "who made", "who created", "who built", "your creator", "your developer", "your author", "made you", "built you", "created you"]):
        return (
            "I'm the assistant for this California housing price prediction app, currently answering "
            "from local model metadata because the hosted chatbot is rate-limited. I don't have information "
            "about the individual developer of this app — that's not something stored in the model or dataset."
        )

    if any(keyword in text for keyword in ["who are you", "what are you", "your name"]):
        return (
            f"I'm an assistant for exploring a {meta['model_type']} trained to predict "
            f"{meta['target_description']}. Gemini is temporarily unavailable, so I'm giving you the "
            f"best local answer I can from the model metadata."
        )

    if any(keyword in text for keyword in ["hi", "hello", "hey", "greetings"]) and len(text.split()) <= 4:
        return (
            "Hi! Gemini is temporarily unavailable right now, but I can still answer questions about "
            "the housing price model, its features, metrics, or the dataset."
        )

    if any(keyword in text for keyword in ["what can you do", "help", "capabilities", "what do you do"]):
        return (
            "I can answer questions about the trained model (type, metrics, feature importances) and "
            "the dataset it was trained on. If you ask something general, Gemini will answer it when "
            "the hosted chatbot is available."
        )

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
        f"Gemini is temporarily unavailable, so I'm falling back to local model metadata. "
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
    return _run_python_code(code)

tools = [run_python]
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
    temperature=0,
    max_retries=0,
).bind_tools(tools, tool_choice="auto")

SYSTEM_PROMPT = (
    f"You are a helpful assistant for a California housing price prediction app. "
    f"Answer general questions directly. Only use tools when the user asks for data, calculations, or model-specific analysis. "
    f"The trained model is a {meta['model_type']} that predicts {meta['target_description']}. "
    "Use the run_python tool only when the answer depends on the dataset or the trained model. "
    "Always compute real answers rather than guessing when the question depends on the data. "
    "Keep answers concise and non-technical unless asked."
)

def _message_value(message: Any, key: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _extract_code_from_tool_call(tool_call: Any) -> str:
    args = _message_value(tool_call, "args", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return ""
    if isinstance(args, dict):
        return str(args.get("code", ""))
    return ""


def _invoke_llm(messages: list[Any]):
    return llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)


def _answer_with_one_tool_round(message: str) -> str:
    user_messages: list[Any] = [("user", message)]
    first_response = _invoke_llm(user_messages)

    tool_calls = _message_value(first_response, "tool_calls", []) or []
    if not tool_calls:
        return _message_value(first_response, "content", "") or ""

    tool_messages: list[ToolMessage] = []
    for tool_call in tool_calls:
        tool_name = _message_value(tool_call, "name", "")
        if tool_name != "run_python":
            continue
        code = _extract_code_from_tool_call(tool_call)
        tool_call_id = _message_value(tool_call, "id", "")
        if not tool_call_id:
            continue
        tool_messages.append(ToolMessage(content=_run_python_code(code), tool_call_id=tool_call_id))

    if not tool_messages:
        return _message_value(first_response, "content", "") or ""

    follow_up_messages: list[Any] = user_messages + [first_response] + tool_messages
    final_response = _invoke_llm(follow_up_messages)
    final_content = _message_value(final_response, "content", "") or ""
    if final_content:
        return final_content
    return _message_value(first_response, "content", "") or ""

def ask_chatbot(message: str) -> str:
    try:
        return _answer_with_one_tool_round(message)
    except Exception as exc:
        if _is_quota_error(exc):
            return _quota_fallback_response(message)
        return "Sorry, the chatbot is temporarily unavailable right now. Please try again later."