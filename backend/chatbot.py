# backend/chatbot.py
import pickle
import pandas as pd
import numpy as np
import io
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
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


# ----------------------------------------------------------------------
# RAG knowledge base
# ----------------------------------------------------------------------
# Instead of leaning on the LLM to call the run_python tool for every
# question (each tool call costs a *second* Gemini call and burns through
# free-tier quota twice as fast), we pre-compute a small set of text
# "documents" describing the model, dataset and features. At request time
# we retrieve the most relevant snippets locally (TF-IDF, no API call) and
# inject them into the prompt. Most questions can then be answered in a
# single Gemini call. run_python remains available as a tool for anything
# that genuinely needs live computation the knowledge base doesn't cover.

FEATURE_GLOSSARY = {
    "MedInc": "Median income of households in the block group (tens of thousands of USD).",
    "HouseAge": "Median age of houses in the block group (years).",
    "AveRooms": "Average number of rooms per household.",
    "AveBedrms": "Average number of bedrooms per household.",
    "Population": "Total population of the block group.",
    "AveOccup": "Average number of household members.",
    "Latitude": "Latitude of the block group.",
    "Longitude": "Longitude of the block group.",
}


def _build_knowledge_base() -> list[str]:
    docs: list[str] = []

    docs.append(
        f"Model overview: the app uses a {meta['model_type']} with "
        f"{meta['n_estimators']} trees and max_depth={meta['max_depth']} to predict "
        f"{meta['target_description']}."
    )

    docs.append(
        f"Model performance metrics: R2 = {meta['metrics']['r2']:.3f}, "
        f"MSE = {meta['metrics']['mse']:.3f}."
    )

    ranked = sorted(meta["feature_importance"].items(), key=lambda kv: kv[1], reverse=True)
    docs.append(
        "Feature importances, most to least influential: "
        + ", ".join(f"{name} ({value:.3f})" for name, value in ranked)
    )

    docs.append(
        f"Dataset overview: {len(df):,} rows. Columns are: {', '.join(meta['features'])}. "
        f"The target predicts {meta['target_description']}."
    )

    for col, description in FEATURE_GLOSSARY.items():
        if col in df.columns:
            docs.append(f"Feature '{col}' meaning: {description}")

    try:
        stats = df.describe()
        for col in stats.columns:
            docs.append(
                f"Statistics for '{col}': mean={stats.loc['mean', col]:.3f}, "
                f"min={stats.loc['min', col]:.3f}, max={stats.loc['max', col]:.3f}, "
                f"std={stats.loc['std', col]:.3f}."
            )
    except Exception:
        pass

    return docs


KNOWLEDGE_BASE: list[str] = _build_knowledge_base()
_vectorizer = TfidfVectorizer(stop_words="english")
_kb_matrix = _vectorizer.fit_transform(KNOWLEDGE_BASE) if KNOWLEDGE_BASE else None


def retrieve_context(query: str, k: int = 4) -> str:
    """Return the k most relevant knowledge-base snippets for a query (local, no API call)."""
    if _kb_matrix is None:
        return ""
    query_vec = _vectorizer.transform([query])
    scores = cosine_similarity(query_vec, _kb_matrix).flatten()
    top_indices = scores.argsort()[::-1][:k]
    relevant = [KNOWLEDGE_BASE[i] for i in top_indices if scores[i] > 0]
    return "\n".join(f"- {doc}" for doc in relevant)


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


def _is_daily_quota_error(exc: Exception) -> bool:
    """True for a per-day (RPD) quota block, e.g. quota_id contains 'PerDay'.
    Retrying the same model won't help here — the whole point is to fall
    through to the next model in MODEL_CHAIN instead."""
    text = str(exc).lower()
    return "perday" in text.replace(" ", "") or "requests_per_day" in text or "per day" in text


def _is_transient_error(exc: Exception) -> bool:
    """Errors that are likely to succeed on a quick retry (as opposed to a
    hard per-day quota block, which won't be fixed by waiting a few seconds)."""
    text = str(exc).lower()
    markers = ["503", "unavailable", "timeout", "deadline exceeded", "internal error", "500", "connection"]
    return any(marker in text for marker in markers)

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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model fallback chain, tried in order: Gemini Flash -> Gemini Flash-Lite ->
# Groq (a different provider entirely, so it has its own independent daily
# quota — genuinely useful once both Gemini tiers are exhausted for the day).
# On a genuine daily-quota (RPD) error, we fall through to the next entry
# instead of going straight to the local metadata fallback. Configure via
# env vars, e.g.:
#   GEMINI_MODEL=gemini-2.5-flash
#   GEMINI_FALLBACK_MODELS=gemini-2.5-flash-lite
#   GROQ_MODEL=llama-3.3-70b-versatile
_gemini_primary = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_gemini_fallbacks = [
    m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash-lite").split(",") if m.strip()
]
_gemini_models = [_gemini_primary] + [m for m in _gemini_fallbacks if m != _gemini_primary]
_groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Each entry is (label, bound client), tried in this exact order.
MODEL_CHAIN: list[tuple[str, Any]] = []

for _model_name in _gemini_models:
    MODEL_CHAIN.append((
        f"gemini:{_model_name}",
        ChatGoogleGenerativeAI(
            model=_model_name,
            api_key=GOOGLE_API_KEY,
            temperature=0,
            max_retries=2,
            timeout=30,
        ).bind_tools(tools, tool_choice="auto"),
    ))

if GROQ_API_KEY:
    MODEL_CHAIN.append((
        f"groq:{_groq_model}",
        ChatGroq(
            model=_groq_model,
            api_key=GROQ_API_KEY,
            temperature=0,
            max_retries=2,
            timeout=30,
        ).bind_tools(tools, tool_choice="auto"),
    ))
else:
    print("[chatbot] GROQ_API_KEY not set — Groq fallback tier is disabled. "
          "Add GROQ_API_KEY to your .env to enable it.")

BASE_SYSTEM_PROMPT = (
    f"You are a helpful, general-purpose assistant embedded in a California housing price "
    f"prediction app. You can and should answer ANY question the user asks — general knowledge, "
    f"coding, math, explanations, advice, casual conversation, or anything else — exactly as a "
    f"capable assistant normally would. You are not limited to app-related topics; do not refuse "
    f"or deflect a question just because it isn't about this app or its dataset. "
    f"\n\n"
    f"For questions that ARE about this app specifically (the dataset, model, backend API, "
    f"frontend, or project architecture): the trained model is a {meta['model_type']} that "
    f"predicts {meta['target_description']}. You may be given a 'Relevant model/dataset "
    "information' section retrieved from a local knowledge base — use it to answer whenever it's "
    "sufficient, and don't call any tool if it already answers the question. Only use the "
    "run_python tool when the question needs a live computation the retrieved information doesn't "
    "cover (e.g. a filter, custom aggregation, or a prediction on new inputs), and always compute "
    "real answers rather than guessing when the answer depends on the data. "
    "\n\n"
    "For everything else, just answer directly from your own knowledge, the same way you would in "
    "any normal conversation — there is no local knowledge base or tool for general topics, and "
    "that's expected. "
    "Keep answers concise and non-technical unless the user asks for more depth."
)


def _build_system_prompt(message: str) -> str:
    context = retrieve_context(message)
    if not context:
        return BASE_SYSTEM_PROMPT
    return f"{BASE_SYSTEM_PROMPT}\n\nRelevant model/dataset information:\n{context}"

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


def _invoke_llm(messages: list[Any], system_prompt: str, max_attempts_per_model: int = 2):
    full_messages = [SystemMessage(content=system_prompt)] + messages
    last_exc: Exception | None = None

    for label, client in MODEL_CHAIN:
        for attempt in range(max_attempts_per_model):
            try:
                response = client.invoke(full_messages)
                if label != MODEL_CHAIN[0][0]:
                    print(f"[chatbot] Answered using fallback model '{label}'.")
                return response
            except Exception as exc:
                last_exc = exc
                is_last_attempt_for_model = attempt == max_attempts_per_model - 1
                if _is_daily_quota_error(exc):
                    # This model is done for the day — no point retrying it,
                    # move straight on to the next model in the chain.
                    print(f"[chatbot] '{label}' hit its daily quota, trying next model in chain.")
                    break
                if _is_transient_error(exc) and not is_last_attempt_for_model:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if _is_quota_error(exc) and not is_last_attempt_for_model:
                    # Likely a short per-minute burst — worth one brief retry
                    # on the same model before moving on.
                    time.sleep(3)
                    continue
                if is_last_attempt_for_model:
                    break
                raise

    # Every model in the chain failed.
    raise last_exc  # type: ignore[misc]


def _answer_with_one_tool_round(message: str) -> str:
    system_prompt = _build_system_prompt(message)
    user_messages: list[Any] = [("user", message)]
    first_response = _invoke_llm(user_messages, system_prompt)

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
    final_response = _invoke_llm(follow_up_messages, system_prompt)
    final_content = _message_value(final_response, "content", "") or ""
    if final_content:
        return final_content
    return _message_value(first_response, "content", "") or ""

def ask_chatbot(message: str) -> str:
    try:
        return _answer_with_one_tool_round(message)
    except Exception as exc:
        # Surface the real error server-side — without this you can't tell a
        # genuine daily quota cap apart from a per-minute burst limit, a bad
        # API key, or a wrong model name, all of which land here differently.
        print(f"[chatbot] Gemini call failed: {type(exc).__name__}: {exc}")
        if _is_quota_error(exc):
            return _quota_fallback_response(message)
        return "Sorry, the chatbot is temporarily unavailable right now. Please try again later."