from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src import llm
from src.config import (
    ALLOWED_MODELS,
    API_KEY_NAME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_ROLE,
    DEFAULT_SENIORITY,
    DEFAULT_TEMPERATURE,
    MAX_ROLE_CHARS,
    MAX_TEMPERATURE,
    MIN_TEMPERATURE,
    MissingAPIKeyError,
    SecretsFileError,
    get_api_key,
)
from src.guard import GuardError, validate_input, validate_role
from src.prompts import INTERVIEW_TYPES, SENIORITY_LEVELS, STRATEGIES, STRATEGY_LABELS

st.set_page_config(page_title="Interview Practice", layout="wide")

# Theme colours come from .streamlit/config.toml so the cards always match the app.
primary = st.get_option("theme.primaryColor") or "#4f46e5"
card_bg = st.get_option("theme.secondaryBackgroundColor") or "#f1f2f9"
text = st.get_option("theme.textColor") or "#1f2430"

st.markdown(
    f"""
    <style>
    [data-testid="stMetric"] {{
        background: {card_bg};
        border-left: 4px solid {primary};
        border-radius: 8px;
        padding: 12px 16px;
    }}
    [data-testid="stMetricLabel"] {{ color: {text}; opacity: 0.75; }}
    [data-testid="stMetricValue"] {{ color: {primary}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Interview Practice")

try:
    get_api_key()
except SecretsFileError:
    st.error(
        "**`.streamlit/secrets.toml` could not be parsed.** Check that the key is in quotes, "
        f'for example `{API_KEY_NAME} = "sk-or-v1-..."`, then reload this page.',
        icon="🔑",
    )
    st.stop()
except MissingAPIKeyError:
    st.error(
        f"**OpenRouter API key missing.** Add `{API_KEY_NAME}` to `.streamlit/secrets.toml` "
        "(copy `.streamlit/secrets.toml.example`) or set it as an environment variable, "
        "then reload this page. See the README for setup steps.",
        icon="🔑",
    )
    st.stop()


@st.cache_data
def load_sessions() -> pd.DataFrame:
    """Return 60 days of made-up practice sessions for the dashboard."""
    # Placeholder data: replace with your real practice-session records.
    days = pd.date_range(end=date.today(), periods=60, freq="D")
    return pd.DataFrame(
        {
            "date": days,
            "questions": [3 + (i * 7) % 6 for i in range(60)],
            "score": [60 + (i * 11) % 40 for i in range(60)],
        }
    )


df = load_sessions()

with st.sidebar:
    st.header("Session settings")
    model = st.selectbox(
        "Model", ALLOWED_MODELS, index=ALLOWED_MODELS.index(DEFAULT_MODEL), key="model"
    )
    # Shows each strategy by its technique label; T5.2 sends the chosen one to the LLM.
    st.selectbox(
        "Prompt strategy",
        list(STRATEGIES),
        format_func=STRATEGY_LABELS.__getitem__,
        key="strategy",
    )
    temperature = st.slider(
        "Temperature",
        MIN_TEMPERATURE,
        MAX_TEMPERATURE,
        DEFAULT_TEMPERATURE,
        step=0.1,
        key="temperature",
    )
    st.selectbox("Interview type", INTERVIEW_TYPES, key="interview_type")
    # Streamlit cuts the value to max_chars on the server too; validate_role checks it again.
    role = st.text_input("Role", DEFAULT_ROLE, max_chars=MAX_ROLE_CHARS, key="role")
    st.selectbox(
        "Seniority",
        SENIORITY_LEVELS,
        index=SENIORITY_LEVELS.index(DEFAULT_SENIORITY),
        key="seniority",
    )
    st.header("Filters")
    picked = st.date_input(
        "Date range",
        value=(date.today() - timedelta(days=29), date.today()),
        min_value=df["date"].min().date(),
        max_value=df["date"].max().date(),
    )

# While the user is mid-selection Streamlit returns a single date.
if len(picked) == 2:
    start, end = picked
    filtered = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)]
else:
    filtered = df.iloc[0:0]

c1, c2, c3 = st.columns(3)
c1.metric("Sessions", len(filtered))
c2.metric("Questions answered", int(filtered["questions"].sum()))
c3.metric("Average score", f"{filtered['score'].mean():.0f}" if len(filtered) else "-")

st.line_chart(filtered.set_index("date")["score"])

# Minimal chat turn so the input guard runs before any API call; T5.2 replaces it with the
# full flow (prompts, history, streaming).
message = st.chat_input("Type your answer or question")
if message is not None:
    try:
        clean = validate_input(message)
        # The role is checked before any call; T5.2 sends the cleaned role in the prompts.
        validate_role(role)
    except GuardError as exc:
        st.warning(str(exc), icon="✋")
    else:
        st.chat_message("user").markdown(clean)
        try:
            reply = llm.complete(
                [{"role": "user", "content": clean}],
                model,
                temperature,
                DEFAULT_MAX_TOKENS,
            )
        except llm.LLMError as exc:
            # Only the fixed message: the chained SDK error holds the raw response body.
            st.error(str(exc))
        else:
            st.chat_message("assistant").markdown(reply)
