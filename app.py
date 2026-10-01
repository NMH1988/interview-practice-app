from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.config import API_KEY_NAME, MissingAPIKeyError, get_api_key

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
