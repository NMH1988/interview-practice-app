from collections.abc import Iterator
from contextlib import closing
from datetime import date, timedelta
from itertools import chain

import pandas as pd
import streamlit as st

from src import llm, rate_limit
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
from src.guard import (
    GuardError,
    check_output,
    clean_input,
    is_blank,
    validate_input,
    validate_role,
)
from src.prompts import (
    INTERVIEW_TYPES,
    SENIORITY_LEVELS,
    STRATEGIES,
    STRATEGY_LABELS,
    build_messages,
    build_user_prompt,
)
from src.rate_limit import RateLimitError, check_rate_limit

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

# Chat state. Each history turn is {"role", "content" (shown in the chat), "sent" (sent to the
# LLM)}. "pending" holds a submitted message until its reply is in; "notice" is a warning or
# error to show once, since the run that sets it ends with st.rerun(): {"kind" (a key of
# NOTICE_STYLES), "text", and "unsent" (the user's message, for a copy box) when there is one}.
st.session_state.setdefault("history", [])
st.session_state.setdefault("pending", None)
st.session_state.setdefault("notice", None)
# When each request went to the LLM (rate_limit.clock() seconds), for the rate limit. Kept by
# "New session", which only clears the chat, so it cannot be used to skip the limit.
st.session_state.setdefault("request_times", [])

INTERRUPTED = "The answer was interrupted before it finished."

# How each kind of notice is shown: (st element, icon, title, copy-box caption), so the user can
# tell at a glance whether to fix the message, wait, or try again.
NOT_SENT = "Your message was not sent. Copy it from here to keep it:"
NO_ANSWER = "Your message got no answer. Copy it from here to keep it:"
NOTICE_STYLES = {
    "guard": (st.warning, "✋", "Message not sent", NOT_SENT),
    "rate_limit": (st.warning, "⏳", "Message limit reached", NOT_SENT),
    "llm": (st.error, "⚠️", "AI service problem", NO_ANSWER),
    "interrupted": (st.error, "⏹️", "Answer interrupted", NO_ANSWER),
}


def queue_message() -> None:
    """Keep the submitted chat message as pending, so this run can lock the input first."""
    st.session_state.pending = st.session_state.chat_box


def new_session() -> None:
    """Forget the chat history and anything still waiting to be sent or shown."""
    st.session_state.history = []
    st.session_state.pending = None
    st.session_state.notice = None


def reply_pieces(
    messages: list[dict],
    system_prompt: str,
    model: str,
    temperature: float,
    clean: str,
    user_prompt: str,
    request_times: list[float],
) -> Iterator[str]:
    """Stream the reply's pieces, saving the turn's outcome before Streamlit gets control back."""
    # The request goes out on this first next(), so from here a stopped run must not resend the
    # message; until the reply is in or has failed, it counts as interrupted. Every st.session_state
    # access is a stop point too (it checks for a rerun before it acts), so the notice comes first:
    # a run stopped between the two writes resends the message under a stale notice rather than
    # dropping it silently.
    # Its own name, not "notice": the module-level notice below holds the previous run's one.
    saved_notice = {"kind": "interrupted", "text": INTERRUPTED, "unsent": clean}
    st.session_state.notice = saved_notice
    st.session_state.pending = None
    received = []
    try:
        # Checks the model and the key now (a failure here sends nothing); the request itself
        # goes out on the stream's first next().
        stream = llm.stream(messages, model, temperature, DEFAULT_MAX_TOKENS)
        # Counted here, just before the request goes out (a failed or cut-short one may still
        # have spent tokens). A plain list append, and nothing from llm.stream to the request
        # touches st.session_state, so no stop point falls between counting and sending: a
        # message resent by a later run is never counted twice.
        request_times.append(rate_limit.clock())
        # closing(): when this generator is closed mid-stream, close the inner one (and its
        # connection) too, rather than leaving that to garbage collection.
        with closing(stream):
            for piece in stream:
                received.append(piece)
                yield piece
    except llm.LLMError as exc:
        # Only the fixed message: the chained SDK error holds the raw response body. The turn
        # stays out of the history, so it alternates user/assistant. Changed in place (a plain
        # dict write, not a stop point), so a rerun already waiting cannot stop the run before
        # the real error replaces "interrupted".
        saved_notice["kind"] = "llm"
        saved_notice["text"] = str(exc)
        raise
    # Saved before st.write_stream draws the final text, where a requested rerun could stop it.
    # The whole reply is checked before it is stored, so a leaked prompt never stays in the chat
    # or reaches the next request; the streamed text is replaced by the rerun that follows.
    reply = check_output("".join(received), system_prompt)
    # Read before anything changes: a stop at this read leaves the "interrupted" notice in place.
    history = st.session_state.history
    # The last stop point: it checks before it clears, and the extend after it is a plain list
    # call, so the notice is never cleared without the reply being saved.
    st.session_state.notice = None
    history.extend(
        [
            {"role": "user", "content": clean, "sent": user_prompt},
            {"role": "assistant", "content": reply, "sent": reply},
        ]
    )


with st.sidebar:
    st.header("Session settings")
    model = st.selectbox(
        "Model", ALLOWED_MODELS, index=ALLOWED_MODELS.index(DEFAULT_MODEL), key="model"
    )
    # Shows each strategy by its technique label; the key picks the system prompt.
    strategy = st.selectbox(
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
    interview_type = st.selectbox("Interview type", INTERVIEW_TYPES, key="interview_type")
    # Streamlit cuts the value to max_chars on the server too; validate_role checks it again.
    role = st.text_input("Role", DEFAULT_ROLE, max_chars=MAX_ROLE_CHARS, key="role")
    # Checked here, not on send, so the chat input is locked before anything is typed. Not
    # logged: this runs on every rerun, and only a message that is sent counts as a request.
    try:
        validate_role(role, log=False)
        role_ok = True
    except GuardError as exc:
        st.warning(str(exc), icon="✋")
        role_ok = False
    seniority = st.selectbox(
        "Seniority",
        SENIORITY_LEVELS,
        index=SENIORITY_LEVELS.index(DEFAULT_SENIORITY),
        key="seniority",
    )
    st.button("New session", on_click=new_session, icon="🔄")
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

for turn in st.session_state.history:
    st.chat_message(turn["role"]).markdown(turn["content"])

notice = st.session_state.notice
if notice is not None:
    show, icon, title, caption = NOTICE_STYLES[notice["kind"]]
    # Only str(exc) is shown, never the exception: a chained SDK error holds the raw response.
    show(notice["text"], icon=icon, title=title)
    # Every notice keeps the message the user typed, unless it was blank.
    if "unsent" in notice:
        st.caption(caption)
        st.code(notice["unsent"], language=None, wrap_lines=True)
    # Cleared only once drawn, so a run stopped mid-draw shows it on the next run instead.
    st.session_state.notice = None

# Drawn before the LLM call and locked while a reply is pending, so a second message cannot
# be sent (and cut this run short) while the first one is waiting.
st.chat_input(
    "Type your answer or question",
    key="chat_box",
    on_submit=queue_message,
    disabled=not role_ok or st.session_state.pending is not None,
)

# Chat turn: guard -> prompts -> streamed LLM reply. Every path clears "pending" and ends in
# st.rerun(), which unlocks the input. A rerun requested before the request is sent (e.g. a
# sidebar click) stops the run with the message still pending, so the next run sends it. One
# requested later stops the run at the next piece drawn; reply_pieces has already saved the
# outcome by then (reply, error or "interrupted"), so the message is never sent twice.
message = st.session_state.pending
if message is not None:
    try:
        try:
            clean = validate_input(message)
            # Also gives the cleaned role, and blocks a message that was queued before the role
            # was blanked (locking the input does not stop it).
            clean_role = validate_role(role)
            # Last, so a message the guard blocks shows the guard's reason and uses no request.
            request_times = st.session_state.request_times
            check_rate_limit(request_times, rate_limit.clock())
        except RateLimitError as exc:
            # The message was fine, so keep it in a copy box for when the wait is over.
            st.session_state.notice = {"kind": "rate_limit", "text": str(exc), "unsent": clean}
            st.session_state.pending = None
        except GuardError as exc:
            # Kept in a copy box too (a block may be a false positive, or the message only too
            # long), unless there is nothing to keep. Built first, so it is one write.
            blocked = {"kind": "guard", "text": str(exc)}
            kept = clean_input(message).strip()
            if not is_blank(kept):
                blocked["unsent"] = kept
            st.session_state.notice = blocked
            st.session_state.pending = None
        else:
            st.chat_message("user").markdown(clean)
            user_prompt = build_user_prompt(clean_role, interview_type, seniority, clean)
            system_prompt = STRATEGIES[strategy](clean_role, interview_type)
            messages = build_messages(system_prompt, st.session_state.history, user_prompt)
            pieces = reply_pieces(
                messages, system_prompt, model, temperature, clean, user_prompt, request_times
            )
            try:
                with st.chat_message("assistant"):
                    # The spinner covers the wait for the first piece (gpt-5 thinks first).
                    with st.spinner("Thinking..."):
                        first = next(pieces)
                    st.write_stream(chain([first], pieces), cursor="▌")
            except llm.LLMError:
                pass  # reply_pieces has saved the error; the rerun shows it.
            finally:
                # A run stopped mid-stream leaves the generator open; close it (and the
                # connection) now rather than whenever it is garbage-collected.
                pieces.close()
    except Exception:
        # A real bug, not Streamlit's rerun signal (a BaseException): without this the message
        # would stay pending, so every later run would lock the input and crash again.
        st.session_state.pending = None
        raise
    st.rerun()
