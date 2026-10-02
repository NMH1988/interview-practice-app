import gc
from pathlib import Path

import pytest
import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_requests import RerunData
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
from streamlit.runtime.state.safe_session_state import SafeSessionState
from streamlit.testing.v1 import AppTest

from src import llm, prompts
from src.config import API_KEY_NAME, DEFAULT_ROLE, DEFAULT_SENIORITY
from src.prompts import INTERVIEW_TYPES, STRATEGIES, build_user_prompt

APP = Path(__file__).resolve().parent.parent / "app.py"

pytestmark = pytest.mark.usefixtures("no_env_key")


def start() -> AppTest:
    """Run the app once with a fake key and return it, ready for interaction."""
    at = AppTest.from_file(str(APP))
    at.secrets[API_KEY_NAME] = "sk-test-not-a-real-key"
    at.run(timeout=30)
    assert not at.exception
    return at


def say(at: AppTest, text: str) -> None:
    """Submit `text` in the chat input and wait for the run (and its rerun) to finish."""
    at.chat_input[0].set_value(text).run(timeout=30)
    assert not at.exception


def shown(at: AppTest) -> list[tuple[str, str]]:
    """Return the chat as (author, text) pairs, in the order they are shown."""
    return [(msg.name, msg.markdown[0].value) for msg in at.chat_message]


def test_two_turns_are_kept_in_order(fake_llm):
    """After two turns the chat shows both questions and both replies, in order."""
    at = start()
    fake_llm.reply = "First reply."
    say(at, "First answer.")
    fake_llm.reply = "Second reply."
    say(at, "Second answer.")
    assert shown(at) == [
        ("user", "First answer."),
        ("assistant", "First reply."),
        ("user", "Second answer."),
        ("assistant", "Second reply."),
    ]
    assert [turn["content"] for turn in at.session_state.history] == [
        "First answer.",
        "First reply.",
        "Second answer.",
        "Second reply.",
    ]
    assert len(fake_llm.calls) == 2


def test_llm_gets_chosen_strategy_and_delimited_message(fake_llm):
    """The LLM gets the chosen strategy's system prompt and the tagged, built user message."""
    at = start()
    at.sidebar.selectbox(key="strategy").set_value("persona")
    at.sidebar.selectbox(key="interview_type").set_value("Technical")
    at.sidebar.text_input(key="role").set_value("Data Analyst")
    at.sidebar.selectbox(key="seniority").set_value("Senior")
    at.run(timeout=30)
    say(at, "I would use a window function.")
    assert fake_llm.calls[0]["messages"] == [
        {"role": "system", "content": STRATEGIES["persona"]("Data Analyst", "Technical")},
        {
            "role": "user",
            "content": build_user_prompt(
                "Data Analyst", "Technical", "Senior", "I would use a window function."
            ),
        },
    ]


def test_second_turn_resends_earlier_turns_inside_tags(fake_llm):
    """The second call sends the earlier turns too, with the old user message still tagged."""
    at = start()
    say(at, "First answer.")
    say(at, "Second answer.")
    system, *turns = fake_llm.calls[1]["messages"]
    assert system["role"] == "system"

    def tagged(text):
        """Build the user prompt the default settings give for `text`."""
        return build_user_prompt(DEFAULT_ROLE, INTERVIEW_TYPES[0], DEFAULT_SENIORITY, text)

    assert turns == [
        {"role": "user", "content": tagged("First answer.")},
        {"role": "assistant", "content": fake_llm.reply},
        {"role": "user", "content": tagged("Second answer.")},
    ]


def test_role_is_cleaned_before_it_reaches_the_prompts(fake_llm):
    """Extra spaces and line breaks in the role are folded before both prompts are built."""
    at = start()
    at.sidebar.text_input(key="role").set_value("  Data \n  Analyst ").run(timeout=30)
    say(at, "Hello.")
    system, user = fake_llm.calls[0]["messages"]
    assert system["content"] == STRATEGIES[next(iter(STRATEGIES))](
        "Data Analyst", INTERVIEW_TYPES[0]
    )
    assert user["content"].startswith("Role: Data Analyst\n")


def test_chat_shows_raw_text_not_the_built_prompt(fake_llm):
    """The chat shows what the user typed, not the escaped, tagged prompt sent to the LLM."""
    at = start()
    say(at, "if a < b && c:")
    assert shown(at)[0] == ("user", "if a < b && c:")
    assert "&lt;" in fake_llm.calls[0]["messages"][-1]["content"]


def test_chat_input_is_unlocked_after_the_reply(fake_llm):
    """The input is locked only while a reply is pending, so the next message can be sent."""
    at = start()
    say(at, "First answer.")
    assert not at.chat_input[0].disabled
    assert at.session_state.pending is None


@pytest.mark.parametrize(
    ("module", "name", "errors_after"),
    [(llm, "stream", 1), (prompts, "build_user_prompt", 0)],
    ids=["crash-in-llm-call", "crash-building-prompt"],
)
def test_chat_input_is_locked_during_the_turn_and_freed_after_a_crash(
    monkeypatch, fake_llm, module, name, errors_after
):
    """The input is drawn locked before the turn runs; a crash in it does not leave it locked."""

    def crash(*args, **kwargs):
        """Fail with an error the app does not handle, so the run ends right here."""
        raise RuntimeError("unexpected bug")

    # app.py looks these up again on every run, so the patch reaches it.
    monkeypatch.setattr(module, name, crash)
    at = start()
    at.chat_input[0].set_value("First answer.").run(timeout=30)
    # AppTest keeps what was drawn before the crash: the input was already locked.
    assert at.exception
    assert at.chat_input[0].disabled
    assert at.session_state.pending is None
    at.run(timeout=30)
    assert not at.exception
    assert not at.chat_input[0].disabled
    # Once the request was due, the turn counts as interrupted, so the next run says so.
    assert len(at.error) == errors_after


def test_settings_changed_mid_session_keep_history(fake_llm):
    """A strategy and role changed after a turn keep the history; the next call uses them."""
    at = start()
    say(at, "First answer.")
    at.sidebar.selectbox(key="strategy").set_value("persona")
    at.sidebar.text_input(key="role").set_value("Data Analyst")
    at.run(timeout=30)
    assert len(at.chat_message) == 2
    say(at, "Second answer.")
    system, first_user, first_reply, second_user = fake_llm.calls[1]["messages"]
    assert system["content"] == STRATEGIES["persona"]("Data Analyst", INTERVIEW_TYPES[0])
    assert first_user["content"] == build_user_prompt(
        DEFAULT_ROLE, INTERVIEW_TYPES[0], DEFAULT_SENIORITY, "First answer."
    )
    assert first_reply == {"role": "assistant", "content": fake_llm.reply}
    assert second_user["content"].startswith("Role: Data Analyst\n")


def failing_stream(pieces_before_failure):
    """Return an llm.stream stand-in that sends some pieces, then fails with a 5xx error."""

    def stream(*args, **kwargs):
        """Yield the given pieces, then fail the way llm.stream does mid-reply."""
        yield from pieces_before_failure
        raise llm.LLMServerError("The AI service is having problems right now.")

    return stream


@pytest.mark.parametrize(
    "pieces_before_failure",
    [[], ["Half of ", "a reply"]],
    ids=["before-first-piece", "mid-stream"],
)
def test_failed_turn_stays_out_of_history(monkeypatch, fake_llm, pieces_before_failure):
    """A stream that fails shows its error and the unsent text; earlier turns stay intact."""
    at = start()
    say(at, "First answer.")

    monkeypatch.setattr(llm, "stream", failing_stream(pieces_before_failure))
    say(at, "Answer that fails.")
    assert len(at.error) == 1
    assert "having problems" in at.error[0].value
    assert at.code[0].value == "Answer that fails."
    # No half reply is kept: the chat and the history hold only the earlier turn.
    assert shown(at) == [("user", "First answer."), ("assistant", fake_llm.reply)]
    assert [turn["content"] for turn in at.session_state.history] == [
        "First answer.",
        fake_llm.reply,
    ]
    assert not at.chat_input[0].disabled

    monkeypatch.setattr(llm, "stream", fake_llm.stream)
    say(at, "Third answer.")
    sent = [msg["content"] for msg in fake_llm.calls[-1]["messages"]]
    assert not any("Answer that fails." in text for text in sent)
    assert not any("Half of" in text for text in sent)
    assert not at.error


def test_streamed_pieces_become_one_reply(fake_llm):
    """A reply streamed in many pieces is the last assistant message, whole and in order."""
    fake_llm.reply = "**Good** start.\n\n- Name the result\n- Add a number"
    at = start()
    say(at, "I fixed the build.")
    assert at.chat_message[-1].name == "assistant"
    assert at.chat_message[-1].markdown[0].value == fake_llm.reply
    assert at.session_state.history[-1] == {
        "role": "assistant",
        "content": fake_llm.reply,
        "sent": fake_llm.reply,
    }


def test_stream_interrupted_by_a_rerun_is_reported_and_not_resent(monkeypatch, fake_llm):
    """A rerun mid-stream (e.g. a sidebar click) drops the turn, says so, and sends no retry."""
    at = start()
    say(at, "First answer.")

    def interrupted_stream(*args, **kwargs):
        """Send one piece, then request a rerun the way a sidebar click does mid-stream."""
        fake_llm.calls.append({"messages": args[0]})
        yield "Half of "
        st.rerun()

    monkeypatch.setattr(llm, "stream", interrupted_stream)
    say(at, "Answer that is cut off.")
    assert len(fake_llm.calls) == 2
    assert len(at.error) == 1
    assert "interrupted" in at.error[0].value
    assert at.code[0].value == "Answer that is cut off."
    assert shown(at) == [("user", "First answer."), ("assistant", fake_llm.reply)]
    assert at.session_state.pending is None
    assert not at.chat_input[0].disabled


def test_stream_paused_when_the_run_stops_is_closed_and_not_resent(monkeypatch, fake_llm):
    """A rerun while a piece is drawn closes the paused stream, says so, and sends no retry."""
    at = start()
    say(at, "First answer.")
    events = []

    def paused_stream(*args, **kwargs):
        """Yield pieces until closed, recording the call and the close."""
        fake_llm.calls.append({"messages": args[0]})
        try:
            yield "Half of "
            yield "a reply."
        finally:
            events.append("closed")

    def rerun_while_drawing(stream, **kwargs):
        """Stop the run the way a sidebar click does when Streamlit draws the next piece."""
        events.append("rerun")
        st.rerun()

    monkeypatch.setattr(llm, "stream", paused_stream)
    # app.py looks st.write_stream up on every run, so the patch reaches it.
    monkeypatch.setattr(st, "write_stream", rerun_while_drawing)
    # The paused generator sits in a reference cycle (module globals -> pieces -> its frame), so
    # the cyclic GC could close it too. With GC off, only app.py's explicit close() can. This
    # relies on AppTest's runner skipping the gc.collect() the real runner does after each run
    # (Streamlit 1.64); if an upgrade adds it, this test stops catching a missing close().
    gc_was_on = gc.isenabled()
    gc.disable()
    try:
        say(at, "Answer that is cut off.")
        seen = list(events)
    finally:
        if gc_was_on:
            gc.enable()
    assert seen == ["rerun", "closed"]
    assert len(fake_llm.calls) == 2
    assert len(at.error) == 1
    assert "interrupted" in at.error[0].value
    assert at.code[0].value == "Answer that is cut off."
    assert shown(at) == [("user", "First answer."), ("assistant", fake_llm.reply)]
    assert at.session_state.pending is None


def queue_rerun():
    """Queue a rerun the way a sidebar click does mid-run, without stopping the run yet."""
    ctx = get_script_run_ctx()
    # The run stops at its next stop point, i.e. the next st call or state access.
    ctx.script_requests.request_rerun(
        RerunData(query_string=ctx.query_string, page_script_hash=ctx.page_script_hash)
    )


def rerun_after_setting(monkeypatch, key, value):
    """Queue a rerun right after session state `key` is first set to `value`; return a flag."""
    original = SafeSessionState.__setitem__
    fired = []

    def set_then_request_rerun(self, name, new_value):
        """Set the value, then (once) queue a rerun."""
        original(self, name, new_value)
        if name == key and new_value == value and not fired:
            fired.append(True)
            queue_rerun()

    monkeypatch.setattr(SafeSessionState, "__setitem__", set_then_request_rerun)
    return fired


def test_rerun_just_after_the_message_leaves_pending_is_reported(monkeypatch, fake_llm):
    """A rerun landing just after the turn takes the message off pending still reports it."""
    at = start()
    fired = rerun_after_setting(monkeypatch, "pending", None)
    say(at, "First answer.")
    assert fired
    # Not dropped silently: the "interrupted" notice was saved before pending was cleared.
    assert len(at.error) == 1
    assert "interrupted" in at.error[0].value
    assert at.code[0].value == "First answer."
    assert at.session_state.history == []
    assert len(fake_llm.calls) == 1


def test_rerun_just_after_the_interrupted_notice_is_set_resends_once(monkeypatch, fake_llm):
    """A rerun between saving the notice and clearing pending resends the message, just once."""
    at = start()
    interrupted = {
        "kind": "error",
        "text": "The answer was interrupted before it finished.",
        "unsent": "First answer.",
    }
    fired = rerun_after_setting(monkeypatch, "notice", interrupted)
    say(at, "First answer.")
    assert fired
    # The request had not gone out, so the message stayed pending and the next run sent it.
    assert shown(at) == [("user", "First answer."), ("assistant", fake_llm.reply)]
    assert not at.error
    assert len(fake_llm.calls) == 1


def test_stream_error_with_a_rerun_waiting_shows_the_real_error(monkeypatch, fake_llm):
    """A stream that fails while a rerun is waiting still shows its own error, not "interrupted"."""
    at = start()

    def failing_with_rerun_waiting(*args, **kwargs):
        """Queue a rerun, then fail the way llm.stream does on a 5xx, before any st call."""
        queue_rerun()
        raise llm.LLMServerError("The AI service is having problems right now.")

    monkeypatch.setattr(llm, "stream", failing_with_rerun_waiting)
    say(at, "Answer that fails.")
    assert len(at.error) == 1
    assert "having problems" in at.error[0].value
    assert at.code[0].value == "Answer that fails."
    assert at.session_state.history == []


def test_rerun_just_after_the_notice_is_cleared_keeps_the_reply(monkeypatch, fake_llm):
    """A rerun landing just after the finished turn clears its notice still finds the reply."""
    at = start()
    fired = rerun_after_setting(monkeypatch, "notice", None)
    say(at, "First answer.")
    assert fired
    assert shown(at) == [("user", "First answer."), ("assistant", fake_llm.reply)]
    assert not at.error
    assert len(fake_llm.calls) == 1


def test_rerun_after_the_last_piece_keeps_the_reply(monkeypatch, fake_llm):
    """A rerun that lands after the last piece, while the final text is drawn, keeps the reply."""

    def drain_then_rerun(stream, **kwargs):
        """Read the whole stream, then stop the run before anything is drawn."""
        for _ in stream:
            pass
        st.rerun()

    # app.py looks st.write_stream up on every run, so the patch reaches it.
    monkeypatch.setattr(st, "write_stream", drain_then_rerun)
    at = start()
    say(at, "First answer.")
    assert shown(at) == [("user", "First answer."), ("assistant", fake_llm.reply)]
    assert not at.error
    assert len(fake_llm.calls) == 1


def test_new_session_clears_history(fake_llm):
    """The "New session" button empties the chat, and the next call has no earlier turns."""
    at = start()
    say(at, "First answer.")
    assert len(at.chat_message) == 2
    button = at.sidebar.button[0]
    assert button.label == "New session"
    button.click().run(timeout=30)
    assert not at.exception
    assert len(at.chat_message) == 0
    assert at.session_state.history == []
    say(at, "Fresh start.")
    assert [msg["role"] for msg in fake_llm.calls[-1]["messages"]] == ["system", "user"]
