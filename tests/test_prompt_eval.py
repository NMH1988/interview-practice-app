import io
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import httpx2
import openai
import pytest
from streamlit.testing.v1 import AppTest

from scripts import prompt_eval
from scripts.prompt_eval import (
    CRITERIA,
    DEFAULT_INPUTS,
    INPUT_FIELDS,
    OPTIONAL_FIELDS,
    EvalInput,
    EvalInputError,
    Job,
    Result,
    RunSettings,
    budget,
    build_jobs,
    load_inputs,
    render_report,
    run_job,
    select_inputs,
)
from src import llm
from src.config import (
    API_KEY_NAME,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    MAX_INPUT_CHARS,
    MAX_JD_CHARS,
    MAX_TOKENS_BY_EFFORT,
    MAX_TOKENS_CAP,
    REASONING_EFFORTS,
)
from src.prompts import (
    JD_ANALYSIS,
    STRATEGIES,
    STRATEGY_LABELS,
    build_messages,
    build_user_prompt,
)
from tests.conftest import FakeLLM
from tests.injection_samples import ATTACKS

SAMPLE = [
    {
        "id": "beh-1",
        "role": "Data Analyst",
        "interview_type": "Behavioural",
        "seniority": "Junior",
        "text": "I once fixed a broken weekly report by rewriting its SQL query.",
    },
    {
        "id": "tech-1",
        "role": "Backend Developer",
        "interview_type": "Technical",
        "seniority": "Senior",
        "text": "A hash map gives O(1) average lookups because keys are hashed to buckets.",
    },
]
ITEM = EvalInput(**SAMPLE[0])
APP = Path(__file__).resolve().parent.parent / "app.py"
# Longer than the normal limit but within the job-description one, and harmless to the guard.
LONG_JD = ("We need someone to build weekly sales reports in SQL. " * 60)[: MAX_INPUT_CHARS + 1000]
FAKE_USAGE_TEXT = "Prompt 1,234 · Completion 567 (reasoning 320) · Total 1,801 tokens"


def _write(tmp_path, entries):
    """Write `entries` as the inputs JSON file and return its path."""
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _settings(strategies=tuple(STRATEGIES), planned=None, results=(), efforts=("medium",)):
    """Return RunSettings for a one-run report at the given efforts and the app's budgets."""
    return RunSettings(
        model=DEFAULT_MODEL,
        strategies=tuple(strategies),
        efforts=tuple(efforts),
        runs=1,
        max_tokens=None,
        commit="abc1234",
        started=datetime(2026, 10, 2, 21, 30),
        planned=len(results) if planned is None else planned,
    )


@pytest.fixture
def no_client(monkeypatch):
    """Make `llm.make_client` return a stand-in, so main() needs no API key."""
    monkeypatch.setattr(llm, "make_client", lambda: "fake-client")


def test_load_inputs_reads_a_valid_file(tmp_path):
    """A valid file gives one EvalInput per entry, in file order."""
    assert load_inputs(_write(tmp_path, SAMPLE)) == [EvalInput(**e) for e in SAMPLE]


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ({"id": "x"}, "non-empty JSON list"),
        ([], "non-empty JSON list"),
        (["text"], "must be a JSON object"),
        ([{**SAMPLE[0], "txt": "typo"}], "unknown field"),
        ([{k: v for k, v in SAMPLE[0].items() if k != "text"}], "missing or empty field.*text"),
        ([{**SAMPLE[0], "role": "  "}], "missing or empty field.*role"),
        ([{**SAMPLE[0], "seniority": 3}], "missing or empty field.*seniority"),
        ([{**SAMPLE[0], "text": "TODO my answer"}], "TODO placeholder"),
        ([{**SAMPLE[0], "interview_type": "Coding"}], "interview_type must be one of"),
        ([{**SAMPLE[0], "seniority": "Intern"}], "seniority must be one of"),
        ([SAMPLE[0], {**SAMPLE[1], "id": "beh-1"}], "entry 2: id 'beh-1' is used twice"),
        ([{**SAMPLE[0], "question": " "}], "question must be non-empty text"),
        ([{**SAMPLE[0], "question": None}], "question must be non-empty text"),
        ([{**SAMPLE[0], "question": "TODO optional"}], "TODO placeholder"),
    ],
)
def test_load_inputs_rejects_bad_entries(tmp_path, entries, message):
    """Each kind of bad inputs file raises EvalInputError saying what to fix."""
    with pytest.raises(EvalInputError, match=message):
        load_inputs(_write(tmp_path, entries))


def test_load_inputs_reads_an_optional_question(tmp_path):
    """An entry may carry the coach's question; one without it has question None."""
    entries = [{**SAMPLE[0], "question": "Tell me about a time you fixed a problem."}, SAMPLE[1]]
    first, second = load_inputs(_write(tmp_path, entries))
    assert first.question == "Tell me about a time you fixed a problem."
    assert second.question is None


def test_load_inputs_reads_a_file_saved_with_a_bom(tmp_path):
    """A file saved as "UTF-8 with BOM" (Notepad, PowerShell 5) is read like any other."""
    path = tmp_path / "bom.json"
    path.write_text(json.dumps(SAMPLE), encoding="utf-8-sig")
    assert load_inputs(path) == [EvalInput(**e) for e in SAMPLE]


def test_load_inputs_allows_todo_inside_a_real_answer(tmp_path):
    """Only a field that starts with TODO counts as unfilled; an answer may mention TODOs."""
    entries = [{**SAMPLE[0], "text": "I track every TODO comment in a ticket."}]
    assert load_inputs(_write(tmp_path, entries))[0].text == entries[0]["text"]


def test_load_inputs_rejects_missing_and_broken_files(tmp_path):
    """A missing file or invalid JSON raises EvalInputError, not a raw OS or JSON error."""
    with pytest.raises(EvalInputError, match="does not exist"):
        load_inputs(tmp_path / "nope.json")
    broken = tmp_path / "broken.json"
    broken.write_text("[{", encoding="utf-8")
    with pytest.raises(EvalInputError, match="not valid JSON"):
        load_inputs(broken)


def test_inputs_file_uses_the_expected_fields():
    """The checked-in inputs file is a list of objects with the required fields, and no others."""
    entries = json.loads(DEFAULT_INPUTS.read_text(encoding="utf-8-sig"))
    assert entries
    for entry in entries:
        assert set(INPUT_FIELDS) <= set(entry) <= set(INPUT_FIELDS) | set(OPTIONAL_FIELDS)


def test_select_inputs_keeps_the_chosen_ids_in_file_order():
    """--ids picks inputs by id in the file's order; no ids means every input."""
    inputs = [EvalInput(**e) for e in SAMPLE]
    assert select_inputs(inputs, ["tech-1", "beh-1"]) == inputs
    assert select_inputs(inputs, ["tech-1"]) == [inputs[1]]
    assert select_inputs(inputs, None) == inputs


def test_select_inputs_rejects_an_unknown_id():
    """A mistyped id fails up front instead of silently running fewer inputs."""
    with pytest.raises(EvalInputError, match="no input with id.*tech-2"):
        select_inputs([EvalInput(**e) for e in SAMPLE], ["beh-1", "tech-2"])


def test_build_jobs_covers_every_combination_once():
    """Jobs cover inputs x efforts x runs x strategies, each combination exactly once."""
    inputs = [EvalInput(**e) for e in SAMPLE]
    jobs = build_jobs(inputs, ["zero_shot", "persona"], ["minimal", "high"], runs=3)
    assert len(jobs) == 2 * 2 * 3 * 2
    assert len({(j.item.id, j.strategy, j.effort, j.run) for j in jobs}) == len(jobs)
    # Grouped so one input's strategies sit next to each other for scoring.
    assert [j.strategy for j in jobs[:2]] == ["zero_shot", "persona"]


@pytest.mark.parametrize("effort", REASONING_EFFORTS)
def test_budget_is_the_apps_unless_set(effort):
    """Without --max-tokens each effort gets the app's budget; with it, the value given."""
    assert budget(effort, None) == MAX_TOKENS_BY_EFFORT[effort]
    assert budget(effort, 1234) == 1234


@pytest.mark.parametrize("strategy", list(STRATEGIES))
def test_run_job_sends_the_same_messages_as_the_app(fake_llm, strategy):
    """A job sends the strategy's system prompt and the app's user prompt, with its settings."""
    result = run_job(Job(ITEM, strategy, "high", 1), "openai/gpt-5-nano", 500, client="c")
    system = STRATEGIES[strategy](ITEM.role, ITEM.interview_type)
    user = build_user_prompt(ITEM.role, ITEM.interview_type, ITEM.seniority, ITEM.text)
    assert fake_llm.calls == [
        {
            "messages": build_messages(system, [], user),
            "model": "openai/gpt-5-nano",
            "reasoning_effort": "high",
            "max_tokens": 500,
            "client": "c",
        }
    ]
    assert result == Result(
        Job(ITEM, strategy, "high", 1), fake_llm.reply, result.seconds, usage=FakeLLM.usage
    )


@pytest.mark.parametrize("effort", REASONING_EFFORTS)
def test_run_job_sends_each_effort_the_apps_budget(fake_llm, effort):
    """With no max_tokens given, a job sends its effort's budget, as the app does."""
    run_job(Job(ITEM, "zero_shot", effort, 1), DEFAULT_MODEL)
    assert fake_llm.calls[0]["max_tokens"] == MAX_TOKENS_BY_EFFORT[effort]


def test_run_job_sends_what_the_app_sends_for_raw_text(fake_llm, no_env_key):
    """For untidy text and role, the script sends what the real app sends and reports its usage."""
    raw = EvalInput(
        id="raw",
        # The control character is only removed by validate_role, not by the prompt builders.
        role="  Data\x07\tAnalyst ",
        interview_type="Technical",
        seniority="Senior",
        text="  I would use a window function.\r\nThen sort.\x07  ",
    )
    at = AppTest.from_file(str(APP))
    at.secrets[API_KEY_NAME] = "sk-test-not-a-real-key"
    at.run(timeout=30)
    at.sidebar.selectbox(key="strategy").set_value("persona")
    at.sidebar.selectbox(key="interview_type").set_value(raw.interview_type)
    at.sidebar.text_input(key="role").set_value(raw.role)
    at.sidebar.selectbox(key="seniority").set_value(raw.seniority)
    at.run(timeout=30)
    at.chat_input[0].set_value(raw.text).run(timeout=30)
    assert not at.exception
    (app_call,) = fake_llm.calls
    result = run_job(Job(raw, "persona", DEFAULT_REASONING_EFFORT, 1), DEFAULT_MODEL)
    script_call = fake_llm.calls[1]
    assert script_call["messages"] == app_call["messages"]
    assert script_call["reasoning_effort"] == app_call["reasoning_effort"]
    assert script_call["max_tokens"] == app_call["max_tokens"]
    # The cleaned text and role were sent, not the raw ones.
    assert "\x07" not in str(app_call["messages"])
    assert "Role: Data Analyst" in app_call["messages"][1]["content"]
    # The report's token line reads like the app's "Token usage" line for the same counts.
    (expander,) = [e for e in at.chat_message[1].expander if e.label == "Token usage"]
    assert prompt_eval._usage_text(result.usage) == expander.caption[0].value


def test_run_job_sends_the_question_as_the_previous_turn(fake_llm):
    """With a question, the coach's question goes between the system prompt and the answer."""
    item = EvalInput(**SAMPLE[0], question="Tell me about a time you fixed a problem.")
    run_job(Job(item, "few_shot", "medium", 1), DEFAULT_MODEL)
    system = STRATEGIES["few_shot"](item.role, item.interview_type)
    user = build_user_prompt(item.role, item.interview_type, item.seniority, item.text)
    assert fake_llm.calls[0]["messages"] == [
        {"role": "system", "content": system},
        {"role": "assistant", "content": item.question},
        {"role": "user", "content": user},
    ]


def test_run_job_allows_a_job_description_as_long_as_the_app_does(fake_llm):
    """A pasted JD over the normal limit is sent in JD mode, like the app, and blocked elsewhere."""
    assert MAX_INPUT_CHARS < len(LONG_JD) <= MAX_JD_CHARS
    jd = EvalInput(**{**SAMPLE[0], "interview_type": JD_ANALYSIS, "text": LONG_JD})
    result = run_job(Job(jd, "zero_shot", "medium", 1), DEFAULT_MODEL)
    assert not result.blocked
    assert LONG_JD in fake_llm.calls[0]["messages"][-1]["content"]
    technical = EvalInput(**{**SAMPLE[1], "text": LONG_JD})
    assert run_job(Job(technical, "zero_shot", "medium", 1), DEFAULT_MODEL).blocked
    assert len(fake_llm.calls) == 1


@pytest.mark.parametrize(
    "item",
    [
        EvalInput(**{**SAMPLE[0], "text": ATTACKS[1]}),
        EvalInput(**{**SAMPLE[0], "text": "x" * (MAX_INPUT_CHARS + 1)}),
        EvalInput(**{**SAMPLE[0], "interview_type": JD_ANALYSIS, "text": "x" * (MAX_JD_CHARS + 1)}),
        EvalInput(**{**SAMPLE[0], "role": ATTACKS[0]}),
    ],
)
def test_run_job_records_a_guard_block_without_calling_the_api(fake_llm, item):
    """An input or role the guard blocks is recorded with the guard's message and never sent."""
    result = run_job(Job(item, "zero_shot", "medium", 1), DEFAULT_MODEL)
    assert result.blocked
    assert not result.reply
    assert fake_llm.calls == []


def test_run_job_records_usage_and_a_cut_off_reply(fake_llm):
    """A reply ended by max_tokens is kept and marked cut off, with its token counts."""
    fake_llm.finish_reason = "length"
    result = run_job(Job(ITEM, "zero_shot", "high", 1), DEFAULT_MODEL)
    assert result.cut_off
    assert result.reply == fake_llm.reply
    assert result.usage == FakeLLM.usage


def test_run_job_does_not_mark_a_finished_or_refused_reply_cut_off(fake_llm):
    """Only "length" means cut off, and a reply the output check replaces is not cut off."""
    assert not run_job(Job(ITEM, "persona", "medium", 1), DEFAULT_MODEL).cut_off
    fake_llm.finish_reason = "length"
    fake_llm.reply = STRATEGIES["persona"](ITEM.role, ITEM.interview_type)
    result = run_job(Job(ITEM, "persona", "medium", 1), DEFAULT_MODEL)
    assert result.refused
    assert not result.cut_off


def _failing_with(body):
    """Return a stand-in for llm.stream that fails like a 400 from OpenRouter with `body`."""
    request = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    cause = openai.BadRequestError(
        f"Error code: 400 - {body}", response=httpx2.Response(400, request=request), body=body
    )

    def failing(*args, **kwargs):
        """Fail like llm.stream's first read does when OpenRouter answers 400."""
        raise llm.LLMError("The request to the AI service failed.") from cause

    return failing


def test_run_job_records_an_llm_error_with_only_the_reason():
    """A failed request is recorded with the status and the body's reason, nothing else."""
    body = {
        "error": {"message": "max_tokens is too large", "code": 400},
        "user_id": "user_secret123",
    }
    job = Job(ITEM, "zero_shot", "low", 1)
    result = run_job(job, DEFAULT_MODEL, 100, stream=_failing_with(body))
    assert (
        result.error == "The request to the AI service failed. [HTTP 400: max_tokens is too large]"
    )
    assert not result.reply


@pytest.mark.parametrize("body", [None, "not json", {"error": "flat"}, {"error": {"code": 1}}])
def test_run_job_records_an_llm_error_whose_body_has_no_reason(body):
    """A body without error.message gives just the status, never the raw body."""
    job = Job(ITEM, "zero_shot", "low", 1)
    result = run_job(job, DEFAULT_MODEL, 100, stream=_failing_with(body))
    assert result.error == "The request to the AI service failed. [HTTP 400]"


def test_run_job_records_an_error_part_way_through_a_reply():
    """A stream that fails after some text records the error and, like the app, keeps no text."""

    def pieces(end):
        """Yield part of a reply, then fail like a read that timed out."""
        yield "The first half"
        raise llm.LLMTimeoutError("The AI service took too long to answer.")

    def breaking(*args, **kwargs):
        """Return a stream that breaks after its first piece."""
        end = llm._StreamEnd()
        return llm.ReplyStream(pieces(end), end)

    result = run_job(Job(ITEM, "zero_shot", "high", 1), DEFAULT_MODEL, stream=breaking)
    assert result.error == "The AI service took too long to answer."
    assert not result.reply


def test_run_job_reraises_a_missing_or_rejected_key():
    """An auth error stops the run, since every other request would fail the same way."""

    def rejected(*args, **kwargs):
        """Fail like llm.stream does for a rejected key."""
        raise llm.LLMAuthError("The OpenRouter API key was rejected.")

    with pytest.raises(llm.LLMAuthError):
        run_job(Job(ITEM, "zero_shot", "medium", 1), DEFAULT_MODEL, 100, stream=rejected)


def test_run_job_flags_a_reply_the_output_check_would_refuse(fake_llm):
    """A reply that repeats the system prompt is flagged, and the original text is kept."""
    fake_llm.reply = STRATEGIES["persona"](ITEM.role, ITEM.interview_type)
    result = run_job(Job(ITEM, "persona", "medium", 1), DEFAULT_MODEL, 100)
    assert result.refused
    assert result.reply == fake_llm.reply


def _results(strategies=tuple(STRATEGIES)):
    """Return one fake result per strategy for ITEM, each reply naming nothing."""
    return [
        Result(Job(ITEM, s, "medium", 1), f"Reply number {n}.", 1.5)
        for n, s in enumerate(strategies)
    ]


def test_render_report_names_strategies_and_adds_a_score_table():
    """A named report shows each strategy's label, reply, size and an empty score row."""
    results = _results()
    report, key = render_report(results, _settings(results=results))
    assert key is None
    assert f"## Input `{ITEM.id}` · Behavioural · Junior · Data Analyst" in report
    assert "### Reasoning effort medium · run 1" in report
    assert f"> {ITEM.text}" in report
    for strategy, result in zip(STRATEGIES, results, strict=True):
        assert f"#### {STRATEGY_LABELS[strategy]} (`{strategy}`)" in report
        assert f"> {result.reply}" in report
        assert f"| {STRATEGY_LABELS[strategy]} (`{strategy}`) |" in report
    assert "| Strategy | " + " | ".join(CRITERIA) + " | Notes |" in report
    assert f"_{len(results[0].reply)} characters · 1.5 s · no token count reported_" in report


def test_render_report_header_names_each_efforts_budget():
    """The header lists the efforts and the max_tokens each one was sent with."""
    report, _ = render_report([], _settings(efforts=("minimal", "high"), planned=0))
    assert "- Reasoning effort: minimal, high" in report
    assert "- max_tokens: minimal 4,000, high 16,000 (the app's budgets)" in report
    settings = RunSettings(**{**vars(_settings(planned=0)), "max_tokens": 2000})
    report, _ = render_report([], settings)
    assert "- max_tokens: 2,000 for every effort (set with --max-tokens)" in report


def test_render_report_shows_token_counts_and_a_cut_off_reply():
    """A reply's line shows the app's token counts; a cut-off reply is marked with ✂️."""
    results = [
        Result(Job(ITEM, "zero_shot", "high", 1), "Long reply", 9.0, usage=FakeLLM.usage),
        Result(
            Job(ITEM, "persona", "high", 1), "Cut reply", 9.0, usage=FakeLLM.usage, cut_off=True
        ),
    ]
    report, _ = render_report(results, _settings(["zero_shot", "persona"], results=results))
    assert f"_10 characters · 9.0 s · {FAKE_USAGE_TEXT}_" in report
    assert report.count("**✂️ Cut off:**") == 1
    assert report.index("**✂️ Cut off:**") > report.index("#### Role / persona")
    assert "0 refused by the output check, 1 cut off, 0 errors" in report


def test_usage_text_leaves_out_unreported_reasoning():
    """Without a reasoning count the completion shows alone, as in the app."""
    usage = llm.Usage(prompt_tokens=10, completion_tokens=2000, total_tokens=2010)
    assert prompt_eval._usage_text(usage) == "Prompt 10 · Completion 2,000 · Total 2,010 tokens"


def test_render_report_shows_the_question_before_the_answer():
    """An input with a question shows it, quoted, above the candidate's message."""
    item = EvalInput(**SAMPLE[0], question="Why did the report break?")
    results = [Result(Job(item, "zero_shot", "medium", 1), "Reply.", 1.0)]
    report, _ = render_report(results, _settings(["zero_shot"], results=results))
    asked = report.index("**Coach asked:**\n\n> Why did the report break?")
    assert asked < report.index(f"**Candidate:**\n\n> {item.text}")


def test_render_report_quotes_every_line_of_a_reply():
    """Every line of a multi-line reply stays inside the blockquote, headings included."""
    results = [Result(Job(ITEM, "zero_shot", "medium", 1), "## Feedback\n\nGood start.", 1.0)]
    report, _ = render_report(results, _settings(["zero_shot"], results=results))
    assert "> ## Feedback\n>\n> Good start." in report
    assert "\n## Feedback" not in report


def test_render_report_shows_blocks_errors_refusals_and_an_early_stop():
    """Blocked, failed and refused results each get their own line; a short run says so."""
    results = [
        Result(Job(ITEM, "zero_shot", "medium", 1), blocked="Your message is too long."),
        Result(Job(ITEM, "few_shot", "medium", 1), seconds=2.0, error="Timed out."),
        Result(Job(ITEM, "persona", "medium", 1), "Leaked prompt.", 1.0, refused=True),
    ]
    report, _ = render_report(results, _settings(planned=5, results=results))
    assert "**Blocked by the input guard:** Your message is too long." in report
    assert "**Error** after 2.0 s: Timed out." in report
    assert "**The output check would refuse this reply**" in report
    assert "Requests: 3 of 5 done; 1 blocked by the input guard, 1 refused" in report
    assert "**Stopped early:**" in report


def test_blind_report_hides_strategies_and_the_key_names_each_letter():
    """A blind report shows only letters; the key maps the letters to every strategy once."""
    results = _results()
    report, key = render_report(results, _settings(results=results), blind=True)
    for strategy in STRATEGIES:
        assert strategy not in report
        assert STRATEGY_LABELS[strategy] not in report
    letters = re.findall(r"#### Strategy ([A-E])", report)
    assert letters == list("ABCDE")
    mapping = dict(re.findall(r"([A-E]): `(\w+)`", key))
    assert sorted(mapping.values()) == sorted(STRATEGIES)
    # Each key row has the header's four columns, so the table renders.
    rows = [line for line in key.splitlines() if line.startswith("| `")]
    assert rows
    assert all(row.count("|") == 5 for row in rows)
    # Letter A's reply in the report is the reply of the strategy the key gives for A.
    by_strategy = {r.job.strategy: r.reply for r in results}
    first = report.split("#### Strategy A", 1)[1].split("####", 1)[0]
    assert f"> {by_strategy[mapping['A']]}" in first


def test_blind_report_hides_times_and_the_key_has_them():
    """A blind report leaves each reply's time out (the terminal showed it); the key keeps it."""
    results = [
        Result(Job(ITEM, "zero_shot", "medium", 1), "Reply.", 4.2),
        Result(Job(ITEM, "persona", "medium", 1), seconds=9.9, error="Timed out."),
    ]
    settings = _settings(["zero_shot", "persona"], results=results)
    report, key = render_report(results, settings, blind=True)
    assert "4.2" not in report
    assert "9.9" not in report
    assert "_6 characters · no token count reported_" in report
    assert "**Error**: Timed out." in report
    assert "`zero_shot` (4.2 s)" in key
    assert "`persona` (9.9 s)" in key


def test_blind_shuffle_is_repeatable_with_a_seed():
    """The same seed gives the same letters, so a run's key can be rebuilt."""
    results = _results()
    first = render_report(results, _settings(results=results), blind=True, rng=random.Random(7))
    again = render_report(results, _settings(results=results), blind=True, rng=random.Random(7))
    assert first == again


def test_main_dry_run_calls_nothing_and_writes_nothing(fake_llm, tmp_path, capsys):
    """--dry-run prints the user prompts and the request count, with no API call or file."""
    out = tmp_path / "out"
    code = prompt_eval.main(
        ["--inputs", str(_write(tmp_path, SAMPLE)), "--out-dir", str(out), "--dry-run"]
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert fake_llm.calls == []
    assert not out.exists()
    assert "<user_input>" in printed
    assert f"would send {2 * len(STRATEGIES)} requests" in printed


def test_main_dry_run_shows_the_question(fake_llm, tmp_path, capsys):
    """--dry-run prints the coach's question that will be sent before the answer."""
    entries = [{**SAMPLE[0], "question": "Why did the report break?"}]
    prompt_eval.main(["--inputs", str(_write(tmp_path, entries)), "--dry-run"])
    assert "Coach asked (sent as the previous turn): Why did the report break?" in (
        capsys.readouterr().out
    )


def test_main_dry_run_shows_a_guard_block(fake_llm, tmp_path, capsys):
    """--dry-run tells you up front which inputs the guard would block."""
    entries = [{**SAMPLE[0], "text": ATTACKS[1]}]
    prompt_eval.main(["--inputs", str(_write(tmp_path, entries)), "--dry-run"])
    assert "BLOCKED by the input guard" in capsys.readouterr().out


def test_main_dry_run_allows_a_long_job_description(fake_llm, tmp_path, capsys):
    """--dry-run uses the job-description limit too, so it agrees with a real run."""
    entries = [{**SAMPLE[0], "interview_type": JD_ANALYSIS, "text": LONG_JD}]
    prompt_eval.main(["--inputs", str(_write(tmp_path, entries)), "--dry-run"])
    assert "BLOCKED" not in capsys.readouterr().out


def test_main_prints_non_english_text_through_a_cp1252_pipe(fake_llm, monkeypatch, tmp_path):
    """Vietnamese input text does not crash a print on a Windows-style cp1252 pipe."""
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", out)
    entries = [{**SAMPLE[0], "text": "Tôi đã sửa báo cáo hàng tuần bị lỗi."}]
    assert prompt_eval.main(["--inputs", str(_write(tmp_path, entries)), "--dry-run"]) == 0
    out.flush()
    assert "Tôi đã sửa báo cáo" in out.buffer.getvalue().decode("utf-8")


def test_main_writes_a_report(fake_llm, no_client, tmp_path):
    """A real run sends every job with its effort and that effort's budget, and writes a report."""
    out = tmp_path / "out"
    args = ["--inputs", str(_write(tmp_path, SAMPLE)), "--out-dir", str(out)]
    code = prompt_eval.main(
        [*args, "--strategies", "few_shot", "persona", "--efforts", "minimal", "high"]
    )
    assert code == 0
    assert len(fake_llm.calls) == 2 * 2 * 2
    assert {(c["reasoning_effort"], c["max_tokens"]) for c in fake_llm.calls} == {
        ("minimal", MAX_TOKENS_BY_EFFORT["minimal"]),
        ("high", MAX_TOKENS_BY_EFFORT["high"]),
    }
    assert {c["client"] for c in fake_llm.calls} == {"fake-client"}
    (report,) = out.glob("run-*.md")
    text = report.read_text(encoding="utf-8")
    assert "Requests: 8 of 8 done" in text
    assert "Few-shot (`few_shot`)" in text
    assert FAKE_USAGE_TEXT in text


def test_main_runs_only_the_chosen_ids(fake_llm, no_client, tmp_path):
    """--ids sends only those inputs, with --max-tokens used for every effort."""
    out = tmp_path / "out"
    args = ["--inputs", str(_write(tmp_path, SAMPLE)), "--out-dir", str(out), "--ids", "tech-1"]
    assert prompt_eval.main([*args, "--max-tokens", "3000"]) == 0
    assert len(fake_llm.calls) == len(STRATEGIES)
    assert all("hash map" in c["messages"][-1]["content"] for c in fake_llm.calls)
    assert {c["max_tokens"] for c in fake_llm.calls} == {3000}


def test_main_rejects_an_unknown_id(fake_llm, tmp_path, capsys):
    """A mistyped --ids value stops the run before any call."""
    assert prompt_eval.main(["--inputs", str(_write(tmp_path, SAMPLE)), "--ids", "beh-2"]) == 2
    assert "no input with id(s): beh-2" in capsys.readouterr().err
    assert fake_llm.calls == []


def test_main_blind_run_writes_a_separate_key(fake_llm, no_client, tmp_path):
    """--blind writes the key to its own file, with the seed so the shuffle can be rebuilt."""
    out = tmp_path / "out"
    args = ["--inputs", str(_write(tmp_path, SAMPLE)), "--out-dir", str(out), "--blind"]
    assert prompt_eval.main([*args, "--seed", "3"]) == 0
    key = next(out.glob("run-*-key.md")).read_text(encoding="utf-8")
    report = next(p for p in out.glob("run-*.md") if not p.stem.endswith("-key"))
    assert "few_shot" not in report.read_text(encoding="utf-8")
    assert "few_shot" in key
    assert "Shuffle seed: 3" in key


def test_main_keeps_replies_when_interrupted(no_client, monkeypatch, tmp_path):
    """Ctrl+C during a run still writes the replies already received."""
    fake = FakeLLM()
    fake.reply = "First reply."

    def interrupt_second(*args, **kwargs):
        """Answer the first request, then act as if the user pressed Ctrl+C."""
        if fake.calls:
            raise KeyboardInterrupt
        return fake.stream(*args, **kwargs)

    monkeypatch.setattr(llm, "stream", interrupt_second)
    out = tmp_path / "out"
    assert prompt_eval.main(["--inputs", str(_write(tmp_path, SAMPLE)), "--out-dir", str(out)]) == 0
    text = next(out.glob("run-*.md")).read_text(encoding="utf-8")
    assert f"Requests: 1 of {2 * len(STRATEGIES)} done" in text
    assert "> First reply." in text


def test_main_rejects_unfilled_inputs(fake_llm, tmp_path, capsys):
    """A file with a TODO field left stops the run before any call."""
    entries = [{**SAMPLE[0], "id": "TODO-short-id"}]
    assert prompt_eval.main(["--inputs", str(_write(tmp_path, entries))]) == 2
    assert "TODO placeholder" in capsys.readouterr().err
    assert fake_llm.calls == []


def test_main_blind_progress_does_not_name_strategies(fake_llm, no_client, tmp_path, capsys):
    """A blind run's progress lines leave the strategy out, so times cannot reveal letters."""
    out = tmp_path / "out"
    args = ["--inputs", str(_write(tmp_path, SAMPLE)), "--out-dir", str(out), "--blind"]
    assert prompt_eval.main(args) == 0
    progress = capsys.readouterr().err
    assert f"[1/{2 * len(STRATEGIES)}] beh-1 · effort=medium · run 1" in progress
    for strategy in STRATEGIES:
        assert strategy not in progress


def test_main_named_progress_names_strategies(fake_llm, no_client, tmp_path, capsys):
    """A named run's progress lines say which strategy each request uses."""
    args = ["--inputs", str(_write(tmp_path, SAMPLE)), "--out-dir", str(tmp_path / "out")]
    prompt_eval.main([*args, "--strategies", "persona"])
    assert "[1/2] beh-1 · persona · effort=medium · run 1" in capsys.readouterr().err


def test_main_writes_reports_with_lf_line_endings(fake_llm, no_client, tmp_path):
    """Report and key use LF endings on Windows too, like the rest of the repo."""
    out = tmp_path / "out"
    args = ["--inputs", str(_write(tmp_path, SAMPLE)), "--out-dir", str(out), "--blind"]
    prompt_eval.main(args)
    files = list(out.glob("run-*.md"))
    assert len(files) == 2
    assert all(b"\r\n" not in path.read_bytes() for path in files)


def test_main_keeps_replies_when_the_key_fails_mid_run(no_client, monkeypatch, tmp_path):
    """A key rejected after some replies still writes them, and the exit code says it failed."""
    fake = FakeLLM()
    fake.reply = "First reply."

    def reject_second(*args, **kwargs):
        """Answer the first request, then fail like a revoked key."""
        if fake.calls:
            raise llm.LLMAuthError("The OpenRouter API key was rejected.")
        return fake.stream(*args, **kwargs)

    monkeypatch.setattr(llm, "stream", reject_second)
    out = tmp_path / "out"
    assert prompt_eval.main(["--inputs", str(_write(tmp_path, SAMPLE)), "--out-dir", str(out)]) == 1
    text = next(out.glob("run-*.md")).read_text(encoding="utf-8")
    assert f"Requests: 1 of {2 * len(STRATEGIES)} done" in text
    assert "> First reply." in text


def test_main_stops_when_the_key_is_missing(fake_llm, monkeypatch, tmp_path, capsys):
    """A missing key stops the run before any request, with the reason."""

    def no_key():
        """Fail like make_client does without a key."""
        raise llm.LLMAuthError("No OpenRouter API key is set.")

    monkeypatch.setattr(llm, "make_client", no_key)
    assert prompt_eval.main(["--inputs", str(_write(tmp_path, SAMPLE))]) == 1
    assert "No OpenRouter API key is set." in capsys.readouterr().err
    assert fake_llm.calls == []


@pytest.mark.parametrize(
    "args",
    [
        ["--efforts", "xhigh"],
        ["--efforts", "none"],
        ["--runs", "0"],
        ["--model", "openai/gpt-4o"],
        ["--strategies", "self_critique"],
        ["--max-tokens", "0"],
        ["--max-tokens", str(MAX_TOKENS_CAP + 1)],
        ["--ids"],
    ],
)
def test_main_rejects_bad_arguments(args):
    """Unknown efforts, models or strategies, bad runs or max tokens, and an empty --ids fail."""
    with pytest.raises(SystemExit):
        prompt_eval.parse_args(args)


def test_parse_args_defaults_follow_the_app():
    """By default a run uses the app's default effort and model, each effort's own budget."""
    args = prompt_eval.parse_args([])
    assert args.efforts == [DEFAULT_REASONING_EFFORT]
    assert args.model == DEFAULT_MODEL
    assert args.max_tokens is None
    assert args.ids is None
    assert prompt_eval.parse_args(["--max-tokens", str(MAX_TOKENS_CAP)]).max_tokens == (
        MAX_TOKENS_CAP
    )


def test_parse_args_drops_repeated_values():
    """Repeated strategies, efforts or ids would only send the same requests twice."""
    args = prompt_eval.parse_args(
        ["--strategies", "persona", "persona", "--efforts", "low", "low", "--ids", "a", "a"]
    )
    assert args.strategies == ["persona"]
    assert args.efforts == ["low"]
    assert args.ids == ["a"]
