"""Run fixed test inputs through every prompt strategy and write a report to score by hand (T3.3).

Run from the repo root; the inputs are in scripts/eval_inputs.json (or pass --inputs):

    python -m scripts.prompt_eval --dry-run
    python -m scripts.prompt_eval
    python -m scripts.prompt_eval --strategies few_shot --efforts minimal high --ids jd-review

It calls the real OpenRouter API (the key comes from .streamlit/secrets.toml or the
OPENROUTER_API_KEY environment variable), so it is run by hand and never in CI.
"""

import argparse
import json
import logging
import random
import string
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from src import llm
from src.config import (
    ALLOWED_MODELS,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    MAX_TOKENS_BY_EFFORT,
    MAX_TOKENS_CAP,
    REASONING_EFFORTS,
)
from src.guard import GuardError, check_output, max_input_chars, validate_input, validate_role
from src.prompts import (
    INTERVIEW_TYPES,
    SENIORITY_LEVELS,
    STRATEGIES,
    STRATEGY_LABELS,
    build_messages,
    build_user_prompt,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUTS = ROOT / "scripts" / "eval_inputs.json"
DEFAULT_OUT_DIR = ROOT / "docs" / "eval"
INPUT_FIELDS = ("id", "role", "interview_type", "seniority", "text")
# The question the coach asked before the candidate's message; sent as the previous turn.
OPTIONAL_FIELDS = ("question",)
# The rubric from docs/PLAN.md (T3.3), each scored 1-5.
CRITERIA = ("Relevance", "Actionability", "Structure", "Tone")
# Marks a field not filled in yet (a new inputs file can start from it); a run refuses to start
# while a field begins with it.
PLACEHOLDER = "TODO"


class EvalInputError(ValueError):
    """Raised when the inputs file is missing, malformed or still holds placeholder values."""


@dataclass(frozen=True)
class EvalInput:
    """One fixed test input: the session settings, the coach's question and the reply to it."""

    id: str
    role: str
    interview_type: str
    seniority: str
    text: str
    question: str | None = None


@dataclass(frozen=True)
class Job:
    """One request to make: an input sent with one strategy at one reasoning effort."""

    item: EvalInput
    strategy: str
    effort: str
    run: int


@dataclass(frozen=True)
class Result:
    """What one job produced: the reply, or why there is none."""

    job: Job
    reply: str = ""
    seconds: float = 0.0
    # The guard's message when the input never reached the model.
    blocked: str | None = None
    # True when check_output() would have replaced the reply with its refusal.
    refused: bool = False
    error: str | None = None
    # Token counts OpenRouter reported, or None if it sent none.
    usage: llm.Usage | None = None
    # True when max_tokens ended the reply, which the app marks with ✂️.
    cut_off: bool = False


@dataclass(frozen=True)
class RunSettings:
    """The settings of one run, printed at the top of the report."""

    model: str
    strategies: tuple[str, ...]
    efforts: tuple[str, ...]
    runs: int
    # None: each effort gets the app's own budget (MAX_TOKENS_BY_EFFORT).
    max_tokens: int | None
    commit: str
    started: datetime
    planned: int


def load_inputs(path: Path) -> list[EvalInput]:
    """Read and check the inputs file; raise EvalInputError saying what to fix."""
    try:
        # utf-8-sig also reads a file saved "with BOM" (Notepad, PowerShell 5 Set-Content).
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise EvalInputError(f"{path} does not exist.") from None
    except json.JSONDecodeError as exc:
        raise EvalInputError(f"{path} is not valid JSON: {exc}") from None
    if not isinstance(raw, list) or not raw:
        raise EvalInputError(f"{path} must hold a non-empty JSON list of inputs.")
    items: list[EvalInput] = []
    seen: set[str] = set()
    for number, entry in enumerate(raw, start=1):
        where = f"{path.name}, entry {number}"
        if not isinstance(entry, dict):
            raise EvalInputError(f"{where}: must be a JSON object.")
        unknown = sorted(set(entry) - set(INPUT_FIELDS) - set(OPTIONAL_FIELDS))
        if unknown:
            raise EvalInputError(f"{where}: unknown field(s): {', '.join(unknown)}.")
        empty = [
            f for f in INPUT_FIELDS if not isinstance(entry.get(f), str) or not entry[f].strip()
        ]
        if empty:
            raise EvalInputError(f"{where}: missing or empty field(s): {', '.join(empty)}.")
        question = entry.get("question")
        if "question" in entry and (not isinstance(question, str) or not question.strip()):
            raise EvalInputError(
                f"{where}: question must be non-empty text; delete the field if there is none."
            )
        # Only at the start, so a real answer may mention "TODO comments".
        filled = [entry[f] for f in INPUT_FIELDS] + ([question] if question is not None else [])
        if any(value.startswith(PLACEHOLDER) for value in filled):
            raise EvalInputError(
                f"{where}: still holds the {PLACEHOLDER} placeholder; replace it with your input."
            )
        if entry["interview_type"] not in INTERVIEW_TYPES:
            raise EvalInputError(
                f"{where}: interview_type must be one of: {', '.join(INTERVIEW_TYPES)}."
            )
        if entry["seniority"] not in SENIORITY_LEVELS:
            raise EvalInputError(
                f"{where}: seniority must be one of: {', '.join(SENIORITY_LEVELS)}."
            )
        if entry["id"] in seen:
            raise EvalInputError(f"{where}: id {entry['id']!r} is used twice.")
        seen.add(entry["id"])
        # Fields spelled out (no **kwargs), as T4.4's source scan requires.
        items.append(
            EvalInput(
                id=entry["id"],
                role=entry["role"],
                interview_type=entry["interview_type"],
                seniority=entry["seniority"],
                text=entry["text"],
                question=question,
            )
        )
    return items


def select_inputs(inputs: Sequence[EvalInput], ids: Sequence[str] | None) -> list[EvalInput]:
    """Return the inputs whose id is in `ids` (all if None), in file order; raise on unknown ids."""
    if ids is None:
        return list(inputs)
    unknown = [i for i in ids if i not in {item.id for item in inputs}]
    if unknown:
        raise EvalInputError(f"no input with id(s): {', '.join(unknown)}.")
    return [item for item in inputs if item.id in ids]


def build_jobs(
    inputs: Sequence[EvalInput],
    strategies: Sequence[str],
    efforts: Sequence[str],
    runs: int,
) -> list[Job]:
    """Return one job per input, effort, run and strategy, grouped in that order."""
    return [
        Job(item, strategy, effort, run)
        for item in inputs
        for effort in efforts
        for run in range(1, runs + 1)
        for strategy in strategies
    ]


def budget(effort: str, max_tokens: int | None) -> int:
    """Return the max_tokens to send: `max_tokens` if set, else the app's budget for `effort`."""
    return MAX_TOKENS_BY_EFFORT[effort] if max_tokens is None else max_tokens


def _error_text(exc: llm.LLMError) -> str:
    """Return the error's message plus the HTTP status and reason OpenRouter gave, if any."""
    # LLMError's own message is the generic one the app shows; the cause says what went wrong,
    # e.g. a 400 when OpenRouter rejects a parameter.
    # Only the body's error message: the rest of the body (and the SDK's `message`, which
    # repeats it all) may hold the account's user_id, and reports are committed.
    cause = exc.__cause__
    status = getattr(cause, "status_code", None)
    if status is None:
        return str(exc)
    body = getattr(cause, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    reason = error.get("message") if isinstance(error, dict) else None
    if not isinstance(reason, str):
        return f"{exc} [HTTP {status}]"
    return f"{exc} [HTTP {status}: {reason}]"


def _history(item: EvalInput) -> list[dict]:
    """Return the earlier turns to send: the coach's question as an assistant turn, if any."""
    # The app's history always starts with a user turn; the message that opened the session
    # is not part of an input, so the question stands alone here.
    if item.question is None:
        return []
    return [{"role": "assistant", "content": item.question, "sent": item.question}]


def run_job(
    job: Job,
    model: str,
    max_tokens: int | None = None,
    *,
    client: OpenAI | None = None,
    stream: Callable[..., llm.ReplyStream] | None = None,
) -> Result:
    """Send one job the way the app does (guard, prompts, LLM, output check) and record it."""
    # Looked up at call time, so tests that patch llm.stream are picked up.
    stream = stream or llm.stream
    item = job.item
    try:
        # The same limit as the app: longer for a pasted job description (app.py).
        text = validate_input(item.text, max_input_chars(item.interview_type))
        role = validate_role(item.role)
    except GuardError as exc:
        return Result(job, blocked=str(exc))
    system_prompt = STRATEGIES[job.strategy](role, item.interview_type)
    user_prompt = build_user_prompt(role, item.interview_type, item.seniority, text)
    messages = build_messages(system_prompt, _history(item), user_prompt)
    start = time.perf_counter()
    try:
        # A stream, like the app: it reports token usage and whether max_tokens cut the reply,
        # and a long think at "high" keeps the connection reading instead of waiting for it all.
        replies = stream(messages, model, job.effort, budget(job.effort, max_tokens), client=client)
        # Read to the end (or to an error, which closes the response itself).
        reply = "".join(replies)
    except llm.LLMAuthError:
        # Every other job would fail the same way, so stop the run.
        raise
    except llm.LLMError as exc:
        # Like the app, a reply that fails part-way keeps none of its text.
        return Result(job, seconds=time.perf_counter() - start, error=_error_text(exc))
    seconds = time.perf_counter() - start
    refused = check_output(reply, system_prompt) != reply
    return Result(
        job,
        reply,
        seconds,
        refused=refused,
        usage=replies.usage,
        # As in the app: a reply the output check replaces is not shown as cut off.
        cut_off=replies.finish_reason == "length" and not refused,
    )


def _quote(text: str) -> str:
    """Return `text` as a Markdown blockquote, so a reply's headings stay inside its block."""
    return "\n".join(f"> {line}" if line.strip() else ">" for line in text.splitlines())


def _usage_text(usage: llm.Usage) -> str:
    """Return the token counts in the app's format: "Prompt ... · Completion ... · Total ..."."""
    # The same wording as app.usage_text, so a report reads like the app's "Token usage" line.
    completion = f"{usage.completion_tokens:,}"
    if usage.reasoning_tokens is not None:
        completion += f" (reasoning {usage.reasoning_tokens:,})"
    return (
        f"Prompt {usage.prompt_tokens:,} · Completion {completion} · "
        f"Total {usage.total_tokens:,} tokens"
    )


def _budget_text(settings: RunSettings) -> str:
    """Return the max_tokens each effort in the run was sent with, for the report header."""
    if settings.max_tokens is not None:
        return f"{settings.max_tokens:,} for every effort (set with --max-tokens)"
    return ", ".join(f"{e} {budget(e, None):,}" for e in settings.efforts) + " (the app's budgets)"


def _status(result: Result, *, blind: bool = False) -> str:
    """Return the one-line summary shown above a reply (without its time in a blind run)."""
    # A blind report leaves the time out: the terminal showed each strategy's time, so it
    # would tell the letters apart. The key file has the times instead.
    if result.blocked:
        return f"**Blocked by the input guard:** {result.blocked}"
    if result.error:
        took = "" if blind else f" after {result.seconds:.1f} s"
        return f"**Error**{took}: {result.error}"
    took = "" if blind else f" · {result.seconds:.1f} s"
    tokens = "no token count reported" if result.usage is None else _usage_text(result.usage)
    line = f"_{len(result.reply):,} characters{took} · {tokens}_"
    if result.cut_off:
        line += (
            "\n\n**✂️ Cut off:** the reply used up its max_tokens; the app would mark it as cut off."
        )
    if result.refused:
        line += (
            "\n\n**The output check would refuse this reply** (it repeats the system prompt); "
            "the app would show its refusal instead of the text below."
        )
    return line


def _named(strategy: str) -> str:
    """Return a strategy's label with its key, e.g. "Few-shot (`few_shot`)"."""
    return f"{STRATEGY_LABELS[strategy]} (`{strategy}`)"


def render_report(
    results: Sequence[Result],
    settings: RunSettings,
    *,
    blind: bool = False,
    rng: random.Random | None = None,
) -> tuple[str, str | None]:
    """Return the report as Markdown and, for a blind run, the key that names each letter."""
    rng = rng or random.Random()
    groups: dict[tuple[str, str, int], list[Result]] = {}
    for result in results:
        job = result.job
        groups.setdefault((job.item.id, job.effort, job.run), []).append(result)

    blocked = sum(1 for r in results if r.blocked)
    errors = sum(1 for r in results if r.error)
    refused = sum(1 for r in results if r.refused)
    cut_off = sum(1 for r in results if r.cut_off)
    shown = "hidden (blind run; see the key file)" if blind else ", ".join(settings.strategies)
    lines = [
        f"# Prompt evaluation run · {settings.started:%Y-%m-%d %H:%M}",
        "",
        f"- Model: `{settings.model}`",
        f"- Strategies: {shown}",
        f"- Reasoning effort: {', '.join(settings.efforts)}",
        f"- Runs per combination: {settings.runs}",
        f"- max_tokens: {_budget_text(settings)}",
        f"- Commit: `{settings.commit}`",
        f"- Requests: {len(results)} of {settings.planned} done; {blocked} blocked by the input "
        f"guard, {refused} refused by the output check, {cut_off} cut off, {errors} errors",
    ]
    if len(results) < settings.planned:
        lines.append("- **Stopped early:** the run was interrupted before every request was sent.")
    key_rows: list[str] = []
    current_input = None
    for (input_id, effort, run), group in groups.items():
        item = group[0].job.item
        if input_id != current_input:
            current_input = input_id
            lines += [
                "",
                f"## Input `{item.id}` · {item.interview_type} · {item.seniority} · {item.role}",
            ]
            if item.question is not None:
                lines += ["", "**Coach asked:**", "", _quote(item.question)]
            lines += ["", "**Candidate:**", "", _quote(item.text)]
        if blind:
            group = rng.sample(group, len(group))
            names = list(string.ascii_uppercase[: len(group)])
            key_rows.append(
                f"| `{input_id}` | {effort} | {run} | "
                + ", ".join(
                    f"{n}: `{r.job.strategy}` ({r.seconds:.1f} s)"
                    for n, r in zip(names, group, strict=True)
                )
                + " |"
            )
        else:
            names = [_named(r.job.strategy) for r in group]
        lines += ["", f"### Reasoning effort {effort} · run {run}"]
        for name, result in zip(names, group, strict=True):
            heading = f"Strategy {name}" if blind else name
            lines += ["", f"#### {heading}", "", _status(result, blind=blind)]
            if result.reply:
                lines += ["", _quote(result.reply)]
        lines += [
            "",
            "#### Scores (1-5)",
            "",
            "| Strategy | " + " | ".join(CRITERIA) + " | Notes |",
            "|---" * (len(CRITERIA) + 2) + "|",
            *(f"| {name} |" + " |" * (len(CRITERIA) + 1) for name in names),
        ]
    report = "\n".join(lines) + "\n"
    if not blind:
        return report, None
    key = "\n".join(
        [
            f"# Key for the blind run of {settings.started:%Y-%m-%d %H:%M}",
            "",
            "Open this only after scoring. Letters are shuffled again for every group.",
            "",
            "| Input | Effort | Run | Letters |",
            "|---|---|---|---|",
            *key_rows,
        ]
    )
    return report, key + "\n"


def _commit() -> str:
    """Return the short commit hash, marked if the tree has uncommitted changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            # Only the code that builds the prompts: the inputs file and earlier reports are
            # usually being edited during a run.
            ["git", "status", "--porcelain", "--", "app.py", "src", "scripts/prompt_eval.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return "unknown"
    return f"{head} (with uncommitted code changes)" if dirty else head


def _positive(value: str) -> int:
    """Parse a --runs value and check it is at least 1."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return number


def _max_tokens(value: str) -> int:
    """Parse a --max-tokens value and check it is between 1 and the app's cap."""
    number = _positive(value)
    if number > MAX_TOKENS_CAP:
        raise argparse.ArgumentTypeError(f"must be at most {MAX_TOKENS_CAP}, got {value}")
    return number


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.prompt_eval",
        description="Send fixed test inputs through the prompt strategies and write a report.",
    )
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS, help="inputs JSON file")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(STRATEGIES),
        default=list(STRATEGIES),
        help="strategy keys to run (default: all)",
    )
    parser.add_argument(
        "--efforts",
        nargs="+",
        choices=REASONING_EFFORTS,
        default=[DEFAULT_REASONING_EFFORT],
        help=f"reasoning effort levels to try (default: {DEFAULT_REASONING_EFFORT})",
    )
    parser.add_argument("--ids", nargs="+", help="only the inputs with these ids (default: all)")
    parser.add_argument("--runs", type=_positive, default=1, help="runs per combination")
    parser.add_argument("--model", choices=ALLOWED_MODELS, default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-tokens",
        type=_max_tokens,
        help=(
            "one reply budget for every effort, at most the app's cap of "
            f"{MAX_TOKENS_CAP} (default: the app's budget for each effort)"
        ),
    )
    parser.add_argument("--blind", action="store_true", help="hide strategy names behind letters")
    parser.add_argument("--seed", type=int, help="seed for the blind shuffle (default: random)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would be sent, without calling the API"
    )
    args = parser.parse_args(argv)
    # Repeats would only send the same requests twice.
    args.strategies = list(dict.fromkeys(args.strategies))
    args.efforts = list(dict.fromkeys(args.efforts))
    if args.ids is not None:
        args.ids = list(dict.fromkeys(args.ids))
    return args


def _dry_run(inputs: Sequence[EvalInput], jobs: Sequence[Job]) -> None:
    """Print each input's guard result and user prompt, and how many requests a run would make."""
    for item in inputs:
        print(f"== {item.id} · {item.interview_type} · {item.seniority} · {item.role}")
        if item.question is not None:
            print(f"Coach asked (sent as the previous turn): {item.question}\n")
        try:
            text = validate_input(item.text, max_input_chars(item.interview_type))
            role = validate_role(item.role)
        except GuardError as exc:
            print(f"BLOCKED by the input guard: {exc}\n")
            continue
        print(build_user_prompt(role, item.interview_type, item.seniority, text) + "\n")
    plural = "" if len(jobs) == 1 else "s"
    print(f"A real run would send {len(jobs)} request{plural}. No API calls were made.")


def _utf8_output() -> None:
    """Make stdout and stderr write UTF-8, so non-English input text cannot crash a print."""
    # On Windows a redirected or piped stream defaults to cp1252, which cannot encode e.g.
    # Vietnamese letters, so print() would raise UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation from the command line and return the exit code."""
    _utf8_output()
    # A guard block is written in the report; its log line would only break the progress line.
    logging.getLogger("src.guard").setLevel(logging.ERROR)
    args = parse_args(argv)
    try:
        inputs = select_inputs(load_inputs(args.inputs), args.ids)
    except EvalInputError as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 2
    jobs = build_jobs(inputs, args.strategies, args.efforts, args.runs)
    if args.dry_run:
        _dry_run(inputs, jobs)
        return 0

    try:
        # One client for the whole run; this also fails fast when the key is missing.
        client = llm.make_client()
    except llm.LLMAuthError as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 1
    settings = RunSettings(
        model=args.model,
        strategies=tuple(args.strategies),
        efforts=tuple(args.efforts),
        runs=args.runs,
        max_tokens=args.max_tokens,
        commit=_commit(),
        started=datetime.now(),
        planned=len(jobs),
    )
    results: list[Result] = []
    code = 0
    try:
        for number, job in enumerate(jobs, start=1):
            # A blind run leaves the strategy out, or the times would give the letters away.
            strategy = "" if args.blind else f"{job.strategy} · "
            print(
                f"[{number}/{len(jobs)}] {job.item.id} · {strategy}"
                f"effort={job.effort} · run {job.run} ...",
                end=" ",
                file=sys.stderr,
                flush=True,
            )
            result = run_job(job, args.model, args.max_tokens, client=client)
            results.append(result)
            outcome = "blocked" if result.blocked else "error" if result.error else "ok"
            print(f"{outcome} ({result.seconds:.1f} s)", file=sys.stderr)
    except llm.LLMAuthError as exc:
        # E.g. the key was revoked mid-run; still keep the replies already paid for.
        print(f"\nStopped: {exc}", file=sys.stderr)
        code = 1
    except KeyboardInterrupt:
        # Keep what was already paid for.
        print("\nInterrupted; writing the replies received so far.", file=sys.stderr)

    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    report, key = render_report(results, settings, blind=args.blind, rng=random.Random(seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"run-{settings.started:%Y%m%d-%H%M%S}"
    report_path = args.out_dir / f"{stem}.md"
    # LF endings on Windows too, so committed reports match the rest of the repo.
    report_path.write_text(report, encoding="utf-8", newline="\n")
    print(f"Report: {report_path}", file=sys.stderr)
    if key is not None:
        key_path = args.out_dir / f"{stem}-key.md"
        key_text = key + f"\nShuffle seed: {seed}\n"
        key_path.write_text(key_text, encoding="utf-8", newline="\n")
        print(f"Key (open after scoring): {key_path}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
