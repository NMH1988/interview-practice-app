# Interview Practice App — Implementation Plan

**Stack:** Python 3.14, Streamlit, OpenRouter (`openai/gpt-5-mini` default), pytest, ruff, GitHub Actions, Streamlit Community Cloud.
**Estimate:** ~5 h core. Ticket sizes: S ≈ 20–30 min, M ≈ 45–60 min.

## Architecture (from the diagram)

```
Streamlit UI ──► Security Guard ──► Prompt Builder (system + user) ──► OpenRouter client ──► LLM
     ▲                                                                       ▲                 │
     └──────────────────── Generated interview answer ◄──────────────────────┴── Model settings (reasoning effort)
```

Proposed module layout:

```
app.py                  # Streamlit UI only (thin)
src/
  config.py             # model names, defaults, secrets loading
  guard.py              # input validation / injection / output safety
  rate_limit.py         # per-session request limits
  prompts.py            # 5+ system prompt strategies + user prompt builder + example prompts
  llm.py                # OpenRouter client wrapper (retries, errors, streaming)
tests/                  # all LLM calls mocked
docs/BRIEF.md  docs/PLAN.md  docs/PROMPT_EVALUATION.md
```

**Product decision (default, change if you prefer):** *Mock interview coach* — the user picks a role + interview type (behavioural, technical, questions-to-ask, job-description analysis), the app asks a question, the user answers, and the app gives feedback.

## Testing strategy
Every ticket lists its own **Tests**, written in the same PR as the code (not saved up for T6.2).

| Layer | Tool | What it proves | When it runs |
|---|---|---|---|
| Unit | pytest | One function in `src/` works, including edge cases and errors | CI, every PR |
| UI flow | `streamlit.testing.v1.AppTest` + fake LLM | `app.py` really wires widgets → guard → prompt → LLM → chat | CI, every PR |
| Deploy smoke | CD job (`/_stcore/health`) + one manual chat turn | The live app is up and really talks to OpenRouter | After merge to `main` |

- **No test calls OpenRouter.** Unit tests mock the HTTP layer; UI flow tests use a `fake_llm` fixture in `tests/conftest.py`, added by the first ticket that needs it.
- UI flow tests patch module attributes (e.g. `src.llm.stream`) with `monkeypatch`. This works because `app.py` looks the name up again on every run (see `tests/test_app_smoke.py`). So call the LLM only from `app.py`, or via `llm.stream(...)`; never through a name another `src` module bound at import time (`from src.llm import stream`), which the patch would not reach.
- Time-based code (retry backoff, rate limits) takes a clock/sleep that tests replace, so tests never really wait.
- What `AppTest` cannot check (layout width, streaming animation, double-click races) is listed as a **Manual** check.
- **No browser E2E suite (Playwright/Selenium) for now.** UI flow tests already run the real `app.py` from input to reply with only the LLM faked, and the deploy smoke test covers the live app. A browser suite would be slow and flaky and would mostly re-test Streamlit itself. Revisit if we add custom components (`st.components`) or more pages.

---

## Epic 1 — Foundation & Project Setup
Goal: a clean, runnable repo skeleton that other epics build on.

### T1.1 Restructure project & config (S)
- Create `src/` package, `config.py` (model list, default `openai/gpt-5-mini`, default temperature).
- Remove placeholder `main.py`; fill README stub; fix `pyproject.toml` description/dependencies (streamlit, openai/httpx).
**Acceptance criteria**
- [ ] `streamlit run app.py` starts with no errors.
- [ ] Allowed models are exactly the three from the brief; default is `gpt-5-mini`.
- [ ] `pyproject.toml` and `requirements.txt` list the same runtime dependencies.

**Tests**
- [ ] Unit: allowed models are exactly the three from the brief; default is `openai/gpt-5-mini` (`tests/test_config.py`).
- [ ] Unit: `pyproject.toml` and `requirements.txt` list the same runtime dependencies (`tests/test_dependencies.py`).
- [ ] UI flow: the app loads with no exception (`tests/test_app_smoke.py`).

### T1.2 Secrets management (S)
- Load `OPENROUTER_API_KEY` from `st.secrets` or env var; `.streamlit/secrets.toml.example`.
**Acceptance criteria**
- [ ] Key is never committed (`secrets.toml`, `.env` are gitignored — already done).
- [ ] Missing key shows a friendly in-app error, not a stack trace.
- [ ] README documents local setup in ≤ 5 steps.

**Tests**
- [ ] Unit: the key is read from `st.secrets`, then the env var; a missing key and an unparseable `secrets.toml` each raise their own error (`tests/test_config.py`).
- [ ] Unit: `secrets.toml` and `.env` are gitignored; the example file holds only a placeholder (`tests/test_secrets_hygiene.py`).
- [ ] UI flow: a missing key and a broken secrets file each show one `st.error` and no exception (`tests/test_app_smoke.py`).

## Epic 2 — OpenRouter Integration
Goal: reliable LLM calls behind one small interface.

### T2.1 OpenRouter client wrapper (M)
- `llm.complete(messages, model, temperature, max_tokens)` using OpenRouter's OpenAI-compatible endpoint. (T2.4 replaced `temperature` with `reasoning_effort`.)
**Acceptance criteria**
- [x] Returns the assistant text for a valid request (verified with a mocked HTTP layer in tests).
- [x] Timeout, 401, 429 and 5xx map to distinct, user-readable exceptions; 429/5xx retried ≤ 2× with backoff.
- [x] Rejects models outside the allowed list.

**Tests**
- [x] Unit (mocked HTTP, no network): a valid request returns the assistant text; model, temperature and `max_tokens` are sent as given. (T2.4: reasoning effort instead of temperature.)
- [x] Unit: timeout, 401, 429 and 5xx each raise their own exception with a readable message.
- [x] Unit: 429/5xx are retried at most 2 times, then raise; 401 is not retried. The backoff sleep is patched so tests stay fast.
- [x] Unit: a model outside the allowed list raises before any HTTP call is made.

### T2.2 Streaming responses (S)
**Acceptance criteria**
- [x] Answer renders incrementally via `st.write_stream`.
- [x] Stream errors mid-response show a message and keep the chat history intact.

**Tests**
- [x] Unit: the stream generator yields the chunks of a mocked stream in order; an error mid-stream raises the mapped exception.
- [x] UI flow: with a fake streaming LLM, the full reply is the last assistant message (`AppTest` sees the final result, not the incremental render).
- [x] UI flow: a stream that fails part-way shows an error, and the earlier chat history is still there.

### T2.3 Cost & usage guardrails (S)
**Acceptance criteria**
- [x] `max_tokens` capped per request (configurable).
- [x] Token usage from the response is shown in an expander (or logged).

**Tests**
- [x] Unit: a requested `max_tokens` above the cap is clamped to the value in `config.py`.
- [x] Unit: token usage is read from the response; a response without usage does not crash.
- [x] UI flow: after a reply, the usage expander shows the token counts.

### T2.4 Replace temperature with reasoning effort (S)
OpenRouter lists no `temperature` support for the three allowed gpt-5 models, so the slider had no effect. Reasoning effort (named in the brief's Easy #8) takes its place.
**Acceptance criteria**
- [x] One live call by the owner confirms what OpenRouter does with `temperature` for gpt-5-mini; the result is in `docs/PROGRESS.md` (accepted but ignored).
- [x] A "Reasoning effort" select (`minimal` / `low` / `medium` / `high`, default `medium` from `config.py`) replaces the slider and is sent as `reasoning={"effort": ...}`.
- [x] `temperature` is no longer sent; `config.py` drops its temperature settings.
- [x] A help text explains the trade-off: thinking time and tokens versus answer depth.
- [x] T3.3 (#10) compares reasoning effort instead of temperature (PLAN T3.3 entry and a comment on #10; the script switches when its branch merges).

**Tests**
- [x] Unit: `stream()` / `complete()` send the chosen effort and no `temperature`; an effort outside the list raises before any request.
- [x] UI flow: the select's value reaches the fake LLM call; no Temperature widget remains.

### T2.5 Token budget large enough for high reasoning effort (S)
At `high`, the sample-JD starter spent 3,648 of the 4,000 tokens thinking and was cut off (T5.8's live check, #73). Each effort now gets its own budget.
**Acceptance criteria**
- [x] The owner chose a budget per effort (not one larger budget for all): `minimal` / `low` / `medium` keep 4,000, `high` gets 16,000; the reason is in `docs/PROGRESS.md`.
- [x] `config.py` holds the budgets (`MAX_TOKENS_BY_EFFORT`) and `MAX_TOKENS_CAP = 16000`; `_send` still clamps to the cap, and its comment says why.
- [x] The reasoning-effort help text still says the thinking counts against the token limit, and names both limits.
- [x] Live check by the owner (real key): the JD sample starter at `high` finishes without ✂️; the token counts go in `docs/PROGRESS.md`.

**Tests**
- [x] Unit: the budget sent for each effort equals the configured value and never exceeds the cap.
- [x] Existing T2.3 tests (cap clamp, cut-off warning) still pass.

## Epic 3 — Prompt Engineering (≥ 5 strategies)
Goal: satisfy the brief's "5 system prompts, pick the best" requirement with evidence.

### T3.1 Prompt registry (S)
**Acceptance criteria**
- [ ] `prompts.py` exposes a dict of named strategies; each is a function `(role, interview_type) -> system prompt`.
- [ ] Unit test asserts ≥ 5 strategies are registered and all return non-empty strings.

**Tests**
- [ ] Unit: ≥ 5 strategies are registered; each returns a non-empty string for every interview type (parametrised).

### T3.2 Implement five strategies (M)
Zero-shot · Few-shot (2–3 example Q&A with feedback) · Chain-of-Thought (reason before scoring) · Role/persona (strict senior interviewer) · Structured-output (rubric + fixed Markdown/JSON sections). Optional 6th: self-critique.
**Acceptance criteria**
- [x] Each strategy is clearly labelled with its technique in code and UI.
- [x] Every prompt instructs the model to stay on interview-prep topics and ignore instructions embedded in user answers.
- [x] Few-shot examples are realistic and contain no real personal data.

**Tests**
- [x] Unit (parametrised over all strategies): every prompt contains the stay-on-topic rule and the ignore-embedded-instructions rule.
- [x] Unit: every strategy has a technique label; the few-shot prompt contains its examples; no prompt contains an email address or phone number.
- [x] UI flow: the strategy select lists every registered strategy by its label.

### T3.3 Prompt evaluation (M)
- Run the same 3–5 fixed test inputs through all strategies at the default reasoning effort (temperature until T2.4); score on a rubric (relevance, actionability, structure, tone, 1–5).
**Acceptance criteria**
- [ ] `docs/PROMPT_EVALUATION.md` contains the test inputs, a results table, and a justified winner.
- [ ] The winning strategy is the app default.

**Tests**
- [ ] Unit: the default strategy in `config.py` exists in the registry.
- [ ] Manual: the evaluation itself uses the real API, so it is run by hand and recorded in `docs/PROMPT_EVALUATION.md`, not in CI.

### T3.4 User-prompt builder (S)
**Acceptance criteria**
- [x] Builds the user message from role, interview type, seniority, and the user's text.
- [x] User text is delimited (e.g. `<user_input>…</user_input>`) so the guard and prompts can reference it.

**Tests**
- [x] Unit: the message contains role, interview type, seniority and the user's text inside `<user_input>…</user_input>`.
- [x] Unit: user text that contains `</user_input>` cannot close the block early (it is escaped or removed).

## Epic 4 — Security Guard (≥ 1 required)
Goal: prevent misuse before any tokens are spent.

### T4.1 Input validation guard (S)
**Acceptance criteria**
- [x] Empty/whitespace input and input over N characters (default 2000) are rejected with a clear message and **no API call is made**.
- [x] Control characters are stripped.

**Tests**
- [x] Unit (parametrised): empty, whitespace-only and over-limit input are rejected; input of exactly the limit is accepted.
- [x] Unit: control characters are stripped; normal newlines are kept.
- [x] UI flow: submitting whitespace-only or too-long input shows the message, and the fake LLM is called 0 times.

### T4.2 Prompt-injection / off-topic guard (M)
**Acceptance criteria**
- [x] Known patterns ("ignore previous instructions", "reveal your system prompt", role-override attempts) are blocked — parametrised tests cover ≥ 10 attack strings and ≥ 5 benign strings (no false positives on normal answers).
- [x] Blocked requests show a neutral refusal and are logged without logging the API key.
- [ ] Stretch: cheap LLM classifier pass (`gpt-5-nano`) for off-topic detection, behind a feature flag.

**Tests**
- [x] Unit (parametrised): ≥ 10 attack strings are blocked, including upper-case and extra-space variants; ≥ 5 normal interview answers are allowed.
- [x] Unit (`caplog`): a blocked request is logged, and the log never contains the API key.
- [x] UI flow: an attack string shows the neutral refusal, and the fake LLM is called 0 times.
- [ ] Stretch: with the flag off, the classifier is never called; with it on, a mocked classifier result is respected.

### T4.3 Rate limiting (S)
**Acceptance criteria**
- [x] Per-session limit (default 10 requests/min, 50/session) enforced via `st.session_state`.
- [x] Exceeding the limit shows remaining wait time; limit values are in `config.py`.

**Tests**
- [x] Unit (fake clock, no real waiting): 10 requests in one minute pass and the 11th is blocked with the right wait time; the window resets after 60 s; the 50-per-session cap holds.
- [x] UI flow: over the limit, the warning shows the wait time, and the fake LLM is not called.

### T4.4 Output safety (S)
**Acceptance criteria**
- [x] Response is rendered as Markdown without `unsafe_allow_html`.
- [x] If the response contains the system prompt text verbatim, it is replaced by a refusal (unit tested).

**Tests**
- [x] Unit: a response that contains the system prompt verbatim is replaced by the refusal; a normal response is unchanged.
- [x] Unit (source scan): model and user text are rendered without `unsafe_allow_html`; the only allowed `unsafe_allow_html=True` is the static theme CSS block in `app.py`.
- [x] UI flow: HTML in the user's message and in the model's reply is shown as text (the Markdown element has `allow_html` off).

## Epic 5 — Streamlit UI
Goal: a polished single-page app matching the diagram.

### T5.1 Layout & inputs (M)
Sidebar: model select, strategy select, temperature slider (replaced by a reasoning effort select in T2.4), interview type, role. Main: chat.
**Acceptance criteria**
- [x] Single page, works at desktop and mobile widths.
- [x] Temperature slider 0.0–1.5 (default 0.7) is passed to the API call.
- [x] Theme colours still come from `.streamlit/config.toml`.

**Tests**
- [x] UI flow: the sidebar shows model (exactly the allowed models, default `gpt-5-mini`), strategy, temperature (0.0–1.5, default 0.7), interview type and role. (T2.4: a reasoning effort select replaced the temperature slider.)
- [x] UI flow: a changed model and temperature reach the fake LLM call. (T2.4: reasoning effort.)
- [x] Manual: check desktop and mobile widths in a browser (`AppTest` cannot measure layout).

### T5.2 Chat flow (M)
**Acceptance criteria**
- [x] `st.chat_input` → guard → prompt → LLM → streamed reply; history kept in `st.session_state`.
- [x] "New session" button clears history.
- [x] A spinner/disabled input prevents double submission.

**Tests**
- [x] UI flow (main happy path): chat input → guard → prompt → fake LLM → reply shown; after two turns, both are in the history in order.
- [x] UI flow: the fake LLM receives the selected strategy's system prompt and the delimited user message.
- [x] UI flow: "New session" clears the history.
- [x] UI flow: a reply that repeats the system prompt (or a long paragraph of it) shows `guard.REFUSAL_MESSAGE` instead; pass every reply through `guard.check_output(reply, system_prompt)` before it is shown or stored (wired in by T4.4, #54, since T5.2 merged first).
- [x] Manual: double submission is blocked while a reply is generating (`AppTest` runs one script run at a time, so it cannot test this race).

### T5.3 Interview modes (M)
**Acceptance criteria**
- [x] Modes: Behavioural Q&A, Technical questions, Questions to ask the interviewer, Job-description analysis (paste JD → prep strategy). The modes exist; T5.8 (#70) added the JD mode's study plan.
- [x] Each mode changes the system/user prompt and the placeholder text.

**Tests**
- [x] Unit (parametrised over modes): each mode gives a different system/user prompt; job-description mode includes the pasted JD.
- [x] UI flow: switching mode changes the chat input placeholder.

### T5.4 Error & empty states (S)
**Acceptance criteria**
- [x] Guard blocks, rate limits, and API errors each show a distinct `st.error`/`st.warning` message.
- [x] Empty state shows example prompts the user can click.

**Tests**
- [x] UI flow: a guard block, a rate limit, and each LLM error (the fake LLM raises it) show their own distinct `st.error`/`st.warning` text and no exception; the shown text never contains the fake API key or the raw response body (only `str(exc)`).
- [x] UI flow: with no history, example prompts are shown; clicking one sends it (the fake LLM receives that text).

### T5.5 Remove placeholder dashboard (S)
**Acceptance criteria**
- [ ] Dummy `load_sessions` data and chart removed (or replaced by a real session-score tracker if Epic 7 is done).

**Tests**
- [ ] UI flow: update `tests/test_app_smoke.py` so no metrics or chart remain (today it asserts 3 metrics).

### T5.6 "Questions to ask the interviewer" generates questions (M)
The brief's starter idea is a generator: company name and role in, 5–8 thoughtful questions to ask at the end of the interview out, tailored to that company (quoted in #62). The mode used to only rate the candidate's own question. The owner chose to do both: suggest questions, and still give feedback on the candidate's own.
**Acceptance criteria**
- [x] `MODE_INSTRUCTIONS["Questions to ask the interviewer"]` (written by the owner) suggests 5–8 questions for the role and seniority, tailored to a company the candidate names (typed in the chat, no new field), using only what the candidate says about it; each suggestion has one short reason naming a criterion; the candidate's own question is reviewed against four criteria.
- [x] The empty-chat starters and caption for this mode match the new behaviour.
- [x] Few-shot Example 3 suggests questions instead of rating one.
- [x] The chat placeholder for this mode matches the new behaviour.
- [x] The user-prompt preface covers a request for questions as well as an own question (the job-description part was done in T5.3).

**Tests**
- [x] Unit: every strategy's prompt for this mode contains the owner's suggestion sentences, after the criteria they refer to; Example 3 has 5–8 questions, each with one reason, and no review headings; the starters pass the guard unchanged and match the mode.
- [x] UI flow: clicking a starter in this mode sends its text to the fake LLM (covered for every starter by `tests/test_app_examples.py`).
- [x] Live check by the owner (real key, gpt-5-mini): results in `docs/PROGRESS.md`; one known limitation is followed up in T3.5 (#76).

### T5.7 Separate developer settings from practice settings (S)
The brief's Medium #9: keep the developer settings (model, system prompts) apart from the user experience, since the user may not know much about LLMs (quoted in #63).
**Acceptance criteria**
- [x] The practice settings (interview type, role, seniority, New session) come first, under a "Practice settings" header.
- [x] The developer settings (model, prompt strategy, reasoning effort) sit in a "Developer settings" expander, collapsed by default. Max tokens joins it when T7.5 (#64) lands.
- [x] Defaults and widget keys are unchanged.

**Tests**
- [x] UI flow: the developer widgets are inside the expander and the practice widgets are not; settings changed inside the expander reach the fake LLM call.
- [x] Manual (Claude, fake key, no LLM call): desktop and mobile widths in a browser; the expander stays open while a setting is changed and while the role warning appears or goes away.

### T5.8 Job-description analysis adds a short study plan (S)
The brief's job description analyser "extracts the key skills, likely interview topics, and a short study plan" (quoted in #70), and T5.3 promises "paste JD → prep strategy", but the JD mode's instructions never asked for one.
**Acceptance criteria**
- [x] The owner's one sentence (English) is added verbatim to the job-description block of `MODE_INSTRUCTIONS`, asking for a short study plan (at most five items, after the owner's live check).
- [x] It also applies to a sample job description written on request (T5.4).

**Tests**
- [x] Unit: every strategy's job-description prompt contains the owner's sentence verbatim; it follows the list it refers to.

## Epic 6 — Quality, CI/CD & Deployment
Goal: every PR is linted, tested, scanned; `main` auto-deploys.

### T6.1 CI pipeline ✅ *(configured in this change)*
`.github/workflows/ci.yml`: ruff lint + format check, pytest with coverage, `pip-audit`.
**Acceptance criteria**
- [ ] CI runs on every PR and push to `main`; all three jobs green on current `main`.
- [ ] CI needs **no** OpenRouter key (LLM calls mocked).
- [ ] Branch protection on `main` requires CI to pass *(manual GitHub setting)*.

### T6.2 Test suite (M)
**Acceptance criteria**
- [ ] Every ticket's **Tests** list is done; gaps (e.g. in `guard`, `prompts`, `llm`) are filled.
- [ ] Coverage ≥ 80% on `src/`, enforced in CI.

**Tests**
- [ ] Shared fakes (`fake_llm`, fake clock) live in `tests/conftest.py`; tests for any gaps left by earlier tickets are added.
- [ ] CI runs `pytest --cov=src --cov-fail-under=80`, so the build fails below 80% coverage.

### T6.3 CD: Streamlit Community Cloud (S)
`.github/workflows/cd.yml` smoke-tests the live URL after CI passes on `main`.
**Acceptance criteria**
- [ ] App deployed from `main` on share.streamlit.io; `OPENROUTER_API_KEY` set in the Cloud secrets UI (never in repo).
- [ ] Repo variable `APP_URL` set; CD job passes `/_stcore/health` check after a merge.
- [ ] Deploy URL in README.

**Tests**
- [ ] Deploy smoke test: the CD job checks `/_stcore/health` on the live URL after each merge to `main`.
- [ ] Manual (the only real end-to-end check): after the first deploy, and after changes to `llm.py`, do one real chat turn on the live URL.

### T6.4 Dependency hygiene (S)
**Acceptance criteria**
- [ ] Dependabot enabled for pip and GitHub Actions *(config added)*.
- [ ] `pyproject.toml` / `requirements.txt` versions pinned and consistent.

**Tests**
- [ ] Unit: extend `tests/test_dependencies.py` to check that every runtime dependency is pinned with `==`.

### T6.5 Code-reviewer subagent (S)
- `.claude/agents/code-reviewer.md`: read-only reviewer for branches/PRs, tailored to this repo's stack and rules.
**Acceptance criteria**
- [ ] Read-only; reviews the current branch vs `main` by default, or named files / PR / ticket.
- [ ] Checklist covers correctness, security (secrets, guard, injection, output safety), mocked tests and project conventions.
- [ ] Runs ruff + pytest, checks the ticket's acceptance criteria, and reports in a fixed severity-ranked format.

### T6.6 Every PR closes its ticket (S)
**Acceptance criteria**
- [ ] PR template starts with `Closes #`.
- [ ] `.github/workflows/pr-checks.yml` fails PRs without `Closes/Fixes/Resolves #<issue>` (Dependabot exempt) and re-runs on description edits.
- [ ] `CLAUDE.md` documents the rule, including retargeting stacked PRs to `main`.

**Tests**
- [ ] Manual: a PR whose description has no `Closes/Fixes/Resolves #N` fails the `linked-issue` job; adding the keyword and editing the description makes it pass.
- [ ] Manual: a Dependabot PR skips the check.

### T6.7 Progress log read before every ticket (S)
**Acceptance criteria**
- [ ] `docs/PROGRESS.md` has one entry per PR (what, why, decisions/gotchas, follow-ups), backfilled with all work so far.
- [ ] `CLAUDE.md` and the `qrspi` skill require reading it (plus open PRs) before a new ticket.
- [ ] Every PR adds its own entry (Dependabot exempt).

### T6.8 Test plan in every ticket (S)
**Acceptance criteria**
- [ ] `docs/PLAN.md` has a "Testing strategy" section: unit tests, UI flow tests (`AppTest` with a fake LLM), the deploy smoke test, and why there is no browser E2E suite.
- [ ] Every ticket that changes code has a **Tests** list in `docs/PLAN.md` and in its GitHub issue.
- [ ] T6.2 becomes "fill gaps + enforce coverage"; the Definition of Done includes the ticket's **Tests** list.

### T6.9 Code conventions in CLAUDE.md (S)
**Acceptance criteria**
- [ ] `CLAUDE.md` has a "Code conventions" section: one-line docstring on every function, method and class (tests and fixtures included), and a plain-language "what I did and why" walkthrough after each task.
- [ ] The `qrspi` skill's Implement phase repeats both rules.
- [ ] A test fails CI when a function or class in `app.py`, `src/` or `tests/` has no one-line docstring.

**Tests**
- [ ] Unit: `tests/test_conventions.py` passes on the current tree.
- [ ] Unit: `missing_docstrings` reports, by `file:line name`, an undocumented function, class, method, nested helper, async function, an empty docstring and a multi-line docstring in a temporary file, and nothing for documented ones.
- [ ] Unit: a UTF-8 BOM file is still checked, and a file that does not parse raises a SyntaxError naming it.
- [ ] Unit: the repo-wide check fails if `src/` or `tests/` is missing, so a renamed folder cannot go unchecked.

### T6.11 Read the project brief before every ticket (S)
**Acceptance criteria**
- [x] `docs/BRIEF.md` summarises the course brief in our own words (the repo is public): mandatory requirements, the "don't put it in a box" freedom, the five starter ideas, the optional tasks with the brief's numbering, the evaluation criteria and the bonus rule.
- [x] `CLAUDE.md` and the `qrspi` skill's Question phase say to read it before every ticket (and the full brief when the exact wording matters), name the brief item the ticket serves (or label it "beyond the brief (owner's request)"), and ask (not infer) when our docs or prompts contradict the brief or a choice changes what the product does. A contradiction with the brief or work beyond it is always flagged, even when the owner asks for it.

## Epic 7 — Optional / Portfolio Extras
- T7.1 Session score tracker (replaces placeholder chart) — AC: scores parsed from structured output and charted per session.
  - Tests: unit — scores are parsed from structured output, and malformed or missing scores are skipped without crashing; UI flow — after scored replies, the chart has one point per scored answer.
- T7.2 Export practice session to Markdown — AC: download button yields the full transcript.
  - Tests: unit — the transcript builder turns the history into Markdown with every message in order; UI flow — with history, the download button is shown.
- T7.3 RAG over a question bank using `qwen/qwen3-embedding-8b` — AC: retrieved questions cited in the prompt; ≥ 20 seed questions.
  - Tests: unit (mocked embeddings) — retrieval returns the top-k most similar questions, the seed file has ≥ 20 questions, and retrieved questions appear cited in the built prompt.
- T7.4 Prompt A/B comparison view — AC: same input run through two strategies side by side.
  - Tests: UI flow — one input is sent to the fake LLM twice with two different system prompts, and both replies are shown side by side.

---

## Suggested order (≈ 5 h)

| Step | Tickets | Time |
|---|---|---|
| 1 | T1.1, T1.2, T6.1 (done) | 0:30 |
| 2 | T2.1, T3.1, T3.4 | 0:50 |
| 3 | T4.1, T4.2, T4.3, T4.4 | 1:00 |
| 4 | T3.2, T5.1, T5.2 | 1:15 |
| 5 | T3.3 evaluation, T5.3, T5.4, T5.5 | 1:00 |
| 6 | T6.2, T6.3, T6.4, README | 0:45 |

## Definition of Done (all tickets)
Code on a feature branch → PR → CI green → acceptance criteria and **Tests** list checked → merged to `main`.
