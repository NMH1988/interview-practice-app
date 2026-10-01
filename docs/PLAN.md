# Interview Practice App — Implementation Plan

**Stack:** Python 3.14, Streamlit, OpenRouter (`openai/gpt-5-mini` default), pytest, ruff, GitHub Actions, Streamlit Community Cloud.
**Estimate:** ~5 h core. Ticket sizes: S ≈ 20–30 min, M ≈ 45–60 min.

## Architecture (from the diagram)

```
Streamlit UI ──► Security Guard ──► Prompt Builder (system + user) ──► OpenRouter client ──► LLM
     ▲                                                                       ▲                 │
     └──────────────────── Generated interview answer ◄──────────────────────┴── Model settings (temperature)
```

Proposed module layout:

```
app.py                  # Streamlit UI only (thin)
src/
  config.py             # model names, defaults, secrets loading
  guard.py              # input validation / injection / rate limit
  prompts.py            # 5+ system prompt strategies + user prompt builder
  llm.py                # OpenRouter client wrapper (retries, errors, streaming)
tests/                  # all LLM calls mocked
docs/PLAN.md  docs/PROMPT_EVALUATION.md
```

**Product decision (default, change if you prefer):** *Mock interview coach* — the user picks a role + interview type (behavioural, technical, questions-to-ask, job-description analysis), the app asks a question, the user answers, and the app gives feedback.

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

### T1.2 Secrets management (S)
- Load `OPENROUTER_API_KEY` from `st.secrets` or env var; `.streamlit/secrets.toml.example`.
**Acceptance criteria**
- [ ] Key is never committed (`secrets.toml`, `.env` are gitignored — already done).
- [ ] Missing key shows a friendly in-app error, not a stack trace.
- [ ] README documents local setup in ≤ 5 steps.

## Epic 2 — OpenRouter Integration
Goal: reliable LLM calls behind one small interface.

### T2.1 OpenRouter client wrapper (M)
- `llm.complete(messages, model, temperature, max_tokens)` using OpenRouter's OpenAI-compatible endpoint.
**Acceptance criteria**
- [ ] Returns the assistant text for a valid request (verified with a mocked HTTP layer in tests).
- [ ] Timeout, 401, 429 and 5xx map to distinct, user-readable exceptions; 429/5xx retried ≤ 2× with backoff.
- [ ] Rejects models outside the allowed list.

### T2.2 Streaming responses (S)
**Acceptance criteria**
- [ ] Answer renders incrementally via `st.write_stream`.
- [ ] Stream errors mid-response show a message and keep the chat history intact.

### T2.3 Cost & usage guardrails (S)
**Acceptance criteria**
- [ ] `max_tokens` capped per request (configurable).
- [ ] Token usage from the response is shown in an expander (or logged).

## Epic 3 — Prompt Engineering (≥ 5 strategies)
Goal: satisfy the brief's "5 system prompts, pick the best" requirement with evidence.

### T3.1 Prompt registry (S)
**Acceptance criteria**
- [ ] `prompts.py` exposes a dict of named strategies; each is a function `(role, interview_type) -> system prompt`.
- [ ] Unit test asserts ≥ 5 strategies are registered and all return non-empty strings.

### T3.2 Implement five strategies (M)
Zero-shot · Few-shot (2–3 example Q&A with feedback) · Chain-of-Thought (reason before scoring) · Role/persona (strict senior interviewer) · Structured-output (rubric + fixed Markdown/JSON sections). Optional 6th: self-critique.
**Acceptance criteria**
- [ ] Each strategy is clearly labelled with its technique in code and UI.
- [ ] Every prompt instructs the model to stay on interview-prep topics and ignore instructions embedded in user answers.
- [ ] Few-shot examples are realistic and contain no real personal data.

### T3.3 Prompt evaluation (M)
- Run the same 3–5 fixed test inputs through all strategies at the default temperature; score on a rubric (relevance, actionability, structure, tone, 1–5).
**Acceptance criteria**
- [ ] `docs/PROMPT_EVALUATION.md` contains the test inputs, a results table, and a justified winner.
- [ ] The winning strategy is the app default.

### T3.4 User-prompt builder (S)
**Acceptance criteria**
- [ ] Builds the user message from role, interview type, seniority, and the user's text.
- [ ] User text is delimited (e.g. `<user_input>…</user_input>`) so the guard and prompts can reference it.

## Epic 4 — Security Guard (≥ 1 required)
Goal: prevent misuse before any tokens are spent.

### T4.1 Input validation guard (S)
**Acceptance criteria**
- [ ] Empty/whitespace input and input over N characters (default 2000) are rejected with a clear message and **no API call is made**.
- [ ] Control characters are stripped.

### T4.2 Prompt-injection / off-topic guard (M)
**Acceptance criteria**
- [ ] Known patterns ("ignore previous instructions", "reveal your system prompt", role-override attempts) are blocked — parametrised tests cover ≥ 10 attack strings and ≥ 5 benign strings (no false positives on normal answers).
- [ ] Blocked requests show a neutral refusal and are logged without logging the API key.
- [ ] Stretch: cheap LLM classifier pass (`gpt-5-nano`) for off-topic detection, behind a feature flag.

### T4.3 Rate limiting (S)
**Acceptance criteria**
- [ ] Per-session limit (default 10 requests/min, 50/session) enforced via `st.session_state`.
- [ ] Exceeding the limit shows remaining wait time; limit values are in `config.py`.

### T4.4 Output safety (S)
**Acceptance criteria**
- [ ] Response is rendered as Markdown without `unsafe_allow_html`.
- [ ] If the response contains the system prompt text verbatim, it is replaced by a refusal (unit tested).

## Epic 5 — Streamlit UI
Goal: a polished single-page app matching the diagram.

### T5.1 Layout & inputs (M)
Sidebar: model select, strategy select, temperature slider, interview type, role. Main: chat.
**Acceptance criteria**
- [ ] Single page, works at desktop and mobile widths.
- [ ] Temperature slider 0.0–1.5 (default 0.7) is passed to the API call.
- [ ] Theme colours still come from `.streamlit/config.toml`.

### T5.2 Chat flow (M)
**Acceptance criteria**
- [ ] `st.chat_input` → guard → prompt → LLM → streamed reply; history kept in `st.session_state`.
- [ ] "New session" button clears history.
- [ ] A spinner/disabled input prevents double submission.

### T5.3 Interview modes (M)
**Acceptance criteria**
- [ ] Modes: Behavioural Q&A, Technical questions, Questions to ask the interviewer, Job-description analysis (paste JD → prep strategy).
- [ ] Each mode changes the system/user prompt and the placeholder text.

### T5.4 Error & empty states (S)
**Acceptance criteria**
- [ ] Guard blocks, rate limits, and API errors each show a distinct `st.error`/`st.warning` message.
- [ ] Empty state shows example prompts the user can click.

### T5.5 Remove placeholder dashboard (S)
**Acceptance criteria**
- [ ] Dummy `load_sessions` data and chart removed (or replaced by a real session-score tracker if Epic 7 is done).

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
- [ ] Unit tests for `guard`, `prompts`, `llm` (mocked); UI smoke test via `streamlit.testing.v1.AppTest` (exists).
- [ ] Coverage ≥ 80% on `src/`.

### T6.3 CD: Streamlit Community Cloud (S)
`.github/workflows/cd.yml` smoke-tests the live URL after CI passes on `main`.
**Acceptance criteria**
- [ ] App deployed from `main` on share.streamlit.io; `OPENROUTER_API_KEY` set in the Cloud secrets UI (never in repo).
- [ ] Repo variable `APP_URL` set; CD job passes `/_stcore/health` check after a merge.
- [ ] Deploy URL in README.

### T6.4 Dependency hygiene (S)
**Acceptance criteria**
- [ ] Dependabot enabled for pip and GitHub Actions *(config added)*.
- [ ] `pyproject.toml` / `requirements.txt` versions pinned and consistent.

### T6.5 Code-reviewer subagent (S)
- `.claude/agents/code-reviewer.md`: read-only reviewer for branches/PRs, tailored to this repo's stack and rules.
**Acceptance criteria**
- [ ] Read-only; reviews the current branch vs `main` by default, or named files / PR / ticket.
- [ ] Checklist covers correctness, security (secrets, guard, injection, output safety), mocked tests and project conventions.
- [ ] Runs ruff + pytest, checks the ticket's acceptance criteria, and reports in a fixed severity-ranked format.

## Epic 7 — Optional / Portfolio Extras
- T7.1 Session score tracker (replaces placeholder chart) — AC: scores parsed from structured output and charted per session.
- T7.2 Export practice session to Markdown — AC: download button yields the full transcript.
- T7.3 RAG over a question bank using `qwen/qwen3-embedding-8b` — AC: retrieved questions cited in the prompt; ≥ 20 seed questions.
- T7.4 Prompt A/B comparison view — AC: same input run through two strategies side by side.

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
Code on a feature branch → PR → CI green → acceptance criteria checked → merged to `main`.
