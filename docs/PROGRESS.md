# Progress log

What each PR did and why, newest first. **Read this before starting a ticket**, together with the open PRs (`gh pr list --state open`), so earlier decisions and loose ends are not lost.

**Adding an entry:** every PR (Dependabot exempt) adds one entry at the top of the log below, in the same PR, so the log reaches `main` together with the work. Keep it short and record what the code does not say. Parallel PRs all add at the top, so they conflict here: keep both entries. `.gitattributes` sets `merge=union` for this file, which handles local merges; GitHub's web conflict editor ignores it.

```
## YYYY-MM-DD · T<x.y> <title> · #<PR> (closes #<issue>)
- **What:** files/behaviour changed.
- **Why:** reason for the approach chosen.
- **Decisions & gotchas:** anything a future session must know.
- **Follow-ups:** what is left, with ticket numbers.
```

Entries marked *(open when logged)* were backfilled while their PR was still open. Check its state on GitHub.

---

## 2026-10-01 · T2.1 OpenRouter client wrapper · #48 (closes #4)
- **What:** `src/llm.py`. `complete(messages, model, temperature, max_tokens, *, client=None)` calls OpenRouter's OpenAI-compatible endpoint through the `openai` SDK and returns the reply text. A model outside `ALLOWED_MODELS` raises `InvalidModelError` before a client is built or the key is read. Timeout, 401, 429 and 5xx raise `LLMTimeoutError`, `LLMAuthError`, `LLMRateLimitError` and `LLMServerError`; anything else (400, connection failure, empty reply) raises the base `LLMError`. All carry fixed user-readable messages, never the SDK text. 429/5xx are retried up to `MAX_RETRIES = 2` times with 1 s then 2 s backoff. `tests/test_llm.py` covers it at 100% with a mocked HTTP transport.
- **Why:** one small interface for the UI (T5.2) and streaming (T2.2) to build on; fixed messages keep response bodies and keys out of what the user sees.
- **Decisions & gotchas:** `openai` 3.x sends requests through **`httpx2`** (not `httpx`); tests mock it with `httpx2.MockTransport` via `make_client(key, http_client=...)`. The SDK retries 2× by default, so `make_client` sets `max_retries=0`; otherwise retries would multiply. Timeouts (30 s, `REQUEST_TIMEOUT`) are not retried, as the ticket only asks for 429/5xx. Backoff sleeps through `llm._sleep` so tests can patch it. No temperature range check and no `max_tokens` cap here. Review fixes: a missing or unparseable key (`MissingAPIKeyError`/`SecretsFileError` from `get_api_key()`) is re-raised by `make_client()` as `LLMAuthError`, so catching `LLMError` covers every failure; a whitespace-only reply counts as empty. Third review: the SDK does not validate 200 responses (a non-JSON body comes back as a plain `str`, malformed JSON raises `json.JSONDecodeError`, a choice may lack `message`), so `complete()` catches `ValueError` and `_reply_text()` reads every field defensively; all of these now end in `LLMError`. Fourth review: that `except ValueError` also catches request-side errors, so `make_client()` rejects a non-ASCII key (smart quotes, zero-width space; HTTP headers are ASCII) with `LLMAuthError` first, rather than letting it show as "unreadable answer". Fifth review: the check is `isascii() and isprintable()`, because a newline, CR or NUL in the key is ASCII but fails in h11 as a generic connection error that quotes the raw key. Gotcha: Claude's tool input turns backslash-u escapes into the real characters, so write test strings with invisible characters through `chr()` and check the file's bytes.
- **Follow-ups:** `max_tokens` cap and usage display in T2.3; streaming in T2.2. Check by hand with one live OpenRouter call in T2.3 (mocked tests cannot show either): (1) gpt-5 models spend reasoning tokens from `max_tokens`, so a small cap may return empty content with `finish_reason="length"`, which today shows as "empty answer, try again" and needs its own "answer was cut off" message; (2) OpenAI only accepts the default temperature for gpt-5, so confirm OpenRouter drops a non-default temperature rather than answering 400. The UI should catch `LLMError` and show only `str(exc)` (T5.2/T5.4), never `st.exception(exc)`: the chained SDK error (`__cause__`) holds the raw response body. From review, not done: friendlier messages for OpenRouter 402 (no credits) / 403 (moderation); honour `Retry-After` on 429; maybe move timeout/retry constants to `config.py`; cache the client with `st.cache_resource` in T5.2.

## 2026-10-01 · T6.8 Test plan in every ticket · #45 (closes #44)
- **What:** `docs/PLAN.md` has a "Testing strategy" section and a **Tests** checklist (Unit / UI flow / Manual) under every ticket that changes code; the same lists were added to GitHub issues #1, #2, #4–#6, #8–#11, #13–#16, #18–#22, #25–#27, #29–#32 and #40. T6.2 is now "fill gaps + coverage ≥ 80% enforced in CI"; the Definition of Done includes the Tests list.
- **Why:** the user wanted tests written with each ticket instead of all at the end, using unit tests and UI flow tests.
- **Decisions & gotchas:** no browser E2E suite (Playwright/Selenium): `AppTest` runs the real `app.py` with only the LLM faked, and T6.3's health check plus one manual live chat turn cover the deployed app. No test calls OpenRouter; UI flow tests use a `fake_llm` fixture in `tests/conftest.py`, created by the first ticket that needs it. Process/docs tickets (T6.1, T6.5, T6.7) have no Tests list. Merged `main` (#37, #43) and kept T6.7 before T6.8 in `docs/PLAN.md`. Review fixes: the T4.4 source scan allows the one static theme-CSS `unsafe_allow_html` in `app.py` (otherwise it would fail on today's code); clarified that patching works because `app.py` re-runs its `from src.x import y` lines, so other `src` modules must not bind LLM functions at import time; T6.6 (#40) got a manual check for the `linked-issue` job. Third review: T5.4 (#21) also checks that error messages never show the API key or the raw response body, since #48's exceptions chain the raw SDK error.
- **Follow-ups:** T6.2 adds `--cov=src --cov-fail-under=80` to CI.

## 2026-10-01 · T6.7 Progress log read before every ticket · #43 (closes #42)
- **What:** this file; `CLAUDE.md` and the `qrspi` skill now require reading it (plus open PRs) before a ticket and adding an entry in every PR.
- **Why:** the user did not want work from earlier sessions forgotten. A file in the repo is visible to the team and to every Claude session, unlike local memory.
- **Decisions & gotchas:** a single file was chosen over one file per PR so there is one place to read. Parallel PRs may conflict at the top of the log; resolve by keeping both entries (`.gitattributes` `merge=union` does this for local merges). Dependabot PRs are exempt, as in #41. If `gh` is not logged in, the `qrspi` skill goes on with only the log. Review fixes: merged `main` (#39) and kept T6.5 before T6.7 in `docs/PLAN.md`.
- **Follow-ups:** once #41 merges, add a "- [ ] `docs/PROGRESS.md` entry added" line to its PR template.

## 2026-10-01 · Repo setting: auto-delete head branches (no PR)
- **What:** turned on "Automatically delete head branches" on GitHub.
- **Why:** when a base PR merges, GitHub deletes its branch and retargets stacked PRs to `main`, so their `Closes #N` fires on merge.
- **Decisions & gotchas:** squash-merging a base PR leaves its commits duplicated in the stacked PR. When a PR has others stacked on it, merge it with a merge commit, or merge `main` into each stacked branch afterwards.

## 2026-10-01 · T6.6 Every PR closes its ticket · #41 (closes #40)
- **What:** `.github/pull_request_template.md` starts with `Closes #`. `.github/workflows/pr-checks.yml` (`linked-issue` job) asks GitHub (`gh pr view --json closingIssuesReferences`) whether the PR closes an issue and fails if not; Dependabot is exempt and the check re-runs when the description or base branch changes. `CLAUDE.md` has a "Pull requests" section.
- **Why:** #36 used `Refs #1`, which does not close a ticket, and nothing checked for it.
- **Decisions & gotchas:** the first version used a regex, but review showed it rejected forms GitHub accepts (`Closes: #40`, `owner/repo#40`, issue URLs, double spaces) and accepted text that closes nothing (`Closes #99999`, keywords in code spans). Asking GitHub is exact. GitHub links issues a few seconds after a description is saved, so the check retries for about 30 s; a failed `gh` call is retried too, because the step runs under `bash -e`. A `concurrency` group keeps one run per PR. GitHub only links issues for PRs into `main`, so a stacked PR falls back to a relaxed keyword regex until it is retargeted. Closing keywords only fire on merge into `main`. #36's description was fixed by hand to `Closes #1`.
- **Follow-ups:** make `linked-issue` a required status check in `main`'s branch protection (manual). After merge, confirm the next Dependabot PR shows `linked-issue` as skipped (not testable before, since Dependabot PRs are based on `main`). On the next stacked PR, check that GitHub's automatic retarget (base branch deleted) re-runs the check; a manual retarget does. Add a "- [ ] `docs/PROGRESS.md` entry added" line to the PR template (from T6.7, separate ticket).

## 2026-10-01 · T6.5 Code-reviewer subagent · #39 (closes #38), merged as `d6d9d19`
- **What:** `.claude/agents/code-reviewer.md`, a read-only reviewer. By default it reviews the current branch vs `main`. Its checklist covers correctness, security (secrets, guard, injection, output safety), mocked tests and conventions; it runs ruff + pytest, checks the ticket's acceptance criteria, and reports in a fixed format.
- **Why:** one agent rather than separate frontend/backend agents, because Streamlit UI and logic are the same Python codebase and most bugs sit where they meet.
- **Decisions & gotchas:** first committed on the T1.2 branch, then moved to its own ticket/branch at the user's request (one ticket per PR). Claude Code loads agents at session start, so start a new session to use it.

## 2026-10-01 · T1.2 Secrets management · #37 (closes #2) *(open when logged)*
- **What:** `src/config.py:get_api_key()` reads `OPENROUTER_API_KEY` from `st.secrets`, falls back to the env var, and raises `MissingAPIKeyError` if blank or missing. `app.py` shows a friendly error and `st.stop()`s. Adds `.streamlit/secrets.toml.example` and README setup in 5 steps. Also adds one-line docstrings to every function and documents that convention in `CLAUDE.md`.
- **Why:** `st.secrets` works locally and on Streamlit Cloud; the env var covers CI and other hosts.
- **Decisions & gotchas:** stacked on #36, so merge #36 first. Tests clear the env var and use fake keys, so CI needs no real key.
- **Follow-ups (from the code-reviewer dry run):** `test_secrets_example_has_placeholder_key_only` only checks that the placeholder exists, not that it is the only key. T1.2 boxes in `docs/PLAN.md` are not ticked.

## 2026-10-01 · T1.1 Restructure project & config · #36 (closes #1) *(open when logged)*
- **What:** `src/` package with `config.py` (`ALLOWED_MODELS`, `DEFAULT_MODEL = "openai/gpt-5-mini"`, temperature 0.0-1.5, default 0.7). Pins `openai==3.22.1` and `streamlit==1.64.0` in `requirements.txt` and `pyproject.toml`, with a test that keeps them identical. Removes `main.py` and adds a README stub. Adds `pythonpath = ["."]` to pytest so CI can import `src`.
- **Why:** gives later tickets (guard, prompts, llm) a home and one place for settings.
- **Decisions & gotchas:** `ALLOWED_MODELS` holds **only** `gpt-5-mini`, as agreed with the user, although the plan says "exactly the three from the brief". That acceptance criterion is knowingly unmet.
- **Follow-ups:** add the other two models later, or update the plan. `app.py` placeholder dashboard stays until T5.5.

## 2026-10-01 · Workflow rules: QRSPI skill · `72fe075` on `main` (no PR)
- **What:** `.claude/skills/qrspi/SKILL.md` and `CLAUDE.md`. Every coding task follows Question -> Research -> Structure -> Plan -> Implement.

## 2026-10-01 · Plan, CI/CD and smoke test · `b207f35` on `main` (no PR)
- **What:** `docs/PLAN.md` (7 epics, tickets with acceptance criteria), `.github/workflows/ci.yml` (ruff, pytest + coverage, `pip-audit`), `.github/workflows/cd.yml` (health check of the Streamlit Cloud URL after CI on `main`), Dependabot config, and the `AppTest` smoke test. GitHub issues #1-#33 were created from the plan; Dependabot opened #34 and #35.
- **Follow-ups:** T6.3 needs the repo variable `APP_URL` and the Cloud secret set by hand.

## 2026-09-30 · Initial commit · `c6cc35c`
