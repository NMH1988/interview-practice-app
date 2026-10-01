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

## 2026-10-01 · T6.9 Code conventions in CLAUDE.md · #47 (closes #46)
- **What:** `CLAUDE.md` gets a "Code conventions" section (one-line docstring on every function/method/class, tests included; a "what I did and why" walkthrough after each task). The `qrspi` skill's Implement phase repeats both. `tests/test_conventions.py` fails if any function or class in `app.py`, `src/` or `tests/` lacks a docstring.
- **Why:** the user asked for both rules during T1.2. The section was first added in #37, then removed there in review because it was out of scope; this ticket restores it on its own.
- **Decisions & gotchas:** the docstring rule is enforced by a test (via `ast`), so it no longer depends on anyone remembering it; nested helper functions count too. The walkthrough rule cannot be tested, so it lives in `CLAUDE.md` and the skill only. Added a **Tests** list (in `docs/PLAN.md` and #46) to follow T6.8's rule. The check lives in a `missing_docstrings` helper with its own unit test on a temporary file, so a broken check (one that never fails) is caught too. Review fix: files are parsed from bytes (`ast.parse(path.read_bytes(), filename=...)`), so a UTF-8 BOM (Notepad) does not break the check and a SyntaxError names the file.
- **Follow-ups:** `docs/PLAN.md` and the top of this file may conflict with #41 / #45; keep all sections and entries.

## 2026-10-01 · T6.7 Progress log read before every ticket · #43 (closes #42)
- **What:** this file; `CLAUDE.md` and the `qrspi` skill now require reading it (plus open PRs) before a ticket and adding an entry in every PR.
- **Why:** the user did not want work from earlier sessions forgotten. A file in the repo is visible to the team and to every Claude session, unlike local memory.
- **Decisions & gotchas:** a single file was chosen over one file per PR so there is one place to read. Parallel PRs may conflict at the top of the log; resolve by keeping both entries (`.gitattributes` `merge=union` does this for local merges). Dependabot PRs are exempt, as in #41. If `gh` is not logged in, the `qrspi` skill goes on with only the log. Review fixes: merged `main` (#39) and kept T6.5 before T6.7 in `docs/PLAN.md`.
- **Follow-ups:** once #41 merges, add a "- [ ] `docs/PROGRESS.md` entry added" line to its PR template.

## 2026-10-01 · Repo setting: auto-delete head branches (no PR)
- **What:** turned on "Automatically delete head branches" on GitHub.
- **Why:** when a base PR merges, GitHub deletes its branch and retargets stacked PRs to `main`, so their `Closes #N` fires on merge.
- **Decisions & gotchas:** squash-merging a base PR leaves its commits duplicated in the stacked PR. When a PR has others stacked on it, merge it with a merge commit, or merge `main` into each stacked branch afterwards.

## 2026-10-01 · T6.6 Every PR closes its ticket · #41 (closes #40) *(open when logged)*
- **What:** `.github/pull_request_template.md` starts with `Closes #`. `.github/workflows/pr-checks.yml` (`linked-issue` job) fails PRs without a `Closes/Fixes/Resolves #N` keyword; Dependabot is exempt and the check re-runs when the description is edited. `CLAUDE.md` has a "Pull requests" section.
- **Why:** #36 used `Refs #1`, which does not close a ticket, and nothing checked for it.
- **Decisions & gotchas:** closing keywords only fire on merge into `main`. #36's description was fixed by hand to `Closes #1`.
- **Follow-ups:** make `linked-issue` a required status check in `main`'s branch protection (manual). `docs/PLAN.md` may conflict with #39 and #43 (all add a ticket after T6.4); keep all sections.

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
