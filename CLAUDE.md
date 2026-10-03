# CLAUDE.md

Interview Practice App: Streamlit + OpenRouter. Brief: `docs/BRIEF.md`. Plan and tickets: `docs/PLAN.md` and the GitHub issues.

## Workflow (mandatory)

For **every coding task** in this repository (features, bug fixes, refactors, issue tickets), invoke the `qrspi` skill (`.claude/skills/qrspi/SKILL.md`) first and follow its phases in order: Question -> Research -> Structure -> Plan -> Implement. Do not edit code before the Implement phase.

**Before starting any ticket**, read `docs/BRIEF.md` (the project brief: what the course asks for; the full brief wins if they differ) and check the ticket against it, then read `docs/PROGRESS.md` (what earlier PRs did, decisions, follow-ups) and run `gh pr list --state open`. If the ticket's idea, our plan or the prompts differ from the brief, say so and ask; don't infer what the brief probably means. **Every PR adds its own entry** at the top of `docs/PROGRESS.md` (Dependabot exempt).

## Code conventions

- Every function, method and class (tests and fixtures included) gets a one-line docstring saying what it does in plain words. `tests/test_conventions.py` enforces this.
- After each task, finish with a plain-language "what I did and why" walkthrough: each changed file, what changed, and why that choice was made.

## Commands

- Install: `pip install -r requirements-dev.txt`
- Run: `streamlit run app.py`
- Lint/format: `ruff check .` / `ruff format .`
- Test: `python -m pytest -q` (LLM calls must be mocked)

CI (`.github/workflows/ci.yml`) runs lint, tests and `pip-audit` on every push and PR.

## Pull requests

- One ticket per PR. The description must contain `Closes #<issue>` (not `Refs`), preferably on the first line, so GitHub closes the ticket on merge. `.github/workflows/pr-checks.yml` asks GitHub whether the PR closes an issue and fails if not (Dependabot is exempt; a stacked PR is checked by keyword until it targets `main`).
- Closing keywords only fire when the PR merges into `main`. Retarget a stacked PR to `main` after its base PR merges, before merging it.
