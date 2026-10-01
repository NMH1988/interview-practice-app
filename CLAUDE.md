# CLAUDE.md

Interview Practice App: Streamlit + OpenRouter. Plan and tickets: `docs/PLAN.md` and the GitHub issues.

## Workflow (mandatory)

For **every coding task** in this repository (features, bug fixes, refactors, issue tickets), invoke the `qrspi` skill (`.claude/skills/qrspi/SKILL.md`) first and follow its phases in order: Question -> Research -> Structure -> Plan -> Implement. Do not edit code before the Implement phase.

## Code conventions

- Every function, method and class (tests and fixtures included) gets a one-line docstring saying what it does in plain words.
- After each task, finish with a plain-language "what I did and why" walkthrough: each changed file, what changed, and why that choice was made.

## Commands

- Install: `pip install -r requirements-dev.txt`
- Run: `streamlit run app.py`
- Lint/format: `ruff check .` / `ruff format .`
- Test: `python -m pytest -q` (LLM calls must be mocked)

CI (`.github/workflows/ci.yml`) runs lint, tests and `pip-audit` on every push and PR.
