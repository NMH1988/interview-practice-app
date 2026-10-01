# CLAUDE.md

Interview Practice App: Streamlit + OpenRouter. Plan and tickets: `docs/PLAN.md` and the GitHub issues.

## Workflow (mandatory)

For **every coding task** in this repository (features, bug fixes, refactors, issue tickets), invoke the `qrspi` skill (`.claude/skills/qrspi/SKILL.md`) first and follow its phases in order: Question -> Research -> Structure -> Plan -> Implement. Do not edit code before the Implement phase.

**Before starting any ticket**, read `docs/PROGRESS.md` (what earlier PRs did, decisions, follow-ups) and run `gh pr list --state open`. **Every PR adds its own entry** at the top of `docs/PROGRESS.md`.

## Commands

- Install: `pip install -r requirements-dev.txt`
- Run: `streamlit run app.py`
- Lint/format: `ruff check .` / `ruff format .`
- Test: `python -m pytest -q` (LLM calls must be mocked)

CI (`.github/workflows/ci.yml`) runs lint, tests and `pip-audit` on every push and PR.
