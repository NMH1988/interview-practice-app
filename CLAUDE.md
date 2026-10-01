# CLAUDE.md

Interview Practice App: Streamlit + OpenRouter. Plan and tickets: `docs/PLAN.md` and the GitHub issues.

## Workflow (mandatory)

For **every coding task** in this repository (features, bug fixes, refactors, issue tickets), invoke the `qrspi` skill (`.claude/skills/qrspi/SKILL.md`) first and follow its phases in order: Question -> Research -> Structure -> Plan -> Implement. Do not edit code before the Implement phase.

## Commands

- Install: `pip install -r requirements-dev.txt`
- Run: `streamlit run app.py`
- Lint/format: `ruff check .` / `ruff format .`
- Test: `python -m pytest -q` (LLM calls must be mocked)

CI (`.github/workflows/ci.yml`) runs lint, tests and `pip-audit` on every push and PR.

## Pull requests

- One ticket per PR. The description's first line is `Closes #<issue>` (not `Refs`), so GitHub closes the ticket on merge. `.github/workflows/pr-checks.yml` fails PRs without it (Dependabot is exempt).
- Closing keywords only fire when the PR merges into `main`. Retarget a stacked PR to `main` after its base PR merges, before merging it.
