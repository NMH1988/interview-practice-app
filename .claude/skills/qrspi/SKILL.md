---
name: qrspi
description: Enforces the QRSPI workflow (Question, Research, Structure, Plan, Implement) for any coding task in this repo - new features, bug fixes, refactors, tickets from GitHub issues. Use before writing or changing code.
---

# QRSPI workflow

Run the five phases **in order** for every coding task. Do not write or edit code before the Implement phase. Announce each phase with a heading (`## Q`, `## R`, `## S`, `## P`, `## I`) so progress is visible.

Trivial edits (typo, comment, formatting, a one-line config change) may collapse Q-R-S-P into a single sentence, but must still state the plan before editing.

## 1. Question
First read `docs/PROGRESS.md` and run `gh pr list --state open`, so earlier decisions, open work and follow-ups shape the questions. If `gh` is not installed or not logged in, say so and go on with only the log. Then break the task into clarifying questions.
- List every unknown that affects the solution: scope, inputs/outputs, edge cases, acceptance criteria, constraints, affected files.
- If the task comes from a GitHub issue, read it (`gh issue view <n>`) and turn each acceptance criterion into at least one question.
- Mark each question as **answerable from the code/docs** or **needs the user**.
- Ask the user the "needs the user" questions and wait for answers before moving on, unless a sensible default exists; if so, state the default explicitly.

## 2. Research
Answer each question **using only facts from the codebase or provided documents** (files, `docs/PLAN.md`, `docs/PROGRESS.md`, issues, README, the user's messages).
- Read the relevant files; run read-only commands (grep, tests, `git log`) as needed.
- For each answer, cite the source (`path:line`, issue number, or document).
- Never answer from assumption or memory. If the repo does not contain the answer, write "Not found in codebase" and either ask the user or record it as an explicit assumption.
- Do not propose solutions yet.

## 3. Structure
Organize the work into **vertical slices**.
- Each slice delivers a thin, end-to-end, independently verifiable behaviour (e.g. UI -> guard -> prompt -> LLM client for one path), not a horizontal layer ("all models first").
- Order slices so the first one is the smallest working path; later slices extend it.
- For each slice give: goal, files touched, how it is verified (test or manual check), and which acceptance criteria it satisfies.

## 4. Plan
Produce a detailed implementation plan from the slices.
- Numbered steps per slice: exact files/functions to create or change, tests to write first or alongside, commands to run.
- Include risks and rollback notes where relevant.
- Present the plan to the user and **wait for approval** before implementing, unless the user already said to proceed.

## 5. Implement
Execute the plan step by step.
- Work one slice at a time; after each slice run `ruff check .`, `ruff format --check .` and `python -m pytest -q` and fix failures before continuing.
- Mock all LLM/network calls in tests; never put API keys in code or the repo.
- Stay inside the plan. If new information invalidates it, stop, go back to the relevant phase, and update the plan.
- Add an entry at the top of `docs/PROGRESS.md` (format at the top of that file) in the same PR: what changed, why, decisions/gotchas, follow-ups.
- Finish by reporting which acceptance criteria are met, what was verified, and anything not done. Commit/push only when the user asks.
