---
name: qrspi
description: Enforces the QRSPI workflow (Question, Research, Structure, Plan, Implement) for any coding task in this repo - new features, bug fixes, refactors, tickets from GitHub issues. Use before writing or changing code.
---

# QRSPI workflow

Run the five phases **in order** for every coding task. Do not write or edit code before the Implement phase. Announce each phase with a heading (`## Q`, `## R`, `## S`, `## P`, `## I`) so progress is visible.

Trivial edits (typo, comment, formatting, a one-line config change) may collapse Q-R-S-P into a single sentence, but must still state the plan before editing.

## 1. Question
First read `docs/BRIEF.md` (the project brief), then `docs/PROGRESS.md`, and run `gh pr list --state open`, so the brief, earlier decisions, open work and follow-ups shape the questions. If `gh` is not installed or not logged in, say so and go on with only the log. Then break the task into clarifying questions.
- List every unknown that affects the solution: scope, inputs/outputs, edge cases, acceptance criteria, constraints, affected files.
- If the task comes from a GitHub issue, read it (`gh issue view <n>`) and turn each acceptance criterion into at least one question.
- Name the brief's requirement, starter idea or optional task the ticket serves, or write "beyond the brief (owner's request)" for work it does not cover (read the full brief when the exact wording matters; ask the user for it if it is not at hand). If the ticket, `docs/PLAN.md` or the prompts (e.g. `MODE_INSTRUCTIONS`) **contradict** the brief, or a choice changes **what the product does** (who speaks in a mode, what a mode produces, which users it serves), list it as a **needs the user** question instead of inferring an answer. Implementation details the brief does not mention follow the normal default rule below.
- Mark each question as **answerable from the code/docs** or **needs the user**.
- Ask the user the "needs the user" questions and wait for answers before moving on, unless a sensible default exists; if so, state the default explicitly. A brief contradiction or a choice that changes what the product does (the brief bullet above) is never settled by a default: always ask, even when the user requested the change.

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
- Give every new function, method and class (tests and fixtures included) a one-line docstring saying what it does.
- Stay inside the plan. If new information invalidates it, stop, go back to the relevant phase, and update the plan.
- Add an entry at the top of `docs/PROGRESS.md` (format at the top of that file) in the same PR: what changed, why, decisions/gotchas, follow-ups.
- Finish by reporting which acceptance criteria are met, what was verified, and anything not done, followed by a plain-language "what I did and why" walkthrough (each changed file, what changed, why that choice). Commit/push only when the user asks.
