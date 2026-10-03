# Project brief (summary)

**Read this before starting any ticket** (see `CLAUDE.md` and the `qrspi` skill). It summarises the course brief *Sprint 1, Part 5: Build an Interview Practice App* (Turing College, AI Engineering) in our own words. The repo is public, so the course text is not copied here. The full brief is on the course platform, and the project owner keeps a local copy. When this summary and the full brief seem to differ, the full brief wins: ask the owner.

## What the project is

Build a **single-page interview-preparation web app** in Streamlit (or Next.js). It calls the OpenRouter API with a system prompt and a user prompt that **we write** to carry the interview-prep instructions (in this repo: `prompts.STRATEGIES` and `build_user_prompt`). It also has security guards. The app should bring together the whole sprint: calling OpenRouter, prompt techniques, LLM settings and security guards. Estimated time: about 5 hours.

Diagram: Streamlit UI → security guard → system prompt + user prompt → OpenRouter API → LLM (model settings such as temperature) → generated interview answer back to the UI.

### Freedom, and what we chose with it

The brief gives the learner a free hand in **what** to practise: *"We don't want you to put it in a box."* It lists interview questions, questions on a specific programming language, the questions to ask at the end of an interview, and job-description analysis as examples, and encourages experimenting. The brief does **not** require the app to cover every field. One optional task (Easy #2) even suggests tuning the prompts to *your own* domain.

**Our design decision** (not a brief rule): the app works for any role and level. The role is free text (T5.1), seniority has four levels, and per-role content such as the sample-job-description starter (T5.4) is built from them rather than fixed to one field. Keep new work consistent with this, unless the owner decides otherwise.

## Mandatory requirements

1. Research and choose the kind of interview preparation the app does. Exploring and being creative is part of the task.
2. Decide how to build the front end (Streamlit components, or HTML/CSS in Next.js).
3. Create an OpenRouter API key for the project.
4. Use one of these models: `openai/gpt-5-mini` (recommended default), `openai/gpt-5-nano` (cheaper) or `openai/gpt-5` (stronger). Embeddings, only if needed (e.g. optional RAG): `qwen/qwen3-embedding-8b` (default), `openai/text-embedding-3-small`, `openai/text-embedding-3-large`.
5. Write five or more system prompts, each using a different technique (e.g. few-shot, chain-of-thought, zero-shot), **then compare them and find the one that works best.**
6. Add **at least one security guard** against misuse.

## Starter ideas (optional, "swap, mix, or extend")

1. **Role-based Q&A generator.** Job title + seniority in, about 8–10 likely interview questions out.
2. **Behavioural answer coach (STAR).** Paste a draft answer, get a critique against Situation/Task/Action/Result and a tighter rewrite.
3. **"Questions to ask the interviewer" generator.** Company name + role in; out comes a short list (5–8) of good questions for the end of the interview, tailored to that company. Our current mode *rates* the user's own question instead; #62 brings it back to the generator.
4. **Job description analyser.** Paste a job description; the app pulls out the main skills and the topics an interviewer is likely to ask about, plus a short plan of what to study.
5. **Self-introduction polisher.** Paste a 30-second pitch, get an interview-ready rewrite and what to keep or cut.

## Optional tasks (numbering as in the brief, so tickets can cite them)

Do these **only after the core app works**. Anyone with software experience is invited to over-engineer the app as a portfolio project, with the extras below **and ideas of our own**. So an Epic 7 extra is allowed even when the brief does not list it. Tags in brackets show our status: *planned* (a ticket exists), *done* (merged).

**Easy**
1. Ask ChatGPT to critique the solution for usability, security and prompt engineering.
2. Improve prompts for your own domain (IT, finance, HR, communication, ...).
3. More security constraints: input validation and system-prompt validation (possibly checked by an LLM).
4. Difficulty levels for the questions (easy, medium, hard).
5. Concise vs. detailed answers through prompting.
6. Have the model draft interviewer guidelines: a structured scoring guide for technical and behavioural interviews.
7. Mock interview with AI personas (strict, neutral, friendly).
8. Change at least one model setting (temperature, max tokens, reasoning effort, ...) and compare the outputs.

**Medium**
1. Let the user set every model setting (model, temperature, max tokens, ...) through sliders or fields. (*planned:* #64)
2. Two or more structured JSON output formats.
3. Show the price of the prompt (pricing from OpenRouter's models endpoint).
4. Read the OpenRouter docs and implement your own improvement.
5. Try to jailbreak your own app and record the results in a spreadsheet.
6. A separate field for the job description you are applying for, with preparation for that position (RAG).
7. Let the user pick from LLMs of different providers (e.g. Gemini, OpenAI). Our `ALLOWED_MODELS` holds only the three OpenAI models, so this is **not** done.
8. Use image generation creatively (`google/gemini-2.5-flash-image`).
9. A security guard, designed for usability: keep developer settings (model, system prompts) separate from the user experience. (*planned:* #63)

**Hard**
1. A full chatbot instead of a one-time call. (*done:* T5.2, T2.2)
2. LangChain chains or agents.
3. A vector database to spot interview data seen before and prompt for new data.
4. Open-source LLMs.
5. Assess prompt or model performance, e.g. LLM-as-a-judge.

## How the project is evaluated

- **Core concepts:** explain the prompting techniques, how LLM settings (temperature, max tokens, ...) change the output, the user/system/assistant roles, and the different output types.
- **Technical implementation:** the app works as intended, so you can prepare for an interview by asking it for help. It calls OpenRouter with the correct parameters and uses a front-end library for the UI.
- **Reflection:** explain the choice of prompt techniques and settings, know the app's potential problems, and suggest improvements.
- **Bonus:** top marks need two medium optional tasks and one hard one, at least.

## Using this file

- At the start of every ticket, name the requirement, idea or optional task the ticket serves.
- When the exact wording matters, or this summary is unclear, read the full brief (ask the owner for it if it is not at hand). Don't guess from the summary.
- If our plan, prompts or mode instructions **contradict** the brief, or a choice changes **what the product does** (who speaks in a mode, what a mode produces, which users it serves), say so and ask the owner. Ordinary implementation choices the brief says nothing about follow the usual QRSPI rule: use a sensible default and state it.
- Submission and the project review go through the GitHub repository the course provides: this repo, `NMH1988/interview-practice-app` (confirmed by the owner). Reviewers read it as submitted, so keep `main`, the README and `docs/` in a state you would hand in.
