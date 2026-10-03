# Project brief (summary)

**Read this before starting any ticket** (see `CLAUDE.md` and the `qrspi` skill). It summarises the course brief *Sprint 1, Part 5: Build an Interview Practice App* (Turing College, AI Engineering) in our own words. The repo is public, so the course text is not copied here. The full brief is on the course platform, and the project owner keeps a local copy. When this summary and the full brief seem to differ, the full brief wins: ask the owner.

## What the project is

Build a **single-page interview-preparation web app** in Streamlit (or Next.js). It calls the OpenRouter API, uses a system prompt and the user's own prompt with interview-prep instructions, and has security guards. It should bring together the whole sprint: calling OpenRouter, prompt techniques, LLM settings and security guards. Estimated time: about 5 hours.

The brief stresses freedom: *"We don't want you to put it in a box."* You can practise interview questions, programming questions, the questions to ask at the end of an interview, or job-description analysis, and mix them freely. So **never lock the app or its content to one field, role or level.** The optional tasks name IT, finance, HR and communication as example fields.

Diagram: Streamlit UI → security guard → system prompt + user prompt → OpenRouter API → LLM (model settings such as temperature) → generated interview answer back to the UI.

## Mandatory requirements

1. Research and choose the kind of interview preparation the app does. Exploring and being creative is part of the task.
2. Decide how to build the front end (Streamlit components, or HTML/CSS in Next.js).
3. Create an OpenRouter API key for the project.
4. Use one of these models: `openai/gpt-5-mini` (recommended default), `openai/gpt-5-nano` (cheaper) or `openai/gpt-5` (stronger). Embeddings, only if needed (e.g. optional RAG): `qwen/qwen3-embedding-8b` (default), `openai/text-embedding-3-small`, `openai/text-embedding-3-large`.
5. Write **at least 5 system prompts with different techniques** (few-shot, chain-of-thought, zero-shot, ...) **and check which one works best.**
6. Add **at least one security guard** against misuse.

## Starter ideas (optional, "swap, mix, or extend")

1. **Role-based Q&A generator.** Job title + seniority in, about 8–10 likely interview questions out.
2. **Behavioural answer coach (STAR).** Paste a draft answer, get a critique against Situation/Task/Action/Result and a tighter rewrite.
3. **"Questions to ask the interviewer" generator.** Company + role in, 5–8 thoughtful questions to ask at the end of the interview out. Our current mode *rates* the user's own question instead; #62 brings it back to the generator.
4. **Job description analyser.** Paste a job description, get the key skills, likely interview topics and a short study plan.
5. **Self-introduction polisher.** Paste a 30-second pitch, get an interview-ready rewrite and what to keep or cut.

## Optional tasks (numbering as in the brief, so tickets can cite them)

**Easy**
1. Ask ChatGPT to critique the solution for usability, security and prompt engineering.
2. Improve prompts for your own domain (IT, finance, HR, communication, ...).
3. More security constraints: input validation and system-prompt validation (possibly checked by an LLM).
4. Difficulty levels for the questions (easy, medium, hard).
5. Concise vs. detailed answers through prompting.
6. Generate interviewer guidelines: structured evaluation criteria for technical and behavioural interviews.
7. Mock interview with AI personas (strict, neutral, friendly).
8. Tune at least one model setting (temperature, max tokens, reasoning effort, ...) and compare the output.

**Medium**
1. Let the user tune all model settings (model, temperature, max tokens, ...) as sliders or fields. (#64)
2. At least two structured JSON output formats.
3. Show the price of the prompt (pricing from OpenRouter's models endpoint).
4. Read the OpenRouter docs and implement your own improvement.
5. Try to jailbreak your own app and record the results in a spreadsheet.
6. A separate field for the job description you are applying for, with preparation for that position (RAG).
7. Let the user choose from a list of LLMs.
8. Use image generation creatively (`google/gemini-2.5-flash-image`).
9. A security guard, designed for usability: keep developer settings (model, system prompts) separate from the user experience. (#63)

**Hard**
1. A full chatbot instead of a one-time call. (Done: T5.2/T2.2.)
2. LangChain chains or agents.
3. A vector database to spot interview data seen before and prompt for new data.
4. Open-source LLMs.
5. Assess prompt or model performance, e.g. LLM-as-a-judge.

## How the project is evaluated

- **Core concepts:** explain the prompting techniques, how LLM settings (temperature, max tokens, ...) change the output, the user/system/assistant roles, and the different output types.
- **Technical implementation:** the app works as intended, so you can prepare for an interview by asking it for help. It calls OpenRouter with the correct parameters and uses a front-end library for the UI.
- **Reflection:** explain the choice of prompt techniques and settings, know the app's potential problems, and suggest improvements.
- **Bonus:** for maximum points, implement **at least 2 medium and 1 hard** optional task.

## Using this file

- At the start of every ticket, name the requirement, idea or optional task the ticket serves. Name the brief's own wording when it matters.
- If our plan, prompts or mode instructions differ from the brief, say so and ask the owner. Don't infer what the brief "probably means".
- Submission goes through the GitHub repository created for the course.
