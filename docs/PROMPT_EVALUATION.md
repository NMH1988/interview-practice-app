# Prompt evaluation

This is the evidence for the brief's mandatory requirement #5: *"Write at least 5 system prompts with different techniques … and check which one works best for you."* It also covers optional task Easy #8 (compare how a model setting changes the output) and Hard #5 (assess the performance of the prompts, here by scoring them by hand). Ticket: T3.3 (#10).

**Result:** Structured output and Zero-shot tied at 92/100. The project owner chose **Structured output** as the app's default strategy, as a product-design preference (see [The winner](#the-winner)). The default reasoning effort stays **medium**.

The raw replies, with the owner's scores and notes, are committed next to this file:

- [`eval/run-20261004-223148.md`](eval/run-20261004-223148.md): the strategy comparison (25 replies).
- [`eval/run-20261005-103856.md`](eval/run-20261005-103856.md): the reasoning-effort comparison (8 replies).

## Method

- **Script:** `python -m scripts.prompt_eval` (`scripts/prompt_eval.py`). It sends each input the way the app does: the input guard with the mode's length limit, the strategy's system prompt, the user prompt from `build_user_prompt`, `llm.stream()`, and then the output check. It records each reply with its length, time and token counts. All LLM calls in the script's tests are mocked; the runs here used the real API.
- **Model and settings:** `openai/gpt-5-mini` (the app's default), reasoning effort `medium` (the app's default) and the app's token budget (4,000). One run per strategy and input: 5 inputs × 5 strategies = 25 requests.
- **Inputs:** written by the project owner, one or two per interview mode, covering all four seniority levels. The companies are fictional, so an invented fact about them is easy to spot. For Behavioural and Technical, the coach's question is sent as the previous turn, so the answer is judged against a question the model has seen.
- **Scoring:** the owner scored every reply from 1 to 5 on four criteria (the rubric in `docs/PLAN.md`): **Relevance**, **Actionability**, **Structure** and **Tone**. That gives up to 20 points per input and 100 per strategy. The scoring was not blind; the strategy names were visible.
- **Two scoring rules**, agreed before the final pass:
  1. A reply that breaks a rule written in the mode's instructions (`MODE_INSTRUCTIONS`, for example the answer-review format in "Questions to ask the interviewer") always loses points, and the note says why.
  2. Each defect is deducted under one criterion only.
- **Score revisions:** the scores were revised during review, after questions about consistency. The owner first chose Structured output when an earlier pass tied it with Zero-shot at 92–92. The final pass, which re-checked every table under the two rules above, tied them again at 93–93, and the owner kept the choice. The PR review then found that no job-description reply ends with its study plan, which the mode requires ("End the analysis by … creating a short, practical study plan"); the owner counted it as a mode rule, so every strategy lost one Structure point on `jd-review` (92–92, ranking unchanged). The report holds only the final scores.

### Test inputs

| Id | Mode | Seniority | Role | Coach's question first |
|---|---|---|---|---|
| `qta-suggest` | Questions to ask the interviewer | Senior | SAP Developer | no |
| `behavioral` | Behavioural | Senior | Marketing Manager | yes |
| `technical` | Technical | Junior | Software Engineer | yes |
| `qta-own` | Questions to ask the interviewer | Lead | Software Engineer | no |
| `jd-review` | Job-description analysis | Mid-level | SAP Developer | no |

**`qta-suggest`** (asks for suggestions, names a company and one fact about it):
> I'm interviewing at Velnora Logistics. They are replacing several legacy warehouse systems with SAP S/4HANA. What would be good questions to ask at the end of the interview?

**`behavioral`**, after the coach asked *"Tell me about a time when you had a disagreement with a teammate. How did you handle the situation, and what was the outcome?"* (weak on purpose: no result, and the candidate did not follow up):
> During a product launch, a teammate and I disagreed about whether to move more budget from email to paid social. We reviewed the previous campaign results and discussed both approaches with the team. In the end, we agreed to test both channels before making a larger budget shift. The launch went ahead on time, but I did not track the final impact of the decision myself.

**`technical`**, after the coach asked *"Can you explain the difference between a REST API and a GraphQL API, and when you would choose one over the other?"* (correct, but brief):
> REST uses multiple endpoints and fixed responses, while GraphQL uses one endpoint and lets the client request exactly the data it needs. I would choose REST for simpler APIs and GraphQL for more flexible data requirements.

**`qta-own`** (asks for feedback on the candidate's own question):
> Please help me evaluate whether this is a good question: "I would like to know more about the upcoming projects that I could be involved in."

**`jd-review`** (a pasted job description):
```
Can you help me prepare for an interview based on this job description?

SAP Developer - Orvexa Manufacturing

Responsibilities:
- Develop and maintain ABAP applications in an SAP S/4HANA environment.
- Build CDS views and OData services for business applications.
- Integrate SAP systems with external services and APIs.
- Analyze and resolve technical issues and performance problems.
- Participate in code reviews and work with functional consultants and other developers.

Requirements:
- At least 3 years of experience with ABAP development.
- Experience with SAP S/4HANA, CDS views, and OData.
- Good knowledge of SQL and REST APIs.
- Experience with Git and modern development workflows.
- Ability to work in an Agile team and communicate with technical and non-technical stakeholders.
- Good English communication skills.
```

## Results

All 25 requests succeeded: none was blocked by the input guard, refused by the output check, cut off by the token limit, or failed.

### Score per input (out of 20) and in total (out of 100)

| Strategy | qta-suggest | behavioral | technical | qta-own | jd-review | **Total** |
|---|---|---|---|---|---|---|
| Structured output | 20 | 17 | 17 | 20 | 18 | **92** |
| Zero-shot | 20 | 18 | 17 | 19 | 18 | **92** |
| Chain-of-thought | 20 | 17 | 16 | 20 | 17 | **90** |
| Role / persona | 19 | 19 | 16 | 17 | 18 | **89** |
| Few-shot | 19 | 18 | 17 | 17 | 17 | **88** |

### Score per criterion (out of 25) and cost

| Strategy | Relevance | Actionability | Structure | Tone | Mode-rule deductions | Completion tokens (5 replies) | Time (5 replies) |
|---|---|---|---|---|---|---|---|
| Structured output | 21 | 25 | 21 | 25 | 1 (`jd-review`) | 8,720 | 76.9 s |
| Zero-shot | 21 | 25 | 21 | 25 | 2 (`qta-own`, `jd-review`) | 7,879 | 79.5 s |
| Chain-of-thought | 20 | 25 | 20 | 25 | 1 (`jd-review`) | 7,829 | 61.0 s |
| Role / persona | 21 | 25 | 18 | 25 | 2 (`qta-own`, `jd-review`) | 7,737 | 72.3 s |
| Few-shot | 19 | 25 | 19 | 25 | 2 (`qta-own`, `jd-review`) | 6,924 | 58.4 s |

Completion tokens include the model's hidden reasoning. Few-shot had the largest prompts (1,176–1,635 tokens, because of its three examples) but the shortest replies.

### What the scores show

- **The top three are not clearly separated.** Structured output, Zero-shot and Chain-of-thought are within two points of each other. With one run per combination, a gap that small can come from run-to-run variation (see [Limitations](#limitations)).
- **Few-shot and Role / persona fell behind on `qta-own`.** Both switched to the answer-review format (`## Evaluation`, `## Feedback`, `## Follow-up Question`, and Persona also `## Expected Answer`), which the mode's instructions forbid for feedback on the candidate's own question. Each lost three points under Structure. Zero-shot lost one point there for adding a list of follow-up questions, which the mode also forbids unless the candidate asks for suggestions. On `jd-review` every strategy broke the same mode rule (see problem 3 below), so that deduction does not change the ranking.
- **Actionability and Tone did not separate the strategies:** every strategy scored 25/25 on both. All the differences come from Relevance and Structure.

### Problems shared by all or most strategies

These do not depend on the technique, so they point to the shared parts of the prompts or to the model, not to the choice of strategy:

1. **Invented facts in sample answers** (`behavioral`). Every strategy except Role / persona wrote a model answer with made-up campaign numbers and results (for example "increasing monthly revenue by 12%"), and some made the candidate the owner of a measurement they said they did not track. A candidate could repeat those numbers as if they were real. Relevance 3 for four strategies, 4 for Role / persona.
2. **Seniority not respected** (`technical`). For a Junior candidate, all five strategies asked for production-level topics such as N+1 queries, persisted queries, rate limiting and query-cost limits. Chain-of-thought and Structured output mentioned the Junior level but still asked for that depth. Relevance 3 for all five.
3. **Long job-description analyses that do not end with the study plan** (`jd-review`). Every reply was 7,861–9,135 characters, repeating material across the skills, topics, keywords, gaps and study-plan sections. Every reply included a study plan of exactly five items, each with one review topic and one practice task, but none ended with it, as the mode requires: all five added general tips and an offer to continue after it. Each strategy lost one Structure point for the repetition and one for the mode rule (Structure 3 for all five).
4. **A few-shot example leaked into another mode.** Few-shot's job-description reply listed "temporal dead zone (not relevant here — ignore)" among the keywords. The term comes from its JavaScript example (Example 1 in `FEW_SHOT_EXAMPLES`).

Things that worked in every reply: `qta-suggest` got 7 or 8 questions (the mode asks for 5–8) with one reason line each, tailored to the S/4HANA fact, with no invented facts about the company; Role / persona and Chain-of-thought were checked on this path too, which T5.6's live check had not covered. No strategy asked the candidate an interview question in "Questions to ask the interviewer".

## The winner

The project owner's reading of the tie:

> 1. The inputs were already clear: Zero-shot had enough task, role, and seniority information to produce strong responses.
> 2. Structured formatting did not prevent content errors: Both strategies invented details in the behavioral case and exceeded Junior expectations in the technical case.
> 3. Their strengths offset each other: Zero-shot scored one point higher in behavioral, while Structured output scored one point higher in qta-own, resulting in 92/100 for both.

A fact from the code that supports point 1: every strategy is the technique's own text joined to the same session context and the same shared rules, including the mode's instructions (`_assemble` in `src/prompts.py`). Zero-shot therefore already carries most of the guidance; only the technique part differs between strategies.

The project owner's choice:

> I selected Structured output as a product-design preference. In the five evaluated inputs, it followed the mode-specific presentation rules, apart from the job-description study-plan rule, which all five strategies broke, and provided a more focused qta-own response than Zero-shot. Its section-based organization in Behavioural and Technical responses may also help users locate feedback and expected answers, although the app renders all responses as Markdown and does not validate their format by mode.
>
> The trade-off is approximately 11% more completion tokens than Zero-shot (8,720 versus 7,879), and an additional scoring rubric in the behavioural response that needs to be reviewed. Both strategies scored 92/100, so this choice does not demonstrate superior overall response quality.

`DEFAULT_STRATEGY = "structured_output"` in `src/config.py` makes it the strategy the sidebar starts with; the user can still pick any of the five under "Developer settings".

## Reasoning effort (Easy #8)

The allowed gpt-5 models ignore `temperature` (T2.4, #68), so the setting compared is **reasoning effort**: how long the model thinks before it answers. Structured output was run on `behavioral` and `jd-review` at all four levels, one run each, with the app's budgets (4,000 tokens, 16,000 at `high`). The brief asks to compare how the output changes, so these replies were compared by reading them, not scored.

| Input | Effort | Reasoning tokens | Visible tokens | Characters | Time |
|---|---|---|---|---|---|
| behavioral | minimal | 0 | 584 | 2,737 | 7.4 s |
| | low | 64 | 665 | 3,132 | 7.2 s |
| | medium | 768 | 811 | 3,453 | 15.1 s |
| | high | 3,904 | 961 | 4,032 | 30.8 s |
| jd-review | minimal | 0 | 1,963 | 9,018 | 23.0 s |
| | low | 64 | 2,001 | 9,391 | 16.4 s |
| | medium | 512 | 1,874 | 8,243 | 21.1 s |
| | high | 4,288 | 1,877 | 8,181 | 46.6 s |

"Visible tokens" is the completion minus the reasoning, that is, the text the user reads.

What changed:

- **Higher effort mostly buys hidden thinking, not longer answers.** Reasoning went from 0 to about 4,000 tokens, while the visible job-description analysis stayed at about 1,900 tokens. The behavioural feedback grew from 584 to 961.
- **Cost and wait:** `high` used about 2.6–3.1 times the completion tokens of `medium` and took about twice as long (31–47 s against 15–21 s). At gpt-5-mini's output price recorded in T2.5 ($2 per million tokens), the `high` job-description reply cost about $0.012 and the `medium` one about $0.005.
- **Content at `high`:** the behavioural feedback added a five-step measurement plan and advice on recovering the missing results after the fact, which fits the candidate's real gap. But its model answer invented numbers like every other level (for example a "$120k budget" and "~250 conversions"), and the job-description plan made manufacturing-domain knowledge one of its five study items, although the description does not ask for it.
- **Content at `minimal` and `low`:** in the behavioural feedback, the rubric was more lenient (Correctness 4 and Depth 3, against 3 and 2 at `medium` and `high`), and the model answer began by echoing the prompt's own condition, for example "(Include because the original answer omitted …)". The structured-output prompt says "Include only when the answer is incorrect or significantly incomplete." The echo paraphrases the prompt rather than copying it, and the output check refuses only a reply that repeats, word for word, the whole system prompt or one of its paragraphs of 80 characters or more. So it is a wording problem, not a leak the app refuses.
- **No timeout:** the `high` job-description request took 46.6 s with 4,288 reasoning tokens (the longest think seen so far) and was not cut off by the 30-second read timeout. The run does not show why the connection stayed alive during the thinking (for example, whether OpenRouter sends keep-alive messages), so T2.5's open timeout question is answered for this case only.

**The project owner's conclusion** (translated from Vietnamese): keep `medium`, because `high` takes too long to answer and costs about three times as much.

## Limitations

- **One run per combination and five inputs.** The same setting gives different replies from run to run. Structured output at `medium` appears in both reports: `behavioral` was 4,064 characters with 704 reasoning tokens in one run and 3,453 with 768 in the other; `jd-review` was 9,135 with 576 and 8,243 with 512. Length differences of that size are as large as those between nearby effort levels, so small score gaps should not be read as a ranking.
- **One model.** Only `gpt-5-mini` was evaluated; `gpt-5-nano` and `gpt-5` may rank the strategies differently.
- **One scorer, not blind.** The owner scored all replies with the strategy names visible, and revised the scores during review (see [Method](#method)).
- **Editor formatting.** Saving the first report in the editor reformatted its Markdown (blank lines inside blockquotes, `1)` lists as `1.`, `*` emphasis as `_`, aligned tables). The checked passages read the same as before; the file is not byte-for-byte what the script wrote.
- **The effort run's header** says `293ad94 (with uncommitted code changes)`. The uncommitted change was `DEFAULT_STRATEGY` in `src/config.py` and the select's default in `app.py`, which the script does not read; `src/prompts.py` was unchanged, so the prompts sent were the committed ones.

## Follow-ups

- **T3.5 (#76)** measures, over several runs, how often feedback on the candidate's own question uses the answer-review format, ends with a question or gives more than one rewording. Data point from `qta-own`: Few-shot and Role / persona used the review headings; every strategy gave two or three rewordings, where the mode allows "a better wording".
- **Not ticketed, from the shared problems above:**
  - sample answers invent facts (problem 1);
  - feedback ignores the Junior level (problem 2);
  - long, repetitive job-description analyses that do not end with the study plan (problem 3);
  - the Few-shot JavaScript example leaking into other modes (problem 4);
  - the `minimal` / `low` prompt echo.
