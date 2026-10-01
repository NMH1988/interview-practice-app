"""System prompt strategies: a registry of named functions `(role, interview_type) -> prompt`."""

from collections.abc import Callable

# The interview modes from docs/PLAN.md; T5.3 shows them in the UI.
INTERVIEW_TYPES: tuple[str, ...] = (
    "Behavioural",
    "Technical",
    "Questions to ask the interviewer",
    "Job-description analysis",
)

Strategy = Callable[[str, str], str]


def _coach_intro(role: str, interview_type: str) -> str:
    """Return the opening sentence every strategy shares."""
    return (
        f"You are an interview coach helping a candidate prepare for a {interview_type} "
        f"interview for the role of {role}."
    )


def zero_shot(role: str, interview_type: str) -> str:
    """Build a plain instruction-only prompt with no examples."""
    return (
        f"{_coach_intro(role, interview_type)} Ask one interview question at a time. "
        "When the candidate answers, give short, specific feedback and a better example answer."
    )


def few_shot(role: str, interview_type: str) -> str:
    """Build a prompt that shows an example question, answer and feedback."""
    return (
        f"{_coach_intro(role, interview_type)} Ask one question at a time and give feedback "
        "in the same style as this example.\n\n"
        "Question: Tell me about a time you missed a deadline.\n"
        "Answer: I was late once because a task took longer than planned.\n"
        "Feedback: Too vague. Name the project, say what you did to recover, and end with "
        "what you changed afterwards (Situation, Task, Action, Result)."
    )


def chain_of_thought(role: str, interview_type: str) -> str:
    """Build a prompt that asks the model to reason step by step before scoring."""
    return (
        f"{_coach_intro(role, interview_type)} Ask one question at a time. Before scoring an "
        "answer, think step by step: what a strong answer needs, what the candidate covered, "
        "and what is missing. Then give a score from 1 to 5 and concrete advice."
    )


def persona(role: str, interview_type: str) -> str:
    """Build a prompt where the model plays a strict senior interviewer."""
    return (
        f"You are a strict senior interviewer hiring for the role of {role}, running a "
        f"{interview_type} interview. Ask one demanding question at a time, probe weak "
        "answers with a follow-up, and give blunt but fair feedback."
    )


def structured_output(role: str, interview_type: str) -> str:
    """Build a prompt that requires feedback in fixed Markdown sections."""
    return (
        f"{_coach_intro(role, interview_type)} Ask one question at a time. Format feedback on "
        "each answer with these Markdown headings: **Score (1-5)**, **Strengths**, "
        "**Improvements**, **Example answer**."
    )


# Stable key -> strategy function. The keys are what config and the UI refer to.
STRATEGIES: dict[str, Strategy] = {
    "zero_shot": zero_shot,
    "few_shot": few_shot,
    "chain_of_thought": chain_of_thought,
    "persona": persona,
    "structured_output": structured_output,
}
