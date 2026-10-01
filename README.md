# Interview Practice App

A Streamlit app that helps you practise job interviews, powered by LLMs via
[OpenRouter](https://openrouter.ai) (default model: `openai/gpt-5-mini`).

## Quick start

```bash
pip install -r requirements-dev.txt
streamlit run app.py
```

## Development

- Lint/format: `ruff check .` / `ruff format .`
- Test: `python -m pytest -q` (LLM calls are mocked; no API key needed)

Model settings live in `src/config.py`. See [docs/PLAN.md](docs/PLAN.md) for the roadmap.
