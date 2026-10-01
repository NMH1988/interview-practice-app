# Interview Practice App

A Streamlit app that helps you practise job interviews, powered by LLMs via
[OpenRouter](https://openrouter.ai) (default model: `openai/gpt-5-mini`).

## Local setup

1. Clone the repo and `cd` into it (requires Python 3.14+).
2. Install dependencies: `pip install -r requirements-dev.txt`
3. Copy the secrets template: `cp .streamlit/secrets.toml.example .streamlit/secrets.toml`
   (Windows: `copy .streamlit\secrets.toml.example .streamlit\secrets.toml`)
4. Paste your [OpenRouter key](https://openrouter.ai/keys) into `.streamlit/secrets.toml`, or
   set the `OPENROUTER_API_KEY` environment variable instead.
5. Run the app: `streamlit run app.py`

`.streamlit/secrets.toml` and `.env` are gitignored, so never commit your key. If the key is
missing, the app shows a setup message instead of starting.

## Development

- Lint/format: `ruff check .` / `ruff format .`
- Test: `python -m pytest -q` (LLM calls are mocked; no API key needed)

Model settings live in `src/config.py`. See [docs/PLAN.md](docs/PLAN.md) for the roadmap.
