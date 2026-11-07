# Resume Parser

Extracts structured data (contact info, summary, work experience, education, skills,
certifications, projects, languages) from PDF/DOCX resumes using an LLM: a **local model via Ollama (default)** or the **Anthropic Claude API** (optional).
FastAPI backend + Streamlit UI.

## Architecture

```
Streamlit UI  --HTTP-->  FastAPI (/api/upload, /api/resume/{id})
                              |-- services/parser.py       PDF/DOCX -> text (PyMuPDF, python-docx)
                              |-- services/llm_service.py  text -> JSON (Ollama or Claude, schema-constrained)
                              |-- services/storage.py      JSON file persistence in data/results
```

Key design choices:
- **Structured outputs**: the Pydantic schema is passed to Ollama's `format` parameter, so the
  model is forced to return valid JSON; output is re-validated with Pydantic and retried once.
- `temperature=0` and an explicit "never invent data" prompt to reduce hallucination.
- `num_ctx` is raised (Ollama's default context is small and would silently truncate resumes).

## Setup

1. **Install Ollama** from https://ollama.com and pull a model:
   ```bash
   ollama pull llama3.1:8b
   ```
2. **Create environment & install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env            # Windows: copy .env.example .env
   ```
3. **Run the API** (terminal 1)
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   Docs: http://localhost:8000/docs
4. **Run the UI** (terminal 2)
   ```bash
   streamlit run ui/streamlit_app.py
   ```

## API

| Method | Path | Description |
|---|---|---|
| POST | `/api/upload` | Upload PDF/DOCX (`multipart/form-data`, field `file`) -> 201 + extracted data + `document_id` |
| GET | `/api/resume/{document_id}` | Fetch stored result, 404 if unknown |
| GET | `/health` | Health check |

Status codes: `400` invalid/corrupt file, `413` too large, `422` no readable text (scanned PDF),
`502` bad LLM output, `503` Ollama not reachable.

```bash
curl -X POST http://localhost:8000/api/upload -F "file=@resume.pdf"
```

## Choosing the LLM provider

Set `LLM_PROVIDER` in `.env`:

- `ollama` (default): fully local, no API key. Needs Ollama running and a pulled model.
- `anthropic`: set `ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_MODEL`). No Ollama needed.
  Output is constrained to the schema via a forced tool call.

`LLM_MODE=multi` (default) splits extraction into three focused calls (basics, experience, skills),
which is far more reliable with small local models. `LLM_MODE=single` uses one call and is faster
with larger models or Claude.

Both providers use the same prompts and Pydantic validation; only `services/llm_service.py` differs.

## Configuration

See `.env.example`. `LLM_MODEL` can be any Ollama model (e.g. `mistral`, `qwen2.5:7b`).

## Tests

The tests mock the LLM, so they need neither Ollama nor an API key:

```bash
pip install -r requirements-dev.txt
pytest -v
```

They cover file validation, PDF/DOCX extraction, the regex/skills fallbacks, hallucination
clean-up, the LLM service (modes, retries, error translation) and the API status codes.

## Limitations / future work

- Scanned (image-only) PDFs need OCR (e.g. Tesseract) - currently returns a clear 422 error.
- Results are stored as JSON files; swap in SQLite/Postgres for production.
- Add a Dockerfile and CI (e.g. GitHub Actions running `pytest`).
