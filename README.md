# Resume Parser – LLM-Based Resume Information Extraction

An AI-powered Resume Parser that extracts structured information from PDF and DOCX resumes using Large Language Models (LLMs).

The application uses **FastAPI** for the backend, **Streamlit** for the frontend, and **Ollama** for local LLM-based information extraction. It also supports Anthropic Claude as an optional LLM provider.

## Key Features

- Upload resumes in PDF and DOCX formats.
- Extract contact information, professional summary, work experience, education, skills, certifications, projects, and languages.
- Generate structured JSON output using LLMs.
- Validate extracted information using Pydantic models.
- Support local LLM inference through Ollama.
- Optional Anthropic Claude API integration.
- Store extracted results as JSON files.
- Retrieve previously processed resumes using a unique document ID.
- Handle invalid files, extraction errors, and LLM failures.
- Provide interactive API documentation using Swagger UI.

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| Frontend | Streamlit |
| LLM | Ollama / Anthropic Claude |
| Data Validation | Pydantic |
| PDF Processing | PyMuPDF |
| DOCX Processing | python-docx |
| Storage | JSON file-based storage |
| Testing | Pytest |

## Project Architecture

```text
                User
                  |
                  v
             Streamlit UI
                  |
               HTTP API
                  |
                  v
              FastAPI
                  |
          Resume Parser Service
                  |
          PDF / DOCX Extraction
                  |
                  v
             LLM Service
             /         \
         Ollama      Claude API
             \         /
                  |
           Structured JSON
                  |
                  v
          Pydantic Validation
                  |
                  v
             JSON Storage
                  |
                  v
          Extracted Resume Data
```

## Project Structure

```text
resume-parser/
│
├── app/
│   ├── main.py
│   ├── models.py
│   ├── services/
│   │   ├── parser.py
│   │   ├── llm_service.py
│   │   ├── storage.py
│   │   ├── fallback.py
│   │   └── postprocess.py
│   └── utils/
│       └── logger.py
│
├── data/
│   └── results/
│
├── scripts/
├── tests/
├── ui/
│   └── streamlit_app.py
│
├── .env.example
├── .gitignore
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/jafarbasha-dev/resume-parser-llm.git
cd resume-parser-llm
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file from the example:

```bash
copy .env.example .env
```

Update the configuration according to your selected LLM provider.

### 5. Configure Ollama (Default Provider)

Install Ollama from:

https://ollama.com

Download the model:

```bash
ollama pull llama3.1:8b
```

Make sure the Ollama service is running before starting the application.

### 6. Start the FastAPI Backend

```bash
uvicorn app.main:app --reload --port 8000
```

API documentation:

http://localhost:8000/docs

### 7. Start the Streamlit Frontend

Open another terminal, activate the virtual environment, and run:

```bash
streamlit run ui/streamlit_app.py
```

The Streamlit interface will open in your browser.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/upload` | Upload and process a PDF/DOCX resume |
| GET | `/api/resume/{document_id}` | Retrieve extracted resume information |
| GET | `/health` | Check API health |

### Example API Request

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@resume.pdf"
```

## LLM Configuration

The application supports two LLM providers.

### Ollama

- Runs models locally.
- Does not require an external API key.
- Default model: `llama3.1:8b`.

### Anthropic Claude

- Uses the Anthropic API.
- Requires an API key.
- Configure the provider and model in the `.env` file.

Example configuration:

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
LLM_MODE=multi
```

Refer to `.env.example` for the available configuration options.

## Data Extraction and Validation

The application extracts structured resume information, including:

- Personal and contact details.
- Professional summary.
- Employment history.
- Educational qualifications.
- Technical and soft skills.
- Certifications.
- Projects and languages.

The extracted response is validated against Pydantic schemas to maintain a consistent JSON structure.

The application also uses explicit prompts and output validation to reduce incorrect or unsupported information.

## Testing

The project includes automated tests using Pytest.

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the test suite:

```bash
pytest -v
```

**Note:** The automated tests mock LLM responses. They do not require a running Ollama service or an external API key.

## Error Handling

The API handles common errors, including:

- Invalid or unsupported file formats.
- Corrupted documents.
- Files exceeding the configured size limit.
- PDFs without extractable text.
- LLM service failures.
- Invalid or missing document IDs.

## Current Limitations and Future Enhancements

- OCR support for scanned or image-based PDF resumes.
- Database integration using SQLite or PostgreSQL.
- Docker-based deployment.
- CI/CD integration using GitHub Actions.
- Additional resume formats and extraction improvements.

## Author

**Jafar Basha**

GitHub: https://github.com/jafarbasha-dev

---

**Project:** Resume Parser – LLM-Based Information Extraction

**Built with:** Python, FastAPI, Streamlit, Ollama, and Pydantic.