# ReadingBuddy

A web application that extracts PDF pages, translates content to Sinhala using AI, and generates a translated PDF.

## Pipeline

1. **PDF Extraction** - Extracts text (digital + OCR) and images from each PDF page
2. **AI Translation** - Translates extracted English text to Sinhala Markdown using OpenRouter
3. **PDF Generation** - Converts Sinhala Markdown back to a polished PDF with proper font rendering

## Setup

### Prerequisites

- Python 3.11+
- Tesseract OCR (for OCR support)
- Playwright Chromium (for PDF generation)

### Install Dependencies

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

On Windows, install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|---|---|---|
| `SECRET_KEY` | JWT secret for session tokens (generate with `python3 -c "import secrets; print(secrets.token_hex(32))"`) | `readingbuddy-secret-change-in-production` |
| `DATABASE_URL` | Database connection string | `sqlite:///./readingbuddy.db` |
| `OPENROUTER_API_KEY` | Backend OpenRouter API key (fallback when users don't have their own) | *(empty)* |
| `OPENROUTER_MODEL` | Default AI model for translation | `openai/gpt-4o-mini` |

### Run the Application

```bash
python3 -m uvicorn webapp.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

## Features

### User Accounts

- Register and login with username/email + password
- Session management via JWT cookies (7-day expiry)
- Passwords hashed with bcrypt

### API Key Fallback

The application resolves the OpenRouter API key in this order:

1. **User's personal key** - Set in Settings page
2. **Backend key** - `OPENROUTER_API_KEY` environment variable
3. **Error** - If neither is configured, processing is blocked

### Document Management

- Upload and manage PDF documents
- View processing history
- Download translated PDFs and Markdown files

### Settings

Each user can configure their own defaults:

- OpenRouter API key and model
- Extraction settings (DPI, OCR mode, language)
- Translation temperature
- PDF output (page size, margin)

## Project Structure

```
ReadingBuddy/
├── poc/                          # Original POC scripts
├── webapp/                       # Web application
│   ├── main.py                   # FastAPI app entry point
│   ├── config.py                 # Configuration
│   ├── auth.py                   # Authentication (JWT, bcrypt)
│   ├── database.py               # SQLite database setup
│   ├── models.py                 # Database models (User, Document, Job)
│   ├── tasks.py                  # Background task runner
│   ├── pipeline/                 # Pipeline modules
│   │   ├── extractor.py          # PDF extraction
│   │   ├── translator.py         # AI translation via OpenRouter
│   │   └── pdf_generator.py      # Markdown to PDF via Playwright
│   ├── static/                   # Static files (CSS)
│   └── templates/                # Jinja2 HTML templates
├── workspace/                    # Job outputs (auto-created)
├── .env.example                  # Environment variable template
├── requirements.txt
└── README.md
```

## Usage

1. Navigate to http://localhost:8000
2. Register a new account or login
3. Configure your OpenRouter API key in **Settings** (or use the backend default)
4. Go to **Dashboard** and upload a PDF
5. Monitor progress on the job detail page
6. Download the final translated PDF or Markdown files when complete
