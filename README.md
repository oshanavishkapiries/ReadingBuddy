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

```bash
# Required for translation
export OPENROUTER_API_KEY="your_openrouter_api_key"

# Optional: default model
export OPENROUTER_MODEL="openai/gpt-4o-mini"
```

### Run the Application

```bash
cd D:\ReadingBuddy
uvicorn webapp.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

## Project Structure

```
ReadingBuddy/
├── poc/                          # Original POC scripts
├── webapp/                       # Web application
│   ├── main.py                   # FastAPI app entry point
│   ├── config.py                 # Configuration
│   ├── database.py               # SQLite database setup
│   ├── models.py                 # Database models
│   ├── tasks.py                  # Background task runner
│   ├── pipeline/                 # Pipeline modules
│   │   ├── extractor.py          # PDF extraction
│   │   ├── translator.py         # AI translation via OpenRouter
│   │   └── pdf_generator.py      # Markdown to PDF via Playwright
│   ├── static/                   # Static files (CSS)
│   └── templates/                # Jinja2 HTML templates
├── workspace/                    # Job outputs (auto-created)
├── requirements.txt
└── README.md
```

## Usage

1. Navigate to the home page
2. Upload a PDF file
3. Optionally configure extraction, translation, and output settings
4. Click "Start Processing"
5. Monitor progress on the job detail page
6. Download the final translated PDF or Markdown files when complete
