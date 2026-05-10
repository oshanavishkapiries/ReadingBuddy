import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_DIR = Path(__file__).resolve().parent

WORKSPACE = BASE_DIR.parent / "workspace"
WORKSPACE.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{BASE_DIR.parent / 'readingbuddy.db'}")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

DEFAULT_EXTRACTION = {
    "dpi": 300,
    "ocr_mode": "auto",
    "lang": "eng",
    "min_text_chars": 80,
    "psm": 6,
    "min_image_area": 2500,
    "save_page_render": True,
}

DEFAULT_TRANSLATION = {
    "model": os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    "temperature": 0.2,
    "max_tokens": 0,
    "retries": 3,
    "timeout": 120,
    "continue_on_error": True,
}

DEFAULT_PDF_GENERATION = {
    "page_size": "A4",
    "margin": "18mm",
    "font_file": str(BASE_DIR.parent / "poc" / "NotoSansSinhala-Regular.ttf"),
}
