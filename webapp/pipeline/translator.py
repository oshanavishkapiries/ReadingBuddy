from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SUPPORTED_LANGUAGES = {
    "sinhala":  "Sinhala",
    "tamil":    "Tamil",
    "hindi":    "Hindi",
    "chinese":  "Chinese (Simplified)",
    "japanese": "Japanese",
    "korean":   "Korean",
    "arabic":   "Arabic",
}

DEFAULT_LANGUAGE = "sinhala"


def build_system_prompt(language: str = DEFAULT_LANGUAGE) -> str:
    lang_name = SUPPORTED_LANGUAGES.get(language.lower(), "Sinhala")
    return f"""You are a careful {lang_name} technical-document translator and Markdown formatter.

Your job:
Convert OCR/PDF extracted English page text into {lang_name} Markdown that is easy to understand, technically accurate, and faithful to the original.

Translation style:
- Use natural, conversational {lang_name} — the way an educated native speaker would actually say it out loud.
- Translate based on language context: ask yourself "would a native {lang_name} speaker say this word in English in everyday speech?" If yes, keep the English word. If no, translate it.
- Do NOT translate English words that native {lang_name} speakers have adopted as everyday loanwords and use in normal speech.
  Examples (keep these in English): car, bus, van, train, phone, mobile, camera, computer, table, chair, office, school, class, teacher, book, pen, bag, shop, market, bank, hospital, doctor, nurse, film, music, song, game, match, team, park, hotel, ticket, bill, tax, form, report, meeting, project, plan, map.
  The rule: if a {lang_name} speaker would say the English word naturally in a sentence, do not replace it with a formal or archaic {lang_name} equivalent.
- Do NOT translate software and engineering technical terms. Keep them in English exactly as written.
- Translate all explanations, instructions, descriptions, and conceptual content into natural {lang_name}.
- Never use a formal or archaic {lang_name} word if the English loanword is the one people actually use in speech.
- Preserve the original meaning exactly. Do not summarize. Do not skip content. Do not invent missing content.
- Correct only obvious OCR mistakes while translating.

Formatting:
- Preserve headings, bullet points, numbering, tables, code blocks, command snippets, API names, file names, paths, URLs, variables, class names, function names, UI labels, product names, and version numbers.
- Return only valid Markdown content.
"""


USER_PROMPT_TEMPLATE = """You are processing one extracted PDF page.

Page folder name: {page_name}
Page number: {page_number}
Target language: {lang_name}

Available image relative paths for this page:
{image_list}

English extracted text from text.txt:
--- BEGIN TEXT ---
{text}
--- END TEXT ---

Translate the page content into {lang_name} while preserving software/technical terms in English.

If images are relevant to a section, insert the Markdown image link near that section.
If relevance is unclear, place all images at the end under a heading in {lang_name}.

Image link format must use the exact relative paths provided, for example:
![image]({example_image_path})

Important translation rules:
- Output {lang_name} Markdown only.
- Do not wrap the answer in code fences.
- Do not mention that you are an AI.
- Do not skip content.
- Do not summarize.
- Keep software engineering terms in English.
- Do NOT translate words like: API, database, server, client, frontend, backend, framework, library, package, module, component, deployment, pipeline, repository, branch, commit, pull request, issue, bug, feature, release, environment, variable, function, class, object, method, interface, endpoint, request, response, payload, authentication, authorization, token, cache, queue, event, service, microservice, container, Docker, Kubernetes, cloud, AWS, Azure, GCP, Linux, command, terminal, script, build, test, debug, log, error, exception, config, JSON, YAML, XML, HTML, CSS, JavaScript, TypeScript, Python, Java, SQL, NoSQL, Git, GitHub.
- Keep product names, tool names, file names, folder paths, commands, code, URLs, and UI labels exactly as they are.
- If text is empty, still create a useful page Markdown file with any available images.
"""


def natural_page_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"(\d+)", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def find_page_folders(input_dir: Path) -> List[Path]:
    folders = [p for p in input_dir.iterdir() if p.is_dir() and re.search(r"page[_-]?\d+", p.name, re.I)]
    if not folders:
        folders = [p for p in input_dir.iterdir() if p.is_dir()]
    return sorted(folders, key=natural_page_key)


def read_text_file(page_dir: Path) -> str:
    preferred = ["text.txt", "ocr_text.txt", "digital_text.txt"]
    for name in preferred:
        path = page_dir / name
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                return text
    text_path = page_dir / "text.txt"
    if text_path.exists():
        return text_path.read_text(encoding="utf-8", errors="replace").strip()
    return ""


def find_images(page_dir: Path, base_dir: Path) -> List[str]:
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    images: List[Path] = []
    image_dir = page_dir / "images"
    search_roots = [image_dir] if image_dir.exists() else [page_dir]
    for root in search_roots:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in image_exts:
                if p.parent == page_dir and p.name.lower() in {"page.png", "page.jpg", "page.jpeg"}:
                    continue
                images.append(p)
    rels = []
    for img in sorted(images, key=natural_page_key):
        rels.append(img.relative_to(base_dir).as_posix())
    return rels


def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: Optional[int],
    retries: int,
    timeout: int,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if max_tokens is not None and max_tokens > 0:
        payload["max_tokens"] = max_tokens

    data = json.dumps(payload).encode("utf-8")
    last_error: Optional[str] = None

    for attempt in range(1, retries + 1):
        req = Request(OPENROUTER_URL, data=data, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                result = json.loads(body)
                choices = result.get("choices") or []
                if not choices:
                    raise RuntimeError(f"OpenRouter returned no choices: {result}")
                message = choices[0].get("message") or {}
                content = message.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(str(part.get("text", part)) for part in content)
                return str(content).strip()
        except HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {e.code}: {err_body}"
            if e.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except (URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as e:
            last_error = str(e)
        if attempt < retries:
            sleep_seconds = min(2 ** attempt, 20)
            time.sleep(sleep_seconds)

    raise RuntimeError(f"OpenRouter request failed after {retries} attempt(s): {last_error}")


IMAGE_RELEVANCE_THRESHOLD = 8  # keep images scoring >= this value (1–10 scale)

_RELEVANCE_PROMPT = """\
Score this image on how relevant it is to educational/learning content on a scale of 1 to 10.

Scoring guide:
10 — Core learning material: technical diagram, architecture diagram, flowchart, algorithm illustration,
     code snippet, data table, scientific figure, mathematical graph, circuit schematic, UML diagram.
7–9 — Supporting content: annotated screenshot, step-by-step UI walkthrough, labelled photo,
      comparison table, process overview.
4–6 — Marginally relevant: generic stock photo loosely related to the topic, simple decorative
      border that also contains text, section divider with some informational value.
1–3 — Not learning content: logo, brand mark, watermark, social media icon, advertisement,
      header/footer decoration, background pattern, publisher colophon.

Reply with a single integer between 1 and 10. Nothing else."""


def score_image_relevance(image_path: Path, api_key: str, model: str) -> int:
    """Return a 1–10 educational relevance score for the image via LLM vision.
    Returns IMAGE_RELEVANCE_THRESHOLD (pass) on any error so images are kept (fail-open)."""
    try:
        raw = image_path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        suffix = image_path.suffix.lower().lstrip(".")
        mime = "image/png" if suffix == "png" else f"image/{suffix}"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                        {"type": "text", "text": _RELEVANCE_PROMPT},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 3,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = json.dumps(payload).encode("utf-8")
        req = Request(OPENROUTER_URL, data=data, headers=headers, method="POST")
        with urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            answer = (result.get("choices") or [{}])[0].get("message", {}).get("content", "")
            if isinstance(answer, list):
                answer = " ".join(str(p.get("text", p)) for p in answer)
            match = re.search(r"\d+", answer.strip())
            if match:
                return max(1, min(10, int(match.group())))
    except Exception:
        pass
    return IMAGE_RELEVANCE_THRESHOLD  # fail-open


def filter_images(images: List[str], base_dir: Path, api_key: str, model: str) -> List[str]:
    """Keep only images whose educational relevance score meets the threshold."""
    kept = []
    for rel_path in images:
        abs_path = base_dir / rel_path
        if not abs_path.exists():
            kept.append(rel_path)
            continue
        score = score_image_relevance(abs_path, api_key, model)
        if score >= IMAGE_RELEVANCE_THRESHOLD:
            kept.append(rel_path)
    return kept


def clean_markdown(md: str) -> str:
    md = md.strip()
    if md.startswith("```"):
        md = re.sub(r"^```(?:markdown|md)?\s*", "", md, flags=re.I)
        md = re.sub(r"\s*```$", "", md)
    return md.strip() + "\n"


def page_number_from_name(page_name: str, fallback: int) -> int:
    match = re.search(r"(\d+)", page_name)
    if match:
        return int(match.group(1))
    return fallback


def build_user_prompt(
    page_dir: Path,
    base_dir: Path,
    index: int,
    language: str = DEFAULT_LANGUAGE,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    filter_decorative: bool = True,
) -> Tuple[str, int, List[str]]:
    text = read_text_file(page_dir)
    images = find_images(page_dir, base_dir)
    if filter_decorative and api_key and model and images:
        images = filter_images(images, base_dir, api_key, model)
    page_number = page_number_from_name(page_dir.name, index)
    image_list = "\n".join(f"- {img}" for img in images) if images else "- No images found for this page."
    example_path = images[0] if images else f"{page_dir.name}/images/crop_001.png"
    lang_name = SUPPORTED_LANGUAGES.get(language.lower(), "Sinhala")
    user_prompt = USER_PROMPT_TEMPLATE.format(
        page_name=page_dir.name,
        page_number=page_number,
        lang_name=lang_name,
        image_list=image_list,
        text=text,
        example_image_path=example_path,
    )
    return user_prompt, page_number, images


def fallback_markdown(page_dir: Path, base_dir: Path, index: int) -> str:
    text = read_text_file(page_dir)
    images = find_images(page_dir, base_dir)
    lines = []
    if text:
        lines.extend([
            "> Translation failed. Original extracted text below.",
            "",
            text,
            "",
        ])
    if images:
        lines.extend(["## Images", ""])
        for i, img in enumerate(images, start=1):
            lines.append(f"![image {i}]({img})")
    return "\n".join(lines).strip() + "\n"


def translate_to_sinhala(
    input_dir: str,
    output_dir: str,
    api_key: str,
    model: str = "openai/gpt-4o-mini",
    language: str = DEFAULT_LANGUAGE,
    temperature: float = 0.2,
    max_tokens: int = 0,
    retries: int = 3,
    timeout: int = 120,
    continue_on_error: bool = True,
    progress_callback: Callable[[float, str, str], None] | None = None,
) -> int:
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    page_dirs = find_page_folders(input_path)
    if not page_dirs:
        raise ValueError(f"No page folders found in {input_dir}")

    total = len(page_dirs)
    processed = 0

    for idx, page_dir in enumerate(page_dirs, start=1):
        page_number = page_number_from_name(page_dir.name, idx)
        out_file = output_path / f"page_{page_number:03d}.md"

        user_prompt, _, images = build_user_prompt(
            page_dir, input_path, idx, language,
            api_key=api_key, model=model, filter_decorative=True,
        )

        try:
            md = call_openrouter(
                api_key=api_key,
                model=model,
                system_prompt=build_system_prompt(language),
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens > 0 else None,
                retries=retries,
                timeout=timeout,
            )
            md = clean_markdown(md)
        except Exception as e:
            if not continue_on_error:
                raise
            md = fallback_markdown(page_dir, input_path, idx)

        out_file.write_text(md, encoding="utf-8")
        processed += 1

        progress = (processed / total) * 100
        detail = f"Page {processed}/{total}: {page_dir.name}"
        lang_name = SUPPORTED_LANGUAGES.get(language.lower(), "Sinhala")
        if progress_callback:
            progress_callback(progress, f"Translating to {lang_name}", detail)

    combined_parts = []
    for md_file in sorted(output_path.glob("page_*.md"), key=natural_page_key):
        combined_parts.append(md_file.read_text(encoding="utf-8", errors="replace"))
    if combined_parts:
        combined_file = output_path / "combined.md"
        combined_file.write_text("\n\n---\n\n".join(part.strip() for part in combined_parts) + "\n", encoding="utf-8")

    return processed
