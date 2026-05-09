from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_SYSTEM_PROMPT = """You are a careful Sinhala technical-document translator and Markdown formatter.

Audience:
Sri Lankan software engineers who understand English technical terms but need the document context explained clearly in casual, natural Sinhala.

Your job:
Convert OCR/PDF extracted English page text into Sinhala Markdown that is easy to understand, technically accurate, and faithful to the original.

Translation style:
- Use casual, natural Sinhala.
- Do NOT use overly formal Sinhala.
- Do NOT translate common software/engineering technical terms.
- Keep technical terms in English exactly where they are commonly used by software engineers.
- Translate explanations, descriptions, instructions, and surrounding context into Sinhala.
- Preserve the original meaning. Do not summarize. Do not invent missing content.
- Correct only obvious OCR mistakes while translating.

Formatting:
- Preserve headings, bullet points, numbering, tables, code blocks, command snippets, API names, file names, paths, URLs, variables, class names, function names, UI labels, product names, and version numbers.
- Return only valid Markdown content.
"""

USER_PROMPT_TEMPLATE = """You are processing one extracted PDF page.

Page folder name: {page_name}
Page number: {page_number}

Available image relative paths for this page:
{image_list}

English extracted text from text.txt:
--- BEGIN TEXT ---
{text}
--- END TEXT ---

Create Sinhala Markdown for this page.

Required output structure:
# පිටුව {page_number_padded}

<Translate the page content into casual Sinhala while preserving software/technical terms in English.>

If images are relevant to a section, insert the Markdown image link near that section.
If relevance is unclear, place all images at the end under:
## රූප

Image link format must use the exact relative paths provided, for example:
![රූපය 1]({example_image_path})

Important translation rules:
- Output Sinhala Markdown only.
- Do not wrap the answer in code fences.
- Do not mention that you are an AI.
- Do not skip content.
- Do not summarize.
- Use casual Sri Lankan Sinhala.
- Keep software engineering terms in English.
- Do NOT translate words like: API, database, server, client, frontend, backend, framework, library, package, module, component, deployment, pipeline, repository, branch, commit, pull request, issue, bug, feature, release, environment, variable, function, class, object, method, interface, endpoint, request, response, payload, authentication, authorization, token, cache, queue, event, service, microservice, container, Docker, Kubernetes, cloud, AWS, Azure, GCP, Linux, command, terminal, script, build, test, debug, log, error, exception, config, JSON, YAML, XML, HTML, CSS, JavaScript, TypeScript, Python, Java, SQL, NoSQL, Git, GitHub.
- Keep product names, tool names, file names, folder paths, commands, code, URLs, and UI labels exactly as they are.
- If an English technical term needs clarification, add a short casual Sinhala explanation after it.
  Example: API කියන්නේ system දෙකක් අතර data හුවමාරු කරන්න තියෙන interface එක.
- If text is empty, still create a useful page Markdown file with the page title and images.
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


def build_user_prompt(page_dir: Path, base_dir: Path, index: int) -> Tuple[str, int, List[str]]:
    text = read_text_file(page_dir)
    images = find_images(page_dir, base_dir)
    page_number = page_number_from_name(page_dir.name, index)
    padded = f"{page_number:03d}"
    image_list = "\n".join(f"- {img}" for img in images) if images else "- No images found for this page."
    example_path = images[0] if images else f"{page_dir.name}/images/crop_001.png"
    user_prompt = USER_PROMPT_TEMPLATE.format(
        page_name=page_dir.name,
        page_number=page_number,
        page_number_padded=padded,
        image_list=image_list,
        text=text,
        example_image_path=example_path,
    )
    return user_prompt, page_number, images


def fallback_markdown(page_dir: Path, base_dir: Path, index: int) -> str:
    text = read_text_file(page_dir)
    images = find_images(page_dir, base_dir)
    page_number = page_number_from_name(page_dir.name, index)
    padded = f"{page_number:03d}"
    lines = [f"# පිටුව {padded}", ""]
    if text:
        lines.extend([
            "> OpenRouter translation failed. පහත දැක්වෙන්නේ original extracted text එකයි.",
            "",
            text,
            "",
        ])
    if images:
        lines.extend(["## රූප", ""])
        for i, img in enumerate(images, start=1):
            lines.append(f"![රූපය {i}]({img})")
    return "\n".join(lines).strip() + "\n"


def translate_to_sinhala(
    input_dir: str,
    output_dir: str,
    api_key: str,
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 0,
    retries: int = 3,
    timeout: int = 120,
    continue_on_error: bool = True,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
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

        user_prompt, _, images = build_user_prompt(page_dir, input_path, idx)

        try:
            md = call_openrouter(
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
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
        if progress_callback:
            progress_callback(progress, "Translating to Sinhala", detail)

    combined_parts = []
    for md_file in sorted(output_path.glob("page_*.md"), key=natural_page_key):
        combined_parts.append(md_file.read_text(encoding="utf-8", errors="replace"))
    if combined_parts:
        combined_file = output_path / "combined.md"
        combined_file.write_text("\n\n---\n\n".join(part.strip() for part in combined_parts) + "\n", encoding="utf-8")

    return processed
