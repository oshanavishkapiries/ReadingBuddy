#!/usr/bin/env python3
"""
OpenRouter-powered PDF extraction post-processor.

Reads folders produced by pdf_page_extractor.py, for example:

output/
  page_001/
    text.txt
    images/
      crop_001.png

Then creates Sinhala Markdown files, one per page:

output_markdown/
  page_001.md
  page_002.md

Usage:
  export OPENROUTER_API_KEY="your_key_here"
  python openrouter_markdown_translator.py output -o output_markdown \
    --model openai/gpt-4o-mini

Optional:
  python openrouter_markdown_translator.py output -o output_markdown \
    --model anthropic/claude-3.5-sonnet \
    --temperature 0.2 \
    --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
                # Avoid including full page render as a normal content image unless it is inside images/.
                if p.parent == page_dir and p.name.lower() in {"page.png", "page.jpg", "page.jpeg"}:
                    continue
                images.append(p)

    def img_key(p: Path) -> Tuple[int, str]:
        return natural_page_key(p)

    rels = []
    for img in sorted(images, key=img_key):
        rels.append(img.relative_to(base_dir).as_posix())
    return rels


def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: Optional[int],
    site_url: Optional[str],
    app_name: Optional[str],
    retries: int,
    timeout: int,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

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
                    # Some providers may return structured content parts.
                    content = "\n".join(str(part.get("text", part)) for part in content)
                return str(content).strip()
        except HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {e.code}: {err_body}"
            # Retry rate limits/server errors, fail fast on bad request/auth.
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
    # Remove accidental code fences.
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
            "> ⚠️ OpenRouter translation failed. පහත දැක්වෙන්නේ original extracted text එකයි.",
            "",
            text,
            "",
        ])
    if images:
        lines.extend(["## රූප", ""])
        for i, img in enumerate(images, start=1):
            lines.append(f"![රූපය {i}]({img})")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate extracted PDF pages to Sinhala Markdown using OpenRouter.")
    parser.add_argument("input_dir", help="Folder produced by pdf_page_extractor.py")
    parser.add_argument("-o", "--output-dir", default=None, help="Folder to write page_XXX.md files")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="OpenRouter model id")
    parser.add_argument("--api-key", default=None, help="OpenRouter API key. Prefer env OPENROUTER_API_KEY.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=0, help="0 means provider default")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Markdown files")
    parser.add_argument("--continue-on-error", action="store_true", help="Write fallback Markdown if an API call fails")
    parser.add_argument("--site-url", default=None, help="Optional OpenRouter HTTP-Referer header")
    parser.add_argument("--app-name", default="PDF Sinhala Markdown Translator", help="Optional OpenRouter X-Title header")
    parser.add_argument("--system-prompt-file", default=None, help="Optional custom system prompt text file")
    parser.add_argument("--combine", action="store_true", help="Also create combined.md containing all pages")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: input_dir does not exist or is not a folder: {input_dir}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir / "markdown"
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Missing OpenRouter API key. Set OPENROUTER_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8", errors="replace")

    page_dirs = find_page_folders(input_dir)
    if not page_dirs:
        print(f"ERROR: No page folders found in {input_dir}", file=sys.stderr)
        return 2

    combined_parts: List[str] = []
    print(f"Found {len(page_dirs)} page folder(s). Writing Markdown to: {output_dir}")

    for idx, page_dir in enumerate(page_dirs, start=1):
        page_number = page_number_from_name(page_dir.name, idx)
        out_file = output_dir / f"page_{page_number:03d}.md"

        if out_file.exists() and not args.overwrite:
            print(f"SKIP {page_dir.name}: {out_file.name} already exists. Use --overwrite to replace.")
            if args.combine:
                combined_parts.append(out_file.read_text(encoding="utf-8", errors="replace"))
            continue

        print(f"PROCESS {page_dir.name} -> {out_file.name}")
        user_prompt, _, images = build_user_prompt(page_dir, input_dir, idx)

        try:
            md = call_openrouter(
                api_key=api_key,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens if args.max_tokens > 0 else None,
                site_url=args.site_url,
                app_name=args.app_name,
                retries=args.retries,
                timeout=args.timeout,
            )
            md = clean_markdown(md)
        except Exception as e:
            if not args.continue_on_error:
                print(f"ERROR on {page_dir.name}: {e}", file=sys.stderr)
                return 1
            print(f"WARN {page_dir.name}: {e}. Writing fallback Markdown.", file=sys.stderr)
            md = fallback_markdown(page_dir, input_dir, idx)

        out_file.write_text(md, encoding="utf-8")
        combined_parts.append(md)

    if args.combine:
        combined_file = output_dir / "combined.md"
        combined_file.write_text("\n\n---\n\n".join(part.strip() for part in combined_parts) + "\n", encoding="utf-8")
        print(f"WROTE {combined_file}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
