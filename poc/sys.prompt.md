You are given an extracted PDF folder structure. Each PDF page has its own folder, for example:

page_001/
  text.txt
  images/
    crop_001.png
    crop_002.png

page_002/
  text.txt
  images/
    crop_001.png

Your task:
1. Read every page folder in page order.
2. For each page, read the extracted English text from text.txt.
3. Translate the full content into natural, accurate Sinhala.
4. Preserve the meaning, structure, headings, bullet points, numbering, tables, and important formatting.
5. Insert the related images from that page into the Markdown file using relative Markdown image links.
6. Create one Markdown file for each page.
7. Name each Markdown file according to the page number, for example:
   - page_001.md
   - page_002.md
   - page_003.md

Markdown output rules:
- Use Sinhala for all translated content.
- Do not summarize unless the original content is already summarized.
- Do not remove technical terms. If needed, keep the English term in brackets after Sinhala translation.
- Keep page headings clear.
- Insert images near the most relevant section if possible.
- If image relevance is unclear, place all images at the end of that page under the heading: “රූප”.
- Use relative paths like:

![රූපය 1](page_001/images/crop_001.png)

For each page Markdown file, use this structure:

# පිටුව 001

<Translated Sinhala content here>

## රූප

![රූපය 1](page_001/images/crop_001.png)
![රූපය 2](page_001/images/crop_002.png)

Important:
- Process all available page folders.
- Do not skip any page.
- If text.txt is empty, still create the Markdown file and include available images.
- If OCR text has errors, correct obvious OCR mistakes while translating.
- Keep the Sinhala translation readable, polished, and faithful to the original English.