package com.readingbuddy.pipeline;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.common.PDRectangle;
import org.apache.pdfbox.pdmodel.font.PDFont;
import org.apache.pdfbox.pdmodel.font.PDType0Font;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.apache.pdfbox.pdmodel.font.Standard14Fonts;

import java.io.FileInputStream;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;

// Python equivalent: pdf_generator.py (generate_pdf using weasyprint / chromium)
//
// Key differences:
//   - Python uses weasyprint / chromium to render Markdown → PDF via HTML/CSS
//   - Java uses Apache PDFBox to build the PDF directly with a content stream
//   - PDType0Font.load() supports Unicode fonts (Noto Sans Sinhala) for Sinhala script
//   - Word wrapping is handled manually because PDFBox does not do it automatically
@Component
public class PdfGenerator {

    private static final float MARGIN = 50f;
    private static final float LINE_SPACING = 1.4f;

    public void generate(List<PageData> pages, Path outputPdf,
                         Path fontPath, float fontSize, String pageSize,
                         BiConsumer<Double, String> progress) throws IOException {

        PDRectangle mediaBox = "A4".equalsIgnoreCase(pageSize)
                ? PDRectangle.A4 : PDRectangle.LETTER;

        try (PDDocument doc = new PDDocument()) {

            // Try to load the custom font (NotoSansSinhala) — falls back to Helvetica if missing.
            // PDType0Font supports Unicode / complex scripts; PDType1Font is Latin-only.
            PDFont font = loadFont(doc, fontPath);

            int total = pages.size();

            for (int i = 0; i < total; i++) {
                PageData page = pages.get(i);
                String text = page.getTranslatedText();
                if (text == null) text = page.getDigitalText();
                if (text == null || text.isBlank()) text = "(no content)";

                PDPage pdfPage = new PDPage(mediaBox);
                doc.addPage(pdfPage);

                float width  = mediaBox.getWidth()  - 2 * MARGIN;
                float startY = mediaBox.getHeight() - MARGIN;

                try (PDPageContentStream stream =
                             new PDPageContentStream(doc, pdfPage)) {

                    stream.setFont(font, fontSize);
                    stream.setLeading(fontSize * LINE_SPACING);
                    stream.beginText();
                    stream.newLineAtOffset(MARGIN, startY);

                    // Manual word wrap — needed because PDFBox showText() has no built-in wrapping
                    for (String line : wrapText(text, font, fontSize, width)) {
                        stream.showText(line);
                        stream.newLine();
                    }

                    stream.endText();
                }

                double pct = ((double) (i + 1) / total) * 100;
                progress.accept(pct,
                        String.format("Generating PDF page %d/%d", page.getPageNumber(), total));
            }

            doc.save(outputPdf.toFile());
        }
    }

    private PDFont loadFont(PDDocument doc, Path fontPath) {
        if (fontPath != null && fontPath.toFile().exists()) {
            try (FileInputStream in = new FileInputStream(fontPath.toFile())) {
                // true = embed the full font subset so Sinhala glyphs render on any viewer
                // PDFBox 3.x removed the File overload — must use InputStream
                return PDType0Font.load(doc, in, true);
            } catch (IOException e) {
                System.err.println("Could not load custom font, falling back: " + e.getMessage());
            }
        }
        return new PDType1Font(Standard14Fonts.FontName.HELVETICA);
    }

    // Splits text into lines that fit within maxWidth points.
    // PDFBox's getStringWidth() returns width in 1/1000 text-space units, so divide by 1000 * fontSize.
    private List<String> wrapText(String text, PDFont font, float fontSize, float maxWidth)
            throws IOException {

        List<String> lines = new ArrayList<>();

        for (String paragraph : text.split("\\n")) {
            String[] words = paragraph.split("\\s+");
            StringBuilder current = new StringBuilder();

            for (String word : words) {
                String candidate = current.isEmpty() ? word : current + " " + word;
                float lineWidth = font.getStringWidth(candidate) / 1000f * fontSize;

                if (lineWidth > maxWidth && !current.isEmpty()) {
                    lines.add(current.toString());
                    current = new StringBuilder(word);
                } else {
                    current = new StringBuilder(candidate);
                }
            }

            if (!current.isEmpty()) lines.add(current.toString());
            lines.add("");  // blank line between paragraphs
        }

        return lines;
    }
}
