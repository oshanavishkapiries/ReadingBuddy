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
                return PDType0Font.load(doc, in, true);
            } catch (IOException e) {
                System.err.println("Could not load custom font, falling back: " + e.getMessage());
            }
        }
        return new PDType1Font(Standard14Fonts.FontName.HELVETICA);
    }

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
