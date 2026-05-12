package com.readingbuddy.pipeline;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.apache.pdfbox.text.PDFTextStripper;
import org.springframework.stereotype.Component;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;

// Python equivalent: extractor.py
//
// Key differences from the Python version:
//   - Apache PDFBox replaces pymupdf (fitz) for text extraction and page rendering
//   - No OCR here — PDFBox only reads embedded text. Adding Tesseract OCR would require
//     net.sourceforge.tess4j:tess4j and an extra preprocessing step (see Python extractor.py).
//   - PDFRenderer.renderImageWithDPI() replaces fitz's page.get_pixmap()
//   - PDFTextStripper replaces page.get_text("text")
@Component
public class PdfExtractor {

    // progress callback: (progressPercent, statusMessage)
    public List<PageData> extract(Path pdfPath, int dpi,
                                  BiConsumer<Double, String> progress) throws IOException {

        List<PageData> pages = new ArrayList<>();

        // Loader.loadPDF() is the PDFBox 3.x entry point (replaces PDDocument.load() from 2.x)
        try (PDDocument doc = Loader.loadPDF(pdfPath.toFile())) {

            int total = doc.getNumberOfPages();
            PDFRenderer renderer = new PDFRenderer(doc);

            // PDFTextStripper extracts text from one page at a time via setStartPage/setEndPage
            PDFTextStripper stripper = new PDFTextStripper();

            for (int i = 0; i < total; i++) {
                int pageNo = i + 1;

                // Extract digital text — replaces extract_digital_text() in Python
                stripper.setStartPage(pageNo);
                stripper.setEndPage(pageNo);
                String text = stripper.getText(doc).strip();

                // Render page to a BufferedImage — replaces render_page() in Python
                // ImageType.RGB matches Python's "RGB" mode in PIL
                BufferedImage image = renderer.renderImageWithDPI(i, dpi, ImageType.RGB);

                pages.add(PageData.builder()
                        .pageNumber(pageNo)
                        .digitalText(text)
                        .renderedImage(image)
                        .digitalChars(text.length())
                        .build());

                double pct = ((double) pageNo / total) * 100;
                progress.accept(pct,
                        String.format("Extracting page %d/%d (%d chars)", pageNo, total, text.length()));
            }
        }

        return pages;
    }
}
