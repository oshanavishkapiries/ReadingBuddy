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

@Component
public class PdfExtractor {

    public List<PageData> extract(Path pdfPath, int dpi,
                                  BiConsumer<Double, String> progress) throws IOException {

        List<PageData> pages = new ArrayList<>();

        try (PDDocument doc = Loader.loadPDF(pdfPath.toFile())) {

            int total = doc.getNumberOfPages();
            PDFRenderer renderer = new PDFRenderer(doc);

            PDFTextStripper stripper = new PDFTextStripper();

            for (int i = 0; i < total; i++) {
                int pageNo = i + 1;

                stripper.setStartPage(pageNo);
                stripper.setEndPage(pageNo);
                String text = stripper.getText(doc).strip();

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
