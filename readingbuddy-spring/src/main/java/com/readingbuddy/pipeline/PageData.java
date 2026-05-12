package com.readingbuddy.pipeline;

import lombok.Builder;
import lombok.Data;

import java.awt.image.BufferedImage;

// Carries per-page extraction results through the pipeline.
// Python equivalent: the per-page output written to page_NNN/ directories in extractor.py
@Data
@Builder
public class PageData {

    private int pageNumber;

    // Text extracted by PDFBox — replaces digital_text.txt in Python
    private String digitalText;

    // Rendered raster image of the page — replaces page.png in Python
    private BufferedImage renderedImage;

    // Translated Sinhala text — filled in by Translator
    private String translatedText;

    // Character counts for pipeline logging
    private int digitalChars;
    private int translatedChars;
}
