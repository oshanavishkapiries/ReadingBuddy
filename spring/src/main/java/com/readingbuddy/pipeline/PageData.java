package com.readingbuddy.pipeline;

import lombok.Builder;
import lombok.Data;

import java.awt.image.BufferedImage;

@Data
@Builder
public class PageData {

    private int pageNumber;

    private String digitalText;
    private BufferedImage renderedImage;
    private String translatedText;
    private int digitalChars;
    private int translatedChars;
}
