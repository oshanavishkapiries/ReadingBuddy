package com.readingbuddy.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class JobStatusDto {
    private String id;
    private String filename;
    private String status;
    private double progress;
    private String currentStep;
    private String stepDetail;
    private String error;
    private String outputPdf;
    private int pageCount;
}
