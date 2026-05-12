package com.readingbuddy.dto;

import lombok.Builder;
import lombok.Data;

// JSON response body for GET /job/{id}/status — polled by the job-detail page.
// Python equivalent: the dict returned by GET /job/{job_id}/status in main.py
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
