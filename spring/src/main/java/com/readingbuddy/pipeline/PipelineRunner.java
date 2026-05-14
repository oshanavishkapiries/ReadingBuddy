package com.readingbuddy.pipeline;

import com.readingbuddy.service.JobService;
import com.readingbuddy.service.UsageService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

// Python equivalent: run_pipeline() in tasks.py
//
// Orchestrates the three pipeline stages: Extract → Translate → Generate PDF.
// Called from JobService.startPipeline() which is @Async — so this runs in a background thread.
@Component
public class PipelineRunner {

    @Autowired private PdfExtractor extractor;
    @Autowired private Translator translator;
    @Autowired private PdfGenerator generator;
    @Autowired private UsageService usageService;

    @Value("${app.workspace}")
    private String workspace;

    @Value("${app.openrouter-api-key:}")
    private String backendApiKey;

    @SuppressWarnings("unchecked")
    public void run(String jobId, Path pdfPath, Map<String, Object> settings, JobService jobService) {
        try {
            Map<String, Object> extSettings   = (Map<String, Object>) settings.getOrDefault("extraction",   Map.of());
            Map<String, Object> transSettings = (Map<String, Object>) settings.getOrDefault("translation",  Map.of());
            Map<String, Object> pdfSettings   = (Map<String, Object>) settings.getOrDefault("pdf_generation", Map.of());

            String apiKey = (String) transSettings.getOrDefault("api_key", backendApiKey);
            if (apiKey == null || apiKey.isBlank()) {
                throw new IllegalStateException("No OpenRouter API key configured.");
            }

            String model       = (String) transSettings.getOrDefault("model", "openai/gpt-4o-mini");
            double temperature = ((Number) transSettings.getOrDefault("temperature", 0.2)).doubleValue();
            int dpi            = ((Number) extSettings.getOrDefault("dpi", 300)).intValue();
            String pageSize    = (String) pdfSettings.getOrDefault("page_size", "A4");
            float fontSize     = ((Number) pdfSettings.getOrDefault("font_size", 16.5)).floatValue();

            Path jobWorkspace = Path.of(workspace, jobId);
            Files.createDirectories(jobWorkspace);
            Path outputPdf = jobWorkspace.resolve("final.pdf");

            // Stage 1: Extract text and render pages
            jobService.updateProgress(jobId, 5, "Extracting PDF", "Starting extraction...");
            List<PageData> pages = extractor.extract(pdfPath, dpi,
                    (pct, msg) -> jobService.updateProgress(jobId, 5 + pct * 0.35, "Extracting PDF", msg));

            if (pages.size() > 100) {
                throw new IllegalStateException(
                        "PDF has " + pages.size() + " pages. Free plan limit is 100 pages.");
            }

            // Stage 2: Translate each page via OpenRouter
            jobService.updateProgress(jobId, 40, "Translating", "Sending pages to OpenRouter...");
            translator.translate(pages, apiKey, model, temperature,
                    (pct, msg) -> jobService.updateProgress(jobId, 40 + pct * 0.45, "Translating", msg));

            // Stage 3: Build the output PDF
            jobService.updateProgress(jobId, 85, "Generating PDF", "Writing pages...");

            Path fontPath = Path.of("poc", "NotoSansSinhala-Regular.ttf");
            generator.generate(pages, outputPdf, fontPath, fontSize, pageSize,
                    (pct, msg) -> jobService.updateProgress(jobId, 85 + pct * 0.14, "Generating PDF", msg));

            // Update usage log with the actual page count
            String userId = (String) settings.get("_user_id");
            if (userId != null) {
                usageService.updatePageCount(jobId, pages.size());
            }

            jobService.markCompleted(jobId, outputPdf.toString());

        } catch (Exception e) {
            jobService.markFailed(jobId, e.getMessage());
            e.printStackTrace();
        }
    }
}
