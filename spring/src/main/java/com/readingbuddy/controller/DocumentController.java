package com.readingbuddy.controller;

import com.readingbuddy.entity.Document;
import com.readingbuddy.entity.Job;
import com.readingbuddy.entity.User;
import com.readingbuddy.security.SecurityUtils;
import com.readingbuddy.service.DocumentService;
import com.readingbuddy.service.JobService;
import com.readingbuddy.service.UsageService;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

@Controller
public class DocumentController extends BaseController {

    @Autowired private SecurityUtils securityUtils;
    @Autowired private DocumentService documentService;
    @Autowired private JobService jobService;
    @Autowired private UsageService usageService;

    @Value("${app.openrouter-api-key:}")
    private String backendApiKey;

    @GetMapping("/documents")
    public String documentsPage(Model model) {
        User user = securityUtils.requireCurrentUser();
        long usageCount = isUsingBackendKey(user) ? usageService.getTodayCount(user.getId()) : 0;
        model.addAttribute("documents", documentService.listDocuments(user.getId(), 50));
        model.addAttribute("hasBackendKey", !backendApiKey.isBlank());
        model.addAttribute("usageCount", usageCount);
        return "documents";
    }

    @PostMapping("/documents/upload")
    public String uploadDocument(@RequestParam("file") MultipartFile file,
                                 HttpServletResponse response) throws Exception {
        User user = securityUtils.requireCurrentUser();
        documentService.uploadDocument(file, user);
        return redirectWithNotify(response, "/documents", "success", "Document uploaded successfully");
    }

    @PostMapping("/documents/{docId}/delete")
    public String deleteDocument(@PathVariable String docId, HttpServletResponse response) {
        User user = securityUtils.requireCurrentUser();
        documentService.deleteDocument(docId, user);
        return redirectWithNotify(response, "/documents", "success", "Document deleted");
    }

    @PostMapping("/documents/{docId}/translate")
    public String translateDocument(@PathVariable String docId,
                                    @RequestParam(defaultValue = "300") int dpi,
                                    @RequestParam(defaultValue = "auto") String ocrMode,
                                    @RequestParam(defaultValue = "eng") String lang,
                                    @RequestParam(defaultValue = "openai/gpt-4o-mini") String model,
                                    @RequestParam(defaultValue = "0.2") double temperature,
                                    @RequestParam(defaultValue = "A4") String pageSize,
                                    @RequestParam(defaultValue = "18mm") String margin,
                                    @RequestParam(defaultValue = "16.5") double fontSize,
                                    HttpServletResponse response) throws Exception {

        User user = securityUtils.requireCurrentUser();

        String apiKey = resolveApiKey(user);
        if (apiKey == null) {
            return redirectWithNotify(response, "/settings", "error",
                    "No API key configured. Add one in Settings.");
        }
        if (isUsingBackendKey(user) && usageService.isLimitReached(user.getId())) {
            return redirectWithNotify(response, "/documents", "warning",
                    "Daily limit reached (3/3). Add your own API key for unlimited access.");
        }

        Document doc = documentService.getDocument(docId, user.getId())
                .orElse(null);
        if (doc == null) return "redirect:/documents";

        Path localPath = documentService.resolveLocalPath(doc, user);
        Map<String, Object> settings = buildSettings(dpi, ocrMode, lang, model, temperature,
                pageSize, margin, fontSize, apiKey, user.getId());

        Job job = jobService.createJob(user, doc, doc.getOriginalName(), settings);
        if (isUsingBackendKey(user)) usageService.logUsage(user, job);
        jobService.startPipeline(job.getId(), localPath, settings);

        return "redirect:/job/" + job.getId();
    }

    @PostMapping("/upload")
    public String uploadAndTranslate(@RequestParam("file") MultipartFile file,
                                     @RequestParam(defaultValue = "300") int dpi,
                                     @RequestParam(defaultValue = "auto") String ocrMode,
                                     @RequestParam(defaultValue = "eng") String lang,
                                     @RequestParam(defaultValue = "openai/gpt-4o-mini") String model,
                                     @RequestParam(defaultValue = "0.2") double temperature,
                                     @RequestParam(defaultValue = "A4") String pageSize,
                                     @RequestParam(defaultValue = "18mm") String margin,
                                     @RequestParam(defaultValue = "16.5") double fontSize,
                                     HttpServletResponse response) throws Exception {

        User user = securityUtils.requireCurrentUser();

        String apiKey = resolveApiKey(user);
        if (apiKey == null) {
            return redirectWithNotify(response, "/settings", "error",
                    "No API key configured. Add one in Settings.");
        }
        if (isUsingBackendKey(user) && usageService.isLimitReached(user.getId())) {
            return redirectWithNotify(response, "/dashboard", "warning",
                    "Daily limit reached (3/3). Add your own API key for unlimited access.");
        }

        Document doc = documentService.uploadDocument(file, user);
        Path localPath = documentService.resolveLocalPath(doc, user);
        Map<String, Object> settings = buildSettings(dpi, ocrMode, lang, model, temperature,
                pageSize, margin, fontSize, apiKey, user.getId());

        Job job = jobService.createJob(user, doc, file.getOriginalFilename(), settings);
        if (isUsingBackendKey(user)) usageService.logUsage(user, job);
        jobService.startPipeline(job.getId(), localPath, settings);

        return "redirect:/job/" + job.getId();
    }

    private String resolveApiKey(User user) {
        if (user.getOpenrouterApiKey() != null && !user.getOpenrouterApiKey().isBlank()) {
            return user.getOpenrouterApiKey();
        }
        return backendApiKey.isBlank() ? null : backendApiKey;
    }

    private boolean isUsingBackendKey(User user) {
        return (user.getOpenrouterApiKey() == null || user.getOpenrouterApiKey().isBlank())
                && !backendApiKey.isBlank();
    }

    private Map<String, Object> buildSettings(int dpi, String ocrMode, String lang,
                                               String model, double temperature,
                                               String pageSize, String margin, double fontSize,
                                               String apiKey, String userId) {
        Map<String, Object> settings = new HashMap<>();
        settings.put("extraction",     Map.of("dpi", dpi, "ocr_mode", ocrMode, "lang", lang));
        settings.put("translation",    Map.of("model", model, "temperature", temperature, "api_key", apiKey));
        settings.put("pdf_generation", Map.of("page_size", pageSize, "margin", margin, "font_size", fontSize));
        settings.put("_user_id",       userId);
        return settings;
    }
}
