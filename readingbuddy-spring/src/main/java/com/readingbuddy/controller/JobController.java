package com.readingbuddy.controller;

import com.readingbuddy.dto.JobStatusDto;
import com.readingbuddy.entity.Job;
import com.readingbuddy.entity.JobStatus;
import com.readingbuddy.entity.User;
import com.readingbuddy.security.SecurityUtils;
import com.readingbuddy.service.JobService;
import com.readingbuddy.service.SharedDocumentService;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;

// Python equivalent: /job/* and /jobs routes in main.py
@Controller
public class JobController extends BaseController {

    @Autowired private SecurityUtils securityUtils;
    @Autowired private JobService jobService;
    @Autowired private SharedDocumentService sharedDocumentService;

    @Value("${app.workspace}")
    private String workspace;

    @GetMapping("/jobs")
    public String jobsPage(Model model) {
        User user = securityUtils.requireCurrentUser();
        model.addAttribute("jobs", jobService.listJobs(user.getId(), 50));
        return "jobs";
    }

    @GetMapping("/job/{jobId}")
    public String jobDetail(@PathVariable String jobId, Model model) {
        User user = securityUtils.requireCurrentUser();
        return jobService.getJob(jobId, user.getId()).map(job -> {
            model.addAttribute("job", job);
            model.addAttribute("shared", sharedDocumentService.findByJobId(jobId).orElse(null));
            return "job-detail";
        }).orElse("redirect:/jobs");
    }

    // JSON polling endpoint — called every few seconds by the job-detail page JS
    // Python equivalent: GET /job/{job_id}/status returning a dict
    @GetMapping("/job/{jobId}/status")
    @ResponseBody
    public ResponseEntity<JobStatusDto> jobStatus(@PathVariable String jobId) {
        User user = securityUtils.requireCurrentUser();
        return jobService.getJob(jobId, user.getId()).map(job -> {
            Map<String, Object> live = jobService.getLiveStatus(jobId);
            return ResponseEntity.ok(JobStatusDto.builder()
                    .id(job.getId())
                    .filename(job.getFilename())
                    .status(job.getStatus().name().toLowerCase())
                    .progress(live != null ? (Double) live.get("progress") : job.getProgress())
                    .currentStep(live != null ? (String) live.get("currentStep") : job.getCurrentStep())
                    .stepDetail(live != null ? (String) live.get("stepDetail") : job.getStepDetail())
                    .error(job.getError())
                    .outputPdf(job.getOutputPdf())
                    .pageCount(job.getPageCount())
                    .build());
        }).orElseGet(() -> ResponseEntity.notFound().build());
    }

    // Streams the translated PDF back to the browser
    @GetMapping("/job/{jobId}/download")
    public ResponseEntity<FileSystemResource> downloadPdf(@PathVariable String jobId) {
        User user = securityUtils.requireCurrentUser();
        return jobService.getJob(jobId, user.getId()).flatMap(job -> {
            if (job.getOutputPdf() == null || job.getOutputPdf().isBlank()) return Optional.empty();
            var file = Path.of(job.getOutputPdf()).toFile();
            if (!file.exists()) return Optional.empty();
            return Optional.of(ResponseEntity.ok()
                    .header(HttpHeaders.CONTENT_DISPOSITION,
                            "attachment; filename=\"readingbuddy_" + jobId + ".pdf\"")
                    .contentType(MediaType.APPLICATION_PDF)
                    .body(new FileSystemResource(file)));
        }).orElseGet(() -> ResponseEntity.notFound().build());
    }

    // Streams the combined markdown file
    @GetMapping("/job/{jobId}/download/markdown")
    public ResponseEntity<FileSystemResource> downloadMarkdown(@PathVariable String jobId) {
        User user = securityUtils.requireCurrentUser();
        return jobService.getJob(jobId, user.getId()).flatMap(job -> {
            var file = Path.of(workspace, jobId, "markdown", "combined.md").toFile();
            if (!file.exists()) return Optional.empty();
            return Optional.of(ResponseEntity.ok()
                    .header(HttpHeaders.CONTENT_DISPOSITION,
                            "attachment; filename=\"readingbuddy_" + jobId + "_markdown.md\"")
                    .contentType(MediaType.TEXT_PLAIN)
                    .body(new FileSystemResource(file)));
        }).orElseGet(() -> ResponseEntity.notFound().build());
    }

    @PostMapping("/job/{jobId}/cancel")
    public String cancelJob(@PathVariable String jobId) {
        User user = securityUtils.requireCurrentUser();
        jobService.getJob(jobId, user.getId()).ifPresent(job -> jobService.cancel(job.getId()));
        return "redirect:/job/" + jobId;
    }

    @PostMapping("/job/{jobId}/share")
    public String shareJob(@PathVariable String jobId,
                           @RequestParam(defaultValue = "") String publicName,
                           HttpServletResponse response) {
        User user = securityUtils.requireCurrentUser();
        jobService.getJob(jobId, user.getId()).ifPresent(job -> {
            if (job.getStatus() == JobStatus.COMPLETED
                    && sharedDocumentService.findByJobId(jobId).isEmpty()) {
                sharedDocumentService.share(job, user, publicName);
            }
        });
        return "redirect:/job/" + jobId;
    }

    @PostMapping("/job/{jobId}/unshare")
    public String unshareJob(@PathVariable String jobId) {
        User user = securityUtils.requireCurrentUser();
        sharedDocumentService.findByJobId(jobId).ifPresent(shared -> {
            if (shared.getUser().getId().equals(user.getId())) {
                sharedDocumentService.unshare(shared);
            }
        });
        return "redirect:/job/" + jobId;
    }
}
