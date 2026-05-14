package com.readingbuddy.controller;

import com.readingbuddy.entity.Job;
import com.readingbuddy.entity.JobStatus;
import com.readingbuddy.security.SecurityUtils;
import com.readingbuddy.service.DocumentService;
import com.readingbuddy.service.JobService;
import com.readingbuddy.service.UsageService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

import java.util.List;

@Controller
public class DashboardController extends BaseController {

    @Autowired private SecurityUtils securityUtils;
    @Autowired private JobService jobService;
    @Autowired private DocumentService documentService;
    @Autowired private UsageService usageService;

    @Value("${app.openrouter-api-key:}")
    private String backendApiKey;

    @GetMapping("/dashboard")
    public String dashboard(Model model) {
        var user = securityUtils.requireCurrentUser();

        List<Job> jobs = jobService.listJobs(user.getId(), 20);
        List<Job> activeJobs = jobs.stream()
                .filter(j -> j.getStatus() == JobStatus.PENDING
                          || j.getStatus() == JobStatus.PROCESSING)
                .toList();

        long usageCount = user.getOpenrouterApiKey() == null || user.getOpenrouterApiKey().isBlank()
                ? usageService.getTodayCount(user.getId()) : 0;

        model.addAttribute("jobs", activeJobs.isEmpty() ? jobs.subList(0, Math.min(5, jobs.size())) : activeJobs);
        model.addAttribute("allJobs", jobs);
        model.addAttribute("documents", documentService.listDocuments(user.getId(), 20));
        model.addAttribute("hasBackendKey", !backendApiKey.isBlank());
        model.addAttribute("usageCount", usageCount);

        return "dashboard";
    }
}
