package com.readingbuddy.controller;

import com.readingbuddy.entity.SharedDocument;
import com.readingbuddy.service.SharedDocumentService;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.nio.file.Path;
import java.util.Optional;

@Controller
public class SharedController {

    @Autowired private SharedDocumentService sharedDocumentService;

    @GetMapping("/shared/{sharedId}")
    public String viewShared(@PathVariable String sharedId, Model model) {
        return sharedDocumentService.findById(sharedId).map(shared -> {
            model.addAttribute("shared", shared);
            model.addAttribute("job", shared.getJob());
            return "shared-view";
        }).orElse("redirect:/explore");
    }

    @GetMapping("/shared/{sharedId}/download")
    public ResponseEntity<FileSystemResource> downloadShared(@PathVariable String sharedId) {
        return sharedDocumentService.findById(sharedId).flatMap(shared -> {
            if (shared.getDirectLink() != null && !shared.getDirectLink().isBlank()) {
                return Optional.empty();
            }
            var job = shared.getJob();
            if (job != null && job.getOutputPdf() != null) {
                var file = Path.of(job.getOutputPdf()).toFile();
                if (file.exists()) {
                    return Optional.of(ResponseEntity.ok()
                            .header(HttpHeaders.CONTENT_DISPOSITION,
                                    "attachment; filename=\"readingbuddy_" + shared.getPublicName() + ".pdf\"")
                            .contentType(MediaType.APPLICATION_PDF)
                            .body(new FileSystemResource(file)));
                }
            }
            return Optional.empty();
        }).orElseGet(() -> ResponseEntity.notFound().build());
    }

    @PostMapping("/shared/{sharedId}/like")
    public String like(@PathVariable String sharedId, HttpServletRequest request) {
        sharedDocumentService.like(sharedId);
        String referer = request.getHeader("Referer");
        return "redirect:" + (referer != null ? referer : "/explore");
    }
}
