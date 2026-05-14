package com.readingbuddy.controller;

import com.readingbuddy.service.DriveService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

// Python equivalent: GET /health in main.py
@RestController
public class HealthController {

    @Autowired private DriveService driveService;

    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of(
                "status", "ok",
                "storage", driveService.isEnabled() ? "google_drive" : "local"
        );
    }
}
