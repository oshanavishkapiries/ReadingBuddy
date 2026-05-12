package com.readingbuddy.controller;

import com.readingbuddy.security.SecurityUtils;
import com.readingbuddy.service.SharedDocumentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

// Python equivalent: GET /explore in main.py
// Public route — no authentication required (optional_user in Python)
@Controller
public class ExploreController {

    @Autowired private SharedDocumentService sharedDocumentService;
    @Autowired private SecurityUtils securityUtils;

    @GetMapping("/explore")
    public String explore(Model model) {
        model.addAttribute("sharedDocs", sharedDocumentService.listPublic(50));
        // currentUser is null for unauthenticated visitors — same as optional_user in Python
        return "explore";
    }
}
