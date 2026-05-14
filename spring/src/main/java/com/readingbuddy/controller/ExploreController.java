package com.readingbuddy.controller;

import com.readingbuddy.security.SecurityUtils;
import com.readingbuddy.service.SharedDocumentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class ExploreController {

    @Autowired private SharedDocumentService sharedDocumentService;
    @Autowired private SecurityUtils securityUtils;

    @GetMapping("/explore")
    public String explore(Model model) {
        model.addAttribute("sharedDocs", sharedDocumentService.listPublic(50));
        return "explore";
    }
}
