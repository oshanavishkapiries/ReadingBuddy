package com.readingbuddy.controller;

import com.readingbuddy.entity.User;
import com.readingbuddy.repository.UserRepository;
import com.readingbuddy.security.SecurityUtils;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class SettingsController extends BaseController {

    @Autowired private SecurityUtils securityUtils;
    @Autowired private UserRepository userRepository;

    @GetMapping("/settings")
    public String settingsPage(Model model) {
        model.addAttribute("user", securityUtils.requireCurrentUser());
        return "settings";
    }

    @PostMapping("/settings")
    public String updateSettings(@RequestParam(defaultValue = "") String openrouterApiKey,
                                 @RequestParam(defaultValue = "google/gemini-2.5-flash") String openrouterModel,
                                 @RequestParam(defaultValue = "300") int extractionDpi,
                                 @RequestParam(defaultValue = "auto") String extractionOcrMode,
                                 @RequestParam(defaultValue = "eng") String extractionLang,
                                 @RequestParam(defaultValue = "sinhala") String translationLanguage,
                                 @RequestParam(defaultValue = "0.2") double translationTemperature,
                                 @RequestParam(defaultValue = "A4") String pdfPageSize,
                                 @RequestParam(defaultValue = "18mm") String pdfMargin,
                                 @RequestParam(defaultValue = "10.0") double pdfFontSize,
                                 HttpServletResponse response) {

        User user = securityUtils.requireCurrentUser();
        user.setOpenrouterApiKey(openrouterApiKey);
        user.setOpenrouterModel(openrouterModel);
        user.setExtractionDpi(extractionDpi);
        user.setExtractionOcrMode(extractionOcrMode);
        user.setExtractionLang(extractionLang);
        user.setTranslationLanguage(translationLanguage);
        user.setTranslationTemperature(translationTemperature);
        user.setPdfPageSize(pdfPageSize);
        user.setPdfMargin(pdfMargin);
        user.setPdfFontSize(pdfFontSize);
        userRepository.save(user);

        return redirectWithNotify(response, "/settings", "success", "Settings saved successfully");
    }
}
