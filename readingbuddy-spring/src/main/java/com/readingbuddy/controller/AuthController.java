package com.readingbuddy.controller;

import com.readingbuddy.entity.User;
import com.readingbuddy.repository.UserRepository;
import com.readingbuddy.security.SecurityUtils;
import com.readingbuddy.service.AuthService;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

// Python equivalent: login / register / logout routes in main.py
@Controller
public class AuthController extends BaseController {

    @Autowired private AuthService authService;
    @Autowired private UserRepository userRepository;
    @Autowired private SecurityUtils securityUtils;

    @GetMapping("/")
    public String index() {
        return securityUtils.getCurrentUser() != null ? "redirect:/dashboard" : "redirect:/login";
    }

    @GetMapping("/login")
    public String loginPage(@RequestParam(required = false) String error, Model model) {
        if (securityUtils.getCurrentUser() != null) return "redirect:/dashboard";
        model.addAttribute("error", error);
        return "login";   // → templates/login.html
    }

    @PostMapping("/login")
    public String login(@RequestParam String username,
                        @RequestParam String password,
                        HttpServletResponse response) {
        return authService.authenticate(username, password)
                .map(user -> {
                    authService.setAuthCookie(response, user.getId());
                    return "redirect:/dashboard";
                })
                .orElse("redirect:/login?error=invalid");
    }

    @GetMapping("/register")
    public String registerPage(@RequestParam(required = false) String error, Model model) {
        if (securityUtils.getCurrentUser() != null) return "redirect:/dashboard";
        model.addAttribute("error", error);
        return "register";   // → templates/register.html
    }

    @PostMapping("/register")
    public String register(@RequestParam String username,
                           @RequestParam String email,
                           @RequestParam String password,
                           @RequestParam String confirmPassword,
                           HttpServletResponse response) {

        if (!password.equals(confirmPassword))  return "redirect:/register?error=password_mismatch";
        if (password.length() < 6)              return "redirect:/register?error=password_short";
        if (userRepository.findByUsername(username).isPresent()) return "redirect:/register?error=username_taken";
        if (userRepository.findByEmail(email).isPresent())       return "redirect:/register?error=email_taken";

        User newUser = authService.register(username, email, password);
        authService.setAuthCookie(response, newUser.getId());
        notify(response, "success", "Welcome to ReadingBuddy, " + username + "!", 5000);
        return "redirect:/dashboard";
    }

    @PostMapping("/logout")
    public String logout(HttpServletResponse response) {
        authService.clearAuthCookie(response);
        return "redirect:/login";
    }
}
