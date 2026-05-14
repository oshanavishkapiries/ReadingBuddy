package com.readingbuddy.security;

import com.readingbuddy.entity.User;
import com.readingbuddy.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.AnonymousAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;

@Component
public class SecurityUtils {

    @Autowired
    private UserRepository userRepository;

    public User getCurrentUser() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || !auth.isAuthenticated() || auth instanceof AnonymousAuthenticationToken) {
            return null;
        }
        return userRepository.findById(auth.getName()).orElse(null);
    }

    public User requireCurrentUser() {
        User user = getCurrentUser();
        if (user == null) throw new IllegalStateException("Not authenticated");
        return user;
    }
}
