package com.readingbuddy.security;

import com.readingbuddy.entity.User;
import com.readingbuddy.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.AnonymousAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;

// Convenience bean — replaces FastAPI's Depends(optional_user) / Depends(require_user) pattern.
// Controllers call securityUtils.getCurrentUser() instead of declaring a dependency parameter.
@Component
public class SecurityUtils {

    @Autowired
    private UserRepository userRepository;

    // Returns null if the request is unauthenticated — equivalent to optional_user
    public User getCurrentUser() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || !auth.isAuthenticated() || auth instanceof AnonymousAuthenticationToken) {
            return null;
        }
        return userRepository.findById(auth.getName()).orElse(null);
    }

    // Throws if unauthenticated — equivalent to require_user
    public User requireCurrentUser() {
        User user = getCurrentUser();
        if (user == null) throw new IllegalStateException("Not authenticated");
        return user;
    }
}
