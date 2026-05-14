package com.readingbuddy.service;

import com.readingbuddy.entity.User;
import com.readingbuddy.repository.UserRepository;
import com.readingbuddy.security.JwtUtil;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
public class AuthService {

    @Autowired private UserRepository userRepository;
    @Autowired private PasswordEncoder passwordEncoder;
    @Autowired private JwtUtil jwtUtil;

    public User register(String username, String email, String password) {
        User user = User.builder()
                .username(username)
                .email(email)
                .hashedPassword(passwordEncoder.encode(password))
                .build();
        return userRepository.save(user);
    }

    public Optional<User> authenticate(String username, String password) {
        return userRepository.findByUsername(username)
                .filter(u -> passwordEncoder.matches(password, u.getHashedPassword()))
                .filter(User::isActive);
    }

    public void setAuthCookie(HttpServletResponse response, String userId) {
        String token = jwtUtil.generateToken(userId);
        Cookie cookie = new Cookie("rb_session", token);
        cookie.setHttpOnly(true);
        cookie.setPath("/");
        cookie.setMaxAge(60 * 60 * 24 * 7);
        response.addCookie(cookie);
    }

    public void clearAuthCookie(HttpServletResponse response) {
        Cookie cookie = new Cookie("rb_session", "");
        cookie.setMaxAge(0);
        cookie.setPath("/");
        response.addCookie(cookie);
    }
}
