package com.readingbuddy.config;

import com.readingbuddy.security.JwtCookieFilter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

// Python equivalent: auth.py (bcrypt + JWT) + FastAPI Depends(require_user)
//
// Spring Security replaces manual cookie checks in every route handler.
// The JwtCookieFilter reads the rb_session cookie and populates SecurityContextHolder,
// so every controller can call securityUtils.getCurrentUser() without caring about tokens.
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Autowired
    private JwtCookieFilter jwtCookieFilter;

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            // Stateless — no HttpSession. Auth state lives in the rb_session JWT cookie.
            .sessionManagement(s -> s.sessionCreationPolicy(SessionCreationPolicy.STATELESS))

            // CSRF disabled: the app uses cookie-based JWT which is already origin-bound.
            // Re-enable for production CSRF protection if needed.
            .csrf(csrf -> csrf.disable())

            .authorizeHttpRequests(auth -> auth
                .requestMatchers(
                    "/login", "/register",
                    "/explore", "/shared/**",
                    "/static/**", "/css/**", "/js/**",
                    "/health", "/h2-console/**"
                ).permitAll()
                .anyRequest().authenticated()
            )

            // JWT filter runs before Spring's own auth filter on every request
            .addFilterBefore(jwtCookieFilter, UsernamePasswordAuthenticationFilter.class)

            // Unauthenticated → redirect to login (matches Python's require_user raising 401)
            .exceptionHandling(ex -> ex
                .authenticationEntryPoint((req, res, e) -> res.sendRedirect("/login"))
                .accessDeniedHandler((req, res, e) -> res.sendRedirect("/login"))
            )

            // Allow H2 console iframes
            .headers(h -> h.frameOptions(fo -> fo.sameOrigin()));

        return http.build();
    }

    // BCrypt replaces Python's bcrypt.hashpw / checkpw
    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    // Prevent Spring Boot from auto-registering JwtCookieFilter as a plain servlet filter.
    // Without this it would run twice: once via the security chain, once outside it.
    @Bean
    public FilterRegistrationBean<JwtCookieFilter> jwtFilterRegistration(JwtCookieFilter filter) {
        FilterRegistrationBean<JwtCookieFilter> registration = new FilterRegistrationBean<>(filter);
        registration.setEnabled(false);
        return registration;
    }
}
