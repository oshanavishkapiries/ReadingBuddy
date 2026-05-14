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

@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Autowired
    private JwtCookieFilter jwtCookieFilter;

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .sessionManagement(s -> s.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
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

            .addFilterBefore(jwtCookieFilter, UsernamePasswordAuthenticationFilter.class)
            .exceptionHandling(ex -> ex
                .authenticationEntryPoint((req, res, e) -> res.sendRedirect("/login"))
                .accessDeniedHandler((req, res, e) -> res.sendRedirect("/login"))
            )

            .headers(h -> h.frameOptions(fo -> fo.sameOrigin()));

        return http.build();
    }

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
