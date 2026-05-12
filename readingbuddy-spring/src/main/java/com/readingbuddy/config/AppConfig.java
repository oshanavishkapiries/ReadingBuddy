package com.readingbuddy.config;

import com.readingbuddy.security.SecurityUtils;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Lazy;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.web.servlet.ModelAndView;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;
import org.springframework.web.servlet.HandlerInterceptor;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;

@Configuration
public class AppConfig implements WebMvcConfigurer {

    // @Lazy prevents a circular-dependency issue: AppConfig (WebMvcConfigurer) is
    // initialised early in the context lifecycle, before JPA repositories are ready.
    // Deferring SecurityUtils to first-use avoids that ordering problem.
    @Autowired
    @Lazy
    private SecurityUtils securityUtils;

    // ThreadPoolTaskExecutor powers @Async methods in JobService.
    // Python equivalent: every start_job() spawns a new daemon thread.
    // Spring consolidates all async work into one managed pool.
    @Bean(name = "taskExecutor")
    public ThreadPoolTaskExecutor taskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(4);
        executor.setMaxPoolSize(10);
        executor.setQueueCapacity(50);
        executor.setThreadNamePrefix("pipeline-");
        executor.initialize();
        return executor;
    }

    // This interceptor adds two things to every Thymeleaf model:
    //   currentUser  — the logged-in User entity (like Jinja2's template context in Python)
    //   notification — parsed rb_notify cookie, cleared after first render
    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(new HandlerInterceptor() {

            @Override
            public void postHandle(HttpServletRequest request, HttpServletResponse response,
                                   Object handler, ModelAndView mav) {
                if (mav == null || mav.getViewName() == null
                        || mav.getViewName().startsWith("redirect:")) {
                    return;
                }

                mav.addObject("currentUser", securityUtils.getCurrentUser());

                Cookie[] cookies = request.getCookies();
                if (cookies == null) return;

                for (Cookie cookie : cookies) {
                    if ("rb_notify".equals(cookie.getName())) {
                        String[] parts = cookie.getValue().split("\\|", 3);
                        if (parts.length == 3) {
                            mav.addObject("notifyType", parts[0]);
                            mav.addObject("notifyMessage", parts[1]);
                            mav.addObject("notifyDuration", parts[2]);
                        }
                        Cookie expire = new Cookie("rb_notify", "");
                        expire.setMaxAge(0);
                        expire.setPath("/");
                        response.addCookie(expire);
                        break;
                    }
                }
            }
        });
    }
}
