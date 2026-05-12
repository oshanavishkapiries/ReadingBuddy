package com.readingbuddy.controller;

import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletResponse;

// Provides the notify() helper used by every controller to set the rb_notify cookie.
// Python equivalent: response.set_cookie(key="rb_notify", value=...) scattered across main.py
// The AppConfig interceptor reads this cookie and adds it to the Thymeleaf model.
public abstract class BaseController {

    protected void notify(HttpServletResponse response, String type, String message, int durationMs) {
        Cookie cookie = new Cookie("rb_notify", type + "|" + message + "|" + durationMs);
        cookie.setMaxAge(5);
        cookie.setPath("/");
        response.addCookie(cookie);
    }

    protected String redirectWithNotify(HttpServletResponse response, String url,
                                        String type, String message) {
        notify(response, type, message, 3000);
        return "redirect:" + url;
    }
}
