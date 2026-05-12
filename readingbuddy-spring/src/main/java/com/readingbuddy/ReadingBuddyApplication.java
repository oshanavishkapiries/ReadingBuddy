package com.readingbuddy;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableAsync;

// @EnableAsync activates Spring's @Async processing — equivalent to threading.Thread in the Python version.
// Background pipeline jobs will be dispatched through a ThreadPoolTaskExecutor defined in AppConfig.
@SpringBootApplication
@EnableAsync
public class ReadingBuddyApplication {

    public static void main(String[] args) {
        SpringApplication.run(ReadingBuddyApplication.class, args);
    }
}
