package com.readingbuddy;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableAsync;

@SpringBootApplication
@EnableAsync
public class ReadingBuddyApplication {

    public static void main(String[] args) {
        SpringApplication.run(ReadingBuddyApplication.class, args);
    }
}
