package com.readingbuddy.service;

import com.readingbuddy.entity.Job;
import com.readingbuddy.entity.UsageLog;
import com.readingbuddy.entity.User;
import com.readingbuddy.repository.UsageLogRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.util.List;

// Python equivalent: get_today_usage / log_usage in models.py
@Service
public class UsageService {

    @Autowired private UsageLogRepository usageLogRepository;

    @Value("${app.daily-limit}")
    private int dailyLimit;

    public long getTodayCount(String userId) {
        return usageLogRepository.countByUserIdAndDate(userId, today());
    }

    public int getTodayPageCount(String userId) {
        return usageLogRepository.findByUserIdAndDate(userId, today())
                .stream().mapToInt(UsageLog::getPageCount).sum();
    }

    public boolean isLimitReached(String userId) {
        return getTodayCount(userId) >= dailyLimit;
    }

    public void logUsage(User user, Job job) {
        UsageLog log = UsageLog.builder()
                .user(user)
                .job(job)
                .date(today())
                .pageCount(0)
                .build();
        usageLogRepository.save(log);
    }

    public void updatePageCount(String jobId, int pageCount) {
        usageLogRepository.findByJobId(jobId).ifPresent(log -> {
            log.setPageCount(pageCount);
            usageLogRepository.save(log);
        });
    }

    private String today() {
        return LocalDate.now().toString();   // YYYY-MM-DD, same format as Python
    }
}
