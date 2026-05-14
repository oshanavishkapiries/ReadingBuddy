package com.readingbuddy.repository;

import com.readingbuddy.entity.UsageLog;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface UsageLogRepository extends JpaRepository<UsageLog, String> {

    List<UsageLog> findByUserIdAndDate(String userId, String date);

    long countByUserIdAndDate(String userId, String date);

    Optional<UsageLog> findByJobId(String jobId);
}
