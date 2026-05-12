package com.readingbuddy.repository;

import com.readingbuddy.entity.Job;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

// Python equivalent: list_jobs / get_job in models.py
public interface JobRepository extends JpaRepository<Job, String> {

    List<Job> findByUserIdOrderByCreatedAtDesc(String userId, Pageable pageable);

    Optional<Job> findByIdAndUserId(String id, String userId);
}
