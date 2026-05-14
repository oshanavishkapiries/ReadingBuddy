package com.readingbuddy.repository;

import com.readingbuddy.entity.Document;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface DocumentRepository extends JpaRepository<Document, String> {

    List<Document> findByUserIdOrderByCreatedAtDesc(String userId, Pageable pageable);
    Optional<Document> findByIdAndUserId(String id, String userId);
}
