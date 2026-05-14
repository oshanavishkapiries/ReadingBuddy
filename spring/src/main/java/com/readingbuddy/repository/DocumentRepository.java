package com.readingbuddy.repository;

import com.readingbuddy.entity.Document;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

// Python equivalent: list_documents / get_document / delete_document in models.py
public interface DocumentRepository extends JpaRepository<Document, String> {

    // SELECT * FROM documents WHERE user_id = ? ORDER BY created_at DESC LIMIT ?
    List<Document> findByUserIdOrderByCreatedAtDesc(String userId, Pageable pageable);

    // Used to enforce ownership — only returns a doc if it belongs to the given user
    Optional<Document> findByIdAndUserId(String id, String userId);
}
