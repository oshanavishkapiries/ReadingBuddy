package com.readingbuddy.repository;

import com.readingbuddy.entity.SharedDocument;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

// Python equivalent: get_shared_document / get_shared_by_job / list_shared_documents in models.py
public interface SharedDocumentRepository extends JpaRepository<SharedDocument, String> {

    Optional<SharedDocument> findByJobId(String jobId);

    // Public explore page — most recent first
    List<SharedDocument> findAllByOrderByCreatedAtDesc(Pageable pageable);
}
