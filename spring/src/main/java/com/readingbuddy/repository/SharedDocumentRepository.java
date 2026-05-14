package com.readingbuddy.repository;

import com.readingbuddy.entity.SharedDocument;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface SharedDocumentRepository extends JpaRepository<SharedDocument, String> {

    Optional<SharedDocument> findByJobId(String jobId);

    List<SharedDocument> findAllByOrderByCreatedAtDesc(Pageable pageable);
}
