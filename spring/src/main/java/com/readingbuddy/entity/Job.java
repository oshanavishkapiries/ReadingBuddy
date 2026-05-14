package com.readingbuddy.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

@Entity
@Table(name = "jobs")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@EqualsAndHashCode(of = "id")
@ToString(exclude = {"user", "document"})
public class Job {

    @Id
    private String id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "document_id")
    private Document document;

    @Column(nullable = false)
    private String filename;

    @Enumerated(EnumType.STRING)
    @Builder.Default
    private JobStatus status = JobStatus.PENDING;

    @Builder.Default
    private double progress = 0.0;

    @Builder.Default
    private String currentStep = "Queued";

    @Column(columnDefinition = "TEXT")
    @Builder.Default
    private String stepDetail = "";

    @Column(columnDefinition = "TEXT")
    @Builder.Default
    private String error = "";

    @Column(columnDefinition = "TEXT")
    @Convert(converter = JsonMapConverter.class)
    @Builder.Default
    private Map<String, Object> settings = new HashMap<>();

    @Builder.Default
    private String outputPdf = "";

    @Builder.Default
    private int pageCount = 0;

    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    @PrePersist
    void prePersist() {
        if (id == null) id = UUID.randomUUID().toString();
        if (createdAt == null) createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }

    @PreUpdate
    void preUpdate() {
        updatedAt = LocalDateTime.now();
    }
}
