package com.readingbuddy.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;
import java.util.UUID;

// Python equivalent: class SharedDocument(Base) in models.py
@Entity
@Table(name = "shared_documents")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@EqualsAndHashCode(of = "id")
@ToString(exclude = {"job", "user"})
public class SharedDocument {

    @Id
    private String id;

    // unique = true mirrors SQLAlchemy's unique=True on job_id
    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "job_id", unique = true, nullable = false)
    private Job job;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(nullable = false)
    private String publicName;

    @Builder.Default
    private int likes = 0;

    private LocalDateTime createdAt;

    @PrePersist
    void prePersist() {
        if (id == null) id = UUID.randomUUID().toString();
        if (createdAt == null) createdAt = LocalDateTime.now();
    }
}
