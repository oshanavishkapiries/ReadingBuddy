package com.readingbuddy.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

// Python equivalent: class Document(Base) in models.py
@Entity
@Table(name = "documents")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@EqualsAndHashCode(of = "id")
@ToString(exclude = {"user", "jobs"})
public class Document {

    @Id
    private String id;

    // @ManyToOne + @JoinColumn replaces SQLAlchemy's ForeignKey("users.id") + relationship("User")
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(nullable = false)
    private String filename;

    @Column(nullable = false)
    private String originalName;

    @Builder.Default
    private long fileSize = 0;

    @Builder.Default
    private int pageCount = 0;

    @Builder.Default
    private String driveFileId = "";

    private LocalDateTime createdAt;

    @OneToMany(mappedBy = "document")
    @Builder.Default
    private List<Job> jobs = new ArrayList<>();

    @PrePersist
    void prePersist() {
        if (id == null) id = UUID.randomUUID().toString();
        if (createdAt == null) createdAt = LocalDateTime.now();
    }
}
