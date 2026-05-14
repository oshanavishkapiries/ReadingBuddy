package com.readingbuddy.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Entity
@Table(name = "users")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@EqualsAndHashCode(of = "id")
@ToString(exclude = {"jobs", "documents"})
public class User {

    @Id
    private String id;

    @Column(unique = true, nullable = false)
    private String username;

    @Column(unique = true, nullable = false)
    private String email;

    @Column(nullable = false)
    private String hashedPassword;

    @Builder.Default
    private String openrouterApiKey = "";

    @Builder.Default
    private String openrouterModel = "openai/gpt-4o-mini";

    @Builder.Default
    private int extractionDpi = 300;

    @Builder.Default
    private String extractionOcrMode = "auto";

    @Builder.Default
    private String extractionLang = "eng";

    @Builder.Default
    private double translationTemperature = 0.2;

    @Builder.Default
    private String pdfPageSize = "A4";

    @Builder.Default
    private String pdfMargin = "18mm";

    @Builder.Default
    private double pdfFontSize = 16.5;

    @Builder.Default
    private boolean active = true;

    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    @OneToMany(mappedBy = "user", cascade = CascadeType.ALL, orphanRemoval = true)
    @Builder.Default
    private List<Job> jobs = new ArrayList<>();

    @OneToMany(mappedBy = "user", cascade = CascadeType.ALL, orphanRemoval = true)
    @Builder.Default
    private List<Document> documents = new ArrayList<>();

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
