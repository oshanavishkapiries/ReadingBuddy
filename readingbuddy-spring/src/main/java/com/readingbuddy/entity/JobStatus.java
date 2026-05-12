package com.readingbuddy.entity;

// Stored as VARCHAR in the database via @Enumerated(EnumType.STRING).
// Python equivalent: the status column string values in models.py.
public enum JobStatus {
    PENDING,
    PROCESSING,
    COMPLETED,
    FAILED,
    CANCELLED
}
