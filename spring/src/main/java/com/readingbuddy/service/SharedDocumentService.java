package com.readingbuddy.service;

import com.readingbuddy.entity.Job;
import com.readingbuddy.entity.SharedDocument;
import com.readingbuddy.entity.User;
import com.readingbuddy.repository.SharedDocumentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class SharedDocumentService {

    @Autowired private SharedDocumentRepository sharedDocumentRepository;
    @Autowired private DriveService driveService;

    public Optional<SharedDocument> findById(String id) {
        return sharedDocumentRepository.findById(id);
    }

    public Optional<SharedDocument> findByJobId(String jobId) {
        return sharedDocumentRepository.findByJobId(jobId);
    }

    public List<SharedDocument> listPublic(int limit) {
        return sharedDocumentRepository.findAllByOrderByCreatedAtDesc(PageRequest.of(0, limit));
    }

    public SharedDocument share(Job job, User user, String publicName) {
        SharedDocument shared = SharedDocument.builder()
                .job(job)
                .user(user)
                .publicName(publicName.isBlank() ? job.getFilename() : publicName)
                .build();
        return sharedDocumentRepository.save(shared);
    }

    public void unshare(SharedDocument shared) {
        if (driveService.isEnabled()
                && shared.getDriveFileId() != null
                && !shared.getDriveFileId().isEmpty()) {
            driveService.deleteFile(shared.getDriveFileId());
        }
        sharedDocumentRepository.delete(shared);
    }

    public SharedDocument like(String sharedId) {
        SharedDocument shared = sharedDocumentRepository.findById(sharedId)
                .orElseThrow(() -> new IllegalArgumentException("Shared document not found"));
        shared.setLikes(shared.getLikes() + 1);
        return sharedDocumentRepository.save(shared);
    }
}
