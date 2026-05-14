package com.readingbuddy.service;

import com.readingbuddy.entity.Document;
import com.readingbuddy.entity.User;
import com.readingbuddy.repository.DocumentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

// Python equivalent: upload logic in main.py (/documents/upload and /upload routes)
@Service
public class DocumentService {

    @Autowired private DocumentRepository documentRepository;
    @Autowired private DriveService driveService;

    @Value("${app.workspace}")
    private String workspace;

    public List<Document> listDocuments(String userId, int limit) {
        return documentRepository.findByUserIdOrderByCreatedAtDesc(userId, PageRequest.of(0, limit));
    }

    public Optional<Document> getDocument(String docId, String userId) {
        return documentRepository.findByIdAndUserId(docId, userId);
    }

    // Saves the uploaded file either to Google Drive or the local workspace.
    // Returns the saved Document entity.
    public Document uploadDocument(MultipartFile file, User user) throws IOException {
        byte[] content = file.getBytes();
        String ext = getExtension(file.getOriginalFilename());
        String savedName = UUID.randomUUID().toString().replace("-", "") + ext;

        if (driveService.isEnabled()) {
            String parentId = driveService.getUserFolder(user.getId(), "uploads");
            String driveId = driveService.uploadBytes(content, savedName, parentId);
            return documentRepository.save(Document.builder()
                    .user(user)
                    .filename(savedName)
                    .originalName(file.getOriginalFilename())
                    .fileSize(content.length)
                    .driveFileId(driveId)
                    .build());
        }

        Path uploadDir = Path.of(workspace, "users", user.getId(), "documents");
        Files.createDirectories(uploadDir);
        Files.write(uploadDir.resolve(savedName), content);

        return documentRepository.save(Document.builder()
                .user(user)
                .filename(savedName)
                .originalName(file.getOriginalFilename())
                .fileSize(content.length)
                .build());
    }

    public void deleteDocument(String docId, User user) {
        documentRepository.findByIdAndUserId(docId, user.getId()).ifPresent(doc -> {
            if (driveService.isEnabled() && doc.getDriveFileId() != null && !doc.getDriveFileId().isEmpty()) {
                driveService.deleteFile(doc.getDriveFileId());
            }
            documentRepository.delete(doc);
        });
    }

    // Resolves the local filesystem path for a document, downloading from Drive if needed.
    public Path resolveLocalPath(Document doc, User user) throws IOException {
        if (driveService.isEnabled() && doc.getDriveFileId() != null && !doc.getDriveFileId().isEmpty()) {
            Path tmp = Path.of(workspace, "temp");
            Files.createDirectories(tmp);
            Path dest = tmp.resolve(doc.getFilename());
            byte[] bytes = driveService.downloadToBytes(doc.getDriveFileId());
            Files.write(dest, bytes);
            return dest;
        }
        return Path.of(workspace, "users", user.getId(), "documents", doc.getFilename());
    }

    private String getExtension(String filename) {
        if (filename == null) return ".pdf";
        int dot = filename.lastIndexOf('.');
        return dot >= 0 ? filename.substring(dot) : ".pdf";
    }
}
