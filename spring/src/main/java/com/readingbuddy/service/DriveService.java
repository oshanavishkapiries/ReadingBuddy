package com.readingbuddy.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Service
public class DriveService {

    @Value("${google.drive.root-folder-id:}")
    private String rootFolderId;

    @Value("${google.drive.credentials-b64:}")
    private String credentialsB64;

    public boolean isEnabled() {
        return rootFolderId != null && !rootFolderId.isBlank()
                && credentialsB64 != null && !credentialsB64.isBlank();
    }

    public String getUserFolder(String userId, String subfolder) {
        throw new UnsupportedOperationException("Google Drive not configured");
    }

    public String uploadBytes(byte[] data, String filename, String parentId) {
        throw new UnsupportedOperationException("Google Drive not configured");
    }

    public byte[] downloadToBytes(String fileId) {
        throw new UnsupportedOperationException("Google Drive not configured");
    }

    public void deleteFile(String fileId) {
    }

    public void makePublic(String fileId) {
        throw new UnsupportedOperationException("Google Drive not configured");
    }

    public String getDirectLink(String fileId) {
        return "https://drive.google.com/uc?id=" + fileId + "&export=download";
    }
}
