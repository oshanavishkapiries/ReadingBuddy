package com.readingbuddy.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

// Python equivalent: DriveManager in gdrive.py
// This is a stub showing the interface. A full implementation would use
// the google-api-services-drive Java library (same service account / base64 credentials flow).
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

    // Returns the Drive folder ID for a user subfolder (uploads / outputs)
    public String getUserFolder(String userId, String subfolder) {
        // In a real implementation:
        //   Drive service = buildDriveService(credentialsB64);
        //   return getOrCreateFolder(userId, usersRootId);
        throw new UnsupportedOperationException("Google Drive not configured");
    }

    public String uploadBytes(byte[] data, String filename, String parentId) {
        throw new UnsupportedOperationException("Google Drive not configured");
    }

    public byte[] downloadToBytes(String fileId) {
        throw new UnsupportedOperationException("Google Drive not configured");
    }

    public void deleteFile(String fileId) {
        // silently ignore if Drive is not configured
    }

    public void makePublic(String fileId) {
        throw new UnsupportedOperationException("Google Drive not configured");
    }

    public String getDirectLink(String fileId) {
        return "https://drive.google.com/uc?id=" + fileId + "&export=download";
    }
}
