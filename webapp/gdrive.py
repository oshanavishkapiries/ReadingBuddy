import base64
import io
import json
import os
from pathlib import Path
from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]

FOLDER_STRUCTURE = {
    "users": {},
    "shared": {},
    "workspace": {},
}


class DriveManager:
    def __init__(self, root_folder_id: str, credentials_b64: str = ""):
        self.root_folder_id = root_folder_id
        self._service = None
        self._folder_cache = {}
        self._initialized = False

        if credentials_b64:
            creds_json = base64.b64decode(credentials_b64)
            creds_dict = json.loads(creds_json)
            creds = service_account.Credentials.from_service_account_info(
                creds_dict, scopes=SCOPES
            )
            self._service = build("drive", "v3", credentials=creds)
        else:
            raise ValueError("Google Drive credentials (base64) are required")

    @property
    def service(self):
        return self._service

    def initialize(self) -> dict:
        if self._initialized:
            return self._folder_cache

        self._folder_cache["root"] = self.root_folder_id

        for folder_name in FOLDER_STRUCTURE:
            folder_id = self._get_or_create_folder(folder_name, self.root_folder_id)
            self._folder_cache[folder_name] = folder_id

        self._initialized = True
        return self._folder_cache

    def _get_or_create_folder(self, name: str, parent_id: str) -> str:
        results = self._service.files().list(
            q=f"name='{name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false",
            fields="files(id, name)",
        ).execute()
        files = results.get("files", [])
        if files:
            return files[0]["id"]

        file_metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        folder = self._service.files().create(body=file_metadata, fields="id").execute()
        return folder["id"]

    def get_user_folder(self, user_id: str, subfolder: str = "uploads") -> str:
        users_folder = self._folder_cache["users"]
        user_folder_id = self._get_or_create_folder(user_id, users_folder)
        return self._get_or_create_folder(subfolder, user_folder_id)

    def get_workspace_folder(self, job_id: str) -> str:
        workspace_folder = self._folder_cache["workspace"]
        return self._get_or_create_folder(job_id, workspace_folder)

    def get_shared_folder(self, shared_id: str) -> str:
        shared_root = self._folder_cache["shared"]
        return self._get_or_create_folder(shared_id, shared_root)

    def upload_file(self, file_path: str, parent_id: str, mime_type: Optional[str] = None) -> dict:
        path = Path(file_path)
        file_metadata = {"name": path.name, "parents": [parent_id]}
        media = MediaFileUpload(str(path), mimetype=mime_type or "application/octet-stream")
        file = self._service.files().create(
            body=file_metadata, media_body=media, fields="id, name, webViewLink, webContentLink"
        ).execute()
        return file

    def upload_bytes(self, data: bytes, filename: str, parent_id: str, mime_type: Optional[str] = None) -> dict:
        file_metadata = {"name": filename, "parents": [parent_id]}
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type or "application/octet-stream")
        file = self._service.files().create(
            body=file_metadata, media_body=media, fields="id, name, webViewLink, webContentLink"
        ).execute()
        return file

    def download_file(self, file_id: str, dest_path: str):
        request = self._service.files().get_media(fileId=file_id)
        with open(dest_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

    def download_to_bytes(self, file_id: str) -> bytes:
        request = self._service.files().get_media(fileId=file_id)
        output = io.BytesIO()
        downloader = MediaIoBaseDownload(output, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return output.getvalue()

    def make_public(self, file_id: str):
        self._service.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
        ).execute()

    def get_direct_link(self, file_id: str) -> str:
        return f"https://drive.google.com/uc?id={file_id}&export=download"

    def get_view_link(self, file_id: str) -> str:
        return f"https://drive.google.com/file/d/{file_id}/view"

    def delete_file(self, file_id: str):
        try:
            self._service.files().delete(fileId=file_id).execute()
        except Exception:
            pass

    def list_files(self, parent_id: str) -> list[dict]:
        results = self._service.files().list(
            q=f"'{parent_id}' in parents and trashed=false",
            fields="files(id, name, mimeType, size, createdTime, webViewLink, webContentLink)",
        ).execute()
        return results.get("files", [])


def get_drive_manager() -> Optional[DriveManager]:
    root_folder_id = os.environ.get("GOOGLE_DRIVE_ROOT_FOLDER_ID", "")
    credentials_b64 = os.environ.get("GOOGLE_DRIVE_CREDENTIALS_B64", "")
    if not root_folder_id:
        return None
    return DriveManager(root_folder_id, credentials_b64)
