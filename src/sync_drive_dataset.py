"""
Sincronizează imaginile din Google Drive în folderul local `dataset/`.
- Folosește OAuth2 (credentials.json/token.json).
- Găsește folderul de start după cale (ex: proiectlicenta/dataset) și descarcă recursiv doar imagini.
"""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
from typing import Dict, List

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm

# Permisiune minimă: citire din Drive
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_PATH = Path("token.json")
CREDENTIALS_PATH = Path("credentials.json")

IMAGE_EXTS = {"jpg", "jpeg", "png", "bmp", "webp"}


def get_creds() -> Credentials:
    """Încarcă/creează credențialele OAuth. Dacă token-ul e expirat, îl reîmprospătează."""
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_PATH.exists():
                raise FileNotFoundError("credentials.json lipsă la rădăcina proiectului")
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            # Deschide automat browserul pentru autorizare pe localhost
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())
    return creds


def find_folder_id(service, root_folder_path: str) -> str:
    """Parcurge ierarhia de foldere din Drive după cale și returnează id-ul folderului final."""
    parts = [p for p in root_folder_path.strip('/').split('/') if p]
    parent = 'root'
    for name in parts:
        q = f"name = '{name}' and mimeType = 'application/vnd.google-apps.folder' and '{parent}' in parents and trashed=false"
        res = service.files().list(q=q, spaces='drive', fields='files(id, name)', pageSize=10).execute()
        files = res.get('files', [])
        if not files:
            raise FileNotFoundError(f"Nu am găsit folderul '{name}' în '{parent}'")
        parent = files[0]['id']
    return parent


def list_children(service, folder_id: str) -> List[Dict]:
    """Listează toate fișierele/subfolderele directe, cu paginare, pentru un folder Drive."""
    items: List[Dict] = []
    page_token = None
    while True:
        res = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType)',
            pageToken=page_token,
            pageSize=1000,
        ).execute()
        items.extend(res.get('files', []))
        page_token = res.get('nextPageToken')
        if not page_token:
            break
    return items


def is_image(name: str) -> bool:
    """Verifică extensia pentru a accepta doar imagini."""
    ext = name.lower().rsplit('.', 1)[-1] if '.' in name else ''
    return ext in IMAGE_EXTS


def sync_folder(service, folder_id: str, local_dir: Path):
    """Descarcă imaginile din folderul Drive dat și apoi procesează recursiv subfolderele."""
    local_dir.mkdir(parents=True, exist_ok=True)
    items = list_children(service, folder_id)

    # Împarte între foldere și fișiere
    folders = [i for i in items if i["mimeType"] == 'application/vnd.google-apps.folder']
    files = [i for i in items if i["mimeType"] != 'application/vnd.google-apps.folder']

    # Descarcă doar imaginile (skip dacă fișierul există local)
    for f in tqdm(files, desc=f"Fișiere în {local_dir.name}"):
        name = f['name']
        if not is_image(name):
            continue
        out_path = local_dir / name
        if out_path.exists():
            continue
        req = service.files().get_media(fileId=f['id'])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        out_path.write_bytes(fh.getvalue())

    # Recursiv pentru subfoldere
    for folder in folders:
        sub_local = local_dir / folder['name']
        sync_folder(service, folder['id'], sub_local)


def main():
    """Parse arguments and sync a Drive folder locally."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--drive-folder', required=True, help="Calea în Drive (ex: proiectlicenta/dataset)")
    ap.add_argument('--local', default='dataset', help='Folder local pentru sincronizare')
    args = ap.parse_args()

    creds = get_creds()
    service = build('drive', 'v3', credentials=creds)

    folder_id = find_folder_id(service, args.drive_folder)
    print(f"Găsit folderul Drive '{args.drive_folder}' (id={folder_id})")

    local_dir = Path(args.local)
    sync_folder(service, folder_id, local_dir)
    print(f"Sincronizare completa in '{local_dir}'")


if __name__ == '__main__':
    main()
