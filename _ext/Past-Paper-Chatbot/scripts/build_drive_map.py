"""
One-off script: recursively walk the Google Drive folder tree (given by
GOOGLE_DRIVE_FOLDER_ID / GOOGLE_DRIVE_ALEVEL_FOLDER_ID in .env) and build a
mapping from each PDF's relative path (matching data/raw layout, e.g.
"Accounting/Accounting Past_Papers/2022/May_June/Paper_1/Variant_1/....pdf")
to its Google Drive file ID.

Output: data/drive_file_map.json  ->  {"<relative_path>": "<drive_file_id>", ...}
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

MAX_WORKERS = 20

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
REFRESH_TOKEN = os.environ["GOOGLE_REFRESH_TOKEN"]
ROOT_FOLDER_ID = os.environ["GOOGLE_DRIVE_FOLDER_ID"]
OUTPUT_PATH = PROJECT_DIR / "data" / "drive_file_map.json"

FOLDER_MIME = "application/vnd.google-apps.folder"


_thread_local = threading.local()


def get_drive_service():
    # httplib2 (used under the hood) isn't thread-safe, so each worker
    # thread needs its own service instance.
    if not hasattr(_thread_local, "service"):
        creds = Credentials(
            token=None,
            refresh_token=REFRESH_TOKEN,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            token_uri="https://oauth2.googleapis.com/token",
        )
        _thread_local.service = build("drive", "v3", credentials=creds, cache_discovery=False)
    return _thread_local.service


def list_children(folder_id):
    service = get_drive_service()
    items = []
    page_token = None
    while True:
        resp = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType)",
                pageSize=1000,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def fetch_folder(entry):
    folder_id, prefix = entry
    return prefix, list_children(folder_id)


def main():
    mapping = {}
    frontier = [(ROOT_FOLDER_ID, "")]
    level = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        while frontier:
            level += 1
            print(f"Level {level}: expanding {len(frontier)} folder(s) concurrently...")
            next_frontier = []
            for prefix, children in pool.map(fetch_folder, frontier):
                for item in children:
                    rel_path = f"{prefix}/{item['name']}" if prefix else item["name"]
                    if item["mimeType"] == FOLDER_MIME:
                        next_frontier.append((item["id"], rel_path))
                    elif item["name"].lower().endswith(".pdf"):
                        mapping[rel_path] = item["id"]
            print(f"  -> {len(mapping)} PDFs mapped, {len(next_frontier)} subfolder(s) to expand next")
            frontier = next_frontier

    OUTPUT_PATH.write_text(json.dumps(mapping, indent=0), encoding="utf-8")
    print(f"Done. Mapped {len(mapping)} PDFs.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
