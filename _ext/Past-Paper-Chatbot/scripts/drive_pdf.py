"""
Renders one exact page of a Drive-hosted PDF as an image, so the app can
show the precise page a question is on without depending on any PDF
viewer's "#page=N" jump-to-page behavior - confirmed unreliable both for
Drive's own /view viewer and for Google Docs Viewer when opened as a direct
link (both stayed on page 1 regardless of the fragment, verified by testing
actual scroll position). Downloading the file and rendering the target page
ourselves sidesteps that entirely: there's no "jump" to fail, the requested
page is just what gets drawn.

Caching (so a paper is only downloaded from Drive once, not on every view)
is the caller's responsibility - see app.py's @st.cache_data wrappers around
these functions.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import fitz  # PyMuPDF

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

_service = None


def get_drive_service():
    global _service
    if _service is None:
        creds = Credentials(
            token=None,
            refresh_token=os.environ["GOOGLE_REFRESH_TOKEN"],
            client_id=os.environ["GOOGLE_CLIENT_ID"],
            client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
            token_uri="https://oauth2.googleapis.com/token",
        )
        _service = build("drive", "v3", credentials=creds, cache_discovery=False)
    return _service


def download_pdf_bytes(file_id: str) -> bytes:
    service = get_drive_service()
    return service.files().get_media(fileId=file_id).execute()


def render_page_image(pdf_bytes: bytes, page_number: int, zoom: float = 2.0) -> bytes:
    """page_number is 1-indexed (matches how pages are referenced everywhere
    else in this project). Returns PNG bytes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        index = max(0, min(page_number - 1, doc.page_count - 1))
        page = doc[index]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png")
    finally:
        doc.close()
