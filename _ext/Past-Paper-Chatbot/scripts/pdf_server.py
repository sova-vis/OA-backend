"""
Local HTTP file server for the past-paper PDFs.

Results-table links need to point at an http:// URL rather than a file://
one: modern browsers block a page served from http://localhost from
navigating to file:// links (a security restriction), even when everything
is on the same machine. Serving the PDFs over plain HTTP on localhost avoids
that restriction entirely.
"""

import functools
import http.server
import socketserver
import threading
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PDF_SERVER_PORT = 8765

_lock = threading.Lock()
_started = False


def start_pdf_server():
    """Idempotent: safe to call every Streamlit rerun. If the port is
    already bound (e.g. by a previous instance of this same server), that's
    fine - it means the PDFs are already being served."""
    global _started
    with _lock:
        if _started:
            return
        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=str(RAW_DIR)
        )
        try:
            httpd = socketserver.TCPServer(("localhost", PDF_SERVER_PORT), handler)
        except OSError:
            _started = True  # already running (this or a prior instance)
            return
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        _started = True
