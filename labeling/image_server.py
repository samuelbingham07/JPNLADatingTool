"""Simple file server with CORS headers for Label Studio image serving."""
import http.server
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent / "data" / "raw"
PORT = 8081


class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET")
        super().end_headers()

    def log_message(self, format, *args):
        pass  # suppress request logs


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    with http.server.HTTPServer(("", port), CORSHandler) as httpd:
        print(f"Image server running at http://localhost:{port}")
        httpd.serve_forever()
