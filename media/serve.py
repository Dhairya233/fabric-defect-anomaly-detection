"""A throwaway local web server for the repository, run inside the build process."""

from __future__ import annotations

import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep build output readable
        pass


def start(root: Path) -> tuple[ThreadingHTTPServer, str]:
    handler = functools.partial(_Quiet, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"
