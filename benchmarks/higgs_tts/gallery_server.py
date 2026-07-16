#!/usr/bin/env python3
"""Serve an audio gallery with HTTP byte-range support."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """Simple static handler that supports one RFC 7233 byte range."""

    range_start: int | None = None
    range_end: int | None = None

    def send_head(self) -> BinaryIO | None:
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            self.range_start = self.range_end = None
            return super().send_head()

        try:
            handle = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        try:
            size = os.fstat(handle.fileno()).st_size
            parsed = self._parse_range(self.headers.get("Range"), size)
            if parsed is False:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                handle.close()
                return None

            content_type = self.guess_type(path)
            self.send_response(
                HTTPStatus.PARTIAL_CONTENT if parsed is not None else HTTPStatus.OK
            )
            self.send_header("Content-type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            if parsed is None:
                start, end = 0, max(0, size - 1)
                self.range_start = self.range_end = None
                content_length = size
            else:
                start, end = parsed
                self.range_start, self.range_end = start, end
                content_length = end - start + 1
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(content_length))
            self.send_header(
                "Last-Modified",
                self.date_time_string(os.fstat(handle.fileno()).st_mtime),
            )
            self.end_headers()
            return handle
        except Exception:
            handle.close()
            raise

    @staticmethod
    def _parse_range(value: str | None, size: int) -> tuple[int, int] | None | bool:
        if value is None:
            return None
        match = _RANGE_RE.fullmatch(value.strip())
        if match is None or size == 0:
            return False
        start_text, end_text = match.groups()
        if not start_text:
            if not end_text or int(end_text) <= 0:
                return False
            length = min(int(end_text), size)
            return size - length, size - 1
        start = int(start_text)
        if start >= size:
            return False
        end = min(int(end_text), size - 1) if end_text else size - 1
        if end < start:
            return False
        return start, end

    def copyfile(self, source: BinaryIO, outputfile: BinaryIO) -> None:
        if self.range_start is None or self.range_end is None:
            shutil.copyfileobj(source, outputfile)
            return
        source.seek(self.range_start)
        remaining = self.range_end - self.range_start + 1
        while remaining:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=22222)
    parser.add_argument("--directory", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handler = partial(RangeRequestHandler, directory=str(args.directory))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"Serving {args.directory.resolve()} on http://{args.host}:{args.port} "
        "with byte-range support",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
