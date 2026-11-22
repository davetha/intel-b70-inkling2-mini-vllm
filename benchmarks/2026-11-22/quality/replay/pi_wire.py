"""Expose one local vLLM endpoint over a Unix socket and record actual provider usage."""

import json
import os
import socketserver
import threading
import time
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit


class SocketServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def server_bind(self):
        super().server_bind()
        os.chmod(self.server_address, 0o666)


class Wire(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        if self.path.split("?")[0] != "/v1/chat/completions":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("model") != self.server.model:
            self.send_error(400, "Unexpected model")
            return
        body["cache_salt"] = self.server.salt
        n = len(body["messages"])
        base = self.server.upstream
        conn = HTTPConnection(base.hostname, base.port, timeout=900)
        stamp = time.monotonic()
        row = {
            "messages": n,
            "request_started": time.time(),
            "settings": {
                k: body.get(k)
                for k in (
                    "model",
                    "reasoning_effort",
                    "thinking_token_budget",
                    "max_tokens",
                    "max_completion_tokens",
                    "temperature",
                    "top_p",
                    "top_k",
                    "seed",
                    "chat_template_kwargs",
                )
                if k in body
            },
            "tools": len(body.get("tools") or []),
            "stream": bool(body.get("stream")),
        }
        with self.server.lock:
            self.server.requests += 1
            row["request"] = self.server.requests
        before = (
            self.server.metrics_fn(self.server.url)
            if row["request"] == 1 and hasattr(self.server, "metrics_fn")
            else None
        )
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                json.dumps(body).encode(),
                {"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.send_response(resp.status)
            self.send_header(
                "Content-Type", resp.getheader("Content-Type", "application/json")
            )
            self.send_header("Connection", "close")
            self.end_headers()
            row["status"] = resp.status
            first = None
            bytes_pending = b""
            while block := resp.read1(65536):
                self.wfile.write(block)
                if resp.status != 200 or not row["stream"]:
                    bytes_pending += block
                    continue
                bytes_pending += block
                *lines, bytes_pending = bytes_pending.split(b"\n")
                for line in lines:
                    if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                        continue
                    d = json.loads(line[6:])
                    if d.get("usage"):
                        row["usage"] = d["usage"]
                    for c in d.get("choices") or []:
                        if first is None and any(
                            c.get("delta", {}).get(k)
                            for k in (
                                "content",
                                "reasoning",
                                "reasoning_content",
                                "tool_calls",
                            )
                        ):
                            first = time.monotonic() - stamp
            if not row["stream"] and resp.status == 200 and bytes_pending:
                row["usage"] = json.loads(bytes_pending).get("usage")
            if resp.status != 200:
                row["error"] = bytes_pending[:1400].decode(errors="replace")
            row["first_seconds"] = first
        except (OSError, ValueError) as exc:
            row["transport_error"] = str(exc)
        finally:
            conn.close()
            row["seconds"] = time.monotonic() - stamp
            if before:
                try:
                    after = self.server.metrics_fn(self.server.url)
                    row["engine_delta"] = {k: after[k] - before[k] for k in before}
                except OSError:
                    pass
            with self.server.lock, self.server.log.open("a") as out:
                print(json.dumps(row), file=out, flush=True)
            self.close_connection = True


def start(path, url, model, salt, log):
    server = SocketServer(str(path), Wire)
    server.url = url
    server.upstream = urlsplit(url)
    server.model = model
    server.salt = salt
    server.log = Path(log)
    server.lock = threading.Lock()
    server.requests = 0
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
