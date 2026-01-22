#!/usr/bin/env python3
"""
ExpliCAS Web Server (Colab/PM2-ready)

- Serves static files from web/
- Proxies /api/eval to cas_cli eval --format json
- Maintains in-memory session state (vars + #n refs)

Colab tips:
- Bind 0.0.0.0
- Use a tunnel (colabcode/ngrok) to access from browser
- Start with pm2:
    pm2 start python3 --name explicas -- web/server.py -- --port 8000
"""

import http.server
import json
import os
import re
import subprocess
import sys
import argparse
from pathlib import Path

# ----------------------------
# Config (env + defaults)
# ----------------------------

DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"

def env_int(*names, default):
    for n in names:
        v = os.environ.get(n)
        if v:
            try:
                return int(v)
            except ValueError:
                pass
    return default

def env_str(*names, default=None):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default

CAS_TIMEOUT = env_int("CAS_TIMEOUT", default=60)  # seconds

# Session state - persists for lifetime of server
session_variables = {}  # name -> result string
session_results = []    # list of all results for #n references


def log(msg):
    print(msg, flush=True)


def project_root_from_script(script_path: Path) -> Path:
    # If script is in web/server.py, project root is parent of web/
    # More robust than os.chdir assumptions under pm2.
    script_dir = script_path.parent.resolve()
    # If directory name is "web", use its parent. Otherwise assume repo root == script_dir.parent.
    if script_dir.name == "web":
        return script_dir.parent
    return script_dir.parent


class CASHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        # directory passed by HTTPServer factory
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format, *args):
        # quieter but still useful
        log("[http] " + (format % args))

    def do_POST(self):
        if self.path == "/api/eval":
            self.handle_eval()
        elif self.path == "/api/clear":
            self.handle_clear()
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        # CORS preflight
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def handle_clear(self):
        global session_variables, session_results
        session_variables = {}
        session_results = []
        self.send_json({"ok": True, "message": "Session cleared"})

    def handle_eval(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8", errors="replace")

        try:
            data = json.loads(body)
            expression = (data.get("expression") or "").strip()
            if not expression:
                self.send_json_error("No expression provided")
                return

            # assignment: name := expr
            assignment_match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*:=\s*(.+)$", expression)
            if assignment_match:
                var_name = assignment_match.group(1)
                expr_part = assignment_match.group(2)
                result = self.eval_with_substitution(expr_part)
                if result.get("ok", False):
                    session_variables[var_name] = result.get("result", "")
                    result["assignment"] = var_name
                    result["input"] = expression
            else:
                result = self.eval_with_substitution(expression)

            session_results.append(result)
            result["ref"] = len(session_results)
            result["variables"] = list(session_variables.keys())
            self.send_json(result)

        except json.JSONDecodeError:
            self.send_json_error("Invalid JSON")
        except Exception as e:
            self.send_json_error(f"Server error: {e}")

    def eval_with_substitution(self, expression: str):
        expr = expression

        # Substitute refs: %n or #n
        def replace_ref(m):
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(session_results):
                ref_result = session_results[idx].get("result", "")
                return f"({ref_result})"
            return m.group(0)

        expr = re.sub(r"[%#](\d+)", replace_ref, expr)

        # Substitute variables carefully (word boundary)
        for var_name, var_value in session_variables.items():
            pattern = r"\b" + re.escape(var_name) + r"\b"
            expr = re.sub(pattern, f"({var_value})", expr)

        return self.call_cas_cli(expr)

    def call_cas_cli(self, expression: str):
        try:
            # Use absolute path if available
            cas_cli = self.server.cas_cli_path
            cmd = [
                str(cas_cli),
                "eval",
                "--format", "json",
                "--max-chars", "500000",
                "--steps", "on",
            ]

            result = subprocess.run(
                cmd,
                input=expression,
                capture_output=True,
                text=True,
                timeout=self.server.cas_timeout,
                cwd=str(self.server.project_root),
            )

            if result.returncode != 0:
                return {
                    "ok": False,
                    "error": (result.stderr.strip() or "Evaluation failed"),
                    "input": expression,
                }

            out = result.stdout.strip()
            if not out:
                return {"ok": False, "error": "Empty response", "input": expression}

            try:
                return json.loads(out)
            except json.JSONDecodeError as e:
                return {
                    "ok": False,
                    "error": f"Invalid JSON from cas_cli: {e}",
                    "input": expression,
                    "raw": out[:800],
                }

        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"Timeout ({self.server.cas_timeout}s)", "input": expression}
        except FileNotFoundError:
            return {
                "ok": False,
                "error": f"cas_cli not found at '{self.server.cas_cli_path}'. Build it or set CAS_CLI env.",
                "input": expression,
            }
        except Exception as e:
            return {"ok": False, "error": f"cas_cli call error: {e}", "input": expression}

    def send_json(self, data):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_json_error(self, message: str):
        payload = json.dumps({"ok": False, "error": message}).encode("utf-8")
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class ExpliCASSrv(http.server.ThreadingHTTPServer):
    # attach config to server instance
    def __init__(self, server_address, RequestHandlerClass, *, directory, project_root, cas_cli_path, cas_timeout):
        self.directory = directory
        self.project_root = project_root
        self.cas_cli_path = cas_cli_path
        self.cas_timeout = cas_timeout
        super().__init__(server_address, lambda *a, **kw: RequestHandlerClass(*a, directory=str(directory), **kw))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=env_int("PORT", "EXPLICAS_PORT", default=DEFAULT_PORT))
    parser.add_argument("--host", type=str, default=env_str("HOST", "EXPLICAS_HOST", default=DEFAULT_HOST))
    parser.add_argument("--cas-cli", type=str, default=env_str("CAS_CLI", default=None))
    parser.add_argument("--timeout", type=int, default=CAS_TIMEOUT)
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = project_root_from_script(script_path)

    web_dir = (project_root / "web").resolve()
    if not web_dir.exists():
        log(f"⚠️  Warning: web directory not found at {web_dir}")
        log("   Static serving may fail.")

    # cas_cli path
    if args.cas_cli:
        cas_cli_path = Path(args.cas_cli).expanduser().resolve()
    else:
        cas_cli_path = (project_root / "target" / "release" / "cas_cli").resolve()

    if not cas_cli_path.exists():
        log(f"⚠️  Warning: cas_cli not found at {cas_cli_path}")
        log("   Build it with: cargo build --release")
        log("   Or set CAS_CLI=/abs/path/to/cas_cli")

    server = ExpliCASSrv(
        (args.host, args.port),
        CASHandler,
        directory=web_dir,
        project_root=project_root,
        cas_cli_path=cas_cli_path,
        cas_timeout=args.timeout,
    )

    log("")
    log("╔════════════════════════════════════════════════╗")
    log("║     🧮 ExpliCAS Web REPL Server (PM2/Colab)    ║")
    log("╠════════════════════════════════════════════════╣")
    log(f"║  Host: {args.host:<39}║")
    log(f"║  Port: {args.port:<39}║")
    log(f"║  Web:  / (static from {str(web_dir)[:25]:<25}...)║")
    log(f"║  API:  /api/eval  |  /api/clear                ║")
    log("║                                                ║")
    log(f"║  cas_cli: {str(cas_cli_path)[:34]:<34}║")
    log(f"║  timeout: {args.timeout:<34}║")
    log("╚════════════════════════════════════════════════╝")
    log("")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("\n👋 Server stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
