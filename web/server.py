#!/usr/bin/env python3
"""
ExpliCAS Web Server

Simple HTTP server that:
1. Serves static files from web/
2. Proxies /api/eval requests to cas_cli eval-json
3. Maintains session state for variable assignments

Usage:
    cd /path/to/math
    python web/server.py
    
Then open http://localhost:8080
"""

import http.server
import json
import os
import re
import subprocess
import sys
import uuid
from urllib.parse import parse_qs
import hashlib
import tempfile
from contextlib import contextmanager, nullcontext


PORT = int(os.environ.get('PORT', 8080))
CAS_CLI = "./target/release/cas_cli"

# CLI session snapshot configuration (for fast #N references without textual substitution)
SESSION_SNAPSHOT_DIR = os.path.join(tempfile.gettempdir(), "explicas_cli_sessions")
os.makedirs(SESSION_SNAPSHOT_DIR, exist_ok=True)

def _session_snapshot_path(session_id: str) -> str:
    """Stable, filesystem-safe path for a given browser session."""
    h = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
    return os.path.join(SESSION_SNAPSHOT_DIR, f"session_{h}.json")

def _delete_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except Exception as e:
        print(f"⚠️  Warning: failed to delete {path}: {e}")

@contextmanager
def _file_lock(lock_path: str):
    """Best-effort cross-process lock (POSIX)."""
    try:
        import fcntl  # type: ignore
    except Exception:
        yield
        return
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


# Session configuration
SESSION_TIMEOUT_SECONDS = 2 * 60 * 60  # 2 hours of inactivity before session expires

# Multi-session support: each browser tab gets its own session
# Sessions are identified by a UUID stored in the browser's sessionStorage
# Each session tracks: variables, results, and last_access time for cleanup
sessions = {}  # session_id -> dict(variables/results/ref_map/cli_ref/session_file/last_access)

def get_session(session_id):
    """Get or create a session by ID, updating last access time"""
    import time
    current_time = time.time()
    
    # Clean up expired sessions periodically
    cleanup_expired_sessions(current_time)
    
    if session_id not in sessions:
        session_file = _session_snapshot_path(session_id)
        # If the server restarted, drop any stale snapshot to avoid reference mismatches
        _delete_file(session_file)
        _delete_file(session_file + ".lock")
        sessions[session_id] = {
            "variables": {},          # name -> display value (for UI)
            "variable_refs": {},      # name -> cli history id (for fast substitution)
            "results": [],             # list of eval results shown in UI
            "ref_map": [],             # UI ref N -> cli history id (or None if eval failed)
            "cli_ref": 0,              # last cli history id stored in snapshot
            "session_file": session_file,
            "last_access": current_time,
        }
    else:
        # Update last access time
        sessions[session_id]["last_access"] = current_time
    
    return sessions[session_id]

def clear_session(session_id):
    """Clear a specific session (but keep the session entry for new use)"""
    import time
    if session_id in sessions:
        session_file = sessions[session_id].get("session_file") or _session_snapshot_path(session_id)
        _delete_file(session_file)
        _delete_file(session_file + ".lock")
        sessions[session_id] = {
            "variables": {},
            "variable_refs": {},
            "results": [],
            "ref_map": [],
            "cli_ref": 0,
            "session_file": session_file,
            "last_access": time.time(),
        }

def cleanup_expired_sessions(current_time):
    """Remove sessions that haven't been accessed in SESSION_TIMEOUT_SECONDS"""
    expired = [
        sid for sid, data in sessions.items()
        if current_time - data.get("last_access", 0) > SESSION_TIMEOUT_SECONDS
    ]
    for sid in expired:
        session_file = sessions.get(sid, {}).get("session_file")
        if session_file:
            _delete_file(session_file)
            _delete_file(session_file + ".lock")
        del sessions[sid]
    if expired:
        print(f"🧹 Cleaned up {len(expired)} expired session(s). Active sessions: {len(sessions)}")

class CASHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve from web/ directory
        super().__init__(*args, directory="web", **kwargs)
    
    def do_GET(self):
        if self.path == '/api/examples':
            self.handle_get_examples()
        else:
            # Default static file serving
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/api/eval':
            self.handle_eval()
        elif self.path == '/api/clear':
            self.handle_clear()
        elif self.path == '/api/import':
            self.handle_import()
        elif self.path == '/api/delete-variable':
            self.handle_delete_variable()
        else:
            self.send_error(404)
    
    def handle_get_examples(self):
        """Return examples from CSV file"""
        import csv
        examples = []
        csv_path = os.path.join(os.path.dirname(__file__), 'examples.csv')
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    examples.append({
                        "expression": row.get('expression', ''),
                        "description": row.get('description', '')
                    })
            self.send_json({"ok": True, "examples": examples})
        except FileNotFoundError:
            self.send_json({"ok": False, "error": "Examples file not found", "examples": []})
        except Exception as e:
            self.send_json({"ok": False, "error": str(e), "examples": []})
    
    def handle_clear(self):
        """Clear session state for a specific session"""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'
        
        try:
            data = json.loads(body) if body else {}
            session_id = data.get('session_id', 'default')
            clear_session(session_id)
            self.send_json({"ok": True, "message": "Session cleared", "session_id": session_id})
        except json.JSONDecodeError:
            self.send_json({"ok": True, "message": "Session cleared", "session_id": "default"})
    
    def handle_import(self):
        """Import session state (variables and results) from client."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            session_id = data.get('session_id')
            if not session_id:
                self.send_json_error("Missing session_id")
                return

            variables = data.get('variables', {})
            results = data.get('results', [])
            mode = data.get('mode', 'replace')

            session = get_session(session_id)

            # Rebuild results list in order by 'ref'
            max_ref = 0
            results_by_ref = {}
            for r in results:
                ref = r.get('ref', 0)
                if ref and ref > 0:
                    results_by_ref[ref] = r
                    max_ref = max(max_ref, ref)

            imported_results = []
            for i in range(1, max_ref + 1):
                if i in results_by_ref:
                    imported_results.append(results_by_ref[i])
                else:
                    imported_results.append({
                        "ref": i,
                        "input": "<missing>",
                        "result": "",
                        "ok": False,
                        "error": "Missing imported result"
                    })

            if mode == 'replace':
                session["variables"] = dict(variables)
                session["results"] = imported_results
                session["ref_map"] = [None] * len(session["results"])
            else:  # append
                session["variables"].update(variables)
                session["results"].extend(imported_results)
                session.setdefault("ref_map", [])
                session["ref_map"].extend([None] * len(imported_results))

            # Imported data invalidates any existing CLI snapshot mappings
            session["variable_refs"] = {}
            session["cli_ref"] = 0
            session_file = session.get("session_file")
            if session_file:
                _delete_file(session_file)
                _delete_file(session_file + ".lock")

            self.send_json({
                "ok": True,
                "message": f"Imported {len(variables)} variables and {len(imported_results)} results",
                "session_id": session_id,
                "next_ref": len(session["results"]) + 1
            })

        except json.JSONDecodeError:
            self.send_json_error("Invalid JSON")
        except Exception as e:
            self.send_json_error(str(e))
    def handle_delete_variable(self):
        """Delete a specific variable from session"""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body)
            session_id = data.get('session_id', 'default')
            session = get_session(session_id)
            var_name = data.get('variable', '')
            
            if var_name and var_name in session["variables"]:
                del session["variables"][var_name]
                session.get("variable_refs", {}).pop(var_name, None)
                self.send_json({"ok": True, "message": f"Variable '{var_name}' deleted"})
            else:
                self.send_json({"ok": False, "error": f"Variable '{var_name}' not found"})
        except json.JSONDecodeError:
            self.send_json_error("Invalid JSON")
        except Exception as e:
            self.send_json_error(str(e))
    
    def handle_eval(self):
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body)
            expression = data.get('expression', '').strip()
            session_id = data.get('session_id', 'default')
            session = get_session(session_id)
            
            if not expression:
                self.send_json_error("No expression provided")
                return
            
            # Check for assignment: name := expr
            assignment_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:=\s*(.+)$', expression)
            
            if assignment_match:
                var_name = assignment_match.group(1)
                expr_part = assignment_match.group(2)
                result, cli_id = self.eval_and_store(expr_part, session)
                
                if result.get('ok', False):
                    # Store the result for this variable (but NOT in results list)
                    session["variables"][var_name] = result.get('result', '')
                    if cli_id is not None:
                        session["variable_refs"][var_name] = cli_id
                    result['assignment'] = var_name
                    result['input'] = expression
                    # Assignments don't get a ref - they don't consume reference numbers
                    result['ref'] = None
                    
            else:
                result, cli_id = self.eval_and_store(expression, session)
                # Only non-assignment evaluations get stored for UI references
                session["results"].append(result)
                result['ref'] = len(session["results"])
                session["ref_map"].append(cli_id if result.get('ok', False) else None)
            
            # Include current variables and session_id in response
            result['variables'] = list(session["variables"].keys())
            result['session_id'] = session_id
            
            self.send_json(result)
            
        except json.JSONDecodeError:
            self.send_json_error("Invalid JSON")
        except Exception as e:
            self.send_json_error(str(e))
    
    def eval_and_store(self, expression, session):
        """Evaluate via cas_cli using the per-session snapshot, updating cli_ref on success."""
        session_file = session.get("session_file")
        lock_path = (session_file + ".lock") if session_file else None

        with (_file_lock(lock_path) if lock_path else nullcontext()):
            result = self.eval_with_substitution(expression, session)

            cli_id = None
            if result.get("ok", False):
                session["cli_ref"] = session.get("cli_ref", 0) + 1
                cli_id = session["cli_ref"]

            return result, cli_id

    def eval_with_substitution(self, expression, session):
        """Evaluate expression, substituting variables and mapping UI #N refs to CLI session refs."""
        expr = expression
        session_results = session.get("results", [])
        session_variables = session.get("variables", {})
        ref_map = session.get("ref_map", [])
        variable_refs = session.get("variable_refs", {})

        # Map UI refs (%n or #n) -> CLI history id to avoid re-parsing huge expressions.
        # Fallback: if a CLI mapping doesn't exist (e.g., imported notebook), substitute stored result text.
        invalid_refs = []

        def replace_ref(match):
            n = int(match.group(2))
            idx = n - 1
            if idx < 0:
                invalid_refs.append(n)
                return match.group(0)

            # Fast path: mapped CLI id
            if 0 <= idx < len(ref_map) and ref_map[idx] is not None:
                return f"#{ref_map[idx]}"

            # Fallback (legacy): textual substitution from stored results (can be slow for huge results)
            if 0 <= idx < len(session_results):
                ref_result = session_results[idx].get("result", "")
                # Avoid accidental blow-ups if someone references an enormous stored string
                if isinstance(ref_result, str) and len(ref_result) > 200000:
                    invalid_refs.append(n)
                    return match.group(0)
                return f"({ref_result})"

            invalid_refs.append(n)
            return match.group(0)

        expr = re.sub(r"([%#])(\d+)", replace_ref, expr)

        if invalid_refs:
            uniq = sorted(set(invalid_refs))
            return {
                "ok": False,
                "error": "Invalid reference(s): " + ", ".join(f"#{n}" for n in uniq),
                "input": expression[:500],
            }

        # Substitute variables (word boundary to avoid partial replacements)
        for var_name, var_value in session_variables.items():
            pattern = r"\b" + re.escape(var_name) + r"\b"
            if var_name in variable_refs:
                # Substitute with CLI session ref (fast, avoids huge text expansions)
                expr = re.sub(pattern, f"(#{variable_refs[var_name]})", expr)
            else:
                # Legacy fallback: substitute display value
                expr = re.sub(pattern, f"({var_value})", expr)

        return self.call_cas_cli(expr, session.get("session_file"))

    def call_cas_cli(self, expression, session_file=None):
        """Call cas_cli eval-json and return parsed result"""
        result = None  # Initialize to handle exception cases
        try:
            # Log expression length for debugging large inputs
            expr_len = len(expression)
            if expr_len > 1000:
                print(f"⚠️  Large expression: {expr_len} chars")

            # Increased timeout for very large expressions
            timeout = 120 if expr_len > 10000 else 60

            # Build CLI command (session snapshot enables fast #N references across calls)
            cmd = [CAS_CLI, "eval", "--format", "json", "--max-chars", "500000", "--steps", "on"]
            if session_file:
                cmd += ["--session", session_file]

            result = subprocess.run(
                cmd,
                input=expression,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                print(f"❌ cas_cli error (exit {result.returncode}): {result.stderr[:500] if result.stderr else 'no stderr'}")
                return {
                    "ok": False,
                    "error": result.stderr or "Evaluation failed",
                    "input": expression[:500] + "..." if len(expression) > 500 else expression
                }

            # Parse JSON output
            output = result.stdout.strip()
            if output:
                return json.loads(output)
            else:
                print(f"❌ cas_cli returned empty output for: {expression[:200]}...")
                return {"ok": False, "error": "Empty response from cas_cli", "input": expression[:500]}

        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout ({timeout}s) for expression of {len(expression)} chars")
            return {"ok": False, "error": f"Timeout ({timeout}s) - expression too complex", "input": expression[:500]}
        except FileNotFoundError:
            return {"ok": False, "error": f"cas_cli not found. Run 'cargo build --release' first.", "input": expression[:500]}
        except json.JSONDecodeError as e:
            stderr_info = result.stderr[:200] if result and result.stderr else "no stderr"
            stdout_preview = result.stdout[:200] if result and result.stdout else "empty"
            print(f"❌ JSON decode error: {e}")
            print(f"   stderr: {stderr_info}")
            print(f"   stdout preview: {stdout_preview}")
            return {"ok": False, "error": f"Invalid JSON from cas_cli: {e}", "input": expression[:500], "raw": stdout_preview}
    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def send_json_error(self, message):
        self.send_response(400)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"ok": False, "error": message}).encode('utf-8'))
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def main():
    # Check if cas_cli exists
    if not os.path.exists(CAS_CLI):
        print(f"⚠️  Warning: {CAS_CLI} not found")
        print("   Run 'cargo build --release' to build the CLI")
        print()
    
    # Change to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # Start server
    with http.server.HTTPServer(('', PORT), CASHandler) as httpd:
        print(f"""
╔════════════════════════════════════════════════╗
║     🧮 ExpliCAS Web REPL Server                ║
╠════════════════════════════════════════════════╣
║  Open:  http://localhost:{PORT}                  ║
║  API:   http://localhost:{PORT}/api/eval         ║
║                                                ║
║  Features:                                     ║
║  • Variable assignments: a := expr             ║
║  • Session references: %1, %2, ...             ║
║  • Clear session: POST /api/clear              ║
║                                                ║
║  Press Ctrl+C to stop                          ║
╚════════════════════════════════════════════════╝
""")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

if __name__ == "__main__":
    main()
