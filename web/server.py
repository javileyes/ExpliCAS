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

PORT = 8080
CAS_CLI = "./target/release/cas_cli"

# Session configuration
SESSION_TIMEOUT_SECONDS = 2 * 60 * 60  # 2 hours of inactivity before session expires

# Multi-session support: each browser tab gets its own session
# Sessions are identified by a UUID stored in the browser's sessionStorage
# Each session tracks: variables, results, and last_access time for cleanup
sessions = {}  # session_id -> {"variables": {}, "results": [], "last_access": timestamp}

def get_session(session_id):
    """Get or create a session by ID, updating last access time"""
    import time
    current_time = time.time()
    
    # Clean up expired sessions periodically
    cleanup_expired_sessions(current_time)
    
    if session_id not in sessions:
        sessions[session_id] = {
            "variables": {},  # name -> result string
            "results": [],    # list of all results for %n references
            "last_access": current_time
        }
    else:
        # Update last access time
        sessions[session_id]["last_access"] = current_time
    
    return sessions[session_id]

def clear_session(session_id):
    """Clear a specific session (but keep the session entry for new use)"""
    import time
    if session_id in sessions:
        sessions[session_id] = {
            "variables": {},
            "results": [],
            "last_access": time.time()
        }

def cleanup_expired_sessions(current_time):
    """Remove sessions that haven't been accessed in SESSION_TIMEOUT_SECONDS"""
    expired = [
        sid for sid, data in sessions.items()
        if current_time - data.get("last_access", 0) > SESSION_TIMEOUT_SECONDS
    ]
    for sid in expired:
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
        """Import session state from notebook data"""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'
        
        try:
            data = json.loads(body) if body else {}
            session_id = data.get('session_id', 'default')
            variables = data.get('variables', {})
            results = data.get('results', [])
            replace = data.get('replace', True)
            
            session = get_session(session_id)
            
            # Build a properly indexed results list
            # The server uses list index to look up #n references (idx = n - 1)
            # So we need results[0] = #1, results[1] = #2, etc.
            max_ref = 0
            results_by_ref = {}
            for r in results:
                ref = r.get('ref', 0)
                if ref > 0:
                    results_by_ref[ref] = r
                    max_ref = max(max_ref, ref)
            
            # Build ordered list with placeholders for gaps
            imported_results = []
            for i in range(1, max_ref + 1):
                if i in results_by_ref:
                    imported_results.append(results_by_ref[i])
                else:
                    # Placeholder for missing refs (shouldn't normally happen)
                    imported_results.append({'result': '', 'ref': i})
            
            if replace:
                # Clear and replace
                session["variables"] = dict(variables)
                session["results"] = imported_results
            else:
                # Append to existing - extend with new results
                session["variables"].update(variables)
                # For append, we need to adjust refs to continue from current count
                current_len = len(session["results"])
                for r in imported_results:
                    session["results"].append(r)
            
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
                result = self.eval_with_substitution(expr_part, session)
                
                if result.get('ok', False):
                    # Store the result for this variable (but NOT in results list)
                    session["variables"][var_name] = result.get('result', '')
                    result['assignment'] = var_name
                    result['input'] = expression
                    # Assignments don't get a ref - they don't consume reference numbers
                    result['ref'] = None
                    
            else:
                result = self.eval_with_substitution(expression, session)
                # Only non-assignment evaluations get stored for %n references
                session["results"].append(result)
                result['ref'] = len(session["results"])
            
            # Include current variables and session_id in response
            result['variables'] = list(session["variables"].keys())
            result['session_id'] = session_id
            
            self.send_json(result)
            
        except json.JSONDecodeError:
            self.send_json_error("Invalid JSON")
        except Exception as e:
            self.send_json_error(str(e))
    
    def eval_with_substitution(self, expression, session):
        """Evaluate expression, substituting known variables from the session"""
        expr = expression
        session_results = session["results"]
        session_variables = session["variables"]
        
        # Substitute session references: %n or #n -> result of expression n
        def replace_ref(match):
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(session_results):
                ref_result = session_results[idx].get('result', '')
                return f"({ref_result})"
            return match.group(0)
        
        # Support both %n and #n syntax
        expr = re.sub(r'[%#](\d+)', replace_ref, expr)
        
        # Substitute variables (careful with word boundaries)
        for var_name, var_value in session_variables.items():
            # Use word boundary to avoid partial replacements
            pattern = r'\b' + re.escape(var_name) + r'\b'
            # Only substitute if not part of a function call or already substituted
            expr = re.sub(pattern, f"({var_value})", expr)
        
        return self.call_cas_cli(expr)
    
    def call_cas_cli(self, expression):
        """Call cas_cli eval-json and return parsed result"""
        try:
            # Run cas_cli with the expression and steps enabled
            result = subprocess.run(
                [CAS_CLI, "eval", "--format", "json", "--max-chars", "500000", "--steps", "on"],
                input=expression,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return {
                    "ok": False,
                    "error": result.stderr or "Evaluation failed",
                    "input": expression
                }
            
            # Parse JSON output
            output = result.stdout.strip()
            if output:
                return json.loads(output)
            else:
                return {"ok": False, "error": "Empty response", "input": expression}
                
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Timeout (60s)", "input": expression}
        except FileNotFoundError:
            return {"ok": False, "error": f"cas_cli not found. Run 'cargo build --release' first.", "input": expression}
        except json.JSONDecodeError as e:
            return {"ok": False, "error": f"Invalid JSON from cas_cli: {e}", "input": expression, "raw": result.stdout[:500]}
    
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
