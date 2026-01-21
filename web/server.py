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
from urllib.parse import parse_qs

PORT = 8080
CAS_CLI = "./target/release/cas_cli"

# Session state - persists for lifetime of server
session_variables = {}  # name -> result string
session_results = []     # list of all results for #n references

class CASHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve from web/ directory
        super().__init__(*args, directory="web", **kwargs)
    
    def do_POST(self):
        if self.path == '/api/eval':
            self.handle_eval()
        elif self.path == '/api/clear':
            self.handle_clear()
        else:
            self.send_error(404)
    
    def handle_clear(self):
        """Clear session state"""
        global session_variables, session_results
        session_variables = {}
        session_results = []
        self.send_json({"ok": True, "message": "Session cleared"})
    
    def handle_eval(self):
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body)
            expression = data.get('expression', '').strip()
            
            if not expression:
                self.send_json_error("No expression provided")
                return
            
            # Check for assignment: name := expr
            assignment_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:=\s*(.+)$', expression)
            
            if assignment_match:
                var_name = assignment_match.group(1)
                expr_part = assignment_match.group(2)
                result = self.eval_with_substitution(expr_part)
                
                if result.get('ok', False):
                    # Store the result for this variable
                    session_variables[var_name] = result.get('result', '')
                    result['assignment'] = var_name
                    result['input'] = expression
                    
            else:
                result = self.eval_with_substitution(expression)
            
            # Store result for #n references
            session_results.append(result)
            result['ref'] = len(session_results)
            
            # Include current variables in response
            result['variables'] = list(session_variables.keys())
            
            self.send_json(result)
            
        except json.JSONDecodeError:
            self.send_json_error("Invalid JSON")
        except Exception as e:
            self.send_json_error(str(e))
    
    def eval_with_substitution(self, expression):
        """Evaluate expression, substituting known variables"""
        expr = expression
        
        # Substitute session references: %n -> result of expression n
        def replace_ref(match):
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(session_results):
                ref_result = session_results[idx].get('result', '')
                return f"({ref_result})"
            return match.group(0)
        
        expr = re.sub(r'%(\d+)', replace_ref, expr)
        
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
            # Run cas_cli with the expression
            result = subprocess.run(
                [CAS_CLI, "eval", "--format", "json", "--max-chars", "500000"],
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
