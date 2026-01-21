#!/usr/bin/env python3
"""
ExpliCAS Web Server

Simple HTTP server that:
1. Serves static files from web/
2. Proxies /api/eval requests to cas_cli eval-json

Usage:
    cd /path/to/math
    python web/server.py
    
Then open http://localhost:8080
"""

import http.server
import json
import os
import subprocess
import sys
from urllib.parse import parse_qs

PORT = 8080
CAS_CLI = "./target/release/cas_cli"

class CASHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve from web/ directory
        super().__init__(*args, directory="web", **kwargs)
    
    def do_POST(self):
        if self.path == '/api/eval':
            self.handle_eval()
        else:
            self.send_error(404)
    
    def handle_eval(self):
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body)
            expression = data.get('expression', '')
            
            if not expression:
                self.send_json_error("No expression provided")
                return
            
            # Call cas_cli eval-json
            result = self.call_cas_cli(expression)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
            
        except json.JSONDecodeError:
            self.send_json_error("Invalid JSON")
        except Exception as e:
            self.send_json_error(str(e))
    
    def call_cas_cli(self, expression):
        """Call cas_cli eval-json and return parsed result"""
        try:
            # Run cas_cli with the expression
            result = subprocess.run(
                [CAS_CLI, "eval", "--format", "json", "--max-chars", "100000"],
                input=expression,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return {
                    "error": result.stderr or "Evaluation failed",
                    "input": expression
                }
            
            # Parse JSON output
            output = result.stdout.strip()
            if output:
                return json.loads(output)
            else:
                return {"error": "Empty response", "input": expression}
                
        except subprocess.TimeoutExpired:
            return {"error": "Timeout (30s)", "input": expression}
        except FileNotFoundError:
            return {"error": f"cas_cli not found. Run 'cargo build --release' first.", "input": expression}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON from cas_cli: {e}", "input": expression, "raw": result.stdout[:500]}
    
    def send_json_error(self, message):
        self.send_response(400)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode('utf-8'))
    
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
║  Press Ctrl+C to stop                          ║
╚════════════════════════════════════════════════╝
""")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

if __name__ == "__main__":
    main()
