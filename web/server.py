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
    
Then open http://localhost:8081
"""

import http.server
import json
import math
import os
import re
import subprocess
import sys
import uuid
from urllib.parse import parse_qs
import hashlib
import tempfile
from contextlib import contextmanager, nullcontext

from app_config import (
    DEFAULT_WEB_MIN_HARD_TIMEOUT_SECONDS,
    DEFAULT_WEB_TIME_BUDGET_MS,
    DEFAULT_WEB_TIMEOUT_GRACE_SECONDS,
    env_int,
    render_frontend_build_config_js,
)

#get from .env
PORT = env_int("PORT", 8080)
CAS_CLI = "./target/release/cas_cli"
# Calibrated below the visible 2s SLA because the engine deadline is cooperative
# and some root shortcuts can overrun the budget slightly before the next checkpoint.
WEB_TIME_BUDGET_MS = env_int("WEB_TIME_BUDGET_MS", DEFAULT_WEB_TIME_BUDGET_MS)
WEB_MIN_HARD_TIMEOUT_SECONDS = env_int(
    "WEB_MIN_HARD_TIMEOUT_SECONDS", DEFAULT_WEB_MIN_HARD_TIMEOUT_SECONDS
)
WEB_TIMEOUT_GRACE_SECONDS = env_int(
    "WEB_TIMEOUT_GRACE_SECONDS", DEFAULT_WEB_TIMEOUT_GRACE_SECONDS
)

# CLI session snapshot configuration (for fast #N references without textual substitution)
SESSION_SNAPSHOT_DIR = os.path.join(tempfile.gettempdir(), "explicas_cli_sessions")
os.makedirs(SESSION_SNAPSHOT_DIR, exist_ok=True)

def _split_top_level_pair(input_str: str):
    """Split `expr1, expr2` on the last top-level comma."""
    depth = 0
    split_pos = None
    for i, ch in enumerate(input_str):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(depth - 1, 0)
        elif ch == "," and depth == 0:
            split_pos = i

    if split_pos is None:
        return None

    left = input_str[:split_pos].strip()
    right = input_str[split_pos + 1 :].strip()
    if not left or not right:
        return None
    return left, right

def _parse_special_pair_command(input_str: str, command: str):
    trimmed = input_str.strip()
    lower = trimmed.lower()
    if lower == command:
        return None

    fn_prefix = f"{command}("
    spaced_prefix = f"{command} "
    if lower.startswith(fn_prefix) and trimmed.endswith(")"):
        content = trimmed[len(command) + 1 : -1].strip()
    elif lower.startswith(spaced_prefix):
        content = trimmed[len(command) :].strip()
    else:
        return None

    return _split_top_level_pair(content)

def _parse_equiv_pair(input_str: str):
    return _parse_special_pair_command(input_str, "equiv")

def _build_derive_command_from_equiv(input_str: str):
    pair = _parse_equiv_pair(input_str)
    if pair is None:
        return None
    lhs, rhs = pair
    return f"derive {lhs}, {rhs}"

def _equiv_result_is_true(result: dict) -> bool:
    return result.get("ok", False) and str(result.get("result", "")).lower() == "true"

def _merge_equiv_with_derive_steps(equiv_result: dict, derive_result: dict | None):
    merged = dict(equiv_result)
    merged["steps"] = []
    merged["steps_count"] = 0
    merged["steps_mode"] = "off"
    merged.pop("strategy", None)

    if derive_result and derive_result.get("ok", False):
        derive_steps = derive_result.get("steps") or []
        if derive_steps:
            merged["steps"] = derive_steps
            merged["steps_count"] = derive_result.get("steps_count", len(derive_steps))
            merged["steps_mode"] = derive_result.get("steps_mode", "on")
            if derive_result.get("strategy"):
                merged["strategy"] = derive_result["strategy"]

            wire = dict(merged.get("wire") or {})
            messages = [
                msg
                for msg in wire.get("messages", [])
                if msg.get("kind") not in {"steps"}
                and not (
                    msg.get("kind") == "info"
                    and str(msg.get("text", "")).startswith("Strategy:")
                )
            ]
            if derive_result.get("strategy"):
                messages.append(
                    {
                        "kind": "info",
                        "text": f"Strategy: {derive_result['strategy']}",
                    }
                )
            messages.append(
                {
                    "kind": "steps",
                    "text": derive_result.get("wire", {})
                    .get("messages", [{}])[-1]
                    .get("text", f"{merged['steps_count']} step(s)"),
                }
            )
            if wire:
                wire["messages"] = messages
                merged["wire"] = wire

    return merged

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


def _coerce_time_budget_ms(raw_value):
    """Parse optional request budget; fall back to the web default."""
    if raw_value is None or raw_value == "":
        return WEB_TIME_BUDGET_MS

    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        raise ValueError("time_budget_ms must be an integer")

    if value < 0:
        raise ValueError("time_budget_ms must be >= 0")

    return value

def _coerce_domain_mode(raw_value):
    """Parse the web domain selector (strict/generic/assume; default generic)."""
    if raw_value is None or raw_value == "":
        return "generic"

    value = str(raw_value).strip().lower()
    if value not in {"strict", "generic", "assume"}:
        raise ValueError("domain must be 'strict', 'generic' or 'assume'")

    return value


def _coerce_language(raw_value):
    """Parse the step-by-step language selector (es/en; default es)."""
    if raw_value is None or raw_value == "":
        return "es"

    value = str(raw_value).strip().lower()
    if value not in {"es", "en"}:
        return "es"

    return value


def _coerce_complex_arithmetic(raw_value):
    """Parse the web complex-arithmetic selector (off/on; default off).

    Maps to the CLI flag --value-domain complex, the axis that activates
    Gaussian arithmetic on `i` (i^2 = -1, conjugate products); --complex is
    a different axis and does not gate this.
    """
    if raw_value is None or raw_value == "":
        return "off"

    value = str(raw_value).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError("complex_arithmetic must be 'off' or 'on'")

    return value


def _coerce_numeric_display(raw_value):
    """Parse the web numeric-display selector (exact/decimal; default exact).

    Maps to the CLI flag --numeric-display, a PRESENTATION-ONLY axis: the
    engine stays exact and symbolic internally and the final result (and
    solution-set members) is approximated at the output boundary.
    """
    if raw_value is None or raw_value == "":
        return "exact"

    value = str(raw_value).strip().lower()
    if value not in {"exact", "decimal"}:
        raise ValueError("numeric_display must be 'exact' or 'decimal'")

    return value


def _coerce_branch_mode(raw_value):
    """Parse the web inverse-trig branch selector (strict/principal; default strict).

    Maps to the CLI flag --inv-trig, the knob that actually gates
    arcfun(fun(x)) principal-range simplifications (--branch is a legacy
    no-op flag kept for wire compatibility).
    """
    if raw_value is None or raw_value == "":
        return "strict"

    value = str(raw_value).strip().lower()
    if value not in {"strict", "principal"}:
        raise ValueError("branch must be 'strict' or 'principal'")

    return value


def _compute_subprocess_timeout_seconds(expr_len: int, time_budget_ms: int | None) -> int:
    legacy_timeout = 120 if expr_len > 10000 else 60
    if time_budget_ms is None:
        return legacy_timeout

    cooperative_timeout = max(
        WEB_MIN_HARD_TIMEOUT_SECONDS,
        math.ceil(time_budget_ms / 1000) + WEB_TIMEOUT_GRACE_SECONDS,
    )
    return min(legacy_timeout, cooperative_timeout)

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
            "functions": {},          # name -> {internal, params, display}
            "fn_counter": 0,           # fresh internal function names (__webfnN)
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
            "functions": {},
            "fn_counter": 0,
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
        elif self.path == '/build-config.js':
            self.handle_build_config_js()
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
        elif self.path == '/api/delete-function':
            self.handle_delete_function()
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
                        "description": row.get('description', ''),
                        "group": row.get('group', '')
                    })
            self.send_json({"ok": True, "examples": examples})
        except FileNotFoundError:
            self.send_json({"ok": False, "error": "Examples file not found", "examples": []})
        except Exception as e:
            self.send_json({"ok": False, "error": str(e), "examples": []})

    def handle_build_config_js(self):
        payload = render_frontend_build_config_js().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/javascript; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    
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
            functions = data.get('functions', {})
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
                merged_functions = dict(functions)
                session["variables"] = dict(variables)
                session["results"] = imported_results
                session["ref_map"] = [None] * len(session["results"])
            else:  # append
                merged_functions = {
                    name: {"params": meta.get("params", []), "display": meta.get("display", "")}
                    for name, meta in session.get("functions", {}).items()
                }
                merged_functions.update(functions)
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

            # The snapshot was invalidated, so every function (kept or
            # imported) must be re-registered under a fresh internal name in
            # the new snapshot; the stored display body is its definition.
            session["functions"] = {}
            session["fn_counter"] = 0
            for name, meta in merged_functions.items():
                params = [str(p) for p in meta.get("params", [])] or ["x"]
                display = str(meta.get("display", "")).strip()
                if not display or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
                    continue
                session["fn_counter"] += 1
                internal = f"__webfn{session['fn_counter']}"
                definition = f"{internal}({', '.join(params)}) := {display}"
                reg_result, _ = self.eval_and_store(
                    definition,
                    session,
                    None,
                    "generic",
                    skip_vars=set(params),
                )
                if reg_result.get('ok', False):
                    session["functions"][name] = {
                        "internal": internal,
                        "params": params,
                        "display": reg_result.get('result', display),
                    }

            self.send_json({
                "ok": True,
                "message": f"Imported {len(variables)} variables, {len(session['functions'])} functions and {len(imported_results)} results",
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
    
    def handle_delete_function(self):
        """Delete a defined function from session (drops the web-side name
        mapping; the orphaned internal definition in the CLI snapshot is
        unreachable and harmless, mirroring variable deletion)."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')

        try:
            data = json.loads(body)
            session_id = data.get('session_id', 'default')
            session = get_session(session_id)
            fn_name = data.get('function', '')

            if fn_name and fn_name in session.get("functions", {}):
                del session["functions"][fn_name]
                self.send_json({"ok": True, "message": f"Function '{fn_name}' deleted"})
            else:
                self.send_json({"ok": False, "error": f"Function '{fn_name}' not found"})
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
            time_budget_ms = _coerce_time_budget_ms(data.get("time_budget_ms"))
            domain_mode = _coerce_domain_mode(data.get("domain"))
            branch_mode = _coerce_branch_mode(data.get("branch"))
            complex_arithmetic = _coerce_complex_arithmetic(data.get("complex_arithmetic"))
            numeric_display = _coerce_numeric_display(data.get("numeric_display"))
            # Per-request step-by-step language, read back in call_cas_cli (same handler instance).
            self._step_language = _coerce_language(data.get("language"))
            session = get_session(session_id)
            
            if not expression:
                self.send_json_error("No expression provided")
                return
            
            # Check for function definition: name(p1, p2, ...) := body
            # The web owns the public name; the CLI session evaluates the
            # definition under a fresh internal name (__webfnN), so deleting
            # a function is just dropping the web-side mapping.
            function_match = re.match(
                r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*'
                r'([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)'
                r'\s*\)\s*:=\s*(.+)$',
                expression,
            )

            if function_match:
                fn_name = function_match.group(1)
                params = [p.strip() for p in function_match.group(2).split(',')]
                body = function_match.group(3)
                session["fn_counter"] = session.get("fn_counter", 0) + 1
                internal = f"__webfn{session['fn_counter']}"
                definition = f"{internal}({', '.join(params)}) := {body}"
                result, _cli_id = self.eval_and_store(
                    definition,
                    session,
                    time_budget_ms,
                    domain_mode,
                    branch_mode,
                    complex_arithmetic,
                    numeric_display,
                    skip_vars=set(params),
                )

                if result.get('ok', False):
                    session["functions"][fn_name] = {
                        "internal": internal,
                        "params": params,
                        "display": result.get('result', ''),
                    }
                    result['function_assignment'] = fn_name
                    result['function_params'] = params
                    result['input'] = expression
                    # Definitions don't get a ref - they don't consume reference numbers
                    result['ref'] = None

                result['variables'] = list(session["variables"].keys())
                result['functions'] = list(session["functions"].keys())
                result['session_id'] = session_id
                result['domain'] = domain_mode
                result['branch'] = branch_mode
                result['complex_arithmetic'] = complex_arithmetic
                result['numeric_display'] = numeric_display
                self.send_json(result)
                return

            # Check for assignment: name := expr
            assignment_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:=\s*(.+)$', expression)
            
            if assignment_match:
                var_name = assignment_match.group(1)
                expr_part = assignment_match.group(2)
                result, cli_id = self.eval_and_store(
                    expr_part,
                    session,
                    time_budget_ms,
                    domain_mode,
                    branch_mode,
                    complex_arithmetic,
                    numeric_display,
                )
                
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
                result, cli_id = self.eval_and_store(
                    expression,
                    session,
                    time_budget_ms,
                    domain_mode,
                    branch_mode,
                    complex_arithmetic,
                    numeric_display,
                )
                # Only non-assignment evaluations get stored for UI references
                session["results"].append(result)
                result['ref'] = len(session["results"])
                session["ref_map"].append(cli_id if result.get('ok', False) else None)
            
            # Include current variables and session_id in response
            result['variables'] = list(session["variables"].keys())
            result['functions'] = list(session["functions"].keys())
            result['session_id'] = session_id
            result['domain'] = domain_mode
            result['branch'] = branch_mode
            result['complex_arithmetic'] = complex_arithmetic
            result['numeric_display'] = numeric_display
            
            self.send_json(result)
            
        except json.JSONDecodeError:
            self.send_json_error("Invalid JSON")
        except ValueError as e:
            self.send_json_error(str(e))
        except Exception as e:
            self.send_json_error(str(e))

    def eval_and_store(
        self,
        expression,
        session,
        time_budget_ms,
        domain_mode,
        branch_mode="strict",
        complex_arithmetic="off",
        numeric_display="exact",
        skip_vars=None,
    ):
        """Evaluate via cas_cli using the per-session snapshot, tracking real stored ids."""
        session_file = session.get("session_file")
        lock_path = (session_file + ".lock") if session_file else None

        with (_file_lock(lock_path) if lock_path else nullcontext()):
            result = self._filter_plumbing_steps(
                self.eval_with_substitution(
                    expression,
                    session,
                    time_budget_ms,
                    domain_mode,
                    branch_mode,
                    complex_arithmetic,
                    numeric_display,
                    skip_vars=skip_vars,
                )
            )

            cli_id = None
            if result.get("ok", False) and result.get("stored_id") is not None:
                cli_id = int(result["stored_id"])
                session["cli_ref"] = max(session.get("cli_ref", 0), cli_id)

            return result, cli_id

    # Internal session-machinery rules that are not didactic math steps:
    # cached-ref resolution leaks CLI history ids (\#N) and meta-function
    # unwrapping is plumbing, so neither belongs in the web step list.
    PLUMBING_STEP_RULES = {"Use cached result", "Evaluate Meta Functions"}

    @classmethod
    def _filter_plumbing_steps(cls, result):
        if not isinstance(result, dict):
            return result
        steps = result.get('steps')
        if isinstance(steps, list) and steps:
            kept = [s for s in steps if s.get('rule') not in cls.PLUMBING_STEP_RULES]
            if len(kept) != len(steps):
                for position, step in enumerate(kept, start=1):
                    if isinstance(step, dict) and 'index' in step:
                        step['index'] = str(position)
                result['steps'] = kept
                if 'steps_count' in result:
                    result['steps_count'] = len(kept)
        return result

    @staticmethod
    def _present_original_input(result, original, rewritten):
        """When the server rewrote the expression (function internals, variable
        and #N ref substitution), the CLI's input/input_latex echo the rewritten
        form and would leak internals like __webfn1 into the card header. Show
        the user's original input as plain text instead."""
        has_ref_tokens = re.search(r'[%#]\d+', original) is not None
        if isinstance(result, dict) and (rewritten != original or has_ref_tokens):
            result['input'] = original
            result.pop('input_latex', None)
        return result

    def eval_with_substitution(
        self,
        expression,
        session,
        time_budget_ms,
        domain_mode,
        branch_mode="strict",
        complex_arithmetic="off",
        numeric_display="exact",
        skip_vars=None,
    ):
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

        # Rewrite defined-function calls to their internal CLI names
        # (longest names first so f2( is not shadowed by f(). The CLI
        # session owns the actual evaluation, including param shadowing.
        for fn_name in sorted(session.get("functions", {}), key=len, reverse=True):
            internal = session["functions"][fn_name]["internal"]
            expr = re.sub(r"\b" + re.escape(fn_name) + r"\s*\(", internal + "(", expr)

        # Substitute variables (word boundary to avoid partial replacements)
        for var_name, var_value in session_variables.items():
            if skip_vars and var_name in skip_vars:
                continue
            pattern = r"\b" + re.escape(var_name) + r"\b"
            if var_name in variable_refs:
                # Substitute with CLI session ref (fast, avoids huge text expansions)
                expr = re.sub(pattern, f"(#{variable_refs[var_name]})", expr)
            else:
                # Legacy fallback: substitute display value
                expr = re.sub(pattern, f"({var_value})", expr)

        derive_expr = _build_derive_command_from_equiv(expr)
        if derive_expr is not None:
            equiv_result = self.call_cas_cli(
                expr,
                session.get("session_file"),
                steps_on=False,
                time_budget_ms=time_budget_ms,
                domain_mode=domain_mode,
                branch_mode=branch_mode,
                complex_arithmetic=complex_arithmetic,
                numeric_display=numeric_display,
            )
            if _equiv_result_is_true(equiv_result):
                derive_result = self.call_cas_cli(
                    derive_expr,
                    session.get("session_file"),
                    steps_on=True,
                    time_budget_ms=time_budget_ms,
                    domain_mode=domain_mode,
                    branch_mode=branch_mode,
                    complex_arithmetic=complex_arithmetic,
                    numeric_display=numeric_display,
                )
                return self._present_original_input(
                    _merge_equiv_with_derive_steps(equiv_result, derive_result),
                    expression,
                    expr,
                )
            return self._present_original_input(
                _merge_equiv_with_derive_steps(equiv_result, None),
                expression,
                expr,
            )

        return self._present_original_input(
            self.call_cas_cli(
                expr,
                session.get("session_file"),
                time_budget_ms=time_budget_ms,
                domain_mode=domain_mode,
                branch_mode=branch_mode,
                complex_arithmetic=complex_arithmetic,
                numeric_display=numeric_display,
            ),
            expression,
            expr,
        )

    def call_cas_cli(
        self,
        expression,
        session_file=None,
        steps_on=True,
        time_budget_ms=None,
        domain_mode="generic",
        branch_mode="strict",
        complex_arithmetic="off",
        numeric_display="exact",
    ):
        """Call cas_cli eval-json and return parsed result"""
        result = None  # Initialize to handle exception cases
        try:
            # Log expression length for debugging large inputs
            expr_len = len(expression)
            if expr_len > 1000:
                print(f"⚠️  Large expression: {expr_len} chars")
            timeout = _compute_subprocess_timeout_seconds(expr_len, time_budget_ms)

            # Build CLI command (session snapshot enables fast #N references across calls)
            cmd = [
                CAS_CLI,
                "eval",
                "--format",
                "json",
                "--max-chars",
                "500000",
                "--steps",
                "on" if steps_on else "off",
                "--lang",
                getattr(self, "_step_language", "es"),
                "--domain",
                domain_mode,
                "--inv-trig",
                branch_mode,
            ]
            if complex_arithmetic == "on":
                cmd += ["--value-domain", "complex"]
            if numeric_display == "decimal":
                cmd += ["--numeric-display", "decimal"]
            if time_budget_ms is not None:
                cmd += ["--time-budget-ms", str(time_budget_ms)]
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
            if time_budget_ms is not None:
                error = (
                    f"Timeout ({timeout}s hard limit after {time_budget_ms}ms "
                    "simplification budget) - expression too complex"
                )
            else:
                error = f"Timeout ({timeout}s) - expression too complex"
            return {"ok": False, "error": error, "input": expression[:500]}
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
