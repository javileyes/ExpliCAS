# ExpliCAS Web Server

Interactive web-based REPL for the ExpliCAS computer algebra system.

## Quick Start

```bash
# From the repository root
# 1) Build the CLI binary used by the web server
cargo build --release -p cas_cli

# 2) Start the web server
python3 web/server.py
```

Then open http://localhost:8080

If `python` on your system already points to Python 3, this also works:

```bash
python web/server.py
```

To configure the web simplification budget from a `.env`, copy the example file first:

```bash
cp .env.example .env
```

Then adjust `WEB_TIME_BUDGET_MS` and rebuild:

```bash
make build-release
```

That build refreshes `web/build-config.js` from `.env`, and the Python server also reads the same `.env` at runtime.

## Features

- **Interactive REPL**: Enter mathematical expressions and see step-by-step solutions
- **Variable Assignments**: `a := 5`, `expr := x^2 + 2*x + 1`
- **Cell References**: Use `%1`, `%2`, etc. to reference previous results
- **Session Isolation**: Each browser tab has its own isolated session
- **Step-by-step Display**: Expandable derivation steps for each simplification

## Session Management

### Per-Tab Isolation

Each browser tab/window gets its own independent session:

- **Variables** defined in one tab are not visible in other tabs
- **Cell references** (`%1`, `%2`) only work within the same tab
- **Clear Session** only affects the current tab

### How It Works

```
┌─────────────────────┐     ┌─────────────────────┐
│    Browser Tab 1    │     │    Browser Tab 2    │
│  sessionStorage     │     │  sessionStorage     │
│  session-abc123...  │     │  session-xyz789...  │
└─────────┬───────────┘     └─────────┬───────────┘
          │                           │
          ▼                           ▼
┌─────────────────────────────────────────────────┐
│                Python Server                     │
│  sessions = {                                    │
│    "session-abc123...": {variables, results},   │
│    "session-xyz789...": {variables, results}    │
│  }                                               │
└─────────────────────────────────────────────────┘
```

- **Client-side**: Session ID stored in `sessionStorage` (persists until tab is closed)
- **Server-side**: Sessions stored in Python `sessions` dictionary

### Session Timeout and Memory Management

Sessions automatically expire after **2 hours of inactivity**.

| Configuration | Default | Location |
|---------------|---------|----------|
| `SESSION_TIMEOUT_SECONDS` | 7200 (2 hours) | `server.py` |

**Cleanup Process**:
- Cleanup runs on every API request
- Expired sessions are automatically removed
- Server logs cleanup: `🧹 Cleaned up X expired session(s). Active sessions: Y`

**Memory Safety**:
- Closing a browser tab removes the session ID from `sessionStorage`
- Server data persists until timeout or explicit clear
- No memory leaks: all sessions eventually expire 

## API Endpoints

### POST `/api/eval`

Evaluate a mathematical expression.

**Request**:
```json
{
  "expression": "x^2 + 2*x + 1",
  "session_id": "session-abc123..."
}
```

**Response**:
```json
{
  "ok": true,
  "input": "x^2 + 2*x + 1",
  "result": "x² + 2·x + 1",
  "result_latex": "x^{2} + 2 \\cdot x + 1",
  "steps": [...],
  "session_id": "session-abc123...",
  "ref": 1,
  "variables": ["a", "b"]
}
```

### POST `/api/clear`

Clear all variables and results for a session.

**Request**:
```json
{
  "session_id": "session-abc123..."
}
```

**Response**:
```json
{
  "ok": true,
  "message": "Session cleared",
  "session_id": "session-abc123..."
}
```

### POST `/api/delete-variable`

Delete a specific variable from a session.

**Request**:
```json
{
  "variable": "a",
  "session_id": "session-abc123..."
}
```

**Response**:
```json
{
  "ok": true,
  "message": "Variable 'a' deleted"
}
```

## Variable Assignment

Assign values to variables for reuse:

```
a := 5          → Stores a = 5
b := a + 3      → Evaluates to 8, stores b = 8
a^2 + b         → Evaluates to 33 (using stored values)
```

## Cell References

Reference previous results:

```
#1  x^2 + 1         → Result stored, ref #1
#2  %1 + x          → Uses result from #1: (x² + 1) + x
#3  2 * %2          → Uses result from #2
```

Both `%n` and `#n` syntax work.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8080 | Server port |
| `WEB_TIME_BUDGET_MS` | 1700 | Frontend/default simplification budget passed to the CLI |
| `WEB_MIN_HARD_TIMEOUT_SECONDS` | 8 | Minimum hard subprocess timeout for the web server |
| `WEB_TIMEOUT_GRACE_SECONDS` | 5 | Grace window added on top of the cooperative engine budget |
| `CAS_CLI` | `./target/release/cas_cli` | Path to CLI binary |
| `SESSION_TIMEOUT_SECONDS` | 7200 | Session expiration (2 hours) |

## Development

### Requirements

- Python 3
- Built `cas_cli` binary at `./target/release/cas_cli`

### File Structure

```
web/
├── server.py      # Python HTTP server
├── index.html     # Single-page web application
└── README.md      # This file
```

### Debugging

Session IDs are logged to the browser console:
```
Session ID: session-60acfc57-f0a0-4457-a353-8516474f025b
```

Server logs show session cleanup:
```
🧹 Cleaned up 3 expired session(s). Active sessions: 5
```
