# Security Policy

ExpliCAS evaluates **untrusted mathematical expressions** in three deployment
shapes — client-side WebAssembly on GitHub Pages, a local development server
that forwards input to the native binary, and an Android FFI layer. That is
the attack surface this policy covers.

This is a solo-maintained project. Reports are handled on a **best-effort
basis**: no response-time guarantees and no supported-versions matrix. The
supported target is the `main` branch and the live Pages deployment built
from it.

## Reporting a vulnerability

Please report security issues **privately** — not as public GitHub issues:

1. **Preferred**: GitHub's private vulnerability reporting — *Security* tab →
   *Report a vulnerability* on this repository.
2. Fallback: email the maintainer at `javiergimenezmoya@gmail.com` with
   `[ExpliCAS security]` in the subject.

Include the input (expression, session JSON, or request) that triggers the
problem and what you observed. A reproducible input is the whole report; the
project's own debugging starts from there.

## Scope

**In scope** — things we want reported privately:

- Code execution, memory unsafety, or sandbox escape triggered by a crafted
  expression, session-import JSON, or wire payload — in the native binary,
  the WASM build, or the Android FFI.
- Injection through the development server (`web/server.py`,
  `web/server_colab.py`): anything that turns an expression or HTTP request
  into shell/command execution or file access beyond the intended evaluation.
- Hangs or resource exhaustion that **bypass the documented budget system**
  (`docs/BUDGET_POLICY.md`): the engine's contract is that every evaluation
  is bounded by rewrite/node/time budgets, so an input that escapes them is
  a security-relevant robustness failure, not just slowness.
- Vulnerabilities introduced through the dependency tree that are reachable
  from expression evaluation.

**Out of scope** — better reported as regular issues:

- Slow evaluations that stay *within* the configured budgets, or inputs the
  engine honestly declines.
- Mathematically wrong results (the project treats these as its highest-class
  ordinary bugs — see the soundness audits under `docs/` — but they are not
  security reports).
- Denial of service against a `web/server.py` instance someone chose to
  expose publicly (see the trust model below).

## Trust model, stated plainly

- **Expressions are always untrusted input.** The parser and engine are
  expected to answer, decline, or fail a budget — never to crash the host or
  execute anything.
- **Session import replays through the engine.** Imported JSON transcripts
  are not trusted as state: each entry is re-evaluated by the same pipeline
  as typed input, so a crafted session file has exactly the power of typing
  the same expressions.
- **The WASM deployment is client-side only.** There is no server component
  on GitHub Pages; the blast radius of a malicious expression there is the
  visitor's own tab.
- **`web/server.py` is a development server, not a product.** Note that it
  binds **all network interfaces**, not just localhost — machines on your
  LAN can reach it while you develop (this is deliberate: the Colab variant
  needs it). Do not deploy it publicly; nothing in it is hardened for that.

## Dependency auditing

`cargo audit` runs in CI (`.github/workflows/audit.yml`) on every change to
the dependency tree and on a weekly schedule against the RustSec advisory
database. Advisory-level *warnings* (unmaintained/unsound notices) are
reviewed but do not fail the build; *vulnerabilities* do.
