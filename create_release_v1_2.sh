#!/bin/bash

set -euo pipefail

NOTES=$(cat <<'EOF'
# ExpliCAS v1.2 — Educational Computer Algebra System

**ExpliCAS** is a modular Computer Algebra System written in Rust, focused on **step-by-step mathematical explanations** for educational use.

🌐 **Try it online:** [https://sanjuanbosco.javiergimenez.es/](https://sanjuanbosco.javiergimenez.es/)

---

## Highlights in v1.2

- **Interactive web application** powered by the Rust engine and a lightweight Python server
- **Step-by-step derivations** with more human-friendly didactic substeps
- **CLI / JSON / web workflow parity** for special commands such as `derive`, `collect`, `limit`, and `solve_system`
- **User-defined functions** and symbolic variable assignments
- **Improved LaTeX presentation**, preserving user input style where appropriate
- **Expanded symbolic coverage** in algebra, calculus, logarithms, trigonometry, linear systems, and number theory

---

## Core Features

### Symbolic Computation
- **Step-by-step simplification** with explicit rule traces
- **Polynomial expansion and factorization**
- **Grouping terms** with `collect(...)`
- **Fraction simplification**
- **Equivalence checking**

### Calculus
- **Symbolic differentiation** with `diff(...)`
- **Symbolic integration** with `integrate(...)`
- **Limits** with `limit(...)`

### Algebra and Solving
- **Equation solving** with `solve(...)`
- **Linear systems** with `solve_system(...)`
- **Derivation between equivalent expressions** with `derive(...)`

### Functions and Number Theory
- **Trigonometric identities**
- **Logarithms and exponentials**
- **Factorials, sums, products, gcd/lcm, modular arithmetic, combinatorics**

---

## Interfaces

### Web Application
- MathJax-rendered expressions
- Step-by-step expansion panel
- Session persistence and result references
- Import/export of notebook-like sessions

### CLI / REPL
- Interactive shell with history
- Unicode pretty output
- Configurable verbosity and step modes
- JSON output via `cas_cli eval --format json`

---

## Installation

```bash
git clone https://github.com/javileyes/ExpliCAS.git
cd ExpliCAS

# Build the engine used by both the CLI and the web app
cargo build --release -p cas_cli

# CLI
./target/release/cas_cli

# Web application
python3 web/server.py
```

Then open `http://localhost:8080`.

**MIT License** · Built by Javier Giménez Moya
EOF
)

gh release create v1.2.0 \
  --title "🧮 ExpliCAS v1.2 — Educational Computer Algebra System" \
  --notes "$NOTES"
