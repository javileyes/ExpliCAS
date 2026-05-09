#!/bin/bash

set -euo pipefail

NOTES=$(cat <<'EOF'
# ExpliCAS v1.3 — Calculus, Real-Domain Semantics, and Presentation

**ExpliCAS** is a modular Computer Algebra System written in Rust, focused on **step-by-step mathematical explanations** for educational use.

🌐 **Try it online:** [https://sanjuanbosco.javiergimenez.es/](https://sanjuanbosco.javiergimenez.es/)

---

## Highlights in v1.3

- **Much stronger symbolic differentiation** for nested functions, inverse trigonometric forms, radicals, exponentials, logarithms, and composed expressions.
- **Expanded conservative symbolic integration** with verified antiderivatives for common polynomial, trigonometric, logarithmic, affine-log, and integration-by-parts families.
- **Antiderivative verification by differentiation** for promoted integration families, reducing the risk of accepting unsafe integration shortcuts.
- **Improved post-calculus presentation**, including more compact forms for reciprocal powers, radicals, factored denominators, and derived calculus results.
- **More rigorous real-domain semantics** for logarithms, absolute values, inverse trigonometric expressions, and mode-sensitive assumptions versus requirements.
- **Better equivalence diagnostics**, including residual reporting for false equivalences where appropriate.
- **Stronger CI and guardrail coverage**, including scorecard lanes, pressure checks, and lints that work even when optional developer tools such as ripgrep are not installed.

---

## Calculus Improvements

### Differentiation
- Nested chain-rule cases such as `diff(sin(e^(x^2)), x)`.
- Inverse trigonometric compositions such as `diff(arctan(sqrt(x)), x)`.
- Public diff results that remain simplifiable when embedded inside larger expressions and residual checks.

### Integration
- Polynomial-trigonometric integration by parts, for example `integrate(x^2 * sin(x), x)`.
- Affine-log integration-by-parts families such as `integrate((x+1)*ln(2*x+1), x)`.
- Conservative domain-aware log/radical forms, with antiderivatives checked by differentiating back to the integrand.

### Presentation
- Cleaner post-calculus simplification for results containing radicals, reciprocal powers, and factored denominators.
- More stable output for embedded calculus residuals, reducing noisy nonzero leftovers when the identity is already known.

---

## Semantics and Modes

- Real-domain policy alignment across logarithms, absolute values, radicals, and equivalence.
- Clearer distinction between requirements produced by the expression and assumptions introduced by an `assume`-style operation.
- Web and engine paths updated toward explicit generic versus assume-mode behavior.

---

## Core Features

### Symbolic Computation
- **Step-by-step simplification** with explicit rule traces
- **Polynomial expansion and factorization**
- **Grouping terms** with `collect(...)`
- **Fraction simplification**
- **Equivalence checking** with residual diagnostics

### Calculus
- **Symbolic differentiation** with `diff(...)`
- **Symbolic integration** with `integrate(...)`
- **Limits** with `limit(...)`
- **Verified antiderivative families** for promoted integration rules

### Algebra and Solving
- **Equation solving** with `solve(...)`
- **Linear systems** with `solve_system(...)`
- **Derivation between equivalent expressions** with `derive(...)`

### Functions and Number Theory
- **Trigonometric identities**
- **Logarithms and exponentials**
- **Real-domain absolute value reasoning**
- **Factorials, sums, products, gcd/lcm, modular arithmetic, combinatorics**

---

## Interfaces

### Web Application
- MathJax-rendered expressions
- Mode-aware symbolic workflows
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

gh release create v1.3.0 \
  --title "🧮 ExpliCAS v1.3 — Calculus, Real-Domain Semantics, and Presentation" \
  --notes "$NOTES"
