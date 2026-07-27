#!/bin/bash

set -euo pipefail

NOTES=$(cat <<'EOF'
# ExpliCAS v1.4 — Differential Equations, Vector Calculus, In-Browser Engine, and the Math Keypad

**ExpliCAS** is a modular Computer Algebra System written in Rust, focused on **step-by-step mathematical explanations** for educational use.

🌐 **Try it online:** [https://sanjuanbosco.javiergimenez.es/](https://sanjuanbosco.javiergimenez.es/)
🚀 **Fully in-browser (WASM, no server):** [https://javileyes.github.io/ExpliCAS/](https://javileyes.github.io/ExpliCAS/)

---

## Highlights in v1.4

- **Ordinary differential equations** with `dsolve(...)`: the complete elementary course — first order (separable, linear via integrating factor, exact, Bernoulli, homogeneous), second order with constant coefficients (including undetermined-coefficient resonances), initial value problems, and 2×2 linear systems. Every emitted solution is **verified exactly** against the ODE before being shown.
- **Multivariable and vector calculus**: `gradient`, `jacobian`, `hessian`, `divergence`, `curl`, `laplacian`, `potential`, line and surface integrals, multivariable limits with **proven** continuity / squeeze arguments and honest DNE-by-paths with cited witnesses, and multivariable Taylor.
- **Nonlinear 2×2 systems** in `solve(...)`: symbolic Cramer, verified isolate-substitute for line + conic, and the **Sylvester resultant** for two conics — every solution pair verified against both equations.
- **Universal rational integration**: closure over the roots of any denominator (`root_sum`), exact residues over ℚ, with algebraic numbers only at render time.
- **The engine now runs fully in the browser** via WebAssembly on GitHub Pages — same JSON wire as the CLI and server, with a stateful session (`#N` references, `:=` assignments) living in the tab.
- **On-screen math keypad** for phones, tablets, and desktop: layers for numbers, functions, and letters, color-coded by mathematical family, with the system keyboard suppressed on touch devices.
- **Single-glyph Greek input in the engine**: `α` ≡ `alpha` as one canonical symbol (with `π`/`φ` reaching the constants), rendered back as real Greek in MathJax.
- **Relative cell references**: `#-1` is the newest cell, `#-2` the one before — normalized to absolute references before parsing so stored sessions replay identically.

---

## Differential Equations

- `dsolve(diff(y,x) = x*y, y, x)` → `y = C·e^(x²/2)`, with narrated steps in Spanish and English.
- Second order: `dsolve(diff(y,x,2) + 4*y = 0, y, x)` → `y = C1·sin(2x) + C2·cos(2x)`; resonant right-hand sides handled by undetermined coefficients.
- Initial value problems: `dsolve(diff(y,x) = -y, y, x, y(0) = 3)` → `y = 3/eˣ`.
- Systems: `dsolve([diff(x,t) = -y, diff(y,t) = x], [x,y], t)` with real solutions built from complex eigenvalues.
- Cauchy-Euler equations and simple integrating factors μ(x)/μ(y); unsupported families decline honestly instead of fabricating.

## Multivariable and Vector Calculus

- Vector verbs over matrix literals: `gradient(x^2*y, [x,y])`, `curl([y,-x,0], [x,y,z])`, `laplacian(ln(x^2+y^2), [x,y])` → 0 (harmonic).
- `potential([2*x*y, x^2], [x,y])` → `x²y`, verified by `∇φ = F`.
- Line and surface integrals over parametrized curves and surfaces: circulation `∮F·dr` over the unit circle → `2π`, flux through the unit cylinder → `2π`.
- Multivariable limits: continuity is **proved** (exact nonzero denominator), squeeze bounds in polar form, and non-existence cites the two witness paths.

## Solving and Systems

- Nonlinear 2×2: `solve([x*y=6, x^2+y^2=13], [x,y])` → `(±2,±3), (±3,±2)` via the Sylvester resultant.
- Parametric linear systems keep exact symbolic coefficients and carry `det ≠ 0` conditions.
- Trigonometric, absolute-value, radical, exponential, and logarithmic equations and inequalities with periodic solution sets as unions.

## In-Browser Engine (WASM)

- The full Rust engine compiled to WebAssembly runs client-side on GitHub Pages: exact BigRational arithmetic, LaTeX, localized step-by-step — zero network after load.
- Stateful browser session: `#N` references, `:=` variable and function definitions, and session clear, all inside the tab.
- The classic Python server deployment is unchanged and keeps full parity.

## Web UX: Math Keypad and Function Menu

- **Calculator keypad** attached to the input bar: digits and operators, `x y z`, constants `π e φ`, brackets and a 2×2 matrix template, `:=`, differential and integral keys, cell-reference chips `#` and `ANT` (`#-1`).
- **f(x) layer** with trigonometry, logarithms and roots, calculus verbs, algebra verbs, and vector/matrix operations (`dot`, `cross`, `norm`, `det`, `inverse`, `transpose`, `matmul`, `gradient`, `divergence`, `curl`, `laplacian`, `jacobian`, `eigenvalues`) — every verb inserts its **template with the first argument pre-selected**.
- **Greek keyboard**: three fixed rows (no scrolling) with all lowercase letters plus Γ Δ Θ Λ Σ Φ Ω; Euler’s `e` is color-coded as a constant.
- Keys are **color-coded by family**: operators, variables, constants, trigonometry, calculus, algebra, vectors/matrices.
- **ƒ() insertion menu**: ~90 function templates in 12 categories, bilingual descriptions, hole-aware insertion that can wrap the current selection.
- Touch-first behavior: the system keyboard is suppressed on phones **and tablets** (portrait and landscape), with one button to swap back; desktop gets a persistent toggle.
- **Fullscreen button** next to the logo (hidden where the browser API does not exist).
- Card polish: headers pan horizontally like every other section, and both the input and the result show their **raw copy-paste form with a copy button**, always visible.
- Light/dark theme contrast pass across steps, sub-steps, panels, and cards.

## Semantics and Didactics

- Step narration is **verified before being published**: didactic steps are checked against the real engine trace, with divergence gates in CI.
- Numeric display mode: `decimal` approximates only the final presentation (12 significant digits) while everything stays exact and symbolic internally; `approx(...)` for one-off numeric values.
- Complex mode: Gaussian arithmetic, Euler identities, principal branches (`i^i → e^(-π/2)`), `Re/Im/conjugate/arg`, exact verification of identities like `e^(iπ) = -1`.

---

## Core Features

### Symbolic Computation
- **Step-by-step simplification** with explicit rule traces
- **Polynomial expansion and factorization**
- **Grouping terms** with `collect(...)`
- **Fraction simplification and partial fractions** with `apart(...)`
- **Equivalence checking** with residual diagnostics

### Calculus
- **Symbolic differentiation** with `diff(...)`
- **Symbolic integration** with `integrate(...)`, antiderivatives verified by differentiation
- **Limits** with `limit(...)` (L’Hôpital, notable limits, one- and multi-variable)
- **Taylor series** with `taylor(...)`, sums and products with `sum(...)` / `product(...)`
- **Differential equations** with `dsolve(...)`

### Algebra and Solving
- **Equation and inequality solving** with `solve(...)`
- **Linear and nonlinear 2×2 systems** with `solve(...)` / `solve_system(...)`
- **Derivation between equivalent expressions** with `derive(...)`

### Linear Algebra, Functions and Number Theory
- **Matrices**: `det`, `inverse`, `transpose`, `rank`, `trace`, `rref`, `charpoly`, `eigenvalues`, `eigenvectors`, `linsolve`
- **Vectors**: `dot`, `cross`, `norm`, and the vector-calculus verbs
- **Trigonometric identities, logarithms, exponentials, absolute-value reasoning**
- **Factorials, gcd/lcm, modular arithmetic, primes, combinatorics**

---

## Interfaces

### Web Application
- Runs either against the Python server or **fully in-browser via WASM**
- MathJax-rendered expressions with raw copy-paste captions
- On-screen math keypad and ƒ() template menu
- Step-by-step expansion panel, localized in Spanish and English
- Session persistence, `#N` / `#-k` references, import/export of sessions

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

Then open `http://localhost:8080` — or use the zero-install WASM build at
[https://javileyes.github.io/ExpliCAS/](https://javileyes.github.io/ExpliCAS/).

**MIT License** · Built by Javier Giménez Moya
EOF
)

gh release create v1.4.0 \
  --title "🧮 ExpliCAS v1.4 — Differential Equations, Vector Calculus, In-Browser Engine, and the Math Keypad" \
  --notes "$NOTES"
