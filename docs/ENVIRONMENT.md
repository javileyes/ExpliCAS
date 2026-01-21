# Session Environment & Store

ExpliCAS provides a **persistent session state** that allows you to store expressions, define variables, and reference previous results. This system mimics notebook-style workflows (like Mathematica or Jupyter) and is designed for educational use.

> [!TIP]
> Both `SessionStore` and `Environment` are part of the **engine core** (`cas_engine`), not tied to the CLI. This means GUI and web interfaces can use the same session functionality.

---

## Session Store (`#id` References)

Every expression or equation you evaluate is automatically stored with a unique **Entry ID** (`#1`, `#2`, etc.). You can reference these IDs in subsequent expressions.

### Basic Usage

```text
> x + 1
Result: x + 1
#1: x + 1

> #1 * 2
Result: 2 * x + 2
#2: 2 * x + 2

> #1 + #2
Result: 3 * x + 3
#3: 3 * x + 3
```

### Equations as Expressions

When you reference an **equation** (`lhs = rhs`) as part of an expression, it is automatically converted to its **residue form**: `(lhs - rhs)`.

```text
> x + 1 = 5
#1: x + 1 = 5  [Eq]

> #1
Result: (x + 1) - 5
```

This is useful for systems of equations where you want to manipulate residuals.

### Chained References

References can be **chained** — if `#2` contains `#1`, resolving `#2` will also resolve `#1`:

```text
> x
#1: x

> #1 + 1
#2: x + 1

> #2 * 2
Result: 2 * x + 2   (resolved: (x + 1) * 2)
```

### Cycle Detection

The engine detects **circular references** and prevents infinite loops:

```text
> #1        (if #1 contains #1)
Error: Circular reference detected involving #1
```

### ID Stability

- IDs are **auto-incrementing** and **never reused**, even after deletion.
- If you delete `#3`, the next entry will still be `#4`, not `#3`.

---

## Session Reference Caching (V2.15.36)

When you reference a previous result with `#N`, the engine uses **cached simplified results** instead of re-running the full simplification pipeline. This provides significant performance improvements and cleaner timelines.

### How It Works

1. **Automatic Caching**: After each evaluation, the simplified result is cached.
2. **Cache Hit**: When `#N` is referenced, the cached result replaces the reference.
3. **Synthetic Step**: Timeline shows `"Used cached simplified result from #1, #3"` instead of repeating derivation.

### Timeline Traceability

When `#N` is resolved from cache, the timeline shows a **synthetic step** instead of repeating the full derivation:

```text
> sin(x)^2 + cos(x)^2
#1: 1
Steps:
1. sin²(x) + cos²(x) = 1  [Pythagorean Chain Identity]
Result: 1

> #1 + 5
#2: 6
Steps:
1. Used cached simplified result from #1  [Use cached result]
   Before: 1 + 5
   Rule: #1 + 5 -> 1 + 5
Result: 6
```

**Multiple references** are listed together:

```text
> x^2 + 2*x + 1
#1: x² + 2·x + 1

> (x+1)^2
#2: (x + 1)²

> #1 + #2 + 10
Steps:
1. Used cached simplified result from #1, #2  [Use cached result]
   ...
Result: 2·x² + 4·x + 12
```

### Cache Invalidation

The cache is automatically invalidated when **domain settings change**:

| Setting Change | Cache Behavior |
|----------------|----------------|
| Same domain mode | Cache hit (fast) |
| Different domain | Cache miss (re-simplify) |

### Memory Management

The cache has built-in **LRU eviction** to prevent memory bloat:

| Setting | Default | Description |
|---------|---------|-------------|
| Max cached entries | 100 | Oldest entries evicted first |
| Max total steps | 5000 | Steps budget across all cached entries |
| Light-cache threshold | 200 | Entries with > 200 steps store only result (no steps) |

> [!TIP]
> The cache is transparent — you don't need to do anything special. It just makes `#N` references faster!

---

## Steps Mode (Performance & Display Control)

The `steps` command controls how simplification steps are recorded and displayed. This affects both performance and output verbosity.

### Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `on` (default) | Full step recording with before/after snapshots | Educational display, debugging |
| `compact` | Record steps without snapshots | Memory-efficient step tracking |
| `off` | No step recording (fastest) | Batch processing, performance optimization |

### Commands

```text
> set steps              # Show current mode
Current steps mode: on
  (use 'set steps on|off|compact' to change)

> set steps off          # Disable step recording
Steps mode: off
  ⚡ Steps disabled (faster). Warnings still enabled.

> set steps compact      # Compact mode (no snapshots)
Steps mode: compact
  Compact steps (no before/after snapshots).

> set steps on           # Full step recording
Steps mode: on
  Full step recording with before/after snapshots.
```

### Prompt Indicator

When steps mode is not `on` (default), the prompt shows the current mode:

```text
> x + 1                           # Default: no indicator

> set steps off
[steps:off] > x + 1               # Indicator visible

> context integrate
[steps:off][ctx:integrate] > x    # Multiple indicators
```

### Domain Warnings Survival

Even with `steps off`, **domain warnings** are preserved. This ensures mathematical safety information (like "assuming sin(u) ≠ 0") is never silently dropped.

```rust
// In code:
let (result, steps) = simplifier.simplify(expr);
let warnings = simplifier.take_domain_warnings(); // Works in all modes
```

### Performance

Benchmarks show `steps off` provides modest performance gains:

| Workload | Steps On | Steps Off | Improvement |
|----------|----------|-----------|-------------|
| Batch (11 inputs) | 1.38 ms | 1.25 ms | ~9% faster |
| Light (i^12345) | 14.6 µs | 13.6 µs | ~7% faster |
| Heavy (multivar GCD) | 7.67 ms | 7.58 ms | ~1% faster |

> [!TIP]
> Use `steps off` for batch processing or when steps aren't needed. The prompt indicator helps you remember the current mode.

---

## Session Commands

| Command | Description |
|---------|-------------|
| `history` or `list` | Show all stored entries with their IDs |
| `show #N` | Display details of entry `#N` (type, raw input, expression) |
| `del #N [#M ...]` | Delete entries by ID (IDs are never reused) |
| `mode [strict\|principal]` | Show or switch simplification mode (strict = safe, principal = educational) |
| `reset` | Clear **all** session state (history + variables) |

### Example: `show #N`

```text
> show #1
Entry #1:
  Type: Expr
  Raw:  x + 1
  Expr: x + 1
```

For equations:
```text
> show #2
Entry #2:
  Type: Eq
  Raw:  x + 1 = 5
  LHS:  x + 1
  RHS:  5
```

---

## Environment (Variables)

The **Environment** allows you to bind expressions to variable names. These bindings are substituted into expressions automatically.

### Assignment Syntax

```text
> let a = 5
> let f = x^2 + 1

> b := a + 10        (alternative syntax)
```

### Transitive Substitution

Variable substitution is **transitive** — chains are fully resolved:

```text
> let b = 3
> let a = b + 1

> a * 2
Result: 8           (a → b+1 → 3+1 = 4, then 4*2 = 8)
```

### Cycle Detection

Recursive definitions are detected and handled safely:

```text
> let x = x + 1

> x * 2
Result: (x + 1) * 2   (x is NOT infinitely expanded)
```

The engine detects the cycle and stops substitution, leaving the variable as-is.

### Shadowing

Some commands temporarily **shadow** variables to prevent substitution:

```text
> let x = 5
> diff(x^2, x)
Result: 2 * x        (x treated as symbol, not 5)
```

This ensures that differentiation variables, solve targets, etc., work correctly even when bound.

### Reserved Names

The following names **cannot** be assigned:

| Category | Examples |
|----------|----------|
| Keywords | `let`, `vars`, `clear`, `solve`, `simplify`, `expand`, `diff`, `integrate` |
| Functions | `sin`, `cos`, `tan`, `log`, `ln`, `sqrt`, `abs`, `exp`, `gcd`, `lcm` |
| Constants | `pi`, `e`, `i`, `inf`, `infinity`, `undefined` |

Attempting to assign a reserved name will show an error.

---

## Environment Commands

| Command | Description |
|---------|-------------|
| `vars` | List all defined variables and their values |
| `clear` | Remove **all** variable bindings |
| `clear <name> [name ...]` | Remove specific variable bindings |
| `reset` | Clear entire session (variables **and** history) |

### Example: `vars`

```text
> let a = 5
> let b = a + 1
> vars
Variables:
  a = 5
  b = a + 1
```

---

## Advanced Usage

### Solving with References

You can use session IDs directly in `solve` commands. This is powerful for solving equations you've just built or manipulated.

```text
> x + 1 = 5
#1: x + 1 = 5

> solve #1, x
Result: x = 4
```

### Equivalence Checking

You can verify if two session entries are mathematically equivalent using `equiv`. The command returns:
- **True** — Expressions are unconditionally equivalent
- **True (conditional)** — Expressions are equivalent under domain conditions (shown as "Requires")
- **False** — Expressions are demonstrably not equivalent
- **Unknown** — Cannot determine equivalence

```text
> (x+1)^2
#1: (x+1)^2

> x^2 + 2x + 1
#2: x^2 + 2x + 1

> equiv #1, #2
Result: True
```

**Conditional Equivalence Examples:**

```text
> equiv (x - 27) / (x^(2/3) + 3*x^(1/3) + 9), x^(1/3) - 3
True (conditional)
ℹ️ Requires:
  • 9 + x^(2/3) + 3·x^(1/3) ≠ 0

> equiv tan(x) * tan(pi/3 - x) * tan(pi/3 + x), tan(3*x)
True (conditional)
ℹ️ Requires:
  • cos(x) ≠ 0
  • cos(x + pi / 3) ≠ 0
  • cos(pi / 3 - x) ≠ 0

> equiv (x-1)/(x-1), 1
True (conditional)
ℹ️ Requires:
  • x - 1 ≠ 0
```

---

## Architecture (for Developers)

The session system is designed for **portability** — the CLI is just one consumer. The `SessionState` struct bundles the store and environment for easy management.

### Engine Modules

| Module | Location | Purpose |
|--------|----------|---------|
| `SessionState` | `cas_engine/src/session_state.rs` | Bundles `SessionStore` + `Environment` |
| `Engine` | `cas_engine/src/eval.rs` | High-level evaluation pipeline |
| `SessionStore` | `cas_engine/src/session.rs` | Stores expressions with `#id` |
| `Environment` | `cas_engine/src/env.rs` | Stores variable bindings |
| `resolve_all()` | `cas_engine/src/session_state.rs` | Resolves both `#id` and variables |
| `substitute()` | `cas_engine/src/env.rs` | Substitutes environment variables |

### Public API

```rust
use cas_engine::eval::{Engine, EvalRequest, EvalAction};
use cas_engine::session_state::SessionState;

// Initialize
let mut engine = Engine::new();
let mut state = SessionState::new();

// Eval request
let req = EvalRequest {
    expression: parsed_expr,
    action: EvalAction::Simplify, // or Solve, Equiv, etc.
    auto_store: true,
};

// Evaluate (handles storage, resolution, and logic)
let output = engine.eval(&mut state, req)?;
println!("Result: {}", output.result);
```

### Key Types

```rust
pub type EntryId = u64;

pub enum EntryKind {
    Expr(ExprId),
    Eq { lhs: ExprId, rhs: ExprId },
}

pub struct Entry {
    pub id: EntryId,
    pub kind: EntryKind,
    pub raw_text: String,
    pub diagnostics: Diagnostics,
    pub simplified: Option<SimplifiedCache>,  // V2.15.36
}

// V2.15.36: Cached simplified result
pub struct SimplifiedCache {
    pub key: SimplifyCacheKey,
    pub expr: ExprId,
    pub requires: Vec<RequiredItem>,
    pub steps: Option<Arc<Vec<Step>>>,  // None = light cache
}

// V2.15.36: Memory management settings
pub struct CacheConfig {
    pub max_cached_entries: usize,   // 0 = unlimited
    pub max_cached_steps: usize,     // 0 = unlimited
    pub light_cache_threshold: Option<usize>,
}
```

### Error Handling

```rust
pub enum ResolveError {
    NotFound(EntryId),          // #id doesn't exist
    CircularReference(EntryId), // cycle detected
}
```

---

## Best Practices

1. **Use `#id` for iterative work**: Build up complex expressions step by step.
2. **Use `let` for constants**: Avoid re-typing values; let the engine substitute.
3. **Combine both**: `let base = #1` to name a previous result.
4. **Use `reset` carefully**: It clears everything. Use `clear` for partial cleanup.
5. **Check `history`**: Before referencing `#id`, verify it exists.

---

## Logging & Debug Output

The CLI uses `tracing` for logging. By default, only warnings and errors are shown.

### Controlling Log Level

Use the `RUST_LOG` environment variable:

```bash
# No extra output (default)
cargo run -p cas_cli

# Show pipeline statistics (rewrites, iterations)
RUST_LOG=info cargo run -p cas_cli

# Show detailed debug info
RUST_LOG=debug cargo run -p cas_cli

# Very verbose (trace level)
RUST_LOG=trace cargo run -p cas_cli

# Disable all logs
RUST_LOG=off cargo run -p cas_cli
```

### Log Levels

| Level | What's Shown |
|-------|--------------|
| `error` | Critical errors only |
| `warn` | Warnings (**default**) |
| `info` | Pipeline stats (iterations, rewrites) |
| `debug` | Detailed debugging (rule applications) |
| `trace` | Everything (very verbose) |

> [!TIP]
> For normal use, keep the default (no `RUST_LOG`). Use `RUST_LOG=info` when debugging performance issues or investigating simplification behavior.
