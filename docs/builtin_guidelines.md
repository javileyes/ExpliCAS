# BuiltinFn Guidelines

`BuiltinFn` is the type-safe enumeration of known built-in functions.

## Quick Reference

| Need to... | Use |
|------------|-----|
| Create `__hold(expr)` | `wrap_hold(ctx, inner)` |
| Check if expr is `__hold` | `is_hold(ctx, id)` or `ctx.is_builtin(fn_id, BuiltinFn::Hold)` |
| Unwrap `__hold` | `unwrap_hold(ctx, id)` |
| Strip all holds recursively | `strip_all_holds(ctx, root)` |
| Compare function name string | `is_hold_name(name)` or `name == BuiltinFn::Hold.name()` |
| Get canonical name | `BuiltinFn::Hold.name()` → `"__hold"` |

## When to Use What

### With `fn_id` available (preferred)
```rust
// O(1) comparison via Context
if ctx.is_builtin(fn_id, BuiltinFn::Hold) { ... }

// Create builtin calls
ctx.call_builtin(BuiltinFn::Sqrt, vec![arg])
```

### With only `&str` (compatibility paths)
```rust
// For functions receiving name: &str
if cas_ast::hold::is_hold_name(name) { ... }

// Get canonical name
let hold = cas_ast::hold::hold_name(); // "__hold"
```

## Where String Names Are Allowed

1. **Parser/CLI compatibility** - legacy input parsing
2. **Evaluation fallbacks** - when signature only provides `&str`
3. **Display/debug output** - user-facing function names

## Adding New Builtins

1. Add variant to `BuiltinFn` enum in `builtin.rs`
2. Add case to `BuiltinFn::name()` method
3. Update `ALL_BUILTINS` array and `COUNT`
4. Add helper functions if needed (like `hold.rs` for Hold)

---

## Internal Wrappers: `__hold`, `__eq__`, `poly_result`

These are **internal IR wrappers**, not user-facing functions:

### `__eq__` — Equation Wrapper

| Need to... | Use |
|------------|-----|
| Create `__eq__(lhs, rhs)` | `wrap_eq(ctx, lhs, rhs)` |
| Check if expr is `__eq__` | `is_eq(ctx, id)` or `ctx.is_builtin(fn_id, BuiltinFn::Eq)` |
| Unwrap `__eq__` | `unwrap_eq(ctx, id)` → `Option<(lhs, rhs)>` |
| Compare function name string | `is_eq_name(name)` |

**Important distinctions:**
- `__eq__`: Internal equation wrapper `lhs = rhs` (for solver residuals, parse trees)
- `Equal`: Symbolic equality comparison operator (semantic)

**Aridad canónica:** 2 (lhs, rhs)

**Display/LaTeX:** Rendered as equation `lhs = rhs` (never as function call)

**Never compare strings directly:** Use `eq::*` helpers or `ctx.is_builtin(.., BuiltinFn::Eq)`

### `__hold` vs `hold` — Simplification Barriers

Two related functions with different visibility:

| Function | Visibility | Display | Use Case |
|----------|------------|---------|----------|
| `__hold(expr)` | Internal | Transparent (hidden) | Rule protection (expand, factor) |
| `hold(expr)` | User-facing | Visible | Didactic, forcing unevaluated form |

**HoldAll Semantics:** Both `__hold` and `hold` prevent child simplification:
- `hold(x + 0)` → `hold(x + 0)` (child preserved)
- `hold(2 * 3)` → `hold(2 * 3)` (not folded to 6)

| Need to... | Use |
|------------|-----|
| Create `__hold(inner)` | `wrap_hold(ctx, inner)` |
| Check if internal __hold | `ctx.is_builtin(fn_id, BuiltinFn::Hold)` |
| Check if any hold | `is_hold(ctx, id)` |
| Unwrap ONLY internal __hold | `unwrap_internal_hold(ctx, id)` ← preserves user `hold()` |
| Unwrap any hold | `unwrap_hold(ctx, id)` |
| Strip all internal holds | `strip_all_holds(ctx, root)` |
| Check name string | `is_internal_hold_name(name)` or `is_hold_name(name)` |

**Output boundaries:** Use `unwrap_internal_hold` at `simplify()` exit to strip internal barriers while keeping user `hold()` visible.

See [Quick Reference](#quick-reference) for creation helpers.

### `poly_result` — Polynomial Store Reference

| Need to... | Use |
|------------|-----|
| Create `poly_result_{id}` | `wrap_poly_result(ctx, poly_id)` |
| Check if expr is poly_result | `is_poly_result(ctx, id)` |
| Extract poly ID | `parse_poly_result_id(ctx, id)` |

