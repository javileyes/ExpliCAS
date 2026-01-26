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
