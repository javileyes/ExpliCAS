# Project Maintenance Guide

This document provides a comprehensive overview of the project's architecture, debugging tools, and extension patterns to facilitate maintenance and future development.

## 1. Debug Logging System

**Status**: ✅ Uses `tracing` for professional structured logging (see [DEBUG_SYSTEM.md](DEBUG_SYSTEM.md) for full guide)

### Quick Start

**No code changes needed!** Control logging via environment variable:

```bash
# Enable debug logging for entire engine
RUST_LOG=cas_engine=debug cargo test

# Specific module only  
RUST_LOG=cas_engine::canonical_forms=debug cargo run -p cas_cli

# Very verbose (trace level)
RUST_LOG=cas_engine=trace ./target/release/cas_cli
```

### Log Levels

- `error` - Critical errors only
- `warn` - Warnings  
- `info` - General information
- `debug` - ⭐ **Recommended for development** - Detailed debugging info
- `trace` - Very verbose, every detail

### What's Logged

Currently instrumented modules:
- `canonical_forms.rs` - Conjugate detection, canonical form checks
- `engine.rs` - AST traversal (`transform_expr_recursive`)
- More modules can be added as needed

### Example Output

```bash
$ RUST_LOG=cas_engine=debug echo "simplify x^2 + 2x + 1" | cargo run -p cas_cli

# Shows detailed trace of expression simplification
```

### Adding Debug Logging

```rust
use tracing::debug;

pub fn my_function() {
    debug!("Checking value: {:?}", value);    // General debugging
    debug!("Processing expr: {:?}", expr);    // AST debugging
}
```

**Benefits over `eprintln!`**:
- ✅ Zero overhead when disabled (compiled out)
- ✅ Granular control per module  
- ✅ Professional industry standard
- ✅ Doesn't pollute benchmarks/tests
- ✅ Change log level without recompiling

**See [DEBUG_SYSTEM.md](DEBUG_SYSTEM.md)** for:
- Pattern Detection debugging
- Component-by-component debugging
- Common troubleshooting scenarios
- End-to-end trace guides

---

**Note**: `Simplifier.enable_debug()` method exists for backward compatibility but is deprecated. All debug logging now uses `tracing` controlled by `RUST_LOG`.

## 2. Architecture Overview

The project is organized as a workspace with several crates:

### `crates/cas_ast`
Defines the core data structures for symbolic expressions.
-   **`Expr`**: Enum representing expression nodes (Add, Mul, Var, etc.).
-   **`ExprId`**: Lightweight handle (index) to an expression node.
-   **`Context`**: Arena allocator that stores all `Expr` nodes. Passed around to manage memory and avoid lifetime issues.

### `crates/cas_engine`
Contains the core logic for simplification and solving.
-   **`Simplifier`**: The main entry point. Manages the `Context` and a collection of `Rule`s.
-   **`Rule` Trait**: Interface for simplification rules.
-   **`LocalSimplificationTransformer`**: Visits the expression tree recursively, applying rules bottom-up.
-   **`SolverStrategy` Trait**: Interface for equation solving strategies (e.g., `QuadraticStrategy`, `SubstitutionStrategy`).

### `crates/cas_parser`
Handles parsing of mathematical strings into AST nodes.
-   Uses a recursive descent parser.

### `crates/cas_format`
Handles formatting of AST nodes into strings (e.g., LaTeX, text).

### Pattern Detection Infrastructure ★ (Added 2025-12)

**New System** for context-aware rule application and Pythagorean identity simplification.

#### Components:

1. **`pattern_marks.rs`**: Lightweight `HashSet<ExprId>` to mark protected expressions
2. **`pattern_scanner.rs`**: Pre-analysis scanner that detects patterns before simplification
3. **`parent_context.rs`**: Context threading from parent to child during transformation
4. **`pattern_detection.rs`**: Helper functions (`is_sec_squared`, `is_tan_squared`, etc.)

#### Key Files:
- **Core**: `crates/cas_engine/src/{pattern_marks.rs, pattern_scanner.rs, parent_context.rs, pattern_detection.rs}`
- **Engine Integration**: `crates/cas_engine/src/{engine.rs, orchestrator.rs}`
- **Rules**: `crates/cas_engine/src/rules/trigonometry.rs` (SecTanPythagoreanRule, CscCotPythagoreanRule, TanToSinCosRule with guard)
- **Tests**: `crates/cas_cli/tests/{pythagorean_variants_test.rs, debug_sec_tan.rs}`

#### Data Flow:
```
Orchestrator::simplify()
  → PatternScanner::scan_and_mark_patterns() [O(n) pre-analysis]
  → Creates PatternMarks
  → Simplifier::apply_rules_loop(expr, &pattern_marks)
  → LocalSimplificationTransformer { initial_parent_ctx: ParentContext::with_marks(marks) }
  → Rules receive parent_ctx with pattern_marks
  → Guards skip premature conversions
  → Direct rules apply identities
```

## 2.5. Pattern Detection System - Maintenance Guide ★

### Debugging Pattern Detection

#### Enable Pattern Scanner Logging

Currently, pattern detection runs silently. To debug:

1. **Add temporary debug output** in `pattern_scanner.rs`:
   ```rust
   pub fn scan_and_mark_patterns(ctx: &Context, expr_id: ExprId, marks: &mut PatternMarks) {
       eprintln!("[PATTERN] Scanning: {:?}", ctx.get(expr_id));
       // ... existing code ...
       
       if is_pythagorean_difference(ctx, *left, *right) {
           eprintln!("[PATTERN] ✓ Found Pythagorean pattern!");
           // ...
       }
   }
   ```

2. **Check what gets marked**:
   ```rust
   // In orchestrator.rs after scanning
   eprintln!("[PATTERN] Marked {} expressions", pattern_marks.protected.len());
   ```

3. **Verify guards are firing**:
   ```rust
   // In TanToSinCosRule::apply
   if let Some(marks) = parent_ctx.pattern_marks() {
       if marks.is_pythagorean_protected(expr) {
           eprintln!("[GUARD] Skipping tan→sin/cos conversion for protected expr");
           return None;
       }
   }
   ```

#### Common Issues and Solutions

**Problem**: Pythagorean identity not simplifying to 1

**Debugging Steps**:
1. Check if pattern scanner is finding the pattern:
   ```bash
   # Add debug output in scan_and_mark_patterns
   echo "sec(x)^2 - tan(x)^2" | cargo run -p cas_cli 2>&1 | grep PATTERN
   ```

2. Verify AST structure (CRITICAL):
   ```rust
   // Remember: a - b is ALWAYS Add(a, Neg(b)), NOT Sub(a, b)
   // In SecTanPythagoreanRule, must match:
   if let Expr::Add(left, right) = expr_data {
       if let Expr::Neg(neg_inner) = ctx.get(neg) {
           // This is the subtraction!
       }
   }
   ```

3. Check if rule is registered:
   ```rust
   // In crates/cas_engine/src/rules/trigonometry.rs::register
   simplifier.add_rule(Box::new(SecTanPythagoreanRule));
   simplifier.add_rule(Box::new(CscCotPythagoreanRule));
   ```

4. Verify `pattern_marks` threading:
   ```bash
   # Check that pattern_marks reaches rules
   # Add debug in LocalSimplificationTransformer::apply_rules
   cargo test --test pythagorean_variants_test -- --nocapture
   ```

**Problem**: Guard not preventing conversion

**Solution**: Ensure `ParentContext` is being extended correctly:
```rust
// In engine.rs, when recursing to children
let child_ctx = parent_ctx.extend(current_expr);
self.apply_rules(rewritten, &child_ctx);  // Pass extended context!
```

### Testing Pattern Detection

#### Running Pattern Detection Tests

```bash
# All pattern detection tests
cargo test -p cas_engine pattern

# Pythagorean variant tests
cargo test --test pythagorean_variants_test

# Debug test with output
cargo test --test debug_sec_tan -- --nocapture
```

#### Test Structure

Tests are organized by concern:

1. **Unit Tests** (`pattern_scanner.rs`):
   - `test_scan_sec_tan_difference` - Basic pattern detection
   - `test_scan_csc_cot_difference` - Csc/cot variant
   - `test_scan_nested_patterns` - Nested expressions
   
2. **Integration Tests** (`pythagorean_variants_test.rs`):
   - `test_sec_tan_equals_one` - Full simplification to 1
   - `test_sec_tan_minus_one_equals_zero` - Alternative form
   - `test_csc_cot_minus_one_equals_zero` - Csc/cot identity

3. **Debug Tests** (`debug_sec_tan.rs`):
   - Manual `pattern_marks` construction
   - Step-by-step verification

#### Adding New Pattern Tests

```rust
#[test]
fn test_my_new_pattern() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "my_pattern_expression");
    let mut marks = PatternMarks::new();
    
    scan_and_mark_patterns(&ctx, expr, &mut marks);
    
    // Verify detection
    assert!(marks.is_protected(expected_expr_id));
    
    // Verify simplification
    let mut simplifier = Simplifier::with_default_rules();
    let (result, _) = simplifier.simplify(expr);
    assert_eq!(format_expr(&ctx, result), "expected_result");
}
```

### Extending Pattern Detection

#### Adding a New Pattern Family

**Example**: Adding sum-to-product trig identities protection

1. **Add pattern detection** in `pattern_scanner.rs`:
   ```rust
   fn is_sum_to_product_pattern(ctx: &Context, left: ExprId, right: ExprId) -> bool {
       // Detect: sin(A) + sin(B) or cos(A) + cos(B)
       matches!(
           (ctx.get(left), ctx.get(right)),
           (Expr::Function(name1, _), Expr::Function(name2, _))
           if (name1 == "sin" && name2 == "sin") || (name1 == "cos" && name2 == "cos")
       )
   }
   
   pub fn scan_and_mark_patterns(...) {
       match ctx.get(expr_id) {
           Expr::Add(left, right) => {
               // Existing Pythagorean check...
               
               // NEW: Sum-to-product check
               if is_sum_to_product_pattern(ctx, *left, *right) {
                   marks.mark_protected(*left);
                   marks.mark_protected(*right);
               }
           }
           // ...
       }
   }
   ```

2. **Add helper** in `pattern_detection.rs`:
   ```rust
   pub fn is_sin_function(ctx: &Context, expr: ExprId) -> Option<ExprId> {
       if let Expr::Function(name, args) = ctx.get(expr) {
           if name == "sin" && args.len() == 1 {
               return Some(args[0]);
           }
       }
       None
   }
   ```

3. **Add direct rule** (optional) in `trigonometry.rs`:
   ```rust
   define_rule!(
       SumToProductRule,
       "Sum-to-Product Identity",
       |ctx, expr| {
           // sin(A) + sin(B) = 2*sin((A+B)/2)*cos((A-B)/2)
           // Check pattern and apply transformation
       }
   );
   ```

4. **Add guard** in existing expansion rule:
   ```rust
   // In SinExpansionRule or similar
   fn apply(&self, ctx: &mut Context, expr: ExprId, parent_ctx: &ParentContext) -> Option<Rewrite> {
       if let Some(marks) = parent_ctx.pattern_marks() {
           if marks.is_sum_to_product_protected(expr) {
               return None; // Skip expansion
           }
       }
       // ... normal logic
   }
   ```

5. **Add tests**:
   ```rust
   #[test]
   fn test_sum_to_product_detection() {
       let expr = parse("sin(x) + sin(y)");
       // ... verify detection and simplification
   }
   ```

#### Pattern Mark Categories

Consider adding categories for different pattern types:

```rust
pub struct PatternMarks {
    pythagorean_protected: HashSet<ExprId>,
    sum_to_product_protected: HashSet<ExprId>,
    // ... other categories
}

impl PatternMarks {
    pub fn mark_pythagorean(&mut self, expr: ExprId) {
        self.pythagorean_protected.insert(expr);
    }
    
    pub fn is_pythagorean_protected(&self, expr: ExprId) -> bool {
        self.pythagorean_protected.contains(&expr)
    }
    
    // ... methods for other categories
}
```

### Performance Considerations

#### Pattern Scanner Cost

- **Complexity**: O(n) where n = AST nodes
- **Frequency**: Once per `Orchestrator::simplify()` call
- **Overhead**: ~5-10µs for typical expressions (100-500 nodes)

#### Optimization Tips

1. **Early exit in scanner**:
   ```rust
   // Only scan relevant subtrees
   if !is_trig_related(ctx, expr_id) {
       return; // Skip non-trig expressions
   }
   ```

2. **Lazy pattern marks**:
   ```rust
   // Don't clone pattern_marks unless needed
   pub fn extend(&self, parent_id: ExprId) -> Self {
       Self {
           ancestors: new_ancestors,
           pattern_marks: self.pattern_marks.clone(), // Only happens on extend
       }
   }
   ```

3. **Benchmark impact**:
   ```bash
   # Before adding new pattern
   cargo bench -p cas_engine -- simplify
   
   # After adding new pattern
   cargo bench -p cas_engine -- simplify
   
   # Compare results in target/criterion/*/report/index.html
   ```

### Troubleshooting Checklist

When pattern detection isn't working:

- [ ] Pattern scanner finds the pattern (add debug output)
- [ ] Expression is marked in `PatternMarks`
- [ ] `ParentContext` is created with marks in `apply_rules_loop`
- [ ] `ParentContext` is extended correctly when recursing
- [ ] Rule receives `parent_ctx` parameter (check signature)
- [ ] Guard checks `parent_ctx.pattern_marks()` correctly
- [ ] AST structure matches expectation (a - b is Add+Neg!)
- [ ] Rule is registered in `register()` function
- [ ] Tests pass: `cargo test pattern`
-   [ ] Pattern scanner finds the pattern (add debug output)
-   [ ] Expression is marked in `PatternMarks`
-   [ ] `ParentContext` is created with marks in `apply_rules_loop`
-   [ ] `ParentContext` is extended correctly when recursing
-   [ ] Rule receives `parent_ctx` parameter (check signature)
-   [ ] Guard checks `parent_ctx.pattern_marks()` correctly
-   [ ] AST structure matches expectation (a - b is Add+Neg!)
-   [ ] Rule is registered in `register()` function
-   [ ] Tests pass: `cargo test pattern`

## 3. Extending the CAS

### Adding a New Simplification Rule

**UPDATED 2025-12**: Rules now receive `ParentContext` parameter for context-aware decisions.

1.  **Create the Rule Struct**: Define a struct that implements the `Rule` trait.

2.  **Implement `apply`** with new signature:
    ```rust
    use crate::parent_context::ParentContext;

    impl Rule for MyRule {
        fn name(&self) -> &str { "My Rule" }

        fn apply(
            &self,
            ctx: &mut Context,
            expr: ExprId,
            parent_ctx: &ParentContext  // ← NEW parameter
        ) -> Option<Rewrite> {
            // Option 1: Ignore parent_ctx if not needed
            let _ = parent_ctx;

            // Option 2: Use parent_ctx for guards
            if let Some(marks) = parent_ctx.pattern_marks() {
                if marks.is_protected(expr) {
                    return None;  // Skip transformation
                }
            }

            // Check pattern and return Some(Rewrite) if matched
            if let Expr::MyPattern(...) = ctx.get(expr) {
                return Some(Rewrite {
                    new_expr: transformed_expr,
                    description: "Description of transformation".to_string(),
                });
            }

            None
        }
    }
    ```

3.  **Register the Rule**: Add it to the appropriate `register` function
    ```rust
    // In crates/cas_engine/src/rules/my_module.rs
    pub fn register(simplifier: &mut Simplifier) {
        simplifier.add_rule(Box::new(MyRule));
    }

    // Then call from crates/cas_engine/src/rules/mod.rs
    pub fn register_all_rules(simplifier: &mut Simplifier) {
        // ... existing registrations ...
        my_module::register(simplifier);
    }
    ```

**Note**: The `parent_ctx` parameter can be safely ignored for rules that don't need context. The compiler will warn about unused parameters; use `let _ = parent_ctx;` to suppress.

### Adding a New Solver Strategy

1.  **Create the Strategy Struct**: Define a struct that implements the `SolverStrategy` trait.
2.  **Implement `apply`**:
    ```rust
    impl SolverStrategy for MyStrategy {
        fn apply(&self, eq: &Equation, var: &str, simplifier: &mut Simplifier) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
            // Return None if strategy doesn't apply
            // Return Some(Result) if it does
        }
    }
    ```
3.  **Register the Strategy**: Add it to the `strategies` vector in `crates/cas_engine/src/solver/mod.rs`.

## 4. Testing Strategy

### Unit Tests
Located in `src/lib.rs` or specific modules. Test individual rules and functions.

### Integration Tests
Located in `tests/`. Test end-to-end flows like `solve()` or `simplify()`.

### Property-Based Tests
Located in `tests/property_tests.rs`. Use `proptest` to generate random expressions and verify invariants (e.g., `simplify(simplify(x)) == simplify(x)`).

### Running Tests

```bash
cargo test
```

### Ejecutar el CLI

#### Opción 1: Compilar release y usar el binario directamente
##### clean si es necesario
cargo clean
##### build release
cargo build --release
##### ejecutar
echo "simplify ((x+1)*(x-1))^2" | target/release/cas_cli | grep -c "^[0-9]"

#### Opción 2: Ejecutar con cargo run --release
echo "simplify ((x+1)*(x-1))^2" | cargo run --release | grep -c "^[0-9]"

## 5. Benchmarks

The project uses `criterion` for benchmarking performance, specifically for simplification rules.

### Running Benchmarks

```bash
cargo bench -p cas_engine
```

### Benchmark Locations

Benchmarks are located in `crates/cas_engine/benches/`.

-   `simplification_bench.rs`: Benchmarks for polynomial and trigonometric simplification.

### Adding New Benchmarks

1.  Create a new function in `crates/cas_engine/benches/simplification_bench.rs` (or a new file).
2.  Use `c.benchmark_group("group_name")` to group related benchmarks.
3.  Use `black_box` to prevent compiler optimizations from removing the code being measured.
4.  Register the new benchmark group with `criterion_group!` and `criterion_main!`.

**Important**: When optimizing code, always run benchmarks before and after changes to verify improvements and ensure no regressions.

### Baseline Results (as of 2025-12-01)

-   `polynomial/expand_binomial_power_10`: ~48 µs
-   `polynomial/combine_like_terms_large`: ~170 µs
-   `trigonometry/pythagorean_identity_nested`: ~10 ms

