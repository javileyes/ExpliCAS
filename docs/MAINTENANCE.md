# Project Maintenance Guide

This document provides a comprehensive overview of the project's architecture, debugging tools, and extension patterns to facilitate maintenance and future development.

## 1. Debug Logging System

**Status**: ✅ Uses `tracing` for professional structured logging (see [DEBUG_SYSTEM.md](DEBUG_SYSTEM.md) for full guide)

### Quick Start

**No code changes needed!** Control logging via environment variable:

```bash
# Enable info logging (engine pipeline stats)
RUST_LOG=info cargo run -p cas_cli

# Enable debug logging for entire engine
RUST_LOG=cas_engine=debug cargo test

# Specific module only  
RUST_LOG=cas_engine::canonical_forms=debug cargo run -p cas_cli

# Very verbose (trace level)
RUST_LOG=cas_engine=trace ./target/release/cas_cli
```

> **Note**: By default, log level is `warn` — no info/debug messages are shown.
> This keeps the CLI output clean. Use `RUST_LOG` to enable verbose output.

### Log Levels

- `error` - Critical errors only  
- `warn` - Warnings (**default level**)
- `info` - Pipeline statistics (rewrites, iterations)
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

### Canonical Utilities ⚠️ (IMPORTANT)

**If you need flatten/predicates/builders/traversal, use the canonical implementation. The CI will fail if you duplicate them.**

| Category | Canonical Location | Don't Redefine |
|----------|-------------------|----------------|
| **hold** | `cas_ast::hold` | `strip_hold`, `unwrap_hold` |
| **flatten** | `cas_engine::nary` | `flatten_add`, `flatten_mul` |
| **predicates** | `cas_engine::helpers` | `is_zero`, `is_one`, `is_negative`, `get_integer` |
| **builders** | `cas_ast::views::MulBuilder`, `Context::build_balanced_mul` | `build_mul_from_factors`, `build_balanced_mul` |
| **traversal** | `cas_ast::traversal` | `count_nodes`, `count_nodes_matching` |

Run `make audit-utils` to see the full registry and verify compliance.

See [POLICY.md](POLICY.md) for detailed contracts.

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

### ⚠️ "Sub is NOT stable" — Critical Hazard for Rule Authors

`CanonicalizeNegationRule` converts `Sub(a, b) → Add(a, Neg(b))` **very early** in the
simplification pipeline. This means:

- **Any rule that pattern-matches on `Expr::Sub` nodes will never fire** in the default
  configuration because `Sub` nodes are rewritten to `Add(…, Neg(…))` before subsequent
  rules see them.

**If your rule needs to detect subtraction, choose ONE of these options:**

| Option | When to Use | Example |
|--------|-------------|---------|
| **(a)** Register BEFORE `canonicalization` | Exotic — rarely needed | — |
| **(b)** Also match `Add(x, Neg(y))` | When rule is a single-expression rewrite | Pattern detection rules |
| **(c)** Equation-level operation in solver | When operation is relational (LHS↔RHS) | `cancel_common_additive_terms` |

**Reference**: See `rules/cancel_common_terms.rs` for the canonical example of option (c),
and `engine/simplifier.rs` `register_default_rules()` for the order contract.

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

## 6. Stress Testing & Rule Orchestration Analysis

### Overview

The project includes a **stress testing infrastructure** to identify performance bottlenecks and potential stack overflows in the simplification engine. This is particularly useful for:

1. Finding rules that trigger excessive re-simplification loops
2. Identifying expression patterns that cause exponential rule application
3. Debugging stack overflow triggers in the recursive simplifier

### Test Profiles

Located in `crates/cas_engine/tests/strategies/mod.rs`:

| Profile  | Depth | Size | Items | Use Case |
|----------|-------|------|-------|----------|
| **SAFE**    | 2  | 8   | 4  | CI/CD, never stack overflows |
| **NORMAL**  | 3  | 15  | 6  | Development, balanced |
| **STRESS**  | 4  | 20  | 10 | Original values, stresses engine |
| **EXTREME** | 5  | 30  | 15 | Deep debugging of specific bottlenecks |

**Parameters**:
- `Depth`: Maximum recursion depth (2^DEPTH possible nesting levels)
- `Size`: Maximum number of nodes in expression tree
- `Items`: Maximum items per collection (affects function arguments)

### Running Stress Tests

Use the `STRESS_PROFILE` environment variable to select complexity profile:

```bash
# Run with SAFE profile (default, never stack overflows)
cargo test --package cas_engine --test stress_test -- --nocapture

# Run with STRESS profile (requires larger stack)
STRESS_PROFILE=STRESS RUST_MIN_STACK=16777216 cargo test --package cas_engine --test stress_test -- --nocapture

# Run batch test (100 cases) to find overflows
cargo test --package cas_engine --test stress_test test_batch_overflow_finder -- --ignored --nocapture
```

**Valid STRESS_PROFILE values**: `SAFE` | `NORMAL` | `STRESS` | `EXTREME`

### Test Summary Output

After running stress tests, you'll see a comprehensive summary:

```
═══════════════════════════════════════════════════════════════════
                    STRESS TEST SUMMARY
═══════════════════════════════════════════════════════════════════
Profile: STRESS (depth=4, size=20, items=10)

📊 TEST RESULTS:
   Passed:       20 / 20 (100.0%)
   ⚠️  Overflows:    2              ← Only shown if there are overflows

📈 RULE STATISTICS:
   Total simplifications:     20
   Total rule applications:   47
   Average rules/expression:  2.4

   Top 10 Rules:
      Combine Constants                          15 ( 31.9%)
      Canonicalize Addition                      12 ( 25.5%)
      Pythagorean Identity                        8 ( 17.0%)
      Evaluate Numeric Power                      5 ( 10.6%)
      ...

🔥 MOST EXPENSIVE EXPRESSION:
   Steps: 23
   Expr:  sin(x)^8 - cos(x)^8

⚠️  STACK OVERFLOW EXPRESSIONS (2):      ← Only if overflows detected
   These expressions caused stack overflow and need investigation:
   1. (sin(x)^2 + cos(x)^2)^10 / ((tan(x) + cot(x))^5)
   2. ln(exp(sin(cos(tan(x)^2)^3)^4))

   💡 TIP: Copy these expressions to test_stress_single() for debugging.
═══════════════════════════════════════════════════════════════════
```

### Debugging Overflow Expressions

When the stress test finds expressions that cause stack overflow, follow this workflow:

1. **Copy the problematic expression** from the summary
2. **Edit `test_stress_single()`** in `stress_test.rs` to build that expression:

```rust
#[test]
fn test_stress_single() {
    let mut ctx = Context::new();
    
    // Build the problematic expression from the overflow report
    let x = ctx.var("x");
    let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![x]));
    let cos_x = ctx.add(Expr::Function("cos".to_string(), vec![x]));
    // ... reconstruct the expression
    
    // Run with profiler to see which rules are firing
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.context = ctx;
    simplifier.profiler = RuleProfiler::new(true);
    
    let (result, steps) = simplifier.simplify(expr);
    
    // See step-by-step trace and profiler report
    eprintln!("{}", simplifier.profiler.report());
}
```

3. **Run the single test** with debug logging:
```bash
RUST_LOG=debug cargo test test_stress_single -- --nocapture
```

### Stack Overflow Prevention

When running with higher profiles (STRESS, EXTREME), increase stack size:

```bash
# 8MB stack (minimum for STRESS)
RUST_MIN_STACK=8388608 cargo test

# 16MB stack (recommended for EXTREME)
RUST_MIN_STACK=16777216 cargo test
```

### Key Files

| File | Purpose |
|------|---------|
| `strategies/mod.rs` | Test profiles (SAFE/NORMAL/STRESS/EXTREME) and expression generators |
| `stress_test.rs` | Stress tests with overflow detection and profiler integration |
| `property_tests.rs` | Property-based tests (idempotency, etc.) |

## 7. Health Instrumentation ★★★ (Added 2025-12)

### Overview

The engine includes a **health instrumentation system** to detect churn (excessive rewrites), hot rules, and node growth. This is critical for preventing performance regressions.

### Key Components

| Component | Purpose |
|-----------|---------|
| `RuleProfiler` | Tracks applied/rejected counts and node delta **per phase** |
| `CycleDetector` | Detects ping-pong patterns using fingerprint ring buffer |
| `PipelineStats` | Aggregates phase-level statistics including cycle info |
| `health_smoke_tests.rs` | CI tests with thresholds to catch regressions |

### Per-Phase RuleProfiler

Stats are tracked separately for each phase (Core, Transform, Rationalize, PostCleanup):

```rust
pub struct RuleProfiler {
    per_phase: [HashMap<String, RuleStats>; 4],
    enabled: bool,
    health_enabled: bool,
}

// Query specific phase
profiler.top_applied_for_phase(SimplifyPhase::Transform, 3)
profiler.health_report_for_phase(Some(SimplifyPhase::Transform))

// Aggregate across all phases
profiler.health_report()  // backward compatible
```

### Cycle Detection

The engine automatically detects "ping-pong" patterns where rules undo each other:

```rust
pub struct CycleDetector {
    buffer: [u64; 64],      // Ring buffer of fingerprints
    phase: SimplifyPhase,
    max_period: usize,      // Check periods 1-8
}
```

**When cycle is detected:**
- Phase exits immediately (treated as fixed-point)
- `PhaseStats.cycle` is populated with `CycleInfo`
- REPL shows: `⚠ Cycle detected: period=2 at rewrite=37`
- Top contributing rules for that phase are shown

### Enabling Health Metrics

```rust
// Enable health tracking (by default: zero overhead)
simplifier.profiler.enable_health();
simplifier.profiler.clear_run();  // Reset for this run

// After simplification
let (result, steps, stats) = simplifier.simplify_with_stats(expr, opts);

// Check for cycles in a phase
if let Some(cycle) = &stats.transform.cycle {
    println!("Cycle detected: period={}", cycle.period);
}

// Print per-phase report
println!("{}", simplifier.profiler.health_report_for_phase(Some(SimplifyPhase::Transform)));
```

### REPL Commands

| Command | Effect |
|---------|--------|
| `set debug on` | Show pipeline diagnostics + health report inline |
| `health on` | Enable background health tracking |
| `health` | Display last health report with cycle info |
| `health reset` | Clear accumulated statistics |

### Health Smoke Tests

Located in `crates/cas_engine/tests/health_smoke_tests.rs`:

| Test | Expression | max_rewrites | max_growth | max_transform |
|------|-----------|--------------|------------|---------------|
| mixed_expression | `x/(1+sqrt(2)) + 2*(y+3)` | 150 | 250 | 80 |
| polynomial | `(x+1)*(x+2)` | 100 | 150 | 60 |
| rationalization_only | `1/(3-2*sqrt(5))` | 100 | 200 | 40 |
| simple_no_op | `x + y` | 20 | 30 | 10 |

### Interpreting Health Issues

| Symptom | Likely Cause | Action |
|---------|--------------|--------|
| High `total_rewrites` | Churn (rules undo each other) | Check top applied rules for A↔B patterns |
| `Cycle detected` | Ping-pong between rules | Look at "Likely contributors" for the phase |
| High `rejected_semantic` | Expensive equality checks | Consider caching or reducing matcher scope |
| High `total_positive_growth` | Expansion-heavy rules | Add growth limits or progress gates |
| Budget saturation | Possible loops | Enable health to find cycling rules |

### Guidelines for Adding New Rules

1. **Determine correct phase:**
   - Core: Safe, non-growing transformations
   - Transform: Distribution, expansion (may grow)
   - Rationalize: Denominator rationalization
   - PostCleanup: Final cleanup (same as Core)

2. **Avoid churn:**
   - Use builders (`mul_many_simpl`, `add_many_simpl`) that simplify during construction
   - Prefer Views for pattern matching (read-only, no intermediate nodes)
   - Never re-expand after rationalization

3. **Add tests:**
   - Unit test: rule produces correct output
   - Idempotency test: applying twice gives same result
   - Health smoke test (for complex rules): verify thresholds

```rust
#[test]
fn health_smoke_my_new_feature() {
    run_health_check(
        "my_expression",
        100,  // max_rewrites (start conservative)
        150,  // max_growth
        50,   // max_transform_rewrites
    );
}
```

### CI Protection

Health smoke tests run on every CI build. If a test fails:

1. Check the failing expression in the error output
2. Look at `Transform rewrites` and `Top Applied Rules`
3. If intentional improvement, update thresholds with a comment
4. If regression, investigate the top rules for churn

## 8. CI System and Zero-Warning Policy ★ (Added 2025-12)

### Overview

The project follows a **Zero-Warning CI Policy** with all warnings treated as errors (`-D warnings`). There are **no crate-level `#![allow]`** suppressions in source code.

### Running CI

```bash
# Full CI suite (recommended)
make ci                    # fmt + lints + clippy + tests + build --release

# Variants
make ci-release            # + test --release
make ci-quick              # skip clippy (faster)
make lint                  # fmt + lints + clippy only
make test                  # tests only
```

### CI Pipeline (`scripts/ci.sh`)

The CI pipeline runs these steps in order:

| Step | Command | Description |
|------|---------|-------------|
| 1. Format | `cargo fmt --check` | Enforce consistent formatting |
| 2. Lints | `scripts/lint_*.sh` | Auto-discovered custom lint scripts |
| 3. Clippy | `cargo clippy -D warnings` | All warnings as errors |
| 4. Tests | `cargo test` | Full test suite |
| 5. Release | `cargo build --release` | Verify release build |

### Custom Lint Scripts

Lint scripts are **auto-discovered** by pattern `scripts/lint_*.sh`:

```bash
# Current scripts
scripts/lint_nary_shape_independence.sh   # Enforce AddView usage
scripts/lint_no_raw_mul.sh                # Prevent raw Mul construction

# Add new lint: just create script and make executable
chmod +x scripts/lint_my_check.sh         # Auto-runs in next CI
```

### Clippy Allow Policy

**Crate-Level `#![allow]`**:
- **Prohibited** in `src/` code (requires explicit approval)
- Currently: **0** crate-level allows in `cas_engine` and `cas_cli`
- Test files may have allows for `format_in_format_args`, `dead_code`, etc.

**Local `#[allow]`**:
- Allowed on specific items with justifying comment
- Format: `#[allow(clippy::lint_name)] // why this is necessary`

**Current Local Exceptions** (7 total in `cas_engine`):

| Lint | Location | Reason |
|------|----------|--------|
| `arc_with_non_send_sync` | profile_cache.rs ×2 | Arc for shared ownership, not threading |
| `too_many_arguments` | gcd_zippel_modp.rs ×2, inverse_trig.rs, step.rs | Math algorithms with distinct params |
| `never_loop` | engine.rs | Intentional loop structure with early returns |

### Tracking Technical Debt

```bash
# List all local #[allow] in source code
make lint-allowlist

# Output example:
==> Local #[allow(clippy::...)] in crates:
crates/cas_engine/src/profile_cache.rs:32:#[allow(clippy::arc_with_non_send_sync)]
...

==> Crate-level #![allow] (should be 0):
  ✓ None (clean)
```

### Adding New Lint Scripts

1. Create `scripts/lint_mycheck.sh`
2. Make executable: `chmod +x scripts/lint_mycheck.sh`
3. Script must exit 0 on success, non-zero on failure
4. CI auto-discovers and runs it

Example lint script structure:

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Running my custom check..."

# Your check logic here
if grep -r "bad_pattern" crates/; then
    echo "❌ Found violations"
    exit 1
fi

echo "✅ Check passed"
exit 0
```

### MSRV (Minimum Supported Rust Version)

```bash
# Run CI on both pinned toolchain and MSRV
make ci-msrv

# MSRV is read from root Cargo.toml:
# rust-version = "1.88"
```

## 9. N-ary Shape Independence Policy ★ (Added 2025-12)

### Overview

Rules that process sums (Add/Sub chains) must use **shape-independent** traversal to avoid regressions caused by tree parentization differences like `((a+b)+c)` vs `(a+(b+c))`.

### Policy

| Pattern | Policy |
|---------|--------|
| `Expr::Add(l, r)` | ❌ **Prohibited** in n-ary sum modules → use `AddView::from_expr()` |
| `Expr::Mul(l, r)` | ✅ Allowed for structural patterns (e.g., `2*cos(kx)` detection) |

### N-ary API (`nary.rs`)

```rust
use crate::nary::{AddView, MulView, Sign, build_balanced_add, build_balanced_mul};

// Flatten any sum tree (handles Add/Sub/Neg chains)
let view = AddView::from_expr(ctx, root);
for &(term, sign) in &view.terms {
    let is_positive = sign == Sign::Pos;
    // Process each term...
}

// Rebuild as balanced tree
let result = build_balanced_add(ctx, &terms);
```

### Monitored Modules

The lint script checks these files for binary `Expr::Add` patterns:

- `telescoping.rs` ← migrated to AddView
- (Add more as migrated)

### Lint Script

```bash
# Run manually
./scripts/lint_nary_shape_independence.sh

# Should pass:
✅ N-ary shape independence: no binary Expr::Add in n-ary sum modules.
```

### Extending the Policy

To add a new file to n-ary enforcement:

1. Migrate the file to use `AddView` for sum traversal
2. Add the file path to `NARY_SUM_FILES` in the lint script
3. Run the lint to verify no violations

To add an exemption for a specific structural pattern:

```rust
// nary-lint: allow-binary (structural pattern match for half-angle)
if let Expr::Mul(l, r) = ctx.get(expr) { ... }
```

### Testing After Migration

Verify shape-independence with these test patterns:

```rust
#[test]
fn test_my_rule_shape_independent() {
    // Same expression with different tree shapes
    let left_assoc = parse("((a+b)+c)+d");  // left-associative
    let right_assoc = parse("a+(b+(c+d))"); // right-associative
    let balanced = parse("(a+b)+(c+d)");    // balanced
    
    // All three must simplify to identical result
    assert_eq!(simplify(left_assoc), simplify(right_assoc));
    assert_eq!(simplify(right_assoc), simplify(balanced));
}
```

### Future: MulView Migration Roadmap

Currently `Expr::Mul(l,r)` is **allowed** because many product rules are structurally binary (e.g., detecting `a * a^-1` or `a * (b/c)`). Prohibiting would cause too many false positives.

**Progressive hardening roadmap:**

| Phase | Policy | Trigger |
|-------|--------|---------|
| **Current** | ✅ Allowed | — |
| Phase 2 | ⚠ Allowed + lint warning | When migrating product rules to `MulView` |
| Phase 3 | ❌ Prohibited in `@nary-product` modules | When MulView coverage is high |

**When to advance:**
- Regressions appear from product tree shapes
- Significant rules migrated to `MulView`
- Lint has low false-positive rate

## 10. Anti-Explosion Budget Policy ★ (Planned)

### Overview

The engine has **fragmented budget systems** that need unification into a coherent anti-explosion policy. See [docs/BUDGET_POLICY.md](docs/BUDGET_POLICY.md) for full design.

### Current State

| Budget Type | Metrics | Error Type |
|-------------|---------|------------|
| `ExpandBudget` | pow_exp, terms | (silent reject) |
| `PolyBudget` | terms, degree | `PolyError::BudgetExceeded` |
| `ZippelBudget` | points, retries | — |
| `PhaseBudgets` | rewrites | Cycle detection |

### Target State

Single unified system:
- **`BudgetConfig`**: Limits per `(Operation, Metric)`
- **`BudgetScope`**: RAII tracking of current operation
- **`BudgetExceeded`**: Uniform error with `{ op, metric, used, limit }`

### Implementation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Infrastructure (`budget.rs`, `ContextStats`) | ✅ Complete |
| 1 | Unify error types (`CasError::BudgetExceeded` wraps struct) | ✅ Complete |
| 2 | Simplify pipeline (`PassStats` flows to caller) | ✅ Complete |
| 3 | Expand / multinomial (`expand_with_stats`) | ✅ Complete |
| 4 | Polynomial ops (`mul_with_stats`, `div`, `gcd`) | ✅ Complete |
| 5 | Zippel GCD (existing `ZippelBudget`) | ✅ Complete |
| 6 | CI lint (`lint_budget_enforcement.sh`) | ✅ Complete |

### Enforcement Layers

```
Layer A: Central    — NodesCreated in Context::add (always counts)
Layer B: Hotspot    — TermsMaterialized, RewriteSteps, PolyOps (module-specific)
Layer C: Estimation — Fail fast before expensive work
```

### Contract Tests (to add)

```rust
#[test]
fn budget_rejects_huge_expansion() {
    // (a+b)^200 → BudgetExceeded(Expand, TermsMaterialized)
}

#[test]
fn budget_stops_runaway_simplify() {
    // expr with 1000+ rewrites → BudgetExceeded(SimplifyCore, RewriteSteps)
}
```

### Implemented Contract Tests

See `tests/budget_contract_tests.rs`:

- `test_expand_fails_fast_without_allocation` — Layer C precheck
- `test_expand_with_stats_reports_metrics` — PassStats reporting  
- `test_budget_preset_*` — Preset limit values
- `test_budget_charge_*` — Charge tracking and limits

### Adding a New Operation (Contributor Guide)

1. Add `Operation` variant in `budget.rs` (if needed)
2. Create `_with_stats` wrapper returning `PassStats`
3. Add to `lint_budget_enforcement.sh` hotspots if critical
4. Add contract test validating PassStats fields
5. Run `make ci && make lint-budget`
