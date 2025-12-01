# Project Maintenance Guide

This document provides a comprehensive overview of the project's architecture, debugging tools, and extension patterns to facilitate maintenance and future development.

## 1. Debug Logging System

A concise debug logging system is built into the `Simplifier` to trace rule applications.

### Usage

To enable debug logging:

```rust
let mut simplifier = Simplifier::with_default_rules();
simplifier.enable_debug();
```

To disable it:

```rust
simplifier.disable_debug();
```

### Output Format

Logs are printed to `stderr` in the following format:

```
[DEBUG] Rule 'RuleName' applied: ExprId(Input) -> ExprId(Output)
[DEBUG] Global Rule 'RuleName' applied: ExprId(Input) -> ExprId(Output)
```

This format is designed to be easily filtered using `grep`:

```bash
cargo run | grep "[DEBUG]"
```

### Implementation

-   **`Simplifier`**: Holds a `debug_mode` flag.
-   **`LocalSimplificationTransformer`**: Checks `debug_mode` before applying rules and prints logs using `eprintln!`.

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

## 3. Extending the CAS

### Adding a New Simplification Rule

1.  **Create the Rule Struct**: Define a struct that implements the `Rule` trait.
2.  **Implement `apply`**:
    ```rust
    impl Rule for MyRule {
        fn name(&self) -> &str { "My Rule" }
        fn apply(&self, ctx: &mut Context, id: ExprId) -> Option<Rewrite> {
            // Check pattern and return Some(Rewrite) if matched
        }
    }
    ```
3.  **Register the Rule**: Add it to `Simplifier::register_default_rules` in `crates/cas_engine/src/engine.rs`.

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
