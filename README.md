# ExpliCAS - Educational Computer Algebra System in Rust

ExpliCAS is a modular Computer Algebra System (CAS) written in Rust, designed to provide **step-by-step mathematical explanations**. It is built with portability and educational utility in mind.

## Features

-   **Step-by-Step Simplification**: Shows every rule applied to transform an expression.
-   **Arithmetic**: Basic operations and simplification (e.g., `x + 0 -> x`, `2 * 3 -> 6`).
-   **Polynomials**: Distribution, combining like terms, and annihilation (e.g., `2*(x+3) + 4*x -> 6*x + 6`).
-   **Exponents**: Power rules (e.g., `x^2 * x^3 -> x^5`, `(x^2)^3 -> x^6`).
-   **Interactive CLI**: Command-line interface with history support.

## Getting Started

### Prerequisites

-   [Rust and Cargo](https://rustup.rs/) installed.

### Running the CLI

To start the interactive demo:

```bash
cargo run -p cas_cli
```

### Examples

Once inside the CLI, try these expressions:

**1. Arithmetic & Polynomials**
```text
> 2 * (x + 3) + 4 * x
```
*Output:* `6 * x + 6` (with steps showing distribution and combining terms)

**2. Exponents**
```text
> x^2 * x^3
```
*Output:* `x^5`

**3. Nested Powers**
```text
> (x^2)^3
```
*Output:* `x^6`

**4. Zero Exponent**
```text
> x^0
```
*Output:* `1`

**5. Fractions**
```text
> 1/2 + 1/3
```
*Output:* `5/6`

**6. Roots**
```text
> sqrt(x) * sqrt(x)
```
*Output:* `x` (Canonicalized to `x^(1/2) * x^(1/2)` then simplified)

### Running Tests

To verify the correctness of the system, run the test suite:

```bash
cargo test
```

This runs unit tests for all crates and integration tests ensuring the parser and engine work together correctly.

## Project Structure

-   `crates/cas_ast`: Core mathematical data structures.
-   `crates/cas_parser`: Parsing logic (String -> AST).
-   `crates/cas_engine`: Simplification rules and step tracer.
-   `crates/cas_cli`: Command-line interface.
