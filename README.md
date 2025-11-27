# ExpliCAS - Educational Computer Algebra System in Rust

ExpliCAS is a modular Computer Algebra System (CAS) written in Rust, designed to provide **step-by-step mathematical explanations**. It is built with portability and educational utility in mind.

## Features

-   **Step-by-Step Simplification**: Shows every rule applied to transform an expression.
-   **Basic Arithmetic**: Addition, subtraction, multiplication, division, exponentiation.
-   **Algebraic Simplification**:
    -   Combining like terms (`2x + 3x -> 5x`).
    -   Polynomial expansion (`expand((x+1)^2) -> x^2 + 2x + 1`).
    -   Polynomial factorization (`factor(2x^2 + 4x) -> 2x(x + 2)`).
    -   Grouping terms (`collect(ax + bx, x) -> (a+b)x`).
    -   Fraction simplification (`(x^2 - 1) / (x + 1) -> x - 1`).
-   **Functions**:
    -   Trigonometry (`sin`, `cos`, `tan`) with identity simplification (`sin(x)^2 + cos(x)^2 -> 1`).
    -   Logarithms (`log(base, x)`, `ln(x)`) with properties (`log(b, b^x) -> x`).
    -   Roots (`sqrt(x)`, `sqrt(x, n)`).
    -   Absolute value (`abs(x)`).
-   **Variables**: Symbolic computation with variables.
-   **Substitution**: Replace variables with values or other expressions.
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

> sqrt(x) * sqrt(x)
```
*Output:* `x` (Canonicalized to `x^(1/2) * x^(1/2)` then simplified)
```

**7. Variable Substitution**
Use the `subst` command to evaluate expressions for specific variable values.

```text
> subst x+1, x=2
```
*Output:* `3`

```text
> subst x^2+x, x=3
```
*Output:* `12` (Calculated as `3^2 + 3` -> `9 + 3` -> `12`)

**8. Equation Solving**
Use the `solve` command to isolate a variable in an equation.

```text
> solve x + 2 = 5, x
```
*Output:* `x = 3`

```text
> solve ln(x) = 1, x
```
*Output:* `x = e`

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
