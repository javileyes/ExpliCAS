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
-   **Constants**: Built-in support for mathematical constants `e` and `pi`.
-   **Equation Solving**: Isolate variables in equations (`solve x+2=5, x`).
-   **Substitution**: Replace variables with values or other expressions.
-   **Interactive CLI**: Command-line interface with history support.
-   **Verbose Mode**: Toggle step-by-step output (`steps on/off`) for educational or performance purposes.

## Getting Started

### Prerequisites

-   [Rust and Cargo](https://rustup.rs/) installed.

### Running the CLI

To start the interactive demo:

```bash
cargo run -p cas_cli
```

### Examples

Once inside the CLI, try these expressions to see the step-by-step engine in action:

#### 1. Trigonometric Identities & Canonicalization
The system automatically reorders terms to match known identities.

```text
> sin(x)^2 + cos(x)^2 + x
```
**Output:**
```
Steps:
1. Reorder addition terms
   -> cos(x)^2 + sin(x)^2 + x
2. cos^2(x) + sin^2(x) = 1  [Pythagorean Identity]
   -> 1 + x
Result: 1 + x
```

#### 2. Equation Solving with Steps
The solver uses the full power of the simplification engine at each step.

```text
> solve 2 * sin(x) = 1
```
**Output:**
```
Steps:
1. Divide both sides by 2
   -> sin(x) = 1/2
2. Take arcsin of both sides
   -> x = arcsin(1/2)
3. arcsin(1/2) = pi/6  [Evaluate Trigonometric Functions]
   -> x = pi / 6
Result: x = pi / 6
```

#### 3. Complex Algebra & Distribution
```text
> 2 * (x + 3) + 4 * x
```
**Output:**
```
Steps:
1. Distribute 2 over (x + 3)
   -> 2 * x + 6 + 4 * x
2. Combine like terms (2x + 4x)
   -> 6 * x + 6
Result: 6 * x + 6
```

#### 4. Logarithms & Exponents
```text
> exp(ln(x * y))
```
**Output:**
```
Steps:
1. b^log(b, x) = x  [Exponential-Log Inverse]
   -> x * y
Result: x * y
```

#### 5. Fractions
```text
> 1/2 + 1/3
```
**Output:** `5/6`

#### 6. Variable Substitution
```text
> subst x^2 + x, x=3
```
**Output:** `12`

#### 7. Verbose Mode
Toggle detailed output:
```text
> steps off
> solve ln(x) = 1
```
*Output:* `x = e` (Instant result)

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
