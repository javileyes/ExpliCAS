# ExpliCAS - Educational Computer Algebra System in Rust

ExpliCAS is a modular Computer Algebra System (CAS) written in Rust, designed to provide **step-by-step mathematical explanations**. It is built with portability and educational utility in mind.

## Features

-   **Step-by-Step Simplification**: Shows every rule applied to transform an expression.
-   **Basic Arithmetic**: Addition, subtraction, multiplication, division, exponentiation.
-   **Calculus**:
    -   Symbolic Integration (`integrate(x^2, x) -> x^3/3`).
    -   Symbolic Differentiation (`diff(sin(x), x) -> cos(x)`).
-   **Number Theory**:
    -   GCD/LCM (`gcd(12, 18) -> 6`, `lcm(4, 6) -> 12`).
    -   Modular Arithmetic (`mod(10, 3) -> 1`, `10 mod 3 -> 1`).
    -   Prime Factorization (`factors(12) -> 2^2 * 3`).
    -   Combinatorics (`fact(5) -> 120`, `choose(5, 2) -> 10`, `perm(5, 2) -> 20`).
-   **Algebraic Simplification**:
    -   Combining like terms (`2x + 3x -> 5x`).
    -   Polynomial expansion (`expand((x+1)^2) -> x^2 + 2x + 1`).
    -   Polynomial factorization (`factor(x^3 - x) -> x(x-1)(x+1)`).
    -   Grouping terms (`collect(ax + bx, x) -> (a+b)x`).
    -   Fraction simplification (`(x^2 - 1) / (x + 1) -> x - 1`).
-   **Functions**:
    -   Trigonometry (`sin`, `cos`, `tan`) with identities (`sin(2x) -> 2sin(x)cos(x)`).
    -   Logarithms (`log(base, x)`, `ln(x)`) with expansion (`ln(xy) -> ln(x) + ln(y)`).
    -   Roots (`sqrt(x)`, `sqrt(x, n)`).
    -   Absolute value (`abs(x)`).
-   **Equivalence Checking**: Verify if two expressions are equal (`equiv sin(x+y), sin(x)cos(y)+...`).
-   **Variables**: Symbolic computation with variables.
-   **Constants**: Built-in support for mathematical constants `e` and `pi`.
-   **Equation Solving**: Isolate variables in equations (`solve x+2=5, x`).
-   **Substitution**: Replace variables with values or other expressions.
-   **Interactive CLI**: Command-line interface with history support.
-   **Configuration**: Enable/disable specific simplification rules (e.g., `root_denesting`, `trig_double_angle`) via the `config` command.
-   **Verbose Mode**: Toggle step-by-step output (`steps on/off`) for educational or performance purposes.
-   **Debug Tools** (Phase 2):
    -   **Rule Profiler**: Track rule application frequency for performance analysis (`profile enable/disable/clear`).
    -   **AST Visualizer**: Export expression trees to Graphviz DOT format (`visualize <expr>`).
    -   **Timeline HTML**: Generate interactive HTML visualization of simplification steps (`timeline <expr>`).
-   **Performance Optimized** (Phase 1):
    -   Conditional multi-pass simplification (-46.8% on complex fractions).
    -   Cycle detection prevents infinite loops.
    -   Early exit optimization for fast path.
-   **Development Debug System** (tracing-based):
    -   Parametrizable debug logging with zero overhead when disabled.
    -   See [DEBUG_SYSTEM.md](DEBUG_SYSTEM.md) for usage details.


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

#### 1. Polynomial Factorization
The engine uses the Rational Root Theorem to factor polynomials.
```text
> factor(x^3 - x)
```
**Output:**
```
Steps:
1. Factor polynomial using Rational Root Theorem
   -> x * (x^2 - 1)
2. Factor difference of squares
   -> x * (x - 1) * (x + 1)
Result: x * (x - 1) * (x + 1)
```

#### 2. Advanced Trigonometry
Simplifies complex trigonometric expressions using double angle and sum identities.
```text
> sin(2*x) + 2*sin(x)*cos(x)
```
**Output:**
```
Steps:
1. sin(2x) = 2sin(x)cos(x) [Double Angle Identity]
   -> 2 * sin(x) * cos(x) + 2 * sin(x) * cos(x)
2. Combine like terms
   -> 4 * sin(x) * cos(x)
Result: 4 * sin(x) * cos(x)
```

#### 3. Logarithm Expansion & Simplification
Automatically expands products/quotients and simplifies inverses.
```text
> ln(x^2 * y) - 2*ln(x)
```
**Output:**
```
Steps:
1. log(b, x*y) = log(b, x) + log(b, y)
   -> ln(x^2) + ln(y) - 2 * ln(x)
2. log(b, x^y) = y * log(b, x)
   -> 2 * ln(x) + ln(y) - 2 * ln(x)
3. Combine like terms (2ln(x) - 2ln(x) = 0)
   -> ln(y)
Result: ln(y)
```

#### 4. Symbolic Integration
Basic indefinite integration support.
```text
> integrate(x^2 + sin(x), x)
```
**Output:**
```
Result: x^3 / 3 - cos(x)
```

#### 5. Symbolic Differentiation
Computes derivatives using product, quotient, and chain rules.
```text
> diff(x * sin(x), x)
```
**Output:**
```
Result: sin(x) + x * cos(x)
```

#### 6. Number Theory
Perform integer arithmetic operations like GCD, LCM, and prime factorization.
```text
> factors(2345)
```
**Output:**
```
Result: 5 * 7 * 67
```
```text
> gcd(12, 18)
```
**Output:**
```
Result: 6
```
```text
> 5!
```
**Output:**
```
Result: 120
```
```text
> choose(5, 2)
```
**Output:**
```
Result: 10
```
```text
> 25 mod 7
```
**Output:**
```
Result: 4
```

#### 7. Pre-Calculus (Absolute Value & Inequalities)
The solver handles absolute values (branching) and inequalities (sign flipping).

**Example: Absolute Value Equation**
```text
> solve |2*x + 1| = 5, x
Steps:
1. Split absolute value (Case 1): 2 * x + 1 = 5
   -> 2 * x = 4
   -> x = 2
2. Split absolute value (Case 2): 2 * x + 1 = -5
   -> 2 * x = -6
   -> x = -3
Result: x = 2
--- Solution 2 ---
Result: x = -3
```

**Example: Inequality**
```text
> solve -2*x < 10, x
Steps:
1. Divide both sides by -2 (flips inequality)
   -> x > -5
Result: (-5, infinity)
```

**Example: Absolute Value Inequality (Intersection)**
```text
> solve |x| < 5, x
Steps:
1. Split absolute value (Case 1): x < 5
2. Split absolute value (Case 2): x > -5
Result: (-5, 5)
```

**Example: Absolute Value Inequality (Union)**
```text
> solve |x| > 5, x
Steps:
1. Split absolute value (Case 1): x > 5
2. Split absolute value (Case 2): x < -5
Result: (-infinity, -5) U (5, infinity)
```

#### 6. Equivalence Checking
Verify if two expressions are mathematically equivalent.
```text
> equiv sin(x+y), sin(x)*cos(y) + cos(x)*sin(y)
```
**Output:** `True`

#### 6. Equation Solving
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
3. arcsin(1/2) = pi/6
   -> x = pi / 6
Result: x = pi / 6
```

#### 7. Variable Substitution
```text
> subst x^2 + x, x=3
```
**Output:** `12`

### Advanced Examples

ExpliCAS can handle complex symbolic identities and simplifications.

#### 1. Ramanujan's Nested Roots
Simplifies nested square roots automatically.
```text
> sqrt(3 + 2*sqrt(2))
```
**Output:** `1 + sqrt(2)`

#### 2. Logarithmic Mirror
Simplifies complex exponential and logarithmic identities.
```text
> x^(1/ln(x))
```
**Output:** `e`

#### 3. Trigonometric Identities
Verifies complex identities like the triple angle formula.
```text
> simplify sin(3*x) - (3*sin(x) - 4*sin(x)^3)
```
**Output:** `0`

#### 4. Rational Simplification
Handles polynomial division and simplification.
```text
> (x^3 - 1) / (x - 1)
```
**Output:** `x^2 + x + 1`

### Configuration & Rule Toggling
 
You can dynamically enable or disable specific simplification rules directly from the CLI. This is useful for educational purposes (showing intermediate steps without full simplification) or for debugging.
 
#### Listing Rules
To see all available configurable rules and their current status:
```text
> config list
```
**Output:**
```
Enabled Rules:
  - Distributive Property
  - Simplify Fractions
  ...
Disabled Rules:
  - (none)
```
 
#### Disabling a Rule
To prevent a specific rule from applying:
```text
> config disable trig_double_angle
Rule 'trig_double_angle' set to false.
```
Now, expressions like `sin(2*x)` will **not** be simplified to `2*sin(x)*cos(x)`.
 
#### Enabling a Rule
To re-enable a rule:
```text
> config enable trig_double_angle
Rule 'trig_double_angle' set to true.
```
 
#### Restoring Defaults
To reset all rules to their default state:
```text
> config restore
All rules restored to default values.
```

### Debug Tools

ExpliCAS includes powerful debugging and visualization tools for developers and educators.

#### Rule Profiler

Track which rules are being applied and how often:

```text
> profile enable
Profiler enabled.

> (x+1)^2
Result: x^2 + 2*x + 1

> profile
Rule Profiling Report
─────────────────────────────────────────────
Rule                                      Hits
─────────────────────────────────────────────
Binomial Expansion                           1
Combine Like Terms                           2
─────────────────────────────────────────────
TOTAL                                        3

> profile clear
Profiler statistics cleared.
```

#### AST Visualizer

Export expression trees to Graphviz DOT format for visual debugging:

```text
> visualize (x+1)*(x-1)
AST exported to ast.dot
Render with: dot -Tsvg ast.dot -o ast.svg
```

Then render the tree:
```bash
$ dot -Tsvg ast.dot -o ast.svg
$ open ast.svg  # Opens beautiful tree visualization
```

#### Timeline HTML

Generate interactive HTML visualization of simplification steps with intelligent filtering:

```text
> timeline 1/(sqrt(x)+1)+1/(sqrt(x)-1)-(2*sqrt(x))/(x-1)
Timeline exported to timeline.html
Open in browser to view interactive visualization.
```

**Features:**
- **Intelligent Step Filtering**: Automatically filters out non-productive steps (where global state doesn't change) and low-importance canonicalization steps
  - Reduces visual clutter (e.g., 47 steps → 13 meaningful steps)
  - Configurable verbosity: `Low`, `Normal`, `Verbose`
- **Compact Layout**: 
  - Shows only the current expression (removed redundant "After" state)
  - Displays rule name and local transformation
  - Final result clearly highlighted
- **Smart LaTeX Rendering**:
  - Global expressions use readable notation (e.g., `√x` for roots)
  - Rule transformations preserve mathematical notation for clarity:
    - Exponent rules show fractional notation: `x^{1/2} → x^{1/2·2}`
    - Other rules use standard notation with roots
- **Professional Styling**:
  - MathJax-rendered expressions
  - Gradient backgrounds and color-coded sections
  - Responsive design with smooth animations

**Step Importance Levels:**
- **Trivial**: Identity operations (Add Zero, Mul By One) - hidden by default
- **Low**: Canonicalization, sorting, constant evaluation - hidden in Normal mode
- **Medium**: Standard algebraic transforms - always shown
- **High**: Major transformations (Factor, Expand, Integrate) - always highlighted

**Customization:**
To add new rules that preserve exponent notation, edit `crates/cas_engine/src/timeline.rs`:
```rust
let should_preserve_exponents = step.rule_name.contains("Multiply exponents")
    || step.rule_name.contains("Power of a Power")
    || step.rule_name.contains("Your Rule Here");
```


### Running Tests

To verify the correctness of the system, run the test suite:

```bash
cargo test
```

This runs unit tests for all crates and integration tests ensuring the parser and engine work together correctly.

## Project Structure

-   `crates/cas_ast`: Core mathematical data structures and LaTeX rendering.
    -   `latex.rs`: Standard LaTeX rendering (converts fractional exponents to roots).
    -   `latex_no_roots.rs`: Specialized LaTeX rendering that preserves exponent notation.
-   `crates/cas_parser`: Parsing logic (String -> AST).
-   `crates/cas_engine`: Simplification rules and step tracer.
    -   `step.rs`: Step importance classification system.
    -   `timeline.rs`: HTML timeline generation with intelligent filtering.
    -   `strategies.rs`: Global state filtering for non-productive steps.
-   `crates/cas_cli`: Command-line interface.
