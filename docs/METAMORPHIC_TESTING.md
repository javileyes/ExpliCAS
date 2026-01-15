# Metamorphic Testing Framework

ExpliCAS includes a **metamorphic testing** framework that validates mathematical correctness through invariant properties, catching bugs that traditional golden tests miss.

## Overview

Metamorphic testing exploits the property that if `A` simplifies to `B`, then `A + e` must equal `B + e` for any expression `e`. This catches:
- Incomplete `requires` conditions
- Rule priority issues
- Cancellation failures in context
- Soundness bugs in transformation rules

## Quick Start

```bash
# Run metatests (CI mode: 50 samples, depth 3)
cargo test -p cas_engine --test metamorphic_simplification_tests

# Stress mode (local: 500 samples, depth 5)
METATEST_STRESS=1 cargo test -p cas_engine --test metamorphic_simplification_tests

# Reproduce with specific seed
METATEST_SEED=12345 cargo test -p cas_engine --test metamorphic_simplification_tests
```

## Configuration

| Environment Variable | Default | Stress Mode | Description |
|---------------------|---------|-------------|-------------|
| `METATEST_STRESS` | `0` | `1` | Enable stress mode |
| `METATEST_SEED` | `0xC0FFEE` | - | RNG seed for reproducibility |

| Parameter | CI Mode | Stress Mode |
|-----------|---------|-------------|
| `samples` | 50 | 200 |
| `min_valid` | 20 | 100 |
| `depth` | 3 | 3 |
| `eval_samples` | 200 | 300 |
| `atol/rtol` | 1e-9 | 1e-9 |

## Expression Generator

The random expression generator uses only "safe" operations to avoid domain issues:

### Allowed
- Variables from the identity
- Small integer constants (-3 to 3)
- `+`, `-`, `*`
- `pow(base, k)` with `k ∈ {0,1,2,3,4}`
- `sin(...)`, `cos(...)` (total functions)

### NOT Allowed (domain issues)
- Division (`/`)
- `log`, `ln`, `sqrt`, `root`
- Negative exponents

## Historical Logging

Every test run is logged to `crates/metatest_log.jsonl` in JSON Lines format:

```json
{"timestamp":1768499153,"test":"pythagorean_identity","seed":12648430,"samples":50,"depth":3,"min_valid":20,"stress":false,"passed":1,"failed":0,"skipped":0}
```

### Analyzing Logs

```bash
# View last 5 runs
tail -5 crates/metatest_log.jsonl

# Filter failures
jq 'select(.failed > 0)' crates/metatest_log.jsonl

# Summary by test
jq -s 'group_by(.test) | map({test: .[0].test, runs: length, passed: [.[] | .passed] | add})' crates/metatest_log.jsonl
```

## Test Cases

| Test | Identity | Status |
|------|----------|--------|
| `pythagorean_identity` | sin²x + cos²x = 1 | ✅ |
| `double_angle_sin` | sin(2x) = 2·sin(x)·cos(x) | ✅ |
| `double_angle_cos` | cos(2x) = cos²x - sin²x | ✅ |
| `add_zero` | x + 0 = x | ✅ |
| `mul_one` | x · 1 = x | ✅ |
| `binomial_square` | (x+1)² = x² + 2x + 1 | ✅ |
| `difference_of_squares` | (x-1)(x+1) = x² - 1 | ✅ |
| `polynomial_simplify` | (x+1)(x-1) + 1 = x² | ✅ |
| `log_product` | ln(2) + ln(3) = ln(6) | ⏸️ Skipped (no vars) |
| `triple_tan_identity` | tan(x)·tan(π/3-x)·tan(π/3+x) = tan(3x) | ✅ |

## Failure Output

When a test fails, it outputs full context for reproduction:

```text
Metatest FAILED (seed=12648430, iter=42)
A = tan(x) * tan(pi/3 - x) * tan(pi/3 + x)
B = tan(3*x)
e = (x) + (sin(x * 2))
A+e = (tan(x) * tan(pi/3 - x) * tan(pi/3 + x)) + ((x) + (sin(x * 2)))
B+e = (tan(3*x)) + ((x) + (sin(x * 2)))
A+e simplified = 1 + tan(3 * x) + cos(x) + x^2
B+e simplified = 1 + cos(x) + x^2 + (3 * sin(x) - 4 * sin(x)^3) / (4 * cos(x)^3 - 3 * cos(x))
Error: Too few valid samples: 0 < 20 (eval_failed=200)
```

## Adding New Tests

```rust
#[test]
fn metatest_my_identity() {
    // Document the identity being tested
    // sum_formula: sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
    assert_metamorphic_addition(
        "sum_formula",           // Test name (for logging)
        "sin(a + b)",            // Expression A
        "sin(a)*cos(b) + cos(a)*sin(b)",  // Expected simplified B
        &["a", "b"],             // Variables (max 1 for now)
    );
}
```

## Implementation Details

### Deterministic RNG

Uses a Linear Congruential Generator (LCG) for reproducibility:

```rust
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
}
```

### Numeric Verification

- Expressions are parsed, simplified, then evaluated at sample points
- Uses `atol + rtol * scale` tolerance for comparison
- Requires `min_valid` successful evaluations (filters NaN/Inf)

## Resolved Issues

### Triple Tan Identity (Fixed)

**Original problem**: When `tan(3*x)` appeared in sum context (B+e), `TanToSinCosRule` expanded it to complex sin/cos forms, while the identity result from A was protected. This caused different canonical forms.

**Fix implemented** (V2.15):
1. **Anti-worsen guard** on `TanToSinCosRule`: Don't expand `tan(n*x)` where n is integer > 1
2. **Inf filtering** in numeric verification: Filter out infinity values from singularities
3. **`__hold` support** in `eval_f64`: Transparently evaluate held expressions

Both A+e and B+e now simplify to `1 + tan(3·x) + cos(x) + x²` and pass numeric verification.
