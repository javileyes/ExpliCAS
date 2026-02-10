# Core Engine Test Documentation

This document describes the critical tests that verify canonicalization invariants and the simplify/expand contract.

## Quick Reference

| Suite | Location | Tests | Purpose |
|-------|----------|-------|---------|
| Property Tests | `cas_engine/tests/property_tests.rs` | 19 | Canonicalization invariants |
| Policy Tests | `cas_cli/tests/policy_tests.rs` | 23 | Simplify vs Expand contract |
| Canonical Tests | `cas_cli/tests/canonical_tests.rs` | 15 | Specific canonical form cases |

---

## Property Tests (Fuzzing)

These tests use `proptest` to generate random expressions and verify invariants hold universally.

### Configuration
```bash
# Default: SAFE profile (depth=2, size=8)
STRESS_PROFILE=NORMAL cargo test -p cas_engine --test property_tests
```

### Structural Invariants (post-`normalize_core`)

| Test | Invariant | Description |
|------|-----------|-------------|
| `test_normalize_core_idempotent` | `normalize_core(normalize_core(e)) == normalize_core(e)` | Applying twice yields same result |
| `test_normalize_core_no_neg_number` | No `Neg(Number(_))` | Negatives absorbed into Number |
| `test_normalize_core_no_double_neg` | No `Neg(Neg(_))` | Double negation eliminated |
| `test_normalize_core_no_nested_add` | No `Add(Add(..), ..)` | Add is right-associative/flat |
| `test_normalize_core_no_nested_mul` | No `Mul(Mul(..), ..)` | Mul is right-associative/flat |
| `test_normalize_core_no_nested_pow` | No `Pow(Pow(x,a), b)` | N3: Pow flattened to `Pow(x, a*b)` |
| `test_normalize_core_order_deterministic` | Same input → same output | Ordering is stable |
| `test_numbers_reduced_form` | `gcd(numer, denom) == 1` | Fractions always reduced |
| `test_simplify_no_pow_one` | No `Pow(x, 1)` | Trivial powers eliminated |

### Metamorphic Properties

| Test | Metamorphic Relation |
|------|----------------------|
| `test_metamorphic_add_zero` | `e + 0` equivalent to `e` |
| `test_metamorphic_mul_one` | `e * 1` equivalent to `e` |
| `test_metamorphic_mul_zero` | `e * 0` simplifies to `0` |

---

## Policy Tests (Contract)

These tests verify the **simplify vs expand contract** defined in Policy A+.

### Style-Only Differences (Fractional Binomials)

| Expression | `simplify` | `expand` |
|------------|-----------|----------|
| `1/2*(√2-1)` | `(√2-1)/2` (factored) | `√2/2 - 1/2` (distributed) |
| `(-1/2)*(1-√2)` | `(√2-1)/2` (flipped) | distributed |

### Structural Patterns (Must Apply)

| Pattern | `simplify` | `expand` |
|---------|-----------|----------|
| `(x-1)(x+1)` | `x² - 1` | Preserved (conjugate guard) |
| `2*(x+1)` | `2x + 2` | `2x + 2` |

### Binomial Preservation (Policy A+)

| Expression | `simplify` | `expand` |
|------------|-----------|----------|
| `(a+b)*(c+d)` | Preserved | Expanded |
| `(x+1)^2` | Config-dependent | `x²+2x+1` |

---

## Key Invariants Summary

### N0: `Neg(Number(n)) → Number(-n)`
- **Enforced in**: `Context::add()` (constructor) + `normalize_core()`
- **Test**: `test_normalize_core_no_neg_number`

### N1: `Neg(Neg(x)) → x`
- **Enforced in**: `Context::add()` + `normalize_core()`
- **Test**: `test_normalize_core_no_double_neg`

### N2: Flatten/Sort Add/Mul
- **Enforced in**: `Context::add()` + `normalize_core()`
- **Tests**: `test_normalize_core_no_nested_add`, `test_normalize_core_no_nested_mul`

### N3: `Pow(Pow(x,a),b) → Pow(x, a*b)`
- **Enforced in**: `normalize_core()`
- **Test**: `test_normalize_core_no_nested_pow`

### Policy A+: Fractional Distribution Guard
- **Guard in**: `DistributeRule` (polynomial/ and algebra/distribution.rs)
- **Bypass**: `expand()` uses `expand.rs` directly
- **Test**: `test_contract_fractional_binomial_style_only`

---

## Running Tests

```bash
# Property tests (fuzzing)
cargo test -p cas_engine --test property_tests

# Policy tests (contract)
cargo test -p cas_cli --test policy_tests

# Canonical tests
cargo test -p cas_cli --test canonical_tests

# Full core test suite
cargo test -p cas_engine --test property_tests && \
cargo test -p cas_cli --test policy_tests && \
cargo test -p cas_cli --test canonical_tests
```
