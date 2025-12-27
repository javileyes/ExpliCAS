# SEMANTICS POLICY

> **Version 1.0** | Last updated: 2025-12-26

This document defines the semantic configuration axes that control how ExpliCAS evaluates and simplifies expressions. Each axis is orthogonal and controls a specific aspect of mathematical semantics.

## Overview

ExpliCAS uses **4 orthogonal semantic axes**:

| Axis | Controls | Values |
|------|----------|--------|
| **DomainMode** | Variable assumptions (≠0, >0, etc.) | `Strict`, `Generic`, `Assume` |
| **ValueDomain** | Universe of constants | `RealOnly`, `ComplexEnabled` |
| **BranchPolicy** | Multi-valued function branches (ℂ only) | `Principal` |
| **InverseTrigPolicy** | Inverse∘function compositions | `Strict`, `PrincipalValue` |

---

## Axis A: DomainMode ✅ (Implemented)

Controls rules that require assumptions about symbolic variables.

### Values

| Value | Behavior | Use Case |
|-------|----------|----------|
| `Strict` | Only cancel/simplify if provably valid | Teaching, formal proofs |
| `Generic` | Allow all simplifications silently (legacy) | Backward compatibility |
| `Assume` | Allow simplifications with traceable warnings | Research, exploration |

### Examples

| Expression | Strict | Generic | Assume |
|------------|--------|---------|--------|
| `x/x` | `x/x` | `1` | `1` + warning |
| `x^0` | `x^0` | `1` | `1` + warning |
| `4x/(2x)` | `2x/x` | `2` | `2` + warning |

### Affected Rules

- `SimplifyFractionRule` (cancellation gate)
- `IdentityPowerRule` (`x^0`, `x^1`)
- `CancelCommonFactorsRule`
- `QuotientOfPowersRule`

---

## Axis B: ValueDomain (Planned PR1)

Defines the universe of values for constant evaluation.

### Values

| Value | Description | `sqrt(-1)` result |
|-------|-------------|-------------------|
| `RealOnly` | ℝ extended with ±∞, undefined | `undefined` |
| `ComplexEnabled` | ℂ with principal branch | `i` |

### Default

`RealOnly` — matches current behavior, no complex arithmetic.

---

## Axis C: BranchPolicy (Planned PR1)

Controls how multi-valued functions are resolved. **Only applies when `ValueDomain = ComplexEnabled`**.

### Values

| Value | Behavior |
|-------|----------|
| `Principal` | Use principal branch (e.g., `log(-1) = iπ`) |

### Future Values (Non-Goals for V1)

- `AllBranches` — return set of values
- `Symbolic` — preserve multi-valued structure

---

## Axis D: InverseTrigPolicy (Planned PR4)

Controls simplification of inverse∘function compositions like `arctan(tan(x))`.

> **Important**: This is NOT the same as BranchPolicy. InverseTrigPolicy applies to inverse trig functions in ℝ, not complex branch cuts.

### Values

| Value | `arctan(tan(x))` | Warning |
|-------|------------------|---------|
| `Strict` | `arctan(tan(x))` (no change) | None |
| `PrincipalValue` | `x` | "assumed x ∈ (-π/2, π/2)" |

### Current State

The existing REPL `mode principal/strict` command currently controls this behavior but is conflated with other settings. PR4 will extract it cleanly.

---

## Rule Dependency Table

| Rule | DomainMode | ValueDomain | BranchPolicy | InverseTrigPolicy |
|------|:----------:|:-----------:|:------------:|:-----------------:|
| `SimplifyFractionRule` | ✅ | - | - | - |
| `IdentityPowerRule` | ✅ | - | - | - |
| `CancelCommonFactorsRule` | ✅ | - | - | - |
| `QuotientOfPowersRule` | ✅ | - | - | - |
| `sqrt(negative literal)` | - | ✅ | ✅ | - |
| `log(negative literal)` | - | ✅ | ✅ | - |
| `ArcTanTanRule` | - | - | - | ✅ |
| `ArcSinSinRule` | - | - | - | ✅ |
| `ArcCosCosRule` | - | - | - | ✅ |
| `i * i → -1` | - | ✅ | - | - |

---

## Hotspots (Audit List)

### DomainMode Hotspots

- Fraction cancellation: `SimplifyFractionRule`, `CancelCommonFactorsRule`
- Power identity: `x^0`, `0^x`, `0^0`
- Root simplification: `sqrt(x^2)` → `|x|` vs `x`
- Algebraic inverse: `f^{-1}(f(x))`

### ValueDomain / BranchPolicy Hotspots

- `sqrt(negative_literal)` — first case to implement
- `log(negative_literal)` — future
- `pow(negative_base, rational_exp)` — future
- Complex trig/hyperbolic — very future

### InverseTrigPolicy Hotspots

- `arctan(tan(x))`, `arcsin(sin(x))`, `arccos(cos(x))`
- Machin-like sums of arctan
- Parent context propagation bugs (historical)

---

## JSON Schema (V1 Compatible)

New fields are **optional** and do not break existing consumers:

```json
{
  "schema_version": 1,
  "ok": true,
  "result": "1",
  "domain": { "mode": "assume" },
  "semantics": {
    "domain_mode": "assume",
    "value_domain": "real",
    "branch": "principal",
    "inv_trig": "strict"
  },
  "warnings": [
    { "code": "W_DOMAIN_ASSUMPTION", "message": "cancelled factor assumed nonzero" }
  ]
}
```

---

## Defaults (Stability Contract)

| Axis | Default | Rationale |
|------|---------|-----------|
| `domain_mode` | `Generic` | Backward compatibility |
| `value_domain` | `RealOnly` | Safe, no complex |
| `branch` | `Principal` | Standard convention |
| `inv_trig` | `Strict` | Safe, no domain assumptions |
| `const_fold` | `Off` | Defer semantic decisions |

---

## Presets (V1.1)

Presets provide **explicit, auditable shortcuts** to configure multiple axes at once. They do not change defaults and are only applied when the user invokes them explicitly.

### Design Principles

1. **Explicit, not magic** — User must invoke `semantics preset <name>`
2. **Traceable** — Apply prints diff of changes
3. **Complete** — Each preset defines all 5 axes
4. **Few and clear** — Only 4 presets, each with distinct purpose

### Available Presets

| Preset | domain | value | branch | inv_trig | const_fold | Purpose |
|--------|--------|-------|--------|----------|------------|---------|
| `default` | generic | real | principal | strict | off | Reset to engine defaults |
| `strict` | strict | real | principal | strict | off | Conservative, no assumptions |
| `complex` | generic | complex | principal | strict | safe | Enable ℂ + materialization |
| `school` | generic | real | principal | principal | off | Classroom mode |

### REPL Commands

```
semantics preset              # List available presets
semantics preset <name>       # Apply preset (prints diff)
semantics preset help <name>  # Show preset configuration
```

### Example

```
> semantics preset complex
Applied preset: complex
Changes:
  value_domain: real → complex
  const_fold:   off → safe

> sqrt(-1)
Result: i
```

### Non-Goals

- Presets do not create aliases for `semantics set`
- Presets do not introduce "auto" or heuristic behavior
- Custom/user-defined presets are not supported in V1

---

## CLI Flags (Current and Planned)

| Flag | Current | Planned |
|------|:-------:|:-------:|
| `--domain strict\|generic\|assume` | ✅ | - |
| `--value-domain real\|complex` | - | PR3 |
| `--inv-trig strict\|principal` | - | PR4 |

---

## REPL Commands

| Command | Current | After PR4 |
|---------|---------|-----------|
| `mode principal` | Changes branch + inv_trig | Changes only `inv_trig` |
| `mode strict` | Changes branch + inv_trig | Changes only `inv_trig` |
| `domain strict` | ✅ Sets DomainMode | Same |
| `domain assume` | ✅ Sets DomainMode | Same |

---

## PR Checklist

For any PR touching semantic axes:

- [ ] Update this document if adding/changing behavior
- [ ] Add contract test(s) for new axis values
- [ ] Ensure axis is reflected in JSON (if user-facing)
- [ ] Update hotspots section if adding new affected rules
- [ ] Verify defaults don't change existing behavior
- [ ] Run `make ci` to ensure no regressions

---

## Non-Goals (V1)

The following are explicitly **not** supported in V1:

- General complex logarithm (`log(z)` for arbitrary z)
- Multi-valued symbolic expressions
- Automatic domain inference from context
- Riemann surface branch tracking
- Symbolic complex conjugate/modulus

These may be considered for future versions.
