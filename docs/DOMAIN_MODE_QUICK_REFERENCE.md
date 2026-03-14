# DomainMode Quick Reference

> Short operational reference for engine design, solver behavior, and metamorphic tests.
>
> This is a **working summary**, not the full normative spec. Canonical sources remain:
> - [SEMANTICS_POLICY.md](./SEMANTICS_POLICY.md)
> - [POLICY_TABLES.md](./POLICY_TABLES.md)
> - [SOLVER_SIMPLIFY_POLICY.md](./SOLVER_SIMPLIFY_POLICY.md)

## Purpose

Use this document when you need a fast answer to questions like:

- What is the real semantic difference between `Generic` and `Assume`?
- Which examples are good to distinguish `Strict`, `Generic`, `Assume`, and `Assume + Wildcard`?
- What should tests assert, and what should the engine surface as `requires`, `assumed`, `warnings`, `conditional`, or `residual`?

## Canonical Rule

`DomainMode` is defined in [cas_solver_core/src/domain_mode.rs](../crates/cas_solver_core/src/domain_mode.rs):

- `Strict`: only proven-safe transforms.
- `Generic`: allows unproven `Definability` conditions.
- `Assume`: allows unproven `Definability` and `Analytic` conditions.

The taxonomy is defined in [cas_solver_core/src/solve_safety_policy.rs](../crates/cas_solver_core/src/solve_safety_policy.rs):

- `Definability`: small holes or definedness requirements.
  Examples: `x ≠ 0`, `expr is defined`.
- `Analytic`: sign/range/branch restrictions.
  Examples: `x > 0`, `x ≥ 0`, inverse-trig principal range.

## Core Matrix

| Mode | Unproven Definability | Unproven Analytic | Typical meaning |
|---|---:|---:|---|
| `Strict` | No | No | Only what is proved from the current AST/oracle |
| `Generic` | Yes | No | "Almost everywhere" algebra, but no new sign/range assumptions |
| `Assume` | Yes | Yes | Exploration mode with explicit mathematical assumptions |

## AssumeScope Matrix

`AssumeScope` is defined in [cas_solver_core/src/assume_scope.rs](../crates/cas_solver_core/src/assume_scope.rs).

Important: it matters only when `DomainMode = Assume`.

| Scope | Meaning |
|---|---|
| `Real` | Assume only within real-domain semantics. If the path needs complex promotion, do not silently cross into it. |
| `Wildcard` | Same core assumption behavior, but if complex would be needed, prefer residual + visible transparency instead of hard failure. |

## Simplifier / Eval: Canonical Examples

| Expression | Strict | Generic | Assume | Why it matters |
|---|---|---|---|---|
| `x/x -> 1` | Block | Allow | Allow | `NonZero(x)` is `Definability` |
| `0*(1/x) -> 0` | Block | Allow | Allow | `Defined(1/x)` is `Definability` |
| `ln(x*y) -> ln(x)+ln(y)` | Block | Block | Allow | `x>0`, `y>0` are `Analytic` |
| `log(b, b^x) -> x` | Block | Block | Allow | symbolic `b>0`, `b≠1` is not allowed in `Generic` |
| `exp(ln(x)) -> x` | Block | Allow | Allow | `x>0` is **inherited** from `ln(x)` already being in the AST |
| `ln(exp(x)) -> x` in `RealOnly` | Allow | Allow | Allow | `exp(x) > 0` is provable, so no assumption is needed |
| `sqrt(x)^2 -> x` | Block | Block | Allow | `x ≥ 0` is `Analytic` unless already inherited/proven |

### Important Nuance

`Generic` does **not** mean "never use analytic conditions".

It means:

- do **not introduce** new analytic conditions,
- but you may preserve or inherit analytic conditions that are already intrinsic to the input AST.

This is why:

- `exp(ln(x)) -> x` is allowed in `Generic`,
- but `ln(x*y) -> ln(x)+ln(y)` is not.

## Solver: Canonical Examples

| Equation | Strict | Generic | Assume + Real | Assume + Wildcard | Why it matters |
|---|---|---|---|---|---|
| `2^x = y` | Solve, usually `Conditional` | Solve, usually `Conditional` | Solve with explicit positive requirement/signal on `y` | Same as `Real` in this case | Not a good example to distinguish `Generic` vs `Assume` |
| `(-2)^x = 5` | Do not use as "assume" example | Do not use as "generic" example | Avoid silent complex promotion | Residual + transparency | Good example for `Wildcard` |
| `(-2)^x = y` | Same caveat | Same caveat | Avoid silent complex promotion | Residual + visible complex/preset transparency | Good `Wildcard` residual contract |
| `(a^x - b) * (x - 1) = 0` | Not the main discriminator | Residual in `Generic + Real` | Discrete in `Assume + Real` | Usually same as `Real` unless complex is needed | Good example to distinguish `Generic` vs `Assume` in solver |

## Practical Testing Guidance

### Good examples to distinguish `Strict` vs `Generic`

Use `Definability`-class rewrites:

- `x/x`
- `0*(1/x)`
- `0/x`

Expected pattern:

- `Strict`: blocked unless proven
- `Generic`: allowed
- `Assume`: allowed

### Good examples to distinguish `Generic` vs `Assume`

Use introduced `Analytic`-class rewrites:

- `ln(x*y)`
- `log(b, b^x)`
- `sqrt(x)^2`

Expected pattern:

- `Generic`: blocked
- `Assume`: allowed with assumptions/transparency

### Good examples to distinguish `Generic` vs `Assume` in solver

Use families where the solver genuinely needs analytic assumptions to keep going:

- `(a^x - b) * (x - 1) = 0`
- symbolic exponential isolation with unknown positive base/rhs

Do **not** use `2^x = y` as the primary discriminator:

- solver can already handle it in `Strict`/`Generic` via conditional guards
- `Assume` mostly changes how the positive requirement is surfaced

### Good examples to distinguish `Assume + Real` vs `Assume + Wildcard`

Use cases that would need complex promotion:

- `(-2)^x = 5`
- `(-2)^x = y`

Expected pattern:

- `Assume + Real`: do not silently jump to complex
- `Assume + Wildcard`: residual + transparency, not garbage and not hidden promotion

## Design Rules for the Engine

1. `Generic` may introduce only `Definability`-class requirements.
2. `Generic` must not introduce new `Analytic` assumptions just to make a rewrite convenient.
3. `Generic` may preserve inherited intrinsic conditions already present in the AST.
4. `Assume` may use `Analytic` assumptions, but they must surface through the structured channels (`required`, `assumed`, `warnings`, `assumption_records`, transparency, or guards depending on the path).
5. `Wildcard` is not "complex mode". It is "do not hard-fail; return a safe residual + visible signal when complex would be needed".

## Design Rules for Tests

1. Every test that tries to separate `Generic` from `Assume` must state whether the condition is:
   - introduced,
   - inherited,
   - or provable.
2. Prefer one canonical family per distinction:
   - `x/x` for `Strict vs Generic`
   - `log(b, b^x)` for `Generic vs Assume`
   - `(-2)^x = 5` for `Assume + Real vs Wildcard`
   - `(a^x - b) * (x - 1) = 0` for solver `Generic vs Assume`
3. Do not use examples whose difference is only formatting or whether a requirement appears as:
   - `required`,
   - `assumed`,
   - or a `Conditional` guard,
   unless that is exactly the contract under test.

## Canonical Test References

Use these as the most trustworthy executable references:

- [domain_mode_contract_tests.rs](../crates/cas_solver/tests/domain_mode_contract_tests.rs)
- [log_exp_domain_contract_tests.rs](../crates/cas_solver/tests/log_exp_domain_contract_tests.rs)
- [solver_isolation_scope_contract_tests.rs](../crates/cas_solver/tests/solver_isolation_scope_contract_tests.rs)
- [solver_assume_scope_contract_tests.rs](../crates/cas_solver/tests/solver_assume_scope_contract_tests.rs)
- [semantic_behavior_contract_expressions.csv](../crates/cas_solver/tests/semantic_behavior_contract_expressions.csv)
- [equation_solution_kind_contract_cases.csv](../crates/cas_solver/tests/equation_solution_kind_contract_cases.csv)

## Anti-Patterns

Avoid these mistakes when adding new rules or tests:

- Treating `Generic` as "Assume but quieter".
- Using `2^x = y` as proof that only `Assume` can solve symbolic exponentials.
- Treating `Wildcard` as implicit complex promotion.
- Mixing inherited intrinsic conditions with introduced assumptions.
- Writing metamorphic tests that expect exact equality of all metadata channels when the real contract is only "minimum semantic floor preserved".
