The idea: **each rule declares its "soundness class"** and the engine decides whether to:

* **apply unconditionally**
* **apply by introducing requires**
* **block**
* **apply only in "assume" mode**
* **apply only in "strict" mode**
* **perform a symbolic skip and leave only the numeric result** (for branch-sensitive rules in metatest / equivalence checking)

---

## Table 1 — Policy by Mode (DomainMode)

| SoundnessLabel / Class | Meaning | Generic | Assume | Strict |
| --- | --- | --- | --- | --- |
| **UnconditionalEquivalence** | Algebraic equivalence that is "always" valid (within the current ValueDomain) | ✅ apply | ✅ apply | ✅ apply |
| **EquivalenceUnderInheritedRequires** | Valid if you **already** have sufficient "requires" (inherited/implicit) | ✅ apply **only if** `requires ⊇ needed` | ✅ apply **only if** `requires ⊇ needed` (or can inherit from implicit_domain if allowed) | ✅ apply **only if** `requires ⊇ needed` |
| **EquivalenceUnderIntroducedRequires** | Valid if conditions are introduced (e.g., `x>0`, `cos(x)≠0`, `base≠1`) | ✅ apply **and emit** `Requires:` | ✅ apply (may also emit `Assumes:` if extra heuristics are involved) | ✅ apply **only if** those "requires" can be justified or were already present (per your Strict definition) |
| **HeuristicAssumption** | Correct "typically," but cannot be guaranteed without assuming (symbolic parity, "x≥0", etc.) | ⛔ block (or mark as "Blocked") | ✅ apply with ⚠️ `Assumes:` (+ `Requires:` if applicable) | ⛔ block |
| **BranchSensitivePrincipal** | Depends on the principal branch / discontinuities (complex log, atan, etc.) | ⛔ block (or ✅ only if `ValueDomain=RealOnly` with strong guards) | ✅ apply if `RealOnly` and conditions met; normally ⛔ in `ComplexEnabled` | ⛔ block (or requires extra verification) |
| **NormalizationOnly** | Normal form rewriting (commutativity/ordering) without changing meaning | ✅ apply | ✅ apply | ✅ apply |

---

## Table 2 — Additional Gate by ValueDomain (RealOnly vs ComplexEnabled)

| Rule / Typical Identity | RealOnly | ComplexEnabled |  |  |
| --- | --- | --- | --- | --- |
| `sqrt(u^2) = u` | ✅ (with its semantics) | ⛔ or changes to `sqrt(u^2)=±u` (non-representable) |  |  |
| `(x^n)^(1/n)` cancellations | ✅ with parity + requires | ⛔ usually (branch issues) |  |  |
| `log(b, b^y)=y` | ✅ with `b>0, b≠1` (+ care if variable) | ⛔ (principal branch) |  |  |
| `exp(ln(x))=x` | ✅ with `x>0` (if `ln` is real) | ⛔ in general (complex log) |  |  |
| `atan` identities with moduli | ✅ (but branch-sensitive) | ⛔ or requires mod (2π) branch comparison |  |  |

---

## Table 3 — How to Handle "Requires" in Each Mode

| Action | Generic | Assume | Strict |
| --- | --- | --- | --- |
| Inherit `Requires:` from intrinsic operator preconditions | ✅ always (see Invariant A) | ✅ | ✅ |
| Introduce `Requires:` (explicit math conditions) | ✅ only for `IntroducedRequires` rules of class Definability | ✅ | ✅ but ideally only if provable or already inherited |
| Introduce `Assumes:` (heuristics, "we assume x≥0") | ⛔ | ✅ | ⛔ |
| Block with "Blocked: requires …" | ✅ when rule would introduce Analytic conditions not backed by the AST | (usually no) | ✅ |

---

## Concrete Recommendation for your Engine (to ensure consistency)

1. Define **a single mapping**: `SoundnessLabel → AllowedIn(Generic/Assume/Strict)`.
2. Define flags per rule:
   * `introduces_requires: bool`
   * `introduces_assumptions: bool`
   * `branch_sensitive: bool`

3. Suggested Policy:
   * **Generic**: allows inherited intrinsic requires and Definability-class introduced requires. Forbids `introduces_assumptions` and Analytic-class introduced requires.
   * **Assume**: allows everything, but flags with ⚠️.
   * **Strict**: allows only `Unconditional` or `UnderInheritedRequires` (or `IntroducedRequires` only if the condition is already proven/inherited).

This explains the current cases:

* `sqrt(x)^2 → x` is allowed in Generic — `sqrt(x)` intrinsically implies `x ≥ 0`, which is inherited.
* `exp(ln(x)) → x` is allowed in Generic — `ln(x)` intrinsically implies `x > 0`, which is inherited. The rule does **not** introduce a new condition, it preserves one already present.
* `0^x → 0` is blocked in Generic — the `x > 0` requirement is not intrinsic to an operator precondition; it's a degenerate rewrite with high solver-corruption risk (see Table 4).

---

## Table 4 — Implicit Conditions: Intrinsic vs. Introduced

Not all domain restrictions are treated equally. The engine distinguishes by **provenance**:

| Feature | sqrt(x), ln(x) | 0^x → 0 |
| --- | --- | --- |
| **Condition Source** | Intrinsic (operator precondition) | Policy-guarded (degenerate operator rewrite) |
| **Typical Require** | sqrt: x ≥ 0 ; ln: x > 0 | x > 0 |
| **Heritable in Generic** | ✅ Yes (must be preserved if witness removed) | ⛔ No (kept blocked by policy) |
| **Semantic Ambiguity** | Minimal — single canonical interpretation | High (integer vs. real, real vs. complex) |
| **Automatic Inference** | ✅ Yes (`infer_implicit_domain`) | ⛔ No in Generic |

---

### Why does `exp(ln(x)) → x` work in Generic but `0^x → 0` does not?

**1. `exp(ln(x)) → x` — Inherited intrinsic condition**

* `ln(x)` is already in the AST and intrinsically requires `x > 0`.
* The simplification `exp(ln(x)) → x` does **not introduce** any new condition — it only **inherits** one already present.
* When `ln(x)` is eliminated, the `Requires: x > 0` is preserved (propagated to the result).
* Therefore, this is safe in Generic: **no new domain was invented**.

**2. `0^x → 0` — Policy-guarded degenerate rewrite**

* `0^x` is a degenerate case of the power operator, not a standard function with canonical preconditions.
* The condition `x > 0` would need to be **introduced** by the rewrite — there is no operator in the AST that already guarantees it.
* At `x = 0`, it falls into `0^0` (indeterminate by convention).
* At `x < 0`, it becomes `0^(-n)` (division by zero).
* **Depends on semantics**: in `RealOnly`, `0^x` is defined only for `x > 0`. In `ComplexEnabled`, it enters log branches.

> [!IMPORTANT]
> Converting a symbolic expression `0^x` to a constant `0` in Generic mode would be **surprising**: the user sees a variable-dependent expression collapse to a constant with a hidden Requires. This is `assume` behavior, not `generic`.

---

### Behavior in Each Mode

| Expression | Generic | Assume | Strict |
| --- | --- | --- | --- |
| `sqrt(x)^2 → x` | ✅ with inherited `Requires: x ≥ 0` | ✅ | ✅ only if proven |
| `exp(ln(x)) → x` | ✅ with inherited `Requires: x > 0` | ✅ | ✅ only if proven |
| `0^x → 0` | ⛔ Blocked (unless `x` is literal > 0) | ✅ with `Assumes: x > 0` | ⛔ Blocked |

---

### Design Criteria (Generic)

The engine decides "heritable in Generic" primarily by **provenance**:

```
Heritable IF:
  - Intrinsic operator precondition already present in the input AST
  - The condition is preserved (propagated) if the witness node is removed
  - Examples: sqrt(x) → x≥0, ln(x) → x>0, log(b,x) → x>0 ∧ b>0 ∧ b≠1

Not allowed in Generic IF:
  - The rewrite would introduce new domain requirements not implied by the input AST
  - The rewrite is a degenerate operator evaluation with high solver-corruption risk
  - Examples: 0^x → 0 (introduces x>0), ln(a·b) → ln(a)+ln(b) (introduces a>0 ∧ b>0)
```

### Three Invariants

1. **Invariant A — No introduced requires in Generic**
   A rule in Generic cannot add Requires that aren't already backed by intrinsic operator preconditions present in the input AST.

2. **Invariant B — Requires must be preserved**
   If a simplification eliminates a node that provided a precondition (e.g., removes `ln`), the `Requires` must be propagated to the result.

3. **Invariant C — Equivalence under current requires**
   A rule only fires if it is an equivalence under the accumulated Requires, without inventing new assumptions.

---

### SolveSafety Classification

Rules declare their safety level for solver contexts via `SolveSafety`:

| Classification | Prepass | Tactic(Generic) | Tactic(Assume) | Tactic(Strict) |
| --- | --- | --- | --- | --- |
| `Always` | ✅ | ✅ | ✅ | ✅ |
| `IntrinsicCondition(class)` | ⛔ | ✅ | ✅ | ⛔ |
| `NeedsCondition(Definability)` | ⛔ | ✅ | ✅ | ⛔ |
| `NeedsCondition(Analytic)` | ⛔ | ⛔ | ✅ | ⛔ |
| `Never` | ⛔ | ⛔ | ⛔ | ⛔ |
