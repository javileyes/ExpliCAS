The idea: **each rule declares its “soundness class”** and the engine decides whether to:

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
| **HeuristicAssumption** | Correct "typically," but cannot be guaranteed without assuming (symbolic parity, "x≥0", etc.) | ⛔ block (or mark as “Blocked”) | ✅ apply with ⚠️ `Assumes:` (+ `Requires:` if applicable) | ⛔ block |
| **BranchSensitivePrincipal** | Depends on the principal branch / discontinuities (complex log, atan, etc.) | ⛔ block (or ✅ only if `ValueDomain=RealOnly` with strong guards) | ✅ apply if `RealOnly` and conditions met; normally ⛔ in `ComplexEnabled` | ⛔ block (or requires extra verification) |
| **NormalizationOnly** | Normal form rewriting (commutativity/ordering) without changing meaning | ✅ apply | ✅ apply | ✅ apply |

**Key Note:** If you want `exp(ln(x))` in RealOnly to behave like `sqrt(x)^2`, then its rule must be classified as `EquivalenceUnderIntroducedRequires` (with `x>0`) and **not** as `HeuristicAssumption/BranchSensitive`.

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
| Introduce `Requires:` (explicit math conditions) | ✅ only for “IntroducedRequires” rules | ✅ | ✅ but ideally only if provable or already inherited |
| Introduce `Assumes:` (heuristics, “we assume x≥0”) | ⛔ | ✅ | ⛔ |
| Block with “Blocked: requires …” | ✅ when rule is heuristic/branch or not allowed in generic | (usually no) | ✅ |

This explains your case:

* `sqrt(x)^2 → x` is allowed in generic as **IntroducedRequires** (and your `implicit_domain` also infers it).
* `exp(ln(x)) → x` is classified as **not allowed in generic**, so it is marked **Blocked** instead of emitting `Requires: x>0`.

---

## Concrete Recommendation for your Engine (to ensure consistency)

1. Define **a single mapping**: `SoundnessLabel → AllowedIn(Generic/Assume/Strict)`.
2. Define flags per rule:
* `introduces_requires: bool`
* `introduces_assumptions: bool`
* `branch_sensitive: bool`


3. Suggested Policy:
* **Generic**: allows `introduces_requires`, forbids `introduces_assumptions`, and forbids `branch_sensitive` (except for a whitelist in RealOnly).
* **Assume**: allows everything, but flags with ⚠️.
* **Strict**: allows only `Unconditional` or `UnderInheritedRequires` (or `IntroducedRequires` only if the condition is already proven/inherited).

---

## Table 4 — Implicit Conditions: Closed vs. Strict

Not all implicit domain restrictions are treated equally. The engine distinguishes between:

| Feature | sqrt(x), ln(x) | 0^x |
| --- | --- | --- |
| **Condition Type** | Closed () | Strict () |
| **Singular Point** | Included in domain (boundary) | Excluded ( indeterminate) |
| **Semantic Ambiguity** | Minimal | High (integer vs. real, real vs. complex) |
| **Automatic Inference** | ✅ Yes | ⛔ No in generic |

---

### Why does `sqrt(x)` infer `x ≥ 0` but `0^x` does not infer `x > 0`?

**1. `sqrt(x)` — Closed, canonical, and direct restriction**

* The condition  is a **direct precondition of the operator** on its argument.
* It is **closed** (it includes the boundary ).
* It does not depend on the type of  (integer, rational, real): it is always .
* Therefore, `infer_implicit_domain` adds it as an inheritable `NonNegative(x)`.

**2. `0^x` — Strict restriction with a singular point and ambiguity**

* The condition is **strict** (), not closed.
* At , it falls into  (indeterminate by convention).
* At , it becomes  (division by zero).
* **Depends on the semantics of `Pow**`:
* In `RealOnly`:  is defined only for .
* In `ComplexEnabled`: It enters log branches (, undefined).
* If integer exponents are allowed:  is defined for .



> [!IMPORTANT]
> Automatically inferring  for  is equivalent to **choosing a strong semantics** and **forcing a strict inequality on a free variable**. This is `assume` behavior, not `generic`.

---

### Behavior in Each Mode

| Expression | Generic | Assume | Strict |
| --- | --- | --- | --- |
| `sqrt(x)^2 → x` | ✅ with `Requires: x ≥ 0` | ✅ | ✅ only if proven |
| `0^x → 0` | ⛔ Blocked (unless `x` is literal > 0) | ✅ with `Assumes: x > 0` | ⛔ Blocked |
| `exp(ln(x)) → x` | ✅ with `Requires: x > 0` | ✅ | ✅ only if proven |

---

### Design Criteria

The engine uses this heuristic to decide if a condition is "implicitly inheritable":

```
Inheritable IF:
  - Closed condition (≥, ≤, ≠) 
  - Direct operator precondition (sqrt, ln, abs)
  - Semantically unambiguous

NOT inheritable IF:
  - Strict condition (>, <) with a singular point
  - Depends on optional semantics (integer vs. real)
  - Requires "choosing" one interpretation among several valid ones

```

