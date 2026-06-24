# Gatekeeper G2 — Educational narrative of limits (scoping)

Scoped 2026-06-24. This is the **sub-cycle sequence** for the Fase-1 gatekeeper
"narrativa educativa de límites" (the educational half of the north star; limit
values are already correct, the narrative is shallow). Each sub-cycle is bounded,
retainable, has its own commit and green-before-commit, and updates only its own
technique's huella. ~7–9 sub-cycles (skill estimate for G2: 6–10).

## The gap (probed)

Limit **values** are correct. The **narrative** names the technique in a single
substep with an empty `detail` and shows NO intermediate work:

| input | value (ok) | current substep | missing |
|---|---|---|---|
| `(x²−1)/(x−1)` @1 | `2` | "Factorizar … y cancelar el factor común" | the factored form, the cancellation, the substitution |
| `(x−sin x)/x³` @0 | `1/6` | "Indeterminación 0/0 … aplica L'Hôpital o Taylor" | the iteration `(1−cos x)/3x² → sin x/6x → cos x/6 → 1/6` |
| `x²+3x+1` @2 | `11` | "Sustitución directa …" | `(2)²+3(2)+1 = 4+6+1 = 11` |
| `sin(x)/x` @0 | `1` | "Aplicar el límite notable: lím(u→0) sin(u)/u = 1" | the `u` identification + indeterminate-form note |

## Architecture (3 layers)

1. **Value** — `cas_math/src/limits_support.rs`: `eval_limit_*` → ~40
   `apply_finite_*_rule` each `-> Option<ExprId>` (value only). Factored num/den,
   L'Hôpital derivative pairs, Taylor series are computed then **discarded**.
   `LimitEvalOutcome {expr, warning}` cannot carry steps.
2. **Engine** — `cas_engine/src/limits/engine.rs`: `steps = Vec::new()`.
   `eval/actions.rs::eval_limit` (538–575) builds **one** `Step`, rule_name ∈
   {`Evaluar límite finito`, `Evaluar límite unilateral finito`,
   `Evaluar límite en infinito`, `Conservar límite residual`}; meta carries only
   `limit_point` (the lone soundness input the narrator uses to claim 0/0).
3. **Didactic** — `cas_didactic/.../focused_rule_substeps.rs::generate_limit_substeps`
   (19593–19605): POST-HOC. Re-derives the technique NAME from before/after/point
   via `notable_limit_name -> Option<String>`, wraps in **one** `SubStep`. `after`
   is the soundness oracle (a structural match with the wrong value declines).

## Key decision — deepen by POST-HOC reconstruction, not trace-threading

The gold-standard narrators (`generate_integration_by_parts_substeps` 12362,
`generate_rational_linear_partial_fraction_integration_substeps` 14437) build
their multi-substep chains by **reconstructing** the work in the didactic layer
(decomposition "in scratch"), NOT by threading a trace out of the value engine.
Mirror that:

- **DO** reconstruct intermediate work in `generate_limit_substeps` (it has `ctx`
  and can factor / differentiate / substitute). Keep `after` as the oracle: only
  emit a reconstruction whose endpoints match the computed value.
- **DO NOT** thread a structured trace through `LimitEvalOutcome` + the ~40
  `apply_finite_*_rule` signatures (large cross-crate change, high risk; the
  "narration may disagree with the engine's chosen rule" risk is contained by the
  after-oracle, since both must land on the same value).

### One-time structural change (folded into SC1)
`notable_limit_name` returns `Option<String>` (single). Add per-technique substep
builders returning `Vec<SubStep>` (mirroring the integration `generate_*_substeps`),
and have `generate_limit_substeps` dispatch to them, **falling back to the current
single-name SubStep** for not-yet-deepened techniques. Behavior-preserving for
every technique except the one deepened in the cycle.

## Huella (precise — where each cycle pays)

- **Per-technique substep wording is pinned ONLY** in the in-file module
  `focused_rule_substeps.rs:21752–22097` (`mod limit_notable_tests`). Each cycle
  updates its technique's needles + the `titles.len()==1` assertions (deepening to
  N substeps breaks `len()==1`).
- **Insulated:** no scorecard fixture, no CLI contract, no smoke pins per-technique
  substep text. CLI contract (`limit_contract_tests.rs:632–679`) and the smoke
  (`engine_limit_command_matrix_smoke.py`) pin only the 4 top-level rule names and
  `steps_count==1`. Substeps live INSIDE the one Step, so `steps_count` is
  unchanged by deepening. `limit_expected_step_substring_count:211` tracks only
  top-level rule strings → won't drift.
- Validation per cycle: `cargo test -p cas_didactic` (the in-file module) is the
  primary gate; full `cargo test --workspace` + scorecard regen + huella compare
  as usual. Scorecard state should stay identical (substep wording isn't tracked).

## Soundness invariants (pinned declines — never loosen)

- **after-oracle:** structural match with the wrong value must NOT narrate
  (`sin(x)/x→2`, `sin(2x)/x→3`, `(2^x−1)/x→ln(3)` decline). Keep exact
  `as_rational_const`/`BigRational` gating (cf. [[soundness-gates-must-be-exact]]).
- **L'Hôpital 0/0** requires `limit_point` set AND `limit_denominator_vanishes_at`
  (exact); declines pinned when absent (21867–21872).
- **Residual/DNE** ("Conservar límite residual") gets no narrator (doesn't match
  the `Evaluar límite` prefix) — SC7 adds an explicit branch.

## Sub-cycle sequence (ordered by ROI)

| # | technique | scope / reuse | notes |
|---|---|---|---|
| **SC1** | **Factor-and-cancel** `(x²−1)/(x−1)→2` | + infra (multi-substep dispatch). Reuse `limit_share_polynomial_factor` + `Polynomial` gcd/factor. Substeps: factor → cancel common → substitute. | First: cheapest, highest frequency, carries the structural change. |
| **SC2** | **Direct substitution (continuity)** | reuse `limit_is_polynomial` + point eval. Substeps: state continuity → substitute → evaluate. | Trivial; cements the pattern. |
| **SC3** | **Notable limits (first-order)** sin/tan/arcsin/…/u, `(e^u−1)/u`, `ln(1+u)/u` | identify `u` + indeterminate form → cite standard limit → apply. | The signature educational content; may split scaled/cross forms off. |
| **SC4** | **Notable limits (second-order & e)** `(1−cos u)/u²=1/2`, `(1+u)^(1/u)=e`, `(1+1/x)^x=e` | same pattern; the `e` forms at infinity. | Pairs naturally with SC3. |
| **SC5** | **L'Hôpital / Taylor iteration** `(x−sin x)/x³→1/6` | reuse `differentiate_symbolic_expr`; RE-derive the iteration post-hoc, each pair still 0/0 (exact), final substitution = `after`. | Marquee technique; meatiest — may be 2 cycles (polynomial 0/0 then transcendental). |
| **SC6** | **Squeeze (sándwich)** `x·sin(1/x)→0` | reuse `limit_is_squeeze_product`. Show bounding `−|xᵏ| ≤ … ≤ |xᵏ| → 0`. | |
| **SC7** | **Dominance at ∞** (ln≪pot≪exp; rational degree) | deepen the 6 `limit_infinity_dominance` strings with the leading-term comparison. | |
| **SC8** | **One-sided & DNE residual** `1/x`@0 | NEW dispatch branch (rule_dispatch.rs:53 only matches `Evaluar límite`). Narrate left vs right → two-sided DNE. | Higher-risk (new branch); last. |

## Entry

Start with **SC1** (factor-and-cancel + infra): retainable on its own, green
before commit, updates only its needles in `limit_notable_tests`. If a future
cycle has no retainable sub-step ready, fall back to a P1 win rather than landing
a half-built narration (skill guardrail for class-L gatekeepers).
