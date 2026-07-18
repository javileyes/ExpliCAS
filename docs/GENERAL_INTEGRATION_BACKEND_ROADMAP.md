# General Integration Backend Roadmap

This document is part of the engine auto-improvement loop. It defines the
long-term hybrid integration direction: keep the current educational
real-domain integrator conservative and didactic, while adding a separate
algorithmic backend for broader integration methods.

## Position

The project should move toward two integration routes over one shared symbolic
core:

- **Educational route.** Curated, real-domain, explainable, conservative, and
  stable. It owns rich didactic traces, table/substitution/by-parts families,
  explicit required conditions, and honest residuals.
- **Algorithmic backend.** Broader methods, summarized traces, bounded search,
  structured assumptions, verification status, and conservative fallback. It may
  attempt methods that are too broad or opaque for the educational route, but it
  must not publish unverified certainty.

These are not two independent engines. They should share:

- AST, parser, formatting, and public command API
- simplification, equivalence, normalization, and domain predicates
- derivative engine used for antiderivative verification
- runtime budgets, recursion guards, scorecards, and corpus promotion rules
- required-condition and residual reporting concepts

The separation is about ownership and trust boundaries: how an antiderivative is
found, how much method detail can be explained, and what evidence is required
before exposing it as a public answer.

## Non-Negotiable Rules

- Do not add broad search inside the current educational `integrate` route.
- Do not accept an antiderivative from the algorithmic backend unless it is
  verified or explicitly marked residual/inconclusive.
- Do not hide real-domain, branch, component, pole, or integration-constant
  assumptions in display cleanup.
- Do not let the backend slow ordinary promoted `integrate` lanes unless the
  retained capability and runtime evidence justify it.
- Do not pretend an algorithmic method has a full didactic derivation when only
  a summarized method trace is available.

## Backend Result Boundary

The first retained backend milestone should be a structured result boundary,
not a new algorithm.

A backend candidate result should be able to carry:

- original integrand and variable
- candidate antiderivative, when one exists
- method tag, such as `rational`, `hermite`, `heurisch_probe`,
  `table_reused`, or `unsupported`
- assumptions and required conditions
- integration-constant policy
- verification status:
  - `verified`
  - `verified_under_conditions`
  - `inconclusive`
  - `failed`
  - `not_attempted`
- residual reason:
  - `unsupported_method`
  - `domain_policy_missing`
  - `branch_policy_missing`
  - `verification_failed`
  - `verification_inconclusive`
  - `budget_exceeded`
  - `disabled_by_mode`
- trace level:
  - `educational_full`
  - `algorithmic_summary`
  - `diagnostic_only`

Public behavior should only consume `verified` or `verified_under_conditions`
results unless a mode explicitly asks for diagnostic output.

Note on condition filtering: `include_backend_conditions`
(`cas_engine` `integration_conditions.rs:64-68`) converts to requires only
`NonZero`/`Positive`/`NonNegative`/`LowerBound` backend predicates and
deliberately discards `Defined`/`InvTrigPrincipalRange`/`EqZero`/`EqOne` at
this boundary.

## Orchestration Policy

Default public `integrate` behavior should stay educational-first:

1. Try existing educational families.
2. If the educational route returns a supported antiderivative, keep its rich
   trace and required conditions.
3. If it returns residual, optionally consult the algorithmic backend only when
   mode, budget, and runtime policy allow it.
4. Verify any backend candidate by differentiating it back to the integrand under
   the stated conditions.
5. If verification is inconclusive or failed, keep the residual and record the
   backend result as discovery/diagnostic, not as an accepted answer.

The backend must be easy to disable. A failed or slow backend attempt must not
make supported educational cases worse.

## Development Phases

Status note (2026-06-10): the backend module was split into an ownership
directory (`general_integration_backend/{probe_runner,model,verification,verification_normalization,methods,tests}.rs`)
with no behavior change. New normalization cases belong in
`verification_normalization`, new method recognition in `methods`; the result
contract and verifier service files should stay stable as families grow.

Phase status (2026-06-10):

- Phase 0: done
- Phase 1: done (boundary + disabled-by-default modes + mode-boundary lane)
- Phase 2: done (shared verifier with structured outcomes in
  `general_integration_backend/verification*.rs`)
- Phase 3: done (rational/Hermite/heurisch probes via shared probe runner)
- Phase 4: in progress (Hermite positive-quadratic regime grid and two
*(re-audit 2026-07-18: el workstream RACIONAL de Phase 4 está COMPLETO con la clausura universal LRT/root_sum (G1 cerrado); lo restante de Phase 4, si algo, es heurística transcendental)*
  rational families promoted 2026-06-10: multi-quadratic partial fractions,
  then general-degree rational integration via Ostrogradsky-Horowitz reduction
  with rational-root/biquadratic splitting — denominators of degree 3..=8,
  expanded forms and repeated factors included; the even-quartic
  symmetric descent closed x^4+4 and x^4+x^2+1 the same day; the resolvent-cubic
  descent closed non-even quartics 2026-06-11 (conservative
  denominator-nonzero condition pinned: expression-level positivity proving
  for non-even products is an observe-only candidate); the symmetric-surd
  even-quartic probe closed the Phi_12 family `c/(x^4+px^2+r)` with one
  quadratic surd (s=sqrt(r) rational, a=sqrt(2s-p) irrational:
  `1/(x^4-x^2+1)`, `1/(x^4-3x^2+4)`) 2026-06-26, differentiation-verified; the
  G1 Cap.A/B surd-split cycles then closed the composite even-quartic
  denominators `1/(x^4-4)` (real-log ratio), `1/(x^6+1)` and `1/(x^8-1)`
  (conjugate surd), plus non-constant numerators over the surd quartic
  (`x^3/(x^4-x^2+1)`, `(x^3+5)/(x^6+1)`) 2026-07-14 (`d557556ea`, `6c4d59afc`,
  `9c48574c4`), differentiation-verified;
  then Cap.C/D + residuals R1-R3 closed the remaining named G1 probes:
  Phi_5 with sqrt(5) `1/(x^5-1)` (C-iii `69a215bf6`), cube-root extensions
  `1/(x^3-2)` (Cap.D `816ab8c1a`), the irrational-real-resolvent quartic
  `1/(x^4-5)` (R2 `4fd29a4ca`) and the doubly-even octic `1/(x^8+1)` (R3
  `046aa0cd7`, beating a live SymPy wrong answer) — all differentiation-verified;
  remaining: the UNIVERSAL closure Cap.E (Lazard-Rioboo-Trager) for denominators
  whose irreducible Q-factors are not one of the hand-coded shapes (single
  irrational poles `1/(x^3-x-1)`, general quartic `1/(x^4+x+1)`, `1/(x^7-1)`).
  Cap.E scoped 2026-07-16 (`ee9f6fcf0`) into sub-cycles; E-i landed the
  standalone subresultant-PRS + Rothstein-Trager resultant primitive over Q[t]
  (`subresultant_prs.rs`), pinned against SymPy; E-ii (driver + Rioboo real
  render) and E-iii (wiring/gating) remain, with non-solvable factors (S_5)
  *(re-audit 2026-07-18: SUPERADO — Cap.E cerró 2026-07-16 (E-i…E-iv): `1/(x^3-x-1)`, `1/(x^4+x+1)`, `1/(x^7-1)` devuelven root_sum cerrados aceptados con approx()/definidas operativas)*
  gated to honest residual)
  *(re-audit 2026-07-18: las quínticas S_5 ya NO se gatean a residual: root_sum evita la resolubilidad por radicales — `1/(x^5-x-1)` recibe antiderivada cerrada)*
- Phase 5: partial (mode-boundary lane exists; trace policy summarized)
- Phase 6: started 2026-06-11 (didactic elevation; first candidate Hermite positive
  quadratic)

### Phase 0. Inventory And Ownership

Goal: map what can be shared before adding any broad method.

Tasks:

- identify current integration route owners: detection, domain, transformation,
  verification, presentation, and steps
- list existing verified families that can be routed through the backend
  boundary without changing public behavior
- add scorecard visibility for backend attempts, successes, verification
  outcomes, residual reasons, and elapsed time

Retain only if behavior remains unchanged and observability improves.

### Phase 1. Boundary Without Behavior Change

Goal: introduce the backend result boundary using an existing green family.

Good first candidates:

- a rational integration family already verified by differentiation
- a table or substitution family with clear `u`/`du` evidence
- a positive quadratic/arctangent family with explicit domain policy

Success condition:

- current public output is unchanged
- the backend boundary records method, assumptions, verification, and residual
  metadata
- tests prove the backend can be disabled without changing educational behavior

### Phase 2. Verification Service

Goal: make antiderivative verification a shared service instead of a local
post-hoc simplification path.

Tasks:

- define bounded `diff(candidate, x) ~ integrand` verification for backend
  candidates
- classify verification failures by domain, presentation, simplification,
  timeout, or true mismatch
- preserve required conditions through verification and display
- avoid deep generic residual routes when a family-specific verifier is safer

Success condition:

- accepted backend candidates are verified
- rejected candidates produce structured residual reasons
- verification cost is visible in scorecards

### Phase 3. Discovery-Only Algorithmic Probes

Goal: explore broader methods without changing public answers.

Allowed probes:

- more systematic rational integration
- Hermite-style rational reduction
- partial fractions over broader real regimes
- small heurisch-like substitution candidates with tight budgets
- limited algebraic/root substitutions where domain policy is explicit

Rules:

- start in stress/discovery, not live/frozen
- add methods through the shared probe runner so method-probe and verification
  budgets are consumed and reported uniformly
- report no-match reasons through the shared probe runner; prefer expanding a
  method only when the reason distribution shows a reusable gap rather than
  broad shape noise
- no accepted public answer until verification and domain policy are green
- log reusable failures in the combination ledger only when they reveal a
  structural weakness, not when they are malformed or duplicate examples

### Phase 4. First Public Algorithmic Family

Goal: promote one backend family that the educational route did not already
cover systematically.

A family is promotable only when it has:

- method-level definition, not a string-shaped matcher
- bounded runtime and no broad route churn
- real-domain and constant policy
- antiderivative verification
- support-matrix rows for supported, residual, and rejected regimes
- summarized trace that is honest about being algorithmic
- fallback residual behavior when disabled or unverified

Prefer rational integration before transcendental heuristics. It has clearer
algebraic invariants, domain conditions, and verification routes.

Named workstream — algebraic verification graduation (prerequisite for
completing this phase with general rational integration):

- today the verifier matches `diff(candidate)` against the integrand through
  case-by-case structural normalization (`verification_normalization`); that
  does not scale to algorithm-produced antiderivatives
- graduate rational-candidate verification to an algebraic zero test: bring
  `diff(candidate) - integrand` to rational normal form over a common
  denominator via `multipoly` and decide equality by polynomial zero testing,
  under the existing verification budgets
- structural normalization stays for non-rational shapes; do not add new
  `normalize_backend_*` cases for shapes the algebraic test can decide
- the 2026-06-10 combination-ledger learnings (builder canonicalization debt,
  folded-coefficient matching) are the two failure modes this graduation
  removes at the root
- Status (2026-06-10): landed as `verification_algebraic.rs` — a fallback
  decision procedure after the existing cascade (new evidence
  `algebraic_zero_test`); rational shapes plus square-root atoms of
  variable-free radicands via the `t^2 = radicand` quotient; existing
  evidence labels and lane counters unchanged by construction

### Phase 5. Mode And Trace Policy

Goal: expose the backend without weakening the educational UX.

Possible modes:

- default: educational route plus verified backend fallback only when safe
- educational-only: never consult algorithmic backend
- algorithmic-diagnostic: show backend attempts, rejected candidates, and
  residual reasons
- future advanced mode: allow broader backend budgets while preserving
  verification and residual policy

Trace policy:

- educational route keeps detailed rule/substitution steps
- backend route emits summarized method steps, such as "Algorithmic rational
  integration" plus verification and conditions
- diagnostic mode may expose rejected candidates, but ordinary users should see
  clear residual explanations instead of internal churn

### Phase 6. Didactic Elevation

Goal: close the gap between verified and teachable. The north star
(CALCULUS_ENGINE_STRATEGY.md) is a universal *and* educational engine, so
`algorithmic_summary` traces are transitional, not terminal.

A stable backend family is elevated by:

- authoring a real step-by-step derivation for the method (for Hermite
  positive quadratic: complete the square, substitute `u`, table
  arctan/log), reusing the educational route's step machinery instead of
  inventing a parallel one
- keeping the backend verification as the safety net behind the steps
- promoting the family's trace level from `algorithmic_summary` to
  `educational_full`, with a matrix row asserting step substrings

Elevation order should follow public usage value, not implementation order.
A family may stay summary-only when the method is genuinely beyond
curriculum scope; document that decision and its reason in the combination
ledger.

Phase status (2026-06-11): started — the canonical Hermite
positive-quadratic reciprocal emits the educational arctan derivation
(inner-derivative rule, affine argument, constant factor), pinned by the
first backend matrix row with expected_step_substrings. Root cause was
didactic-side (double-held results hidden from the substep generators;
fixed by fixpoint unwrap at the generator entry — engine untouched).
The mixed-numerator ln+arctan narration landed the same day for the
compact shapes (separate the log part, integrate the denominator's
derivative as a log, arctan rule), pinned by substrings on both compact
mixed rows. The expanded-affine shapes landed next (complete-the-square
narration recovered from the result's own ln argument, with an ln-only
branch for derivative-multiple numerators) - all six Hermite
positive-quadratic matrix rows now assert educational substrings. The
multi-quadratic partial-fraction rows landed next with a REAL
intermediate (the backend's decomposition rebuilt on a scratch context:
N/prod(q_i) -> sum of simple terms -> result). The Ostrogradsky and
quartic-descent narrations landed next (rational-part separation quoted
with the literal remaining integral, factorization, partial fractions,
term-by-term integration) - every supported backend family row in the
matrix now asserts educational substrings. Remaining: the Heurisch probe
family (diagnostic-only today) and deciding which summary-only families,
if any, are documented as beyond curriculum scope.

## Promotion Checklist

Before retaining a backend change, answer:

- Which route owns the behavior: educational, backend, shared verifier, or
  orchestration?
- Is public behavior unchanged, or is one verified algorithmic family promoted?
- What method tag and residual reasons are represented?
- How is `diff(candidate, x) ~ integrand` verified?
- What real-domain, branch, pole, component, and constant assumptions are
  represented?
- What happens when the backend is disabled?
- What happens when verification is inconclusive?
- Which support-matrix cells were added or updated?
- What runtime guardrail proves ordinary educational cases did not regress?
- Is the trace educational-full, algorithmic-summary, or diagnostic-only?
- Is the family elevated to `educational_full` steps, scheduled for didactic
  elevation (Phase 6), or documented summary-only with a reason?

## Recommended First Iterations

1. [done — model.rs] Add a backend result model and disabled-by-default boundary.
2. [done — model.rs] Route one already-supported integration family through the boundary without
   changing output.
3. [done — verification.rs] Extract a shared antiderivative verifier with structured outcomes.
4. [done — scorecard lanes] Add backend scorecard metrics for attempts, verification outcomes, residual
   reasons, and runtime.
5. [done — probe_runner.rs] Add a shared probe runner and scorecard budget split for method-probe versus
   verification-budget exhaustion.
6. [done — no-match reasons] Add method-probe candidate/no-match counts and no-match reason attribution.
7. [done — discovery probes] Add discovery-only rational integration probes.
8. [done — multi-quadratic partial fractions promoted 2026-06-10 with the
   algebraic verifier as its gate; Hermite positive-quadratic regime grid
   promoted the same day, see combination ledger] Promote one verified
   rational backend family once the boundary and verifier are stable.
9. [done — verification_algebraic.rs, 2026-06-10] Graduate rational-candidate
   verification to the multipoly algebraic zero test (see the Phase 4 named
   workstream).
10. [done — Phase 6 status, 2026-06-11] Elevate the first backend family
    (Hermite positive quadratic) to `educational_full` steps: the canonical
    reciprocal emits the arctan derivation (inner-derivative rule, affine
    argument, constant factor), pinned by the six Hermite matrix rows'
    `expected_step_substrings`.

## Stop Conditions

Pause backend work and return to architecture or educational integration when:

- backend attempts affect existing educational outputs unexpectedly
- verification relies on broad unbounded simplification
- runtime pressure spreads beyond the family under development
- domain or branch assumptions cannot be represented
- traces become misleading answer-only shortcuts
- the change is just another local integration case instead of a backend
  capability
