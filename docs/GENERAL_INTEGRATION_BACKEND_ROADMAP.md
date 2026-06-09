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

## Recommended First Iterations

1. Add a backend result model and disabled-by-default boundary.
2. Route one already-supported integration family through the boundary without
   changing output.
3. Extract a shared antiderivative verifier with structured outcomes.
4. Add backend scorecard metrics for attempts, verification outcomes, residual
   reasons, and runtime.
5. Add a shared probe runner and scorecard budget split for method-probe versus
   verification-budget exhaustion.
6. Add method-probe candidate/no-match counts and no-match reason attribution.
7. Add discovery-only rational integration probes.
8. Promote one verified rational backend family once the boundary and verifier
   are stable.

## Stop Conditions

Pause backend work and return to architecture or educational integration when:

- backend attempts affect existing educational outputs unexpectedly
- verification relies on broad unbounded simplification
- runtime pressure spreads beyond the family under development
- domain or branch assumptions cannot be represented
- traces become misleading answer-only shortcuts
- the change is just another local integration case instead of a backend
  capability
