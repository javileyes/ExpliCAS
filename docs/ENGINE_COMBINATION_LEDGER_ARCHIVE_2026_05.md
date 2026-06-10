# Engine Combination Ledger - Archive 2026-05

Entries rotated from [ENGINE_COMBINATION_LEDGER.md](ENGINE_COMBINATION_LEDGER.md).
Scorecard discovery metrics still read this file; treat it as
read-only history and do not add new entries here.

## 2026-05-26 - Discovery observe-only: shifted sqrt reciprocal-trig residual verification times out

- area:
  - calculus / post-calculus residual verification / reciprocal trig shifted
    sqrt products
- status:
  - `discovery/observe-only`
- observed:
  - while promoting compact derivative presentation for
    `sec(b - sqrt(x))*k` and `-csc(b - sqrt(x))*k`, a residual matrix attempt
    using
    `diff(result, x) - (-k*sec(b-sqrt(x))*tan(b-sqrt(x))/(2*sqrt(x)))`
    still timed out under the 4s smoke budget
  - the direct `diff(...)` probes were fast and produced the compact
    `sec/csc·tan/cot` form with only the pole condition and `x > 0`, so the
    weakness is not derivative presentation itself
- decision:
  - retain direct derivative presentation and matrix verification by exact
    compact derivative result in this cycle
  - do not force residual-equivalence promotion until a bounded residual route
    can run before reciprocal-trig target expansion
- retained learning:
  - the reusable weakness is the residual verification path for already
    compact reciprocal-trig products: parsing or simplifying the target
    integrand can expand `sec/csc/tan/cot` into sin/cos traffic and trigger
    `cycle_detected`/`depth_overflow` churn before exact cancellation
  - a future candidate should add a narrow pre-general residual signature for
    compact reciprocal-trig product equality, or teach the matrix verifier a
    bounded structural equality mode for exact post-calculus derivative shapes
- follow-up resolution:
  - retained a narrow engine-side residual signature for shifted sqrt
    reciprocal-trig derivative products; the promoted matrix rows now verify
    by residual equality without entering the previous reciprocal-trig
    expansion churn
- `local win in shared helper + move to exact call-site`
- `hotspot shift + second patch that attacks the shifted hotspot`

Bad combination patterns:

- two broad helper-level fast paths that both add traffic
- more canonicalization in already shared hot paths
- combinations whose only rationale is “both looked good locally”

The burden of proof stays the same:

- rerun the narrow local slice
- rerun the scorecard with baseline
- rerun the direct release corpus if the scorecard is good enough

## Current Entries

## 2026-05-28 - Observe-only discovery: condition display must not scale-normalize periodic arguments

- area:
  - calculus / domain-condition display / reciprocal trig residuals
- status:
  - `discovery/observe-only`
- observed:
  - while compacting residual integration conditions for
    `integrate(1/sin(x^2+0), x)` and `integrate(1/tan(x^2+0), x)`, an
    initial helper reused sign-preserving condition normalization inside
    unary function arguments
  - the local integration rows improved, but `make engine-fast` caught a
    `calculus_diff_contract` regression: `diff(sec((3*x+2)/2), x)` published
    `cos(3*x+2) != 0` instead of preserving
    `cos((3*x+2)/2) != 0`
- decision:
  - retain only additive-zero cleanup inside display-only unary condition
    arguments
  - reject scale/content normalization inside periodic arguments, even when it
    is sign-preserving for polynomial inequalities
- retained learning:
  - nonzero conditions over periodic functions are sensitive to argument scale;
    `cos(k*u) != 0` is not equivalent to `cos(u) != 0`
  - future condition-display cleanup may remove inert additive-zero noise, but
    must not reuse polynomial sign-preserving normalization inside
    `sin/cos/tan/sec/csc/cot` arguments without a zero-set proof
- follow-up resolution:
  - retained a public CLI contract for `diff(sec((3*x+2)/2), x)` that asserts
    `required_display` preserves `cos((3*x+2)/2) != 0` and explicitly rejects
    the previously observed scale-normalized `cos(3*x+2) != 0`
  - this complements the lower-level domain-normalization unit without
    broadening condition cleanup beyond additive-zero display normalization

## 2026-05-26 - Observe-only discovery: shifted sqrt-chain reciprocal trig external scale enters fragile simplification route

- area:
  - calculus / integration / sqrt-chain reciprocal trig derivative products
- status:
  - `discovery/observe-only`
- observed:
  - during the bounded sibling sweep for the retained exact sqrt-chain external
    scale candidate, `integrate(-k*sec(b-sqrt(x))*tan(b-sqrt(x))/(2*sqrt(x)), x)`
    timed out under a 5s CLI probe with repeated `cycle_detected` and
    `depth_overflow` warnings
  - the promoted representative stayed limited to `u = sqrt(x)` with an
    x-free symbolic external scale, where the antiderivative verifies by
    differentiation and the didactic trace is concrete
- decision:
  - do not promote shifted `b - sqrt(x)` orientation in this cycle
  - do not broaden the matcher around non-polynomial affine-like offsets until
    the route has a cheap fragility guard or a dedicated argument-normalization
    strategy
- resolved by:
  - later retained shifted sqrt-chain reciprocal-trig integration rows cover
    both `b - sqrt(x)` and `sqrt(x) - b` with symbolic external scale,
    preserved pole/positive-domain conditions, and bounded residual
    verification
- retained learning:
  - the reusable weakness is the interaction between shifted sqrt orientation,
    reciprocal-trig expansion, and symbolic external scale, not a malformed
    generated case
- next candidate:
  - isolate whether `b - sqrt(x)` should be accepted as a safe sqrt-chain
    orientation with bounded presentation/domain handling or intentionally left
    residual behind a fragility guard

## 2026-05-26 - Observe-only discovery: reciprocal trig derivative products do not yet share symbolic external-scale handling

- area:
  - calculus / integration / reciprocal trig derivative products
- status:
  - `discovery/observe-only`
- observed:
  - during the bounded sibling sweep for the retained hyperbolic external-scale
    integration candidate, `integrate(k*a*sec(a*x+b)*tan(a*x+b), x)` remained
    residual as `integrate(a·k·sin(a·x + b) / cos(a·x + b)^2, x)`
  - the cosecant/cotangent sibling
    `integrate(k*a*csc(a*x+b)*cot(a*x+b), x)` timed out under a 4s CLI probe
- decision:
  - do not broaden the retained hyperbolic fix into reciprocal trig products in
    this cycle
  - keep the promoted rows limited to the hyperbolic family where the matcher,
    domain policy, and didactic trace were validated
- resolved by:
  - later retained reciprocal-trig derivative product rows cover symbolic
    external scale for both `sec(u)*tan(u)` and `csc(u)*cot(u)` with explicit
    pole conditions and bounded antiderivative verification
- retained learning:
  - the reusable weakness is a missing shared symbolic external-scale policy for
    reciprocal derivative products, not a malformed generated case
  - a future candidate should first isolate why the `csc/cot` branch enters the
    slow path, then decide whether a shared cofactor-scale helper can cover trig
    and hyperbolic products without changing route order or pole conditions

## 2026-05-25 - Observe-only discovery: inverse-tangent symbolic denominator shortcut leaves condition and external-scale gaps

- area:
  - calculus / differentiation / inverse-trig root presentation
- status:
  - `observe-only`
- generated candidate:
  - generalize the retained `arctan(sqrt(r)/(2*a))` shortcut across affine
    radicands, fractional denominator scales, dual orientation, negative
    orientation, and external constant scale
- local lane:
  - `target/release/cas_cli eval 'diff(arctan(sqrt(2*x+2)/(2*a)), x)' --format json`
  - `target/release/cas_cli eval 'diff(3*arctan(sqrt(2*x+2)/(2*a)), x)' --format json`
- retained result:
  - the direct affine-radicand shortcut avoids the previous
    `depth_overflow` warning and returns the compact derivative
    `a / (sqrt(2*x + 2)*(2*a^2 + x + 1))`
  - the promoted diff matrix row now forbids `depth_overflow` for that direct
    case
  - a follow-up retained the external-scale sibling
    `3*arctan(sqrt(2*x+2)/(2*a))` with compact public output
    `3*a / (sqrt(2*x + 2)*(2*a^2 + x + 1))`; the promoted matrix row now
    forbids `depth_overflow` for that route as well
  - a later follow-up retained a core condition-normalization implication for
    `positive_gap + c*base^2 != 0` under `positive_gap > 0`, `c > 0`, and a
    visible `base != 0`, so the direct and external-scale rows no longer
    publish the redundant `2*a^2 + x + 1 != 0` condition
  - a later follow-up generalized the implication to nonnegative square terms
    without requiring a `base != 0` guard, and to
    `positive_constant + c*base^2*positive_gap != 0`; this removed the
    redundant guard from the retained numerator-scale row
    `diff(arctan(a*sqrt(x+1)), x)`
  - a later follow-up reused the same narrow implication for positive
    conditions, so `diff(asinh(a*sqrt(x+1)), x)` no longer publishes the
    redundant `a^2*x + a^2 + 1 > 0` condition while preserving `x > -1`
  - a later follow-up retained a separate open-interval implication for
    `Positive(1 - (base*sqrt(gap))^2)` dominating the equivalent expanded
    boundary `NonZero`; this removed the redundant
    `a^2*x + a^2 - 1 != 0` condition from
    `diff(atanh(a*sqrt(x+1)), x)` while preserving both real-domain guards
- observed gaps:
  - none remaining for the promoted affine-radicand denominator condition or
    numerator-scale positive-gap/open-interval condition family
- resolved by:
  - the current diff matrix covers the direct and external-scale symbolic
    rational denominator affine-radicand rows, the numerator-scale positive-gap
    row, and the corresponding inverse-hyperbolic positive-gap/open-interval
    representatives
  - 2026-05-27 reprobes of the original local lane plus the numerator-scale
    `arctan`, `asinh`, and `atanh` siblings returned compact results, no
    warnings, and only the retained real-domain conditions
- reusable weakness:
  - calculus shortcuts can avoid fragile derivative construction, but the
    condition layer needs explicit, narrow implication checks when a compact
    denominator mixes a positive calculus gap with parameter squares
  - a naive external-scale route can still degrade public output unless the
    compact result is retained behind the transformer's internal `Hold` barrier
- next candidate:
  - move to another calculus matrix cell unless a sibling exposes a distinct
    domain, trace, or presentation regime beyond the retained numerator-scale
    positive-gap and open-interval representatives

## 2026-05-24 - Observe-only discovery: limit command residuals cannot enter algebraic wrapper matrix

- area:
  - calculus / limit / residual matrix harness boundary
- status:
  - `observe-only`
- generated candidate:
  - promote a calculus residual matrix row for
    `limit((x^2-1)/(x-1),x,1)-2` under the standard algebraic wrappers
- local lane:
  - `python3 scripts/engine_calculus_residual_probe_smoke.py --matrix-residual 'limit((x^2-1)/(x-1),x,1)-2' --matrix-residual-name limit_removable_quadratic_factor --matrix-wrapper plus --matrix-wrapper nested_den --matrix-wrapper double_nested_den --expect pass --forbid-warnings --json --summary-json --ensure-release-cas-cli`
  - `target/release/cas_cli eval 'limit((x^2-1)/(x-1),x,1)' --format json`
  - `target/release/cas_cli eval '((limit((x^2-1)/(x-1),x,1)-2)+1)/(x+2)' --format json`
- observed result:
  - top-level `limit(...)` routes through the limit command and returns a safe
    finite-point residual with a warning and `x != 1` requirement
  - the same `limit(...)` nested inside an algebraic wrapper is treated as an
    undefined ordinary function: `función [limit] no definida`
  - the default residual wrapper matrix therefore cannot directly promote
    `limit` command behavior today
- resolved by:
  - the command-level `limit` matrix now owns this surface explicitly and covers
    finite removable rational cancellation as
    `finite_removable_rational_cancellation`, with result `2`, required
    condition `x != 1`, and no warning
  - the algebraic residual probe now classifies `ok=false` CLI JSON as
    `cli_error` instead of reporting a misleading `result_mismatch`, so
    command-surface boundary failures are visible as harness/observability
    findings rather than engine result mismatches
- reusable weakness:
  - calculus command coverage is asymmetric: `diff` and `integrate` can be
    composed inside the expression-level residual matrix, while `limit` still
    needs either a command-aware matrix harness or a safe expression-level
    residual function route
- next candidate:
  - add a dedicated command-level `limit` matrix for finite/residual/infinity
    policies, or define a narrow expression-level `limit` residual route before
    trying algebraic wrapper-spread promotion

## 2026-05-24 - Observe-only discovery: mul-div target narrowing changes post-calculus diff routes

- area:
  - fractions / target-kind dispatch / post-calculus differentiation presentation
- status:
  - `resolved`
- candidate:
  - narrow `SimplifyMulDivRule` to `TargetKindSet::MUL`, because
    `try_rewrite_simplify_mul_div_expr` begins by requiring a multiplication
    root through `as_mul(ctx, expr)?`
- local lane:
  - `cargo test -p cas_engine simplify_mul_div_rule_targets_mul_only -- --nocapture`
  - `cargo test -p cas_solver --test diff_step_contract_tests
    arctan_sqrt_diff_uses_post_calculus_reciprocal_root_presentation
    -- --nocapture`
  - `cargo test -p cas_cli --test integrate_contract_tests
    integrate_contract_sqrt_chain_hyperbolic_reciprocal_squares_verify
    -- --nocapture`
- local result:
  - the target-kind unit slice passed while the candidate was present
  - the focused diff and integrate presentation contracts passed while the
    candidate was present
- global result:
  - `make engine-fast` failed in `calculus_diff_contract`
  - failing cases:
    `constant_base_log_abs_diff_uses_direct_domain_safe_log_rule` exposed a
    noisy 6-step fixed-base log route, and
    `bounded_inverse_trig_sqrt_diff_uses_post_calculus_root_denominator_presentation`
    failed derivative-equivalence verification for the compact root
    presentation
- why it regressed globally:
  - even when a fraction cleanup helper is syntactically `Mul`-rooted, moving
    it into target-indexed dispatch can alter post-calculus simplification
    ordering and step visibility across diff families that were not covered by
    the local presentation probes
- what could make it combinable later:
  - first harden the post-calculus diff presentation filter/equivalence route
    for fixed-base log and bounded inverse-trig sqrt outputs, then retry any
    `SimplifyMulDivRule` dispatch narrowing under full `calculus_diff_contract`
- retry note:
  - 2026-05-24 combination retry after later calculus-presentation hardening
    still reproduced both original blockers before promotion:
    `constant_base_log_abs_diff_uses_direct_domain_safe_log_rule` emitted a
    6-step fixed-base log route, and
    `bounded_inverse_trig_sqrt_diff_uses_post_calculus_root_denominator_presentation`
    left a nonzero residual for the `(x+1)/(x+3)` arccos case
  - the narrowing was reverted; keep this entry `observe-only` until those two
    post-calculus diff routes are hardened directly
- retained follow-up:
  - 2026-05-24 calculus cycle hardened the fixed-base `log(base, abs(u))`
    derivative path by emitting the compact divided-by-`ln(base)` form directly
    and filtering terminal post-diff cleanup noise; with a local
    `SimplifyMulDivRule` target probe, the
    `constant_base_log_abs_diff_uses_direct_domain_safe_log_rule` contract
    passed
  - 2026-05-24 calculus cycle hardened bounded inverse-trig sqrt affine-quotient
    residual matching by normalizing `-(a/b)` versus `(-a)/b` quotient
    orientation; with a local `SimplifyMulDivRule` target probe,
    `bounded_inverse_trig_sqrt_diff_uses_post_calculus_root_denominator_presentation`
    passed for the `(x+1)/(x+3)` arccos residual
  - known focused blockers for a `SimplifyMulDivRule` target-narrowing retry
    are now closed; promotion is still deferred until a separate combination
    cycle reruns the full diff contract plus fast/guardrail/pressure lanes with
    the candidate patch present
- rejected retry:
  - 2026-05-24 combination retry changed only `SimplifyMulDivRule` to
    `TargetKindSet::MUL`; focused gcd-cancel, historical diff blockers,
    integration presentation probe, and `make engine-fast` passed
  - `make engine-scorecard` was terminated after the release
    `cas_cli --test integrate_contract_tests` lane spent more than 6 minutes in
    `integrate_contract_positive_sqrt_antiderivative_rationalized_residual_survives_shifted_reciprocal_difference`
  - a 3s sample showed the hot route under
    `RationalizeLinearSqrtDenRule` / `GeneralizedRationalizationRule` /
    `rationalize_diff_squares_support::try_rewrite_generalized_rationalization_expr`,
    recursively expanding through `expand_ops::distribute_single` and
    `compare_expr`
  - the narrowing was reverted; keep this entry `observe-only` until the
    positive-sqrt rationalized residual integration route is hardened or gated
    under the narrowed dispatch shape
- retained follow-up:
  - 2026-05-24 robustness cycle hardened the positive-sqrt rationalized
    residual route by resolving embedded half-power `integrate(...)` calls
    before general simplification, dropping redundant positivity requirements
    for strictly positive quadratic radicands, and compacting the nonmatching
    shifted reciprocal residual instead of sending it into rationalization and
    expansion
  - with a local `SimplifyMulDivRule` target probe, the focused
    `integrate_contract_positive_sqrt_antiderivative_rationalized_residual_survives_shifted_reciprocal_difference`
    contract passed in `0.02s`
  - this closes the known positive-sqrt focused blocker; keep the
    `SimplifyMulDivRule` promotion deferred until a separate `combination`
    cycle reruns full diff, integrate, fast, guardrail, and pressure lanes with
    the candidate narrowing present
- rejected retry:
  - 2026-05-24 combination retry changed only `SimplifyMulDivRule` to
    `TargetKindSet::MUL`; focused gcd-cancel target, the three historical
    post-calculus diff blockers, the two integration probes, `cargo fmt
    --check`, `git diff --check`, and `make engine-fast` all passed
  - `make engine-scorecard` rejected the candidate in
    `embedded_equivalence_context`: `1485/1488` passed, `3` failed, elapsed
    `19.70s`
  - failing embedded signatures were all l3 wrapper/composition rows:
    `additive_passthrough_zero x finite_telescoping`,
    `scaled_difference_zero x finite_telescoping`, and
    `common_denominator_zero x calculus_diff`
  - the narrowing was reverted; keep this entry `observe-only` until any next
    retry first hardens or gates the embedded finite-telescoping wrapper
    residuals and the common-denominator log2/sqrt calculus-diff residual under
    the narrowed dispatch shape
- retained follow-up plus rejected probe:
  - 2026-05-24 robustness cycle retained a narrower arithmetic hardening for
    reciprocal half-power residual cancellation: the matcher can now factor
    additive denominator scales such as `q*log2(x^2+1)+q` and
    `2*q*ln(2)+2*q*ln(2)*x^2`, and can compare the induced
    `scale/denominator` quotient by product cross-multiplication
  - focused units for the unfactored `log2`/`sqrt` residual variants passed,
    including both the `sqrt(base)/base` presentation and the shared
    reciprocal-sqrt denominator presentation
  - a local `SimplifyMulDivRule -> TargetKindSet::MUL` probe still left the
    embedded `common_denominator_zero x calculus_diff` row nonzero after the
    full eval pipeline, even though the parsed residual variants now collapse
    directly
  - discovery: the remaining blocker is not the direct arithmetic matcher
    alone; the narrowed dispatch shape leaves a post-calculus residual in an
    accumulated pipeline form that does not re-enter the exact common-scale
    route successfully before final presentation
  - the narrowing was reverted again; keep this entry `observe-only`. A next
    combination attempt should first harden the post-calculus residual cleanup
    call-site for this accumulated form and still address the two
    finite-telescoping embedded blockers before any promotion retry
- retained finite-telescoping follow-up:
  - 2026-05-24 combination cycle hardened finite-sum residual matching and the
    eval post-pass for embedded `sum/product` residuals
  - under a local `SimplifyMulDivRule -> TargetKindSet::MUL` probe, the
    documented embedded blockers
    `additive_passthrough_zero x finite_telescoping` and
    `scaled_difference_zero x finite_telescoping` both passed `5/5`
  - the same probe still failed `common_denominator_zero x calculus_diff` on
    the accumulated `log2`/`sqrt` residual, so the narrowing remains reverted
    and this ledger item stays `observe-only`
  - next combination retry should first harden that post-calculus accumulated
    residual form, then rerun the full embedded, fast, guardrail, and pressure
    lanes with the narrowing present before promotion
- rejected probe / discovery:
  - 2026-05-24 combination retry again applied only the local
    `SimplifyMulDivRule -> TargetKindSet::MUL` probe and targeted the remaining
    `common_denominator_zero x calculus_diff` blocker
  - the focused embedded filter still failed `2/3` with the accumulated
    `log2`/`sqrt` common-denominator residual, so the narrowing was reverted
  - minimal probes showed the rendered residual, when reparsed into a clean
    context, is accepted by the existing common-scaled reciprocal half-power
    matcher, while the eval-produced graph with the same public rendering is
    not accepted
  - discovery: the remaining blocker is not another finite-telescoping-style
    post-pass; it is a structural mismatch between eval-produced fraction-like
    calculus residuals and the current half-power/common-scale matcher input
    shape
  - no code was retained from this probe. A next attempt should first add a
    focused structural probe around the eval-produced graph shape, then harden
    the half-power matcher to consume that shape without render/reparse
    fallback before retrying the target narrowing
- retained follow-up:
  - 2026-05-24 combination cycle added the focused structural probe and
    hardened the half-power/common-scale matcher with a bounded fraction-like
    extractor that accepts a product containing exactly one division factor,
    such as the eval-produced `2 * (num / den)` residual shape
  - with that hardening, the local `SimplifyMulDivRule -> TargetKindSet::MUL`
    promotion was retained: the focused embedded
    `common_denominator_zero x calculus_diff` filter passed `3/3`, and fast,
    guardrail, and pressure scorecards all passed with `failed=0`
  - this resolves the documented mul-div target narrowing blocker without a
    render/reparse fallback, without broad fraction simplification, and without
    changing calculus domain, branch, or integration-constant policy

## 2026-05-24 - Observe-only discovery: cancel-power-fraction target narrowing changes calculus canonical presentation

- area:
  - fractions / target-kind dispatch / integration presentation
- status:
  - `resolved`
- candidate:
  - narrow `CancelPowerFractionRule` to `TargetKindSet::DIV`, because
    `try_rewrite_cancel_power_fraction_expr` begins by requiring a division
    root
- local lane:
  - `cargo test -p cas_engine gcd_cancel -- --nocapture`
  - `cargo test -p cas_cli --test integrate_contract_tests
    integrate_contract_polynomial_derivative_acosh_substitution_preserves_real_domain
    -- --nocapture`
- local result:
  - the `gcd_cancel` target-kind unit slice passed while the candidate was
    present
  - the focused integration contract failed with
    `acosh(5^(-1/2) * (x^2 + x))` instead of
    `acosh((x^2 + x) / sqrt(5))`
  - removing only the `CancelPowerFractionRule` narrowing made the focused
    integration contract pass again
- global result:
  - `make engine-scorecard` failed in `calculus_integrate_contract`
  - embedded, derive, didactic, strict simplify, diff, limit, presimplify, and
    residual matrix lanes were green before that failure
- why it regressed globally:
  - this rule participates in canonical post-integration presentation, not only
    in local fraction cancellation; moving it into target-indexed dispatch
    changes a public antiderivative argument from division-by-root form to
    negative-power multiplication form
- what could make it combinable later:
  - split the public calculus presentation dependency from the generic
    cancellation rule, or add a narrower presentation cleanup that preserves
    `a / sqrt(k)` before retrying any dispatch narrowing
- retained follow-up:
  - a subsequent calculus presentation hardening preserved `a / sqrt(k)` for
    `k^(-1/2) * a` acosh arguments, then this combination cycle retained
    `CancelPowerFractionRule` as `TargetKindSet::DIV`
  - focused acosh integration, fast, guardrail, and pressure lanes passed with
    failed=0; no domain, branch, or integration-constant policy changed

## 2026-05-24 - Observe-only discovery: product-power target narrowing surfaced calculus presentation noise

- area:
  - exponents / target-kind dispatch / post-calculus presentation trace
- status:
  - `resolved`
- candidate:
  - narrow `ProductPowerRule` from untargeted dispatch to `TargetKindSet::MUL`,
    because `try_rewrite_product_power_expr` immediately requires a `Mul` root
- local lane:
  - `cargo test -p cas_engine rules::exponents::tests -- --nocapture`
  - `cargo test -p cas_solver --test fire_exponents_tests -- --nocapture`
- local result:
  - exponents unit tests and fire exponent contracts passed
- global result:
  - `make engine-fast` failed in `calculus_diff_contract`
  - the failing case was
    `asinh_sqrt_diff_uses_post_calculus_root_denominator_presentation`
  - the public step trace exposed
    `Rationalize Product Denominator`, `Product of Powers`, and
    `Present calculus result in compact form`, violating the compact
    post-calculus presentation contract
- why it regressed globally:
  - moving this broad product-power simplifier into target-indexed dispatch can
    change which low-level simplification steps are surfaced during
    post-calculus presentation, even when the final algebraic result remains
    correct
- what could make it combinable later:
  - separate internal presentation cleanup from user-visible step emission, or
    add a narrower call-site that preserves the compact calculus trace before
    retrying product-power dispatch narrowing
- retained follow-up:
  - 2026-05-24 combination cycle added `Product of Powers` and
    `Evaluate Numeric Power` to the strict post-calculus presentation noise
    filter, then retained the `ProductPowerRule` `TargetKindSet::MUL`
    narrowing
  - the previously failing
    `asinh_sqrt_diff_uses_post_calculus_root_denominator_presentation`
    contract now passes, so the runtime dispatch narrowing no longer leaks a
    rationalize/product-power/present round trip into public `diff` steps
  - no calculus domain, branch, or constant policy changed; this resolves the
    recorded presentation-noise blocker rather than adding a new calculus
    family

## 2026-05-23 - Observe-only discovery: affine hyperbolic odd-power antiderivative residuals cycle under equivalence

- area:
  - calculus / integration / hyperbolic odd powers / post-calculus
    equivalence
- status:
  - `resolved`
- candidate:
  - extend pure odd-power primitives from `sinh(x)^3`, `cosh(x)^3`,
    `sinh(x)^5`, and `cosh(x)^5` to shifted or scaled affine arguments such
    as `sinh(2*x+1)^3`
- local probe:
  - direct `integrate(...)` returns a closed-form primitive quickly for affine
    arguments
  - strict residual simplification leaves terms such as
    `sinh(x) * (cosh(x)^2 - 1) - sinh(x)^3`
  - public `equiv(diff(integrate(sinh(2*x+1)^3,x),x),sinh(2*x+1)^3)`
    repeatedly reports simplifier cycle warnings and exceeded the 8s probe
    budget
- why not retained:
  - this is not a formula error; it exposes a reusable post-calculus
    simplification/equivalence gap around hyperbolic Pythagorean products with
    shifted/scaled arguments
- next useful probe:
  - add a bounded normalizer for `f(u) * (cosh(u)^2 - 1)` /
    `f(u) * (1 + sinh(u)^2)` when it directly reduces a residual, then retry
    the affine odd-power promotion
- follow-up probe:
  - 2026-05-23 robustness cycle retained the direct-argument
    `sinh(u)*(cosh(u)^2 - 1) - sinh(u)^3 -> 0` bridge in the active
    arithmetic cancellation route
  - affine public probes such as
    `sinh(2*x+1)*(cosh(2*x+1)^2 - 1) - sinh(2*x+1)^3` and the already
    supported `cosh(2*x+1)*(1+sinh(2*x+1)^2) - cosh(2*x+1)^3` still exceed the
    8s smoke budget before promotion
  - do not promote affine odd-power integration until the public simplifier has
    a root-first or pre-child exact bridge for these affine factored
    hyperbolic residuals
- retained follow-up:
  - 2026-05-23 robustness cycle added that root-first exact bridge by invoking
    the existing hyperbolic Pythagorean factor cancellation before child
    simplification
  - affine residual probes now simplify to `0` without cycle warnings, but
    `integrate(sinh(2*x+1)^3, x)` still returns unevaluated in the public CLI;
    the full `equiv(diff(integrate(...),x), ...)` probe still exceeds the 8s
    budget because the affine primitive is not retained yet
  - a later calculus cycle can implement/promote affine odd-power integration
    as a separate candidate using this root bridge as the verification
    prerequisite
- observe-only follow-up:
  - 2026-05-23 calculus candidate extended the existing `sinh/cosh(x)^(3|5)`
    primitive generator to scaled affine arguments and added a compact-hold
    presentation guard
  - internal `cas_math` generation passed, and public `integrate(sinh(2*x+1)^3,
    x)` / `integrate(cosh(2*x+1)^3, x)` returned closed forms quickly without
    warnings under the guard
  - promotion was rejected because the required verification probes
    `diff(integrate(sinh(2*x+1)^3,x),x)-sinh(2*x+1)^3` and
    `equiv(diff(integrate(sinh(2*x+1)^3,x),x),sinh(2*x+1)^3)` still exceeded
    the 15s smoke budget with repeated cycle warnings
  - explicit equivalent primitive shapes did not solve the verification route;
    one denominator-style shape also triggered deep expansion of
    `cosh(2*x+1)` into `sinh(1)`/`cosh(1)` terms
  - next retained candidate should target the post-diff residual path or prevent
    affine hyperbolic addition expansion during verification before retrying
    public affine odd-power integration
- retained follow-up:
  - 2026-05-23 robustness cycle added a pre-general-simplification derivative
    and residual shortcut for affine hyperbolic cubic primitive forms such as
    `1/2*(1/3*cosh(2*x+1)^3-cosh(2*x+1))`
  - explicit public residuals now close before the fragile generic derivative
    expansion route, avoiding the previous cycle warnings and timeouts
  - this intentionally does not promote affine odd-power `integrate(...)` yet;
    the next calculus retry should use this derivative verification path as a
    prerequisite
- observe-only follow-up:
  - 2026-05-23 calculus cycle promoted the positive-slope affine cubic subset,
    for example `integrate(sinh(2*x+1)^3,x)` and
    `integrate(cosh(2*x+1)^3,x)`, using the retained derivative residual route
  - the negative-orientation probe `integrate(sinh(1-2*x)^3,x)` still produced
    repeated simplifier cycle warnings before promotion, even though the
    explicit primitive residual verifies quickly
  - the retained scope is therefore limited to positive affine slopes; a future
    sign/orientation candidate should make the negative affine presentation
    route preserve the compact primitive before broadening this integration
    family
- retained follow-up:
  - 2026-05-23 robustness cycle bounded the hyperbolic angle sum/difference
    cancellation fallback so it only expands direct `sinh/cosh(a±b)` calls and
    simple scalar multiples, not arbitrary powered/product expressions
  - the explicit negative-orientation affine cubic primitives now simplify
    without cycle/depth warnings, while public `sinh(x+y)` cancellation still
    closes through the direct expansion route
  - integration promotion remains deferred; the next calculus candidate can
    retry `integrate(sinh(1-2*x)^3,x)` and `integrate(cosh(1-2*x)^3,x)` against
    this retained robustness prerequisite
- retained follow-up:
  - 2026-05-23 calculus cycle promoted the negative-slope affine cubic subset:
    `integrate(sinh(1-2*x)^3,x)` and `integrate(cosh(1-2*x)^3,x)` now return
    public primitives and verify through the bounded residual route
  - the retained code change was the minimal slope gate broadening from
    positive affine coefficient to nonzero affine coefficient; no new domain,
    branch, or constant-of-integration condition was introduced
  - this closes the prior orientation blocker without adding integration search;
    broader affine odd powers remain deliberately out of scope
- retained follow-up:
  - 2026-05-23 robustness cycle extended the explicit affine hyperbolic
    primitive residual shortcut from cubic to fifth-power primitives
  - before the fix,
    `diff(1/2*(1/5*cosh(2*x+1)^5-2/3*cosh(2*x+1)^3+cosh(2*x+1)),x)-sinh(2*x+1)^5`
    exceeded the 8s smoke budget, while the analogous `cosh` residual returned
    a nonzero factored hyperbolic expression
  - positive- and negative-slope explicit `sinh/cosh` fifth primitive residuals
    now reduce to `0` through the public residual route without warnings
  - affine fifth `integrate(...)` remains intentionally unpromoted in this
    cycle; the next calculus candidate can retry that gate against this
    retained residual prerequisite
- retained follow-up:
  - 2026-05-23 calculus cycle promoted the minimal affine fifth subset:
    `integrate(sinh/cosh(2*x+1)^5,x)` plus the negative-slope
    `integrate(sinh/cosh(1-2*x)^5,x)` orientations now return public
    primitives
  - the retained implementation removed only the temporary non-affine guard
    for power 5 and kept the public compact-presentation predicate scoped to
    nontrivial affine arguments, preserving the existing pure `sinh/cosh(x)^5`
    public form
  - all promoted fifth cases verify through the bounded public residual route
    without new domain, branch, warning, or constant-of-integration conditions
- resolved by:
  - 2026-05-23 calculus and robustness follow-ups retained the bounded
    verification path and public affine odd-power integration coverage needed
    for the original cubic/fifth discovery scope
  - the public contracts also now cover affine seventh explicit residuals and
    promoted positive- and negative-slope seventh primitives, so this ledger
    item should no longer count as open generated-discovery pressure

## 2026-05-23 - Observe-only discovery: shifted arcsin residual condition implication exceeds exact smoke expectations

- area:
  - calculus / integration residuals / inverse-trig kernels / domain condition
    representation
- status:
  - `superseded`
- candidate:
  - promote
    `diff(integrate(1/sqrt(4-(x+1)^2),x),x)-1/sqrt(4-(x+1)^2)`
    into the calculus residual smoke matrix
- local probe:
  - the custom matrix reduced all 12 wrapper cases to the expected algebraic
    results, but failed exact required-condition checks when the probe expected
    `4 - (x + 1)^2`
  - the public conditions are emitted as the equivalent polynomial
    `3 - x^2 - 2·x` plus the stronger interval `-3 < x < 1`
  - some nested denominator cases omit explicit `x + 3` / `x + 4` conditions
    because the interval already implies those denominators are nonzero
- why not retained:
  - this is not an integration-result failure; it exposes that the residual
    smoke harness currently checks required conditions by exact display string
    and cannot express equivalent domain predicates or implied denominator
    guards
- next useful probe:
  - isolate `plus` and `nested_den` wrappers with the actual emitted
    conditions, then decide whether the next retained change should canonicalize
    radicand-domain displays in the engine or teach the harness a conservative
    condition implication/equivalence check
- superseded by:
  - the residual smoke harness now accepts this conservative condition shape:
    `4 - (x + 1)^2` can match the equivalent public display
    `3 - x^2 - 2·x`, and the interval `-3 < x < 1` can satisfy wrapper
    denominator requirements such as `x + 3` and `x + 4`
  - `shifted_arcsin_kernel` is now promoted as a live residual-matrix
    representative with the factored radicand-domain expectation retained

## 2026-05-23 - Rejected broad wire no-op payload filter

- area:
  - didactic wire payload / calculus presentation
- status:
  - `rejected`
- local lane:
  - `cargo test --release -q -p cas_cli --test wire_smoke_tests test_eval_json_derive_calculus_diff_shadow_omits_noop_wire_steps -- --exact --nocapture`
- local win:
  - a broad `before == after` and no-substep wire-step filter removed the
    visible no-op from the calculus-diff derive shadow payload for
    `derive diff(arctan(sqrt(x)),x), 1/(2*sqrt(x)*(x+1))`
- global result:
  - `make engine-scorecard` rejected the broad filter in
    `calculus_integrate_contract`: `integrate_contract_sparse_quartic_exp_by_parts_keeps_direct_trace`
    expected the retained direct integration trace to contain two steps, but
    the broad filter collapsed it to one
- why it regressed globally:
  - display equality alone is not a safe proxy for didactic redundancy; some
    calculus traces intentionally retain a rule step even when the rendered
    `before` and `after` strings match
- what could make it combinable later:
  - keep no-op wire filtering rule-scoped, or add explicit trace metadata that
    marks a step as presentation-only instead of inferring it from rendered
    equality

## 2026-05-23 - Retained runtime: bounded exact-zero noise stripping for sqrt-chain cosh residual

- area:
  - calculus / post-calculus residuals / exact additive noise / wrapper
    runtime
- status:
  - `retained`
- candidate:
  - promote `sqrt_chain_cosh_recip_square` after fixing its reproducible
    `plus_double_noise` timeout under the calculus residual wrapper matrix
- local probe:
  - before the fix, the custom residual matrix passed 11/12 and timed out only
    on `sqrt_chain_cosh_recip_square:plus_double_noise` under the 8s smoke
    budget
  - bounded exact-zero noise stripping in post-calculus additive contexts
    reduced the custom probe to 12/12 and allowed the live matrix promotion
- global learning:
  - a blind version of the stripping won locally but lost globally by causing
    `arctan_sqrt_additive_trig` noise wrappers to time out; that family keeps
    a specialized residual route and must not be pre-stripped by this helper
- retained constraint:
  - the helper strips exact zero noise only for bounded post-calculus additive
    contexts outside `atan`/`arctan`, preserving the established arctan route
- promotion:
  - `sqrt_chain_cosh_recip_square` is now a live residual-matrix family with
    11 default wrappers and `x > -1/3` as the base domain requirement

## 2026-05-23 - Observe-only discovery: sqrt-chain residual wrappers lose wrapper conditions

- area:
  - calculus / integration residuals / sqrt-chain substitutions / wrapper
    condition propagation
- status:
  - `superseded`
- candidate:
  - promote generated sqrt-chain reciprocal trig and hyperbolic residuals into
    the calculus residual smoke matrix
- local probe:
  - `sqrt_chain_sec_log` and `sqrt_chain_csc_log` reduced all 12 wrapper cases
    to the expected algebraic results, but every case dropped wrapper
    denominator requirements such as `x + 2` / `x + 3`, retaining only the
    base `cos(sqrt(3·x + 1))` or `sin(sqrt(3·x + 1))` plus `3·x + 1`
  - `sqrt_chain_cosh_recip_square` showed the same condition-loss signature and
    one `plus_double_noise` timeout under the 8s smoke budget
- why not retained:
  - the failures are not result mismatches; they point to reusable condition
    propagation and boundedness gaps that need a focused engine/robustness
    iteration rather than promotion as coverage
- next useful probe:
  - isolate a minimal wrapper where the residual result is `1 / (x + 2)` but
    only the sqrt-chain base conditions survive, then fix condition merge before
    adding these families to the live residual matrix
- superseded by:
  - current custom residual probes for `sqrt_chain_sec_log`,
    `sqrt_chain_csc_log`, and `sqrt_chain_cosh_recip_square` now pass 12/12
    with wrapper denominator conditions preserved and no warnings
  - the live residual matrix now promotes the stable minimal representatives
    for these sqrt-chain condition-propagation paths

## 2026-05-20 - Retained robustness: additive-pair residual closure for sec-log reciprocal-root arctan

- area:
  - calculus / differentiation / post-calculus residuals / wrapper spread
- status:
  - `retained`
- candidate:
  - resolve a small opposite-signed additive pair that contains a post-calculus
    residual before the general simplifier handles the surrounding wrapper
- retained:
  - `((diff(arctan(sqrt(sec(x)+ln(x)+1/sqrt(x)+x)),x) - ...)+1)/(x+2)`
    now returns `1 / (x + 2)` in 0.237s instead of timing out with an 8s smoke
    budget
  - the retained rewrite carries the wrapper denominator guard internally; the
    public JSON suppresses `x + 2 != 0` because existing requires include
    `x > 0`, which makes `x + 2 != 0` redundant
  - the existing promoted `arctan_sqrt_additive_trig` matrix stays
    `pass=11`, `fail=0`, `timeout=0`
  - `engine-fast`, `engine-scorecard`, and `engine-scorecard-pressure` remain
    green
- partial rejection:
  - the same generated residual still times out in `one_minus` and
    `minus_const` wrappers with a 4s smoke budget; those orientations need a
    separate bounded candidate rather than broadening this retained patch
- learning:
  - for this family, direct naked residual closure was already available; the
    missing capability was additive-context pair extraction before the wrapper
    enters general simplification

## 2026-05-20 - Retained observability: custom residual wrapper smoke matrix

- area:
  - tooling / calculus residual discovery / wrapper matrix
- status:
  - `retained`
- candidate:
  - add `--matrix-residual` and `--matrix-residual-name` to
    `engine_calculus_residual_probe_smoke.py` so a generated calculus residual
    can be wrapped through the standard smoke matrix without a one-off Python
    probe
- retained:
  - custom residual runs can be filtered with `--matrix-wrapper` and can reuse
    `--require`, timeout, warning, slow, and JSON status checks
  - the previous observe-only residual
    `diff(arctan(sqrt(sec(x)+ln(x)+1/sqrt(x)+x)),x) - ...` is now reproducible
    as `arctan_sqrt_sec_log_recip_root:plus` and reports `timeout=1` with a
    2s budget
  - the existing `arctan_sqrt_additive_trig` filtered matrix remains
    `pass=11`, `timeout=0`
  - `engine-fast` remains green
- learning:
  - the attempted engine patch that merely enabled the small-additive arctan
    resolver for child residuals did not change the timeout, so the next engine
    iteration needs matcher-level inspection rather than another registration
    change

## 2026-05-20 - Retained observability: calculus residual smoke matrix filters

- area:
  - tooling / calculus residual discovery / wrapper matrix
- status:
  - `retained`
- candidate:
  - add `--matrix-base` and `--matrix-wrapper` filters to
    `engine_calculus_residual_probe_smoke.py`
- retained:
  - the smoke harness can isolate a single residual family such as
    `arctan_sqrt_additive_trig` with 11 cases instead of running the full
    matrix
  - filtered run `--default-matrix --matrix-base arctan_sqrt_additive_trig`
    reports `pass=11`, `fail=0`, `timeout=0`
  - `engine-fast` and guardrail remain green
- learning:
  - wrapper-spread discovery needs first-class filters; ad hoc probes made it
    too easy to conflate a failed generated candidate with a retained engine
    change

## 2026-05-20 - Observe-only discovery: sec-log reciprocal-root arctan residual wrapper timeouts

- area:
  - calculus / differentiation / post-calculus residuals / wrapper spread
- status:
  - `observe-only`
- resolved by:
  - 2026-05-22 coverage promotion:
    `sec_log_reciprocal_sqrt_arctan_residual_subtractive_wrappers_stay_bounded`
    covers the minimal `one_minus` and `minus_const` public wrappers after the
    bounded child-residual rewrite was retained
- candidate:
  - extend child-level residual closure from `arctan(sqrt(additive trig))` to
    `arctan(sqrt(sec(x)+ln(x)+1/sqrt(x)+x))`
- local probe:
  - naked residual resolves to `0` with required `x > 0`,
    `sec(x)+ln(x)+1/sqrt(x)+x > 0`, `cos(x) != 0`
  - all 11 smoke-style wrappers around the residual timed out at 8s before
    promotion
- why not retained:
  - adding `try_diff_arctan_sqrt_small_additive_elementary_residual_zero_local`
    to child-level closure did not change the wrapper timeouts, so the real
    matching signature is not the same as the previous retained family
- next useful probe:
  - use the new smoke matrix filters plus a minimal temporary base to inspect
    which matcher closes the naked residual before attempting another engine
    patch

## 2026-05-20 - Retained robustness: arctan sqrt additive-trig residual wrapper spread

- area:
  - calculus / differentiation / post-calculus residuals / wrapper spread
- status:
  - `retained`
- candidate:
  - resolve the narrow `diff(arctan(sqrt(additive trig polynomial)), x)` residual when it appears as a child inside simple additive, multiplicative, and denominator wrappers, then let the existing simplifier process the wrapper
- retained:
  - the residual for `diff(arctan(sqrt(sin(2*x)+cos(x)+4)), x)` now passes all default smoke wrappers instead of timing out under `(({residual})+1)/(x+2)` and related contexts
  - the calculus residual smoke matrix expanded from 201 to 212 cases and passes with `fail=0`, `timeout=0`, `slow=0`
  - existing integration/trig nested-denominator presentation expectations remain unchanged after narrowing the rewrite to the new arctan-sqrt family
  - `fast`, `guardrail`, and `pressure` profiles stay at `failed = 0`
- learning:
  - direct residual closure is not enough; post-calculus residuals need narrow child-level closure for wrapper spread, but broad child rewriting can regress presentation of unrelated families

## 2026-05-20 - Retained robustness: arctan sqrt additive-trig residual short-circuit

- area:
  - calculus / differentiation / post-calculus residuals / arctan root denominators
- status:
  - `retained`
- candidate:
  - add a narrow early residual resolver for `diff(arctan(sqrt(additive trig polynomial)), x)` so the compact derivative presentation can cancel against a factored expected target without entering general simplification
- retained:
  - `diff(arctan(sqrt(sin(2*x)+cos(x)+4)), x) - (cos(2*x)-sin(x)/2)/(sqrt(sin(2*x)+cos(x)+4)*(sin(2*x)+cos(x)+5))` now resolves to `0` in the focused contract
  - direct presentation remains compact as `(cos(2*x)-1/2*sin(x))/(sqrt(sin(2*x)+cos(x)+4)*(sin(2*x)+cos(x)+5))`
  - bounded-positive radicands keep no public required conditions
  - `fast`, `guardrail`, and `pressure` profiles stay at `failed = 0`
- learning:
  - direct post-calculus presentation and residual cancellation need separate early hooks; relying on general fraction/equivalence cleanup can turn a correct residual into a timeout

## 2026-05-20 - Retained calculus: arctan sqrt additive-trig radicand presentation

- area:
  - calculus / differentiation / post-calculus presentation / arctan root
    denominators
- status:
  - `retained`
- candidate:
  - reuse the compact additive-trig square-root derivative presentation for
    `arctan(sqrt(radicand))` so the public derivative keeps the denominator as
    `sqrt(radicand) * (radicand + 1)` instead of expanding the shifted radicand
- retained:
  - `diff(arctan(sqrt(sin(x^2)+cos(x)+4)), x)` now renders as
    `(2*x*cos(x^2)-sin(x))/(2*sqrt(sin(x^2)+cos(x)+4)*(sin(x^2)+cos(x)+5))`
  - the compact residual for that promoted expression collapses to `0`
  - the bounded-positive radicand keeps `required_conditions` empty
  - `fast`, `guardrail`, and `pressure` profiles stay at `failed = 0`
- partial rejection:
  - `diff(arctan(sqrt(sin(2*x)+cos(x)+4)), x)` also renders compactly, but the
    matching residual
    `diff(arctan(sqrt(sin(2*x)+cos(x)+4)), x) - (cos(2*x)-sin(x)/2)/(sqrt(sin(2*x)+cos(x)+4)*(sin(2*x)+cos(x)+5))`
    was not promoted because the focused residual contract did not complete in
    a reasonable inner-loop window
- learning:
  - direct post-calculus presentation can be improved safely by reusing the
    existing root derivative route, but residual cancellation for half-scaled
    double-angle arctan/root fractions still needs a separate bounded
    cancellation candidate before promotion

## 2026-05-20 - Retained coverage: non-unit exp-chain root derivative factor

- area:
  - calculus / differentiation / post-calculus presentation / exp-chain root
    denominators
- status:
  - `retained`
- candidate:
  - promote the `exp(sin(2*x))` orientation of the bounded exp-chain root
    derivative presentation so the chain factor `2*cos(2*x)` is preserved in
    the compact common-denominator numerator
- retained:
  - `diff(sqrt(exp(sin(2*x))+ln(x)+sqrt(x)+1/sqrt(x)+x), x)` stays compact as
    `(2*sqrt(x)+2*x*sqrt(x)+4*x*cos(2*x)*sqrt(x)*e^sin(2*x)+x-1)/(4*x*sqrt(x)*sqrt(ln(x)+sqrt(x)+e^sin(2*x)+1/sqrt(x)+x))`
  - `diff(arctan(sqrt(exp(sin(2*x))+ln(x)+sqrt(x)+1/sqrt(x)+x)), x)` reuses
    the same compact numerator and denominator with the expected
    `(radicand+1)` factor
  - residuals for both promoted expressions collapse to `0`
  - required conditions remain explicit: `x > 0` and radicand positivity
  - `fast`, `guardrail`, and `pressure` profiles stay at `failed = 0`
- learning:
  - the bounded exp-chain route handles simple non-unit chain factors without
    new engine code; promoting this orientation protects a distinct chain-rule
    path beyond unit `sin(x)` and sign-only variants

## 2026-05-20 - Retained coverage: negative-scale exp-chain root derivative orientation

- area:
  - calculus / differentiation / post-calculus presentation / exp-chain root
    denominators
- status:
  - `retained`
- candidate:
  - promote the `-exp(sin(x))` orientation of the bounded exp-chain root
    derivative presentation so a negative external scale is preserved in the
    compact common-denominator numerator
- retained:
  - `diff(sqrt(-exp(sin(x))+ln(x)+sqrt(x)+1/sqrt(x)+x), x)` stays compact as
    `(2*sqrt(x)+2*x*sqrt(x)+x-2*x*cos(x)*sqrt(x)*e^sin(x)-1)/(4*x*sqrt(x)*sqrt(ln(x)+sqrt(x)+1/sqrt(x)+x-e^sin(x)))`
  - `diff(arctan(sqrt(-exp(sin(x))+ln(x)+sqrt(x)+1/sqrt(x)+x)), x)` reuses
    the same compact numerator and denominator with the expected
    `(radicand+1)` factor
  - residuals for both promoted expressions collapse to `0`
  - required conditions remain explicit: `x > 0` and radicand positivity
  - `fast`, `guardrail`, and `pressure` profiles stay at `failed = 0`
- learning:
  - the bounded exp-chain route handles negative external scale separately
    from negative inner derivative; promoting this orientation protects a
    distinct sign path without new engine code

## 2026-05-20 - Retained coverage: negative exp-chain root derivative orientation

- area:
  - calculus / differentiation / post-calculus presentation / exp-chain root
    denominators
- status:
  - `retained`
- candidate:
  - promote the `exp(cos(x))` orientation of the bounded exp-chain root
    derivative presentation so the negative inner derivative is preserved in
    the compact common-denominator numerator
- retained:
  - `diff(sqrt(exp(cos(x))+ln(x)+sqrt(x)+1/sqrt(x)+x), x)` stays compact as
    `(2*sqrt(x)+2*x*sqrt(x)+x-2*x*sin(x)*sqrt(x)*e^cos(x)-1)/(4*x*sqrt(x)*sqrt(ln(x)+sqrt(x)+e^cos(x)+1/sqrt(x)+x))`
  - `diff(arctan(sqrt(exp(cos(x))+ln(x)+sqrt(x)+1/sqrt(x)+x)), x)` reuses
    the same compact numerator and denominator with the expected
    `(radicand+1)` factor
  - residuals for both promoted expressions collapse to `0`
  - required conditions remain explicit: `x > 0` and radicand positivity
  - `fast`, `guardrail`, and `pressure` profiles stay at `failed = 0`
- learning:
  - the bounded exp-chain route handles negative inner derivatives without a
    new engine patch; promoting this orientation protects the sign behavior
    that `exp(sin(x))` alone did not cover

## 2026-05-20 - Retained robustness: bounded exp-chain root derivative presentation

- area:
  - calculus / differentiation / post-calculus presentation / exp-chain root
    denominators
- status:
  - `retained`
- candidate:
  - add a narrow exponential-chain derivative collector for bounded trig
    inners so `exp(sin(x))` can reuse the already-retained
    `ln(x)+sqrt(x)+1/sqrt(x)` common-denominator presentation
- retained:
  - `diff(sqrt(exp(sin(x))+ln(x)+sqrt(x)+1/sqrt(x)+x), x)` now returns with
    the compact denominator
    `4*x*sqrt(x)*sqrt(ln(x)+sqrt(x)+e^sin(x)+1/sqrt(x)+x)`
  - `diff(arctan(sqrt(exp(sin(x))+ln(x)+sqrt(x)+1/sqrt(x)+x)), x)` reuses
    the compact inner denominator and appends the expected `(radicand+1)`
    factor
  - residuals for both promoted expressions collapse to `0`
  - required conditions remain explicit: `x > 0` and radicand positivity;
    `exp(sin(x))` adds no domain condition
  - `fast`, `guardrail`, and `pressure` profiles stay at `failed = 0`
- learning:
  - the timeout was not a missing global chain-rule capability; it was a
    presentation extractor boundary where only linear exponential arguments
    were admitted into the bounded denominator route
  - collecting only bounded trig inners keeps the route narrow and avoids
    broad expression search

## 2026-05-20 - Retained robustness: exp-log sqrt reciprocal-root derivative presentation

- area:
  - calculus / differentiation / post-calculus presentation / mixed root
    denominators
- status:
  - `retained`
- candidate:
  - let the bounded common-denominator root derivative presentation handle
    `exp(x)+ln(x)+sqrt(x)+1/sqrt(x)+x`, even without a reciprocal-trig term,
    when the already-supported `ln(x)`, `sqrt(x)`, and `1/sqrt(x)` components
    provide the full denominator witness
- retained:
  - `diff(sqrt(exp(x)+ln(x)+sqrt(x)+1/sqrt(x)+x), x)` now returns with the
    compact denominator
    `4*x*sqrt(x)*sqrt(ln(x)+sqrt(x)+e^x+1/sqrt(x)+x)`
  - `diff(arctan(sqrt(exp(x)+ln(x)+sqrt(x)+1/sqrt(x)+x)), x)` reuses the
    compact inner denominator and appends the expected `(radicand+1)` factor
  - residuals for both promoted expressions collapse to `0`
  - required conditions remain explicit: `x > 0` and radicand positivity
  - `fast`, `guardrail`, and `pressure` profiles stay at `failed = 0`
- learning:
  - the common-denominator `ln+sqrt+1/sqrt` route was artificially tied to
    reciprocal-trig terms; removing that requirement for the already-bounded
    denominator witness avoids a timeout without broadening global
    simplification

## 2026-05-20 - Observe-only discovery: nonlinear exp-chain in mixed root denominator

- area:
  - calculus / differentiation / post-calculus presentation / exp-chain root
    denominators
- status:
  - `superseded`
- discovery:
  - `diff(sqrt(exp(sin(x))+ln(x)+sqrt(x)+1/sqrt(x)+x), x)` and the matching
    `arctan(sqrt(...))` wrapper still time out after the retained `exp(x)`
    route
  - a residual probe exposed repeated depth-overflow warnings around the
    derivative of `e^sin(x)` inside the same denominator shape
- reason not retained:
  - this is not the same capability as the retained linear-exponential route;
    the current presentation extractor recognizes linear exponential terms, so
    `exp(sin(x))` needs a separate bounded chain-derivative presentation
    hypothesis
- next combinable hypothesis:
  - add a narrow exp-chain derivative collector for bounded inner derivatives
    such as `sin(x)` only if it can reuse the same denominator construction and
    preserve `failed = 0` in embedded and pressure profiles
- superseded by:
  - `2026-05-20 - Retained robustness: bounded exp-chain root derivative
    presentation`

## 2026-05-20 - Retained robustness: triple root-denominator derivative presentation

- area:
  - calculus / differentiation / post-calculus presentation / mixed root
    denominators
- status:
  - `retained`
- candidate:
  - compose the bounded reciprocal-trig root derivative presentation when a
    radicand contains `ln(x)`, `sqrt(x)`, `1/sqrt(x)`, and a linear term, so
    the result uses the compact denominator
    `4*x*sqrt(x)*sqrt(radicand)` instead of falling into a timeout
- retained:
  - `diff(sqrt(sec(x)+ln(x)+sqrt(x)+1/sqrt(x)+x), x)` now returns with the
    compact denominator
    `4*x*sqrt(x)*sqrt(sec(x)+ln(x)+sqrt(x)+1/sqrt(x)+x)`
  - `diff(sqrt(csc(x)+ln(x)+sqrt(x)+1/sqrt(x)+x), x)` now returns with the
    compact denominator
    `4*x*sqrt(x)*sqrt(csc(x)+ln(x)+sqrt(x)+1/sqrt(x)+x)`
  - the corresponding `arctan(sqrt(...))` wrappers reuse the compact inner
    denominator and append the expected `(radicand+1)` factor
  - residuals for all four promoted expressions collapse to `0`
  - required conditions remain explicit: `x > 0`, radicand positivity, and
    `sin(x) != 0` or `cos(x) != 0`
  - `fast`, `guardrail`, and `pressure` profiles stay at `failed = 0`
- learning:
  - the previously retained `ln+sqrt` and `ln+1/sqrt` routes were not
    automatically composable; explicit bounded composition avoids a
    no-termination path without changing global simplification

## 2026-05-20 - Retained calculus: reciprocal-sqrt log orientation presentation

- area:
  - calculus / differentiation / post-calculus presentation / term-order
    robustness
- status:
  - `retained`
- candidate:
  - preserve a discovered `1/sqrt(x)` component when it appears before the
    `ln(x)` term that establishes the common denominator, instead of degrading
    it immediately to an `x^(-3/2)` derivative term
- retained:
  - `diff(sqrt(csc(x)-2*ln(x)+3/sqrt(x)+x), x)` now returns with the compact
    denominator `4*x*sqrt(x)*sqrt(csc(x)-2*ln(x)+3/sqrt(x)+x)`
  - the corresponding `arctan(sqrt(...))` derivative reuses that compact inner
    denominator and appends the expected `(radicand+1)` factor
  - residuals for both promoted expressions collapse to `0` before generic
    cleanup
  - required conditions remain explicit: `x > 0`, radicand positivity, and
    `sin(x) != 0`
  - `fast`, `guardrail`, and `pressure` profiles stay at `failed = 0`
- learning:
  - the previous reciprocal-sqrt presentation route was correct only when the
    log denominator was discovered before the reciprocal-root term; retaining
    the structural reciprocal-root signal until route selection makes the
    presentation orientation-robust

## 2026-05-20 - Retained calculus: reciprocal-sqrt log root derivative presentation

- area:
  - calculus / differentiation / post-calculus presentation / reciprocal
    square-root denominators
- status:
  - `retained`
- candidate:
  - keep `1/sqrt(x)` as a structured reciprocal-root component when a
    reciprocal-trig square-root radicand also contains `ln(x)`, so the final
    derivative can common-denominate over `4*x*sqrt(x)*sqrt(radicand)` instead
    of exposing `x*x^(-3/2)` in the numerator
- retained:
  - `diff(sqrt(sec(x)+ln(x)+1/sqrt(x)+x), x)` now returns
    `(2*sqrt(x)+2*x*sqrt(x)+2*x*tan(x)*sec(x)*sqrt(x)-1)/(4*x*sqrt(x)*sqrt(sec(x)+ln(x)+1/sqrt(x)+x))`
  - the `csc/cot` dual uses the same compact denominator and avoids raw
    half-powers
  - the corresponding `arctan(sqrt(...))` derivatives reuse the compact inner
    denominator and add the expected `(radicand+1)` factor
  - residuals for all four promoted expressions collapse to `0` before generic
    cleanup
  - required conditions remain explicit: `x > 0`, radicand positivity, and the
    relevant reciprocal-trig nonzero denominator
  - `fast`, `guardrail`, and `pressure` profiles stay at `failed = 0`
- learning:
  - the derivative was already correct; the retained value is preserving a
    reciprocal-root presentation invariant instead of letting the local route
    degrade into `x^(-3/2)` noise

## 2026-05-20 - Retained robustness: sec/csc log-sqrt root derivative presentation

- area:
  - calculus / differentiation / post-calculus presentation / reciprocal trig
    with mixed log and radical denominators
- status:
  - `retained`
- candidate:
  - extend the bounded reciprocal-trig square-root derivative presentation so
    radicands such as `sec(x)+ln(x)+sqrt(x)+x` and
    `csc(x)+ln(x)+sqrt(x)+x` can combine the `ln(x)` denominator and the
    `sqrt(x)` derivative denominator directly
- retained:
  - `diff(sqrt(sec(x)+ln(x)+sqrt(x)+x), x)` now returns directly as
    `(2*sqrt(x)+2*x*sqrt(x)+2*x*tan(x)*sec(x)*sqrt(x)+x)/(4*x*sqrt(x)*sqrt(sec(x)+ln(x)+sqrt(x)+x))`
    instead of falling into a slow unresolved route
  - the `csc/cot` dual returns directly as
    `(2*sqrt(x)+2*x*sqrt(x)+x-2*x*csc(x)*cot(x)*sqrt(x))/(4*x*sqrt(x)*sqrt(csc(x)+ln(x)+sqrt(x)+x))`
  - residual checks against both displayed forms collapse to `0` through the
    post-calculus residual route
  - required conditions remain explicit: `x > 0`, square-root radicand
    positivity, and the relevant reciprocal-trig nonzero denominator
  - `fast`, `guardrail`, and `pressure` profiles stay at `failed = 0`
- learning:
  - the missing capability was not a new derivative rule; it was a missing
    local composition of two already-supported presentation denominators
    (`ln(x)` and `sqrt(x)`) inside a reciprocal-trig square-root result

## 2026-05-20 - Retained robustness: cot-csc root derivative residual

- area:
  - calculus / differentiation / post-calculus presentation / reciprocal trig
    residuals
- status:
  - `retained`
- candidate:
  - extend the bounded tan/sec square-root derivative presentation to its
    cot/csc dual when the radicand is a small additive expression with `cot`,
    `exp`, and polynomial terms
  - teach the local residual matcher that `csc(x)^2*sin(x)^2` can cancel to
    `1` in the same bounded post-calculus residual context that already handles
    `sec(x)^2*cos(x)^2`
- retained:
  - `diff(sqrt(cot(x)+exp(x)+x), x)` now returns directly as
    `(sin(x)^2 + e^x*sin(x)^2 - 1) / (2*sin(x)^2*sqrt(cot(x)+e^x+x))`
    instead of timing out
  - `diff(sqrt(cot(x)+exp(x)+x), x) - (1-csc(x)^2+e^x)/(2*sqrt(cot(x)+exp(x)+x))`
    now collapses to `0` before general cleanup
  - the affine negative exp sibling with `exp(-2*x)` also returns quickly and
    its `csc^2` residual collapses to `0`
  - required conditions remain explicit: `sin(x) != 0` and the square-root
    radicand is positive
  - `fast`, `guardrail`, and `pressure` profiles stay at `failed = 0`
- learning:
  - the timeout was not a missing derivative rule for `cot`; it was a missing
    presentation/residual route for the reciprocal-trig dual of an already
    supported tan/sec family

## 2026-05-20 - Retained robustness: sec-squared tan-exp root derivative residual

- area:
  - calculus / differentiation / post-calculus residuals / reciprocal trig
    presentation
- status:
  - `retained`
- candidate:
  - teach the bounded post-calculus residual matcher that `sec(x)^2*cos(x)^2`
    can cancel to `1` when comparing a direct derivative rendered over
    `cos(x)^2` with an educational target rendered with `sec(x)^2`
- retained:
  - `diff(sqrt(tan(x)+exp(x)+x), x) - (sec(x)^2+e^x+1)/(2*sqrt(tan(x)+exp(x)+x))`
    now collapses to `0` before general cleanup instead of timing out after
    repeated depth overflows
  - the negative affine exp sibling
    `diff(sqrt(tan(x)+exp(-2*x)+x), x) - (sec(x)^2-2*e^(-2*x)+1)/(2*sqrt(tan(x)+exp(-2*x)+x))`
    also collapses to `0`
  - required conditions remain explicit: `cos(x) != 0` and the square-root
    radicand is positive
  - `fast`, `guardrail`, and `pressure` profiles stay at `failed = 0`
- learning:
  - the direct derivative route was already correct; the retained value was
    preventing equivalent educational residual forms from falling through into
    fragile deep trig cleanup

## 2026-05-20 - Retained robustness: negative sqrt term common-denominator trig-root residual

- area:
  - calculus / differentiation / post-calculus residuals / signed radical
    denominator factors
- status:
  - `retained`
- candidate:
  - allow the local exact denominator-factor remover used by post-calculus
    residual matching to pair `sqrt(...)` factors whose radicands are the same
    signed additive terms, even when the AST orientation differs
- retained:
  - `diff(sqrt(sin(2*x)+cos(x)-sqrt(x)), x) - (4*sqrt(x)*cos(2*x)-1-2*sqrt(x)*sin(x))/(4*sqrt(x)*sqrt(sin(2*x)+cos(x)-sqrt(x)))`
    now collapses to `0` before generic cleanup
  - the rebuilt debug CLI probe passes in about 0.49s with required conditions
    `x > 0` and `sin(2*x)+cos(x)-sqrt(x) > 0`
  - the existing positive `+sqrt(x)` common-denominator sibling still passes
    in about 0.48s
  - `fast`, `guardrail`, and `pressure` profiles stay at `failed = 0`
- learning:
  - the derivative and the inline residual were already correct; the timeout
    was caused by a too-strict factor-removal step inside the bounded
    denominator-commoning residual matcher
  - the retained change stays local to post-calculus residual verification and
    does not change global simplification or the public derivative form

## 2026-05-20 - Retained robustness: signed sqrt term in elementary sqrt derivative residual

- area:
  - calculus / differentiation / post-calculus residuals / sign orientation
- status:
  - `retained`
- candidate:
  - compare square-root denominator factors whose radicands are equivalent
    signed additive sums in different orders, without changing global
    simplification order
- retained:
  - `diff(sqrt(exp(sin(x))+ln(x)-sqrt(x)), x) - (2*sqrt(x)+2*x*sqrt(x)*cos(x)*e^sin(x)-x)/(4*x*sqrt(x)*sqrt(ln(x)-sqrt(x)+e^sin(x)))`
    now collapses to `0` before generic cleanup
  - the rebuilt debug CLI probe passes in about 0.53s with the real-domain
    requirements `x > 0` and `ln(x)-sqrt(x)+e^sin(x) > 0`
  - the retained change is local to the post-calculus residual matcher and
    leaves promoted embedded, strict, diff, limit, and integrate lanes green
- learning:
  - the public derivative was already correct; the brittle point was matching
    `sqrt(ln(x)+e^sin(x)-sqrt(x))` against
    `sqrt(ln(x)-sqrt(x)+e^sin(x))` inside a denominator factor
  - a bounded signed-additive comparison for square-root radicands is enough
    here and avoids broad signed-product equivalence in global simplification

## 2026-05-20 - Observe-only discovery: signed sqrt term in elementary sqrt derivative residual

- area:
  - calculus / differentiation / post-calculus residuals / sign orientation
- status:
  - `superseded`
- observed:
  - the positive polynomial sibling was retained in this cycle, but
    `diff(sqrt(exp(sin(x))+ln(x)-sqrt(x)), x) - (2*sqrt(x)+2*x*sqrt(x)*cos(x)*e^sin(x)-x)/(4*x*sqrt(x)*sqrt(ln(x)-sqrt(x)+e^sin(x)))`
    still times out under the 4s calculus residual probe
- superseded by:
  - the retained signed-radicand residual matcher entry above
- why it was not promoted:
  - it adds a signed-radicand orientation issue on top of the product-power
    residual gap fixed here
  - retaining both would mix two hypotheses in one cycle
- next candidate:
  - investigate signed additive terms in the elementary sqrt derivative
    residual matcher, preserving the explicit radicand positivity condition
    and avoiding broad signed-product equivalence in global simplification

## 2026-05-20 - Retained robustness: polynomial power term in exp-trig-log sqrt derivative residual

- area:
  - calculus / differentiation / post-calculus residuals / product-power
    matching
- status:
  - `retained`
- candidate:
  - allow the bounded residual matcher to compare product terms such as
    `x*x*sqrt(x)` with user-facing terms such as `x^2*sqrt(x)`
- retained:
  - `diff(sqrt(exp(sin(x))+ln(x)+sqrt(x)+x^2), x) - (2*sqrt(x)+2*x*sqrt(x)*cos(x)*e^sin(x)+x+4*x^2*sqrt(x))/(4*x*sqrt(x)*sqrt(ln(x)+sqrt(x)+e^sin(x)+x^2))`
    now collapses to `0` before generic cleanup
  - the arctan wrapper sibling with the same radicand also collapses quickly
  - the probe on the rebuilt debug CLI passes in about 0.52s with required
    conditions `x > 0` and `ln(x)+sqrt(x)+e^sin(x)+x^2 > 0`
- learning:
  - the derivative presentation intentionally emits repeated factors in some
    polynomial terms, while educational targets often use small powers
  - keeping the expansion bounded to small positive integer powers inside the
    residual matcher avoids a global simplification preference change

## 2026-05-20 - Retained robustness: exp-trig-log sqrt derivative residual

- area:
  - calculus / differentiation / post-calculus residuals / sqrt presentation
- status:
  - `retained`
- candidate:
  - add a bounded early residual route for
    `diff(sqrt(exp(sin(x))+ln(x)+sqrt(x)), x)` when the target derivative is
    written as an equivalent common-denominator fraction
- retained:
  - the direct residual
    `diff(sqrt(exp(sin(x))+ln(x)+sqrt(x)), x) - (2*sqrt(x)+2*x*sqrt(x)*cos(x)*e^sin(x)+x)/(4*x*sqrt(x)*sqrt(ln(x)+sqrt(x)+e^sin(x)))`
    now collapses to `0` before generic cleanup
  - the route preserves the real-domain requirements `x > 0` and
    `ln(x)+sqrt(x)+e^sin(x) > 0`
  - the CLI probe moved from a 4s timeout on the stale route to a pass on the
    rebuilt debug CLI in about 0.53s
- learning:
  - the calculus result was already mathematically available; the fragile part
    was residual comparison after denominator alignment
  - the retained fix reuses the existing elementary sqrt derivative
    presentation and bounded post-calculus residual matchers instead of
    expanding or invoking broader simplification

## 2026-05-20 - Retained positive-shift nonzero suppression for calculus conditions

- area:
  - calculus / differentiation / post-calculus presentation / domain conditions
- status:
  - `retained`
- candidate:
  - suppress the redundant public `NonZero(R+1)` denominator witness when a
    visible real-domain condition already contains `Positive(R)`
- retained:
  - `diff(arctan(sqrt(tan(x)+sqrt(x)+1/sqrt(x)+x)), x)` still renders the
    compact common-denominator derivative, but its public requirements now keep
    only `x > 0`, `tan(x)+sqrt(x)+1/sqrt(x)+x > 0`, and `cos(x) != 0`
  - the scaled sibling
    `diff(arctan(sqrt(tan(x)+2*sqrt(x)-3/sqrt(x)+x)), x)` gets the same
    cleanup
  - matching residuals still collapse to `0` before cleanup
- learning:
  - this is a condition-presentation improvement, not a new calculus rule:
    over the real domain, `R > 0` is enough to prove `R + c != 0` for a
    strictly positive constant `c`
  - the rule is intentionally bounded to exact additive positive-constant
    shifts; it keeps `R-1 != 0` under `R > 0`

## 2026-05-20 - Retained arctan-sqrt wrapper over mixed tan-root presentation

- area:
  - calculus / differentiation / post-calculus presentation / wrapper robustness
- status:
  - `retained`
- candidate:
  - extend the retained
    `sqrt(tan(x)+sqrt(x)+1/sqrt(x)+x)` presentation through the public wrapper
    `arctan(sqrt(...))`, where the direct CLI probe and matching residual both
    timed out before promotion
- retained:
  - `diff(arctan(sqrt(tan(x)+sqrt(x)+1/sqrt(x)+x)), x)` now reuses the inner
    common-denominator derivative and renders:
    `(2*x*sqrt(x) + 2*x*sqrt(x)*sec(x)^2 + x - 1)/(4*x*sqrt(x)*sqrt(tan(x)+sqrt(x)+1/sqrt(x)+x)*(tan(x)+sqrt(x)+1/sqrt(x)+x+1))`
  - the scaled sibling
    `diff(arctan(sqrt(tan(x)+2*sqrt(x)-3/sqrt(x)+x)), x)` uses the same
    route
  - matching residuals against both compact forms collapse to `0` before the
    general simplification pipeline
- learning:
  - this is a wrapper robustness/presentation issue, not a new derivative
    formula: the derivative is the compact `sqrt(R)` derivative divided by
    `R+1`
  - preserving the raw `diff` target and resolving the public direct/residual
    path before general simplification avoids repeated depth-overflow traffic
  - public conditions currently include an explicit `R+1 != 0` denominator
    witness in addition to `R > 0`; it is sound but redundant, and a future
    domain-presentation cleanup can suppress `NonZero(R+1)` when `R > 0` is
    already visible

## 2026-05-20 - Retained tan-root mixed sqrt/reciprocal-sqrt presentation

- area:
  - calculus / differentiation / post-calculus presentation / radical fractions
- status:
  - `retained`
- candidate:
  - improve the public presentation for
    `diff(sqrt(tan(x)+sqrt(x)+1/sqrt(x)+x), x)`, which already computed the
    derivative but exposed `sqrt(x)*x^(-3/2)` and timed out when embedded in a
    residual against the compact common-denominator form
- retained:
  - the direct derivative now keeps both radical contributions under a shared
    `x*sqrt(x)` denominator:
    `(2*x*sqrt(x) + 2*x*sqrt(x)*sec(x)^2 + x - 1)/(4*x*sqrt(x)*sqrt(tan(x)+sqrt(x)+1/sqrt(x)+x))`
  - the scaled sibling
    `diff(sqrt(tan(x)+2*sqrt(x)-3/sqrt(x)+x), x)` uses the same route and
    renders the numerator as `2*x*sqrt(x) + 2*x*sqrt(x)*sec(x)^2 + 2*x + 3`
  - residual checks against both compact forms collapse to `0`; required
    conditions remain `x > 0`, outer radicand positivity, and `cos(x) != 0`
- learning:
  - when `sqrt(x)` and `1/sqrt(x)` appear together, flushing the reciprocal
    component into generic derivative terms is too early and leaves half-power
    noise in the public result
  - keeping both local components until presentation time fixes the visible
    result without changing global simplification policy for fractional powers

## 2026-05-20 - Retained tan-root reciprocal-sqrt post-calculus presentation

- area:
  - calculus / differentiation / post-calculus presentation / radical fractions
- status:
  - `retained`
- candidate:
  - improve the public presentation for
    `diff(sqrt(tan(x)+1/sqrt(x)+x), x)` and its signed sibling, which already
    computed the derivative but exposed `x^(-3/2)` and could timeout when
    embedded in a residual against the expected educational form
- retained:
  - the direct derivative now uses a local `x*sqrt(x)` denominator:
    `(2*x*sqrt(x) + 2*x*sqrt(x)*sec(x)^2 - 1)/(4*x*sqrt(x)*sqrt(tan(x)+1/sqrt(x)+x))`
  - the negative reciprocal-sqrt sibling keeps the same compact shape with the
    final `+ 1` numerator term
  - residual checks against both compact forms collapse to `0` before the
    general simplifier; required conditions remain `x > 0`, outer radicand
    positivity, and `cos(x) != 0`
- learning:
  - this is a retained post-calculus presentation route, not a broad
    simplifier policy for all negative fractional powers
  - treating `k/sqrt(x)` as an explicit local derivative component avoids a
    fragile `x^(-3/2)` residual without changing the internal canonical forms

## 2026-05-20 - Retained tan-root sqrt-variable post-calculus presentation

- area:
  - calculus / differentiation / post-calculus presentation / radical fractions
- status:
  - `retained`
- candidate:
  - improve the direct public presentation for
    `diff(sqrt(tan(x)+sqrt(x)+x), x)`, which already terminated and verified
    but still exposed `x^(-1/2)` inside the numerator
- retained:
  - the direct derivative now uses a local `sqrt(x)` denominator:
    `(2*sqrt(x) + 2*sqrt(x)*sec(x)^2 + 1)/(4*sqrt(x)*sqrt(tan(x)+sqrt(x)+x))`
  - the affine sibling
    `diff(sqrt(tan(x)+sqrt(x)+2*x+1), x)` keeps the same compact shape
  - residual checks still collapse to `0`, and required conditions remain
    `cos(x) != 0`, outer radicand positivity, and `x > 0`
- learning:
  - this is a presentation improvement, not a new derivative rule: the calculus
    route already knew the derivative, but needed a local common-denominator
    shape for `sqrt(x)` to avoid half-power noise in public output
  - keep this local to the tan-root presentation path; do not promote a global
    preference for rewriting every `x^(-1/2)` into a displayed root

## 2026-05-20 - Observe-only reciprocal trig-root diff subtraction orientation still times out

- area:
  - calculus / differentiation / post-calculus presentation / reciprocal
    radicand orientation
- status:
  - `resolved`
- candidate:
  - while promoting the bounded direct route for
    `diff(sqrt(sin(2*x)+cos(x)+k/x), x)`, probe the sign-oriented sibling
    `diff(sqrt(sin(2*x)+cos(x)-2/x), x)`
- observation:
  - the additive forms now terminate directly:
    `diff(sqrt(sin(2*x)+cos(x)+1/x), x)` and
    `diff(sqrt(sin(2*x)+cos(x)+2/x), x)`
  - the equivalent negative-coefficient spelling
    `diff(sqrt(sin(2*x)+cos(x)+(-2)/x), x)` also terminates directly
  - the subtraction spelling
    `diff(sqrt(sin(2*x)+cos(x)-2/x), x)` still times out under the same
    public smoke probe
- learning:
  - this is an orientation/AST-shape gap, not a missing derivative formula:
    the retained presentation path can handle a negative reciprocal
    coefficient once it is represented as an additive negative term
  - do not broaden the retained `k/x` promotion to claim subtraction-oriented
    coverage until the raw derivative gate and presentation recognizer agree on
    `Sub`-shaped radicands
- resolved by:
  - resolved by teaching the raw derivative gate and additive trig-root
    presentation path to treat `Sub` radicands as signed add terms, while
    keeping the derivative route local to post-calculus presentation
  - retained contract:
    `diff(sqrt(sin(2*x)+cos(x)-2/x), x)` now returns the compact one-step
    derivative
    `(2*cos(2*x)*x^2 + 2 - sin(x)*x^2)/(2*x^2*sqrt(sin(2*x)+cos(x)-2/x))`
    with `x != 0` and `sin(2*x)+cos(x)-2/x > 0`
  - the matching residual now collapses to `0`; domain-condition display also
    drops the redundant `radicand >= 0` when the same radicand is already
    required positive

## 2026-05-20 - Observe-only shifted sqrt residual does not bridge sqrt and half-power forms

- area:
  - calculus / differentiation / post-calculus presentation / residual
    equivalence
- status:
  - `resolved`
- candidate:
  - after promoting the compact negative shifted-root derivative
    `diff(1/(sqrt(x)*(sqrt(x)-1)), x)`, verify the residual against the same
    expression typed manually:
    `diff(1/(sqrt(x)*(sqrt(x)-1)), x) + (2*sqrt(x)-1)/(2*x*sqrt(x)*(sqrt(x)-1)^2)`
- observation:
  - the public derivative now returns compactly with minimal conditions:
    `-(2*sqrt(x)-1)/(2*x*sqrt(x)*(sqrt(x)-1)^2)`,
    requiring `x > 0` and `sqrt(x)-1 != 0`
  - the residual probe does not yet collapse to `0`; the manually parsed
    denominator normalizes parts of the expression to `x^(1/2)` and
    `x^(3/2)`, so exact fraction cancellation misses the match
- learning:
  - this is not a calculus rule failure; it is a reusable bridgeability gap
    between protected post-calculus `sqrt` presentation and canonical
    half-power denominator forms
  - do not broaden the retained derivative route further to hide this; handle
    it as a separate simplification/equivalence bridge candidate
- follow-up:
  - resolved by extending fraction-denominator matching so powers with
    equivalent `sqrt`/half-power additive bases can participate in opposite
    fraction cancellation, while preserving the public post-calculus `sqrt`
    presentation
  - retained contract:
    `diff(1/(sqrt(x)*(sqrt(x)-1)), x) + (2*sqrt(x)-1)/(2*x*sqrt(x)*(sqrt(x)-1)^2)`
    now collapses to `0`

## 2026-05-20 - Observe-only quadratic arctan reciprocal-root diff has redundant domain guard

- area:
  - calculus / differentiation / domain-condition minimization / post-calculus
    presentation
- status:
  - `resolved`
- candidate:
  - promote the non-linear cofactor variant
    `diff(arctan(1/(sqrt(x)*(x^2+1))), x)` alongside the existing
    `sqrt(x)*(x+1)` arctan reciprocal-root contracts
- observation:
  - the public derivative is correct and compact:
    `-(5*x^2 + 1)/(2*sqrt(x)*(x*(x^2+1)^2+1))`
  - the composed residual also collapses to `0`:
    `diff(arctan(1/(sqrt(x)*(x^2+1))), x) + (10*x^2+2)/(4*sqrt(x)*(x*(x^2+1)^2+1))`
  - both probes retain an extra required condition
    `x^5 + 2*x^3 + x + 1 != 0`
  - under the already-required `x > 0`, that condition is redundant because
    `x*(x^2+1)^2 + 1` is strictly positive
- learning:
  - this is not a calculus capability failure; it is a reusable
    domain-condition minimization gap for positive products plus a positive
    offset after denominator expansion
  - do not promote the quadratic cofactor contract until the required
    conditions collapse to the minimal real-domain frontier, expected here as
    `x > 0`
- follow-up:
  - resolved by extending required-condition normalization so positive powers
    of a nonnegative base are also recognized as nonnegative under display
    conditions; this lets `x > 0` prove the expanded denominator
    `x^5 + 2*x^3 + x + 1` is positive and drops the redundant nonzero guard

## 2026-05-19 - Observe-only negative scaled elementary trig-root diff still loops

- area:
  - calculus / differentiation / post-calculus presentation / mixed radicands
- status:
  - `resolved`
- candidate:
  - extend the direct `diff(sqrt(sin(2*x)+cos(x)+k*f(x)), x)` presentation
    route from positive scaled `ln(x)` and `exp(x)` terms to negative scaled
    variants such as `-3*ln(x)` and `-3*exp(x)`
- observation:
  - positive scaled probes now terminate directly:
    `diff(sqrt(sin(2*x)+cos(x)+2*ln(x)), x)` and
    `diff(sqrt(sin(2*x)+cos(x)+2*exp(x)), x)`
  - the negative scaled probes still entered depth-overflow/post-cleanup loops
    before producing stable JSON under the same `--steps on` public route:
    `diff(sqrt(sin(2*x)+cos(x)-3*ln(x)), x)` and
    `diff(sqrt(sin(2*x)+cos(x)-3*exp(x)), x)`
- learning:
  - this is not a parser typo or duplicate test; the remaining weakness is a
    signed elementary-term presentation/post-cleanup interaction after the
    direct derivative form is constructed
  - do not promote negative signed elementary radicands until the result can be
    returned without re-entering trig expansion/factorization cleanup
- follow-up:
  - resolved by a narrow rationalization guard for square-root denominators
    whose radicand combines trig terms with `ln`/`exp` terms; the negative
    `ln` and `exp` probes now return in one direct derivative step with no
    `depth_overflow`

## 2026-05-19 - Observe-only non-polynomial trig-root diff can timeout

- area:
  - calculus / differentiation / post-calculus presentation / mixed radicands
- status:
  - `resolved`
- candidate:
  - extend the direct `diff(sqrt(sin(2*x)+cos(x)+p(x)), x)` presentation route
    beyond polynomial terms to non-polynomial additive terms such as `ln(x)`
- observation:
  - the logarithmic probe timed out before producing JSON:
    `timeout 12s cargo run -q -p cas_cli -- eval 'diff(sqrt(sin(2*x)+cos(x)+ln(x)), x)' --format json --steps on`
  - nearby polynomial probes terminate, and the cubic polynomial case was
    promoted separately with a bounded degree gate
- learning:
  - the safe retained move is to broaden the polynomial gate conservatively;
    non-polynomial radicand terms need a separate robustness/presentation route
    with explicit termination guards
- follow-up:
  - resolved by the bounded `ln(x)`-only direct presentation route for
    `diff(sqrt(sin(2*x)+cos(x)+ln(x)), x)`, guarded by a wire smoke that
    requires one step, no `depth_overflow`, and both `radicand > 0` plus
    `x > 0`
  - keep `exp`/nested-root additive terms as separate candidates; they were not
    promoted by this fix

## 2026-05-19 - Observe-only raw tangent sqrt diff route can loop

- area:
  - calculus / differentiation / post-calculus presentation / tangent domains
- status:
  - `resolved`
- candidate:
  - preserve raw `tan` inside `diff(sqrt(tan(x)+sin(x)+x), x)` and present the
    result directly as `(cos(x) + 1/cos(x)^2 + 1)/(2*sqrt(sin(x)+tan(x)+x))`
  - intended domain cleanup was to keep `sin(x)+tan(x)+x > 0` and add
    `cos(x) != 0` while dropping the transformed-radicand guard
    `sin(x)+sin(x)/cos(x)+x >= 0`
- observation:
  - the direct raw route hung before producing JSON under a short smoke probe:
    `timeout 8s cargo run -q -p cas_cli -- eval 'diff(sqrt(tan(x)+sin(x)+x), x)' --steps on --format json`
  - the previous expanded route terminates, but has noisy steps and a leaked
    transformed-radicand condition
- learning:
  - tangent-bearing root derivatives need a domain/presentation route that can
    preserve `tan` publicly without sending the simplifier into a raw
    tangent/root loop
  - do not promote raw tangent preservation until a cheaper structural gate or
    presentation-only wrapper avoids that loop
- follow-up:
  - resolved by promoting an isolated public diff/residual contract for
    `diff(sqrt(tan(x)+sin(x)+x), x)`: the route now preserves public `tan`,
    returns in one step, keeps only `cos(x) != 0` plus outer radicand
    positivity, and avoids the transformed-radicand guard.

## 2026-05-18 - Observe-only scaled inverse-trig polynomial substitution trace gap

- area:
  - calculus / integration / post-calculus presentation / didactic trace
- status:
  - `resolved`
  - resolved in the 2026-05-18 retained calculus cycle that taught the
    inverse-trig polynomial substitution integrator to split algebraic
    variable-free numerator factors before matching the square-root kernel.
- local lane:
  - direct retained probe:
    `cargo run -q -p cas_cli -- eval 'integrate((4*x^3+6*x^2+6*x+2)/sqrt(2-3*(x^2+x+1)^4), x)' --format json --steps on`
  - rejected scaled probe:
    `cargo run -q -p cas_cli -- eval 'integrate(2*(2*x^3+3*x^2+3*x+1)*sqrt(3)/sqrt(2-3*(x^2+x+1)^4), x)' --format json --steps on`
- local win:
  - preserving the raw integrand for the unscaled arcsin polynomial
    substitution removes a noisy `Expandir binomio` step and leaves a single
    direct integration step with the same result and domain requirement.
- global result:
  - only the unscaled semantic detector was retained. A broader syntactic
    raw-preservation guard for nested polynomial powers under square-root
    denominators was rejected before promotion because it made the scaled
    `sqrt(3)` variant stop integrating.
- why it regressed globally:
  - the scaled variant is mathematically in the same family, but the current
    direct recognizer still depends on pre-integration normalization/expansion
    to expose the supported shape. Preserving the raw scaled integrand too
    early hides the form from the existing integration matcher.
- what could make it combinable later:
  - teach the inverse-trig polynomial substitution recognizer to handle
    algebraic constant factors such as `sqrt(3)` directly, then re-enable raw
    preservation for that narrower semantic family instead of a broad syntactic
    radicand-power guard.
- resolution:
  - the scaled probe now integrates directly with one `Calcular la integral`
    step, no `Expandir binomio`, the same real-domain requirement, and
    antiderivative verification retained in the integration contract.

## 2026-05-18 - Observe-only multi-trig ln/sqrt diff trace cost

- area:
  - calculus / diff / domain cleanup / post-calculus presentation
- status:
  - `superseded`
- local lane:
  - direct probes:
    `cargo run -q -p cas_cli -- eval 'diff(ln(1+sqrt(sin(x)+cos(x)+3)), x)' --format json --steps on`
    and
    `cargo run -q -p cas_cli -- eval 'diff(ln(1+sqrt(2*sin(x)+cos(x)+4)), x)' --format json --steps on`
- local win:
  - a conservative real-domain L1 bound in `prove_sign` can prove
    `sin(x)+cos(x)+3 > 0` and `2*sin(x)+cos(x)+4 > 0`, so the public result no
    longer needs redundant `Requires` for these radicands.
- global result:
  - the core sign-proof improvement was retained behind unit coverage, but
    the two multi-trig `steps-on` public cases were not promoted to
    `wire_smoke_tests`: each direct public probe took roughly 46-52s.
- why it regressed globally:
  - the cost appears to be the existing public diff/trace route for
    multi-term trig radicands, not the L1 sign proof itself; unit sign probes
    are instant, while the CLI calculus path is slow before it can return the
    otherwise correct one-step result.
- what could make it combinable later:
  - add observability or a narrower calculus trace shortcut for multi-term
    trig radicands before promoting these shapes into a hot public smoke lane.
- superseded by:
  - the broad opaque-root expand/cancel guard removed the public trace cost for
    these shapes, and the minimal unit-weight and coefficient-weighted
    multi-trig cases are now promoted in `wire_smoke_tests`.

## 2026-05-18 - Superseded expanded-square diff presentation trace cost

- area:
  - calculus / diff / post-calculus presentation / didactic trace
- status:
  - `superseded`
- local lane:
  - direct probe:
    `timeout 60s cargo run -q -p cas_cli -- eval 'diff(sqrt(x)/(x^2+2*x+1)^2, x)' --steps on --format json`
  - promoted non-trace contract:
    `cargo test -p cas_cli --test wire_smoke_tests test_eval_json_diff_sqrt_power_of_expanded_affine_square_quotient_presentation_cancels_common_factor -- --exact --nocapture`
- local win:
  - the public no-steps result can be presented compactly as
    `(1 - 7·x) / (2·sqrt(x)·(x + 1)^5)` while preserving the `x > 0`
    condition and proving the residual against the expanded denominator input.
- global result:
  - the no-steps presentation was retained and guardrails passed, but the
    corresponding `steps-on` assertion was not promoted because the trace probe
    exceeded 60s before producing output.
- why it regressed globally:
  - didactic tracing for this high-power expanded affine-square quotient appears
    to explore a much larger path than the result computation; lower-power
    neighbors stayed cheap, so the issue is trace cost rather than semantic
    correctness.
- what could make it combinable later:
  - add a narrower post-calculus trace shortcut, trace budget, or reusable
    explanation for compacting expanded affine-square denominator powers before
    promoting this shape with a `steps-on` assertion.
- superseded by:
  - a follow-up robustness iteration added a narrow steps-on shortcut for
    `sqrt(polynomial)/(expanded affine square)^n`, `n>=2`. The promoted probe
    now returns the compact result with two visible steps, derivative plus
    post-calculus presentation, while preserving `x > 0`.

## 2026-05-17 - Observe-only sqrt denominator square presentation gap

- area:
  - calculus / diff / post-calculus presentation / formatter boundary
- status:
  - `superseded`
- local lane:
  - direct probes:
    `cargo run -q -p cas_cli -- eval 'diff(arctan(sqrt(x)) + sqrt(x)/(x+1), x)' --no-pretty`
    and
    `cargo run -q -p cas_cli -- eval '1/(sqrt(x)*(x+1)^2)' --no-pretty`
- local win:
  - a bounded post-diff recognizer can identify the mathematically compact
    shape produced by differentiating
    `arctan(sqrt(x)) + sqrt(x)/(x+1)` and
    `8*arctan(2*sqrt(x)) + 4*sqrt(x)/(x+1/4)`.
- global result:
  - the strict presentation target
    `1/(sqrt(x)*(x+a)^2)` was not promoted. The expression renderer/canonical
    display path rewrites that shape back toward `sqrt(x)/(x*(x+a)^2)`, and
    forcing it through internal holds would couple a domain-conditional
    calculus presentation to global display semantics.
- why it regressed globally:
  - `sqrt(x)/x = 1/sqrt(x)` is only real-safe under `x > 0`; the current
    display layer has no domain-condition channel, so a formatter-level rule
    would be too broad.
- what could make it combinable later:
  - add an explicit domain-aware post-calculus display channel, or a dedicated
    final-form marker that can render reciprocal-root denominators only when
    the calculus result carries the required `x > 0` condition.
- retained narrower follow-up:
  - the current iteration retained the safer compact form
    `x^(-1/2)/(x+a)^2`, preserving the public `x > 0` requirement and
    avoiding a global formatter change.
- superseded by:
  - a follow-up display-only presentation change renders quotient numerators
    of the form `base^(-1/2)` as reciprocal square-root denominators, so the
    retained calculus result now displays as `1/(sqrt(x)*(x+a)^2)` without
    changing the internal AST or simplification route.

## 2026-05-17 - Observe-only sqrt-linear-square arctan antiderivative residual

- area:
  - calculus / integration / antiderivative verification / residual simplification
- status:
  - `superseded`
- local lane:
  - `cargo test -p cas_math symbolic_integration_support::tests::integrates_arctan_sqrt_reciprocal_linear_square_kernel -- --exact`
  - direct residual probe:
    `timeout 15s cargo run -q -p cas_cli -- eval 'diff(arctan(sqrt(x)) + sqrt(x)/(x+1), x) - 1/(sqrt(x)*(x+1)^2)' --format json`
- local win:
  - a bounded table extension for `integrate(1/(sqrt(x)*(a*x+b)^2), x)`,
    `a,b > 0`, produced the expected antiderivative family via
    `u = sqrt(x)`, for example
    `arctan(sqrt(x)) + sqrt(x)/(x+1)`.
- global result:
  - not promoted. The candidate passed its narrow construction test, but the
    public residual verification for
    `diff(arctan(sqrt(x)) + sqrt(x)/(x+1), x) - 1/(sqrt(x)*(x+1)^2)`
    hit `depth_overflow` and timed out under the 15s probe.
- why it regressed globally:
  - simplification combines the arctan-plus-rational antiderivative over the
    shared `(x+1)` denominator and then differentiates a much larger quotient;
    the residual does not cheaply cancel the duplicated `arctan(sqrt(x))`
    terms before depth pressure appears.
- what could make it combinable later:
  - first add a residual/presentation simplification path that preserves or
    recovers `arctan(sqrt(x)) + sqrt(x)/(x+1)` as a sum for differentiation, or
    a bounded cancellation for the resulting duplicated arctan quotient shape;
    then retry the conservative integration family.
- superseded by:
  - a follow-up robustness iteration added a bounded pre-simplification
    residual recognizer for
    `diff(c*arctan(sqrt(x)) + c*sqrt(x)/(x+1), x) - c/(sqrt(x)*(x+1)^2)`,
    preserving the `x > 0` condition and avoiding the depth-overflow path.

## 2026-05-17 - Superseded scaled conditional post-integration presentation

- area:
  - calculus / integration / post-calculus presentation / residual verification
- status:
  - `superseded`
- local lane:
  - `cargo test -q -p cas_cli --test integrate_contract_tests integrate_contract_polynomial_derivative_over_fractional_denominator_power_substitution -- --exact --nocapture`
- local win:
  - broadening fractional denominator-power presentation from strictly positive
    quadratics to all positive-leading conditional quadratics made outputs such
    as `integrate((4*x+2)/(2*x^2+2*x-3)^(7/2), x)` eligible for compact
    `sqrt(base) * base^n` denominator rendering while preserving the positive
    base condition.
- global result:
  - existing scaled-shifted residual assertions regressed before promotion:
    differentiating the displayed antiderivative left a nonzero-looking
    residual for the `5/2` scaled base case.
- why it regressed globally:
  - residual normalization already handles the unpresented fractional power
    form and the monic conditional presentation path, but not the scaled
    conditional displayed denominator shape.
- what could make it combinable later:
  - add a residual-normalization path for scaled conditional
    `sqrt(base) * base^n` denominators, then re-open presentation beyond monic
    conditional quadratics.
- superseded by:
  - a follow-up residual parser change that sums integer powers of the same
    base inside half-power denominator products, allowing the scaled conditional
    presentation to be promoted with antiderivative verification intact.

## 2026-05-16 - Rejected broad residual display: two-argument integration as integral text

- area:
  - calculus / integration / post-calculus presentation / CLI residual display
- status:
  - `rejected`
- local lane:
  - `cargo test -p cas_cli eval_unresolved_integration_residual_uses_integral_text_display -- --nocapture`
  - direct probe:
    `cargo run --release -q -p cas_cli -- eval 'integrate(sin(x^2), x)' --format json --steps off`
- local win:
  - two-argument unsupported integration residuals rendered as textbook-style
    plain text, e.g. `int sin(x^2) , dx`, matching the existing `result_latex`
    integral presentation
- global result:
  - `make engine-scorecard` kept embedded, derive, simplify, diff, and limit
    lanes green, but failed `calculus_integrate_contract`
  - failures were contract assertions that intentionally distinguish unsupported
    residuals by the function-call text form, e.g.
    `integrate(sin(x^2), x)`
- why it regressed globally:
  - the two-argument `integrate(expr, x)` form is already used throughout the
    integration contract suite as a stable textual residual sentinel
  - changing that display globally is broader than a post-calculus polish fix
    and requires a deliberate contract migration
- what could make it combinable later:
  - introduce an explicit unsupported/residual field or status in CLI/wire, then
    migrate integration contracts away from function-call text as the residual
    sentinel before changing two-argument presentation
- retained narrower route:
  - keep the display improvement only for one-argument public
    `integrate(expr)`, where the default-variable surface already presents as
    `dx` in LaTeX and does not collide with the existing integration residual
    contract lane

## 2026-05-16 - Observe-only discovery: compact sec/csc log presentation adds redundant domain requirements

- area:
  - calculus / integration / post-calculus presentation / domain presentation
- status:
  - `observe-only`
- resolved by:
  - 2026-05-22 observability close-out:
    current public integration contracts and release probes cover the compact
    `sec`/`csc` logarithmic primitives without redundant nonzero requirements;
    the scorecard entry was stale discovery signal, not an open engine gap
- generated candidate:
  - protect the `sec(u)` and `csc(u)` logarithmic primitive arguments with
    `__hold`, so public integration could render the same form used in the
    didactic trace:
    - `integrate(2*x*sec(x^2), x) -> ln(|tan(x^2) + sec(x^2)|)`
    - `integrate(2*x*csc(x^2), x) -> ln(|csc(x^2) - cot(x^2)|)`
- local win:
  - polynomial substitution probes rendered the standard textbook primitives
    instead of the current ratio forms
- rejection reason:
  - the held compact log argument introduced redundant public requirements:
    - `tan(x^2) + sec(x^2) != 0` in addition to `cos(x^2) != 0`
    - `csc(x^2) - cot(x^2) != 0` in addition to `sin(x^2) != 0`
  - the affine `sec(2*x+1)`/`csc(2*x+1)` siblings also degraded to reciprocal
    quotient spelling inside the held argument, so the change was not a clean
    presentation-only improvement
- reusable weakness:
  - post-calculus presentation for reciprocal-trig log primitives needs a
    domain-presentation proof/filter before preserving compact `sec/tan` or
    `csc/cot` log arguments; under `cos(u) != 0`, `sec(u)+tan(u)` is nonzero,
    and under `sin(u) != 0`, `csc(u)-cot(u)` is nonzero, but the public
    condition layer does not currently encode that redundancy
- next candidate:
  - add a narrow required-condition redundancy filter or proof helper for these
    reciprocal-trig log arguments, then re-evaluate compact presentation with
    direct residual verification and `failed = 0`

## 2026-05-16 - Observe-only discovery: impossible nonzero condition on finite-limit residual

- area:
  - calculus / limit / domain requirement presentation
- status:
  - `superseded`
- observed probe:
  - `limit(1/(sqrt(x^2+1)-sqrt(x^2+1)), x, -2)`
- observed result:
  - the finite-limit evaluator correctly keeps the expression residual and does
    not divide by the zero denominator
  - the public domain envelope still renders an impossible requirement:
    `0 != 0`
- reusable weakness:
  - when denominator requirements are collected after simplification has
    collapsed a denominator to literal zero, the residual output can surface a
    mathematically impossible nonzero requirement instead of suppressing or
    classifying it as an unsatisfied-domain residual
- next candidate:
  - add a narrow domain-presentation cleanup for residual expressions whose
    collected `NonZero` requirement is literal zero, preserving the residual
    and warning while avoiding impossible `Requires` text
- superseded by:
  - a narrow public-condition presentation filter for residual `limit(...)`
    outputs that suppresses literal `NonZero(0)` while retaining the residual
    result and finite-limit warning

## 2026-05-15 - Observe-only discovery: negative acosh fused residual after public diff

- area:
  - calculus / diff-integrate residual verification / reciprocal square-root
    radicand scaling
- status:
  - `superseded`
- generated candidate:
  - promote fused-radicand `acosh` coverage for
    `integrate(±2/sqrt((2*x-1)^2-1), x)`
- retained subset:
  - promoted the positive and negative fused-radicand integration rows
  - added a bounded residual helper for scaled reciprocal square-root
    polynomial radicands, so the positive public residual
    `diff(acosh(2*x-1), x) - 2/sqrt((2*x-1)^2-1)` now closes to `0`
- observed gap:
  - the negative public residual
    `diff(acosh(1-2*x), x) - (-2/sqrt((2*x-1)^2-1))` still renders as
    `(x^2 - x)^(-1/2) - 2*(-2*x*(2 - 2*x))^(-1/2)` under `x < 0`
  - the promoted integration contract still verifies the antiderivative, so
    this is a residual presentation/re-entry gap rather than an incorrect
    integration result
- superseded by:
  - retained the narrow post-diff `acosh` residual re-entry gate for affine
    arguments against fused reciprocal-square-root radicands
  - the positive residual now closes to `0` with `Requires: x > 1`
  - the negative residual now closes to `0` with `Requires: x < 0`
- reusable weakness:
  - after differentiating a negatively oriented inverse-hyperbolic primitive,
    the simplifier can expose equivalent reciprocal square-root radicands
    whose scalar content is sign-normalized differently, and the public
    top-level residual route may not re-enter the scaled-radicand cancellation
    helper
- next candidate:
  - add a narrow post-diff residual re-entry gate for reciprocal square-root
    terms with same-sign coefficients and polynomial radicands differing by a
    positive square factor, while preserving explicit branch/domain conditions
    and measuring embedded runtime

## 2026-05-15 - Observe-only discovery: negative sqrt reciprocal-trig trace

- area:
  - calculus / integrate / didactic trace / post-calculus presentation
- status:
  - `superseded`
- observed result:
  - direct positive sqrt-chain products such as
    `integrate(sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x)), x)` now receive a
    specific `sec(u)·tan(u)` trace with concrete `u` and `du`
  - the normalized negative sibling
    `integrate(-sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x)), x)` still reaches the
    final integration step as a raw
    `-sin(sqrt(x))/(2*sqrt(x)*cos(sqrt(x))^2)`-shaped quotient and falls back
    to the generic `Usar sustitución` substep
- decision:
  - retain only the direct positive/product-shaped trace in this iteration
  - do not broaden the matcher to raw negative quotients without a separate,
    focused trace-normalization hypothesis
- reusable weakness:
  - sign-pulled reciprocal-trig derivative products can cross from the public
    `sec/csc * tan/cot` surface into a raw `sin/cos^2` quotient before
    didactic substeps are generated, so trace recognition needs a narrow
    normalized-form bridge if we want parity for negative variants
- next candidate:
  - add a focused didactic bridge for raw reciprocal-trig derivative quotients
    with sqrt-chain cofactors, then require rule-specific substeps for the
    negative `sec/csc` siblings without changing integration semantics
- superseded by:
  - a retained didactic bridge for raw sqrt-chain reciprocal-trig derivative
    quotients
  - `integrate(-sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x)), x)` now exposes
    `Usar la regla de sec(u)·tan(u) -> sec(u)`, `Identificar u y du`, and
    `Ajustar el factor constante`
  - `integrate(-csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x)), x)` now exposes the
    analogous `csc(u)·cot(u)` trace
  - guardrails passed with `failed = 0`

### 2026-05-15: Cross-Family Csc Residual Quotient Discovery

- area:
  - calculus / residual verification / cross-family quotient wrappers
- status:
  - `superseded`
- generated candidate:
  - promote a shifted quotient of a domain-sensitive reciprocal-trig residual
    over another calculus residual, for example:
    `((diff(integrate(csc(2*x+1),x),x)-csc(2*x+1))+1) /
    ((diff(integrate(1/(x^2+1),x),x)-1/(x^2+1))+1)`
- local lane:
  - `scripts/engine_calculus_residual_probe_smoke.py --expr ... --cas-cli
    target/debug/cas_cli --expect-result 1 --require 'sin(2·x + 1)'
    --forbid-warnings --timeout-seconds 4 --slow-wall-seconds 4 --json`
  - focused direct CLI probe against the same `csc` numerator and the heavier
    rational denominator
- local finding:
  - the simpler cross-family shifted quotient timed out under the 4s smoke
    budget
  - the heavier rational-denominator variant eventually returned `1`, but took
    about 49s, emitted a `depth_overflow` warning on stderr, and leaked
    residual required conditions such as the unsimplified denominator residual
    plus `1`
- retained subset:
  - do not promote the generated case to the matrix in this iteration
  - retain only a harness observability fix so `--forbid-warnings` also catches
    `WARN` lines emitted on stderr, not only JSON `warnings`
- what could make it combinable later:
  - a narrow residual-first simplification gate for shifted quotients whose
    numerator or denominator is a calculus residual with known domain
    conditions, plus condition cleanup before quotient nonzero requirements are
    collected
- superseded by:
  - a retained reciprocal-trig integral residual constant-passthrough quotient
    shortcut
  - the simpler cross-family shifted quotient now returns `1` with
    `sin(2·x + 1) != 0`, no warnings, and about 0.54s wall time under the
    debug CLI smoke
  - the heavier rational-denominator variant now returns `1` with
    `sin(2·x + 1) != 0` and `x + 1 != 0`, no residual denominator condition,
    no warnings, and about 0.04s wall time under the debug CLI smoke

### 2026-05-15: Shifted Hyperbolic Csch-Squared Residual Recheck

- area:
  - calculus / symbolic differentiation / hyperbolic reciprocal-square residuals
- status:
  - `superseded`
- attempted case:
  - promote `integrate(1/sinh(2*x+1)^2, x)` as part of the live bounded
    public residual verification set
- local lane:
  - CLI residual probe:
    `cargo run --release -q -p cas_cli -- eval 'diff(integrate(1/sinh(2*x + 1)^2, x), x) - 1/sinh(2*x + 1)^2' --format json --budget small`
- local result:
  - the residual still eventually simplifies to `0` and preserves
    `sinh(2*x + 1) != 0`
  - the probe took about 26.4s and emitted many `cycle_detected` and
    `depth_overflow` warnings
- global result:
  - not promoted as live coverage in this iteration
  - retained coverage was limited to the cheaper sibling set:
    `1/cosh(2*x+1)^2`, `sinh(2*x+1)/cosh(2*x+1)^2`, and
    `cosh(2*x+1)/sinh(2*x+1)^2`
- best current explanation:
  - the `csch^2` residual can still detour through expanded shifted
    hyperbolic-square forms before reaching the compact reciprocal-square
    identity
- plausible follow-up:
  - add a narrow pre-expansion residual gate for `csch(u)^2` derivative checks,
    then retry the case as pressure before promoting it to live coverage
- superseded by:
  - a bounded `diff(integrate(...), x) - integrand` residual matcher for
    `integrate_symbolic_is_hyperbolic_quotient_substitution_target`
  - the same probe now returns `0` with `sinh(2*x + 1) != 0`, no warnings, and
    about 3.6ms of engine time

### 2026-05-15: Negative Shifted Arctan Compact Derivative Discovery

- area:
  - calculus / integration / post-calculus presentation
- status:
  - `observe-only`
- generated candidate:
  - compact the public primitive for `integrate(x^2*arctan(1-x), x)` from
    separated `arctan(1 - x)` terms into a single polynomial cofactor
- local lane:
  - focused CLI probes for `diff(integrate(x^2*arctan(1-x)+x*arctan(1-x), x), x)`
    residuals and explicit compact-antiderivative residuals
- local finding:
  - `diff(integrate(target), x) - target` can be made robust with a structural
    polynomial-times-arctan-affine shortcut, but directly differentiating the
    rendered compact negative-slope antiderivative still enters a deep rational
    simplification route and emits `depth_overflow`
- retained subset:
  - keep the public `diff(integrate(...), x)` shortcut for structural
    polynomial-times-arctan-affine integrands
- what could make it combinable later:
  - a compact-derivative recognizer for polynomial-cofactor arctan-affine
    by-parts primitives, or a narrower rational cancellation route for the
    `1 + (1 - x)^2` denominator orientation

### 2026-05-14: Shifted Hyperbolic Square Integration Discovery

- area:
  - calculus / integration / post-calculus presentation
- status:
  - `observe-only`
- generated candidate:
  - promote `integrate(sinh(2*x+1)^2, x)` and
    `integrate(cosh(2*x+1)^2, x)` with the same power-reduction row as the
    basic `sinh(x)^2` / `cosh(x)^2` cases
- local lane:
  - focused CLI probes and `integrate_contract_affine_hyperbolic_square_power_reduction`
- local finding:
  - the direct primitive is mathematically valid, but public presentation of
    shifted products such as `sinh(2*x+1)*cosh(2*x+1)` entered a slow
    simplification route before promotion
- retained subset:
  - keep the minimal public cases `integrate(sinh(x)^2, x)` and
    `integrate(cosh(x)^2, x)`
  - a follow-up retained `integrate(sinh(2*x)^2, x)` by adding a bounded
    product-square row for the pre-expanded
    `4*sinh(x)^2*cosh(x)^2` shape
- what could make it combinable later:
  - a post-calculus presentation rule that keeps shifted hyperbolic products
    compact without entering the expensive product-to-sum route

### 2026-05-14: Constant-Base Log Square-Root Diff Presentation

- area:
  - calculus / post-calculus presentation / logarithmic derivatives
- status:
  - `retained`
- retained case:
  - `diff(sqrt(log10(x)), x)`
  - residual:
    `diff(sqrt(log10(x)), x) - 1/(2*x*ln(10)*sqrt(log10(x)))`
- local lane:
  - focused CLI probes and public JSON smoke contracts for constant-base log
    square-root derivative presentation
- local result:
  - direct diff now renders as
    `1 / (2·x·ln(10)·sqrt(log10(x)))`
  - the residual against that rendered form collapses to `0`
  - sibling probe `diff(sqrt(log2(x)), x)` renders with `ln(2)` in the same
    bounded path
  - required conditions remain the intrinsic log-domain pair:
    `log10(x) > 0` and `x > 0`
- implementation:
  - extended the bounded `sqrt(elementary function)` post-diff presenter with
    `log2`/`log10` constant-base denominator factors
  - did not widen to arbitrary two-argument `log(base, x)` in this iteration
- domain safety:
  - no new assumption is introduced; the conditions are inherited from the
    original logarithm witness and outer square-root radicand
- bridge decision:
  - no derive case promoted; this is final calculus presentation, not an
    independent target-family transition
- calculus/precalculus decision:
  - reuses existing constant-base log differentiation and residual
    simplification rather than changing global canonical simplification
- validation:
  - focused public CLI probes, public wire presentation/residual tests,
    `calculus_diff_contract`, `make engine-fast`, and `make engine-scorecard`
    passed with `failed = 0`

### 2026-05-14: Direct `sqrt(atanh(x))` Diff Presentation With Open-Interval Guard

- area:
  - calculus / post-calculus presentation / domain preservation
- status:
  - `retained`
- retained case:
  - `diff(sqrt(atanh(x)), x)`
  - residual:
    `diff(sqrt(atanh(x)), x) - 1/(2*(1-x^2)*sqrt(atanh(x)))`
- local lane:
  - focused CLI probes and public JSON smoke contracts for the direct `atanh`
    square-root derivative presentation
- local result:
  - direct diff now renders as
    `1 / (2·(1 - x^2)·sqrt(atanh(x)))`
  - the residual against that rendered form collapses to `0`
  - required conditions preserve the intrinsic real-domain guard:
    `1 - x^2 > 0` and `atanh(x) > 0`
- implementation:
  - extended the bounded `sqrt(elementary function)` post-diff presenter with
    a non-radical `1 - arg^2` denominator shape for `atanh`
  - preserved nested `atanh(...)` open-interval conditions from the original
    diff input so presentation does not degrade them to weaker nonzero factor
    guards
- domain safety:
  - no new assumption is introduced; the open-interval condition is intrinsic
    to the `atanh` witness already present in the input AST
- bridge decision:
  - no derive case promoted; this is final calculus presentation and domain
    retention, not an independent algebraic transition
- calculus/precalculus decision:
  - reuses existing atanh open-interval condition construction and public
    residual verification instead of adding a calculus-only answer shortcut
- validation:
  - focused public CLI probes, public wire presentation/residual tests,
    `calculus_diff_contract`, `make engine-fast`, and `make engine-scorecard`
    passed with `failed = 0`

### 2026-05-14: Direct `sqrt(acosh(x))` Diff Presentation With Dominated Product Guard

- area:
  - calculus / post-calculus presentation / domain normalization
- status:
  - `retained`
- retained case:
  - `diff(sqrt(acosh(x)), x)`
  - residual:
    `diff(sqrt(acosh(x)), x) - 1/(2*sqrt(x-1)*sqrt(x+1)*sqrt(acosh(x)))`
- local lane:
  - focused CLI probes and public JSON smoke contracts for the direct `acosh`
    square-root derivative presentation
- local result:
  - direct diff now renders as
    `1 / (2·sqrt(x - 1)·sqrt(x + 1)·sqrt(acosh(x)))`
  - the residual against that rendered form collapses to `0`
  - required conditions remain the minimal useful pair:
    `x - 1 > 0` and `acosh(x) > 0`
- implementation:
  - retained the compact split-radical presentation for the direct `acosh(x)`
    case
  - extended condition dominance so a `NonNegative(product)` guard is dropped
    only when known positive factors and intrinsic positive factors imply it,
    including after factoring a polynomial factor
- domain safety:
  - no new assumption is introduced; the removed guard is implied by already
    present positive conditions
- bridge decision:
  - no derive case promoted; this is a public calculus presentation and domain
    normalization improvement, not a new didactic source-to-target route
- calculus/precalculus decision:
  - reuses pre-calculus condition normalization and product/factor reasoning
    rather than adding a calculus-only special case
- validation:
  - focal domain-normalization unit, public wire presentation/residual tests,
    public CLI probe, `calculus_diff_contract`, `make engine-fast`,
    `make engine-scorecard`, and `make engine-scorecard-pressure` passed with
    `failed = 0`

### 2026-05-14: Positive-Quadratic Log By-Parts Rendered-Residual Gap

- area:
  - calculus / integration by parts / antiderivative residual verification
- status:
  - `retained-partial`
- discovered case:
  - `integrate(ln(x^2+x+1), x)`
  - direct rendered residual for the promoted linear case:
    `diff(1/2*x^2*ln(x^2+x+1) - 3/2*arctan((2*x+1)/sqrt(3))/sqrt(3) - 1/2*x^2 + 1/4*ln(x^2+x+1) + 1/2*x, x) - x*ln(x^2+x+1)`
- local lane:
  - focused CLI probes while extending positive-quadratic log by-parts from
    quadratic cofactors to a linear cofactor
- local result:
  - `integrate(x*ln(x^2+x+1), x)` now produces an explicit primitive
  - `diff(integrate(x*ln(x^2+x+1), x), x) - x*ln(x^2+x+1)` collapses to `0`
  - retained follow-up:
    `integrate(ln(x^2+x+1), x)` now produces an explicit primitive and its
    public `diff(integrate(...), x) - ln(...)` residual collapses to `0`
  - later retained follow-up:
    differentiating the rendered standalone primitive for
    `integrate(ln(x^2+x+1), x)` now collapses directly to `0`
  - later retained follow-up:
    differentiating the rendered linear primitive for `integrate(x*ln(x^2+x+1), x)`
    now collapses directly to `0` without `depth_overflow`
- why a residual discovery remains:
  - the retained linear rule is safe because it stays behind a positive
    quadratic log-by-parts gate and verifies through the public integrate
    residual path
  - the retained standalone rule is safe for the same reason: `Q` must be
    positive quadratic and the residual route verifies the integrate call
  - the rendered standalone and linear cases were closed by signed additive term
    matching plus rational-scale normalization of additive terms
- remaining follow-up:
  - none for this shifted positive-quadratic log-by-parts family; future work
    should move to a distinct positive-quadratic log shape or a broader
    verification abstraction, not another near-duplicate of this case

### 2026-05-13: Symbolic Beta-Sqrt Product Presentation Conflicts With Residual Verification

- area:
  - calculus / post-calculus presentation / integration residual
- status:
  - `retained-after-combination`
- local lane:
  - focused CLI probes for
    `diff(integrate(a/(2*sqrt(x)*sqrt(1-x)), x), x)` and
    `diff(integrate(a/(sqrt(2*x+1)*sqrt(3-2*x)), x), x)`
- local win:
  - allowing numerator factors free of `x` in the arcsin inverse-sqrt-product
    detector produced compact public results:
    `a / (2*sqrt(x)*sqrt(1-x))` and
    `a / (sqrt(2*x+1)*sqrt(3-2*x))`
- global result:
  - the focused integration contract regressed before guardrail promotion:
    `diff(integrate(a/(2*sqrt(x)*sqrt(1-x)), x), x) - a/(2*sqrt(x)*sqrt(1-x))`
    no longer collapsed to `0` in one engine pass
- why it regressed globally:
  - the compact presentation has to be protected from rationalization with an
    internal hold, especially for affine radicands
  - inside a residual difference, that hold prevents the existing
    inverse-differentiation verification route from normalizing the compact
    root-product side to the same internal negative-half-power form as the
    comparison side
- what could make it combinable later:
  - a narrower top-level-only post-calculus presentation path for this family
  - or a residual verifier that can compare protected compact
    `sqrt(A)*sqrt(B)` denominators with the merged `(A*B)^(-1/2)` form without
    re-entering broad rationalization
- retained follow-up:
  - the later combination kept the compact public presentation and added a
    narrow residual matcher for scaled reciprocal half-power products, including
    the fraction-addition result shape `(compact * scale - merged) / const`
  - focused residual, `engine-fast`, `engine-scorecard`, and
    `engine-scorecard-pressure` all passed with failed = 0

### 2026-05-13: Reciprocal-Shifted Integration By Parts Trig Residual Needs Denominator Re-Entry

- area:
  - arithmetic / embedded reciprocal wrapper / calculus integration residual
- status:
  - `retained`
- discovered case:
  - `1/((integrate(x^2*cos(x),x))+c) - 1/((2*x*cos(x)+(x^2-2)*sin(x))+c)`
  - `1/((integrate(x^2*sin(x),x))+c) - 1/((2*x*sin(x)+(2-x^2)*cos(x))+c)`
- local lane:
  - focused CLI probes while selecting a minimal `calculus_integrate`
    `reciprocal_shifted_difference_zero` corpus representative
- local result:
  - direct antiderivative residuals such as
    `diff(integrate(x^2*cos(x),x),x)-x^2*cos(x)` simplify to `0`
  - direct reciprocal differences without the shared `+c` also simplify to `0`
  - the retained fix adds a narrow reciprocal-denominator re-entry for
    `1/D1 - 1/D2` when both denominators are small additive trig/hyperbolic
    sums and existing simplification proves `D1-D2 = 0`
- promoted coverage:
  - `calculus_integrate_poly_cos_by_parts_trig_denominator_reentry_zero`
    covers the shifted reciprocal wrapper without adding a new integration
    rule
- remaining caution:
  - the emitted `NonZero` requirements can still contain equivalent duplicate
    denominator forms; normalize that separately if it becomes user-visible

### 2026-05-13: Additive Passthrough Diff Residual Needs Re-Entry After Cancelling Noise

- area:
  - orchestrator / calculus residual / additive passthrough
- status:
  - `resolved`
  - originally logged as `discovery/observe-only`
- discovered case:
  - `((diff(exp(sin(x)),x)+m) - (cos(x)*e^sin(x)+m))`
  - `((diff(sin(e^(x^2)),x)+m) - (2*x*cos(e^(x^2))*e^(x^2)+m))`
- local lane:
  - focused CLI probes while selecting a minimal `calculus_diff` wrapper-spread
    corpus candidate
- local result:
  - direct residuals such as
    `diff(e^sin(x), x) - cos(x)*e^sin(x)` simplify to `0`
  - denominator-preserving wrappers such as
    `diff(exp(sin(x)),x)/q - (cos(x)*e^sin(x))/q` simplify to `0`
  - the additive passthrough shape cancels the shared `+m` noise but leaves the
    direct diff residual unsimplified
- why it was not promoted as coverage:
  - promoting an easier additive `calculus_diff` row would mark the wrapper as
    covered while hiding a reproducible routing gap for exp/trig chain
    derivatives
- what could make it combinable later:
  - a narrow re-entry after shared additive passthrough cancellation that sends
    the remaining diff residual back through the exact-zero calculus residual
    route without adding broad no-match traffic

### 2026-05-13: Isolated Context Transplant Must Reintern Function Symbols

- area:
  - orchestrator / isolated simplification / symbolic calculus calls
- status:
  - `observe-only`
- discovered case:
  - `(diff(sin(e^(x^2)),x) - 2*x*cos(e^(x^2))*e^(x^2)) + (u*v+u*w-u*(v+w))`
- local lane:
  - focused CLI probe while searching for a minimal `calculus_diff`
    cross-family composition candidate
- local result:
  - the generated candidate panicked inside `SymbolTable::resolve` after a
    `diff(...)` function call was transplanted into an isolated context
  - the reusable weakness was that `transplant_expr_subtree` reinterned
    variables but preserved source-context function `SymbolId`s
- why it was not promoted as coverage first:
  - the candidate exposed a structural robustness defect before corpus
    promotion, so the retained work was the context-transplant fix plus a unit
    regression
- what could make it combinable later:
  - after the robustness fix remains green under pressure, retry a minimal
    `calculus_diff` mixed-composition corpus representative if the scorecard
    still shows that family under target

### 2026-05-13: Broad Post-Calculus Trace Compaction Hides Meaningful Presentation Steps

- area:
  - calculus / didactic trace / post-calculus presentation
- status:
  - `rejected`
- local lane:
  - focused `diff_step_contract_tests` probes for reciprocal `asinh(sqrt(...))`
    and `atanh(sqrt(...))` derivatives after removing an internal
    rationalize route
- local win:
  - the broad trace compaction removed a now-redundant final
    `Present calculus result in compact form` step when the calculus result was
    already display-equivalent to the final compact form
- global result:
  - `make engine-fast` failed `calculus_diff_contract` with 11 regressions
    where tests intentionally require the compact post-calculus presentation
    step to remain visible
- why it regressed globally:
  - display-equivalence between a calculus step and final result is not enough
    to decide that presentation is didactically redundant
  - several derivative families use the presentation step as the visible,
    meaningful bridge from power/root internals to public reciprocal-root form
- what could make it combinable later:
  - a narrow trace-quality classifier that distinguishes redundant cleanup
    roundtrips from meaningful presentation transitions
  - or family-specific metadata on calculus rewrites indicating whether the
    presentation step is explanatory or purely cosmetic

### 2026-05-11: High-Power Log Product Integration Needs Residual Verification Narrowing

- area:
  - calculus / integration / by-parts log powers / residual simplification
- status:
  - `observe-only`
- resolved by:
  - 2026-05-22 calculus promotion:
    high log-power product integration now allows positive-leading quadratic
    bases at powers `4..=5` only with the existing explicit `f(x) > 0` domain
    condition, and the bounded public residual verifier closes the promoted
    conditional cases to `0`
- discovered case:
  - `integrate((2*x+1)*ln(x^2+x+1)^5, x)`
  - `integrate(2*x*ln(x^2-1)^4, x)`
- local lane:
  - focused CLI probes while extending `f'(x)*ln(f(x))^k` by-parts support
    from `k=2..3` toward `k=4..5`
- local result:
  - the formal by-parts primitive is generated, but the public residual
    `diff(integrate(...), x) - integrand` does not simplify to `0` for shifted
    or conditional polynomial bases at `k>=4`
  - the narrower positive even monic quadratic base, for example
    `integrate(2*x*ln(x^2+1)^5, x)`, does verify publicly by differentiation
- why it was not promoted broadly:
  - promoting the general `f'(x)*ln(f)^k` route would expose correct-looking
    antiderivatives that do not satisfy the current public verification
    guardrail
  - the retained route was narrowed to bases where residual simplification is
    already strong enough
- what could make it combinable later:
  - a residual simplification improvement for high-power log-polynomial
    products over shifted or conditional polynomial bases
  - or a verified presentation route that keeps a residual-friendly expanded
    form internally while rendering a compact factored primitive publicly
- retained follow-up:
  - a later calculus contract promoted the shifted positive quadratic
    subcase `integrate((2*x+1)*ln(x^2+x+1)^5, x)` after the public residual
    `diff(integrate(...), x) - (2*x+1)*ln(x^2+x+1)^5` simplified to `0`
  - this retains the conservative positive-base slice without promoting the
    broader conditional-base family
- observe-only follow-up:
  - a later robustness cycle kept `integrate(2*x*ln(x^2-1)^4, x)` explicitly
    unsupported while preserving `x^2 - 1 > 0`, because the formal primitive
    residual does not simplify to `0` publicly and `equiv(diff(primitive),
    integrand)` only succeeds after a slow path with repeated `depth_overflow`
    warnings
  - the shifted conditional probe
    `diff(primitive(x^2+x-1, 4), x) - (2*x+1)*ln(x^2+x-1)^4` expanded into a
    much larger residual and exceeded a practical probe budget, so it remains a
    residual/cancellation discovery rather than an integration promotion

### 2026-05-10: Product-Wrapped Reciprocal Trig Derivative Integration Hits Public Routing Cliff

- area:
  - calculus / integration / reciprocal trig derivative / product-wrapper routing
- status:
  - `promoted-after-root-gate`
- discovered case:
  - `integrate((2*x+1)*sin(x^2+x)/cos(x^2+x)^2, x)`
  - `integrate((2*x+1)*cos(x^2+x)/sin(x^2+x)^2, x)`
- local lane:
  - focused CLI JSON probes with 4s timeouts after raw reciprocal trig
    derivative quotient coverage was already green
  - temporary low-level `cas_math::integrate_symbolic_expr` probe that
    combined the external polynomial cofactor into the quotient numerator
- local result:
  - direct derivative quotient shape
    `integrate((2*x+1)/cos(x^2+x)^2, x)` returns `tan(x^2 + x)` quickly
  - the low-level matcher can integrate the product-wrapped raw shapes when
    presented as a combined quotient numerator
  - the public CLI spelling with the polynomial factor outside the quotient
    still times out and emits repeated `depth_overflow` warnings before
    promotion
- why it was not promoted:
  - the public route appears to pre-simplify or recurse through the
    product/quotient expression before the root symbolic integration matcher
    sees the reusable `u' * trig(u) / trig(u)^2` structure
  - retaining only the low-level helper would make an internal test green while
    leaving the public engine behavior unfixed
- retained action:
  - a later combination pass retained a narrow root gate before generic
    required-condition and presentation probes for public `integrate(...)`
  - the gate only accepts polynomial reciprocal trig derivative quotients and
    returns the same nonzero denominator condition as the underlying symbolic
    integration matcher
  - the public antiderivative residual path now reuses the same gate and can
    compare raw `sin(u)/cos(u)^2` and `cos(u)/sin(u)^2` forms against the
    derivative of the compact `sec(u)` / `-csc(u)` primitive
- what could make it combinable later:
  - the retained root gate can be used as the pattern for future calculus
    routing cliffs, but only when a cheap family recognizer can return both the
    antiderivative and its required conditions before broad probing

### 2026-05-10: Raw Negative/Noise Calculus Residual Wrappers Needed Additive Gate

- area:
  - calculus / residual simplification / discovery harness
- status:
  - `superseded-by-retained-additive-gate`
- discovered case:
  - generated wrappers around public antiderivative residuals:
    `((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))-1)/(x+2)`
    and
    `(((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)/(x+2)) + (x-x)`
- local lane:
  - `scripts/engine_calculus_residual_probe_smoke.py` exploratory probes with
    an 8s wall budget before harness promotion
- local result:
  - raw negative and additive-noise wrappers timeout for the representative
    `sinh` and `cosh` by-parts residual families
  - a narrower signed orientation wrapper using an external constant factor is
    stable and was promoted only to the discovery harness:
    `(-1)*(({residual}+1)/(x+2)) -> -1/(x+2)`
- retained action:
  - a later robustness pass retained a bounded helper update:
    `core - constant` now compacts through the existing constant-passthrough
    quotient route, and exact syntactic zero noise such as `x-x` is stripped
    only around the residual quotient helper
  - the retained discovery harness now covers both:
    `(({residual})-1)/(x+2) -> -1/(x+2)` and
    `(({residual}+1)/(x+2))+(x-x) -> 1/(x+2)`
- why it was not promoted before the retained gate:
  - the raw subtraction/noise forms re-enter the heavier hyperbolic residual
    path before the proven-zero residual is compacted
  - promoting them to default or live coverage would add known timeout traffic
    rather than a retained public calculus capability
- retained learning:
  - the reusable fix was not a broader hyperbolic simplifier; it was a narrow
    pre-route around already-supported residual quotient compaction

### 2026-05-10: Deep Hyperbolic Antiderivative Residual Wrapper Timeout After One-Level Quotient Retention

- area:
  - calculus / residual simplification / discovery harness
- status:
  - `observe-only`
- resolved by:
  - 2026-05-22 observability close-out:
    later retained robustness work covers the deeper hyperbolic
    constant-passthrough quotient residual wrappers; current public probes and
    the focused smoke matrix show `hyperbolic_sinh:double_nested_den` and
    `hyperbolic_cosh:double_nested_den` passing quickly without warnings while
    preserving `x + 2`, `x + 3`, and `x + 4` nonzero requirements
- discovered case:
  - generated wrapper probes around public antiderivative residuals:
    `2*(((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)/(x+2))`
    and
    `((((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)/(x+2))/(x+3))`
- local lane:
  - `scripts/engine_calculus_residual_probe_smoke.py` exploratory probes with
    2s and 8s wall budgets before corpus promotion
- local result:
  - the product-denominator wrapper
    `(({residual})+1)/((x+2)*(x+3))` passes quickly across the representative
    calculus residual families and was promoted only to the discovery harness
  - the scaled quotient wrapper was later retained by allowing a single
    nonzero constant factor around the existing constant-passthrough quotient
    route:
    `2*(({residual}+1)/(x+2)) -> 2/(x+2)`
  - the one-level nested quotient wrapper was later retained by compacting the
    inner quotient first and then multiplying denominators:
    `(({residual}+1)/(x+2))/(x+3) -> 1/((x+2)*(x+3))`
  - deeper repeated quotient wrappers remain intentionally unpromoted
- why it was not promoted:
  - the remaining deeper quotient failure is a reusable structural runtime
    cliff in wrapped hyperbolic antiderivative residual simplification, not a
    stable public capability
  - promoting arbitrary quotient nesting to live/contract coverage would add
    hot timeout pressure rather than retained mathematical completeness
- what could make it combinable later:
  - a cheap residual-shape gate that recognizes a proven zero calculus residual
    before deeper repeated quotient wrappers trigger the heavier
    hyperbolic-by-parts simplification path

### 2026-05-09: Sparse Affine Hyperbolic By-Parts Internal Verification Needed A Public Residual Gate

- area:
  - calculus / integration / antiderivative verification harness
- status:
  - `promoted-after-harness-follow-up`
- discovered case:
  - `integrate((x^3+x)*sinh(2*x+1), x)`
  - public residual:
    `diff(integrate((x^3+x)*sinh(2*x+1), x), x) - (x^3+x)*sinh(2*x+1)`
- local lane:
  - focused CLI residual probe after adding the bounded hyperbolic integral
    residual matcher
  - attempted promotion to
    `REPRESENTATIVE_ANTIDERIVATIVE_VERIFICATION_CASES`
- local result:
  - public CLI residual now returns `0` with no required conditions and no
    `depth_overflow`
  - focused contract row for the same expression is cheap and exposes
    `Usar integración por partes repetida`
- why it was not promoted:
  - the representative antiderivative verifier first differentiates and
    simplifies the rendered antiderivative through an internal path before the
    public residual fallback can apply
  - that internal verification path stayed running for more than three minutes
    on the sparse affine hyperbolic cubic case, so live representative
    promotion would add a slow test even though the public residual is now fast
- retained action:
  - promoted after a harness follow-up: the representative verifier now uses
    the bounded public `diff(integrate(...), x) - integrand` residual route
    first for this sparse affine hyperbolic case
  - kept the focused CLI integration contract as the step-quality and result
    presentation guardrail
- what could make it combinable later:
  - generalize the public-residual-first gate only for future families with a
    measured internal verification cliff and a bounded public residual matcher

### 2026-05-08: Affine Hyperbolic By-Parts Compact-Presentation Gate Fires Too Late

- area:
  - calculus / integration / post-calculus presentation
- status:
  - `promoted-after-follow-up`
- discovered case:
  - `integrate((2*x+3)*sinh(2*x+1), x)`
  - `integrate((2*x+3)*cosh(2*x+1), x)`
- local lane:
  - focused CLI JSON probes with `--steps on` while trying to remove
    expand/refactor noise from affine hyperbolic integration-by-parts traces
- local result:
  - public results are correct and compact:
    `1/2*(cosh(2*x+1)*(2*x+3)-sinh(2*x+1))`
    and
    `1/2*(sinh(2*x+1)*(2*x+3)-cosh(2*x+1))`
  - adding an `IntegrateRule` compact-preservation gate for
    `P1(x)*sinh/cosh(ax+b)` does not remove the earlier
    `Expandir la expresión` step
- why it was not promoted:
  - initial attempt was not promoted because by the time `IntegrateRule` runs,
    the integrand has already been expanded
    to a sum such as `3*sinh(2*x+1) + 2*x*sinh(2*x+1)`, so the compact product
    detector no longer sees the original product
  - forcing compactness at this point would require a narrower pre-integration
    simplification/orchestration gate, not another post-result hold
  - a follow-up attempt to preserve the raw `integrate(...)` target with the
    symbolic integration detector was rejected on runtime: the focused
    didactic contract stayed green but took about 67s, which violated the
    retained-ROI guardrail
- retained action:
  - kept the didactic improvement that recognizes linear by-parts terms inside
    an already-expanded integrand and exposes `Usar integración por partes`
  - promoted a cheap syntactic guard in the historical polynomial
    `DistributeRule`: preserve `sinh/cosh(...) * (additive)` products instead
    of distributing them by default. This lets supported affine hyperbolic
    by-parts integrals start directly with `Calcular la integral`, preserves
    compact post-calculus presentation, and also keeps rational affine
    hyperbolic by-parts results factored as `2/3*f(...)*(x+1)`
- what could make it combinable later:
  - extend the same presentation policy to trig/exp by-parts only if a
    similarly cheap syntactic guard keeps `failed=0` and does not hide useful
    explicit expansion in non-calculus paths

### 2026-05-07: Quadratic `arcsec/arccsc` Residual Gap Expansion Promoted With Narrow Residual Gates

- area:
  - calculus / differentiation / residual simplification
- status:
  - `promoted`
- discovered case:
  - factored-gap residuals collapse but are slow enough to avoid promoting as a
    hot contract row:
    `diff(arcsec(x^2+x+3), x) - (2*x+1)/((x^2+x+3)*sqrt((x^2+x+3)^2-1))`
    -> `0`
  - the analogous `arccsc` factored-gap residual also collapses:
    `diff(arccsc(x^2+x+3), x) + (2*x+1)/((x^2+x+3)*sqrt((x^2+x+3)^2-1))`
    -> `0`
  - expanded-gap residuals originally did not collapse:
    `diff(arcsec(x^2+x+3), x) - (2*x+1)/((x^2+x+3)*sqrt(x^4+2*x^3+7*x^2+6*x+8))`
- local lane:
  - CLI probes while considering promotion of a shifted positive-quadratic
    inverse reciprocal trig residual after direct `diff` coverage already
    existed
- local result:
  - standalone derivative presentation succeeds:
    `diff(arcsec(x^2+x+3), x)` ->
    `(2*x+1)/((x^2+x+3)*sqrt(x^4+2*x^3+7*x^2+6*x+8))`
  - standalone `arccsc` presentation succeeds with the opposite sign
  - the factored-gap residual can prove the equality but takes multiple seconds
    in the CLI timing path
  - the expanded-gap residual leaves a difference of two equivalent radical
    denominators instead of reusing the existing proof that the inner
    polynomial gaps are equivalent
  - minimal probes confirm the algebraic subproblem itself is solvable:
    `sqrt(x^4+2*x*x^2+x^2+6*x^2+6*x+9-1) - sqrt(x^4+2*x^3+7*x^2+6*x+8)`
    -> `0`
- why it was not originally promoted:
  - promoting the factored residual would add a slow row to the public `diff`
    contract without adding a new calculus family
  - promoting the expanded residual would encode a known failure rather than a
    retained capability
  - the reusable weakness is residual re-entry/cancellation across equivalent
    polynomial radical denominators, not the derivative rule for `arcsec` or
    `arccsc`
- retained action:
  - promoted on 2026-05-07 by extending polynomial-denominator fraction
    residual cancellation with an extended component-gated budget and a
    one-square-root numerator equivalence check
  - added a positive-quadratic inverse-reciprocal-trig pre-order matcher for
    arguments whose minimum is greater than `1`, so public expanded residuals
    close before the expensive derivative route
  - promoted minimal `diff` contract rows for `arcsec` and `arccsc` with no
    required conditions on `x^2+x+3`
- what could make it combinable later:
  - a narrow residual-cancellation route for two fractions whose numerators are
    opposite and whose denominator factors differ only by small univariate
    polynomial-normalized radical arguments
  - a cheap gate that avoids running polynomial/radical equivalence unless both
    sides are already two-term calculus residuals with matching non-radical
    factors

### 2026-05-07: Quadratic Self-Normalized Inverse-Trig Residual Needs Polynomial-Denominator Normalization

- area:
  - calculus / differentiation / residual simplification
- status:
  - `superseded`
- discovered case:
  - public derivative presentation succeeds:
    `diff(arccos((x^2+x+1)/sqrt((x^2+x+1)^2+5)), x)` ->
    `-sqrt(5)*(2*x+1)/((x^2+x+1)^2+5)`
  - nested residual probe does not currently collapse:
    `diff(arccos((x^2+x+1)/sqrt((x^2+x+1)^2+5)), x)
    + sqrt(5)*(2*x+1)/((x^2+x+1)^2+5)`
- local lane:
  - CLI probes during bounded `diff` post-calculus presentation coverage for
    self-normalized inverse-trig projections
- local result:
  - the standalone derivative is correct and compact
  - `equiv(x^4 + 2*x^3 + 3*x^2 + 2*x + 6, (x^2+x+1)^2+5)`
    returns `true`
  - the residual remains as a sum of fractions whose denominators are
    equivalent but rendered through different polynomial expansion paths
- why it was not promoted:
  - promoting the residual as an active contract would exercise a broader
    polynomial-denominator equivalence gap rather than the calculus
    presentation capability itself
  - the likely fix belongs in reusable residual/equivalence normalization for
    denominators, not in an answer-only calculus shortcut
- retained action:
  - retained a cheap helper-level regression for the public quadratic
    self-normalized projection presentation
  - promoted the nested residual on 2026-05-07 with a bounded
    polynomial-denominator fraction residual cancellation helper
  - the promoted route is gated to two fractions, small expressions, opposite
    numerators, and univariate polynomial-equivalent denominators
- what could make it combinable later:
  - a narrow denominator-normalization pass inside residual cancellation that
    canonicalizes small univariate polynomial denominators before comparing or
    combining matching fractions
  - an equivalence-aware fraction residual route with a cheap polynomial gate

### 2026-05-07: Scaled Affine `asinh` Reciprocal-Root Integration Needs Radical-Scale Normalization

- area:
  - calculus / integration / inverse-hyperbolic reciprocal-root kernels
- status:
  - `partially-promoted`
- discovered case:
  - `integrate(1/((6-2*x)*sqrt(8-2*x)), x)`
  - equivalent primitive shape:
    `sqrt(2)/2 * asinh(sqrt(1/(3-x)))`
- local lane:
  - CLI probes while extending conservative `asinh(sqrt(c/base))`
    inverse-differentiation coverage
- original local result:
  - direct supported normalized neighbor succeeds:
    `integrate(1/((3-x)*sqrt(4-x)), x)` ->
    `2*asinh(sqrt(1/(3-x)))` with `3-x > 0`
  - scaled candidate originally remained unsolved and returned an `integrate(...)`
    residual with `x-3 != 0` and `4-x > 0`
  - explicit verification probe
    `equiv(diff(sqrt(2)/2*asinh(sqrt(1/(3-x))), x),
    1/((6-2*x)*sqrt(8-2*x)))` returns `true`
  - that verification emits a `depth_overflow` warning and routes through a
    high-depth radical/absolute-value residual, so it was not promotion-ready
- promoted result:
  - the scaled integral is now promoted through the existing reciprocal-root
    family with an explicit radical scale:
    `integrate(1/((6-2*x)*sqrt(8-2*x)), x)` ->
    `1/2*asinh(sqrt(1/(3-x)))*sqrt(2)` with `3-x > 0`
  - the contract verifies the antiderivative by differentiation and confirms
    that nested `diff(integrate(...), x)` does not leave an integration
    residual
  - direct `diff(integrate(...), x)` now renders back to the compact input
    kernel instead of exposing the intermediate radical/absolute-value route
- why it was originally not promoted:
  - historical:
    - retaining it cleanly required representing an irrational outer scale
      such as `sqrt(2)/2` in the antiderivative family and normalizing the
      witnessed condition to `3-x > 0`
    - adding a direct answer-only shortcut would have hidden a real
      normalization and verification weakness behind a one-off integration
      result
- remaining gap:
  - retaining it cleanly would require representing an irrational outer scale
    in a prettier public order without relying on multiplication reordering
  - the manual residual probe
    `diff(integrate(...), x) - 1/((6-2*x)*sqrt(8-2*x))` can still emit
    `depth_overflow` before reducing to `0`; this is residual-simplification
    debt, not a reason to reject the promoted integration family
- what could make it combinable later:
  - post-calculus presentation that renders the outer factor compactly as
    `sqrt(2)/2 * asinh(sqrt(1/(3-x)))`, while preserving `3-x > 0`
  - residual simplification that avoids the transient `0 / (...)`
    `depth_overflow` warning

### 2026-05-06: Real-Domain Sqrt Product Equivalence Needs Condition-Aware Proof

- area:
  - equivalence / real-domain radicals / calculus feedback
- status:
  - `partially-superseded`
- discovered case:
  - `equiv(sqrt(1/2)/(sqrt(2*x+3)*sqrt(-x-1)), 1/(sqrt(2*x+3)*sqrt(-2*x-2)))`
- local lane:
  - CLI probes while hardening `diff(arcsin/arccos(sqrt(2*x+3)), x)` post-calculus presentation
- local result:
  - numeric evaluation and chain-rule reasoning show the forms agree under the
    shared real-domain conditions `2*x+3 > 0` and `-2*x-2 > 0`
  - symbolic `equiv(...)` originally returned `false`
- why it was not promoted:
  - the tempting general identity `sqrt(a*b) = sqrt(a)*sqrt(b)` is not valid
    over all real inputs without sign/domain conditions, so a global
    simplifier/equivalence shortcut would be unsound
- retained action:
  - keep non-square gap content inside the post-calculus `sqrt(gap)` for the
    affected `diff` presentation, avoiding reliance on this unimplemented
    equivalence
  - follow-up retained a narrow residual-cancellation matcher for `a - a` and
    `a + (-a)` forms by comparing squared products of affine radical factors;
    this closes the concrete verifier gap without adding a global
    `sqrt(a*b)` simplification rule
  - safety probes also showed that a separate pre-existing public
    simplification route can still collapse split root products such as
    `sqrt((x+1)*(x+2)) - sqrt(x+1)*sqrt(x+2)`; follow-up inspection showed
    that route is the public `RootMergeMulRule`, and it carries explicit
    non-negative `Requires` conditions in generic mode while staying disabled
    in strict mode
  - CLI semantic contracts now pin the `sqrt(a)*sqrt(b)` and
    `sqrt(a)/sqrt(b)` root-merge behavior to those required conditions
  - follow-up assume-mode contracts keep the same intrinsic root-domain facts
    under `Requires`, not `Assume`, because these conditions come from the
    source `sqrt` definedness rather than a heuristic assumption introduced by
    the rewrite
- what could make it combinable later:
  - a condition-aware radical product proof that carries explicit positive
    radicand assumptions through `equiv`/residual verification


### 2026-05-06: Sqrt-Scaled Cosh Log Presentation Exposes Common-Factor Residual Gap

- area:
  - calculus / post-integration presentation / residual simplification
- status:
  - `superseded`
- attempted case:
  - compact the public result for `integrate(tanh(sqrt(2*x))/sqrt(2*x), x)`
    from `ln(|cosh(sqrt(2 * x))|)` to `ln(cosh(sqrt(2 * x)))`, relying on
    real positivity of `cosh`
- local lane:
  - `cargo test -q -p cas_cli --test integrate_contract_tests integrate_contract_sqrt_chain_hyperbolic_tangent_logs_verify -- --exact --nocapture`
  - CLI probes for the residual forms:
    `sinh(sqrt(2*x))/cosh(sqrt(2*x)) - tanh(sqrt(2*x))` and
    `sinh((2*x)^(1/2)) * (2*x)^(-1/2) / cosh((2*x)^(1/2)) - tanh((2*x)^(1/2)) * (2*x)^(-1/2)`
- local result:
  - the direct quotient identity reduces to `0`
  - the differentiated-antiderivative residual with a shared
    `(2*x)^(-1/2)` multiplier remains unsimplified, while an explicitly
    factored equivalent reduces to `0`
- global result:
  - superseded on 2026-05-06 by adding a narrow residual verifier for
    `sinh(u)*k/cosh(u) - tanh(u)*k` and a direct
    `diff(ln(cosh(sqrt(p(x)))), x)` verification path for constant-derivative
    radicands
  - promoted `integrate(tanh(sqrt(2*x))/sqrt(2*x), x)` to render as
    `ln(cosh(sqrt(2 * x)))` with the original `x > 0` condition
- best current explanation:
  - residual simplification does not currently factor a common multiplier
    across `A*B/C - D*B` before applying the existing `sinh(u)/cosh(u) =
    tanh(u)` identity
- plausible follow-up:
  - extend only after another verified representative requires it; avoid
    broad post-order factorization unless embedded runtime remains stable

### 2026-05-06: Trig `u'/(1+u^2)` Integration Probe Hit Slow Pre-Simplification

- area:
  - calculus / integration / trig substitution pre-simplification
- status:
  - `superseded`
- attempted case:
  - probe the same conservative `u'/(1+u^2) -> arctan(u)` shape for
    trigonometric arguments, for example
    `integrate(cos(x)/(1+sin(x)^2), x)` and
    `integrate(2*cos(2*x+1)/(1+sin(2*x+1)^2), x)`
- local lane:
  - CLI probes:
    `cargo run -q -p cas_cli -- eval "integrate(cos(x)/(1+sin(x)^2), x)" --format text`
    and
    `cargo run -q -p cas_cli -- eval "integrate(2*cos(2*x+1)/(1+sin(2*x+1)^2), x)" --format text`
- local result:
  - the first probe emitted `depth_overflow` warnings and returned an
    unsolved `integrate(...)` residual after the denominator was rewritten into
    a double-angle-shaped form
  - the affine trig probe exceeded one minute and was manually terminated
- global result:
  - superseded on 2026-05-06 by promoting the minimal trig representatives to
    the public integration contract after the current integration route
    resolved them quickly and verified the antiderivatives by differentiation
  - retained both the hyperbolic and trig `u'/(1+u^2)` representatives whose
    denominators remain structurally stable under the current route
- best current explanation:
  - generic pre-simplification rewrites `1+sin(arg)^2` into a cos/double-angle
    equivalent before the integration matcher can see the substitution shape
- plausible follow-up:
  - keep this as a regression signal rather than an active combination
    candidate; any broader trig substitution family still needs the same
    antiderivative verification and failed=0 guardrails before promotion

### 2026-05-05: Shifted Arctan Affine Antiderivative Verification Probe

- area:
  - calculus / integration verification / inverse-trig affine by-parts residuals
- status:
  - `superseded-by-retained-residual-closure`
- attempted case:
  - generalize the retained `integrate(arctan(a*x), x)` by-parts rule to
    shifted affine arguments such as `integrate(arctan(2*x+1), x)` and
    `integrate(arctan(1-2*x), x)`
- local lane:
  - CLI residual probes:
    `cargo run -q -p cas_cli -- eval 'diff(integrate(arctan(2*x+1), x), x)-arctan(2*x+1)' --no-pretty`
    and
    `cargo run -q -p cas_cli -- eval 'diff(integrate(arctan(1-2*x), x), x)-arctan(1-2*x)' --no-pretty`
- local result:
  - the direct primitive is mathematically standard and can be made to reduce
    to `0` when internal hold barriers preserve the compact affine square
  - the rendered antiderivative reparsed as plain input still expands into a
    deep rational residual and emits `depth_overflow` warnings before failing
    to reliably reduce to `0`
- global result:
  - not promoted as public integration coverage in this iteration
  - retained only the safer post-integration presentation cleanup for the
    existing zero-offset `integrate(arctan(a*x), x)` family
- best current explanation:
  - shifted affine `arctan` by-parts verification lacks the compact derivative
    recognition already added for shifted reciprocal `arctan`; without that
    recognition, the residual expands through polynomial rational forms before
    cancellation
- plausible follow-up:
  - add a narrow derivative/residual recognizer for the compact
    `u/a*arctan(u) - ln(1+u^2)/(2a)` by-parts form, then promote shifted
    affine `arctan` only if both internal and rendered antiderivative
    verification are quiet and reduce to `0`

### 2026-05-03: Broad Positive-Factor Nonzero Dominance For Post-Calculus Sqrt Denominators

- area:
  - calculus / domain normalization / post-diff presentation
- status:
  - `rejected`
- attempted case:
  - allow `diff(arccos(x), x)` and `diff(arccos(2*x+1), x)` to render as
    `-1 / sqrt(1 - x^2)` and `-1 / sqrt(-x^2 - x)` by dropping denominator
    boundary guards dominated by the strict positive radicand
- local lane:
  - `cargo test -p cas_solver bounded_inverse_trig_diff_evaluates_with_strict_required_domain_conditions --test diff_step_contract_tests`
  - `cargo test -p cas_solver affine_arcsin_diff_drops_scaled_nonnegative_domain_shadow --test diff_step_contract_tests`
- local win:
  - the broad factor-containment dominance removed the redundant `x - 1 != 0`,
    `x + 1 != 0`, `x != 0`, and `x + 1 != 0` guards for the targeted
    `arccos` presentation probes
- global result:
  - rejected after
    `cargo test -p cas_solver_core domain_normalization --lib` failed
    `atomic_positive_factors_dominate_composite_log_argument_positive_condition`
  - the retained version is narrower: only one-variable polynomial positive
    conditions may dominate nonzero polynomial factors
- why it regressed globally:
  - a positive composite quotient can be removed later as redundant under
    separate positive factors; using that transient condition to remove an
    atomic `NonZero` guard loses a required displayed assumption
- what could make it combinable later:
  - dependency-aware dominance ordering, or a proof tag that the dominating
    positive condition is retained in the final displayed condition set

### 2026-05-02: Affine Reciprocal Arctan Antiderivative Verification Probe

- area:
  - calculus / integration verification / inverse-trig reciprocal residuals
- status:
  - `superseded`
- attempted case:
  - promote the shifted reciprocal inverse-trig family behind
    `integrate(arccot(2*x+1), x)` / `integrate(arctan(1/(2*x+1)), x)`
- local lane:
  - CLI residual probe:
    `cargo run -q -p cas_cli -- eval 'diff((2*x+1)*arctan(1/(2*x+1))/2 + 1/4*ln((2*x+1)^2+1), x) - arctan(1/(2*x+1))' --format json`
  - companion public residual probe:
    `cargo run -q -p cas_cli -- eval 'diff((2*x+1)*arccot(2*x+1)/2 + 1/4*ln((2*x+1)^2+1), x) - arccot(2*x+1)' --format json`
- local result:
  - the original probes preserved the expected `2*x + 1 ≠ 0` condition, but the
    residual did not reduce to `0`
  - the failed version emitted repeated `depth_overflow` warnings and took
    about 4.7s before returning a nonzero residual expression
  - superseded on 2026-05-02 by preserving the raw shifted reciprocal
    `arctan` by-parts derivative target and recognizing the compact
    `u/a*arctan(1/u) + ln(u^2+1)/(2a)` derivative form
- global result:
  - promoted as public integration coverage for
    `integrate(arctan(1/(2*x+1)), x)` and `integrate(arccot(2*x+1), x)`
  - antiderivative verification now reduces to `0` while preserving
    `2*x + 1 ≠ 0`
- best current explanation:
  - the shifted reciprocal residual expands into polynomial rational forms
    before recognizing the compact cancellation between the inverse-trig term
    and the log derivative
- plausible follow-up:
  - no longer an open observe-only discovery; future work should generalize
    additional affine orientations only through the same verified by-parts
    path and public contract discipline

### 2026-05-01: Shifted Quadratic Arcsin Antiderivative Verification Probe

- area:
  - calculus / integration verification / polynomial-square presentation
- status:
  - `superseded`
- attempted case:
  - promote `integrate((2*x+2)/sqrt(3-(x^2+2*x+1)^2), x)` as a public
    shifted-polynomial `arcsin` antiderivative contract with diff verification
- local lane:
  - `cargo test -q -p cas_cli --test integrate_contract_tests integrate_contract_shifted_polynomial_arcsin_surd_width_dedupes_positive_domain -- --nocapture`
  - probe:
    `cargo run --release -q -p cas_cli -- eval 'simplify(3-(x^2+2*x+1)^2)' --format json --budget small`
- local result:
  - integration produced the expected `arcsin(1/3 * 3^(1/2) * (x^2 + 2*x + 1))`
    shape, but antiderivative verification did not reduce the residual to `0`
  - a later probe showed the suspected expansion sign issue was not a semantic
    polynomial failure: the expanded AST proves equivalent to
    `2 - x^4 - 4*x^3 - 6*x^2 - 4*x`; the rendered flat display remains easy to
    misread because it does not expose grouping around the subtracted expanded
    sum
- global result:
  - the factored-radicand case is now promoted as a public integrate contract
    after integration compacts the antiderivative argument to `(x + 1)^2`
  - the retained contract verifies the antiderivative by `diff` and preserves
    the compact strict condition `3 - (x + 1)^4 > 0`
- best current explanation:
  - verification was blocked by presentation/form mismatch in the
    antiderivative argument, not by a wrong polynomial expansion
- plausible follow-up:
  - consider a separate display or normalization iteration for subtracting
    expanded sums so public strings like `3 - (...)` cannot be misread as a
    different flat polynomial

### 2026-05-01: Shifted Hyperbolic Sech-Squared Verification Probe

- area:
  - calculus / symbolic differentiation / hyperbolic reciprocal-square residuals
- status:
  - `observe-only`
- attempted case:
  - promote `integrate(sinh(2*x+1)/cosh(2*x+1)^2, x)` into the generic
    antiderivative-verifies-by-`diff` contract
- local lane:
  - CLI residual probe:
    `cargo run --release -q -p cas_cli -- eval 'diff(-(1/(2*cosh(2*x+1))), x) - sinh(2*x+1)/cosh(2*x+1)^2' --format json --budget small`
- local result:
  - the residual eventually simplified to `0` with the expected
    `cosh(2*x + 1) ≠ 0` condition
  - the probe took about 45.6s and emitted repeated `depth_overflow` warnings,
    making it too hot and noisy for promotion as a normal contract row
  - the adjacent `csch^2` residual
    `diff(-(1/(2*tanh(2*x+1))), x) - 1/sinh(2*x+1)^2` also simplified to
    `0`, but took about 3.2s and emitted repeated `cycle_detected` warnings
- global result:
  - not promoted in this iteration
  - retained work is limited to the cheaper `1/tanh(2*x + 1)` antiderivative
    verification path
- best current explanation:
  - shifted reciprocal-square hyperbolic residuals can expand into angle-sum
    and double-angle rational shapes before recognizing the compact
    reciprocal-square identity
- plausible follow-up:
  - add a cheap pre-expansion residual gate for `sech(u)^2` / `csch(u)^2`
    antiderivative checks, then reattempt as pressure rather than default live
    contract coverage

### 2026-05-01: Affine `tan` Anti-Expansion Integrate Coupling

- area:
  - calculus / trig normalization / symbolic integration
- status:
  - `superseded`
- attempted case:
  - retain `tan(2*x+1)` during post-diff simplification so
    `diff(2*x*tan(2*x+1), x)` avoids expanding through
    `sin(2*x+1)/cos(2*x+1)`
- local lane:
  - CLI probe:
    `cargo run --release -q -p cas_cli -- eval 'diff(2*x*tan(2*x+1), x)' --format json --time-budget-ms 50`
- local result:
  - derivative reduced to
    `2*tan(2*x+1) + (4*x)/cos(2*x+1)^2`
  - no `Simplification Time Budget` warning under the 50 ms probe
- global result:
  - first guardrail run failed `calculus_integrate_contract` because
    `integrate(sec(2*x+1)*tan(2*x+1), x)` simplified to the now-preserved
    `tan(2*x+1)/cos(2*x+1)` form, which the integrator did not recognize
- best current explanation:
  - the diff-side anti-expansion policy changed an integration-facing
    intermediate representation from `sin(u)/cos(u)^2` to `tan(u)/cos(u)`
    without a matching conservative antiderivative recognizer
- retained follow-up:
  - added a narrow `tan(u)/cos(u)` and `cot(u)/sin(u)` integration recognizer
    guarded by polynomial `u'` matching and explicit nonzero denominator
    conditions
  - guardrail and pressure reruns are green after the paired fix

### 2026-05-01: Shifted Hyperbolic Diff Residual Probe

- area:
  - calculus / symbolic differentiation / hyperbolic equivalence
- status:
  - `superseded`
- attempted case:
  - promote the residual
    `diff(ln(abs(sinh(2*x+1))), x)/2 - cosh(2*x+1)/sinh(2*x+1)`
    as a cheap contract case
- local lane:
  - CLI and contract probes around
    `diff(ln(abs(sinh(2*x+1))), x)` and the equivalent residual above
- local result:
  - the direct public derivative can be compacted to `1 / tanh(2*x + 1)`
    for the divided-by-two case, with the expected `sinh(2*x + 1) ≠ 0`
    domain condition
  - the residual eventually simplifies to `0`, but it still emits repeated
    `depth_overflow` / `cycle_detected` warnings and is too hot to promote as
    a normal contract representative
- global result:
  - not promoted as a retained residual test in this iteration
  - retained work is limited to compact direct `diff` output and reusable
    `cosh(u)/sinh(u) -> 1/tanh(u)` normalization
- best current explanation:
  - equivalence/simplification of shifted hyperbolic residuals can expand
    `sinh(2*x + 1)` / `cosh(2*x + 1)` into sum and double-angle forms before
    recognizing the compact reciprocal-`tanh` identity
- plausible follow-up:
  - add a cheap equivalence gate or normal-form route for reciprocal hyperbolic
    quotients before angle-sum expansion, then reattempt the residual as a
    pressure or stress representative rather than a default live contract case
- retained follow-up:
  - 2026-05-01 added a narrow root/embedded gate for
    `diff(ln(abs(sinh(u))), x)/u'` and `diff(ln(abs(cosh(u))), x)/u'`
    residuals against compact `tanh`/reciprocal-`tanh` forms, including
    quotient spellings like `cosh(u)/sinh(u)` before angle-sum expansion.
  - public probes now reduce the shifted residuals to `0` without
    `depth_overflow` or `cycle_detected`, with guardrail and pressure lanes
    still green.

### 2026-04-30: Scaled `arcsin` Antiderivative Verification Probe

- area:
  - calculus / integration verification / radical normalization
- status:
  - `superseded`
- attempted case:
  - promote `integrate(2*x/sqrt(4-x^4), x)` into the generic
    antiderivative-verifies-by-`diff` contract
- local lane:
  - CLI residual probe:
    `cargo run -q -p cas_cli -- eval 'diff(arcsin(1/2*x^2), x) - 2*x/sqrt(4-x^4)' --format json`
- original local result:
  - the residual did not simplify to zero
  - the surviving shape was
    `(x*(1/4*(4 - x^4))^(1/2) - 1/2*x*(4 - x^4)^(1/2))/(1/4*(4 - x^4))`
  - the probe also surfaced `x^2 - 2 ≠ 0` alongside the expected
    `4 - x^4 > 0`
- global result:
  - retained follow-up promoted `integrate(2*x/sqrt(4-x^4), x)` into the
    generic antiderivative-verifies-by-`diff` contract
  - the residual now simplifies to `0` with the expected `4 - x^4 > 0`
    condition after positive rational perfect-power extraction under radicals
- closure explanation:
  - residual proof for the scaled `arcsin` antiderivative needed a
    domain-safe radical scaling step such as
    `sqrt((4 - x^4)/4) -> sqrt(4 - x^4)/2`; the retained follow-up supplies
    that as a reusable positive-rational extraction under radicals
- closure:
  - closed by a narrow positive-rational coefficient extraction rule for
    `(c*u)^(1/n)` with `c > 0`; no reciprocal-root workaround was needed

### 2026-04-30: Variable-Base Log Diff Both-Variable Probe

- area:
  - calculus / symbolic differentiation / quotient simplification
- status:
  - `observe-only`
- attempted case:
  - `diff(log(x, x + 1), x)` via the change-of-base derivative of
    `ln(x + 1) / ln(x)`
- local lane:
  - CLI probe:
    `cargo run -q -p cas_cli -- eval 'diff(log(x, x+1), x)' --format json --steps on --no-pretty`
- local result:
  - the initial symbolic differentiation step produced the expected quotient
    derivative:
    `(ln(x)/(x + 1) - ln(x + 1)/x)/ln(x)^2`
  - a later `Reconocer un cociente notable` simplification collapsed a
    nonzero log quotient expression to `0`
- global result:
  - rejected before promotion; the retained calculus change is limited to
    variable-base logs whose argument is constant with respect to the
    differentiation variable
- best current explanation:
  - the notable-quotient recognizer is overmatching after nested-fraction and
    distribution steps introduce log terms in both numerator and denominator
- plausible follow-up:
  - add a narrow guard to the notable-quotient simplifier so it proves the
    numerator cancellation structurally before returning `0`, then reattempt
    the full two-variable `log(base,arg)` derivative

### 2026-04-30: Scaled `tan/cot` Integration Presentation Probe

- area:
  - calculus / symbolic integration / trig canonicalization
- status:
  - `observe-only`
- attempted case:
  - `integrate(sec(2*x + 1)^2, x) -> 1/2 * tan(2*x + 1)`
  - `integrate(csc(2*x + 1)^2, x) -> -1/2 * cot(2*x + 1)`
- local lane:
  - `cargo test --release -q -p cas_math squared_kernel -- --nocapture`
  - `cargo test --release -q -p cas_engine squared_kernel -- --nocapture`
  - CLI probes for `tan(2*x + 1)/2` and `1/2*tan(2*x + 1)`
- local result:
  - constructing the antiderivative internally as `tan(u)` / `-cot(u)` is
    straightforward in the symbolic helper
  - the public root case can then lose the required nonzero condition unless
    extra metadata is added for `sec(u)^2` / `csc(u)^2`
  - the public linear case is still rewritten to
    `sin(u)/(a*cos(u))` / `-cos(u)/(a*sin(u))`, and can duplicate conditions
    if narrow metadata is layered on top of the quotient output
- global result:
  - rejected as a presentation/domain improvement before promotion; retained
    only conservative public coverage for the current domain-preserving output
- best current explanation:
  - direct `tan(x)` remains presentable, but scaled `tan(u)/a` is canonicalized
    through quotient form by existing trig simplification traffic
  - the domain hook currently recognizes the quotient kernel more reliably than
    a direct `tan/cot` antiderivative emitted by integration
- plausible follow-up:
  - add a narrow, cycle-safe canonicalization policy for scaled trig quotients
    and a matching condition hook for direct `sec^2/csc^2` table outputs before
    expecting integration to expose `tan(u)/a` and `-cot(u)/a`

### 2026-04-29: `arcsin/arccos` Diff Radical-Domain Probe

- area:
  - calculus / symbolic differentiation / required-conditions
- status:
  - `observe-only`
- attempted case:
  - `diff(arcsin(x), x) -> 1/sqrt(1-x^2)`
  - `diff(arccos(x), x) -> -1/sqrt(1-x^2)`
- local lane:
  - `cargo test -p cas_math symbolic_differentiation_support`
  - `cargo test -p cas_solver --test diff_step_contract_tests inverse_function_diff_evaluates_with_required_domain_conditions -- --exact`
- local result:
  - the derivative formulas were easy to produce, but the public contract saw
    only `1 - x^2 >= 0` from the surviving `sqrt(1-x^2)` denominator
  - the derivative is undefined at the endpoints, so the retained condition must
    be equivalent to `1 - x^2 > 0`
- global result:
  - rejected before promotion; no guardrail regression was introduced
- decision:
  - retain only total-domain inverse derivatives in this cycle:
    `arctan` and `asinh`
- best current explanation:
  - required-condition extraction distinguishes positive radicands after some
    reciprocal-root simplifications, but the symbolic-diff constructed
    `Div(1, sqrt(...))` path can surface only non-negativity
- plausible follow-up:
  - add a domain-preserving representation or condition hook for reciprocal
    square-root denominators, then reattempt `arcsin/arccos` derivatives with a
    contract requiring `1 - x^2 > 0`

### 2026-04-29: `sin(4x)` Expanded Quadruple-Angle Derive Probe

- area:
  - derive planner / trig multiple-angle expansion
- status:
  - `retained`
- local lane:
  - `printf 'derive sin(4*x), 4*sin(x)*cos(x)^3 - 4*sin(x)^3*cos(x)\n' | target/debug/cas_cli --no-pretty repl`
  - paired control:
    `printf 'derive sin(4*x), 4*sin(x)*cos(x)*(cos(x)^2 - sin(x)^2)\n' | target/debug/cas_cli --no-pretty repl`
- local result:
  - the expanded polynomial/product target did not reach a result within the
    cheap 5s probe window
  - the factored target succeeds quickly through `Strategy: factor`
- retained follow-up:
  - added a bounded `expand trig` route for
    `sin(4*x) -> 4*sin(x)*cos(x)^3 - 4*sin(x)^3*cos(x)`
  - promoted the case into `derive_pairs.csv` as
    `expand_trig_quadruple_angle_sine_expanded_product`
  - kept the route target-aware/presentational so it does not replace the
    existing factored-target path through `Strategy: factor`

### 2026-04-29: `log_exp_inverse` Power Alias Derive Shadow Superseded

- area:
  - derive planner / embedded equivalence shadow pressure
  - `log_exp_inverse`
- status:
  - `superseded`
- attempted case:
  - `exp(y*log(x)) -> x^y`
  - root pair from `embedded_equivalence_context_corpus.csv`
- local lane:
  - CLI probe:
    `cargo run --release -q -p cas_cli -- eval "derive exp(y*log(x)), x^y"`
  - exact shadow:
    `cargo test --release -q -p cas_solver --test derive_contract_tests derive_engine_identity_shadow_pressure_reports_reachability -- --exact --nocapture`
- local result:
  - the expression derived successfully, but only through generic
    `Strategy: simplify`
  - shadow would become `sampled=46 derived=46 unsupported=0 not_equivalent=0`
  - `generic_simplify_strategy_successes` would regress from `0` to `1`
    with `embedded_log_exp_inverse_power_alias`
- global result:
  - rejected before guardrail promotion; no runtime or corpus change retained
- decision:
  - do not promote this shadow case until `derive` has a specific
    `log_exp_inverse` / `rewrite exponentials` route for the power alias
- best current explanation:
  - the engine can simplify/prove the power alias, but `derive` lacks a
    target-aware visible transition for `exp(y*log(x)) -> x^y`
  - this would create exactly the kind of magical one-step bridge that the
    shadow lane is meant to expose
- plausible follow-up:
  - extend the derive exponential/log strategy to classify and rewrite
    `exp(k*log(base)) -> base^k`
  - then re-promote the same embedded root as a shadow case and require
    `generic_simplify_strategy_successes=0`
- superseded by:
  - retained 2026-04-29 coverage fix adding a target-aware exponential
    rewrite for `exp(k*log(base)) -> base^k`
  - promoted `embedded_log_exp_inverse_power_alias` into derive shadow
    pressure, with the route attributed to `rewrite exponentials` instead of
    generic `simplify`

### 2026-04-28: Factor-Out-With-Division Identity Seed Rejected

- area:
  - corpus generation / derive shadow pressure / conditional factor bridge
- status:
  - `observe-only`
- attempted case:
  - `a*x^2 + b*x + c -> x*(a*x + b + c/x)`
  - identity seed used `conditional_requires` with `away_from(0;eps=0.01)`
- local lane:
  - CLI derive probe:
    `printf 'derive a*x^2 + b*x + c, x*(a*x + b + c/x)\n' | cargo run -q -p cas_cli -- repl --no-pretty`
  - exact shadow:
    `cargo test --release -q -p cas_solver --test derive_contract_tests derive_engine_identity_shadow_pressure_reports_reachability -- --exact --nocapture`
- local result:
  - `derive` used `Strategy: factor out with division`
  - shadow improved from `sampled=39` / `distinct_actual_strategies=24`
    to `sampled=40` / `distinct_actual_strategies=25`
  - `generic_simplify_strategy_successes` stayed `0`
- global result:
  - rejected during `make engine-scorecard`
  - `simplify_strict` multiplication combinations began timing out after the
    new seed changed the stratified identity sample; by `9000/11175` mul
    combinations it had reported `T/O 93`
  - the run was killed and the identity/shadow promotion was reverted
- why it regressed globally:
  - the seed itself is a valid derive bridge, but promoting it through
    `identity_pairs.csv` perturbed the strict metamorphic sample enough to pair
    many families with the hot `Multiple angle formulas` representative
    `cos(4*x)`
  - this exposed a product-combination timeout corridor around multiple-angle
    trig expansion, not a correctness issue in factor-out-with-division
- what could make it combinable later:
  - isolate and fix/gate the `mul x Multiple angle formulas` timeout corridor
  - or add a narrower derive-shadow source mechanism that can sample a
    documented engine seed without perturbing the broad strict identity
    selection
- decision:
  - do not promote this seed to `identity_pairs.csv` yet
  - keep `factor out with division` as the next bridge candidate only after the
    multiple-angle product timeout corridor is addressed or isolated

### 2026-04-28: Derive Hyperbolic Negative Exponential Probe Hang

- area:
  - derive planner / hyperbolic exponential decomposition
- status:
  - `superseded`
- attempted case:
  - `derive cosh(x) - sinh(x), exp(-x)`
- local lane:
  - CLI probe: `printf '%s\n' 'derive cosh(x) - sinh(x), exp(-x)' | cargo run -q -p cas_cli`
- local result:
  - the sibling positive decomposition `derive sinh(x) + cosh(x), exp(x)`
    derived quickly through a named `Hyperbolic Sum to Exponential` step
  - the negative-exponent sibling did not finish during the interactive probe
    window and was killed instead of being promoted
- global result:
  - rejected before guardrails; no runtime or corpus change was retained for
    this sibling candidate
- decision:
  - promote only the stable positive decomposition representative
  - keep the negative-exponent orientation as a planner robustness discovery
- best current explanation:
  - the target `exp(-x)` appears to route into a much heavier planner or
    equivalence path than the positive target, despite being a closely related
    hyperbolic/exponential identity
- plausible follow-up:
  - add a cheap target-aware hyperbolic sum/difference-to-exponential route for
    the negative decomposition, or gate planner preference for this orientation
    before it enters an expensive fallback path
- superseded by:
  - retained 2026-04-28 robustness fix adding a direct target-aware
    `cosh(u)-sinh(u) -> exp(-u)` derive route and promoting
    `hyperbolic_contract_negative_exp_decomposition`

### 2026-04-28: Derive Sqrt Arithmetic Multi-Extraction Snapshot Discovery

- area:
  - derive didactic quality / step snapshot preservation
  - `Combine Like Terms` over numeric radical arithmetic
- status:
  - `observe-only`
- attempted case:
  - `derive sqrt(8)+sqrt(18), 5*sqrt(2)`
- local lane:
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_sqrt_arithmetic_multi_extract_sum_shows_each_hidden_radical_extraction -- --nocapture`
- local result:
  - `derive` reached the target, but the attempted didactic assertion failed:
    only one hidden radical extraction was available to the substep generator
    where the candidate expected two
  - CLI showed the step snapshot as `sqrt(8) + 3*sqrt(2)` before combining,
    even though the original source had both `sqrt(8)` and `sqrt(18)`
- global result:
  - rejected before global guardrails; no runtime change was retained for this
    candidate
- decision:
  - do not promote the multi-extraction derive row yet
  - promote the smaller subtraction-orientation row instead
- best current explanation:
  - one radical normalization can happen before the visible derive snapshot for
    the `Combine Like Terms` step, so the didactic layer cannot reliably
    reconstruct every raw radical from `step.global_before`
- plausible follow-up:
  - preserve the original source/focused pre-normalization expression in step
    metadata, or emit real root-canonicalization prep steps before combining
    like radicals
  - avoid fabricating missing radical-extraction substeps from display text

### 2026-04-27: `log_exp_inverse` Log10 Factor Coverage Runtime Rejection

- area:
  - corpus generation / embedded equivalence candidate promotion
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `combined_additive_zero` x `log_exp_inverse`
- status:
  - `rejected as live coverage row`
- attempted row:
  - `log10_power_alias_factor_combined_zero`
  - `(10^(y*log10(x)) - x^y) + (p^2-q^2 - (p-q)*(p+q))`
- local lane:
  - exact candidate smoke with `scripts/engine_embedded_candidate_smoke.py`
  - focused slice: `combined_additive_zero` x `log_exp_inverse`
- local result:
  - candidate smoke passed:
    - `1/1` cases
    - `runner_elapsed=6.55ms`
  - focused slice passed:
    - `6/6` cases
    - `Elapsed: 111.44ms`
  - family balance would improve from 12 low families to 11 low families
- global result:
  - `make engine-scorecard` passed correctness:
    - `embedded_equivalence_context: 1387/1387`
    - `combined_additive_zero: 136`
    - `log_exp_inverse: 18`
  - but embedded runtime rose from the recent retained baseline:
    - `Elapsed: 5.59s -> 6.26s`
    - `+0.67s`, about `+12.0%`
- decision:
  - reverted the live corpus row
  - keep the expression as a discovered stress candidate, not as retained
    coverage
- best current explanation:
  - the base-10 logarithmic power alias is semantically correct, but this
    forward orientation under a factor companion appears to activate heavier
    contextual log/power matching than its coverage value justifies as one live
    row
- plausible combination later:
  - retry only after a cheaper base-10 alias gate or narrower contextual route
  - or promote a lower-cost representative for the same `log_exp_inverse`
    coverage gap

### 2026-04-27: `integrate_prep` Dirichlet Additive-Composition Timeout Discovery

- area:
  - generated discovery / embedded equivalence candidate smoke
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `combined_additive_zero` x `integrate_prep`
- status:
  - `resolved; minimal live row promoted`
- attempted row:
  - `integrate_prep_dirichlet_factor_collect_three_core_combined_zero`
  - `(sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))) + (p^2-q^2 - (p-q)*(p+q)) + (u*v + u*w - u*(v+w))`
- local lane:
  - `python3 scripts/engine_embedded_candidate_smoke.py --json --timeout-seconds 6 --row ...`
- local result:
  - attempted three-core candidate timed out before live promotion:
    - `status=timeout`
    - `wall_elapsed_seconds=6.004`
    - no runner summary was emitted before timeout
  - isolating probes:
    - root Dirichlet `combined_additive_zero` shape: pass,
      `runner_elapsed=3.59ms`
    - Dirichlet plus factor two-core shape: timeout at `10.004s`
    - Dirichlet plus collect two-core shape: timeout at `10.004s`
- resolution:
  - added a direct seven-term `combined_additive_zero` route for Dirichlet
    plus one independent exact-zero core
  - promoted smallest live row:
    - `integrate_prep_dirichlet_factor_combined_zero`
    - `(sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))) + (p^2-q^2 - (p-q)*(p+q))`
  - post-fix candidate smokes:
    - Dirichlet plus factor: `pass`, `runner_elapsed=9.44ms`
    - Dirichlet plus collect: `pass`, `runner_elapsed=10.08ms`
- decision:
  - promote the two-core row to live coverage
  - keep the original three-core row as a stress/discovery shape until a later
    cycle validates that exact three-core composition under full guardrails
- best current explanation:
  - the weakness was the additive-composition route, not Dirichlet itself
  - the arithmetic core could prove each piece, but the orchestrator rejected
    the seven-term composition and fell through to expensive generic paths
- plausible follow-up:
  - isolate whether the timeout is caused by additive-zero decomposition,
    trigonometric Dirichlet matching after flattening, or equivalence fallback
    search when the Dirichlet core is not the only additive term
  - retain a focused regression only after the failing route is understood;
    the generated three-core row is too expensive for live promotion today

### 2026-04-27: `radical_power` Passthrough Three-Core Discovery

- area:
  - generated discovery / embedded equivalence candidate smoke
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `combined_additive_zero` x `radical_power`
- status:
  - `resolved; live row promoted`
- attempted row:
  - `radical_power_passthrough_factor_collect_three_core_combined_zero`
  - `((sqrt(x^3)+a) - (abs(x)*sqrt(x)+a)) + (p^2-q^2 - (p-q)*(p+q)) + (u*v + u*w - u*(v+w))`
- local lane:
  - `python3 scripts/engine_embedded_candidate_smoke.py --json --timeout-seconds 6 --row ...`
- local result:
  - attempted three-core candidate failed before live promotion:
    - `status=fail`
    - `passed=0`
    - `failed=1`
    - `runner_elapsed=23.78ms`
  - isolating probes passed:
    - root radical passthrough shape: pass
    - radical passthrough plus factor two-core shape: pass
    - radical three-core shape without the `+ a` passthrough: pass
- decision:
  - do not promote the passthrough row to the live corpus
  - keep this as discovery pressure for a future `radical_power` isolation or
    runtime cycle
- best current explanation:
  - this is not evidence that `radical_power` is generally broken
  - the current signature points to a narrower composition gap: the radical
    passthrough core becomes fragile only when it is part of a three-core
    additive-zero composition
- plausible follow-up:
  - isolate whether the miss is caused by additive passthrough cancellation
    inside the radical core or by exact-zero decomposition after the first
    radical rewrite

### 2026-04-26: `collect` Common-Tail Three-Core Discovery

- area:
  - generated discovery / embedded equivalence candidate smoke
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `combined_additive_zero` x `collect`
- status:
  - `resolved`
- attempted row:
  - `collect_common_symbolic_factor_trigexpand_three_core_combined_zero`
  - `(x*y + x*z + w - (x*(y+z) + w)) + (p^2-q^2 - (p-q)*(p+q)) + (sin(2*t) - 2*sin(t)*cos(t))`
- local lane:
  - `python3 scripts/engine_embedded_candidate_smoke.py --json --timeout-seconds 6 --row ...`
- local result:
  - attempted three-core candidate failed before live promotion:
    - `status=fail`
    - `passed=0`
    - `failed=1`
    - `runner_elapsed=11.15ms`
  - isolating probes passed:
    - root collect shape with the `+ w` passthrough: pass
    - collect plus factor two-core shape: pass
    - collect plus trig two-core shape: pass
    - three-core shape without the `+ w` passthrough: pass
  - related probes failed:
    - reordered three-core shape with the `+ w` passthrough: fail
    - same collect/factor shape with trig reciprocal instead of trig expand: fail
- 2026-04-27 rejected runtime attempt:
  - tried a bounded three-core additive-zero partition route for generated
    compositions with one shared-passthrough collect core plus two small exact
    zero cores
  - local candidate became correct:
    - `status=pass`
    - `passed=1`
    - `failed=0`
    - `runner_elapsed=443.23ms`
  - global embedded guardrail rejected the promotion:
    - `embedded_equivalence_context`: `1377/1377`, `failed=0`
    - elapsed increased to `9.24s` for the guardrail run, far above the prior
      `5.27s` order of magnitude for only one extra row
  - runtime changes and live corpus promotion were not retained
- 2026-04-27 adjacent coverage retained:
  - promoted the minimal stable two-core representative instead of the failed
    three-core discovery:
    - `collect_common_symbolic_factor_with_tail_factor_combined_zero`
    - `(x*y + x*z + w - (x*(y + z) + w)) + (p^2-q^2 - (p-q)*(p+q))`
  - local smoke before promotion:
    - `status=pass`
    - `runner_elapsed=12.76ms`
  - this does not resolve the observe-only three-core row; it preserves the
    already-stable shared-tail collect shape in live coverage while keeping the
    hotter composition as discovery pressure
- decision:
  - do not promote the row to the live corpus
  - keep the expression as generated-discovery pressure for a future `collect`
    isolation or runtime cycle
- best current explanation:
  - the weakness was the additive-zero partition route, not `collect` itself
  - the engine could prove the collect passthrough core and the two other cores
    separately, but the live combined route did not allow a five-term
    passthrough core inside a three-core composition
  - the rejected fix widened the route enough to trigger expensive direct
    identity probing in embedded traffic
- plausible follow-up:
  - optimize a narrower partition recognizer before attempting promotion again;
    a local pass is not sufficient while the one-row smoke remains hundreds of
    milliseconds
  - isolate the reordered and trig-reciprocal variants before broadening the
    route again

### 2026-04-25: Nested-Fraction Three-Core Live Corpus Promotion

- area:
  - corpus generation / embedded equivalence live promotion
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
- status:
  - `rejected`
- local lane:
  - `python3 scripts/engine_embedded_candidate_smoke.py --json --row '((u*v)/(u+v) - 1/(1/u + 1/v)) + (sec(y) - 1/cos(y)) + (ln(a)+ln(b)-ln(a*b)),0,combined_additive_zero,nested_fraction_trigreciprocal_log_contract_three_core_combined_zero,nested_fraction,(u*v)/(u+v),1/(1/u + 1/v),collapse exact zero additive subexpressions'`
  - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --wrapper combined_additive_zero --family nested_fraction`
- local value:
  - smoke passed `1/1`
  - local `nested_fraction` slice passed `4/4`
  - projected coverage gain was `combined_additive_zero.multi_core_families: 2/23 -> 3/23`
- global result:
  - retained baseline before the attempt:
    - `embedded_equivalence_context`: `1332/1332`, `Elapsed: 3.87s`
  - attempted promotion:
    - first guardrail: `1333/1333`, `Elapsed: 5.06s`
    - rerun guardrail: `1333/1333`, `Elapsed: 5.10s`
  - pressure suite still passed:
    - `simplify_zero_mixed`: `450/450`
  - decision:
    - rejected as live corpus promotion
- best current explanation:
  - the three-core row combines nested-fraction cancellation, trig reciprocal,
    and symbolic log contraction in a way that is semantically valid but too
    expensive for a single live row
  - the smoke lane itself took `1.35s`, so the cost appears attached to this
    candidate shape rather than to scorecard noise
- plausible combination later:
  - keep this shape as a discovery/stress candidate, not live
  - find a cheaper `nested_fraction` multi-core representative without symbolic
    log contraction, or first add profiling that isolates which sub-block makes
    this exact composition hot

### 2026-04-21: Fast Solve-Prep Neg-Rewritten Default-Simplify Route

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `try_build_fast_solve_prep_exact_zero_scope_rewrite(...)`
- status:
  - `candidate for combination`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 CAS_PROFILE_ORCHESTRATOR_SHORTCUT_FILTER=rule.direct_identity.try.zero_scope_fast_solve_prep,rule.fast_solve_prep.,rule.solve_prep. cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --limit 480`
- attempted change:
  - promote an extra `neg_rewritten` + `default_simplify` comparison before
    falling back to `candidate_total_zero`
- local profiler signal:
  - `rule.fast_solve_prep.route.neg_rewritten_default_simplify_match`: `4` hits
  - `rule.fast_solve_prep.route.candidate_total_zero`: `8 -> 4`
  - the moved traffic stayed concentrated in the same shape bucket:
    - `add(add, div) || neg(mul)`
- global runtime result:
  - retained baseline on the same profiled embedded slice:
    - `Elapsed: 762.60ms`
    - `rule.direct_identity.try.zero_scope_fast_solve_prep`: `47.855ms`
    - `TOTAL`: `72.885ms`
  - attempted route enabled:
    - `Elapsed: 805.35ms`
    - `rule.direct_identity.try.zero_scope_fast_solve_prep`: `102.883ms`
    - `TOTAL`: `127.485ms`
  - decision:
    - rejected as runtime change
- best current explanation:
  - the extra whole-expression `default_simplify` comparison is broad enough
    that it costs more than the saved `candidate_total_zero` traffic on the
    shared `fast_solve_prep` path
  - the local win is real, but the call-site is still too broad for this check
    to pay for itself as a default branch
- follow-up combination check:
  - paired later with a retained `remaining_shifted_square` gate in
    `fast_solve_prep`
  - local movement improved:
    - `rule.fast_solve_prep.route.candidate_total_zero`: `8 -> 4`
    - `rule.fast_solve_prep.route.neg_rewritten_default_simplify_match`: `4`
  - but the global guarded slice still regressed:
    - retained gated baseline:
      - `Elapsed: 753.61ms`
      - `TOTAL: 71.558ms`
    - gated + neg-rewritten route:
      - `Elapsed: 824.95ms`
      - `TOTAL: 128.302ms`
  - conclusion:
    - `remaining_shifted_square` is not a narrow enough partner for this route
- plausible combination later:
  - pair it with a still narrower syntactic gate specific to the surviving
    `add(add, div) || neg(mul)` pocket, not merely “remaining has shifted
    square”
  - or reuse caller metadata so the extra negated-compare only runs on a
    subset already known to have exact cancellation plausibility

### 2026-04-18: Binary-Add Surface Plain-Algebraic Gate

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `try_build_exact_trig_phase_shift_zero_scope_rewrite(...)`
  - `Expr::Add(lhs, rhs)` branch feeding `rule.phase_shift.binary_add_match`
- status:
  - `candidate for combination`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus`
- attempted change:
  - reject `binary_add_match` when either side is a surface plain algebraic term
    (`a`, `-a`, `a*k`, `a/b`, etc.) before entering the full phase-shift matcher
- profiler signal:
  - `rule.phase_shift.binary_add_match`: `2336.640ms -> 1569.232ms`
  - `rule.phase_shift.binary_add_match.forward_try`: `822.854ms -> 789.083ms`
  - `rule.phase_shift.binary_add_match.reverse_try`: `1512.833ms -> 779.327ms`
  - filtered pair count stayed aligned with the previous broad trig-presence gate:
    - `516 -> 444`
- useful retained signal:
  - the surviving impossible traffic is concentrated in superficially plain
    algebraic terms paired against exact shifted trigs
  - the new surface classifier leaves a clearer split:
    - `surface_pair_shape.lhs_plain_algebraic`
    - `surface_pair_shape.rhs_plain_algebraic`
    - `surface_pair_shape.neither_plain_algebraic`
- global runtime result:
  - clean release rerun:
    - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
    - `1116/1116`
    - `0` fallos
    - `Elapsed: 7.58s`
  - retained baseline before the attempt:
    - `6.98s`
  - decision:
    - rejected as runtime change
- best current explanation:
  - even the cheap surface classifier still adds enough hot-path work at the
    binary-add call-site to lose globally
  - the fact that it filters almost exactly the same `72` pairs as the broader
    trig-presence gate suggests the real issue is not the screen cost alone, but
    that this caller is not large enough to justify any extra runtime filtering
- plausible combination later:
  - reuse term-shape metadata already produced by the surrounding additive view
    instead of recomputing per pair
  - or push the screen into an upstream phase where pair construction is cheaper
    and the same metadata is already materialized

### 2026-04-18: Binary-Add Trig-Pair Gate

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `try_build_exact_trig_phase_shift_zero_scope_rewrite(...)`
  - `Expr::Add(lhs, rhs)` branch feeding `rule.phase_shift.binary_add_match`
- status:
  - `candidate for combination`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus`
- attempted change:
  - require both binary-add terms to contain `sin/cos` before entering
    `try_find_trig_phase_shift_cancellation_match(...)`
- profiler signal:
  - `rule.phase_shift.binary_add_match.trig_pair_gate`: `516` attempts, `444` hits, `72` misses
  - `rule.phase_shift.binary_add_match`: `2336.640ms -> 1573.269ms`
  - `rule.phase_shift.binary_add_match.forward_try`: `822.854ms -> 788.003ms`
  - `rule.phase_shift.binary_add_match.reverse_try`: `1512.833ms -> 784.381ms`
- useful retained signal:
  - the filtered traffic is real and impossible:
    - `a  ||  sin(x + 1/4*pi) * sqrt(2)`
    - `-a  ||  -(sin(x + 1/4*pi) * sqrt(2))`
    - `a*k  ||  k * sin(x + 1/4*pi) * sqrt(2)`
  - so `binary_add_match` really is receiving non-trig terms that cannot
    participate in phase-shift cancellation
- global runtime result:
  - clean release rerun:
    - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
    - `1116/1116`
    - `0` fallos
    - `Elapsed: 7.09s`
  - retained baseline before the attempt:
    - `6.98s`
  - decision:
    - rejected as runtime change
- best current explanation:
  - the gate cuts real dead traffic, but the recursive builtin scan at this
    binary-add call-site still costs more than the saved matcher work in release
- plausible combination later:
  - reuse trig-shape metadata already known by the caller instead of rescanning
    both terms
  - or add a stronger O(1) syntactic screen for the specific non-trig families
    that dominate the samples

### 2026-04-17: Direct-Core Trig Presence Gate

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `try_build_direct_core_equivalence_rewrite(...)`
- status:
  - `candidate for combination`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus`
- attempted change:
  - gate the direct-core phase-shift matcher call so it only runs when both
    cores contain `sin/cos`
- local profiler signal:
  - `rule.phase_shift.direct_core_pair_candidate_gate`: `35529` attempts,
    `34879` hits, `650` misses
  - `rule.phase_shift.route.exact_try`: `73588` attempts vs previous `74380`
  - `rule.phase_shift.route.final_linear_compare_try`: `72680` attempts vs
    previous `74364`
- interpretation:
  - the gate was correct but weak
  - only a small fraction of direct-core traffic was actually algebraic enough
    to be filtered by “both sides must contain trig”
  - the true hotspots remained overwhelmingly large after the gate
- global runtime result:
  - clean release rerun:
    - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
    - `1116/1116`
    - `0` fallos
    - `Elapsed: 28.63s`
  - decision:
    - rejected as runtime change
- best current explanation:
  - this call-site does contribute some impossible traffic, but not enough to
    justify even a cheap builtin scan
  - the dominant residual cost still comes from genuinely trig-shaped but
    ultimately non-matching traffic deeper in `exact_try` and
    `final_linear_compare_try`
- plausible combination later:
  - reuse caller metadata stronger than bare trig presence
  - or split the late `exact_try` traffic by exact-shift plausibility without
    adding another generic recursive scan at the direct-core entry

### 2026-04-17: Exact-Scope Pair Candidate Gate

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `try_build_exact_trig_phase_shift_zero_scope_rewrite(...)`
- status:
  - `candidate for combination`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family trig_contract --limit 24`
- attempted change:
  - add a call-site-only plausibility gate before `rule.phase_shift.exact_scope_pair_match`
  - require:
    - both sides contain `sin/cos`
    - and at least one side contain `pi` or inverse-trig shift markers
- local result:
  - no meaningful gain on the narrow slice
  - `Elapsed`: stayed at `1.28s`
  - `rule.phase_shift.exact_scope_pair_candidate_gate`: `24/24` hits
  - so the local `trig_contract` lane was already mostly inside the real family
- global profiler result:
  - `rule.phase_shift.exact_scope_pair_match`: `984 -> 186`
  - `rule.phase_shift.exact_scope_pair_candidate_gate`: `984 attempts`, `186` hits
  - but the true global hotspots barely moved:
    - `rule.phase_shift.route.exact_try`: `74380 -> 73588`
    - `rule.phase_shift.route.final_linear_compare_try`: `74364 -> 73572`
- global runtime result:
  - clean release rerun:
    - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
    - `1116/1116`
    - `0` fallos
    - `Elapsed: 29.70s`
  - decision:
    - rejected as runtime change
- best current explanation:
  - the gate successfully cuts dead traffic inside `exact_scope_pair_match`
  - but that traffic is not the dominant contributor to the retained global cost
  - the additional recursive scans for `sin/cos` and `pi/atan` at this call-site
    cost more than the saved work in the true hot path
- useful retained signal:
  - `exact_scope_pair_match` does contain plenty of impossible algebraic pairs
  - but reducing that subroute alone is not enough; the real cost still lives in
    later broad routes such as `exact_try` and `final_linear_compare_try`
- plausible combination later:
  - reuse metadata already computed while forming `focus_expr` / `remaining_expr`
    instead of rescanning both expressions
  - or move the route split even later, closer to the actual `exact_try` caller,
    where the semantic information is stronger and the gate can be cheaper

### 2026-04-17: Pair-Candidate Gate At Phase-Shift Matcher Entry

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `try_find_trig_phase_shift_cancellation_match(...)`
- status:
  - `candidate for combination`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family trig_contract --limit 24`
- attempted change:
  - add a cheap gate at matcher entry requiring:
    - both sides to contain `sin/cos`
    - and at least one side to contain a phase-shift marker (`pi` or `atan`)
- local win:
  - `Elapsed`: `1.34s -> 1.28s`
  - profiler total: `934.146ms -> 901.985ms`
  - `rule.phase_shift.route.linear_focus_try`: `160 attempts -> 24`
  - `rule.phase_shift.route.general_try`: `140 attempts -> 4`
- global profiler result:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus`
  - `rule.phase_shift.route.exact_try`: `74380 attempts -> 1978`
  - `rule.phase_shift.route.final_linear_compare_try`: `74364 attempts -> 1962`
  - new gate cut:
    - `rule.phase_shift.route.pair_candidate_gate`: `72058` attempts, `2050` hits
- global runtime result:
  - clean release rerun:
    - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
    - `1116/1116`
    - `0` fallos
    - `Elapsed: 27.99s`
  - baseline retained before the attempt:
    - `26.79s`
  - decision:
    - rejected as runtime change
- best current explanation:
  - the gate removes a huge amount of profiler-visible impossible traffic,
    especially before `exact_try` and `final_linear_compare_try`
  - but the recursive builtin and `pi/atan` scans at matcher entry are still too
    expensive on the normal hot path when the profiler is off
  - in other words, this was the right *semantic filter* at the wrong *cost
    location*
- useful retained signal:
  - the bad traffic is real
  - the profitable direction is still to cut phase-shift matching before
    `exact_try`
  - but the cut must reuse information already known at the caller instead of
    rescanning both expressions at matcher entry
- plausible combination later:
  - call-site-specific gating in `binary_add_match` or `direct_core_equivalence`
    using already available structure
  - cached term metadata for:
    - contains `sin/cos`
    - contains `pi`
    - contains inverse-trig shift markers
  - route splitting before `try_find_trig_phase_shift_cancellation_match(...)`
    so the matcher does not have to rediscover phase-shift plausibility itself

### 2026-04-17: Exact Target Raw Supported-Arg Fast Path

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `extract_exact_phase_shift_term_data_for_cancellation(...)`
- status:
  - `candidate for combination`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family trig_contract --limit 24`
- attempted change:
  - after the existing `pi` gate, resolve these exact raw shapes directly before
    falling back to the general supported-arg extractor:
    - `pi / 4 + x`
    - `x - pi / 4`
    - `pi / 3 + x`
    - `pi / 6 + x`
- local win:
  - `rule.phase_shift.exact_scope_rewrite`: `44.411ms -> 9.546ms`
  - `rule.phase_shift.exact_scope_pair_match`: `43.971ms -> 9.130ms`
  - `root.div.03.shifted_quotient_nested_zero_core`: `16.221ms -> 7.156ms`
- global result:
  - baseline scorecard:
    - `/tmp/engine_improvement_scorecard_embedded_after_linear_focus_exact_target_pi_gated_fastpath.json`
  - scorecard result:
    - `26.88s -> 29.41s`
  - decision:
    - rejected as runtime change
- best current explanation:
  - the direct raw fast path is excellent for the narrow `trig_contract` slice,
    but too expensive when installed inside a shared extraction path that is hit
    by much broader traffic in `embedded`
- useful retained signal:
  - raw exact shapes are common in the hot slice
  - the real retained hotspot still sits in the general path:
    - `rule.phase_shift.exact_target_extract.shifted_arg_match`
    - `rule.phase_shift.supported_arg.normalized_simplify_fallback`
- plausible combination later:
  - do **not** revive this as a helper-global fast path
  - instead, combine it with one of:
    - call-site-only reuse in `exact_scope_pair_match`
    - cached exact target signature
    - a narrower route that proves the caller is already in the exact phase-shift
      family before paying the raw matcher

### 2026-04-17: Exact Target Raw Third/Sixth Fast Path

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `extract_exact_phase_shift_term_data_for_cancellation(...)`
- status:
  - `candidate for combination`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family trig_contract --limit 24`
- attempted change:
  - after the existing `pi` gate, and only when `has_sqrt_two == false`,
    resolve raw exact `third/sixth` targets directly before calling the general
    supported-arg extractor
  - intended covered shapes:
    - `pi / 3 + x`
    - `pi / 6 + x`
    - `x - pi / 6`
- local win:
  - `rule.phase_shift.exact_scope_rewrite`: `43.741ms -> 9.855ms`
  - `rule.phase_shift.exact_scope_pair_match`: `43.272ms -> 9.433ms`
  - `root.div.03.shifted_quotient_nested_zero_core`: `15.749ms -> 7.155ms`
  - `rule.phase_shift.exact_target_extract.shifted_arg_match`: `34.237ms -> 0.118ms`
- global result:
  - baseline scorecard:
    - `/tmp/engine_improvement_scorecard_embedded_after_linear_focus_exact_target_pi_gated_fastpath.json`
  - scorecard result:
    - `26.88s -> 28.83s`
  - decision:
    - rejected as runtime change
- best current explanation:
  - even scoped to `third/sixth`, the raw fast path still adds enough shared
    extraction traffic to regress `embedded`, despite being excellent in the
    narrow `trig_contract` slice
- useful retained signal:
  - the expensive misses in the retained path are genuinely concentrated in
    `quarter -> third/sixth` fallback churn
  - the profitable part is the *routing information*, not the raw fast path
    itself
- plausible combination later:
  - pair this idea with exact-target signature reuse that is already computed by
    the caller, instead of re-extracting inside `extract_exact_phase_shift_term_data_for_cancellation(...)`
  - or narrow it even further to a route that has already proven it is in the
    `exact_scope_pair_match -> linear_to_shifted` family before attempting raw
    extraction

### 2026-04-17: Exact Target Kind-Hinted Extractor

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `find_linear_focus_phase_shift_cancellation_match(...)`
  - `extract_exact_phase_shift_term_data_for_cancellation(...)`
- status:
  - `candidate for combination`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family trig_contract --limit 24`
- attempted change:
  - reuse the already known `exact_data.kind` from `linear_focus`
  - force exact-target extraction to try only the single denominator compatible
    with that expected `Quarter`/`Third`/`Sixth` family
- local win:
  - `rule.phase_shift.exact_scope_rewrite`: `43.741ms -> 10.903ms`
  - `rule.phase_shift.exact_scope_pair_match`: `43.272ms -> 10.452ms`
  - `root.div.03.shifted_quotient_nested_zero_core`: `15.749ms -> 8.024ms`
- global result:
  - baseline scorecard:
    - `/tmp/engine_improvement_scorecard_embedded_after_linear_focus_exact_target_pi_gated_fastpath.json`
  - scorecard result:
    - `26.88s -> 29.64s`
  - decision:
    - rejected as runtime change
- best current explanation:
  - even though the hint is semantically correct for the local
    `linear_to_shifted` family, specializing the exact-target extractor in place
    still regresses broader `embedded` traffic
  - that implies the global cost is not only “too many denominators tried”; it
    is also *where* the extraction happens and how often the route is reached
- useful retained signal:
  - denominator pruning alone is not enough
  - the next viable combination still points to one-shot target-signature reuse,
    not to further specialization inside the extractor call itself
- plausible combination later:
  - compute exact target signature once in `exact_scope_pair_match` and reuse it
    across `linear_focus` and later compare stages
  - or add a cache keyed by the target term before it enters the extractor

### 2026-04-18: Target-Exact Cache Inside Shared Matcher

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `try_find_trig_phase_shift_cancellation_match(...)`
  - `find_linear_focus_phase_shift_cancellation_match(...)`
  - `final_linear_compare`
- status:
  - `candidate for combination`
- attempted change:
  - cache `extract_exact_phase_shift_term_data_for_cancellation(target_expr)` once
    per matcher call and reuse it across:
    - `linear_focus.exact_target_match`
    - `exact_try.target_exact_extract`
    - `final_linear_compare.target_exact_linear`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family trig_contract --limit 24`
  - result:
    - `Elapsed: 1.21s -> 1.23s`
    - no meaningful local win
- global result:
  - release rerun:
    - `6.49s -> 6.70s`
    - `1116/1116`, `0` fallos
  - decision:
    - rejected as runtime change
- best current explanation:
  - even scoped to the target side only, the extra cache plumbing inside the
    shared matcher adds more overhead than the repeated target extraction it
    removes
  - this suggests the profitable reuse point is not “inside the matcher”, but
    one layer above, where caller metadata can be passed in without reopening
    more mutable state and branching in the hot path
- plausible combination later:
  - reuse an already materialized exact-target signature from the caller of
    `exact_scope_pair_match`, instead of introducing cache logic inside
    `try_find_trig_phase_shift_cancellation_match(...)`

### 2026-04-21: One-Pass Quadratic Term Analyzer

- area:
  - [quadratic_coeffs.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver_core/src/quadratic_coeffs.rs)
  - `extract_quadratic_coefficients(...)`
  - `analyze_term(...)`
- status:
  - `rejected as runtime change`
- attempted change:
  - replace the current `contains_var`-driven branching in `analyze_term(...)`
    with a one-pass recursive analyzer for `Mul`/`Div`/`Neg`
  - keep constant subtrees opaque and only rebuild coefficients after recursive
    degree analysis
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 CAS_PROFILE_ORCHESTRATOR_SHORTCUT_FILTER=rule.direct_identity.try.zero_scope_fast_solve_prep,rule.fast_solve_prep.,rule.solve_prep. cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family solve_prep`
  - result:
    - retained baseline before attempt:
      - `Elapsed: 37.85ms`
      - `TOTAL: 42.830ms`
      - `rule.solve_prep.build_candidate.extract_coeffs: 10.264ms`
    - attempted change, hot rerun:
      - `Elapsed: 41.36ms`
      - `TOTAL: 46.217ms`
      - `rule.solve_prep.build_candidate.extract_coeffs: 10.755ms`
- global result:
  - guardrail lane:
    - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 CAS_PROFILE_ORCHESTRATOR_SHORTCUT_FILTER=rule.direct_identity.try.zero_scope_fast_solve_prep,rule.fast_solve_prep.,rule.solve_prep. cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --limit 480`
  - retained baseline before attempt:
    - `Elapsed: 706.67ms`
    - `TOTAL: 38.352ms`
    - `rule.solve_prep.build_candidate.extract_coeffs: 9.063ms`
  - attempted change, hot rerun:
    - `Elapsed: 716.12ms`
    - `TOTAL: 41.026ms`
    - `rule.solve_prep.build_candidate.extract_coeffs: 9.715ms`
  - decision:
    - reverted
- best current explanation:
  - the repeated `contains_var(...)` scans are not the dominant cost in the
    retained `solve_prep` pocket
  - replacing them with a more generic recursive analyzer adds its own control
    flow and coefficient rebuilding overhead before any real candidate is
    formed
- useful retained signal:
  - `extract_coeffs` should not be attacked with a broad analyzer rewrite in
    shared core
  - the next viable move should be narrower:
    - route-specific prefilters before quadratic extraction
    - or observability inside coefficient extraction to separate
      coefficient-shape cost from target-variable scans

### 2026-04-21: Deferred `solve_prep` `c` Simplification

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `extract_profiled_solve_prep_nonzero_quadratic_coefficients(...)`
- status:
  - `rejected as runtime change`
- attempted change:
  - stop simplifying the extracted constant term `c` inside `solve_prep`
  - rely on later `tail` / final-candidate simplification to normalize it
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 CAS_PROFILE_ORCHESTRATOR_SHORTCUT_FILTER=rule.direct_identity.try.zero_scope_fast_solve_prep,rule.fast_solve_prep.,rule.solve_prep. cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family solve_prep`
  - retained baseline before attempt:
    - `Elapsed: 47.44ms`
    - `TOTAL: 56.502ms`
    - `rule.solve_prep.extract.simplify_c: 4.958ms`
  - attempted change:
    - `Elapsed: 57.03ms`
    - `TOTAL: 113.983ms`
    - `rule.solve_prep.extract.defer_simplify_c: 0.001ms`
    - but `rule.solve_prep.build_candidate.extract_coeffs: 10.973ms -> 16.229ms`
    - and `rule.fast_solve_prep.gate.focus_remaining_var_mismatch: 16 -> 56`
- global result:
  - guardrail lane:
    - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 CAS_PROFILE_ORCHESTRATOR_SHORTCUT_FILTER=rule.direct_identity.try.zero_scope_fast_solve_prep,rule.fast_solve_prep.,rule.solve_prep. cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --limit 480`
  - retained baseline before attempt:
    - `Elapsed: 723.97ms`
    - `rule.solve_prep.build_candidate.extract_coeffs: 9.735ms`
  - attempted change:
    - `Elapsed: 761.06ms`
    - `rule.solve_prep.build_candidate.extract_coeffs: 15.329ms`
    - `rule.fast_solve_prep.gate.focus_remaining_var_mismatch: 16 -> 56`
  - decision:
    - reverted
- best current explanation:
  - `simplify_c` is expensive, but removing it broadens the upstream shape
    space enough to reopen a larger dead-traffic pocket than the cost it saves
  - the saved work is real; the placement is wrong
- plausible combination later:
  - defer `c` normalization only after a narrower route-specific prefilter
  - or only for one build-route once the `focus_remaining_var_mismatch` pocket
    is already closed

### 2026-04-21: Narrow `pos_generic` Negative-Linear `c` Deferral

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `extract_profiled_solve_prep_nonzero_quadratic_coefficients(...)`
- status:
  - `rejected as runtime change`
- attempted change:
  - defer `c` only for the `pos_generic` pocket measured as:
    - `a.shape = atom`
    - `b.shape = neg_atom`
    - `c.shape = addsub_with_div`
    - `focus_no_shifted_square`
  - this was narrow enough to make the focused regression
    `a*x^2 - b*x + (c + d - d)` pass in fast `solve_prep`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 CAS_PROFILE_ORCHESTRATOR_SHORTCUT_FILTER=rule.solve_prep.extract.simplify_a,rule.solve_prep.extract.simplify_a.shape.,rule.solve_prep.extract.simplify_b,rule.solve_prep.extract.simplify_b.shape.,rule.solve_prep.extract.simplify_c,rule.solve_prep.extract.simplify_c.build.,rule.solve_prep.extract.simplify_c.shape.,rule.solve_prep.extract.pos_generic.,rule.solve_prep.extract.defer_simplify_c.,rule.direct_identity.try.zero_scope_fast_solve_prep,rule.fast_solve_prep.,rule.solve_prep. cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family solve_prep`
  - retained baseline before attempt:
    - `Elapsed: 43.27ms`
    - `TOTAL: 52.250ms`
    - `rule.fast_solve_prep.try.collect_rewrites: 16`
    - `rule.solve_prep.build_candidate.extract_coeffs: 16`
  - attempted change:
    - `Elapsed: 47.28ms`
    - `TOTAL: 57.535ms`
    - `rule.fast_solve_prep.try.collect_rewrites: 16 -> 72`
    - `rule.solve_prep.build_candidate.extract_coeffs: 16 -> 104`
    - `rule.solve_prep.extract.defer_simplify_c.pos_generic_scale: 4`
    - sample dead traffic reopened as `:: b`
- decision:
  - reverted
- best current explanation:
  - the pocket is real, but skipping `c` normalization there is still early enough
    to broaden the rewrite search around the raw focus
  - the next viable move is not another `defer_c` variant
  - it should be either:
    - a dedicated negative-linear symbolic route that keeps `c` normalized
    - or a prefilter earlier than coefficient extraction

### 2026-04-25: `nested_fraction` Three-Core Coverage Candidate

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `combined_additive_zero`
- status:
  - `rejected as live coverage row`
- attempted row:
  - `nested_fraction_trigreciprocal_factor_three_core_combined_zero`
  - `(1/(1/u + 1/v) - (u*v)/(u+v)) + (sec(y) - 1/cos(y)) + (a^2-b^2 - (a-b)*(a+b))`
- local result:
  - candidate smoke passed:
    - `1/1` cases
    - `runner_elapsed=1.34s`
    - `wall=1.664s`
  - focused slice passed:
    - `combined_additive_zero` x `nested_fraction`
    - `4/4` cases
    - `Elapsed: 1.30s`
  - `make engine-fast` passed
- global result:
  - `make engine-scorecard` passed correctness:
    - `embedded_equivalence_context: 1344/1344`
    - `combined_additive_zero: 102`
    - `nested_fraction: 4`
    - `multi_core_family_count: 14`
  - but the embedded lane cost rose from the retained baseline:
    - `Elapsed: 4.89s -> 6.13s`
- decision:
  - reverted the live corpus row
  - keep the expression as a discovered stress candidate, not as retained
    coverage
- best current explanation:
  - the candidate proves the missing structural axis, but the nested fraction
    core is expensive enough that composing it with two additional exact-zero
    cores is a poor live guardrail tradeoff
- plausible combination later:
  - find a cheaper `nested_fraction` representative for the same three-core
    axis
  - or improve nested fraction runtime before promoting this shape

### 2026-04-25: `fraction_decompose` Three-Core Coverage Candidate

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `combined_additive_zero`
- status:
  - `rejected before live promotion`
- attempted row:
  - `fraction_decompose_trigreciprocal_factor_three_core_combined_zero`
  - `((a*x+b)/(x+c) - (a + (b-a*c)/(x+c))) + (sec(y) - 1/cos(y)) + (p^2-q^2 - (p-q)*(p+q))`
- local lane:
  - `python3 scripts/engine_embedded_candidate_smoke.py --timeout-seconds 6 --expect pass --row ...`
- local result:
  - timed out before promotion:
    - `status=timeout`
    - `wall=6.005s`
- decision:
  - did not add the row to the live corpus
  - keep this exact shape as stress/discovery only
- best current explanation:
  - the `fraction_decompose` core is stable in existing wrappers, but this
    three-core composition is too hot for live guardrail promotion
  - the low family count should not be closed by adding a broad
    `trig reciprocal + factor` companion around this core
- plausible combination later:
  - look for a cheaper `fraction_decompose` multi-core representative
  - or improve the fraction-decompose residual route before attempting this
    coverage axis again

### 2026-04-27: Exact Trig Two-Term Zero-Scope Runtime Candidate

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `simplify_zero_mixed` pressure `sum`
- status:
  - `retained`
- change:
  - added a normalized two-term path for
    `try_build_exact_trig_equivalence_zero_scope_rewrite`
  - two-term exact trig scopes compare `lhs`/`rhs` directly and avoid the
    generic subset loop
  - three-or-more-term exact trig scopes still use the existing generic path
- local pressure result:
  - focused filtered baseline:
    - `rule.direct_identity.try.zero_scope_exact_trig_equivalence`:
      `11489` attempts, `11040.624ms`, `960.97us` avg
  - after change:
    - `rule.direct_identity.try.zero_scope_exact_trig_equivalence`:
      `5221` attempts, `2083.946ms`, `399.15us` avg
  - `sum@0+100` simplify avg:
    - `68.36ms -> 54.84ms` in the filtered profile
    - `55.48ms` in the full local profile
- guardrails:
  - targeted arithmetic tests:
    - `exact_trig_equivalence_zero_scope`: `8/8`
    - `maybe_exact_trig_equivalence_zero_scope_candidate`: `5/5`
    - `direct_small_zero_additive_combination_shortcut`: `9/9`
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `13.37s -> 10.56s`
    - `sum` simplify avg `65.77ms -> 51.67ms`
- decision:
  - retained as runtime improvement
- next:
  - the new pressure head is phase-shift expansion/comparison on trig-heavy
    two-term scopes; investigate a narrower gate or a cheaper exact path there

### 2026-04-27: Phase-Shift Single-Fragment Runtime Gate

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `simplify_zero_mixed` pressure `sum`
- status:
  - `retained`
- change:
  - reject binary phase-shift bridge pairs where one side is a single plain
    trig term and the other side is an exact/general shifted trig term
  - keep full additive linear-combination vs shifted pairs active
- local pressure result:
  - focused filtered baseline:
    - `rule.direct_identity.try.expand_trig_phase_shift`:
      `306` attempts, `2099.771ms`, `6862us` avg
    - `rule.phase_shift.route.shifted_try`:
      `1326` attempts, `1748.453ms`
    - `rule.phase_shift.binary_add_match`:
      `333` attempts, `1193.440ms`
  - after change:
    - `rule.direct_identity.try.expand_trig_phase_shift`:
      `102` attempts, `330.292ms`, `3238us` avg
    - `rule.phase_shift.route.shifted_try`:
      `252` attempts, `258.625ms`
    - `rule.phase_shift.binary_add_match.productive_term_family_gate`:
      `111` attempts, `237.065ms`
  - `sum@0+100` simplify avg:
    - `52.89ms -> 48.82ms` in the filtered profile
    - `50.37ms` in the full local profile
- guardrails:
  - targeted phase-shift tests:
    - `expand_trig_phase_shift_rule`: `5/5`
    - `trig_phase_shift_cancellation_match`: `6/6`
    - `collapse_exact_zero_three_term_subset_rule_matches_phase_shift`: `2/2`
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `10.56s -> 9.16s`
    - `sum` simplify avg `51.67ms -> 44.75ms`
- decision:
  - retained as runtime improvement
- next:
  - remaining pressure head is the double-angle embedded factor family around
    `2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))`; investigate a
    narrower direct route or cached target simplification for that family

### 2026-04-27: Embedded Double-Angle Factor Runtime Narrowing

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `simplify_zero_mixed` pressure `sum`
- status:
  - `retained`
- change:
  - restricted `zero_scope_embedded_double_angle_factor` to embedded
    `sin(2*x)` factors, because embedded `cos(2*x)` mixed-polynomial cases
    are already covered by the dedicated `cos_variant` /
    `mixed_double_angle_poly` routes
  - added an arity/default-fallback guard for the exact
    `cos(2*x)` polynomial zero-scope path, while preserving factored
    remaining forms with a top-level additive-factor check
- local pressure result:
  - baseline `sum@0+50`:
    - `50/50`, simplify avg `49.85ms`
    - `rule.direct_identity.try.zero_scope_embedded_double_angle_factor`:
      `314` attempts, `2` hits, `650.356ms`
  - after change `sum@0+50`:
    - `50/50`, simplify avg `47.12ms`
    - slow double-angle mixed cases around `#33/#189`: steady-state
      `~529-562ms` instead of the previous `~672ms` local baseline
    - `rule.direct_identity.try.zero_scope_embedded_double_angle_factor`:
      `2` attempts, `2` hits, `5.740ms`
  - pressure scorecard:
    - `simplify_zero_mixed`: `450/450`
    - elapsed `8.11s`
    - `sum` simplify avg `39.56ms`
- guardrails:
  - targeted arithmetic tests:
    - `embedded_double_angle`: `8/8`
    - `mixed_double_angle`: `7/7`
    - `exact_zero_cos_double_angle_polynomial_matches_factored_remaining_regression`: `1/1`
  - `cargo fmt -- --check`: passed
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
- rejected local candidates:
  - early small-core mixed double-angle polynomial route improved targeted
    coverage but regressed the pressure profile, so it was not promoted
  - arithmetic small log/double-angle partition probes and additional
    orchestrator root/base partition probes did not move the hot eval route
    and were reverted
- decision:
  - retained as runtime improvement
- next:
  - remaining pressure heads are exact trig equivalence and phase-shift routes
    for product-to-sum families such as `2*cos(x)*cos(y) - cos(x+y) - cos(x-y)`;
    investigate similarly narrow family gates there

### 2026-04-27: Exact-Trig Nested Default Fallback Runtime Gate

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `simplify_zero_mixed` pressure `sum`
- status:
  - `retained`
- change:
  - disabled default-simplify fallback comparisons inside
    `try_build_exact_trig_equivalence_zero_scope_rewrite` when already running
    under `run_default_simplify`
  - kept exact/structural cancellation checks active, and kept outermost
    default-simplify fallback behavior unchanged
  - applied the same nested fallback guard to the two-term exact trig zero-scope
    helper
- local pressure result:
  - `case#89` product-to-sum/log composition:
    - steady-state simplify `~267.75ms -> ~181.62ms`
    - `rule.direct_identity.try.zero_scope_exact_trig_equivalence`:
      `3000` attempts, `0` hits, `~1436.6ms -> ~722.6ms`
  - `sum@0+100` profiled slice:
    - `100/100`
    - simplify avg `41.50ms -> 38.17ms`
    - exact-trig label `9505` attempts / `3992.559ms` -> `5221` attempts /
      `908.348ms`
  - pressure scorecard:
    - `simplify_zero_mixed`: `450/450`
    - elapsed `8.11s -> 7.25s`
    - `sum` simplify avg `39.56ms -> 35.24ms`
- guardrails:
  - targeted arithmetic tests:
    - `exact_trig_equivalence_zero_scope`: `8/8`
    - `maybe_exact_trig_equivalence_zero_scope_candidate`: `5/5`
  - `cargo fmt -- --check`: passed
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
- rejected local candidates:
  - generic direct product-to-sum zero-scope route passed unit tests but did not
    change the hot eval route and worsened `case#89`
  - structural product-to-sum candidate rejection also left the exact-trig
    no-match traffic unchanged
- decision:
  - retained as runtime improvement
- next:
  - after exact-trig fallback cost drops, the next pressure head shifts back to
    phase-shift expansion/comparison; investigate a similarly nested-safe
    fallback gate for phase-shift generated candidate comparisons

### 2026-04-27: Phase-Shift Binary Fragment Pre-Reject

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `simplify_zero_mixed` pressure `sum`
- status:
  - `retained`
- change:
  - added a cheap surface pre-reject for binary phase-shift fragments where one
    side is a plain trig term and the other side carries a phase-shift signal
  - used that pre-reject before direct two-term phase-shift identity and
    expansion probes, avoiding expensive generated-candidate routes for
    fragments that cannot cancel alone
  - added a direct fast zero-scope path for the full general arctan phase-shift
    triple so the complete `a*sin(x)+b*cos(x)-r*sin(x+atan(b/a))` shape keeps
    its didactic two-step rewrite when reached as a whole
- local pressure result:
  - `case#49` general phase-shift/log composition:
    - steady-state simplify `~218.67ms -> ~17.30ms`
    - initial single run simplify `237.44ms -> 28.13ms`
    - hot `two_term_phase_shift_identity`, `expand_trig_phase_shift`, and
      `shifted_try` probes dropped out of the case top profile after the
      pre-reject
  - profiled `sum@0+100` slice:
    - `100/100`
    - simplify avg `36.24ms -> 32.45ms`
  - pressure scorecard:
    - `simplify_zero_mixed`: `450/450`
    - elapsed `7.25s -> 6.45s`
    - `sum` simplify avg `35.24ms -> 31.23ms`
    - `sum@0+100` simplify avg `34.47ms -> 30.45ms`
    - `sum@700+100` simplify avg `36.00ms -> 32.00ms`
- guardrails:
  - targeted arithmetic tests:
    - `phase_shift`: `54/54`
  - `cargo fmt -- --check`: passed
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
  - `git diff --check`: passed
- rejected local candidates:
  - a nested `default_simplify` fallback gate inside supported phase-shift
    argument normalization passed focused tests but worsened `sum@0+100`
    (`36.24ms -> 38.34ms`) and was reverted
  - the general arctan triple fast path alone was correct but did not move the
    mixed pressure profile until paired with the binary fragment pre-reject
- decision:
  - retained as runtime improvement
- next:
  - remaining pressure heads are double-angle mixed fragments such as
    `2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))`; investigate a narrow
    early gate or fast zero-scope route for those fragments next

### 2026-04-27: Surface Trig Power Numeric Fallback Reject

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `simplify_zero_mixed` pressure `sum`
- status:
  - `retained`
- change:
  - added a direct-core pre-reject before expensive `default_simplify` for
    scaled symbolic `sin`/`cos` power monomials against numeric atoms, e.g.
    `4*cos(x)^2` vs `1`
  - extended the existing surface trig power-gap reject to scaled plain-vs-power
    pairs, e.g. `2*sin(x)` vs `3*sin(x)^2`
  - preserved special-angle traffic by requiring symbolic trig arguments without
    `pi` constants or nested function calls
- local pressure result:
  - `case#33` double-angle/log composition:
    - steady-state simplify `~578.33ms -> ~513.19ms`
    - profiled steady-state simplify `~577.06ms -> ~523.45ms`
  - `case#189` double-angle/nested-fraction composition:
    - steady-state simplify `~580.10ms -> ~517.39ms`
  - profiled `case#33` route movement:
    - `rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.other`:
      `1600` attempts / `~332.304ms -> 32` attempts / `~13.275ms`
    - new reject
      `rule.direct_core_equivalence.route.default_simplify_surface_trig_power_numeric_atom_reject`:
      `1568` hits
  - profiled `sum@0+100` slice:
    - `100/100`
    - simplify avg `30.45ms -> 29.80ms`
  - pressure scorecard:
    - `simplify_zero_mixed`: `450/450`
    - elapsed `6.45s -> 6.08s`
    - `sum` simplify avg `31.23ms -> 29.37ms`
    - `sum@0+100` simplify avg `30.45ms -> 28.68ms`
    - `sum@700+100` simplify avg `32.00ms -> 30.07ms`
- guardrails:
  - targeted arithmetic tests:
    - `reject_scaled_surface_trig_power_vs_numeric_atom_before_default_simplify`: `2/2`
    - `reject_plain_surface_trig_power_gap_before_default_simplify_matches_scaled_gap`: `1/1`
  - `cargo fmt -- --check`: passed
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
- rejected local candidates:
  - adding pure log cores to `small_direct_zero_core` and the two-core
    composition gate passed focused tests for
    `ln(x^3)+ln(y^2)-ln(x^3*y^2)`, but did not move the live route and worsened
    `case#33` steady-state (`~578ms -> ~604-618ms`); reverted
  - extending the same composition gate to the concrete nested-fraction core in
    `case#189` failed the focused rewrite test before promotion; reverted
- decision:
  - retained as runtime improvement
- next:
  - the remaining hot heads are still full double-angle mixed compositions; a
    future cycle should investigate a direct exact-zero route for the complete
    `2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))` composition rather than
    broadening small-core partition gates

### 2026-04-27: Compact Direct Small-Zero Pair Before Didactic Chunking

- area:
  - [orchestrator.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs)
  - `simplify_zero_mixed` pressure `sum`
- status:
  - `retained`
- change:
  - added a `StepsMode::Compact`-only early route for
    `try_standard_direct_small_zero_pair_shortcut` before the targeted
    additive-combination didactic path
  - this preserves the richer chunked explanation path for full step collection,
    but avoids building expensive chunk-pair substeps on canonical eval calls
    where user-visible steps are off and `Compact` is only used internally
  - added a cached-profile compact pipeline regression for the
    `log_product + mixed double-angle` sum to lock the root-shortcut route
- local pressure result:
  - `case#33` double-angle/log composition:
    - steady-state simplify `~445.53ms -> ~0.09ms`
    - single profiled simplify `471.59ms -> 0.85ms`
    - profile now shows
      `root.addsub.00.direct_small_zero_pair.compact_first`: `4/4` hits
  - `case#3001` ln-abs/double-angle composition:
    - steady-state simplify `~477.53ms -> ~0.09ms`
    - single profiled simplify `~497.92ms -> 0.99ms`
  - pressure scorecard:
    - `simplify_zero_mixed`: `450/450`
    - elapsed `6.08s -> 0.24s`
    - `sum` simplify avg `29.37ms -> 0.26ms`
    - `sum@0+100` simplify avg `28.68ms -> 0.12ms`
    - `sum@700+100` simplify avg `30.07ms -> 0.40ms`
- guardrails:
  - targeted orchestrator tests:
    - `trig_mixed_double_angle_sum_regression`: `2/2`
  - `cargo fmt -- --check`: passed
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
- decision:
  - retained as runtime improvement
- next:
  - the new pressure head is `case#3145`:
    `(tan(x)*cot(x) - 1) + (sin(x)^2 - (1 - cos(2*x))/2)`, around
    `30.7ms` steady-state; investigate a compact/direct pair route for
    tan-cot plus half-angle square without weakening the full didactic path

### 2026-04-27: Compact Tan-Cot Half-Angle Pair Shortcut

- area:
  - [orchestrator.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs)
  - `simplify_zero_mixed` pressure `sum@700+100`
- status:
  - `retained`
- investment:
  - `investment_class`: runtime
  - `primary_dimension`: runtime
  - `secondary_dimension`: observability/robustness
  - `cohesion_scope`: compact-only root add/sub small-zero pair routing
  - `behavior_change_expected`: yes, runtime route and compact-step granularity
    only; full didactic step collection remains on the existing path
- change:
  - added a narrow compact matcher for the exact pair
    `tan(x)*cot(x)-1` with `sin(x)^2 - (1 - cos(2*x))/2`
  - installed it before the generic
    `root.addsub.00.direct_small_zero_pair.compact_first` route, only when
    `collect_steps && StepsMode::Compact`
  - added a regression test for the compact shortcut so the optimized path is
    covered independently from the broader generic matcher
- local pressure result:
  - `case#3145`:
    - before: steady-state simplify `~32.25ms`
    - after: initial simplify `1.10ms`, steady-state simplify `~0.07ms`
    - profile now shows
      `root.addsub.00.tan_cot_half_angle_pair.compact_first`: `4/4` hits,
      `0.312ms` total
  - local `sum@700+100` slice:
    - `100/100`
    - elapsed `29.29ms`
    - simplify avg `0.12ms`
  - pressure scorecard:
    - `simplify_zero_mixed`: `450/450`
    - elapsed `0.24s -> 0.21s`
    - `sum` simplify avg `0.26ms -> 0.11ms`
    - `sum@0+100` simplify avg `0.12ms -> 0.12ms`
    - `sum@700+100` simplify avg `0.40ms -> 0.09ms`
- guardrails:
  - targeted orchestrator test:
    - `compact_tan_cot_half_angle_pair_shortcut_handles_sum_regression`: `1/1`
  - `cargo fmt -- --check`: passed
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
- decision:
  - retained as runtime improvement
- next:
  - current pressure heads are now low-ms/sub-ms hyperbolic/log combinations;
    inspect `case#3049`
    `(2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y))) + (cosh(x)^2 - sinh(x)^2 - 1)`
    or the `sum@0+100` head `case#25` before widening any generic matcher

### 2026-04-27: Direct Hyperbolic Pythagorean Zero-Scope Orientation

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `simplify_zero_mixed` pressure `sum@700+100`
- status:
  - `retained`
- investment:
  - `investment_class`: runtime
  - `primary_dimension`: runtime
  - `secondary_dimension`: robustness/coverage-reuse
  - `cohesion_scope`: direct hyperbolic equivalence helper in `arithmetic.rs`
  - `behavior_change_expected`: yes, exact-zero route selection only
- change:
  - reused the existing hyperbolic Pythagorean rewrite inside the exact
    hyperbolic cancellation helper
  - added signed-additive recognition for the `AddView` subset form of
    `cosh(x)^2 - sinh(x)^2`, so
    `cosh(x)^2 - sinh(x)^2 - 1` can prove directly through
    `zero_scope_exact_hyperbolic_equivalence`
  - added focused tests for both the public exact-zero additive rule and the
    private exact hyperbolic zero-scope helper
- local pressure result:
  - `case#3049`:
    `(2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y))) + (cosh(x)^2 - sinh(x)^2 - 1)`
    - before: profiled steady-state simplify `~0.23ms`
    - after: profiled steady-state simplify `~0.19ms`
    - route movement:
      - `rule.direct_identity.try.zero_scope_exact_hyperbolic_equivalence`:
        `8` attempts / `0` hits -> `4` attempts / `4` hits
      - direct-identity probe attempts: `56 -> 28`
  - local `sum@700+100` slice:
    - `100/100`
    - `case#3049` steady-state simplify `~0.17ms` in the local window run
  - pressure scorecard:
    - `simplify_zero_mixed`: `450/450`
    - elapsed `0.21s -> 0.206s`
    - `sum` simplify avg held at `0.11ms`
    - `sum@700+100` simplify avg held at `0.09ms`
- guardrails:
  - targeted arithmetic tests:
    - `collapse_exact_zero_additive_subexpression_matches_direct_hyperbolic_pythagorean_zero_scope`: `1/1`
    - `direct_exact_hyperbolic_zero_scope_matches_additive_pythagorean_orientation`: `1/1`
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
- rejected local candidates:
  - simply calling the existing `try_rewrite_hyperbolic_pythagorean_sub_expr`
    did not move the live profile because the zero-scope subset builder presents
    the focus as a signed additive expression rather than the original `Sub`
    node; retained only after adding the signed-additive recognizer
- decision:
  - retained as a small runtime/robustness improvement
- next:
  - the next pressure head is again `case#25`
    `(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (cosh(x) + sinh(x) - e^x)`;
    current profiling suggests no broad orchestrator cliff, so prefer a
    narrow log/hyperbolic helper only if a new profile shows repeated misses
    rather than adding another generic additive shortcut

### 2026-04-27: Pressure Steady-Rerun Millisecond Rendering

- area:
  - [engine_improvement_scorecard.py](/Users/javiergimenezmoya/developer/math/scripts/engine_improvement_scorecard.py)
  - [test_engine_improvement_scorecard.py](/Users/javiergimenezmoya/developer/math/scripts/test_engine_improvement_scorecard.py)
  - `simplify_zero_mixed` pressure observability
- status:
  - `retained`
- investment:
  - `investment_class`: observability
  - `primary_dimension`: observability
  - `secondary_dimension`: runtime-selection
  - `cohesion_scope`: scorecard markdown rendering only
  - `behavior_change_expected`: no engine behavior change; report precision
    only
- trigger:
  - profiled the current pressure heads before touching files:
    - `case#25` showed steady-state simplify around `0.13ms`; profile hit
      `rule.direct_identity.try.two_term_safe_hyperbolic` `4/4`
    - `case#53` showed steady-state simplify around `0.15ms`; same
      hyperbolic direct-identity route hit `4/4`
  - this did not expose a broad runtime cliff, but the generated Markdown was
    still printing steady rerun medians as `0.00s`, hiding the sub-ms ordering
- change:
  - added `format_runtime_duration` so sub-second runtime values render in ms
    rather than second-rounded `0.00s`
  - applied it to the pressure scorecard's steady-state engine rerun summary
  - added unit coverage for both the formatter and the rendered Markdown
    pressure section
- local pressure result:
  - generated pressure Markdown now reports the top steady reruns as:
    - `case#25`: `median_simplify=0.11ms`, `median_wire=0.13ms`,
      `median_wall=0.37ms`
    - `case#53`: `median_simplify=0.13ms`, `median_wire=0.16ms`,
      `median_wall=0.39ms`
    - `case#57`: `median_simplify=0.12ms`, `median_wire=0.14ms`,
      `median_wall=0.38ms`
  - pressure scorecard:
    - `simplify_zero_mixed`: `450/450`
    - elapsed `207.56ms`
    - `sum` avg simplify `0.11ms`
- guardrails:
  - `python3 -m unittest scripts/test_engine_improvement_scorecard.py`: passed
    - `12` tests
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
- decision:
  - retained as an observability improvement; this does not make the engine
    more powerful directly, but it prevents future cycles from losing useful
    runtime signal once pressure cases are sub-ms
- next:
  - keep the current `case#25` / `case#53` profiles as low-priority runtime
    candidates unless a later profile shows repeated misses or a broader
    non-hyperbolic route; otherwise continue corpus/coverage growth

### 2026-04-27: Pressure Aggregate Millisecond Rendering

- area:
  - [engine_improvement_scorecard.py](/Users/javiergimenezmoya/developer/math/scripts/engine_improvement_scorecard.py)
  - [test_engine_improvement_scorecard.py](/Users/javiergimenezmoya/developer/math/scripts/test_engine_improvement_scorecard.py)
  - `simplify_zero_mixed` pressure observability
- status:
  - `retained`
- investment:
  - `investment_class`: observability
  - `primary_dimension`: observability
  - `secondary_dimension`: runtime-selection
  - `cohesion_scope`: scorecard markdown rendering only
  - `behavior_change_expected`: no engine behavior change; report precision
    only
- trigger:
  - the live `combined_additive_zero` corpus is now balanced at the current
    target (`23/23` families, each non-`simplify` family at `6` cases), so a
    coverage promotion from the old `collect` discovery was not the highest ROI
  - after the previous steady-rerun formatter improvement, the same pressure
    Markdown still rounded aggregate rows such as `difference simplify` to
    `0.00s`, hiding real `4ms`-range costs
- change:
  - reused `format_runtime_duration` for `Mixed Zero Pressure` aggregate
    elapsed/simplify/wall fields in composition hotspots, engine hotspots, and
    window slices
  - added Markdown assertions that block regressions back to second-rounded
    `simplify=0.00s`
- pressure result:
  - generated pressure Markdown now reports:
    - composition hotspots:
      - `sum`: `elapsed=77.29ms`, `simplify=21.38ms`
      - `shifted_quotient`: `elapsed=59.35ms`, `simplify=9.15ms`
      - `difference`: `elapsed=26.35ms`, `simplify=4.42ms`
    - engine hotspots:
      - `sum simplify=21.38ms wall=77.29ms`
      - `difference simplify=4.42ms wall=26.35ms`
    - window slices now render as ms, including
      `sum@700+100 elapsed=23.14ms`
  - pressure scorecard:
    - `simplify_zero_mixed`: `450/450`
    - elapsed `205.04ms`
- guardrails:
  - `python3 -m unittest scripts/test_engine_improvement_scorecard.py`: passed
    - `12` tests
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
- decision:
  - retained as a harness/observability improvement; it does not increase
    mathematical power directly, but it preserves the timing signal now that
    promoted pressure lanes are mostly sub-second and often sub-100ms
- next:
  - use the clearer pressure aggregates to decide between sparse-wrapper
    coverage growth and a narrow runtime pass only if a future profile shows
    repeated misses rather than already-hot successful direct routes

### 2026-04-27: Squared Passthrough Collect Common-Factor Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x collect`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: shell_depth / source-shape breadth
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero` remained sparse after the previous observability
    pass
  - `collect` had only linear representatives under this wrapper, while
    common symbolic coefficient collection was already a stable low-cost shape
    in other contexts
- change:
  - promoted one live row:
    `(((x*y + x*z + w)^2) + m) - (((x*(y + z) + w)^2) + m) -> 0`
  - this keeps the promotion minimal and avoids turning a single stable smoke
    into a batch expansion
- candidate smoke:
  - `1/1` passed
  - runner elapsed `4.28ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
- filtered result:
  - `squared_passthrough_zero --family collect`: `3/3`
  - elapsed `2.94ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `206.39ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1399/1399`
    - average case runtime `2.852ms`
    - sparse bucket: `squared_passthrough_zero x collect total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as a coverage promotion; it does not increase runtime power by
    itself, but it adds a guarded representative proving this wrapper/family
    combination remains correct for a broader collect source shape
- next:
  - continue sparse-wrapper coverage with another one-row smoke from the
    remaining `squared_passthrough_zero` families at count `2`, or spend a
    cycle on observability if sparse-wrapper selection becomes hard to audit

### 2026-04-27: Squared Passthrough Finite Product Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x finite_telescoping`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: finite telescoping source-shape breadth
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x finite_telescoping` only covered the sum
    telescoping shape
  - the product telescoping shape was already stable in simple wrappers and in
    combined additive coverage, but not under the squared passthrough wrapper
- change:
  - promoted one live row:
    `(((product((k+1)/k, k, 1, n))^2) + m) - (((n+1)^2) + m) -> 0`
  - this is the minimal product representative; no double-squared variant and
    no extra additive/composition core were added
- candidate smoke:
  - `1/1` passed
  - runner elapsed `16.41ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
- filtered result:
  - `squared_passthrough_zero --family finite_telescoping`: `3/3`
  - elapsed `37.96ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `221.86ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1400/1400`
    - average case runtime `3.164ms`
    - sparse bucket: `squared_passthrough_zero x finite_telescoping total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
  - repeated embedded-only check:
    - `1400/1400`
    - elapsed `4.39s`
- decision:
  - retained as coverage, not as runtime work
  - the global embedded elapsed was higher than the previous guardrail run, but
    the candidate is locally bounded (`16.41ms` smoke, `37.96ms` filtered
    family), has zero failures, and adds a distinct finite-product shape rather
    than another sum/noise variant
- next:
  - continue sparse-wrapper growth only with low-cost one-row smokes; prefer a
    cheaper non-finite family next if embedded elapsed stays around `4.4s`

### 2026-04-27: Squared Passthrough Expand Difference Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x expand`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: sign/orientation robustness
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x expand` only covered the common-factor sum
    representative and its double-squared shell
  - after the finite-product promotion showed higher embedded timing, the next
    coverage row needed to be a cheap non-finite representative
- change:
  - promoted one live row:
    `(((a*(b-c))^2) + m) - (((a*b - a*c)^2) + m) -> 0`
  - this adds subtraction distribution under the squared passthrough wrapper
    without adding the double-squared variant or extra composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `4.41ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
- filtered result:
  - `squared_passthrough_zero --family expand`: `3/3`
  - elapsed `2.99ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `209.54ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1401/1401`
    - average case runtime `2.948ms`
    - sparse bucket: `squared_passthrough_zero x expand total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as a low-cost coverage promotion
  - it broadens the squared wrapper's expand coverage from positive
    distribution to subtraction/sign handling while keeping local and global
    runtime within guardrail expectations
- next:
  - continue one-row sparse-wrapper coverage in a remaining count-`2` family,
    favoring similarly cheap algebraic representatives before heavier log,
    radical, or telescoping shapes

### 2026-04-27: Squared Passthrough Fraction Difference Combine Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x fraction_combine`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: sign/orientation robustness
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x fraction_combine` only covered the same
    denominator sum representative and its double-squared shell
  - the same denominator difference representative was already stable in
    simple wrappers but not under the squared passthrough wrapper
- change:
  - promoted one live row:
    `(((a/d - b/d)^2) + m) - ((((a-b)/d)^2) + m) -> 0`
  - this adds fraction subtraction/combine coverage without adding a
    double-squared variant, general-denominator variant, or composition core
- candidate smoke:
  - `1/1` passed
  - runner elapsed `21.08ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
- filtered result:
  - `squared_passthrough_zero --family fraction_combine`: `3/3`
  - elapsed `16.44ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `205.74ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1402/1402`
    - average case runtime `2.889ms`
    - sparse bucket: `squared_passthrough_zero x fraction_combine total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - local filtered cost is bounded and the final embedded guardrail improved
    relative to the previous run, so this does not create the kind of runtime
    tax seen in heavier finite/log candidates
- next:
  - continue sparse-wrapper coverage with another count-`2` family; the next
    cheap candidates are likely `power_merge`, `polynomial_product`, or
    `fraction_expand` before heavier log/radical/telescoping shapes

### 2026-04-27: Squared Passthrough Fraction Difference Expand Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x fraction_expand`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: sign/orientation robustness and fraction
    expansion subtraction
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x fraction_expand` covered the same denominator
    sum representative and its double-squared shell
  - the previous retained `fraction_combine` difference row confirmed the
    inverse orientation is cheap, so this completes the paired expansion
    shape under the squared wrapper
- change:
  - promoted one live row:
    `((((a-b)/d)^2) + m) - (((a/d - b/d)^2) + m) -> 0`
  - this covers `(a-b)/d -> a/d - b/d` without adding a double-squared
    variant, a general-denominator variant, or a composition core
- candidate smoke:
  - `1/1` passed
  - runner elapsed `16.48ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
- filtered result:
  - `squared_passthrough_zero --family fraction_expand`: `3/3`
  - elapsed `18.27ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `201.81ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1403/1403`
    - average case runtime `2.865ms`
    - sparse bucket: `squared_passthrough_zero x fraction_expand total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - local and global runtime stayed bounded while the sparse squared-wrapper
    bucket moved out of the count-`2` set
- next:
  - continue low `squared_passthrough_zero` coverage with cheap
    `power_merge` or `polynomial_product` before heavier
    log/radical/telescoping shapes

### 2026-04-27: Squared Passthrough Symbolic Power Merge Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x power_merge`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: symbolic exponent regime under squared passthrough
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x power_merge` was still at count `2`
  - the squared rows only covered fixed fractional powers and the
    double-squared variant, while simple wrappers already covered symbolic
    `x^a*x^b -> x^(a+b)`
  - a competing `polynomial_product` cube-difference squared candidate also
    passed smoke, but was less cheap locally (`13.56ms` vs `3.54ms`)
- change:
  - promoted one live row:
    `(((x^a*x^b)^2) + m) - (((x^(a+b))^2) + m) -> 0`
  - this adds symbolic exponent coverage without adding a double-squared
    variant, quotient-power variant, or cross-family composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `3.54ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
- filtered result:
  - `squared_passthrough_zero --family power_merge`: `3/3`
  - elapsed `4.39ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `201.34ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1404/1404`
    - average case runtime `2.942ms`
    - sparse bucket: `squared_passthrough_zero x power_merge total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - the candidate broadens the squared wrapper from fixed fractional exponents
    to symbolic exponent merge with low filtered cost and no global failures
- next:
  - continue low `squared_passthrough_zero` coverage with
    `polynomial_product` as the next cheap candidate, then reassess heavier
    log/radical/telescoping families

### 2026-04-27: Squared Passthrough Polynomial Product Difference Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x polynomial_product`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: sign/orientation robustness for polynomial product
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x polynomial_product` was still at count `2`
  - the squared rows covered the cube-sum product and its double-squared shell,
    while the cube-difference product was already stable in simple wrappers
- change:
  - promoted one live row:
    `((((x-a)*(x^2+a*x+a^2))^2) + m) - (((x^3-a^3)^2) + m) -> 0`
  - this covers `(x-a)*(x^2+a*x+a^2) -> x^3-a^3` without adding a
    double-squared variant, higher-degree product, or cross-family composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `3.93ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
- filtered result:
  - `squared_passthrough_zero --family polynomial_product`: `3/3`
  - elapsed `2.97ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `203.59ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1405/1405`
    - average case runtime `2.868ms`
    - sparse bucket: `squared_passthrough_zero x polynomial_product total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - it adds the missing sign variant for the squared polynomial-product
    representative while keeping local and global runtime bounded
- next:
  - continue low `squared_passthrough_zero` coverage with the cheapest
    remaining simple algebraic families, likely `radical_power` or
    `rationalize`, before heavier log/telescoping cases

### 2026-04-27: Squared Passthrough Radical Power Sqrt-Cube Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x radical_power`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: radical/power regime after prior simplification
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x radical_power` was still at count `2`
  - existing squared rows covered `x^(3/2) -> abs(x)*sqrt(x)` and its
    double-squared shell, while simple wrappers already covered the explicit
    radical source `sqrt(x^3) -> abs(x)*sqrt(x)`
  - a competing `rationalize_linear_root_squared` candidate passed
    symbolically but exceeded the local smoke slow threshold (`70.10ms`), so it
    was not promoted in this cycle
- change:
  - promoted one live row:
    `(((sqrt(x^3))^2) + m) - (((abs(x)*sqrt(x))^2) + m) -> 0`
  - this covers explicit radical source shape without adding a double-squared
    variant, passthrough variant, or cross-family composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `13.94ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
- filtered result:
  - `squared_passthrough_zero --family radical_power`: `3/3`
  - elapsed `5.05ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `201.32ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1406/1406`
    - average case runtime `2.859ms`
    - sparse bucket: `squared_passthrough_zero x radical_power total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - it broadens the squared wrapper from odd half-power syntax to explicit
    radical syntax with bounded local and global cost
- next:
  - revisit `rationalize` only with a cheaper representative or stress-only
    treatment; otherwise continue low `squared_passthrough_zero` coverage with
    simple non-log families such as `fraction_decompose` or `solve_prep`

### 2026-04-27: Squared Passthrough Fraction Decompose Scaled Linear Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x fraction_decompose`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: scaled linear fraction decomposition
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x fraction_decompose` was still at count `2`
  - existing squared rows covered the unscaled denominator shape
    `(a*x+b)/(x+c)`, including a double-squared shell
  - simple wrappers already covered the scaled linear denominator
    `(a*x+b)/(c*x+d) -> a/c + (b-a*d/c)/(c*x+d)`
  - `solve_prep` remained a possible low-family target, but prior ledgered
    runtime sensitivity made this narrower fraction-decomposition promotion
    the lower-risk candidate for this cycle
- change:
  - promoted one live row:
    `((((a*x+b)/(c*x+d))^2) + m) - (((a/c + (b-a*d/c)/(c*x+d))^2) + m) -> 0`
  - this covers the scaled linear decomposition shape under a shared squared
    passthrough without adding a double-squared variant or a cross-family
    composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `6.51ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `20`
- filtered result:
  - `squared_passthrough_zero --family fraction_decompose`: `3/3`
  - elapsed `8.75ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `200.24ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1407/1407`
    - average case runtime `2.843ms`
    - sparse bucket: `squared_passthrough_zero x fraction_decompose total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - it closes another low squared-wrapper family with a scaled denominator
    representative already known to be stable in simpler wrappers
- next:
  - continue low `squared_passthrough_zero` coverage with remaining count-2
    families, preferring cheap non-log candidates before revisiting
    `rationalize` or the historically sensitive `solve_prep` lane

### 2026-04-27: Squared Passthrough Log Exp Inverse Log10 Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x log_exp_inverse`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: base-10 log/power alias
  - `cohesion_scope`: one embedded corpus row plus discovery notes
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x log_exp_inverse` was still at count `2`
  - existing squared rows covered the natural alias
    `exp(y*log(x)) -> x^y`, including a double-squared shell
  - simple wrappers already covered the base-10 alias
    `10^(y*log10(x)) -> x^y`
  - a prior base-10 combined-additive candidate was rejected for global
    embedded runtime cost, so this cycle required a minimal non-composed smoke
    before promotion
- rejected discovery probes before promotion:
  - `conditional_factor` higher-power extraction under squared passthrough
    failed before live promotion:
    - quartic `x^2` extraction squared: `0/1`, elapsed `15.67ms`
    - matching additive passthrough quartic probe: `1/1`, elapsed `23.12ms`
    - septic `x^3` extraction squared: `0/1`, elapsed `11.43ms`
  - this is a reusable structural gap: conditional factorization by higher
    powers is stable in simpler wrappers but not when both sides are wrapped as
    squared expressions
  - additional heavier squared candidates were not promoted:
    - `integrate_prep` Dirichlet reverse squared: `0/1`, elapsed `14.80ms`
    - `telescoping_fraction` difference-squares split squared: `0/1`,
      elapsed `11.81ms`
    - `trig_expand` phase-shift pair squared: `0/1`, elapsed `73.31ms`
    - `solve_prep` negative linear coefficient squared: `0/1`, elapsed
      `73.60ms`
- change:
  - promoted one live row:
    `(((10^(y*log10(x)))^2) + m) - (((x^y)^2) + m) -> 0`
  - this covers base-10 log-exp inverse under a shared squared passthrough
    without adding double-squared depth or additive cross-family composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `3.35ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `12`
- filtered result:
  - `squared_passthrough_zero --family log_exp_inverse`: `3/3`
  - elapsed `4.23ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `200.75ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1408/1408`
    - average case runtime `2.834ms`
    - sparse bucket: `squared_passthrough_zero x log_exp_inverse total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - the minimal log10 squared row closed a low sparse-wrapper family without
    repeating the earlier global runtime rejection from the composed log10
    candidate
- next:
  - stop trying arbitrary heavy squared wrappers as live rows; either improve
    the squared-passthrough route for a documented structural gap such as
    higher-power `conditional_factor`, or continue coverage only with
    smoke-cheap minimal representatives such as simple `log_contract`

### 2026-04-27: Squared Passthrough Log Contract Basic Sum Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x log_contract`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: basic log-sum contraction
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x log_contract` was still at count `2`
  - existing squared rows covered grouped-power contraction
    `ln(x^2)+ln(y^2) -> ln((x*y)^2)`, including a double-squared shell
  - simple wrappers already covered the basic contraction
    `ln(x)+ln(y) -> ln(x*y)`
  - the prior cycle showed that heavy squared candidates should not be promoted
    blindly, so this cycle used a minimal smoke-cheap representative
- change:
  - promoted one live row:
    `(((ln(x)+ln(y))^2) + m) - (((ln(x*y))^2) + m) -> 0`
  - this covers basic log-sum contraction under a shared squared passthrough
    without adding double-squared depth, reciprocal noise, or cross-family
    composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `26.07ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `13`
- filtered result:
  - `squared_passthrough_zero --family log_contract`: `3/3`
  - elapsed `17.47ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `213.69ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1409/1409`
    - average case runtime `3.102ms`
    - sparse bucket: `squared_passthrough_zero x log_contract total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - embedded elapsed moved from the prior `3.99s` to `4.37s`, which is below
    the scorecard's material runtime delta threshold while preserving
    correctness and closing the target sparse-wrapper bucket
- next:
  - remaining count-2 squared-wrapper families are mostly heavier or already
    documented as fragile; prefer either a runtime/robustness iteration for the
    `conditional_factor` squared gap or another smoke-cheap minimal family such
    as `log_inverse_power`

### 2026-04-27: Squared Passthrough Log Inverse Power Reversed Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x log_inverse_power`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: sign/orientation robustness for log inverse power
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x log_inverse_power` was still at count `2`
  - existing squared rows covered the forward orientation and a double-squared
    shell
  - this cycle used the reversed orientation as a minimal representative
    instead of adding a deeper or cross-family composition
- change:
  - promoted one live row:
    `(((log(x))^2) + m) - (((x^(log(log(x))/log(x)))^2) + m) -> 0`
  - this tests that the same log inverse-power identity is stable when the
    enclosing subtraction is reversed under the shared squared passthrough
- candidate smoke:
  - `1/1` passed
  - runner elapsed `13.89ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `11`
- filtered result:
  - `squared_passthrough_zero --family log_inverse_power`: `3/3`
  - elapsed `4.64ms`
  - average wrapper overhead nodes `12.33`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
    - unit smoke/scorecard tests: `20/20`
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `215.82ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1410/1410`
    - average case runtime `3.078ms`
    - sparse bucket: `squared_passthrough_zero x log_inverse_power total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - the row closed the target sparse-wrapper family without material embedded
    runtime degradation; embedded moved from `1409/1409` at `3.102ms/case` to
    `1410/1410` at `3.078ms/case`
- next:
  - continue only with smoke-cheap minimal squared-wrapper representatives for
    remaining count-2 families, or switch to a runtime/robustness iteration for
    the documented `conditional_factor` higher-power squared gap

### 2026-04-27: Squared Passthrough Telescoping Affine Shift-Gap Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x telescoping_fraction`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: affine symbolic shift-gap telescoping core
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x telescoping_fraction` was still at count `2`
  - existing squared rows covered the shifted quadratic form
    `1/(x+b)-1/(x+c) -> (c-b)/(x^2+(b+c)*x+b*c)`
  - simple wrappers already covered the affine symbolic gap form
    `1/(c-b)*(1/(a*n+b)-1/(a*n+c)) -> 1/((a*n+b)*(a*n+c))`
  - alternative smoke candidates for reversed `trig_expand` and reversed
    `rationalize` passed, but were mostly orientation repeats; this row adds a
    distinct telescoping parameterization under the squared wrapper
- change:
  - promoted one live row:
    `(((1/(c-b)*(1/(a*n+b) - 1/(a*n+c)))^2) + m) - (((1/((a*n+b)*(a*n+c)))^2) + m) -> 0`
  - this covers affine symbolic denominator shifts without adding
    double-squared depth, reciprocal noise, or cross-family composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `17.54ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `22`
- filtered result:
  - `squared_passthrough_zero --family telescoping_fraction`: `3/3`
  - elapsed `16.81ms`
  - average wrapper overhead nodes `22.00`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
    - unit smoke/scorecard tests: `20/20`
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `211.38ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1411/1411`
    - average case runtime `3.040ms`
    - sparse bucket: `squared_passthrough_zero x telescoping_fraction total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - the row closed the target sparse-wrapper family without material embedded
    runtime degradation; embedded moved from `1410/1410` at `3.078ms/case` to
    `1411/1411` at `3.040ms/case`
- next:
  - remaining count-2 squared-wrapper families are
    `conditional_factor`, `integrate_prep`, `rationalize`, `solve_prep`, and
    `trig_expand`; prefer a smoke-cheap minimal representative unless choosing
    to address the documented `conditional_factor` higher-power gap directly

### 2026-04-27: Squared Passthrough Trig Product-To-Sum Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x trig_expand`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: product-to-sum trig expansion under squared shell
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x trig_expand` was still at count `2`
  - existing squared rows covered only the `sin(2*x) -> 2*sin(x)*cos(x)`
    double-angle shape
  - simple wrappers already covered product-to-sum forms with independent
    angles, so the smallest useful promotion was one squared row for
    `2*sin(x)*cos(y) -> sin(x+y)+sin(x-y)`
  - alternative smoke candidates for reversed `rationalize` and reversed
    `integrate_prep` passed, but they mainly tested orientation rather than a
    distinct core shape
- rejected probe before promotion:
  - `solve_prep_complete_square_symbolic_leading_coeff_squared` failed
    smoke as `0/1`, runner elapsed `45.70ms`
  - the same leading-coefficient complete-square form exists in simpler
    wrappers, so this is a useful future robustness/runtime candidate rather
    than a live promotion
- change:
  - promoted one live row:
    `(((2*sin(x)*cos(y))^2) + m) - (((sin(x+y) + sin(x-y))^2) + m) -> 0`
  - this covers product-to-sum trig expansion without adding double-squared
    depth, reciprocal noise, or cross-family composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `13.47ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `16`
- filtered result:
  - `squared_passthrough_zero --family trig_expand`: `3/3`
  - elapsed `3.73ms`
  - average wrapper overhead nodes `15.33`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
    - unit smoke/scorecard tests: `20/20`
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `216.25ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1412/1412`
    - average case runtime `3.123ms`
    - sparse bucket: `squared_passthrough_zero x trig_expand total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - the row closed the target sparse-wrapper family; embedded moved from
    `1411/1411` at `3.040ms/case` to `1412/1412` at `3.123ms/case`, below the
    material runtime threshold
- next:
  - remaining count-2 squared-wrapper families are `conditional_factor`,
    `integrate_prep`, `rationalize`, and `solve_prep`; prefer either
    smoke-cheap minimal representatives or a focused robustness fix for the
    repeated `solve_prep`/`conditional_factor` squared gaps

### 2026-04-27: Solve Prep Leading-Coefficient Squared Discovery

- area:
  - generated discovery / embedded equivalence candidate smoke
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero` x `solve_prep`
- status:
  - `resolved`
- attempted row:
  - `solve_prep_complete_square_symbolic_leading_coeff_squared`
  - `(((a*x^2 + b*x + c)^2) + m) - (((a*(x + b/(2*a))^2 + c - b^2/(4*a))^2) + m)`
- local lane:
  - `python3 scripts/engine_embedded_candidate_smoke.py --json --row ...`
- local result:
  - attempted squared candidate failed before live promotion:
    - `status=fail`
    - `passed=0`
    - `failed=1`
    - `runner_elapsed=45.70ms`
- structural read:
  - the same leading-coefficient complete-square identity is already live in
    additive, scaled, common-denominator, and shifted-quotient wrappers
  - the monic complete-square variant already has squared passthrough coverage
  - the miss therefore points to a reusable gap in non-monic `solve_prep`
    complete-square handling under squared wrappers, not to a typo or malformed
    candidate
- decision:
  - resolved by the later square-base fallback robustness iteration
  - the same representative is now promoted live as
    `solve_prep_complete_square_symbolic_leading_coeff_squared`

### 2026-04-27: Squared Passthrough Rationalize Reverse Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x rationalize`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: sign/orientation robustness for radical
    rationalization
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x rationalize` was still at count `2`
  - existing squared rows covered direct orientation and double-squared depth
    for `1/(sqrt(a)+sqrt(b)) -> (sqrt(a)-sqrt(b))/(a-b)`
  - combined-additive rows already covered reversed rationalization in noisier
    contexts, so the smallest missing live representative was the reversed
    squared wrapper with no extra family composition
- rejected probe before promotion:
  - `factor_out_square_with_division_quartic_squared` for
    `squared_passthrough_zero x conditional_factor` failed smoke as `0/1`,
    runner elapsed `20.21ms`
  - this was not re-entered as a new discovery because the same higher-power
    conditional-factor squared gap is already documented in this ledger
- change:
  - promoted one live row:
    `((((sqrt(a)-sqrt(b))/(a-b))^2) + m) - (((1/(sqrt(a)+sqrt(b)))^2) + m) -> 0`
  - this covers reversed radical rationalization without adding double-squared
    depth, reciprocal noise, or cross-family composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `2.70ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `16`
- filtered result:
  - `squared_passthrough_zero --family rationalize`: `3/3`
  - elapsed `5.03ms`
  - average wrapper overhead nodes `17.33`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
    - unit smoke/scorecard tests: `20/20`
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `212.24ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1413/1413`
    - average case runtime `3.057ms`
    - sparse bucket: `squared_passthrough_zero x rationalize total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - the row closed the target sparse-wrapper family; embedded moved from
    `1412/1412` at `3.123ms/case` to `1413/1413` at `3.057ms/case`, below the
    material runtime threshold
- next:
  - remaining count-2 squared-wrapper families are `conditional_factor`,
    `integrate_prep`, and `solve_prep`; prefer `integrate_prep` if continuing
    low-risk corpus closure, or switch to robustness for the documented
    `conditional_factor` and `solve_prep` squared gaps

### 2026-04-27: Integrate Prep Dirichlet Squared Discovery

- area:
  - generated discovery / embedded equivalence candidate smoke
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero` x `integrate_prep`
- status:
  - `resolved`
- attempted rows:
  - `integrate_prep_dirichlet_basic_squared`
  - `(((1 + 2*cos(x) + 2*cos(2*x))^2) + m) - (((sin(5*x/2)/sin(x/2))^2) + m)`
  - `integrate_prep_dirichlet_reverse_squared`
  - `(((sin(5*x/2)/sin(x/2))^2) + m) - (((1 + 2*cos(x) + 2*cos(2*x))^2) + m)`
- local lane:
  - `python3 scripts/engine_embedded_candidate_smoke.py --json --row ...`
- local result:
  - direct squared candidate failed before live promotion:
    - `status=fail`
    - `passed=0`
    - `failed=1`
    - `runner_elapsed=15.26ms`
  - reversed squared candidate failed before live promotion:
    - `status=fail`
    - `passed=0`
    - `failed=1`
    - `runner_elapsed=4.56ms`
- resolution probe:
  - after the square-base fallback robustness work, both attempted orientations
    pass candidate smoke:
    - direct: `1/1`, runner elapsed `15.18ms`
    - reverse: `1/1`, runner elapsed `3.34ms`
  - promoted only the direct row as the minimal live representative:
    `integrate_prep_dirichlet_basic_squared`
- structural read:
  - the same Dirichlet kernel identity is already live in additive, scaled,
    common-denominator, shifted-quotient, and combined-additive wrappers
  - both squared orientations fail, while the Morrie product identity remains
    stable under squared wrappers
  - this points to an `integrate_prep` subfamily gap for Dirichlet forms under
    squared passthrough, not to a malformed candidate
- decision:
  - resolved by the square-base fallback robustness route
  - close the observe-only discovery and retain one minimal live corpus row for
    Dirichlet squared passthrough coverage

### 2026-04-27: Squared Passthrough Integrate Prep Morrie Reverse Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x integrate_prep`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: sign/orientation robustness for Morrie product
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x integrate_prep` was still at count `2`
  - existing squared rows covered direct orientation and double-squared depth
    for `cos(x)*cos(2*x)*cos(4*x) -> sin(8*x)/(8*sin(x))`
  - the Dirichlet squared candidates exposed a subfamily gap and were kept as
    observe-only discovery, so the smallest stable live promotion was the
    reversed Morrie product under the same squared wrapper
- rejected probe before promotion:
  - `integrate_prep_dirichlet_basic_squared` failed smoke as `0/1`, runner
    elapsed `15.26ms`
  - `integrate_prep_dirichlet_reverse_squared` failed smoke as `0/1`, runner
    elapsed `4.56ms`
  - both are documented in the adjacent observe-only discovery section
- change:
  - promoted one live row:
    `(((sin(8*x)/(8*sin(x)))^2) + m) - (((cos(x)*cos(2*x)*cos(4*x))^2) + m) -> 0`
  - this covers reversed Morrie product preparation without adding
    double-squared depth, reciprocal noise, or cross-family composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `13.50ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `18`
- filtered result:
  - `squared_passthrough_zero --family integrate_prep`: `3/3`
  - elapsed `3.47ms`
  - average wrapper overhead nodes `19.33`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
    - unit smoke/scorecard tests: `20/20`
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `220.27ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1414/1414`
    - average case runtime `3.098ms`
    - sparse bucket: `squared_passthrough_zero x integrate_prep total=3 failed=0`
    - observe-only discoveries: `3`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - the row closed the target sparse-wrapper family; embedded moved from
    `1413/1413` at `3.057ms/case` to `1414/1414` at `3.098ms/case`, below the
    material runtime threshold
- next:
  - only `conditional_factor` and `solve_prep` remain at count `2` for
    `squared_passthrough_zero`; both have documented squared-wrapper gaps, so
    prefer a robustness iteration over more corpus closure

### 2026-04-27: Squared Passthrough Conditional Factor Square-Base Robustness

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x conditional_factor`
- status:
  - `retained`
- investment:
  - `investment_class`: robustness
  - `primary_dimension`: squared wrapper robustness for conditional factor
  - `secondary_dimension`: square-base equivalence after passthrough
    cancellation
  - `cohesion_scope`: narrow exact-zero helpers in `arithmetic.rs`
  - `behavior_change_expected`: yes
- trigger:
  - `conditional_factor` was one of the last count-2
    `squared_passthrough_zero` families
  - the quartic factor-out base already passed additive/scaled/common/
    shifted wrappers, but the squared passthrough candidate failed before
    promotion
- baseline probe:
  - candidate row:
    `(((a*x^4 + b*x^3 + c*x^2 + d)^2) + m) - (((x^2*(a*x^2 + b*x + c + d/x^2))^2) + m) -> 0`
  - smoke before change: `0/1`, `status=fail`, runner elapsed `8.58ms`
  - failure shape after the normal flow canceled `+m`: residual stayed as
    `(a*x^4 + b*x^3 + c*x^2 + d)^2 - (x^2*(...))^2`
- change:
  - added square-base equivalence handling for shared passthrough cores
  - added exact-zero handling for a bare difference of equivalent squares
    after the passthrough terms have already been canceled
  - added focused unit coverage for both the shared-passthrough form and the
    exposed `A^2 - B^2` residual
  - promoted one live corpus row for the quartic squared conditional-factor
    representative
- candidate smoke:
  - after change: `1/1` passed
  - runner elapsed `6.61ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `28`
- filtered result:
  - `squared_passthrough_zero --family conditional_factor`: `3/3`
  - elapsed `4.51ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - unit tests:
    - `shared_passthrough_square_base_equivalence_keeps_quartic_conditional_factor_regression`
    - `collapse_exact_zero_additive_subexpression_matches_quartic_conditional_factor_square_difference`
  - `make engine-fast`: passed
    - unit smoke/scorecard tests: `20/20`
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `223.47ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1415/1415`
    - average case runtime `3.286ms`
    - `squared_passthrough_zero`: `70/70`
    - sparse bucket: `squared_passthrough_zero x conditional_factor total=3 failed=0`
    - observe-only discoveries: `3`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`, `0` timeouts
  - isolated embedded rerun after guardrail:
    - `1415/1415`, elapsed `4.64s`
- decision:
  - retained as robustness plus one minimal live promotion
  - the iteration converts a previously documented structural weakness into a
    bounded engine route; no failures or timeouts were introduced
- next:
  - `solve_prep` remains the only count-2 `squared_passthrough_zero` family;
    prefer a robustness iteration for its non-monic leading-coefficient
    squared gap before adding more simple corpus rows

### 2026-04-27: Squared Passthrough Solve Prep Square-Base Robustness

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x solve_prep`
- status:
  - `retained`
- investment:
  - `investment_class`: robustness
  - `primary_dimension`: squared wrapper robustness for non-monic
    `solve_prep`
  - `secondary_dimension`: bare `A^2 - B^2` residual after passthrough
    cancellation
  - `cohesion_scope`: narrow exact-zero fallback in `arithmetic.rs`
  - `behavior_change_expected`: yes
- trigger:
  - `solve_prep` was the last count-2 `squared_passthrough_zero` family
  - the leading-coefficient complete-square candidate had already been
    documented as a structural miss under squared passthrough
- baseline probe:
  - candidate row:
    `(((a*x^2 + b*x + c)^2) + m) - (((a*(x + b/(2*a))^2 + c - b^2/(4*a))^2) + m) -> 0`
  - smoke before change: `0/1`, `status=fail`, runner elapsed `290.95ms`
  - negative-linear sibling probe also failed before this change:
    `0/1`, runner elapsed `239.15ms`
- change:
  - extended the difference-of-equivalent-square-bases helper with a fallback
    that proves the base residual using `try_build_exact_zero_identity_rewrite_direct`
  - added focused unit coverage for the exposed `A^2 - B^2` residual and the
    full squared-passthrough wrapper
  - promoted one live corpus row for the non-monic leading-coefficient
    complete-square representative
  - closed the prior pending `solve_prep` discovery section as resolved
- candidate smoke:
  - after change: `1/1` passed
  - runner elapsed `19.00ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `20`
- filtered result:
  - `squared_passthrough_zero --family solve_prep`: `3/3`
  - elapsed `12.51ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - unit tests:
    - `collapse_exact_zero_additive_subexpression_matches_solve_prep_square_difference`
    - `collapse_exact_zero_additive_subexpression_matches_solve_prep_squared_passthrough`
  - `make engine-fast`: passed
    - unit smoke/scorecard tests: `20/20`
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `223.12ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1416/1416`
    - average case runtime `3.305ms`
    - `squared_passthrough_zero`: `71/71`
    - sparse bucket: `squared_passthrough_zero x solve_prep total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`, `0` timeouts
- decision:
  - retained as robustness plus one minimal live promotion
  - all `squared_passthrough_zero` families now have at least three live cases
    with zero embedded failures
- next:
  - with sparse squared-wrapper family counts closed, prefer an observability
    or runtime iteration over further corpus growth unless a new structural
    miss appears in the generated discovery ledger

### 2026-04-27: Squared Passthrough Dirichlet Discovery Closure

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x integrate_prep`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: Dirichlet kernel semantic subfamily under squared
    passthrough
  - `secondary_dimension`: stale observe-only discovery closure
  - `behavior_change_expected`: no engine code change; prior square-base
    fallback work already unlocked the candidate
- trigger:
  - `Integrate Prep Dirichlet Squared Discovery` was the only remaining
    observe-only discovery in the generated scorecard
  - rerunning both originally failed squared orientations showed that the
    current engine now proves them
- probes:
  - direct candidate smoke:
    - `integrate_prep_dirichlet_basic_squared`: `1/1`
    - runner elapsed `15.18ms`
  - reverse candidate smoke:
    - `integrate_prep_dirichlet_reverse_squared`: `1/1`
    - runner elapsed `3.34ms`
- change:
  - promoted one minimal live row:
    `integrate_prep_dirichlet_basic_squared`
  - marked the prior Dirichlet squared discovery as `resolved`
  - did not promote the reverse row because it is now confirmed by smoke and
    would mainly duplicate the same squared Dirichlet subfamily in live corpus
- filtered result:
  - `squared_passthrough_zero --family integrate_prep`: `4/4`
  - elapsed `3.82ms`
- guardrails:
  - `make engine-fast`: passed
    - unit smoke/scorecard tests: `20/20`
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `219.47ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1417/1417`
    - average case runtime `3.169ms`
    - `squared_passthrough_zero`: `72/72`
    - sparse bucket: `squared_passthrough_zero x integrate_prep total=4 failed=0`
    - observe-only discoveries: `0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`, `0` timeouts
- decision:
  - retained as coverage
  - this closes the last generated discovery without changing runtime code
- next:
  - with generated discoveries at zero, prefer an observability/runtime cycle
    over more live corpus growth unless a fresh structural miss appears

### 2026-04-27: Direct Small-Zero Pair Flag-Scan Cache Rejection

- area:
  - [orchestrator.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs)
  - `root.addsub.00.direct_small_zero_pair`
  - `root.direct_small_zero_composition.candidate.*`
- status:
  - `rejected`
- investment:
  - `investment_class`: runtime
  - `primary_dimension`: direct-small-zero no-match cost
  - `secondary_dimension`: scanner reuse / cheap-gate discipline
  - `behavior_change_expected`: no semantic behavior change
- trigger:
  - the profiled embedded slice showed persistent no-match cost:
    - `root.addsub.00.direct_small_zero_pair`: `905` attempts, `895` misses
    - `root.addsub.00.direct_small_zero_pair.compact_first`: `321` attempts,
      `296` misses
    - `root.direct_small_zero_composition.candidate.three_core_groups`: `75`
      attempts, `56` misses
- rejected candidate:
  - computing `HotDirectSmallZeroFamilyFlags` for both sides up front in
    `matches_direct_small_zero_pair_root`
  - a narrower variant that replaced only the log/division partner-side scans
    with one flag scan
- focused result:
  - baseline filtered profile:
    - total profiled time `175.419ms`
    - add/sub direct route `46.597ms`
    - compact direct route `40.722ms`
  - up-front flag-cache attempt:
    - total profiled time `191.111ms`
    - add/sub direct route `50.616ms`
    - compact direct route `44.505ms`
  - narrow log/division scan-cache attempt:
    - repeat totals `190.931ms` and `184.403ms`
    - still no retained local win over baseline
- guardrails:
  - targeted direct-small-zero tests stayed green:
    - `cargo test -q -p cas_engine direct_small_zero_pair_shortcut -- --nocapture`
      `28/28`
  - `cargo fmt -- --check`: passed
  - `git diff --check`: passed
- decision:
  - rejected and reverted; no runtime code retained
  - the evidence says not to scan/cache broad flags before a narrower candidate
    gate proves they are needed
- next:
  - prefer a more specific gate for the actual `three_core_groups` miss shapes
    or richer observability of the full expression/context before trying more
    broad scan reuse in direct-small-zero pair matching

### 2026-04-29: Derive Power-Merge Quotient Route Capture

- area:
  - [power_merge.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/power_merge.rs)
  - [derive_contract_tests.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_contract_tests.rs)
  - embedded root `merge_same_base_symbolic_quotient_powers`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: derive bridge for same-base quotient powers
  - `secondary_dimension`: target-form/orientation coverage for quotient powers
  - `behavior_change_expected`: `derive` should route through `combine powers`
    instead of the generic `simplify` fallback
- trigger:
  - `derive x^a/x^b, x^(a-b)` and `derive 2^a/2^b, 2^(a-b)` both succeed,
    but the CLI reports `Strategy: simplify`
  - this means the simplifier knows the algebra, while the target-aware derive
    bridge does not expose it as a specific power-merge move
- hypothesis:
  - `power_merge` handles n-ary multiplication and canonicalized roots, but its
    target-aware provider does not inspect explicit `Div` quotient shapes
  - adding a bounded same-base quotient rewrite before the multiplication-only
    path should classify the target as `PowerMerged` and select `PowerMerge`
    before `Simplify`
- success condition:
  - `derive x^a/x^b, x^(a-b)` reports `Strategy: combine powers`
  - exact shadow pressure includes
    `embedded_power_merge_symbolic_quotient_powers`
  - sampled shadow pressure becomes `49/49` with `generic_simplify_strategy_successes=0`
  - `make engine-fast` and `make engine-scorecard` stay green
- reject condition:
  - the only successful path remains generic `simplify`
  - quotient handling needs broad normalization outside a bounded target-aware
    provider
  - equivalence or runtime guardrails regress
- local result:
  - added a bounded explicit-division branch to the target-aware power-merge
    provider
  - added unit coverage for `x^a/x^b -> x^(a-b)` and `2^a/2^b -> 2^(a-b)`
  - promoted `embedded_power_merge_symbolic_quotient_powers` into derive shadow
    pressure
  - CLI probe now reports `Strategy: combine powers` for
    `derive x^a/x^b, x^(a-b)`
- guardrails:
  - target-aware power merge unit: passed
  - exact derive shadow pressure: `sampled=49 derived=49 unsupported=0 not_equivalent=0`
  - generic simplify shadow successes: `0`
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1417/1417`
    - `derive_contract`: `319` derived, `0` unsupported, `1` expected
      not-equivalent case
    - `simplify_strict`: `16518/16518`, `0` failures/timeouts
- decision:
  - retained as coverage plus a small derive-runtime bridge improvement
  - this is not merely corpus growth; the visible derive route no longer
    depends on generic `simplify` for the representative quotient-power root

### 2026-04-29: Derive Log10 Power-Alias Route Capture

- area:
  - [exponentials.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/exponentials.rs)
  - [derive_contract_tests.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_contract_tests.rs)
  - embedded root `log10_power_alias`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: derive bridge for base-aware exponential/log inverse
  - `secondary_dimension`: log10/general-base orientation for `log_exp_inverse`
  - `behavior_change_expected`: `derive` should route through
    `rewrite exponentials` instead of generic `simplify`
- structural_axis:
  - non-`e` base power alias, specifically `10^(y*log10(x)) -> x^y`
- why_this_is_not_a_duplicate:
  - the existing retained shadow covers natural `exp(y*log(x)) -> x^y`
  - this root uses an explicit numeric base and unary `log10`, which exercises
    the generalized `b^(c*log_b(x))` engine capability
- discovery_or_promotion:
  - discovered by probing an embedded root that already simplifies correctly
    but currently reports `Strategy: simplify` in `derive`
- if_promoted_why_minimal_representative:
  - naked root from the embedded corpus, no wrapper and no extra factors
- trigger:
  - `derive 10^(y*log10(x)), x^y` succeeds only through `Strategy: simplify`
  - `derive exp(y*log(x)), x^y` already succeeds through
    `Strategy: rewrite exponentials`
- hypothesis:
  - the derive exponential provider has a natural-log-specific extractor while
    `cas_math::logarithm_inverse_support` already exposes a generalized
    exponential-log inverse rewrite for `b^(c*log(b,x))`
  - reusing that bounded support in the target-aware provider should cover
    log10 and general-base cases without widening search
- success condition:
  - `derive 10^(y*log10(x)), x^y` reports `Strategy: rewrite exponentials`
  - exact shadow pressure includes `embedded_log10_power_alias`
  - sampled shadow pressure becomes `50/50` with
    `generic_simplify_strategy_successes=0`
- relevant_lanes:
  - exponential target-aware unit
  - CLI derive probe
  - derive shadow pressure
  - engine-fast
  - engine-scorecard
- promotion_target:
  - exact derive shadow pressure case, not live embedded growth
- derive_bridge_check:
  - visible route must be `rewrite exponentials` and carry the same positive
    subject requirement as the engine rewrite
- engine_feedback_check:
  - if generalized support cannot be reused directly, defer and record the
    gap rather than adding a one-off derive-only hack
- retain_if:
  - unit and CLI prove the route is target-aware, shadow pressure stays generic
    simplify free, and guardrails stay green
- reject_if:
  - the case still reaches only through `simplify`, introduces broad search, or
    regresses embedded/derive guardrails
- local result:
  - reused
    `cas_math::logarithm_inverse_support::try_rewrite_exponential_log_inverse_expr`
    inside the target-aware derive exponential provider
  - generalized the visible description from natural `exp(k*log(u))` to
    `b^(k*log_b(u))`
  - added target-aware unit coverage for `10^(y*log10(x)) -> x^y` and
    `b^(c*log(b,x)) -> x^c`
  - promoted `embedded_log10_power_alias` into derive shadow pressure
  - CLI probes now report `Strategy: rewrite exponentials` and preserve
    `Requires: x > 0`
- guardrails:
  - generalized exponential-log inverse unit: passed
  - exact derive shadow pressure:
    `sampled=50 derived=50 unsupported=0 not_equivalent=0`
  - generic simplify shadow successes: `0`
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1417/1417`
    - `derive_contract`: `319` derived, `0` unsupported, `1` expected
      not-equivalent case
    - `derive_shadow_pressure`: `50/50`
    - `simplify_strict`: `16518/16518`, `0` failures/timeouts
- decision:
  - retained as derive bridge coverage
  - no engine runtime change was needed because the reusable engine capability
    already existed; the gap was target-aware derive exposure

### 2026-04-29: Derive Log10 Power-Alias Contract Promotion Capture

- area:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - `log_exp_inverse`
  - embedded shadow root `embedded_log10_power_alias`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: curated derive contract coverage for base-aware
    exponential/log inverse
  - `secondary_dimension`: didactic audit pressure for the visible
    `rewrite exponentials` step
  - `behavior_change_expected`: no runtime behavior change; the existing
    target-aware route should become part of the stable derive contract
- structural_axis:
  - non-`e` base power alias:
    `10^(y*log10(x)) -> x^y`
- why_this_is_not_a_duplicate:
  - `derive_pairs.csv` currently has only `ln(exp(x)) -> x` in
    `log_exp_inverse`
  - shadow pressure already proves the log10 alias, but shadow cases are
    diagnostic; promoting one naked root makes the scorecard and didactic audit
    track this family permanently
- discovery_or_promotion:
  - promotion of a retained shadow/root behavior after the previous runtime
    bridge removed the generic `simplify` fallback
- if_promoted_why_minimal_representative:
  - naked root from the embedded corpus, no wrapper and no unrelated algebraic
    families
- trigger:
  - CLI now reports `Strategy: rewrite exponentials` for
    `derive 10^(y*log10(x)), x^y`
  - curated contract still under-represents `log_exp_inverse` as only the
    natural inverse case
- hypothesis:
  - adding the single log10 alias to the curated derive corpus should increase
    durable `derive_contract` and `derive_didactic_audit` coverage without
    changing runtime or accepting a magical generic-simplify route
- success condition:
  - `derive_contract` accepts the row with expected strategy
    `rewrite exponentials`
  - `derive_didactic_audit` remains flag-free and now audits two
    `log_exp_inverse` cases
  - `make engine-fast` and `make engine-scorecard` stay green
- relevant_lanes:
  - CSV contract row
  - CLI derive probe
  - derive contract test
  - derive didactic audit
  - engine-fast
  - engine-scorecard
- promotion_target:
  - `derive_pairs.csv`, not embedded corpus growth
- derive_bridge_check:
  - the row is only valid if the visible strategy remains
    `rewrite exponentials`, not `simplify`
- engine_feedback_check:
  - if the contract row fails or creates audit flags, do not weaken assertions;
    keep the weakness recorded and return to runtime/didactic repair
- retain_if:
  - all relevant lanes pass and the scorecard shows one additional curated
    derived case
- reject_if:
  - the case routes through generic `simplify`, requires relaxed expectations,
    or destabilizes didactic rendering/audit output
- local result:
  - added
    `log_exp_inverse_log10_power_alias,log_exp_inverse,10^(y*log10(x)),x^y,derived,rewrite exponentials`
    to the curated derive contract
  - the first didactic audit run correctly exposed a weakness: the new
    `Exponential-Log Power Inverse` web step had no structured substeps
  - added focused web substeps that explain the transformation as
    `b^(k*log_b(u)) = (b^log_b(u))^k -> u^k`
  - added a didactic regression test requiring the log10 alias to show:
    - `Usar que 10^(log10(u)) = u`
    - `Aplicar el factor exterior como exponente`
- guardrails:
  - CLI probe:
    `derive 10^(y*log10(x)), x^y` reports
    `Strategy: rewrite exponentials`
  - derive generic-simplify guard: passed
  - targeted didactic test for the log10 power alias: passed
  - derive didactic audit: `404` cases, `0` flags, `424` web substeps
  - `cargo fmt -- --check`: passed
  - `git diff --check`: passed
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1417/1417`
    - `derive_contract`: `320` derived, `0` unsupported, `1` expected
      not-equivalent case
    - `derive_shadow_pressure`: `50/50`, generic simplify successes `0`
    - `derive_didactic_audit`: `404` cases, `0` flags
    - `simplify_strict`: `16518/16518`, `0` failures/timeouts
- decision:
  - retained as coverage plus didactic robustness
  - no mathematical runtime rewrite was added in this cycle; the runtime route
    already existed, and the cycle made it permanently measured and no longer
    opaque in web didactic output

### 2026-04-29: Derive Quotient Power-Merge Contract Promotion Capture

- area:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [focused_rule_substeps.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs)
  - `power_merge`
  - embedded shadow root `embedded_power_merge_symbolic_quotient_powers`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: curated derive coverage for same-base quotient powers
  - `secondary_dimension`: didactic explanation for quotient-to-negative-power
    orientation
  - `behavior_change_expected`: no runtime rewrite change; the existing
    target-aware `combine powers` route should become permanently measured and
    less opaque
- structural_axis:
  - quotient orientation for same-base powers:
    `x^a/x^b -> x^(a-b)`
- why_this_is_not_a_duplicate:
  - `derive_pairs.csv` covers product-oriented power merge
    `x^a*x^b -> x^(a+b)` and root/fractional variants
  - it does not cover the quotient orientation that subtracts exponents, even
    though shadow pressure already proves it through `combine powers`
- discovery_or_promotion:
  - promotion of a retained shadow/root behavior after the previous
    target-aware power-merge bridge removed the generic `simplify` fallback
- if_promoted_why_minimal_representative:
  - naked symbolic quotient root, no wrapper, no numeric-base duplicate, no
    extra surrounding algebra
- trigger:
  - CLI reports `Strategy: combine powers` for
    `derive x^a/x^b, x^(a-b)`
  - curated `power_merge` coverage is still product-heavy and misses the
    subtraction-of-exponents target form
- hypothesis:
  - promoting the minimal quotient row should increase durable
    `derive_contract` coverage without changing engine runtime
  - adding focused quotient substeps should avoid a magical one-step display by
    showing `x^a/x^b -> x^a*x^(-b) -> x^(a-b)`
- success_condition:
  - `derive_contract` accepts the new row with expected strategy
    `combine powers`
  - the web/JSON derive audit reports the quotient row with two meaningful
    substeps and no flags
  - `make engine-fast` and `make engine-scorecard` stay green
- relevant_lanes:
  - CLI derive probe
  - derive contract row
  - focused derive didactic test
  - derive didactic audit
  - engine-fast
  - engine-scorecard
- promotion_target:
  - `derive_pairs.csv`
- derive_bridge_check:
  - the row is valid only if the visible strategy remains `combine powers`,
    not generic `simplify`
- engine_feedback_check:
  - no reusable engine gap is expected; if the row fails, return to the
    power-merge provider instead of adding a derive-only exception
- retain_if:
  - the quotient row is derived specifically, audit stays flag-free, and
    global scorecard metrics remain green
- reject_if:
  - the case falls back to generic `simplify`, the didactic substeps duplicate
    the parent step, or embedded/derive guardrails regress
- local result:
  - added
    `merge_same_base_symbolic_quotient_powers,power_merge,x^a/x^b,x^(a-b),derived,combine powers`
    to the curated derive contract
  - added quotient-specific web/JSON substeps for same-base power merge:
    - `Reescribir la división como potencia negativa`
    - `Sumar los exponentes de la misma base`
  - added a focused didactic regression test for the quotient row
  - CLI probe still reports `Strategy: combine powers` for
    `derive x^a/x^b, x^(a-b)`
- guardrails:
  - derive generic-simplify guard: passed
  - targeted didactic quotient test: passed
  - release derive contract: `321` derived, `0` unsupported, `1` expected
    not-equivalent case
  - derive didactic audit: `405` cases, `0` flags, `426` web substeps
  - `cargo fmt -- --check`: passed
  - `git diff --check`: passed
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1417/1417`
    - `derive_contract`: `321` derived, `0` unsupported, `1` expected
      not-equivalent case
    - `derive_shadow_pressure`: `50/50`, generic simplify successes `0`
    - `derive_didactic_audit`: `405` cases, `0` flags
    - `simplify_strict`: `16518/16518`, `0` failures/timeouts
- decision:
  - retained as coverage plus didactic robustness
  - no engine runtime rewrite was needed in this cycle; it promotes an existing
    target-aware power-merge route and makes the quotient orientation
    explainable in web/JSON output

### 2026-04-29: Derive Log-Inverse-Power Natural Alias Promotion Capture

- area:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
  - `log_inverse_power`
  - embedded shadow root `embedded_log_inverse_power_unary_natural_alias`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: curated derive coverage for natural-log alias handling
    inside log-inverse-power
  - `secondary_dimension`: didactic audit pressure for nested functional target
    recovery
  - `behavior_change_expected`: no runtime behavior change; the existing
    target-aware route should become part of the stable derive contract
- structural_axis:
  - unary natural-log alias and nested target:
    `x^(log(log(x))/log(x)) -> log(x)`
- why_this_is_not_a_duplicate:
  - the existing curated row is `x^(ln(y)/ln(x)) -> y`
  - this one exercises parser-level `log` alias normalization and a nested
    function target recovered by the same `log inverse power` path
- discovery_or_promotion:
  - promotion of a shadow/root behavior already proven by
    `derive_shadow_pressure`
- if_promoted_why_minimal_representative:
  - naked embedded root with no wrapper, no numeric constants, and no unrelated
    algebraic families
- trigger:
  - CLI reports `Strategy: log inverse power` for
    `derive x^(log(log(x))/log(x)), log(x)`
  - curated `log_inverse_power` coverage still has only one row and does not
    cover the natural `log` alias spelling
- hypothesis:
  - adding one contract row will make alias/nested-target support durable
    without adding search or runtime cost
  - the existing didactic substeps should remain concrete enough to avoid audit
    flags
- success_condition:
  - `derive_contract` accepts the row with expected strategy
    `log inverse power`
  - `derive_didactic_audit` remains flag-free and audits two
    `log_inverse_power` cases
  - `make engine-fast` and `make engine-scorecard` stay green
- relevant_lanes:
  - CLI derive probe
  - derive contract row
  - focused derive didactic test
  - derive didactic audit
  - engine-fast
  - engine-scorecard
- promotion_target:
  - `derive_pairs.csv`
- derive_bridge_check:
  - the row is valid only if the visible strategy remains
    `log inverse power`, not generic `simplify`
- engine_feedback_check:
  - no reusable engine gap is expected; if the row fails, inspect the
    log-inverse-power provider rather than adding a derive-only exception
- retain_if:
  - the row is derived specifically, audit stays flag-free, and scorecard
    guardrails stay green
- reject_if:
  - the case routes through generic `simplify`, requires relaxed expectations,
    or destabilizes didactic rendering/audit output
- local result:
  - added
    `log_inverse_power_unary_natural_alias,log_inverse_power,x^(log(log(x))/log(x)),log(x),derived,log inverse power`
    to the curated derive contract
  - added a focused didactic regression for the alias/nested-target case
  - CLI reports `Strategy: log inverse power` and preserves
    `Requires: x > 0`
- guardrails:
  - derive generic-simplify guard: passed
  - targeted didactic alias test: passed
  - release derive contract: `322` derived, `0` unsupported, `1` expected
    not-equivalent case
  - derive didactic audit: `406` cases, `0` flags, `428` web substeps
  - `cargo fmt -- --check`: passed
  - `git diff --check`: passed
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1417/1417`
    - `derive_contract`: `322` derived, `0` unsupported, `1` expected
      not-equivalent case
    - `derive_shadow_pressure`: `50/50`, generic simplify successes `0`
    - `derive_didactic_audit`: `406` cases, `0` flags
    - `simplify_strict`: `16518/16518`, `0` failures/timeouts
- decision:
  - retained as coverage plus didactic guardrail
  - no engine runtime rewrite was needed; the cycle promotes the existing
    target-aware log-inverse-power route and makes alias/nested-target recovery
    permanently measured

### 2026-04-29: Derive Exponential-Log Natural Power Alias Promotion Capture

- area:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
  - `log_exp_inverse`
  - embedded shadow root `embedded_log_exp_inverse_power_alias`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: curated derive coverage for natural-log alias handling
    in exponential-log power inversion
  - `secondary_dimension`: didactic pressure for the symbolic outer exponent
    after inverse cancellation
  - `behavior_change_expected`: no runtime behavior change; promote an
    existing shadow-supported route into the stable derive contract
- structural_axis:
  - natural-base exponential with symbolic multiplier:
    `exp(y*log(x)) -> x^y`
- why_this_is_not_a_duplicate:
  - the existing curated `log_exp_inverse` rows cover `ln(exp(x))` and
    base-10 `10^(y*log10(x)) -> x^y`
  - this case exercises the natural `log` alias spelling through `exp(...)`,
    not `10^...`, and preserves the symbolic outer exponent
- discovery_or_promotion:
  - promotion of a shadow/root behavior already proven by
    `derive_shadow_pressure`
- if_promoted_why_minimal_representative:
  - naked embedded root with one base, one symbolic outer exponent, no wrapper,
    and no unrelated product/log-sum composition
- trigger:
  - CLI reports `Strategy: rewrite exponentials` for
    `derive exp(y*log(x)), x^y`
  - curated `log_exp_inverse` coverage remains small and currently lacks the
    natural `log` power-alias spelling
- hypothesis:
  - adding one contract row will make the natural `exp(k*log(u)) -> u^k`
    route durable without adding search or runtime cost
  - existing didactic substeps should explain both inverse cancellation and the
    outer exponent without audit flags
- success_condition:
  - `derive_contract` accepts the row with expected strategy
    `rewrite exponentials`
  - focused didactic regression sees the inverse-cancel and outer-exponent
    substeps for the natural alias case
  - `make engine-fast` and `make engine-scorecard` stay green
- relevant_lanes:
  - CLI derive probe
  - derive contract row
  - focused derive didactic test
  - derive didactic audit
  - engine-fast
  - engine-scorecard
- promotion_target:
  - `derive_pairs.csv`
- derive_bridge_check:
  - the row is valid only if the visible strategy remains
    `rewrite exponentials`, not generic `simplify`
- engine_feedback_check:
  - no reusable engine gap is expected; if the row fails, inspect the
    exponential-log inverse provider rather than adding a derive-only exception
- retain_if:
  - the row is derived specifically, the natural alias substeps render
    concretely, and scorecard guardrails stay green
- reject_if:
  - the case routes through generic `simplify`, loses the domain requirement,
    requires relaxed expectations, or destabilizes didactic audit output
- local result:
  - added
    `log_exp_inverse_natural_log_power_alias,log_exp_inverse,exp(y*log(x)),x^y,derived,rewrite exponentials`
    to the curated derive contract
  - added a focused didactic regression for the natural `exp(y*log(x))`
    alias case
  - CLI reports `Strategy: rewrite exponentials` and preserves
    `Requires: x > 0`
- guardrails:
  - derive generic-simplify guard: passed
  - targeted didactic natural alias test: passed
  - release derive contract: `323` derived, `0` unsupported, `1` expected
    not-equivalent case
  - derive didactic audit: `407` cases, `0` flags, `430` web substeps
  - `cargo fmt -- --check`: passed
  - `git diff --check`: passed
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1417/1417`
    - `derive_contract`: `323` derived, `0` unsupported, `1` expected
      not-equivalent case
    - `derive_shadow_pressure`: `50/50`, generic simplify successes `0`
    - `derive_didactic_audit`: `407` cases, `0` flags
    - `simplify_strict`: `16518/16518`, `0` failures/timeouts
- decision:
  - retained as coverage plus didactic guardrail
  - no engine runtime rewrite was needed; this promotes the existing
    exponential-log inverse route for natural `log` alias spelling and makes
    symbolic outer-exponent recovery permanently measured

### 2026-04-29: Derive Nested-Fraction Reciprocal Base Promotion Capture

- area:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
  - [inner_fraction.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/nested_fractions/general/inner_fraction.rs)
  - `nested_fraction`
  - identity shadow root `identity_nested_fraction_reciprocal_inverse`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: curated derive coverage for the base reciprocal
    nested-fraction identity
  - `secondary_dimension`: didactic substep quality for a general
    nested-fraction denominator with no inner sum
  - `behavior_change_expected`: no engine runtime behavior change; web/JSON
    didactic output should gain concrete substeps for `n/(p/q)`-style nested
    fractions when the inner fraction is the direct denominator
- structural_axis:
  - pure nested reciprocal simplification: `1/(1/x) -> x`
- why_this_is_not_a_duplicate:
  - existing curated `nested_fraction` rows cover denominator sums and scalar
    division forms, but not the base reciprocal-in-denominator form
  - the generated audit already covers several structured nested fractions, yet
    the general inner-fraction substep provider is currently empty
- discovery_or_promotion:
  - promotion of an identity-pair shadow behavior already reported as derived
    by `derive_shadow_pressure`
- if_promoted_why_minimal_representative:
  - naked root with one reciprocal layer, one symbol, no additive denominator,
    and no extra scalar/noise wrapper
- trigger:
  - CLI reports `Strategy: nested fraction` for `derive 1/(1/x), x`
  - `identity_pairs.csv` contains the same base identity, but the curated
    derive contract lacks this representative
- hypothesis:
  - adding the row will make the base nested reciprocal route durable
  - adding concrete didactic substeps for direct inner-fraction denominators
    will prevent the promoted row from becoming an opaque one-step jump
- success_condition:
  - `derive_contract` accepts the row with expected strategy
    `nested fraction`
  - focused didactic regression sees inverse-and-cleanup substeps for
    `nested_fraction_reciprocal_inverse`
  - `derive_didactic_audit` remains flag-free
  - `make engine-fast` and `make engine-scorecard` stay green
- relevant_lanes:
  - CLI derive probe
  - derive contract row
  - focused derive didactic test
  - derive didactic audit
  - engine-fast
  - engine-scorecard
- promotion_target:
  - `derive_pairs.csv`
- derive_bridge_check:
  - the row is valid only if the visible strategy remains `nested fraction`,
    not generic `simplify`
- engine_feedback_check:
  - no reusable engine gap is expected; if the row fails, inspect the
    nested-fraction transition provider rather than adding a derive-only
    exception
- retain_if:
  - the row is derived specifically, the new substeps are concrete and
    non-duplicative, and global guardrails stay green
- reject_if:
  - the case routes through generic `simplify`, emits noisy template substeps,
    duplicates the parent snapshot, or destabilizes didactic audit output
- local result:
  - added
    `nested_fraction_reciprocal_inverse,nested_fraction,1/(1/x),x,derived,nested fraction`
    to the curated derive contract
  - implemented concrete web/JSON substeps for direct `n/(p/q)`
    nested-fraction denominators
  - added a focused didactic regression for the reciprocal base case
  - CLI reports `Strategy: nested fraction`
- guardrails:
  - derive generic-simplify guard: passed
  - targeted didactic reciprocal inverse test: passed
  - release derive contract: `324` derived, `0` unsupported, `1` expected
    not-equivalent case
  - derive didactic audit: `408` cases, `0` flags, `432` web substeps
  - simplify didactic audit: `14` cases, `0` flags, `26` wire substeps
  - `cargo fmt -- --check`: passed
  - `git diff --check`: passed
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1417/1417`
    - `derive_contract`: `324` derived, `0` unsupported, `1` expected
      not-equivalent case
    - `derive_shadow_pressure`: `50/50`, generic simplify successes `0`
    - `derive_didactic_audit`: `408` cases, `0` flags
    - `simplify_didactic_audit`: `14` cases, `0` flags, `26` wire substeps
    - `simplify_strict`: `16518/16518`, `0` failures/timeouts
- decision:
  - retained as coverage plus didactic quality improvement
  - no engine runtime rewrite was needed; the cycle promotes an existing
    nested-fraction transition and makes the direct inner-fraction denominator
    explanation reusable for derive and simplify traces

### 2026-04-29: Derive Cube-Root Rationalization Promotion Capture

- area:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [rationalization.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/rationalization.rs)
  - `rationalize`
  - identity shadow root `identity_cube_root_rationalization`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: curated derive coverage for rationalizing a
    cube-root sum denominator
  - `secondary_dimension`: didactic substep quality for the hidden sum-of-cubes
    denominator rewrite
  - `behavior_change_expected`: no engine runtime behavior change; web/JSON
    didactic output should expose the cubic conjugate and `u^3 + 1`
    denominator collapse
- structural_axis:
  - rationalization with fractional-power denominator:
    `1/(1 + x^(1/3)) -> (1 - x^(1/3) + x^(2/3))/(1 + x)`
- why_this_is_not_a_duplicate:
  - existing rationalize rows cover linear square-root conjugates and a
    radical notable quotient, but not denominator rationalization by a cubic
    conjugate
  - the case already appears in `identity_pairs.csv` and shadow pressure, yet
    it is not a stable derive contract row
- discovery_or_promotion:
  - promotion of an identity-pair shadow behavior already reported as derived
    by `derive_shadow_pressure`
- if_promoted_why_minimal_representative:
  - one reciprocal, one cube-root term, no extra scalar, no passthrough, no
    wrapper noise
- trigger:
  - CLI reports `Strategy: rationalize` for
    `derive 1/(1+x^(1/3)), (1-x^(1/3)+x^(2/3))/(1+x)`
  - current CLI step is a single rationalization move; the didactic layer should
    expose the hidden cubic conjugate rather than leave the jump opaque
- hypothesis:
  - adding one curated row makes the cube-root rationalization route durable
  - adding didactic substeps for `1/(1+u)` with `u^3=x` makes the same
    rationalization explanation reusable outside `derive`
- success_condition:
  - `derive_contract` accepts the row with expected strategy `rationalize`
  - focused didactic regression sees cubic-conjugate and sum-of-cubes substeps
    for the promoted row
  - `derive_didactic_audit` remains flag-free
  - `make engine-fast` and `make engine-scorecard` stay green
- relevant_lanes:
  - CLI derive probe
  - derive contract row
  - focused derive didactic test
  - derive didactic audit
  - engine-fast
  - engine-scorecard
- promotion_target:
  - `derive_pairs.csv`
- derive_bridge_check:
  - the row is valid only if the visible strategy remains `rationalize`, not
    generic `simplify`
- engine_feedback_check:
  - no reusable runtime gap is expected; if the row fails, inspect the
    rationalization provider or target classifier before adding a derive-only
    exception
- retain_if:
  - the row is derived specifically, cubic substeps are concrete and
    non-duplicative, and global guardrails stay green
- reject_if:
  - the case routes through generic `simplify`, substeps only restate the parent
    rationalization, or the didactic audit destabilizes
- local result:
  - added
    `rationalize_cube_root_sum_denominator,rationalize,1/(1+x^(1/3)),(1-x^(1/3)+x^(2/3))/(1+x),derived,rationalize`
    to the curated derive contract
  - added a cube-root denominator rationalization substep provider for
    `1/(1+u)` with `u = x^(1/3)`
  - mapped `Rationalize Cube Root Denominator` to the public Spanish rule
    `Racionalizar el denominador`
  - added a focused derive didactic regression requiring:
    - `Multiplicar por el conjugado cúbico`
    - `Aplicar suma de cubos en el denominador`
  - the first focused didactic run exposed the expected weakness: the step fell
    through to the square-root conjugate narrative before the cubic provider was
    selected
- guardrails:
  - CLI probe:
    `derive 1/(1+x^(1/3)), (1-x^(1/3)+x^(2/3))/(1+x)` reports
    `Strategy: rationalize`
  - derive generic-simplify guard: passed
  - targeted didactic cube-root rationalization test: passed
  - release derive contract: `325` derived, `0` unsupported, `1` expected
    not-equivalent case
  - derive didactic audit: `409` cases, `0` flags, `434` web substeps
  - simplify didactic audit: `14` cases, `0` flags, `26` wire substeps
  - `cargo fmt -- --check`: passed
  - `git diff --check`: passed
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1417/1417`, avg case `2.555ms`
    - `derive_contract`: `325` derived, `0` unsupported, `1` expected
      not-equivalent case
    - `derive_shadow_pressure`: `50/50`, generic simplify successes `0`
    - `derive_didactic_audit`: `409` cases, `0` flags
    - `simplify_didactic_audit`: `14` cases, `0` flags, `26` wire substeps
    - `simplify_strict`: `16518/16518`, `0` failures/timeouts
- decision:
  - retained as coverage plus didactic quality improvement
  - no engine runtime rewrite was needed; the cycle promotes an existing
    rationalization capability and makes the hidden cubic-conjugate step
    explicit for derive/web traces

### 2026-04-29: Derive Nested-Radical Denesting Promotion Capture

- area:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
  - `rewrite radicals`
  - identity shadow root `identity_nested_radical_denesting`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: curated derive coverage for denesting a nested radical
  - `secondary_dimension`: didactic path quality for a two-step radical route
    through an absolute-value cleanup
  - `behavior_change_expected`: no engine runtime behavior change; promote an
    existing shadow-supported radical route into the stable derive contract
- structural_axis:
  - nested radical denesting:
    `sqrt(6 + 2*sqrt(5)) -> sqrt(5) + 1`
- why_this_is_not_a_duplicate:
  - existing curated radical rows cover perfect-square roots, root powers, and
    odd half-powers, but not a nested radical that first becomes
    `|sqrt(5)+1|`
  - the identity appears in `identity_pairs.csv` and in shadow pressure, yet it
    is not a stable derive contract row
- discovery_or_promotion:
  - promotion of an identity-pair shadow behavior already reported as derived
    by `derive_shadow_pressure`
- if_promoted_why_minimal_representative:
  - one nested square root, one numeric inner radical, no passthrough, no
    symbolic parameters, and no extra wrapper
- trigger:
  - CLI reports `Strategy: rewrite radicals` for
    `derive sqrt(6 + 2*sqrt(5)), sqrt(5)+1`
  - the route has two visible steps, so it adds path-quality coverage rather
    than only another one-step simplification
- hypothesis:
  - adding one curated row makes the denesting route durable in `derive`
  - a focused didactic regression will keep the hidden perfect-square
    recognition visible instead of letting the radical denesting look magical
- success_condition:
  - `derive_contract` accepts the row with expected strategy `rewrite radicals`
  - focused didactic regression sees concrete perfect-square-root substeps for
    `nested_radical_denesting`
  - `derive_didactic_audit` remains flag-free
  - `make engine-fast` and `make engine-scorecard` stay green
- relevant_lanes:
  - CLI derive probe
  - derive contract row
  - focused derive didactic test
  - derive didactic audit
  - engine-fast
  - engine-scorecard
- promotion_target:
  - `derive_pairs.csv`
- derive_bridge_check:
  - the row is valid only if the visible strategy remains `rewrite radicals`,
    not generic `simplify`
- engine_feedback_check:
  - no reusable runtime gap is expected; if the row fails, inspect the radical
    transition provider or target classifier before adding a derive-only
    exception
- retain_if:
  - the row is derived specifically, the radical substeps are concrete, and
    global guardrails stay green
- reject_if:
  - the case routes through generic `simplify`, emits opaque radical substeps,
    or destabilizes didactic audit output
- local_result:
  - added
    `nested_radical_denesting,simplify,sqrt(6 + 2*sqrt(5)),sqrt(5)+1,derived,rewrite radicals`
    to the curated derive contract
  - added a focused derive didactic regression requiring the
    perfect-square-root step to expose both hidden substeps:
    `Reescribir el radicando como un cuadrado perfecto` and
    `La raíz de un cuadrado da un valor absoluto`
  - CLI reports `Strategy: rewrite radicals` and preserves the two-step path
    through `|sqrt(5)+1|`
- guardrails:
  - CLI derive probe: passed
  - derive generic-simplify guard: passed
  - targeted didactic nested-radical test: passed
  - release derive contract: `326` derived, `0` unsupported, `1` expected
    not-equivalent case
  - derive didactic audit: `410` cases, `0` flags, `436` web substeps
  - simplify didactic audit: `14` cases, `0` flags, `26` wire substeps
  - `cargo fmt -- --check`: passed
  - `git diff --check`: passed
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1417/1417`, avg case `2.625ms`
    - `derive_contract`: `326` derived, `0` unsupported, `1` expected
      not-equivalent case
    - `derive_shadow_pressure`: `50/50`, generic simplify successes `0`
    - `derive_didactic_audit`: `410` cases, `0` flags
    - `simplify_didactic_audit`: `14` cases, `0` flags, `26` wire substeps
    - `simplify_strict`: `16518/16518`, `0` failures/timeouts
- decision:
  - retained as derive coverage plus didactic path-quality guardrail
  - no engine runtime rewrite was needed; this cycle promotes an existing
    radical transition and permanently measures its two-step route in derive

### 2026-04-29: Derive Half-Angle Tangent Expansion Capture

- area:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [trig.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/trig.rs)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
  - identity shadow root `identity_tangent_half_angle`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: curated derive coverage for tangent half-angle
    expansion
  - `secondary_dimension`: didactic route specificity for target-aware trig
    expansion
  - `behavior_change_expected`: route selection should prefer the specific
    half-angle tangent transition over the broader tangent-to-sine/cosine
    expansion when the target is a half-angle form
- structural_axis:
  - tangent half-angle expansion:
    `tan(x/2) -> sin(x)/(1 + cos(x))`
- why_this_is_not_a_duplicate:
  - existing curated rows cover half-angle sine/cosine squares, tangent
    quotient contraction, and tangent triple-angle, but not the tangent
    half-angle expansion direction from the shadow identity
  - the current CLI probe derives the target, but reports the broad
    `Expand tangent to sine over cosine` step; that makes the route less
    specific than the engine capability it is using
- discovery_or_promotion:
  - promotion of an identity-pair shadow behavior already reported as derived
    by `derive_shadow_pressure`
  - local probe exposed a route-specificity/didactic weakness: the broad
    tangent expansion wins before the half-angle tangent provider
- if_promoted_why_minimal_representative:
  - one tangent call, one half-angle argument, one standard target form, no
    passthrough and no composition wrapper
- trigger:
  - CLI reports `Strategy: expand trig` for
    `derive tan(x/2), sin(x)/(1 + cos(x))`
  - the visible step currently uses the generic tangent expansion description
    instead of the half-angle tangent identity
- hypothesis:
  - moving the half-angle tangent expansion provider ahead of generic
    reciprocal trig expansion makes the route specific without broad search
  - adding one derive row and focused didactic regression keeps the identity
    stable and prevents the half-angle step from becoming magical
- success_condition:
  - CLI reports a half-angle tangent identity step for
    `derive tan(x/2), sin(x)/(1 + cos(x))`
  - `derive_contract` accepts the row with expected strategy `expand trig`
  - focused didactic regression sees the specific half-angle tangent rule and
    no generic tangent-to-sine/cosine substeps
  - `derive_didactic_audit` remains flag-free
  - `make engine-fast` and `make engine-scorecard` stay green
- relevant_lanes:
  - CLI derive probe
  - focused solver derive command tests for half-angle tangent
  - derive contract row
  - focused derive didactic test
  - derive didactic audit
  - engine-fast
  - engine-scorecard
- promotion_target:
  - `derive_pairs.csv`
- derive_bridge_check:
  - promoted because it adds a real target-family bridge from an existing
    identity shadow case and improves route specificity, not just row count
- engine_feedback_check:
  - gap classified as target-aware strategy ordering plus didactic-quality
    rendering; no new reusable simplification engine rule is expected
- retain_if:
  - the case derives through the half-angle tangent identity, the didactic step
    remains direct and specific, and global guardrails stay green
- reject_if:
  - the case still routes through broad tangent-to-sine/cosine expansion,
    creates generic placeholder substeps, or destabilizes trig derive/audit
- local_result:
  - added
    `expand_trig_half_angle_tangent_sin_over_one_plus_cos,trig_expand,tan(x/2),sin(x)/(1+cos(x)),derived,expand trig`
    to the curated derive contract
  - normalized the doubled argument inside the half-angle tangent expansion
    provider so `tan(x/2)` targets the concrete form `sin(x)/(1+cos(x))`
    through `Half-Angle Tangent Identity`
  - added a focused solver regression for `tan(x/2) -> sin(x)/(1+cos(x))`
    using `HalfAngleTangentExpandSinOverOnePlusCos`
  - added a focused derive didactic regression requiring the public rule
    `Aplicar identidad de tangente de ángulo mitad` with no generic substeps
  - CLI now reports:
    `Expand tan(u) as sin(2u)/(1 + cos(2u)) [Half-Angle Tangent Identity]`
- guardrails:
  - CLI derive probe: passed
  - focused solver half-angle tangent provider test: passed
  - derive generic-simplify guard: passed
  - focused derive didactic half-angle tangent test: passed
  - release derive contract: `327` derived, `0` unsupported, `1` expected
    not-equivalent case
  - derive didactic audit: `411` cases, `0` flags, `436` web substeps
  - `cargo fmt -- --check`: passed
  - `git diff --check`: passed
  - `make engine-fast`: passed
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1417/1417`, avg case `2.661ms`
    - `derive_contract`: `327` derived, `0` unsupported, `1` expected
      not-equivalent case
    - `derive_shadow_pressure`: `50/50`, generic simplify successes `0`
    - `derive_didactic_audit`: `411` cases, `0` flags
    - `simplify_didactic_audit`: `14` cases, `0` flags, `26` wire substeps
    - `simplify_strict`: `16518/16518`, `0` failures/timeouts
- decision:
  - retained as coverage plus target-aware derive route specificity
  - no new simplification runtime rule was needed; the engine already had the
    half-angle tangent capability, but the provider needed to recognize the
    simplified doubled argument so derive could expose the correct rule

### 2026-04-29: Derive Tangent Double-Angle Route Specificity Capture

- area:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [trig.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/trig.rs)
  - [visible_rule_names.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/visible_rule_names.rs)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: both
    `derive tan(2*x), 2*tan(x)/(1-tan(x)^2)` and
    `derive 2*tan(x)/(1-tan(x)^2), tan(2*x)` use a specific tangent
    double-angle rule instead of generic tangent expansion or noisy
    `simplify`
  - `primary_dimension`: derive target-family route specificity for tangent
    double-angle forms
  - `secondary_dimension`: didactic path quality for a currently magical/noisy
    trig route
  - `hypothesis`: adding a bounded, target-aware tangent double-angle provider
    in both directions reuses existing algebraic capability, improves exact
    `source -> target` bridgeability, and avoids broad search or runtime risk
  - `relevant_lanes`: CLI derive probes, focused solver trig unit, derive
    generic-simplify guard, focused derive didactic regression, release derive
    contract, derive didactic audit, `make engine-fast`,
    `make engine-scorecard`
  - `promotion_target`: `derive_pairs.csv`
  - `derive_bridge_check`: promoted because the engine can already prove the
    identity, but derive lacks a clean explicit route in at least one
    direction
  - `engine_feedback_check`: gap classified as target-aware strategy and
    didactic-quality routing, not a missing reusable simplification rule
  - `retain_if`: both directions derive through the specific tangent
    double-angle identity, the route has no generic filler substeps, and global
    guardrails remain green
  - `reject_if`: either direction still falls through generic tangent
    expansion/simplify, introduces formula-template noise, or regresses
    guardrail runtime/coverage
- structural_axis:
  - tangent double-angle expansion/contraction:
    `tan(2*x) <-> 2*tan(x)/(1 - tan(x)^2)`
- why_this_is_not_a_duplicate:
  - existing rows cover sine/cosine double-angle, tangent triple-angle, and
    hyperbolic tangent double-angle, but not the ordinary tangent double-angle
    bridge
  - current probes show a concrete route-quality gap: expansion is labeled as
    generic tangent-to-sine/cosine and contraction falls back to a long
    `simplify` trace
- discovery_or_promotion:
  - promotion of a stable identity family already supported semantically by the
    engine
- if_promoted_why_minimal_representative:
  - one tangent call, one symbolic argument, no passthrough, no negation, and no
    additional wrapper
- local_result:
  - added
    `expand_trig_double_tangent,trig_expand,tan(2*x),2*tan(x)/(1-tan(x)^2),derived,expand trig`
    and
    `contract_trig_double_tangent,trig_contract,2*tan(x)/(1-tan(x)^2),tan(2*x),derived,contract trig`
    to the curated derive contract
  - added target-aware tangent double-angle expansion and contraction providers
    using the public rule `Tangent Double-Angle Identity`
  - mapped that public rule to
    `Aplicar identidad de tangente de ángulo doble`
  - marked the new visible rule as a direct self-explanatory derive step in the
    didactic audit, with focused regression requiring no filler substeps
  - CLI now reports:
    - `Expand tangent double-angle form [Tangent Double-Angle Identity]`
    - `Recognize tangent double-angle form [Tangent Double-Angle Identity]`
- guardrails:
  - CLI derive probes for both directions: passed
  - focused solver tangent double-angle provider test: passed
  - derive generic-simplify guard: passed
  - focused derive didactic tangent double-angle test: passed
  - release derive contract: `329` derived, `0` unsupported, `1` expected
    not-equivalent case
  - derive didactic audit: `413` cases, `0` flags, `436` web substeps
  - `cargo fmt -- --check`: passed
  - `git diff --check`: passed
  - `make engine-fast`: passed
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1417/1417`, avg case `2.668ms`
    - `derive_contract`: `329` derived, `0` unsupported, `1` expected
      not-equivalent case
    - `derive_shadow_pressure`: `50/50`, generic simplify successes `0`
    - `derive_didactic_audit`: `413` cases, `0` flags
    - `simplify_didactic_audit`: `14` cases, `0` flags, `26` wire substeps
    - `simplify_strict`: `16518/16518`, `0` failures/timeouts
- decision:
  - retained as derive coverage plus route-specific didactic quality
  - no new simplify runtime rule was needed; the retained value is converting
    an already provable trig identity into an explicit, target-aware derive
    bridge in both directions

### 2026-04-29: Derive Tangent Angle Sum/Difference Route Specificity Capture

- area:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [trig.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/trig.rs)
  - [visible_rule_names.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/visible_rule_names.rs)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: tangent angle sum/difference derives use a specific
    tangent angle sum/difference identity in both expansion and contraction,
    and no direction falls back to generic tangent expansion or noisy
    `simplify`
  - `primary_dimension`: derive target-family route specificity for tangent
    angle sum/difference identities
  - `secondary_dimension`: sign/orientation robustness for tangent sum vs
    tangent difference denominators
  - `hypothesis`: adding a bounded, target-aware tangent angle
    sum/difference provider reuses identities already present in
    `identity_pairs.csv`, closes a current noisy contraction path, and avoids
    planner search
  - `relevant_lanes`: CLI derive probes, focused solver trig unit, derive
    generic-simplify guard, focused derive didactic regression, release derive
    contract, derive didactic audit, `make engine-fast`,
    `make engine-scorecard`
  - `promotion_target`: `derive_pairs.csv`
  - `derive_bridge_check`: promoted because the engine already proves
    `tan(x+y)` and `tan(x-y)` identities, but derive currently exposes generic
    or noisy routes instead of the reusable target-family bridge
  - `engine_feedback_check`: gap classified as target-aware strategy plus
    didactic-quality routing, not a missing reusable simplification rule
  - `retain_if`: sum/difference expansion and contraction derive through the
    specific tangent angle rule, didactic audit remains flag-free, and global
    guardrails stay green
  - `reject_if`: any representative still falls through generic
    tangent-to-sine/cosine expansion, `simplify`, filler substeps, or embedded
    runtime regresses materially
- structural_axis:
  - tangent angle sum/difference:
    `tan(x+y) <-> (tan(x)+tan(y))/(1 - tan(x)*tan(y))`
    and
    `tan(x-y) <-> (tan(x)-tan(y))/(1 + tan(x)*tan(y))`
- why_this_is_not_a_duplicate:
  - existing derive rows cover sine/cosine angle sum/difference, tangent
    double-angle, tangent half-angle, and tangent triple-angle, but not ordinary
    tangent angle sum/difference
  - current probes show route-quality gaps: expansion is labeled as generic
    tangent-to-sine/cosine and tangent-difference contraction falls back to a
    9-step `simplify` trace
- discovery_or_promotion:
  - promotion of existing engine/metamorphic identities from `identity_pairs.csv`
- if_promoted_why_minimal_representative:
  - one symbolic sum case and one symbolic difference case, each in expansion
    and contraction direction, with no passthrough or extra wrappers
- local_result:
  - added four curated derive rows:
    `expand_trig_tangent_angle_sum`,
    `contract_trig_tangent_angle_sum`,
    `expand_trig_tangent_angle_difference`, and
    `contract_trig_tangent_angle_difference`
  - added target-aware tangent angle sum/difference expansion and contraction
    providers using the public rule `Tangent Angle Sum/Diff Identity`
  - mapped that rule to
    `Aplicar identidad de tangente de suma/diferencia de ángulos`
  - marked the new visible rule as self-explanatory in the didactic audit and
    added a focused regression requiring no filler substeps
  - CLI probes now report one-step routes:
    - `tan(x+y) -> (tan(x)+tan(y))/(1-tan(x)*tan(y))` via `expand trig`
    - `(tan(x)+tan(y))/(1-tan(x)*tan(y)) -> tan(x+y)` via `contract trig`
    - `tan(x-y) -> (tan(x)-tan(y))/(1+tan(x)*tan(y))` via `expand trig`
    - `(tan(x)-tan(y))/(1+tan(x)*tan(y)) -> tan(x-y)` via `contract trig`
- guardrails:
  - CLI derive probes for all four directions: passed
  - focused solver tangent angle sum/difference provider test: passed
  - derive generic-simplify guard: passed
  - focused derive didactic tangent angle sum/difference test: passed
  - release derive contract: `333` derived, `0` unsupported, `1` expected
    not-equivalent case
  - derive didactic audit: `417` cases, `0` flags, `436` web substeps
  - `cargo fmt -- --check`: passed
  - `git diff --check`: passed
  - `make engine-fast`: passed
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1417/1417`, avg case `2.632ms`
    - `derive_contract`: `333` derived, `0` unsupported, `1` expected
      not-equivalent case
    - `derive_shadow_pressure`: `50/50`, generic simplify successes `0`
    - `derive_didactic_audit`: `417` cases, `0` flags
    - `simplify_didactic_audit`: `14` cases, `0` flags, `26` wire substeps
    - `simplify_strict`: `16518/16518`, `0` failures/timeouts
- decision:
  - retained as derive coverage plus route-specific didactic quality
  - no new simplify runtime rule was needed; the value is turning already
    provable tangent sum/difference identities into explicit target-aware
    derive bridges in both directions

### 2026-04-29: Derive Inverse Atan Right-Triangle Composition Capture

- area:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [inverse_trig_composition_support.rs](/Users/javiergimenezmoya/developer/math/crates/cas_math/src/inverse_trig_composition_support.rs)
  - [derive_command.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: `derive sin(arctan(x)), x/sqrt(1+x^2)` and
    `derive cos(arctan(x)), 1/sqrt(1+x^2)` use the explicit
    `rewrite inverse trigs` strategy with no generic `simplify` fallback and no
    zero-multiplication filler step
  - `primary_dimension`: derive target-family coverage for inverse-trig
    right-triangle compositions
  - `secondary_dimension`: didactic path quality for a currently generic/noisy
    one-step simplification route
  - `hypothesis`: extending the existing inverse-trig composition planner with
    the bounded `sin(arctan(u))` and `cos(arctan(u))` formulas reuses engine
    capability already present in `identity_pairs.csv` and avoids broader
    planner search
  - `relevant_lanes`: CLI derive probes, focused cas_math planner unit,
    direct derive command regression, derive generic-simplify guard, focused
    derive didactic regression, release derive contract, derive didactic audit,
    `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: `derive_pairs.csv`
  - `derive_bridge_check`: promoted because the engine can already prove the
    identities, but `derive` currently classifies them as generic `simplify`
    instead of the existing inverse-trig rewrite family
  - `engine_feedback_check`: gap classified as target classifier / strategy
    provider coverage plus didactic-quality routing, not a missing reusable
    simplify runtime rule
  - `retain_if`: both identities derive through `rewrite inverse trigs`, the
    focused didactic test sees the intended inverse-trig rule, and global
    guardrails stay green
  - `reject_if`: either identity remains generic `simplify`, keeps the
    zero-property filler step, requires broad search, or regresses embedded
    runtime/guardrail coverage
- structural_axis:
  - inverse tangent right-triangle compositions:
    `sin(arctan(x)) -> x/sqrt(1+x^2)`
    and
    `cos(arctan(x)) -> 1/sqrt(1+x^2)`
- why_this_is_not_a_duplicate:
  - existing derive rows cover `asin(x/sqrt(x^2+1)) -> arctan(x)` and inverse
    tangent reciprocal sums, but not direct trig-of-arctan right-triangle
    projections
  - current probes show `sin(arctan(x))` reaches the target via generic
    `simplify`, while `cos(arctan(x))` adds a redundant
    `Zero Property of Multiplication` step after the inverse composition
- discovery_or_promotion:
  - promotion of stable identities already present in `identity_pairs.csv`
- if_promoted_why_minimal_representative:
  - two primitive projections from one arctangent argument, no passthrough,
    no extra wrapper, and no branch-sensitive inverse target
- local_result:
  - added `sin_arctan_right_triangle_projection` and
    `cos_arctan_right_triangle_projection` to the derive contract
  - extended the inverse-trig composition planner with `SinArctan` and
    `CosArctan`
  - added focused planner, direct derive, target-classifier, and didactic
    regressions
  - added focused didactic substeps for the route: calculate the arctangent
    triangle hypotenuse and read the sine/cosine projection from that triangle
  - CLI probes now report `rewrite inverse trigs` with one top-level step and
    no zero-multiplication filler
- guardrails:
  - CLI derive probes for both projections: passed
  - focused cas_math planner unit: passed
  - direct derive command regression: passed
  - target classifier regression: passed
  - derive generic-simplify guard: passed
  - focused derive didactic right-triangle test: passed
  - release derive contract: `335` derived, `0` unsupported, `1` expected
    not-equivalent case
  - derive didactic audit: `419` cases, `0` flags, `440` web substeps
  - `cargo fmt -- --check`: passed
  - `git diff --check`: passed
  - `make engine-fast`: passed
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1417/1417`, avg case `2.689ms`
    - `derive_contract`: `335` derived, `0` unsupported, `1` expected
      not-equivalent case
    - `derive_shadow_pressure`: `50/50`, generic simplify successes `0`
    - `derive_didactic_audit`: `419` cases, `0` flags
    - `simplify_didactic_audit`: `14` cases, `0` flags, `26` wire substeps
    - `simplify_strict`: `16518/16518`, `0` failures/timeouts
- decision:
  - retained as derive bridge coverage and didactic path-quality improvement
  - no new simplify runtime rule was needed; the engine already knew these
    identities, and this cycle made the route target-aware and explainable

### 2026-04-29: Sign-Preserving Required-Condition Normalization Capture

- area:
  - [domain_normalization.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver_core/src/domain_normalization.rs)
  - [expr_normalization.rs](/Users/javiergimenezmoya/developer/math/crates/cas_math/src/expr_normalization.rs)
  - [condition_normalization_tests.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/condition_normalization_tests.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: robustness
  - `success_condition`: `sqrt(1-x^2)`, `sqrt(y-x^2)`, and derived
    inverse-trig traces display sign-preserving required conditions such as
    `1 - x^2 ≥ 0`, while `NonZero` guards still canonicalize sign-insensitive
    pairs such as `2 - x ≠ 0` to `x - 2 ≠ 0`
  - `primary_dimension`: semantic/domain transparency for analytic required
    conditions
  - `secondary_dimension`: derive path quality, because inverse-trig projection
    traces inherit these required-condition displays
  - `hypothesis`: condition normalization currently flips leading polynomial
    signs for all condition kinds; that is valid for `NonZero` but unsound for
    `NonNegative` and `Positive`, so splitting sign-preserving analytic
    normalization from sign-insensitive nonzero normalization fixes a reusable
    domain-display bug without changing simplification runtime
  - `relevant_lanes`: focused condition-normalization tests, CLI JSON probes
    for sqrt/derive required displays, `make engine-fast`,
    `make engine-scorecard`
  - `promotion_target`: unit
  - `derive_bridge_check`: no new derive row is promoted in this cycle; derive
    benefits by receiving correct domain requirements for existing inverse-trig
    and radical routes
  - `engine_feedback_check`: gap classified as required-condition
    normalization/domain-transparency, not a missing algebraic simplification
    or derive planner capability
  - `retain_if`: sign-sensitive conditions preserve orientation, nonzero
    dedupe remains sign-insensitive, focused tests and global guardrails remain
    green
  - `reject_if`: sign-insensitive nonzero dedupe regresses, condition rendering
    becomes less canonical for ordinary denominator guards, or scorecard
    guardrails fail
- local_result:
  - added sign-preserving condition normalization for `NonNegative` and
    `Positive` predicates
  - kept the existing sign-insensitive normalization path for `NonZero`
    predicates, so denominator guards such as `2 - x != 0` still display as
    `x - 2 != 0`
  - added focused unit coverage for sign-preserving polynomial conditions in
    `cas_math` and `cas_solver`
  - CLI probes now display:
    - `sqrt(1-x^2)`: `1 - x^2 >= 0`
    - `sqrt(y-x^2)`: `y - x^2 >= 0`
    - `ln(1-x^2)`: `1 - x^2 > 0`
    - `1/(2-x)`: `x - 2 != 0`
  - existing inverse-trig derive traces that produce `sqrt(1-x^2)` now inherit
    `1 - x^2 >= 0` instead of the inverted `x^2 - 1 >= 0`
- guardrails:
  - focused `cas_math` sign-preserving normalization unit: passed
  - focused `cas_solver` nonnegative/positive sign-preserving units: passed
  - focused `cas_solver` nonzero sign normalization unit: passed
  - full `condition_normalization_tests`: `10/10` passed
  - CLI probes for sqrt/log/nonzero and inverse-trig derive required displays:
    passed
  - `cargo fmt -- --check`: passed
  - `make engine-fast`: passed
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1417/1417`, avg case `2.717ms`
    - `derive_contract`: `335` derived, `0` unsupported, `1` expected
      not-equivalent case
    - `derive_shadow_pressure`: `50/50`, generic simplify successes `0`
    - `derive_didactic_audit`: `419` cases, `0` flags
    - `simplify_didactic_audit`: `14` cases, `0` flags, `26` wire substeps
    - `simplify_strict`: `16518/16518`, `0` failures/timeouts
- decision:
  - retained as robustness/domain-transparency improvement
  - no new derive case was promoted; the derive bridge value is inherited by
    existing traces through correct required-condition display

### 2026-04-29: Derive Arcsin/Arccos Complement Projection Capture

- area:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [inverse_trig_composition_support.rs](/Users/javiergimenezmoya/developer/math/crates/cas_math/src/inverse_trig_composition_support.rs)
  - [derive_command.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs)
  - [focused_rule_substeps.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: `derive cos(arcsin(x)), sqrt(1-x^2)` and
    `derive sin(arccos(x)), sqrt(1-x^2)` use the explicit
    `rewrite inverse trigs` strategy with one top-level step, no generic
    `simplify` fallback, no zero-multiplication filler, and sign-correct
    `1 - x^2 >= 0` required-condition display
  - `primary_dimension`: derive target-family coverage for inverse-trig
    right-triangle complement projections
  - `secondary_dimension`: didactic path quality for a currently generic/noisy
    simplification route
  - `hypothesis`: extending the existing inverse-trig composition planner with
    bounded `cos(arcsin(u))` and `sin(arccos(u))` projections reuses engine
    capability already present in `identity_pairs.csv`, consumes the previous
    required-condition fix, and avoids broader derive search
  - `relevant_lanes`: CLI derive probes, focused cas_math planner unit,
    direct derive command regression, target classifier regression, derive
    generic-simplify guard, focused derive didactic regression, release derive
    contract, derive didactic audit, `make engine-fast`,
    `make engine-scorecard`
  - `promotion_target`: `derive_pairs.csv`
  - `derive_bridge_check`: promoted because the engine can already simplify
    both identities, but `derive` currently classifies them as generic
    `simplify`; `cos(arcsin(x))` additionally emits a redundant
    zero-multiplication step
  - `engine_feedback_check`: gap classified as target classifier / strategy
    provider coverage plus didactic-quality routing, not a missing reusable
    simplify runtime rule
  - `retain_if`: both identities derive through `rewrite inverse trigs`, the
    focused didactic test sees inverse-trig right-triangle substeps, and global
    guardrails stay green
  - `reject_if`: either identity remains generic `simplify`, keeps the
    zero-property filler step, loses the sign-correct required condition, or
    regresses embedded runtime/guardrails
- structural_axis:
  - inverse sine/cosine complement projections:
    `cos(arcsin(x)) -> sqrt(1-x^2)`
    and
    `sin(arccos(x)) -> sqrt(1-x^2)`
- why_this_is_not_a_duplicate:
  - existing derive rows cover `sin(arctan(x))`,
    `cos(arctan(x))`, and `asin(x/sqrt(x^2+1)) -> atan(x)`, but not the
    complementary arcsin/arccos projections to `sqrt(1-x^2)`
  - current probes show `sin(arccos(x))` reaches the target via generic
    `simplify`, while `cos(arcsin(x))` adds a redundant
    `Zero Property of Multiplication` step
- discovery_or_promotion:
  - promotion of stable identities already present in `identity_pairs.csv`
- if_promoted_why_minimal_representative:
  - two primitive one-argument projections, no passthrough, no extra wrapper,
    and no branch-changing inverse target
- local_result:
  - added explicit inverse-trig composition planner support for
    `cos(arcsin(x)) -> sqrt(1-x^2)` and
    `sin(arccos(x)) -> sqrt(1-x^2)`
  - both CLI probes now return `strategy: rewrite inverse trigs` with
    `steps_count: 1`, required display `1 - x^2 >= 0`, and no generic
    `simplify` / `Zero Property of Multiplication` filler
  - didactic output now gives two right-triangle substeps for each projection:
    compute the remaining cathetus, then read the projected sine/cosine
- guardrails:
  - `cargo test -q -p cas_math inverse_trig_composition_support::tests::inverse_trig_composition_plan_detects_arcsin_arccos_complement_projections -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive_command::tests::direct_derive_rewrites_arcsin_arccos_complement_projections_without_simplify -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive::target_classifier::tests::classifies_tabulated_inverse_trig_rewritten_targets -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_arcsin_arccos_complement_projections_use_inverse_trig_language -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`
    reported `derived=337 unsupported=0 not_equivalent=1`,
    `generic_simplify_expected=0`, and `rewrite inverse trigs: 6`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`
    reported `cases=421 flagged=0 no_web_substeps=0 no_web_steps=0 total_web_substeps=444`
  - `cargo fmt -- --check`
  - `git diff --check`
  - `make engine-fast`
  - `make engine-scorecard`: `embedded_equivalence_context 1417/1417`,
    `derive_contract derived=337 unsupported=0 not_equivalent=1`,
    `derive_shadow_pressure 50/50`, `derive_didactic_audit 421 cases 0 flags`,
    `simplify_didactic_audit 14 cases 0 flags`, and
    `simplify_strict 16518/16518 proved-symbolic`
- decision:
  - retained as coverage improvement: the engine already had the semantic
    simplification, so this cycle improves derive completeness and didactic
    transparency without adding a broader runtime simplification rule

### 2026-04-29: Derive Tan Arcsin Tangent Projection Capture

- area:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [inverse_trig_composition_support.rs](/Users/javiergimenezmoya/developer/math/crates/cas_math/src/inverse_trig_composition_support.rs)
  - [derive_command.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs)
  - [focused_rule_substeps.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: `derive tan(arcsin(x)), x/sqrt(1-x^2)` uses the
    explicit `rewrite inverse trigs` strategy with one top-level step, no
    generic `simplify` fallback, no ordinary `expand trig` classification, and
    required-condition display `1 - x^2 > 0`
  - `primary_dimension`: derive target-family coverage for inverse-trig
    right-triangle tangent projection
  - `secondary_dimension`: didactic path quality for a currently opaque
    one-step trig expansion route
  - `hypothesis`: extending the existing inverse-trig composition planner with
    the bounded `tan(arcsin(u))` projection reuses an identity already present
    in `identity_pairs.csv`, preserves the stricter denominator domain
    condition, and avoids routing this inverse-trig case through generic trig
    expansion
  - `relevant_lanes`: CLI derive probe, focused cas_math planner unit, direct
    derive command regression, target classifier regression, derive
    generic-simplify guard, focused derive didactic regression, release derive
    contract, derive didactic audit, `make engine-fast`,
    `make engine-scorecard`
  - `promotion_target`: `derive_pairs.csv`
  - `derive_bridge_check`: promoted because the engine already derives the
    target but labels the route as `expand trig`; the target form is a stable
    inverse-trig right-triangle projection and should be represented in the
    inverse-trig derive family
  - `engine_feedback_check`: gap classified as target classifier / inverse-trig
    strategy coverage plus didactic-quality routing, not a missing reusable
    simplify runtime rule
  - `retain_if`: the identity derives through `rewrite inverse trigs`, the
    focused didactic test sees right-triangle tangent substeps, the positive
    `1 - x^2 > 0` condition is preserved, and global guardrails stay green
  - `reject_if`: the identity remains classified as `expand trig`, loses its
    strict denominator-domain condition, needs generic `simplify`, or regresses
    embedded runtime/guardrails
- structural_axis:
  - inverse sine tangent projection:
    `tan(arcsin(x)) -> x/sqrt(1-x^2)`
- why_this_is_not_a_duplicate:
  - existing derive rows cover arctangent sine/cosine projections and arcsin/
    arccos sine/cosine complement projections, but not the tangent projection
    from an arcsine-defined right triangle
  - current probe reaches the target via `expand trig`, so it is semantically
    supported but assigned to the wrong explanatory family
- discovery_or_promotion:
  - promotion of a stable identity already present in `identity_pairs.csv`
- if_promoted_why_minimal_representative:
  - one primitive one-argument tangent projection, no passthrough, no extra
    wrapper, and no branch-changing inverse target
- local_result:
  - added `TanArcsin` to the inverse-trig composition planner and public rule
    descriptions
  - promoted `tan_arcsin_tangent_projection` to the derive contract with
    expected strategy `rewrite inverse trigs`
  - added a small inverse-trig fast-path before the ordinary trig fast-path so
    inverse-trig compositions are not preempted by `expand trig`
  - CLI probe now returns `strategy: rewrite inverse trigs`, `steps_count: 1`,
    required display `1 - x^2 > 0`, and two right-triangle substeps: compute
    the remaining cathetus, then read the tangent
- guardrails:
  - `cargo test -q -p cas_math inverse_trig_composition_support::tests::inverse_trig_composition_plan_detects_tan_arcsin_projection -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive_command::tests::direct_derive_rewrites_tan_arcsin_projection_without_trig_expand -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive::target_classifier::tests::classifies_tabulated_inverse_trig_rewritten_targets -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_arcsin_projection_steps_use_inverse_trig_language -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`
    reported `derived=338 unsupported=0 not_equivalent=1`,
    `generic_simplify_expected=0`, and `rewrite inverse trigs: 7`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`
    reported `cases=422 flagged=0 no_web_substeps=0 no_web_steps=0 total_web_substeps=446`
  - `cargo fmt -- --check`
  - `git diff --check`
  - `make engine-fast`
  - `make engine-scorecard`: `embedded_equivalence_context 1417/1417`,
    `derive_contract derived=338 unsupported=0 not_equivalent=1`,
    `derive_shadow_pressure 50/50`, `derive_didactic_audit 422 cases 0 flags`,
    `simplify_didactic_audit 14 cases 0 flags`, and
    `simplify_strict 16518/16518 proved-symbolic`
- decision:
  - retained as coverage and derive path-quality improvement: the semantic
    simplification already existed, and this cycle moved the teachable route
    into the inverse-trig family while preserving the stricter tangent-domain
    condition

### 2026-04-29: Derive Inverse-Trig Double-Angle Expansion Capture

- area:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [target_classifier.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/target_classifier.rs)
  - [derive_command.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs)
  - [focused_rule_substeps.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: the four minimal inverse-trig double-angle rows
    `sin(2*arcsin(x))`, `cos(2*arcsin(x))`, `sin(2*arccos(x))`, and
    `cos(2*arccos(x))` derive with expected strategy `expand trig`, keep one
    top-level `Expandir ángulo doble` step, emit concrete expansion/projection
    substeps, preserve domain-condition display, and keep guardrails green
  - `primary_dimension`: derive coverage for inverse-trig double-angle
    compositions already represented in the engine identity corpus
  - `secondary_dimension`: didactic quality for transformations that currently
    derive as a magical one-step trig expansion
  - `hypothesis`: the engine already proves these identities through target-aware
    trig expansion; promoting the smallest complete sine/cosine by
    arcsin/arccos grid and adding focused substeps should increase derive
    bridgeability and teachability without runtime-rule risk
  - `relevant_lanes`: CLI derive probes, target classifier regression, direct
    derive command regression, focused derive didactic regression, derive
    generic-simplify guard, release derive contract, derive didactic audit,
    `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: `derive_pairs.csv`
  - `derive_bridge_check`: promoted because the identities are already stable in
    `identity_pairs.csv` and derive succeeds, but the path is absent from the
    curated derive corpus and lacks the intermediate projection explanation
  - `engine_feedback_check`: classified as didactic-quality plus derive corpus
    coverage; no reusable runtime simplification rule is missing in this cycle
  - `retain_if`: all four rows are derived through `expand trig`, focused
    didactic tests see two concrete substeps, release contract/audit counts
    increase without flags, and global scorecard remains green
  - `reject_if`: any row needs generic `simplify`, loses equivalence/domain
    conditions, emits duplicate parent substeps, or regresses embedded/runtime
    guardrails
- structural_axis:
  - inverse-trig double-angle projection grid: outer `sin/cos` crossed with
    inner `arcsin/arccos`
- why_this_is_not_a_duplicate:
  - existing derive rows cover plain double-angle expansion and first-order
    inverse-trig projections, but not the composed route that expands
    `sin(2u)`/`cos(2u)` and then projects `u = arcsin(x)` or `u = arccos(x)`
- discovery_or_promotion:
  - promotion of four stable identities already present in `identity_pairs.csv`
- if_promoted_why_minimal_representative:
  - the four rows are the smallest complete grid for the primitive sine/cosine
    double-angle projections; wrappers, passthrough terms, and deeper
    compositions stay out of the curated derive corpus for now
- local_result:
  - promoted four inverse-trig double-angle identities from `identity_pairs.csv`
    into the curated derive contract under the `trig_expand` family
  - added bounded `TrigExpanded` classifier recognition for the
    `sin/cos(2*arcsin/arccos(x))` projection grid, closing the local
    classifier miss found during the focused test
  - added focused derive command and didactic audit regressions for the four
    cases
  - added two concrete web substeps for the composed double-angle route:
    expand with the double-angle identity, then substitute the inverse-trig
    projections
  - CLI probe for `derive sin(2*arcsin(x)), 2*x*sqrt(1-x^2)` keeps
    `strategy: expand trig`, one top-level `Double Angle Expansion` step, and
    required display `1 - x^2 ≥ 0`
- guardrails:
  - `cargo test -q -p cas_solver derive::target_classifier::tests::classifies_tabulated_trig_expanded_targets -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive_command::tests::direct_derive_expands_inverse_trig_double_angle_projections -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_inverse_trig_double_angle_expansions_show_projection -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`
    reported `derived=342 unsupported=0 not_equivalent=1`,
    `generic_simplify_expected=0`, and `expand trig: 83`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`
    reported `cases=426 flagged=0 no_web_substeps=0 no_web_steps=0 total_web_substeps=454`
  - `cargo fmt -- --check`
  - `git diff --check`
  - `make engine-fast`
  - `make engine-scorecard`: `embedded_equivalence_context 1417/1417`,
    `derive_contract derived=342 unsupported=0 not_equivalent=1`,
    `derive_shadow_pressure 50/50`, `derive_didactic_audit 426 cases 0 flags`,
    `simplify_didactic_audit 14 cases 0 flags`, and
    `simplify_strict 16518/16518 proved-symbolic`
- decision:
  - retained as a coverage and didactic-quality improvement: no runtime
    simplification rule was needed, but the cycle promoted a stable
    engine-known family into derive, closed target-classifier coverage, and
    made the previously magical one-step route explainable

### 2026-04-29: Derive Half-Angle Tangent Alternate Argument Capture

- area:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [target_classifier.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/target_classifier.rs)
  - [derive_command.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs)
  - [analysis_command_eval_tests.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/analysis_command_eval_tests.rs)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: `derive tan(x/2), (1-cos(x))/sin(x)` uses expected
    strategy `expand trig`, keeps one direct `Aplicar identidad de tangente de
    ángulo mitad` step with no redundant substeps, preserves required-condition
    display `sin(x) ≠ 0`, and keeps guardrails green
  - `primary_dimension`: derive contract coverage for the alternate
    half-angle tangent target form with a simplified half-angle argument
  - `secondary_dimension`: semantic/domain regime coverage because the two
    tangent half-angle variants have different visible denominators
  - `hypothesis`: the target-aware trig expansion already supports both
    half-angle tangent variants; promoting the missing `1-cos(x)` over
    `sin(x)` representative closes a stable identity-pair gap without adding
    runtime search or didactic noise
  - `relevant_lanes`: CLI derive probe, target classifier regression, direct
    derive command regression, focused derive didactic regression, derive
    generic-simplify guard, release derive contract, derive didactic audit,
    `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: `derive_pairs.csv`
  - `derive_bridge_check`: promoted because the engine already bridges the
    identity and `identity_pairs.csv` contains both tangent half-angle variants,
    but the primary derive corpus only covered the `sin(x)/(1+cos(x))`
    simplified-argument form
  - `engine_feedback_check`: classified as target-family coverage; no reusable
    runtime simplification rule is missing
  - `retain_if`: the new row derives through `expand trig`, the didactic audit
    accepts the direct step without a no-substeps flag, the domain condition is
    preserved, and global guardrails stay green
  - `reject_if`: the route needs generic `simplify`, emits duplicate/noisy
    substeps, loses the `sin(x) ≠ 0` condition, or regresses scorecard lanes
- structural_axis:
  - tangent half-angle alternate denominator: `(1-cos(x))/sin(x)` paired with
    the existing `sin(x)/(1+cos(x))` representative
- why_this_is_not_a_duplicate:
  - the existing primary derive row covers the conjugate denominator
    `1+cos(x)`, while this row covers the alternate denominator `sin(x)` and
    therefore different visible domain pressure
- discovery_or_promotion:
  - promotion of a stable identity already present in `identity_pairs.csv`
- if_promoted_why_minimal_representative:
  - one primitive half-angle source with no passthrough and no extra scaling;
    broader scaled variants stay in focused tests and discovery pressure
- local_result:
  - promoted `expand_trig_half_angle_tangent_one_minus_cos_over_sin` to the
    primary derive contract, next to the existing conjugate-denominator
    half-angle tangent row
  - added target-classifier, direct derive-command, CLI-domain, and didactic
    regressions for both simplified half-angle tangent target variants
  - confirmed the CLI route for `derive tan(x/2), (1-cos(x))/sin(x)` emits
    strategy `expand trig`, one `Half-Angle Tangent Identity` step, result
    `(1 - cos(x)) / sin(x)`, and required condition `sin(x) ≠ 0`
  - discovery note: the direct step metadata does not carry this quotient
    domain condition; command-level domain inference emits it correctly, so the
    retained guardrail lives at the CLI evaluation layer
- guardrails:
  - `cargo fmt`
  - `cargo test -q -p cas_solver derive::target_classifier::tests::classifies_tabulated_trig_expanded_targets -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive_command::tests::direct_derive_expands_simplified_argument_half_angle_tangent_variants -- --exact --nocapture`
  - `cargo test -q -p cas_solver analysis_command_eval_tests::tests::evaluate_derive_command_lines_expands_simplified_half_angle_tangent_alt_with_domain -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_half_angle_tangent_simplified_argument_uses_specific_identity -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=343`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `expand trig=84`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=427`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=454`
  - CLI probe: `derive tan(x/2), (1-cos(x))/sin(x)` reported `Strategy:
    expand trig` and `Requires: sin(x) ≠ 0`
  - `cargo fmt -- --check`
  - `git diff --check`
  - `make engine-fast`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=343 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `427 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as a coverage/domain-regime improvement; no runtime rule was
    needed because target-aware trig expansion already handled the identity,
    and the missing value was the primary derive contract plus visible-domain
    regression

### 2026-04-29: Derive Hyperbolic Half-Angle Sinh Coverage Capture

- area:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [hyperbolic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/hyperbolic.rs)
  - [target_classifier.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/target_classifier.rs)
  - [derive_command.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs)
  - [focused_rule_substeps.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs)
  - [visible_rule_names.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/visible_rule_names.rs)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: `derive sinh(x/2)^2, (cosh(x)-1)/2` is promoted to
    the primary derive contract, both hyperbolic half-angle square variants use
    strategy `rewrite hyperbolics`, emit a visible half-angle rule rather than
    a generic hyperbolic Pythagorean title, include one concrete didactic
    substep, and keep guardrails green
  - `primary_dimension`: derive contract coverage for the missing
    `sinh(x/2)^2 -> (cosh(x)-1)/2` hyperbolic half-angle representative
  - `secondary_dimension`: didactic path quality for hyperbolic half-angle
    square routes that were previously correct but labelled as Pythagorean
    rewrites
  - `hypothesis`: target-aware hyperbolic rewrite support already contains both
    half-angle square variants; a cheap route-ordering and didactic-label
    adjustment can make the existing capability teachable and promote the
    missing `sinh` counterpart without adding search or runtime rules
  - `relevant_lanes`: CLI derive probes, hyperbolic rewrite unit tests, target
    classifier regression, direct derive-command regression, focused derive
    didactic regression, derive generic-simplify guard, release derive contract,
    derive didactic audit, `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: `derive_pairs.csv`
  - `derive_bridge_check`: promoted because `identity_pairs.csv` already
    contains both hyperbolic half-angle square identities while the primary
    derive corpus only covered the `cosh(x/2)^2` representative
  - `engine_feedback_check`: classified as route ordering plus didactic-quality
    coverage; no reusable simplification runtime rule is missing
  - `retain_if`: the new row derives through `rewrite hyperbolics`, direct tests
    see `Hyperbolic Half-Angle Squares`, didactic audit sees one concrete
    substep for each `cosh/sinh` variant, no generic simplify expectation is
    introduced, and global guardrails stay green
  - `reject_if`: the change steals unrelated hyperbolic routes, loses
    equivalence, emits noisy duplicate substeps, or regresses embedded/runtime
    scorecard lanes
- structural_axis:
  - hyperbolic half-angle square symmetry: `cosh(x/2)^2` and `sinh(x/2)^2`
    mapped to their `cosh(x) ± 1` forms
- why_this_is_not_a_duplicate:
  - the existing derive row covers only the `+1`/`cosh` representative; this
    adds the complementary `-1`/`sinh` representative and corrects the shared
    visible route name for both variants
- discovery_or_promotion:
  - promotion of a stable identity already present in `identity_pairs.csv` plus
    route-label cleanup discovered by the CLI smoke probe
- if_promoted_why_minimal_representative:
  - one primitive no-passthrough `sinh` half-angle square row completes the
    minimal two-case hyperbolic half-angle square grid; scaled, negated, and
    passthrough variants remain in focused unit pressure
- local_result:
  - promoted `hyperbolic_half_angle_sinh_forward` to the primary derive
    contract under `rewrite hyperbolics`
  - moved the target-aware hyperbolic half-angle square route ahead of the
    broader Pythagorean-factor route so `cosh(x/2)^2` and `sinh(x/2)^2` now
    render as `Hyperbolic Half-Angle Squares`
  - added classifier, direct derive-command, focused didactic, and visible-rule
    regressions for the two-case hyperbolic half-angle square grid
  - CLI probe for `derive sinh(x/2)^2, (cosh(x)-1)/2` now reports `Strategy:
    rewrite hyperbolics`, one step, and `Hyperbolic Half-Angle Squares`
- guardrails:
  - `cargo fmt`
  - `cargo test -q -p cas_solver derive::hyperbolic::tests::target_aware_hyperbolic_rewrite_contracts_sinh_half_angle_square -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive::target_classifier::tests::classifies_hyperbolic_half_angle_square_targets -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive_command::tests::direct_derive_rewrites_hyperbolic_half_angle_squares_with_specific_rule -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_hyperbolic_half_angle_squares_use_specific_identity -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=344`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `rewrite hyperbolics=27`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=428`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=456`
  - CLI probe: `derive sinh(x/2)^2, (cosh(x)-1)/2` reported `Strategy:
    rewrite hyperbolics` and `[Hyperbolic Half-Angle Squares]`
  - `make engine-fast`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=344 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `428 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
  - `cargo fmt -- --check`
  - `git diff --check`
- decision:
  - retained as coverage plus didactic path-quality improvement: no new
    runtime simplification capability was needed, but the missing derive bridge
    is now contractual and the already-supported half-angle route no longer
    presents as a generic Pythagorean rewrite

### 2026-04-29: Derive Trig Negative Parity Path-Quality Capture

- area:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [trig.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/trig.rs)
  - [derive_command.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs)
  - [focused_rule_substeps.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs)
  - [visible_rule_names.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/visible_rule_names.rs)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: `derive tan(-x), -tan(x)` is promoted to the primary
    derive contract under `expand trig`, renders as `Trig Parity (Odd/Even)`
    rather than tangent expansion, has a parity substep in the didactic audit,
    and keeps scorecard guardrails green
  - `primary_dimension`: engine-to-derive bridge coverage for a negative
    trigonometric argument identity already present in `identity_pairs.csv`
  - `secondary_dimension`: didactic route quality for sign/orientation
    robustness
  - `hypothesis`: the reusable runtime parity rule already exists; derive was
    falling through to the generic tangent-expansion path and retargeting the
    final expression, so a target-aware parity route ahead of that fallback can
    correct the path without adding search or runtime rules
  - `relevant_lanes`: target-aware trig rewrite unit test, direct derive-command
    regression, focused derive didactic regression, CLI derive probe, derive
    generic-simplify guard, release derive contract, release derive didactic
    audit, `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: `derive_pairs.csv`
  - `derive_bridge_check`: promoted because the negative tangent identity is a
    stable metamorphic identity and exposed a concrete derive path-quality bug
  - `engine_feedback_check`: classified as route ordering plus didactic-quality
    coverage; no reusable simplification runtime rule is missing
  - `retain_if`: CLI and direct tests report `Trig Parity (Odd/Even)`, the new
    row derives through `expand trig`, didactic audit sees the odd-function
    formula substep, no generic simplify expectation is introduced, and global
    guardrails stay green
  - `reject_if`: the route steals unrelated trig expansion/contract cases,
    emits duplicate parent-snapshot substeps, or regresses embedded/runtime
    scorecard lanes
- structural_axis:
  - sign/orientation robustness for negative arguments of trig functions
- why_this_is_not_a_duplicate:
  - the corpus already covered cofunction and angle/product identities, but not
    the negative-argument parity bridge; this representative also fixed an
    observed wrong rule label (`Expand tangent to sine over cosine`)
- discovery_or_promotion:
  - promotion of a stable identity plus path-quality correction discovered by a
    CLI smoke probe
- if_promoted_why_minimal_representative:
  - `tan(-x) -> -tan(x)` is the smallest representative that reproduced the
    mislabeled route; `sec/csc/cot` and passthrough variants are covered by the
    reusable target-aware unit test rather than extra primary corpus rows
- local_result:
  - added `expand_trig_negative_tangent_parity` to the derive contract
  - added `TrigOddEvenParity` as a target-aware trig expansion kind before the
    generic tangent-expansion fallback
  - added a visible Spanish title and a parity formula substep for web didactic
    output
  - CLI probe for `derive tan(-x), -tan(x)` now reports `Strategy: expand trig`
    and `[Trig Parity (Odd/Even)]`
- guardrails:
  - `cargo fmt`
  - `cargo test -q -p cas_solver derive::trig::tests::rewrites_negative_trig_parity_variants_target_aware -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive_command::tests::direct_derive_rewrites_negative_tangent_parity_with_specific_rule -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_trig_negative_parity_uses_specific_identity -- --exact --nocapture`
  - CLI probe: `derive tan(-x), -tan(x)` reported `Strategy: expand trig` and
    `[Trig Parity (Odd/Even)]`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=345`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `expand trig=85`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=429`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=457`
  - `make engine-fast`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=345 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `429 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as coverage plus didactic path-quality improvement: the engine
    already had the parity rule, and the retained value is making `derive`
    select and teach that real rule instead of presenting an unrelated tangent
    expansion

## 2026-05-01 - Discovery observe-only: displaced hyperbolic by-parts outputs

- area:
  - calculus / integration / hyperbolic simplification
- status:
  - `observe-only`
- resolved by:
  - 2026-05-22 observability close-out:
    later retained integration and presentation work covers the displaced
    positive-affine hyperbolic by-parts cases; current public probes for
    `integrate((2*x+3)*sinh/cosh(2*x+1), x)` return compact antiderivatives
    quickly without warnings or required conditions, and the focused integration
    contracts preserve the visible by-parts substep
- attempted case:
  - promote `integrate((2*x+3)*sinh(2*x+1), x)` and
    `integrate((2*x+3)*cosh(2*x+1), x)` while adding hyperbolic integration by
    parts
- local lane:
  - smoke probes:
    `gtimeout 15s cargo run -q -p cas_cli -- eval 'integrate((2*x+3)*sinh(2*x+1), x)' --format json`
    and the analogous `cosh` probe
  - direct-output probes:
    `gtimeout 15s cargo run -q -p cas_cli -- eval '((2*x+3)*cosh(2*x+1)-sinh(2*x+1))/2' --format json`
    and the analogous `sinh/cosh` swap
- local result:
  - simple unit-argument probes `integrate(x*sinh(x), x)` and
    `integrate(x*cosh(x), x)` completed in milliseconds and were retained
  - displaced/scaled argument probes timed out at 10-15s before promotion
  - smaller direct probes showed the reusable signature: subtractive
    hyperbolic sums with shifted linear arguments and linear coefficients can
    spend seconds in simplification even when no integration remains
- global result:
  - the displaced/scaled cases were not promoted
  - the retained patch was narrowed to unit-argument `linear * sinh(x)` /
    `linear * cosh(x)` so public `integrate` does not inherit the slow route
- why it regressed globally:
  - the antiderivative shape introduces a subtractive linear-combination of
    `sinh(linear)`/`cosh(linear)` terms; existing hyperbolic simplification can
    explore expensive rewrite paths for shifted linear arguments
- what could make it combinable later:
  - a cheap simplifier gate for subtractive hyperbolic linear combinations, or a
    canonical compact form that avoids triggering the expensive hyperbolic
    rewrite path
- decision:
  - observe-only discovery; do not retry the displaced/scaled integration
    promotion until the simplification cost signature has a bounded fix

## 2026-05-01 - Discovery observe-only: negative-affine hyperbolic by-parts orientation

- area:
  - calculus / integration / hyperbolic simplification / orientation robustness
- status:
  - `observe-only`
- resolved by:
  - 2026-05-22 robustness promotion:
    the public residual gate now accepts the existing
    `linear_times_hyperbolic_linear` integration detector in addition to the
    polynomial-by-parts detector, so
    `diff(integrate((2*x+3)*sinh/cosh(1-2*x), x), x) - ...` closes through the
    bounded shortcut in milliseconds without `depth_overflow` while preserving
    the compact public antiderivative and visible by-parts substep
- attempted case:
  - reduce runtime for `integrate((2*x+3)*sinh(1-2*x), x)` and
    `integrate((2*x+3)*cosh(1-2*x), x)` after shifted positive-affine
    hyperbolic by-parts became promotable
- local lane:
  - CLI probes for the two negative-affine cases
  - targeted contract smoke:
    `cargo test -p cas_cli integrate_contract_linear_times_hyperbolic_linear_by_parts -- --nocapture`
- local result:
  - baseline probes returned correct antiderivatives with no required
    conditions, but spent roughly 12-14s and emitted `depth_overflow` warnings
  - a local normalization attempt that rewrote negative affine arguments through
    hyperbolic parity did not bound the public CLI path
  - treating `integrate/int` as HoldAll avoided child pre-simplification in
    principle, but regressed already-promoted positive-affine hyperbolic
    by-parts probes from milliseconds to multi-second runs
- global result:
  - no runtime or calculus code from the attempt was retained
  - the existing positive-affine contracts were restored to their prior fast
    path; targeted hyperbolic by-parts contract passes again in about 0.07s
- reusable signature:
  - negative orientation of affine hyperbolic arguments can force the
    bottom-up simplifier into expensive hyperbolic angle expansion before or
    after integration, even when the calculus rule itself has a simple
    antiderivative
- what could make it combinable later:
  - a narrow preorder/root guard that canonicalizes `sinh/cosh(a-b*x)` to a
    positive-slope argument before expensive hyperbolic expansion, without
    making all `integrate` calls HoldAll
  - or a cheap rejection gate for nonzero linear combinations of
    `sinh/cosh(affine)` with polynomial coefficients before recursive
    hyperbolic expansion
- decision:
  - observe-only discovery; do not promote negative-affine hyperbolic by-parts
    until the simplifier has a bounded orientation fix that preserves embedded
    and positive-affine integration runtime

## 2026-05-01 - Discovery observe-only: raw rational-affine hyperbolic integration boundary

- area:
  - calculus / integration / pre-simplification boundary
- status:
  - `observe-only`
- resolved by:
  - 2026-05-22 observability cycle: the current low-level
    `cas_math` unit now retains the raw rational-affine hyperbolic by-parts
    helper path for `(x+1)*sinh((3*x+2)/2)` and the analogous `cosh`
    case; the public CLI contract also verifies positive and negative
    rational-affine forms through the bounded public residual route with no
    required conditions.
- attempted case:
  - add a low-level `cas_math` unit for
    `integrate_symbolic_expr((x+1)*sinh((3*x+2)/2), x)` and the analogous
    `cosh` case while promoting the same public CLI inputs
  - add the same two public inputs to the heavyweight
    antiderivative-by-`diff` verification list
- local lane:
  - `cargo test -p cas_math integrates_linear_times_hyperbolic_linear_by_parts -- --nocapture`
  - `cargo test --release -q -p cas_cli --test integrate_contract_tests integrate_contract_supported_antiderivatives_verify_by_differentiation -- --exact --nocapture`
  - public smoke probes:
    `cargo run -q -p cas_cli -- eval 'integrate((x+1)*sinh((3*x+2)/2), x)' --format json`
    and the analogous `cosh` probe
- local result:
  - public CLI probes completed in roughly 17-19ms, returned simplified
    antiderivatives, and emitted no required conditions
  - the direct `cas_math` helper path returned `None` for the raw rational
    affine argument, showing that this helper currently relies on the public
    pre-simplification/normalization boundary for this shape
  - the antiderivative-by-`diff` list exceeded 60s in release with these two
    cases added, so that heavyweight promotion was rejected
- global result:
  - the low-level unit promotion was not retained
  - the heavyweight antiderivative-by-`diff` promotion was not retained
  - the narrow public CLI no-condition/output contract remains the correct
    promoted surface for this iteration
- reusable signature:
  - rational affine arguments represented with explicit division can be public
    `integrate` successes while still being below the raw
    `integrate_symbolic_expr` helper's accepted polynomial-normalized input
    shape
- what could make it combinable later:
  - either normalize rational affine arguments before calling low-level
    integration helpers, or teach the polynomial extractor to accept these raw
    rational coefficient shapes directly
  - add a cheaper antiderivative verification path for rational-affine
    hyperbolic outputs before putting them in the global differentiation list
- decision:
  - observe-only discovery; do not broaden low-level unit expectations or the
    heavyweight differentiation verification list until the helper boundary and
    residual simplification cost are made explicit or normalized

## 2026-05-01 - Discovery observe-only: rational-affine hyperbolic residual verification remains slow

- area:
  - calculus / integration / antiderivative verification / hyperbolic simplification
- status:
  - `observe-only`
- resolved by:
  - 2026-05-22 observability cycle: the current rational-affine hyperbolic
    by-parts contract evaluates residuals such as
    `diff(integrate((x+1)*sinh((3*x+2)/2), x), x) - (x+1)*sinh((3*x+2)/2)`
    to `0` through the bounded public residual route in milliseconds, so this
    stale slow-verifier discovery should no longer steer candidate selection.
- attempted case:
  - after teaching polynomial extraction to accept division by a constant
    polynomial, re-probe the residual for
    `integrate((x+1)*sinh((3*x+2)/2), x)` before promoting it to the
    heavyweight antiderivative-by-`diff` contract
- local lane:
  - `gtimeout 30s cargo run -q -p cas_cli -- eval 'diff(cosh(1/2*(3*x+2))*(2/3*x+2/3) - 4/9*sinh(1/2*(3*x+2)), x) - (x+1)*sinh((3*x+2)/2)' --format json`
- local result:
  - the probe still timed out at 30s, so the earlier low-level polynomial
    boundary fix did not make the hyperbolic residual cheap enough for the
    global differentiation verifier
  - the analogous rational-affine trigonometric residuals simplified to `0`
    quickly and were promoted instead
- reusable signature:
  - rational-affine hyperbolic by-parts antiderivatives remain dominated by
    residual simplification cost, not by low-level argument extraction
- what could make it combinable later:
  - add a bounded simplification path for subtractive affine
    `sinh`/`cosh` linear combinations before retrying heavyweight
    antiderivative verification
- decision:
  - observe-only discovery; keep the hyperbolic rational-affine public output
    contract, but do not add it to the global by-`diff` verifier yet

## 2026-05-01 - Discovery observe-only: linear hyperbolic reciprocal residual verifier

- area:
  - calculus / integration / antiderivative verification / hyperbolic reciprocal simplification
- status:
  - `observe-only`
- resolved by:
  - 2026-05-22 observability cycle: current bounded public residual routes and
    compact hyperbolic log-abs residual support close the original reciprocal
    cases to `0` without `depth_overflow`, `cycle_detected`, or budget
    warnings. The public probes for `1/sinh(2*x+1)^2`,
    `sinh(2*x+1)/cosh(2*x+1)^2`,
    `cosh(2*x+1)/sinh(2*x+1)^2`, `1/tanh(2*x+1)`, and the direct
    `diff(ln(abs(sinh(2*x+1)))/2,x)-1/tanh(2*x+1)` residual now terminate
    under a 250 ms CLI time budget while preserving the expected
    `sinh(2*x+1) != 0` domain condition where applicable.
- attempted case:
  - promote linear hyperbolic reciprocal antiderivatives such as
    `integrate(1/sinh(2*x+1)^2, x)`,
    `integrate(sinh(2*x+1)/cosh(2*x+1)^2, x)`,
    `integrate(cosh(2*x+1)/sinh(2*x+1)^2, x)`, and
    `integrate(1/tanh(2*x+1), x)` into the heavyweight
    antiderivative-by-`diff` verifier
- local lane:
  - public CLI residual probes comparing `diff(antiderivative, x)` against the
    original integrand before adding the cases to
    `integrate_contract_supported_antiderivatives_verify_by_differentiation`
- local result:
  - some probes eventually reduce to `0`, but they emit repeated
    `depth_overflow` and `cycle_detected` warnings
  - the `ln(abs(sinh(2*x+1)))` residual for `1/tanh(2*x+1)` stayed on the
    expensive expansion path long enough to require manual termination
- reusable signature:
  - composed `sinh`/`cosh` reciprocal residuals can expand affine hyperbolic
    arguments into large products before proving the compact derivative
    identity
- what could make it combinable later:
  - add a bounded residual simplification path for `coth`/`sech`/`csch`-like
    derivatives, or verify these antiderivatives through a compact staged
    identity before the global residual simplifier expands affine
    `sinh`/`cosh` terms
- decision:
  - observe-only discovery; do not promote these hyperbolic reciprocal cases to
    the global by-`diff` verifier until the residual path is bounded

## 2026-05-01 - Discovery observe-only: shifted denominator-square integration after child expansion

- area:
  - calculus / integration / traversal order / expanded denominator powers
- status:
  - `observe-only`
- resolved by:
  - 2026-05-22 observability cycle: the retained follow-up is now verified by
    current contracts and CLI probes. The public syntactic form
    `integrate((2*x+1)/(x^2+x-1)^2, x)` reaches the direct `u'/u^n`
    integration step, returns `-1 / (x^2 + x - 1)`, emits no budget warnings,
    and preserves the compact `x^2 + x - 1 != 0` domain condition without a
    child `Small Multinomial Expansion` detour.
- attempted case:
  - promote the syntactic public form
    `integrate((2*x+1)/(x^2+x-1)^2, x)` after adding exact expanded-square
    denominator recovery to the integration helper
- local lane:
  - `cargo test --release -q -p cas_cli --test integrate_contract_tests integrate_contract_shifted_polynomial_derivative_over_expanded_denominator_square_preserves_nonzero_domain -- --exact --nocapture`
  - `cargo run --release -q -p cas_cli -- eval 'integrate((2*x+1)/(x^2+x-1)^2, x)' --steps on --format json`
  - `cargo run --release -q -p cas_cli -- eval 'integrate((2*x+1)/(x^4+2*x^3-x^2-2*x+1), x)' --format json`
- local result:
  - the already-expanded denominator form integrates to
    `-(1 / (x^2 + x - 1))` with required condition
    `x^2 + x - 1 ≠ 0`
  - the syntactic denominator-square form is expanded by a child
    `Small Multinomial Expansion` step inside `integrate(...)` and remains a
    residual integral, so the root integration rule is not reaching the newly
    expanded shape in that pass
- reusable signature:
  - shifted polynomial denominator powers can be integrable after expansion,
    but child expansion under `integrate(...)` can preempt the root calculus
    rule and leave a residual
- what could make it combinable later:
  - either run the root `integrate` rule before expanding children in
    `integrate(...)`, or add a bounded revisit/hold policy for calculus calls
    that preserves integration candidates before transform-phase expansion
- decision:
  - retain the low-level expanded-square integration capability and public
    expanded-denominator contract; do not promote the syntactic
    denominator-square case until the traversal-order issue is addressed
- retained follow-up:
  - a later cycle kept `Small Multinomial Expansion` out of `integrate(...)`
    descendants, preserving the syntactic `u'/u^2` denominator-power shape
    without making all `integrate` calls HoldAll
  - the syntactic public form was promoted as
    `integrate((2*x+1)/(x^2+x-1)^2, x)` with compact nonzero condition
    `x^2 + x - 1 ≠ 0`

## 2026-05-01 - Rejected combination: global diff HoldAll traversal

- area:
  - calculus / differentiation / traversal order / domain conditions
- status:
  - `rejected-combination`
- attempted combination:
  - make every `diff(...)` call HoldAll so direct derivative rules see the raw
    target before child simplification
- local lane:
  - `diff(tan(2*x+1), x)` with `--time-budget-ms 50`
- local result:
  - local runtime improved and the public result stayed
    `2 / cos(2*x + 1)^2` with `cos(2*x + 1) != 0`
- global guardrail failure:
  - `cargo test --release -q -p cas_solver --test diff_step_contract_tests -- --nocapture`
    failed
    `variable_base_power_log_diff_simplifies_constant_with_minimal_domain_conditions`
  - the global HoldAll policy dropped the normalized `x - 1 != 0` condition for
    `diff(log(x^2, x^3), x)`, leaving only `x > 0`
- reusable signature:
  - some `diff` families rely on pre-differentiation simplification to expose
    normalized domain conditions, while `tan(u)` benefits from preserving the
    raw direct target before trig quotient expansion
- retained decision:
  - reject global `diff` HoldAll
  - retain only the narrow root-`tan(...)` preservation path for direct
    derivative dispatch

## 2026-05-01 - Discovery observe-only: non-unit linear tan product diff still pressures post-simplification

- area:
  - calculus / differentiation / traversal order / product-rule trig
- status:
  - `observe-only`
- resolved by:
  - 2026-05-22 observability cycle: current affine linear-times-`tan`
    differentiation contracts and strict public probes retain the broader
    product-rule shape without `Simplification Time Budget`. The public forms
    `diff((x+1)*tan(2*x+1), x)`, `diff(2*x*tan(2*x+1), x)`, and
    `diff((3*x+2)*tan(2*x+1), x)` now return compact `tan(u) + p/cos(u)^2`
    outputs under a 50 ms time budget while preserving the single
    `cos(2*x+1) != 0` condition. The earlier integration coupling is also
    retained: `sec(u)*tan(u)` and `tan(u)/cos(u)` integrate to `sec(u)/2`.
- attempted case:
  - broaden the narrow `diff` target preservation from `x*tan(u)` to
    `linear-polynomial*tan(u)`
- local lane:
  - `cargo run --release -q -p cas_cli -- eval 'diff((x+1)*tan(2*x+1), x)' --format json --time-budget-ms 50`
  - `cargo run --release -q -p cas_cli -- eval 'diff(2*x*tan(2*x+1), x)' --format json --time-budget-ms 50`
- local result:
  - preserving the broader raw product target avoids the first destructive
    child expansion, but post-differentiation simplification still tries to
    combine tangent/product-rule terms into deeper `sin`/`cos` quotients and
    can hit `Simplification Time Budget`
- reusable signature:
  - non-unit or offset polynomial cofactors around `tan(u)` need a bounded
    post-diff presentation policy; target preservation alone is not enough
- decision:
  - do not promote the broader `linear-polynomial*tan(u)` route
  - retain only the minimal `x*tan(u)` representative that exits under budget
    with the existing product rule and `cos(u) != 0` condition

## 2026-05-01 - Discovery observe-only: shifted hyperbolic coth product diff pressures post-simplification

- area:
  - calculus / differentiation / hyperbolic quotient / post-diff simplification
- status:
  - `observe-only`
- resolved by:
  - 2026-05-22 robustness cycle: exposed the direct
    `cas_math` affine hyperbolic coth product derivative to the public
    pre-simplification `diff` path, preserving the visible
    `Symbolic Differentiation` step while bypassing the expensive compact
    reciprocal-hyperbolic sum simplification route.
- attempted case:
  - directly differentiate affine cofactors around
    `cosh(2*x+1)/sinh(2*x+1)`, for example
    `diff((x+1)*cosh(2*x+1)/sinh(2*x+1), x)`
- local lane:
  - `cargo run --release -q -p cas_cli -- eval 'diff((x+1)*cosh(2*x+1)/sinh(2*x+1), x)' --format json --time-budget-ms 50`
  - direct `cas_math` differentiation probe for the same product family
- local result:
  - the direct derivative can be built, but the public eval path still spends
    seconds in post-simplification and emits `Simplification Time Budget` for
    compact shifted `sinh(u)*cosh(u)/sinh(u)^2` outputs
  - a separate observability bug surfaced while reporting cycle events for
    displays containing `·`; that byte-boundary panic is handled separately as
    a robustness fix
- reusable signature:
  - shifted hyperbolic quotient product-rule outputs need a bounded
    presentation/normalization path before the family is safe to promote
- decision:
  - observe-only discovery; do not promote affine hyperbolic coth product diff
    until the shifted `sinh(u)*cosh(u)` post-simplification path is bounded
- retained follow-up:
  - a later cycle added a bounded direct derivative for affine cofactors around
    `cosh(u)/sinh(u)` when `u` is linear, including already-distributed
    numerator forms
  - the public no-artificial-budget contract now returns compactly as
    `1/tanh(2*x+1) - (2*x+2)/sinh(2*x+1)^2` with required condition
    `sinh(2*x+1) != 0`
  - the strict `--time-budget-ms 50` public probe now also returns the compact
    derivative in the direct path without `Simplification Time Budget`,
    preserving `sinh(2*x+1) != 0`

## 2026-05-01 - Discovery observe-only: reciprocal tanh fraction guard does not bound compact coth diff sum

- area:
  - calculus / differentiation / fraction combination / post-diff runtime
- status:
  - `observe-only`
- resolved by:
  - 2026-05-22 robustness cycle: the public eval path now preserves compact
    same-argument reciprocal-hyperbolic sums shaped like
    `1/tanh(u) - p/sinh(u)^2` before the general simplifier. The original
    strict probe for `1/tanh(2*x+1) - (2*x+2)/sinh(2*x+1)^2` returns the same
    compact form without `Simplification Time Budget` under a 50 ms budget,
    while the public required conditions normalize to `sinh(2*x+1) != 0`.
- attempted case:
  - block `AddFractions` only for compact same-argument reciprocal-hyperbolic
    sums shaped like `1/tanh(u) - p/sinh(u)^2`
- local lane:
  - `cargo test -q -p cas_math fraction_pair_guard_support::tests:: -- --nocapture`
  - `cargo test -q -p cas_engine eval::simplify_action::tests::eval_simplify_steps_off_diff_shifted_linear_times_coth_avoids_post_diff_timeout -- --nocapture`
  - `cargo run --release -q -p cas_cli -- eval '1/tanh(2*x+1) - (2*x+2)/sinh(2*x+1)^2' --format json --time-budget-ms 50 --budget small`
- local result:
  - the guard can recognize the intended denominator signature, but the public
    eval path still emits `Simplification Time Budget`
  - probes showed the expensive path is broader than a single root-level
    `AddFractions` application and includes repeated multiplication
    canonicalization/rule-attempt traversal on the compact output
- reusable signature:
  - compact reciprocal-hyperbolic sums need a bounded presentation/runtime path
    above individual fraction-pair guards; blocking one common-denominator rule
    is insufficient
- decision:
  - reject the guard and leave no code retained from this attempt

## 2026-05-01 - Discovery observe-only: product-form surd scale in inverse reciprocal trig diff

- area:
  - calculus / differentiation / inverse reciprocal trig / surd-scale
    normalization
- status:
  - `observe-only`
- resolved by:
  - 2026-05-22 robustness cycle: the public `steps_off` diff path now reuses
    the existing positive-quadratic surd-scale presentation before general
    simplification for `arcsec/arccsc(sqrt(c)*q(x))`. The original product-form
    probes for `sqrt(2)*(x^2+x+3)` return the compact
    `+/- (2*x+1)/((x^2+x+3)*sqrt(2*(x^2+x+3)^2-1))` form under a 50 ms budget
    with no `Simplification Time Budget`, no `depth_overflow`, and empty
    required conditions.
- attempted case:
  - use the product form `sqrt(2)*(x^2+x+3)` as the argument of `arcsec` and
    `arccsc`
- local lane:
  - `cargo run --release -q -p cas_cli -- eval 'diff(arcsec(sqrt(2)*(x^2+x+3)), x)' --format json --budget small`
  - `cargo run --release -q -p cas_cli -- eval 'diff(arccsc(sqrt(2)*(x^2+x+3)), x)' --format json --budget small`
- local result:
  - the quotient spelling `(x^2+x+3)/sqrt(1/2)` reaches a compact derivative
    and is now handled by the scaled sign-proof improvement
  - the product spelling triggers `depth_overflow` warnings and produces a
    much larger derivative/condition shape before the same semantic gap can be
    used safely
- reusable signature:
  - inverse reciprocal trig derivatives are sensitive to equivalent surd-scale
    orientation; quotient-form positive gaps are compact, while product-form
    surd scaling still needs a bounded normalization/presentation route
- decision:
  - observe-only discovery; do not promote the product spelling in this cycle
  - retain only the reusable scaled sign proof plus the compact quotient-form
    public contract
- retained follow-up:
  - a later cycle made the inverse reciprocal trig derivative use a
    value-preserving primitive positive gap and added a narrow raw-target
    preservation gate for `arcsec/arccsc(sqrt(c)*q(x))` when `q` is a strictly
    positive quadratic
  - the public product spelling now returns the same compact derivative as the
    quotient spelling, with no `depth_overflow` warnings and empty required
    conditions for the promoted representative

## 2026-05-02 - Discovery observe-only: shifted bounded inverse-trig integration by parts needs public residual closure

- area:
  - calculus / integration / inverse trig / affine-shifted by-parts
- status:
  - `observe-only`
- resolved by:
  - 2026-05-22 observability cycle: the documented public residual probes for
    `integrate(arcsin/arccos(2*x+1), x)` and
    `integrate(arcsin/arccos(1-2*x), x)` now all simplify through
    `diff(integrate(...), x) - integrand` to `0` without warnings, while
    preserving the explicit real-domain conditions for the shifted bounded
    inverse-trig representatives.
- attempted case:
  - broaden the retained scaled-variable rule from `integrate(arcsin(2*x), x)`
    to shifted affine arguments such as `integrate(arcsin(2*x+1), x)`
- local lane:
  - `cargo run -q -p cas_cli -- --no-pretty eval 'diff(1/2*(2*x+1)*arcsin(2*x+1)+sqrt(-x^2-x), x) - arcsin(2*x+1)'`
  - `cargo run -q -p cas_cli -- --no-pretty eval 'diff(integrate(arcsin(2*x+1), x), x) - arcsin(2*x+1)'`
- local result:
  - the manually parsed antiderivative simplifies to residual `0`
  - the public nested `diff(integrate(...), x)` path leaves a nonzero residual
    shape involving `(-x^2-x)^(-1/2)`, so the shifted affine case is not yet a
    safe public promotion
- reusable signature:
  - shifted inverse-trig by-parts needs a bounded presentation/residual
    normalization path for the exact result shape emitted by `integrate`
- decision:
  - retain only zero-offset rational scaling in this cycle
  - defer shifted affine bounded inverse-trig integration until the public
    nested verification closes without relying on a manually re-parsed formula
- retained follow-up:
  - a later cycle promoted the positive-slope shifted affine subset for
    `arcsin/arccos(2*x+1)` using the scaled by-parts form
    `1/a * (u*F(u) +/- sqrt(1-u^2))`
  - the public nested residual now closes to `0` and carries the explicit
    positive condition for the promoted representative
  - negative-slope shifted affine inputs remain deferred because the local
    probe can close mathematically while still emitting a depth-overflow signal
  - a later cycle promoted the negative-slope shifted subset for
    `integrate(arcsin/arccos(1-2*x), x)` using a factored by-parts radicand
    `(4*(x-x^2))^(1/2)` and the explicit condition `x - x^2 > 0`
  - `arccos(1-2*x)` also closes the public nested residual
    `diff(integrate(...), x) - arccos(1-2*x)` to `0`
  - a later cycle closed the remaining `arcsin(1-2*x)` public nested residual
    by extracting additive square content from reciprocal square-root powers,
    so `diff(integrate(arcsin(1-2*x), x), x) - arcsin(1-2*x)` now reduces to
    `0` while preserving `x - x^2 > 0`

## 2026-05-02 - Discovery observe-only: log-cube by-parts factored form still stresses residual closure

- area:
  - calculus / integration / logarithmic by-parts / post-diff simplification
- status:
  - `observe-only`
- resolved by:
  - 2026-05-22 observability cycle: current public integration and
    differentiation contracts retain the compact factored log-cube by-parts
    primitive for `integrate(2*x*ln(x^2+1)^3, x)`. The original direct
    `diff((x^2+1)*P(ln(x^2+1)), x) - 2*x*ln(x^2+1)^3` probe and the nested
    `diff(integrate(...), x) - integrand` probe both simplify to `0` without
    warnings or required conditions, so this entry is closed as stale
    discovery signal rather than open engine work.
- attempted case:
  - promote `integrate(2*x*ln(x^2+1)^3, x)` using the compact factored
    antiderivative `(x^2+1)*(ln(x^2+1)^3 - 3*ln(x^2+1)^2 + 6*ln(x^2+1) - 6)`
- local lane:
  - `cargo run -q -p cas_cli -- eval 'diff((x^2+1)*(ln(x^2+1)^3-3*ln(x^2+1)^2+6*ln(x^2+1)-6), x) - 2*x*ln(x^2+1)^3' --no-pretty`
  - `cargo run -q -p cas_cli -- eval 'diff(integrate(2*x*ln(x^2+1)^3, x), x) - 2*x*ln(x^2+1)^3' --no-pretty`
- local result:
  - the factored antiderivative leaves a nonzero quotient/log-power residual
    on the direct simplification path
  - the retained implementation emits the same antiderivative as a bounded
    sum of terms, which the staged contract verifier differentiates back to
    the integrand
- reusable signature:
  - products shaped as `u * P(ln(u))` with degree-three log polynomials still
    need a bounded quotient/log-power cancellation path if we want compact
    factored antiderivatives and nested `diff(integrate(...), x)` residuals to
    close directly
- decision:
  - retain the public `integrate` capability for the expanded by-parts form
  - defer compact factored presentation and direct nested residual closure to
    a future pre-calculus/simplification iteration
- retained follow-up:
  - a later cycle extended the bounded symbolic differentiator for
    `u * P(ln(u))` from the degree-two by-parts pattern to the degree-three
    pattern `ln(u)^3 - 3*ln(u)^2 + 6*ln(u) - 6`
  - direct public `diff` now maps the compact factored antiderivative back to
    `u' * ln(u)^3`, and `diff(integrate(2*x*ln(x^2+1)^3, x), x) -
    2*x*ln(x^2+1)^3` reduces to `0`
  - guardrails retained the change with `calculus_diff_contract` increasing
    to `79` cases and global failed/timeouts remaining `0`
- compact presentation follow-up:
  - a later cycle changed the retained `integrate(2*x*ln(x^2+1)^3, x)`
    presentation from the expanded by-parts sum to the factored form
    `(ln(x^2+1)^3 - 3*ln(x^2+1)^2 + 6*ln(x^2+1) - 6) * (x^2 + 1)`
  - the public nested residual still closes to `0`, so compact presentation is
    now backed by the same differentiation verification instead of being an
    answer-only shortcut

## 2026-05-02 - Discovery observe-only: rational-affine sec/csc diff still pressures half-angle cleanup

- area:
  - calculus / differentiation / reciprocal trig / post-cleanup presentation
- status:
  - `observe-only`
- resolved by:
  - 2026-05-22 observability cycle: current public differentiation contracts
    retain the rational-affine `sec`/`csc` derivative presentation. The
    strict probes for `diff(sec((3*x+2)/2), x)` and
    `diff(csc((2-3*x)/2), x)` return compact `3/2*sec(u)*tan(u)` and
    `3/2*csc(u)*cot(u)` forms under a 50 ms budget with no `depth_overflow`,
    no half-angle `+ 1 - 1` cleanup residue, and the expected
    `cos(u) != 0` / `sin(u) != 0` pole conditions.
- attempted case:
  - extend the retained rational-affine `tan`/`cot` derivative construction to
    product-shaped reciprocal trig derivatives:
    `diff(sec((3*x+2)/2), x)` and `diff(csc((2-3*x)/2), x)`
- local lane:
  - `cargo run --release -q -p cas_cli -- eval 'diff(sec((3*x+2)/2), x)' --format json`
  - `cargo run --release -q -p cas_cli -- eval 'diff(csc((2-3*x)/2), x)' --format json`
  - direct probes of equivalent compact forms such as
    `3*sin((3*x+2)/2)/(2*cos((3*x+2)/2)^2)`
- local result:
  - the mathematically compact forms still route through half-angle/post-cleanup
    and emit `depth_overflow`
  - representative public output for `sec` stayed shaped like
    `(sin(1/2*(3*x+2))*3)/((2*cos(1/2*(3*x+2))^2 + 1 - 1))`
- reusable signature:
  - rational-affine reciprocal trig derivatives with a numerator
    `sin(u)`/`cos(u)` over a squared denominator need a bounded presentation
    rule that cancels additive identity noise after half-angle normalization,
    independent of how the derivative is initially constructed
- decision:
  - observe-only discovery; do not retain the ineffective `sec`/`csc`
    construction changes in this cycle
  - retain only the `tan`/`cot` rational-affine denominator-square split that
    measurably removes warnings/nontermination in the public diff lane

## 2026-05-06 - Discovery observe-only: positive sqrt-chain csc/cot presentation is not preserved through nested diff-integrate

- area:
  - calculus / integration verification / post-calculus presentation /
    reciprocal trig sqrt chains
- status:
  - `observe-only`
- resolved by:
  - 2026-05-22 observability cycle: current public calculus lanes retain the
    compact sqrt-chain reciprocal-trig product presentation that motivated this
    discovery. The reproduced probes
    `diff(integrate(csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x)), x), x)` and
    `diff(integrate(sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x)), x), x)` now render
    as the compact product-over-`sqrt(x)` forms with no warnings and deduped
    `sin(sqrt(x)) != 0` / `cos(sqrt(x)) != 0` plus `x > 0` requirements;
    `diff(-csc(sqrt(x)), x)` also retains
    `csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x))`. Direct simplification of the raw
    integrand can still choose the canonical `sin`/`cos` quotient, so this
    closeout is scoped to the public differentiation/integration verification
    lanes rather than promoted as a general trig presentation simplifier.
- attempted case:
  - make `diff(integrate(csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x)), x), x)`
    render directly as `csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x))`
  - related probe:
    `diff(integrate(sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x)), x), x)`
- local lane:
  - `cargo run -q -p cas_cli -- eval 'diff(integrate(csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x)), x), x)'`
  - `cargo run -q -p cas_cli -- eval 'diff(-csc(sqrt(x)), x)'`
  - targeted contract probes in `integrate_contract_sqrt_chain_secant_cosecant_products_verify`
    and `elementary_sqrt_chain_rule_diff_uses_explicit_root_denominator_presentation`
- local result:
  - direct `diff(sec(sqrt(x)), x)` already preserves the compact
    `sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x))` form
  - the positive `csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x))` shape is reopened by
    later simplification into `cos(x^(1/2))*x^(-1/2)/(2*sin(x^(1/2))^2)` or
    equivalent quotient variants, even when built as a local presentation
    result
  - nested `diff(integrate(...), x)` also accumulates duplicate-looking
    conditions across `x^(1/2)` and `sqrt(x)` renderings in the failing probe
- reusable signature:
  - positive reciprocal-trig products with sqrt-chain arguments need a bounded
    post-calculus preservation or condition-normalization route; simply
    returning the compact product from the calculus rule is not retained
- decision:
  - reject the implementation changes from this cycle
  - keep this as a future calculus/pre-calculus presentation candidate, ideally
    by addressing the product preservation and condition dedupe together rather
    than adding another answer-only shortcut

## 2026-05-06 - Discovery observe-only: sqrt-chain tan/cot log-derivative presentation needs reconstruction

- area:
  - calculus / integration verification / post-calculus presentation /
    trig sqrt-chain log derivatives
- status:
  - `observe-only`
- resolved by:
  - 2026-05-22 observability cycle: current public calculus lanes retain the
    compact sqrt-chain `tan`/`cot` log-derivative presentation that motivated
    this discovery. The reproduced probes
    `diff(integrate(tan(sqrt(x))/(2*sqrt(x)), x), x)`,
    `diff(integrate(cot(sqrt(x))/(2*sqrt(x)), x), x)`, and
    `diff(integrate(tan(sqrt(2*x))/sqrt(2*x), x), x)` now render as the
    compact `tan`/`cot` over explicit `sqrt(...)` forms with no warnings and
    normalized domain requirements. Direct simplification of
    `tan(sqrt(x))/(2*sqrt(x))` and `cot(sqrt(x))/(2*sqrt(x))` also preserves
    the compact form, so the old reconstruction signature is stale for the
    promoted public integration verification lanes.
- attempted case:
  - preserve compact nested results for
    `diff(integrate(tan(sqrt(x))/(2*sqrt(x)), x), x)` and
    `diff(integrate(cot(sqrt(x))/(2*sqrt(x)), x), x)`
  - related affine probe:
    `diff(integrate(tan(sqrt(2*x))/sqrt(2*x), x), x)`
- local lane:
  - `cargo +1.91.1 run -q -p cas_cli -- eval 'diff(integrate(tan(sqrt(x))/(2*sqrt(x)), x), x)' --format json`
  - `cargo +1.91.1 run -q -p cas_cli -- eval 'diff(integrate(cot(sqrt(x))/(2*sqrt(x)), x), x)' --format json`
  - `cargo +1.91.1 run -q -p cas_cli -- eval 'tan(sqrt(x))/(2*sqrt(x))' --format json`
- local result:
  - direct simplification of the compact `tan/cot` integrand rewrites it to a
    `sin/cos` quotient with `x^(-1/2)` or equivalent sqrt-power factors
  - adding a narrow recognizer for the original `tan/cot(sqrt(p))/sqrt(p)`
    shape was not retained because the post-calculus target often arrives
    after this trig canonicalization has already happened
  - the nested output therefore remains mathematically correct but less compact,
    and can carry duplicate-looking conditions rendered across `x^(1/2)` and
    `sqrt(x)`
- reusable signature:
  - preserving these public calculus results requires a bounded presentation
    reconstructor from the canonical `sin/cos` quotient back to `tan/cot` with
    the same sqrt-chain radicand, plus condition display dedupe
- decision:
  - do not retain the ineffective syntactic recognizer in this cycle
  - keep the successfully retained hyperbolic `tanh`/`1/tanh` sqrt-chain log
    presentation as the promoted change, because those targets are not reopened
    by trig canonicalization

## 2026-05-06 - Local win / global fail: broad post-calculus presentation pass rebuilt non-target integration results

- area:
  - calculus / post-calculus presentation / integration contract stability
- status:
  - `rejected-broad-variant`
- attempted case:
  - compact `integrate(tanh(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)` from
    `ln(cosh((3*x+1)^(1/2)))` to `ln(cosh(sqrt(3*x+1)))`
- local lane:
  - `cargo +1.91.1 test -p cas_cli --test integrate_contract_tests integrate_contract_sqrt_chain_hyperbolic_tangent_logs_verify -- --exact --nocapture`
- local result:
  - adding the final presentation pass directly compacted the target case and
    preserved the displayed domain condition
- global failure signature:
  - `make engine-scorecard` failed `calculus_integrate_contract` because the
    broad pass recursively rebuilt unrelated integration results
  - failures were text-order churn in already-correct expressions, such as
    swapped product factors in by-parts antiderivatives and reordered additive
    terms in linear trig integrals
- retained correction:
  - keep only a narrow `has_compactable_ln_abs_cosh_sqrt` detection before
    rebuilding presentation output
  - this preserves the promoted hyperbolic sqrt-chain result without changing
    unrelated integration result ordering
- decision:
  - reject the broad no-op reconstruction variant
  - retain the gated presentation pass plus the hyperbolic derivative verifier

## 2026-05-06 - Local win / global fail: denominator-scale folding created fractional numerator presentation

- area:
  - calculus / post-calculus presentation / held integration results
- status:
  - `rejected-broad-variant`
- attempted case:
  - compact shifted sqrt-chain hyperbolic reciprocal derivative results from
    `-1*3/(3*cosh((3*x+1)^(1/2)))` to
    `-1 / cosh(sqrt(3*x+1))`
- local lane:
  - `cargo +1.91.1 test -p cas_cli --test integrate_contract_tests integrate_contract_sqrt_chain_hyperbolic_reciprocal_derivatives_verify -- --exact --nocapture`
  - `cargo +1.91.1 test -p cas_engine rules::calculus::compact_hold_tests::fold_numeric_mul_constants_for_hold_cancels_denominator_scale -- --exact --nocapture`
- local result:
  - a denominator-scale fold successfully cancelled `3/(3*f)` after
    reconstructing the half-power argument as `sqrt(...)`
- global failure signature:
  - `make engine-scorecard` failed `calculus_integrate_contract` because the
    broad fold also rewrote stable results like `-1 / (3*q)` into
    `-1/3 / q`
- retained correction:
  - only cancel denominator numeric factors when the quotient remains an
    integer scale, so exact scale noise like `3/(3*f)` is removed without
    creating new fractional numerators
  - add a unit guard preserving `-1 / (3 * (x^2 + x - 1))`
- decision:
  - reject the broad denominator-scale fold
  - retain the integer-only scale cancellation plus the compact hyperbolic
    reciprocal derivative presentation

## 2026-05-06 - Discovery / observe-only: negative-slope affine arctan-sqrt integration verifies poorly

- area:
  - calculus / integrate-by-diff verification / sign-orientation robustness
- status:
  - `discovery-observe-only`
- resolved by:
  - 2026-05-22 observability cycle: current public integration verification
    lanes retain the negative-slope affine `arctan(sqrt(...))` capability that
    motivated this discovery. The reproduced
    `integrate(-1/(2*sqrt(5-3*x)*(2-x)), x)` probe returns
    `arctan(sqrt(5 - 3*x))` with the explicit `x < 5/3` condition, and
    `diff(integrate(-1/(2*sqrt(5-3*x)*(2-x)), x), x) -
    (-1/(2*sqrt(5-3*x)*(2-x)))` now simplifies to `0` under the small public
    budget with no warnings. Direct `diff(arctan(sqrt(5-3*x)), x)` also renders
    the compact negative-orientation derivative; the slower direct residual
    written without the `integrate(...)` context remains a separate runtime/NF
    candidate rather than an open integration verification blocker.
- observed case:
  - `integrate(-1/(2*sqrt(5-3*x)*(2-x)), x)` structurally matches the same
    affine inverse-derivative family as `arctan(sqrt(q))` with `q = 5 - 3*x`
- local probes:
  - `cargo +1.91.1 run -q -p cas_cli -- eval 'integrate(-1/(2*sqrt(5-3*x)*(2-x)), x)' --format json`
  - `cargo +1.91.1 run -q -p cas_cli -- eval 'diff(integrate(-1/(2*sqrt(5-3*x)*(2-x)), x), x) - (-1/(2*sqrt(5-3*x)*(2-x)))' --format json`
- local result:
  - the candidate antiderivative renders as `arctan((5 - 3*x)^(1/2))`
  - the derivative is mathematically equivalent to the input, but the current
    residual simplifier does not reduce the sign-oriented affine denominator
    expression to `0`
- reusable signature:
  - negative-slope affine radicands can create equivalent denominators such as
    `12 - 6*x`, `10 - 6*x`, and `x - 2` whose sign/content relationship is not
    normalized strongly enough for antiderivative verification
- decision:
  - promote only the positive-slope affine arctan-sqrt inverse kernel in this
    cycle, where `diff` verification reaches residual `0`
  - leave negative-slope support for a later sign/orientation robustness cycle

## 2026-05-06 - Promotion note: negative-slope affine arctan-sqrt integrated via equivalence verification

- area:
  - calculus / integrate-by-diff verification / sign-orientation robustness
- status:
  - `superseded`
- promoted case:
  - `integrate(-1/(2*sqrt(5-3*x)*(2-x)), x)` now returns
    `arctan((5 - 3*x)^(1/2))` with required condition `5 - 3*x > 0`
- verification:
  - `equiv(diff(arctan(sqrt(5-3*x)), x), -1/(2*sqrt(5-3*x)*(2-x)))`
    returns `true`
- previous retained limitation:
  - the raw residual form did not simplify to `0` because affine
    denominator sign/content normalization is not strong enough across
    expressions such as `12 - 6*x`, `10 - 6*x`, and `x - 2`
- decision:
  - retain the public integration capability because the derivative equivalence
    verifier proves the antiderivative under explicit real-domain conditions
  - keep a future non-calculus candidate open for residual/NF convergence of
    sign-oriented affine denominator fractions
- superseded by:
  - the follow-up coverage cycle that normalizes sign-oriented affine
    denominator fractions with matching reciprocal square-root powers, allowing
    the raw residual
    `diff(arctan(sqrt(5-3*x)), x) - (-1/(2*sqrt(5-3*x)*(2-x)))` to simplify
    to `0`

## 2026-05-07 - Discovery / observe-only: arccot shifted reciprocal sqrt misses the direct diff shortcut

- area:
  - calculus / diff runtime / inverse tangent reciprocal sqrt presentation
- status:
  - `discovery-observe-only`
- resolved by:
  - 2026-05-22 observability cycle: current public differentiation lanes retain
    the bounded shifted reciprocal-sqrt `arccot` route that motivated this
    discovery. The reproduced
    `diff(arccot(sqrt(1/(x+1))), x)` probe returns
    `1/(2*sqrt(x+1)*(x+2))` with `x > -1`, no warnings, and a 3-step trace;
    the residual against `1/(2*sqrt(x+1)*(x+2))` simplifies to `0` under the
    small public budget. The retained contract also guards that this case avoids
    the old `Abs Squared Identity` / `Heuristic Poly Normalize` cleanup route.
    The public trace may still show the bounded `arccot(x) -> arctan(1/x)`
    transition, so this closeout is scoped to the resolved runtime cleanup gap,
    not to a pure arccot-only didactic route.
- observed case:
  - `diff(arccot(sqrt(1/(x+1))), x)` returns the compact result
    `1 / (2*sqrt(x+1)*(x+2))` with required condition `x + 1 > 0`, but still
    routes through the generic `arccot(x) -> arctan(1/x)` rewrite and the
    `abs((x+1)^(-1/2))` cleanup path
- local probes:
  - `cargo run -q -p cas_cli -- eval 'diff(arccot(sqrt(1/(x+1))), x)' --format json --steps on`
- reusable signature:
  - the direct reciprocal-sqrt inverse-tangent derivative shortcut now handles
    the arctan form after `sqrt(1/p(x)) -> abs(p(x)^(-1/2))`, but arccot can be
    rewritten to `arctan(1/abs(...))` before the calculus shortcut sees it
- decision:
  - retain only the arctan runtime shortcut in the current cycle
  - leave arccot as a later bounded calculus-routing candidate rather than
    broadening this iteration into another inverse-trig normalization path

## 2026-05-07 - Promotion note: arccot shifted reciprocal sqrt uses the direct compact diff shortcut

- area:
  - calculus / diff runtime / inverse tangent reciprocal sqrt presentation
- status:
  - `superseded`
- promoted case:
  - `diff(arccot(sqrt(1/(x+1))), x)` now returns
    `1 / (2*sqrt(x+1)*(x+2))` with required condition `x + 1 > 0`
- retained correction:
  - recognize the post-rewrite shape `arctan(1/abs((x+1)^(-1/2)))` as the
    domain-guarded `arctan(sqrt(x+1))` derivative target
  - keep the explicit positive radicand condition instead of relying on the
    `abs` cleanup route
- decision:
  - promote the bounded arccot runtime shortcut because it removes the generic
    `Abs Squared Identity` / polynomial-normalization cleanup path while
    preserving the public result and real-domain condition

## 2026-05-07 - Discovery / observe-only: compact diff presentation is not terminal across later phases

- area:
  - calculus / post-calculus presentation / didactic trace quality
- status:
  - `discovery-observe-only`
- resolved by:
  - 2026-05-22 observability cycle: current public differentiation lanes retain
    the compact post-calculus presentation trace that motivated this discovery.
    The reproduced
    `diff(arcsin((x^2+x+1)^2/sqrt(2/3))*1/sqrt(3), x)` probe returns
    `2*(2*x^3 + 3*x^2 + 3*x + 1)/sqrt(2 - 3*(x^2+x+1)^4)` with
    `2 - 3*(x^2+x+1)^4 > 0`, no warnings, and a single public
    `Symbolic Differentiation` step; the residual against the compact target
    simplifies to `0` under the small public budget. The retained contract
    also guards that the public trace suppresses redundant `Expand Expression`,
    `Expand Binomial`, and `Present calculus result in compact form` repair
    noise. This closeout is scoped to the promoted public trace case, not to a
    general terminal-presentation phase policy.
- observed case:
  - `diff(arcsin((x^2+x+1)^2/sqrt(2/3))*1/sqrt(3), x)` now starts with the
    direct `Symbolic Differentiation` step and returns the compact result
    `2*(2*x^3 + 3*x^2 + 3*x + 1)/sqrt(2 - 3*(x^2+x+1)^4)`, but later phases
    can still expand numerator/gap fragments before the final
    `Present calculus result in compact form` repair
- local probes:
  - `cargo run -q -p cas_cli --release -- eval "diff(arcsin((x^2+x+1)^2/sqrt(2/3))*1/sqrt(3), x)" --steps on --format json`
- reusable signature:
  - preserving the raw `diff` target is enough to avoid pre-derivative binomial
    expansion, but compact calculus results are not yet treated as terminal
    presentation forms across rationalize/expand phases
- decision:
  - retain the narrow raw-target preservation in this cycle
  - leave terminal post-calculus presentation as a later bounded phase-policy
    candidate, not as an ad hoc global simplifier prettification rule

## 2026-05-07 - Promotion: compact direct diff presentation collapses redundant post-diff expansion trace

- promoted case:
  - `diff(arcsin((x^2+x+1)^2/sqrt(2/3))*1/sqrt(3), x)`
- structural signature:
  - the first `Symbolic Differentiation` step already contains enough
    information to render the same compact post-calculus presentation as the
    final public result
  - later expansion and presentation-repair steps do not add mathematical
    explanation when the rendered first-step presentation exactly matches the
    final presentation
- retained correction:
  - collapse the public trace to the first derivative step at the eval
    presentation boundary only when `try_post_calculus_presentation` renders
    the first step exactly like the final result
  - keep the final result and required conditions unchanged
  - do not change global simplification, canonical algebra, or domain policy
- decision:
  - retained as a local calculus presentation improvement
  - supersedes the observe-only row above for this bounded public trace case

## 2026-05-08 - Local win / guardrail fail: broad reciprocal sqrt product merge

- area:
  - coverage / reciprocal-root products / calculus residual verification
- status:
  - `superseded-by-narrower-guarded-rule`
- local win:
  - a broad rule for `a^(-1/2)*b^(-1/2) -> (a*b)^(-1/2)` closed the residual
    `diff(arcsin(sqrt(2*x-1)), x) - 1/(sqrt(2*x-1)*sqrt(2-2*x))`
- guardrail failure:
  - `make engine-fast` failed `calculus_diff_contract` because the broad rule
    also touched direct derivative presentation cases such as
    `diff(acosh(sqrt(x)), x)` and `diff(arcsin(sqrt(x)), x)`, adding or
    reordering required conditions in already-stable public contracts
- retained learning:
  - reciprocal-root product merging is useful for residual contexts, but it
    should not run broadly inside final calculus presentation paths
  - the retained implementation requires an additive/subtractive residual
    ancestor and composed symbolic bases, preserving existing direct `diff`
    contracts while closing the targeted residual

## 2026-05-08 - Local win / guardrail fail: broad raw integrate-route preservation

- area:
  - calculus / integration runtime / nested diff verification
- status:
  - `superseded-by-family-gated-preservation`
- local win:
  - preserving raw `diff(integrate(...), x)` targets closed the residual probe
    for `integrate((x^3+x)*sin(2*x+1), x)` without the previous
    `depth_overflow` warning
- guardrail failure:
  - the broad preservation changed many stable integration-contract
    presentations by bypassing established pre/post-calculus shaping
- retained learning:
  - raw integration targets are valuable only when the integrand is a bounded
    repeated integration-by-parts kernel with a polynomial factor and direct
    trig affine argument
  - the retained implementation gates raw preservation to that family instead
    of treating all public `integrate` calls as terminal presentation forms

## 2026-05-08 - Discovery observe-only: acosh real-domain lower bound is not representable in implicit-domain facts

- area:
  - robustness / domain-regime / sign-aware abs cleanup
- status:
  - `discovery-observe-only`
- resolved by:
  - 2026-05-22 observability cycle: current public domain lanes retain the
    lower-bound representation this discovery required. The reproduced
    `abs(x)-x+acosh(x)-acosh(x)` and
    `sqrt(x^2)-x+acosh(x)-acosh(x)` probes now both simplify to `0` while
    publishing the explicit `x >= 1` lower-bound requirement with no warnings.
    The retained wire contract also covers affine lower-bound implication and
    the negative-orientation guard, so this scorecard entry is stale discovery
    signal rather than an open domain-model blocker.
- observed probes:
  - `abs(x)-x+acosh(x)-acosh(x)` returns `|x| - x` with no required
    conditions
  - `sqrt(x^2)-x+acosh(x)-acosh(x)` returns `|x| - x` with no required
    conditions
- reusable signature:
  - `acosh(u)` in the real domain requires `u >= 1`
  - the current implicit-domain vocabulary can express `u >= 0`, `u > 0`,
    and `u != 0`, but not a shifted lower bound such as `u - 1 >= 0`
  - recording only `u > 0` would let `abs(u) -> u` but would understate the
    real domain after `acosh` cancels, so it is not a retainable fix
- decision:
  - do not patch `acosh` as a weaker positivity condition
  - treat this as a future domain-model extension candidate before promoting
    `acosh`-driven abs cleanup cases

## 2026-05-08 - Retained: acosh lower-bound domain feeds sign-aware abs cleanup

- area:
  - robustness / domain-regime / sign-aware abs cleanup
- retained correction:
  - represent `acosh(u)`'s real input domain as an explicit lower-bound
    condition `u >= 1`
  - allow lower-bound facts to imply weaker nonnegative/positive targets only
    through exact polynomial-margin checks
- promoted cases:
  - `abs(x)-x+acosh(x)-acosh(x)` -> `0`, requiring `x >= 1`
  - `abs(x)-x+acosh(2*x+1)-acosh(2*x+1)` -> `0`, requiring
    `2*x + 1 >= 1`
  - `abs(x)-x+acosh(1-x)-acosh(1-x)` -> `-2*x`, requiring `x <= 0`
- decision:
  - supersedes the observe-only row above for the bounded `acosh` lower-bound
    case
  - this is not a general inequality solver; it is a conservative exact
    lower-bound implication used by existing domain-aware simplification

## 2026-05-08 - Local win / guardrail fail: numeric-first fraction numerator display

- area:
  - calculus / post-calculus presentation / formatter
- status:
  - `rejected`
- local win:
  - changing fraction numerator display to prefer numeric factors first improved
    the probe `integrate(-2*x^2/(sin(x^3)^2), x)` from
    `(cos(x^3) * 2)/(3 * sin(x^3))` to
    `(2 * cos(x^3))/(3 * sin(x^3))`
- guardrail failure:
  - `make engine-fast` failed `calculus_diff_contract`
  - the broad formatter change altered stable derivative contract strings such
    as log/abs quotient outputs from `(x * 2)/(...)` to `(2 * x)/(...)`
  - one inverse-trig derivative presentation lost the visible leading negative
    sign in `diff(arccos(sqrt(3/2)*(x^2+x+1)^2), x)`, making the broad display
    rewrite unsafe as a presentation-only change
- retained learning:
  - numerator factor ordering for final calculus presentation is valuable, but
    it cannot be changed broadly in `FractionDisplayView` without auditing sign
    extraction and existing derivative contracts
  - the next attempt should be a narrower post-integration presentation layer or
    a formatter change with explicit sign-preservation tests for inverse-trig
    derivative quotients

## 2026-05-09 - Discovery observe-only: reparsed compact quadratic-square antiderivative residual

- area:
  - calculus / integrate / post-calculus presentation / residual simplification
- status:
  - `superseded-by-retained`
- superseded by:
  - 2026-05-22 observability cycle: current public integration and residual
    lanes retain the bounded rational residual matcher from the follow-up row.
    The reproduced `integrate(1/(4*x^2+1)^2, x)` probe still renders the
    compact `1/4*arctan(2*x) + x/(2*(4*x^2+1))` primitive with no synthetic
    required conditions, and both
    `diff(integrate(1/(4*x^2+1)^2, x), x) - 1/(4*x^2+1)^2` and
    `diff(1/4*arctan(2*x)+x/(2*(4*x^2+1)), x) - 1/(4*x^2+1)^2` now simplify
    to `0` under the small public budget with no warnings. The remaining note
    about the lower-level exhaustive helper stays a separate harness/core
    alignment concern rather than an open public integration blocker.
- observed probes:
  - `integrate(1/(4*x^2+1)^2, x)` now renders the compact primitive
    `1/4*arctan(2*x) + x/(2*(4*x^2+1))`
  - `diff(integrate(1/(4*x^2+1)^2, x), x) - 1/(4*x^2+1)^2` reduces to `0`
  - reparsing the rendered primitive in
    `diff(1/4*arctan(2*x) + x/(2*(4*x^2+1)), x) - 1/(4*x^2+1)^2`
    leaves an algebraic residual instead of reducing to `0`
- reusable signature:
  - the supported integration path can verify the antiderivative while the
    standalone residual simplifier still misses a common-denominator collapse
    involving the derivative of a compact rational tail plus an arctan term
- decision:
  - retain the public `integrate(...)` capability with a nested-diff contract
  - do not promote the scaled case into a rendered-antiderivative round-trip
    lane until the public residual simplifier closes this standalone form

## 2026-05-09 - Retained: short polynomial rational residual sums close public rendered antiderivative check

- area:
  - calculus / integrate / post-calculus presentation / residual simplification
- retained correction:
  - add a bounded exact residual check for short sums of rational terms with
    polynomial denominators
  - compute a small polynomial LCM and prove the combined numerator is zero
    before collapsing the residual
- promoted case:
  - `diff(1/4*arctan(2*x) + x/(2*(4*x^2+1)), x) - 1/(4*x^2+1)^2` -> `0`
  - `integrate(1/(4*x^2+1)^2, x)` now has a public rendered-antiderivative
    verification contract
- decision:
  - supersedes the observe-only row above for this bounded rational residual
  - the matcher stays deliberately small: at most four rational terms,
    polynomial denominators, and a low-degree common denominator
  - the lower-level exhaustive helper still rejects the reparsed primitive
    because it bypasses the public engine residual route; keep that as a
    separate harness/core-alignment candidate rather than forcing promotion

## 2026-05-09 - Rejected: broad public-engine antiderivative helper

- area:
  - calculus / integrate / verification harness / runtime
- status:
  - rejected
- local win:
  - replacing the internal `Simplifier::simplify` verification helper with a
    public `Engine::eval` residual route lets the scaled compact primitive
    `integrate(1/(4*x^2+1)^2, x)` verify through the same route users exercise
- rejection reason:
  - applying that route to every existing antiderivative helper call made the
    representative smoke materially slower and left the exhaustive ignored lane
    running for more than two minutes before cancellation
- retained learning:
  - public rendered-antiderivative round-trips are valuable as targeted
    coverage, but the broad helper should not be swapped wholesale without a
    cheaper shared residual path
  - the retained direction is to promote minimal rendered round-trip assertions
    for compact public forms, not to route the entire integration verification
    suite through the public engine twice per case

## 2026-05-09 - Retained: antiderivative helper uses public fallback only after internal residual miss

- area:
  - calculus / integrate / verification harness / post-calculus residuals
- retained correction:
  - keep the internal `Simplifier::simplify` antiderivative verification as
    the fast path
  - if that direct residual does not reduce to `0`, re-check the rendered
    primitive through the public engine residual route before failing the test
- promoted case:
  - `integrate(1/(4*x^2+1)^2, x)` is now included in both the representative
    antiderivative verification smoke and the ignored exhaustive verification
    lane
- validation signal:
  - representative antiderivative smoke stayed cheap after the fallback change
  - the ignored exhaustive lane completed successfully instead of requiring
    cancellation after the broad public-route attempt
- decision:
  - this is a harness alignment improvement, not a new integration rule
  - retain the fallback because it preserves the fast path for existing cases
    while letting compact public primitives exercise the same residual route
    users see

## 2026-05-09 - Retained: antiderivative fallback observability stays sparse

- area:
  - observability / calculus / integrate / verification harness
- retained correction:
  - make the antiderivative verification helper report whether it used the
    public rendered-antiderivative fallback
  - share the representative antiderivative verification case list with a
    sparse-fallback contract
- promoted signal:
  - among the representative supported antiderivative cases, only
    `integrate(1/(4*x^2+1)^2, x)` requires the public residual fallback
- decision:
  - keep the fallback as a harness escape, not the normal verification route
  - do not move residual support into the lower-level simplifier until broader
    fallback usage appears

## 2026-05-09 - Retained: exhaustive antiderivative fallback contract stays sparse

- area:
  - observability / calculus / integrate / verification harness
- retained correction:
  - extend the ignored exhaustive antiderivative verification lane so it records
    every case that needs the public rendered-antiderivative residual fallback
  - keep the lane non-CI by default, but make it fail if fallback usage broadens
    silently
- observed signal:
  - the exhaustive supported-antiderivative list still reports only
    `integrate(1/(4*x^2+1)^2, x)` as a fallback case
- decision:
  - retain the fallback as a measured harness escape
  - defer moving this support into the lower-level simplifier until a broader
    fallback cluster appears

## 2026-05-09 - Retained: affine trig cube integration

- area:
  - calculus / integrate / trig powers / bounded power reduction
- retained correction:
  - add conservative antiderivatives for `sin(linear)^3` and `cos(linear)^3`
  - reuse the existing affine extraction and post-integration simplification
- promoted cases:
  - `integrate(sin(x)^3, x)` -> `1/3*(cos(x)^3 - 3*cos(x))`
  - `integrate(cos(x)^3, x)` -> `1/3*(3*sin(x) - sin(x)^3)`
  - affine variants such as `integrate(sin(2*x + 1)^3, x)` are verified by
    differentiating the returned antiderivative
- domain decision:
  - no new required conditions are introduced for real-domain integer powers of
    sine and cosine
- engine feedback:
  - the new cases verify through the existing antiderivative differentiation
    helper and do not broaden public residual fallback usage

## 2026-05-09 - Rejected: affine trig fifth integration before residual verification hardening

- area:
  - calculus / integrate / trig powers / antiderivative verification
- status:
  - rejected
- local win:
  - a bounded extension from cubic to fifth powers can construct direct
    primitives for `sin(linear)^5` and `cos(linear)^5`
- rejection reason:
  - public residual probes such as
    `diff(integrate(sin(x)^5, x), x) - sin(x)^5` and affine variants did not
    return cheaply enough for promotion
  - this violates the current integration policy that promoted families must be
    verified by differentiating the chosen antiderivative
- retained learning:
  - odd trig powers beyond cubic need residual simplification hardening before
    public integration promotion
  - the next retained candidate should target the reusable residual closure for
    fifth-power trig primitives, not add more answer-only integration patterns

## 2026-05-09 - Discovery observe-only: sqrt-log reciprocal residual still misses shared scalar factor

- area:
  - calculus / diff / post-calculus presentation / residual simplification
- status:
  - `superseded-by-retained`
- superseded by:
  - 2026-05-22 observability cycle: current public differentiation and
    residual lanes retain the bounded reciprocal half-power matcher from the
    follow-up row. The reproduced `diff(sqrt(ln(x)), x)` probe now renders as
    `1/(2*x*sqrt(ln(x)))`, and both
    `ln(x)^(-1/2)/(2*x) - ln(x)^(1/2)/(2*x*ln(x))` and
    `diff(sqrt(ln(x)), x) - 1/(2*x*sqrt(ln(x)))` simplify to `0` under the
    small public budget with no warnings while preserving `ln(x) > 0` and
    `x > 0`. The plus-constant follow-up also keeps the same bounded residual
    route covered for `diff(sqrt(ln(x)+c), x)` forms.
- observed probes:
  - `diff(sqrt(ln(x)), x)` returns the correct internal form
    `ln(x)^(-1/2) / (2*x)` with `ln(x) > 0` and `x > 0`
  - `ln(x)^(-1/2)/x - ln(x)^(1/2)/(x*ln(x))` reduces to `0`
  - `diff(sqrt(ln(x)), x) - 1/(2*x*sqrt(ln(x)))` still leaves
    `ln(x)^(-1/2) / (2*x) - ln(x)^(1/2) / (2*x*ln(x))`
- retained learning:
  - positive-log condition normalization can remove the redundant
    `x - 1 != 0` guard, but residual closure still misses the same
    reciprocal-power equivalence when a shared scalar product such as `2*x`
    is present
- decision:
  - retain only the domain-condition cleanup in this iteration
  - treat the remaining residual as a separate bounded simplification candidate
    for shared-denominator reciprocal fractional powers

## 2026-05-09 - Retained: sqrt-log shared denominator reciprocal residual

- area:
  - coverage / calculus feedback / reciprocal half-power residuals
- retained correction:
  - add a bounded root residual matcher for
    `u^(-1/2)/k - u^(1/2)/(k*u) -> 0`
  - keep the matcher exact: the denominator factor set must match after
    removing one copy of `u` from the expanded side
- promoted probes:
  - `ln(x)^(-1/2)/(2*x) - ln(x)^(1/2)/(2*x*ln(x)) -> 0`
  - `diff(sqrt(ln(x)), x) - 1/(2*x*sqrt(ln(x))) -> 0`
- domain decision:
  - no new assumptions are introduced by the matcher
  - public output keeps `ln(x) > 0` and `x > 0`
- rejection guard:
  - mismatched scales such as
    `ln(x)^(-1/2)/(2*x) - ln(x)^(1/2)/(3*x*ln(x))` are rejected
- engine feedback:
  - the retained value is reusable residual closure for reciprocal
    fractional powers with functional bases, discovered through a calculus
    presentation miss

## 2026-05-09 - Retained: sqrt-log plus-constant residual fast path

- area:
  - runtime / calculus feedback / post-diff residuals
- retained correction:
  - add a bounded root residual matcher for
    `diff(sqrt(ln(x)+c), x) - 1/(2*x*sqrt(ln(x)+c)) -> 0`
  - keep the matcher exact: the log argument must be the differentiation
    variable and the denominator must contain precisely `2`, that variable,
    and the matching square root
- promoted probes:
  - `diff(sqrt(ln(x)+1), x) - 1/(2*x*sqrt(ln(x)+1)) -> 0`
  - `diff(sqrt(ln(x)+2), x) - 1/(2*x*sqrt(ln(x)+2)) -> 0`
- domain decision:
  - public output keeps both required conditions:
    `ln(x)+c > 0` and `x > 0`
  - mismatched variables and mismatched scales are rejected
- engine feedback:
  - some correct residuals were reaching expensive general simplification
    after differentiation; a cheap pre-route is retainable when the full
    chain-rule identity and domain witnesses are syntactically visible

## 2026-05-09 - Discovery observe-only: shifted quadratic cubic repeated-pole residual does not close

- area:
  - calculus / integrate / rational partial fractions / residual simplification
- status:
  - `superseded-by-retained`
- superseded by:
  - 2026-05-22 observability cycle: the adjacent retained residual closure now
    covers this discovery in the public calculus path. Reproduced probes show
    `integrate(1/((x+1)^3*(x^2+2*x+2)), x)` still renders the compact primitive
    `1/2*ln(x^2+2*x+2) - ln(abs(x+1)) - 1/(2*(x+1)^2)`, while both the nested
    `diff(integrate(...), x)` residual and the explicit rendered primitive
    residual simplify to `0` under the small public budget with no warnings and
    only the retained `x + 1 != 0` domain condition.
- observed probes:
  - `integrate(1/((x+1)^3*(x^2+2*x+2)), x)` now constructs a compact
    primitive:
    `1/2*ln(x^2 + 2*x + 2) - ln(|x + 1|) - 1/(2*(x + 1)^2)`
  - `diff(integrate(1/((x+1)^3*(x^2+2*x+2)), x), x) - 1/((x+1)^3*(x^2+2*x+2))`
    does not currently collapse to `0`
- retained learning:
  - cubic repeated-pole integration over a shifted definite quadratic can
    produce a valid-looking primitive, but public residual closure is weaker
    than for the centered quadratic representative
  - promote only the centered minimal representative in the current iteration
    and treat shifted-quadratic residual closure as a separate simplification
    candidate

## 2026-05-10 - Retained: shifted quadratic cubic repeated-pole residual closure

- area:
  - engine / coverage / rational residual simplification after calculus
- status:
  - `retained`
- retained probes:
  - `diff(integrate(1/((x+1)^3*(x^2+2*x+2)), x), x) - 1/((x+1)^3*(x^2+2*x+2)) -> 0`
  - `Requires: x + 1 ≠ 0`
- retained learning:
  - the integration formula was already valid; the weak point was public
    post-calculus residual closure when eval runs hidden simplification in
    compact step mode
  - a final bounded polynomial-denominator residual check in the
    post-calculus residual lane is enough to close the case without widening
    integration rules

## 2026-05-10 - Discovery observe-only: shifted polynomial-arctan by-parts verification is too expensive

- area:
  - calculus / integrate / inverse trig by parts / residual simplification
- status:
  - `observe-only`
- superseded by:
  - 2026-05-22 observability cycle: current public calculus residual lanes now
    cover the exact shifted affine polynomial-arctan probe cheaply. Reproduced
    probes show `integrate((x+1)*arctan(2*x+1), x)` returns an explicit
    primitive with no required conditions, while both
    `diff(integrate((x+1)*arctan(2*x+1), x), x) - (x+1)*arctan(2*x+1)` and
    `equiv(diff(integrate((x+1)*arctan(2*x+1), x), x), (x+1)*arctan(2*x+1))`
    complete under the small public budget with no warnings. This is covered by
    the retained inverse-trig antiderivative residual shortcut and the
    polynomial-arctan affine by-parts family rather than a new integration rule.
- observed probes:
  - `integrate((x+1)*arctan(2*x+1), x)` can construct an explicit primitive
    through the same by-parts identity used for unshifted polynomial-arctan
    products
  - `diff(integrate((x+1)*arctan(2*x+1), x), x) - (x+1)*arctan(2*x+1)` did
    not collapse cheaply during contract promotion
  - `equiv(diff(integrate((x+1)*arctan(2*x+1), x), x), (x+1)*arctan(2*x+1))`
    returned `true`, but only after repeated depth-overflow warnings and an
    expensive simplification path
- retained learning:
  - shifted affine polynomial-arctan by-parts is mathematically reusable, but
    current residual/equivalence closure is not cheap enough for live
    promotion
  - retain the unshifted `arctan(a*x)` subset first and treat shifted affine
    verification as a separate post-calculus simplification/runtime candidate

## 2026-05-10 - Discovery observe-only: degree-six polynomial-arctan by-parts residual misses cancellation

- area:
  - calculus / integrate / inverse trig by parts / residual simplification
- status:
  - `superseded-by-retained`
- superseded by:
  - 2026-05-22 observability cycle: the adjacent retained symbolic
    polynomial-denominator residual closure now covers this discovery in the
    public calculus path. Reproduced probes show
    `integrate(x^6*arctan(x), x)` still returns the compact degree-six
    by-parts primitive, while both
    `diff(integrate(x^6*arctan(x), x), x) - x^6*arctan(x)` and the explicit
    rendered-primitive residual simplify to `0` under the small public budget
    with no warnings and no required conditions. The later inverse-trig
    antiderivative residual shortcut keeps the same case cheap without
    broadening integration.
- observed probes:
  - temporary promotion of `integrate(x^6*arctan(x), x)` constructed a compact
    primitive:
    `1/7*x^7*arctan(x) - 1/42*x^6 - 1/14*x^2 + 1/14*ln(x^2 + 1) + 1/28*x^4`
  - focused public contract verification failed on
    `diff(integrate(x^6*arctan(x), x), x) - x^6*arctan(x)`
  - the residual kept a rational expression with terms like
    `x^3 + x - x*(x^2 + 1)` instead of canceling the polynomial numerator to
    zero
- retained learning:
  - the by-parts construction and positive-quadratic rational integral can
    produce a plausible degree-six primitive, but public residual closure is
    not yet strong enough for live promotion
  - retain the bounded degree-five subset first; treat degree-six as a future
    post-calculus residual simplification candidate rather than broadening the
    integration family further

## 2026-05-10 - Retained: symbolic polynomial-denominator residual closure unlocks degree-six arctan by parts

- area:
  - engine / coverage / post-calculus rational residual simplification
- status:
  - `retained`
- retained probes:
  - `diff(integrate(x^6*arctan(x), x), x) - x^6*arctan(x) -> 0`
  - `integrate(x^6*arctan(x), x)` now returns
    `1/7*x^7*arctan(x) - 1/42*x^6 - 1/14*x^2 + 1/14*ln(x^2 + 1) + 1/28*x^4`
- retained learning:
  - the reusable missing capability was not a broader integration rule; it was
    a bounded residual cancellation where polynomial denominators carry
    symbolic opaque coefficients such as `arctan(x)` in the numerator
  - keeping the checker gated by small denominator degree, small term count,
    and polynomial-identity proof preserves the global guardrails while
    allowing the integration cap to move from degree five to degree six

## 2026-05-10 - Retained: suppress non-actionable depth warning for closed embedded calculus residual

- area:
  - engine / robustness / post-calculus residual verification
- status:
  - `retained`
- retained probes:
  - `diff(integrate(x^6*arctan(x), x), x) - x^6*arctan(x) -> 0`
  - the same public residual no longer emits `depth_overflow` on stderr
- retained learning:
  - embedded calculus residuals can briefly traverse a deep internal form
    before the post-calculus residual lane closes them to zero
  - suppressing depth-overflow warnings only for add/sub residuals that already
    contain embedded calculus calls keeps the public output clean without
    changing the final simplification semantics

## 2026-05-10 - Retained: inverse-trig antiderivative residual root shortcut

- area:
  - engine / runtime / post-calculus residual verification
- status:
  - `retained`
- retained probes:
  - `diff(integrate(x^6*arctan(x), x), x) - x^6*arctan(x) -> 0`
  - local CLI timing for that residual dropped from roughly `386-397ms` to
    `4-9ms`
- retained learning:
  - once a bounded integration family is already verified by construction,
    `diff(integrate(f, x), x) - f` can use a narrow root shortcut instead of
    paying for full expansion and residual simplification
  - reusing the exact polynomial-arctan integration matcher keeps the shortcut
    conservative: it does not broaden integration, assume new domains, or mask
    unrelated residual failures

## 2026-05-10 - Retained: degree-five rational antiderivative residual gate

- area:
  - engine / runtime / post-calculus rational residual verification
- status:
  - `retained`
- retained probes:
  - `diff(integrate(1/((x+1)^3*(x^2+2*x+2)), x), x) - 1/((x+1)^3*(x^2+2*x+2)) -> 0`
  - `Requires: x + 1 ≠ 0`
  - local CLI timing for that residual dropped from roughly `162ms` to
    `36-37ms`
- retained learning:
  - the integration family was already public and verified; the expensive
    part was residual recognition falling through to full simplification
  - degree-five rational residual recognition is safe when it is gated by the
    same repeated-linear-times-positive-quadratic decomposition used by the
    integrator, rather than by constructing the full antiderivative only to
    decide if the shortcut applies

## 2026-05-10 - Retained: direct conditions for degree-five rational residuals

- area:
  - engine / runtime / post-calculus rational residual verification
- status:
  - `retained`
- retained probes:
  - `diff(integrate(1/((x+1)^3*(x^2+2*x+2)), x), x) - 1/((x+1)^3*(x^2+2*x+2)) -> 0`
  - `Requires: x + 1 ≠ 0`
  - local CLI timing for that residual moved from roughly `36-37ms` to
    `30-32ms`
- retained learning:
  - after a degree-five rational residual has already matched the structural
    repeated-linear-times-positive-quadratic family, routing conditions
    through the general integration condition aggregator is unnecessary work
  - returning the nonzero linear factor from the same decomposition preserves
    domain reporting while reducing residual verification cost without
    broadening the integration rule

## 2026-05-10 - Retained: degree-four numerator over positive quadratic cube residual closure

- area:
  - calculus / integration / post-calculus residual verification
- status:
  - `retained`
- discovered case:
  - `integrate(x^4/(x^2+1)^3, x)`
  - candidate antiderivative:
    `3/8*arctan(x) + (3*x^3+5*x)/(8*(x^2+1)^2) - x/(x^2+1)`
- local lane:
  - focused CLI probe during bounded `q(x)^3` numerator-degree reduction
- local result:
  - the integrator can construct a mathematically valid primitive by reducing
    `x^4/q^3` into existing `q^-2` and `q^-3` subfamilies
  - public residual did not collapse to `0`:
    `diff(integrate(x^4/(x^2+1)^3,x),x)-x^4/(x^2+1)^3`
- promotion result:
  - the rendered primitive mixes an arctan term with rational terms over
    `q`, `q^2`, `q^3`, and `q^4`, and now verifies publicly to `0` through the
    bounded polynomial-fraction residual checker
- retained learning:
  - degree-three reduction is safe to promote because it verifies directly
  - degree-four positive-quadratic-cube reduction is retainable once the
    bounded post-calculus residual path proves the rendered antiderivative
  - the existing polynomial-fraction residual checker can use its
    already-bounded LCM degree 8 path for individual denominator powers up to
    degree 8 without broadening the integration rule blindly

## 2026-05-10 - Retained: bounded cross-symbol exact-zero residual noise

- area:
  - engine / robustness / post-calculus residual wrappers
- status:
  - `retained`
- pre-promotion probe:
  - `(((((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)/(x+2))+(x-x))+(y-y))`
- pre-promotion result:
  - the cross-symbol `+(y-y)` variant panicked before the residual pre-route
    could close it
- retained result:
  - the bounded residual pre-route now strips two exact-zero additive noise
    layers and returns `1 / (x + 2)` with only `x + 2 != 0`
  - the retained smoke representative uses one same-symbol and one
    cross-symbol exact-zero noise layer rather than duplicating `x-x`

## 2026-05-10 - Retained: third exact-zero residual noise layer

- area:
  - engine / robustness / post-calculus residual wrappers
- status:
  - `retained`
- pre-promotion probe:
  - `((((((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)/(x+2))+(x-x))+(y-y))+(z-z))`
- pre-promotion result:
  - hiperbólicas timed out under the public CLI residual probe, while the
    rational representative still reached `1 / (x + 2)` slowly
- retained result:
  - the residual pre-route now strips up to three exact-zero additive noise
    layers before quotient matching
  - the smoke matrix keeps a single representative `plus_triple_noise`; deeper
    noise remains intentionally outside this cycle rather than becoming an
    unbounded simplification rule

## 2026-05-10 - Retained: neutral multiplicative wrapper after triple residual noise

- area:
  - engine / robustness / post-calculus residual wrappers
- status:
  - `retained`
- pre-promotion probe:
  - `((((((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)/(x+2))+(x-x))+(y-y))+(z-z))*1`
- pre-promotion result:
  - simple `*1`, `1*`, and `/1` wrappers already passed, but the composition
    of `plus_triple_noise` with outer `*1` timed out for hyperbolic residuals
    and was slow for the rational representative
- retained result:
  - after removing a constant factor, the residual quotient pre-route now
    re-applies bounded exact-zero noise stripping before quotient matching
  - the smoke matrix promotes only the composed `plus_triple_noise_times_one`
    representative rather than all neutral multiplicative variants

## 2026-05-10 - Retained: shifted quotient denominator residual conditions

- area:
  - engine / robustness / post-calculus residual shifted quotient
- status:
  - `retained`
- pre-promotion probes:
  - `((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)/((diff(integrate(x^4*cosh(2*x+1),x),x)-x^4*cosh(2*x+1))+1)`
  - `((diff(integrate(1/((x+1)^3*(x^2+2*x+2)),x),x)-1/((x+1)^3*(x^2+2*x+2)))+1)/((diff(integrate(1/(x^2+1),x),x)-1/(x^2+1))+1)`
- pre-promotion result:
  - the numerator residual compacted, but the denominator residual stayed in
    the public required conditions as `NonZero(R + 1)`
  - this produced correct-looking results with a noisy, structurally provable
    condition
- retained result:
  - shifted residual denominators are compacted to their nonzero constant for
    quotient matching
  - structural `NonZero(R + c)` conditions are filtered only when `R` is a
    supported integral residual and `c != 0`, while preserving real required
    conditions needed to prove `R = 0`
  - the residual smoke matrix promotes two representatives: same-family
    hyperbolic and cross-family rational-over-reciprocal-trig

## 2026-05-10 - Retained: negative shifted quotient denominator presentation

- area:
  - engine / calculus / post-calculus residual presentation
- status:
  - `retained`
- pre-promotion probe:
  - `((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)/((diff(integrate(x^4*cosh(2*x+1),x),x)-x^4*cosh(2*x+1))-1)`
- pre-promotion result:
  - the denominator residual was correctly proved as `-1`, but the public
    result rendered as `-1 / 1`
- retained result:
  - quotient residual presentation now divides by any nonzero rational
    denominator constant produced by the residual proof, not only by `1`
  - the smoke matrix promotes one minimal negative shifted denominator
    representative and keeps broader scalar variants as discovery-only unless
    they expose a distinct failure signature

## 2026-05-10 - Retained: scaled shifted quotient denominator residuals

- area:
  - engine / calculus / post-calculus residual presentation
- status:
  - `retained`
- pre-promotion probe:
  - `((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)/(2*((diff(integrate(x^4*cosh(2*x+1),x),x)-x^4*cosh(2*x+1))+1))`
- pre-promotion result:
  - the numerator residual compacted, but the denominator stayed as
    `2*(R + 1)` and left a residual `NonZero(R + 1)` condition
- retained result:
  - shifted residual denominator compaction now accepts one nonzero rational
    factor outside the residual passthrough
  - structural condition filtering uses the same bounded factor stripping, so
    the public result is `1/2` with no residual required condition
  - the smoke matrix promotes one scaled-denominator representative rather
    than all scalar variants

## 2026-05-10 - Retained: scaled numerator and denominator residual quotient

- area:
  - engine / robustness / post-calculus residual wrappers
- status:
  - `retained`
- pre-promotion probe:
  - `3*((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)/(2*((diff(integrate(x^4*cosh(2*x+1),x),x)-x^4*cosh(2*x+1))+1))`
- pre-promotion result:
  - denominator-side scaling was already retained, but adding a nonzero
    rational scale on the numerator made the public residual probe time out
    before the direct residual quotient route could recognize `R + c`
- retained result:
  - the quotient residual pre-route now strips one nonzero rational factor from
    the numerator before matching the shifted residual passthrough
  - numerator and denominator scales are combined as rationals, so the public
    result is `3/2` with no residual required condition
  - the smoke matrix promotes one scalar composition representative rather
    than enumerating equivalent scalar variants

## 2026-05-10 - Retained: residual factor inside product denominator

- area:
  - engine / robustness / post-calculus residual wrappers
- status:
  - `retained`
- pre-promotion probe:
  - `3*((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)/(((diff(integrate(x^4*cosh(2*x+1),x),x)-x^4*cosh(2*x+1))+1)*(x+2))`
- pre-promotion result:
  - the numerator residual compacted, but the denominator product kept a
    structurally provable residual factor as a public `NonZero(R + 1)`
    condition
  - this produced a correct-looking quotient only after carrying avoidable
    post-calculus condition noise
- retained result:
  - denominator compaction now replaces one proved nonzero residual factor
    inside a product while preserving `NonZero` requirements for the external
    denominator factors
  - the public result is `3 / (x + 2)` with only `x + 2` as a required
    condition
  - the smoke matrix promotes one representative composition rather than all
    equivalent product-denominator scalar variants

## 2026-05-10 - Retained: negative residual factor product denominator presentation

- area:
  - calculus / post-calculus residual presentation
- status:
  - `retained`
- pre-promotion probe:
  - `3*((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)/((x+2)*((diff(integrate(x^4*cosh(2*x+1),x),x)-x^4*cosh(2*x+1))-1)*(x+3))`
- pre-promotion result:
  - the residual denominator factor was correctly proved as `-1`, but the
    public result kept the sign in the denominator as
    `3 / (-(x + 2)·(x + 3))`
- retained result:
  - quotient presentation now removes one exact `-1` product factor from a
    compact residual denominator and moves the sign to the numerator
  - the public result is `-3 / ((x + 2)·(x + 3))`
  - required conditions stay attached only to the real external denominator
    factors: `x + 2` and `x + 3`

## 2026-05-10 - Retained: residual numerator inside fraction denominator

- area:
  - calculus / post-calculus residual presentation
- status:
  - `retained`
- pre-promotion probe:
  - `3*((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)/(((diff(integrate(x^4*cosh(2*x+1),x),x)-x^4*cosh(2*x+1))-1)/(x+2))`
- pre-promotion result:
  - the residual denominator numerator stayed visible as
    `3 / ((R - 1) / (x + 2))`
  - the public required conditions also retained the structural residual
    condition `NonZero(R - 1)` even though the residual is proved as `0`
- retained result:
  - fraction denominators now compact one proved nonzero residual numerator and
    divide by the resulting constant
  - the public result is `-3·(x + 2)`
  - the structural residual condition is filtered while preserving the real
    external denominator condition `x + 2`

## 2026-05-10 - Retained coverage: scaled residual fraction denominator

- area:
  - calculus / post-calculus residual smoke coverage
- status:
  - `retained`
- pre-promotion probes:
  - `3*(R1+1)/((R2+1)/(x+2))`
  - `3*(R1+1)/(2*((R2-1)/(x+2)))`
  - `3*(R1+1)/(2*((R2+1)/(x+2)))`
- retained result:
  - the engine already compacted the family correctly, so no engine code
    change was needed
  - the smoke matrix promotes only the minimal scaled negative representative:
    `3*(R1+1)/(2*((R2-1)/(x+2))) -> -3/2·(x + 2)`
  - required conditions stay limited to the external denominator factor:
    `x + 2`
- promotion rationale:
  - this protects the combination of a fraction denominator, proved nonzero
    residual numerator, sign orientation, and rational external scale without
    promoting near-duplicate positive-orientation variants

## 2026-05-10 - Retained coverage: positive quadratic residual fraction denominator

- area:
  - calculus / post-calculus residual smoke coverage
- status:
  - `retained`
- pre-promotion probes:
  - `3*(R1+1)/((R2-1)/(x^2+1))`
  - `3*(R1+1)/((R2-1)/((x-1)*(x+1)))`
  - cross-family `rational_quad` over `recip_trig` with denominator `x^2+1`
- retained result:
  - the engine already compacted the positive quadratic denominator case
    correctly, so no engine code change was needed
  - the smoke matrix promotes only the minimal positive-quadratic
    representative:
    `3*(R1+1)/((R2-1)/(x^2+1)) -> -3·(x^2 + 1)`
  - required conditions are empty because `x^2 + 1` is nonzero over the real
    domain
- promotion rationale:
  - previous promoted fraction-denominator cases covered linear external
    factors with explicit `NonZero` requirements
  - this adds the semantic/domain regime where the external denominator is
    structurally present but contributes no real-domain requirement

## 2026-05-12 - Discovery observe-only: sqrt(tan) post-calculus presentation

- area:
  - calculus / post-calculus presentation
- status:
  - `resolved`
- candidate:
  - render `diff(sqrt(tan(x)), x)` as
    `1 / (2*cos(x)^2*sqrt(tan(x)))`
- smoke outcome:
  - the direct public result can be rendered compactly, but the corresponding
    verification
    `equiv(diff(sqrt(tan(x)), x), 1/(2*cos(x)^2*sqrt(tan(x))))`
    returns `false`
  - the residual mixes `tan(x)` with `sin(x) / cos(x)` under reciprocal square
    roots and trips a `depth_overflow` in post-cleanup
- learning:
  - do not promote tan/cot square-root derivative presentation until the
    trig-quotient equivalence path can normalize `sqrt(tan(x))` and
    `sqrt(sin(x) / cos(x))` without cycling
  - this is an equivalence/normalization weakness, not a calculus derivative
    rule gap

## 2026-05-13 - Discovery observe-only: squared wrapper over integration-by-parts antiderivative

- area:
  - coverage / calculus-integrate contextual embedding
- status:
  - `resolved`
- candidate:
  - promote the squared passthrough wrapper
    `(((integrate(x^2*sin(x),x))^2)+m) - (((2*x*sin(x)+(2-x^2)*cos(x))^2)+m)`
    as a `calculus_integrate` embedded-equivalence row
- smoke outcome:
  - the direct public antiderivative is stable:
    `integrate(x^2*sin(x),x) -> 2*x*sin(x) + (2 - x^2)*cos(x)`
  - additive, scaled, common-denominator, and shifted-quotient wrappers collapse
    quickly
  - the squared wrapper did not return within the cheap probe budget and was
    killed before promotion
- learning:
  - start `calculus_integrate` embedded coverage with basic wrappers only
  - defer squared wrappers for integration-by-parts antiderivatives until there
    is a focused runtime/normalization hypothesis for trig-polynomial products

## 2026-05-13 - Discovery observe-only: reversed integration-by-parts residual orientation

- area:
  - coverage / calculus-integrate contextual embedding
- status:
  - `discovery/observe-only`
- superseded by:
  - 2026-05-22 observability cycle: the adjacent retained signed common-factor
    residual work now covers this reversed orientation in the public embedded
    calculus-integrate lane. Reproduced probes show
    `((2*x*sin(x)+(2-x^2)*cos(x)) - integrate(x^2*sin(x),x)) + (u*v + u*w - u*(v+w))`
    simplifies to `0` under the small public budget with no warnings and no
    required conditions, and the promoted
    `calculus_integrate_poly_sin_by_parts_reversed_collect_combined_zero`
    embedded row is present in the live corpus. The retained value is the
    algebraic signed common-factor cancellation, not a broader integration
    shortcut.
- candidate:
  - promote the reversed combined residual
    `((2*x*sin(x)+(2-x^2)*cos(x)) - integrate(x^2*sin(x),x)) + (u*v + u*w - u*(v+w))`
    as a `calculus_integrate` orientation row
- smoke outcome:
  - forward combined, depth4 nested-fraction, and factor-mix wrappers collapse
    to `0`
  - `equiv(integrate(x^2*sin(x),x), 2*x*sin(x)+(2-x^2)*cos(x))` returns `true`
  - the reversed residual remains as
    `2 * cos(x) + 2 * x * sin(x) - (2 * x * sin(x) + (2 - x^2) * cos(x)) - cos(x) * x^2`
- learning:
  - do not mark `calculus_integrate` orientation as covered until the residual
    path can normalize the public antiderivative form and the user target form
    in both subtraction directions
  - this is a contextual residual/orientation gap, not an integration rule gap

## 2026-05-13 - Retained: signed common-factor residual closes integration orientation

- area:
  - coverage / exact-zero core / calculus-integrate contextual embedding
- status:
  - `retained`
- retained result:
  - the structural common-factor zero core now accepts a binary sum factor with
    internal sign, such as `cos(x)*(2-x^2)`
  - this closes the reversed residual
    `((2*x*sin(x)+(2-x^2)*cos(x)) - integrate(x^2*sin(x),x)) + (u*v + u*w - u*(v+w))`
    without adding an integration-specific shortcut
- promotion rationale:
  - the promoted embedded row covers `calculus_integrate` orientation with a
    minimal collect companion core
  - the reusable engine capability is algebraic signed common-factor
    cancellation, not broader integration search

## 2026-05-13 - Rejected partial: broad small exact-zero core re-entry

- area:
  - robustness / transform re-entry / exact-zero additive core
- status:
  - `rejected`
- local win:
  - wiring `try_build_small_direct_zero_core_rewrite` directly into the
    `Add/Sub` transform path collapsed the isolated residual
    `2*cos(x) + 2*x*sin(x) - (2*x*sin(x)+(2-x^2)*cos(x)) - x^2*cos(x)`
    after child simplification
- global result:
  - `make engine-scorecard` regressed `calculus_integrate_contract` with a stack
    overflow in
    `integrate_contract_repeated_linear_times_definite_quadratic_partial_fraction`
- retained subset:
  - only the narrower exact-zero additive-combination re-entry was kept
  - the failing broad direct-core re-entry was removed
- learning:
  - exact-zero core helpers may run deeper proof/simplify paths than their name
    suggests
  - transform-level re-entry should stay behind narrow candidate gates or
    condition-free additive-combination shapes, not a general small-core probe

## 2026-05-13 - Discovery observe-only: broad quadratic affine-log by-parts residuals

- area:
  - calculus / integrate / residual verification
- status:
  - `discovery/observe-only`
- superseded by:
  - 2026-05-22 calculus cycle: the bounded quadratic affine-log by-parts path now
    accepts nonzero affine slopes, including the negative-slope representative
    `integrate(x^2*ln(1-2*x), x)`, while preserving the explicit real-domain
    condition `x < 1/2`. A narrow residual shortcut for the same supported
    `diff(integrate(target,x),x) - target` family keeps the public verification
    at `0` under the small budget with no warnings. The earlier mixed positive
    cofactor representative `integrate((x^2+x)*ln(x+1), x)` also verifies
    cleanly, so the retained value is the bounded affine-log by-parts vertical
    plus residual closure, not a general integration search.
- candidate:
  - broaden affine-log integration by parts from `x^2*ln(2*x+1)` to mixed
    quadratic cofactors and negative affine arguments, such as
    `integrate((x^2+x)*ln(x+1), x)` and `integrate(x^2*ln(1-2*x), x)`
- smoke outcome:
  - the retained positive-slope monomial case verifies cleanly:
    `diff(integrate(x^2*ln(2*x+1), x), x) - x^2*ln(2*x+1) -> 0`
  - the broader mixed-cofactor residual stayed nonzero and emitted
    `depth_overflow` warnings in cheap probes
  - the negative-slope monomial produced a plausible antiderivative, but its
    residual did not collapse to `0`
- learning:
  - keep the promoted affine-log by-parts slice restricted to the minimal
    monomial quadratic case with positive affine argument
  - broader polynomial affine-log support should wait for a residual
    normalization/presentation hypothesis rather than being promoted as a
    larger integration rule

## 2026-05-13 - Discovery observe-only: arctan orientation presentation vs residual reachability

- area:
  - calculus / post-calculus presentation / inverse-trig orientation
- status:
  - `discovery/observe-only`
- superseded by:
  - 2026-05-22 observability cycle: current public calculus and simplification
    paths already close the reproducible arctan-orientation signature. The
    primitive for `integrate(x^2*arctan(1-x), x)` renders with `arctan(1-x)`,
    `diff(integrate(x^2*arctan(1-x),x),x) - x^2*arctan(1-x)` collapses to
    `0`, the composed residual
    `diff(integrate(x^2*arctan(1-x)+x*arctan(1-x),x),x) - (x^2*arctan(1-x)+x*arctan(1-x))`
    also collapses to `0`, and the public simplifier proves
    `arctan(1-x)+arctan(x-1) -> 0` under the small budget with no warnings. The
    retained value is therefore ledger hygiene and scorecard signal accuracy,
    not a new integration or orientation rewrite.
- candidate:
  - orient the rational correction inside
    `integrate(x^2*arctan(1-x), x)` from `arctan(x-1)` to `arctan(1-x)` using
    the real-domain oddness identity `arctan(-u) = -arctan(u)`
- smoke outcome:
  - the direct primitive rendered more coherently and
    `diff(integrate(x^2*arctan(1-x), x), x) - x^2*arctan(1-x)` still collapsed
    to `0`
  - the composed residual
    `diff(integrate(x^2*arctan(1-x)+x*arctan(1-x), x), x) - (x^2*arctan(1-x)+x*arctan(1-x))`
    regressed to a nonzero rational residual
  - a follow-up pair-planner probe for `arctan(1-x)+arctan(x-1)` worked at the
    local planner level but did not reach the public simplification route
- learning:
  - do not promote post-calculus arctan orientation by rewriting the held
    rational correction alone
  - the next retained move should first observe why inverse-trig additive rules
    that work for numeric atan pairs do not reach symbolic opposite-affine
    pairs after alias/child rewrites
  - keep the existing verified antiderivative form until residual reachability
    and public presentation can improve together

## 2026-05-13 - Rejected partial: global direct-difference equivalence first

- area:
  - robustness / equivalence / calculus verification
- status:
  - `rejected`
- local win:
  - trying `simplify(A-B)` before `expand(A-B)` made the public verification
    `equiv(diff(integrate(x^2*cos(2*x+1)+x*cos(2*x+1), x), x), x^2*cos(2*x+1)+x*cos(2*x+1))`
    return `true` immediately
- global result:
  - applying that order globally stalled `make engine-fast` inside
    `calculus_diff_contract`
- retained subset:
  - keep the direct-first route only when either side of the public equivalence
    still contains a calculus call (`diff`, `integrate`, or `limit`)
  - preserve the previous expand-first behavior for ordinary equivalence
    traffic, with the existing direct fallback only after the expand attempt
- learning:
  - direct residual proof is valuable for public calculus verification, where
    an unevaluated calculus call can make the expand route much more expensive
    than the direct simplifier route
  - broad equivalence ordering changes are runtime-sensitive and should stay
    behind structural gates until pressure lanes show a wider safe pattern

## 2026-05-13 - Discovery observe-only: direct diff of linear partial-fraction integrals

- area:
  - calculus / diff-integrate presentation / rational partial fractions
- status:
  - `discovery/observe-only`
- superseded by:
  - 2026-05-13 retained follow-up plus 2026-05-22 observability revalidation:
    current public calculus presentation already preserves the raw supported
    `diff(integrate(...), x)` target for the rational linear partial-fraction
    representative. Reproduced probes show
    `diff(integrate(1/((x-2)*(x-1)*x*(x+1)*(x+2)), x), x)` returns the compact
    integrand directly, the residual against the same integrand simplifies to
    `0`, and the public `equiv(...)` probe returns `true`, all under the small
    budget with no warnings while preserving the five expected nonzero
    conditions. The retained value is therefore the adjacent robustness
    follow-up; this discovery is closed as stale scorecard signal rather than a
    new calculus implementation target.
- candidate:
  - route `diff(integrate(f,x),x)` for rational linear partial-fraction targets
    directly to `f`, using the same detector as public residual verification
- retained subset:
  - the root residual/equivalence form is retained:
    `equiv(diff(integrate(1/((x-2)*(x-1)*x*(x+1)*(x+2)), x), x), 1/((x-2)*(x-1)*x*(x+1)*(x+2)))`
    now returns `true` without `depth_overflow`
- smoke outcome:
  - the direct form
    `diff(integrate(1/((x-2)*(x-1)*x*(x+1)*(x+2)), x), x)` still reaches the
    differentiated log antiderivative and emits `depth_overflow` warnings
    before rendering the compact rational result
- learning:
  - the residual/equivalence path and the standalone derivative presentation
    path are distinct: the former can be fixed by the root residual matcher,
    while the latter needs an earlier raw `diff(integrate(...),x)` gate before
    child integration rewrites consume the call
  - the next retained move should target that direct presentation route without
    changing the already verified residual/equivalence behavior

## 2026-05-13 - Retained follow-up: direct diff of linear partial-fraction integrals

- area:
  - robustness / calculus / diff-integrate presentation
- status:
  - `retained`
- retained result:
  - `diff(integrate(1/((x-2)*(x-1)*x*(x+1)*(x+2)), x), x)` now preserves the
    raw supported `integrate(...)` target long enough for `DiffRule` to return
    the compact integrand directly
  - the direct route no longer differentiates the expanded logarithmic
    antiderivative and no longer emits `depth_overflow`
- implementation note:
  - the transform pre-pass now preserves raw `diff(integrate(...),x)` targets
    only when the integrand is accepted by the existing rational linear
    partial-fraction detector
  - the existing residual/equivalence route is unchanged
- validation:
  - focal direct and public equivalence probes stayed quiet
  - `make engine-fast`, `make engine-scorecard`, and
    `make engine-scorecard-pressure` all passed with `failed=0`

## 2026-05-13 - Discovery observe-only: sine-square antiderivative presentation under affine product verification

- area:
  - calculus / post-calculus presentation / trig residual verification
- status:
  - `discovery/observe-only`
- superseded by:
  - 2026-05-22 observability cycle: current public integration presentation and
    residual verification already retain the square primitive that motivated
    this discovery. Reproduced probes show
    `integrate(3*sin(2*x+1)*cos(2*x+1), x)` renders as
    `3/4*sin(2*x+1)^2`, the nested public residual
    `diff(integrate(3*sin(2*x+1)*cos(2*x+1), x), x) - 3*sin(2*x+1)*cos(2*x+1)`
    simplifies to `0`, and the explicit square-primitive residual
    `diff(3/4*sin(2*x+1)^2, x) - 3*sin(2*x+1)*cos(2*x+1)` also simplifies to
    `0`, all under the small budget with no warnings or required conditions.
    The retained value is therefore the existing bounded affine
    trig-product integration and residual route, not a new presentation rule.
- candidate:
  - render `integrate(sin(u)*cos(u), x)` as the compact square primitive
    `sin(u)^2/(2*u')` for affine `u`
- smoke outcome:
  - standalone integration rendered correctly for `u = 2*x + 1`
  - direct differentiation of the rendered primitive was mathematically correct
  - the public residual
    `diff(integrate(3*sin(2*x+1)*cos(2*x+1), x), x) - 3*sin(2*x+1)*cos(2*x+1)`
    did not collapse to `0`; it stayed in the equivalent double-angle/product
    residual family
- learning:
  - for this cycle, retain the product-to-double-angle primitive
    `-cos(2*u)/(4*u')`, which avoids the square-power residual presentation
    path
  - a future presentation pass can revisit the prettier `sin(u)^2/(2*u')`
    form once affine double-angle/product residual normalization is stronger

## 2026-05-14 - Discovery observe-only: negative affine hyperbolic power residual verification

- area:
  - calculus / integration contract / explicit derivative residual
- status:
  - `discovery/observe-only`
- candidate:
  - promote affine sign/order variants for
    `integrate(2*cosh(2*x+1)*sinh(2*x+1)^2, x)`, including the negative
    orientation
    `integrate(-2*cosh(2*x+1)*sinh(2*x+1)^2, x)`
- smoke outcome:
  - standalone integration succeeds and renders
    `-1/3 * sinh(2 * x + 1)^3`
  - the additive residual
    `diff(-1/3*sinh(2*x+1)^3, x) + 2*cosh(2*x+1)*sinh(2*x+1)^2`
    collapses to `0`
  - the subtractive residual shape
    `diff(-1/3*sinh(2*x+1)^3, x) - (-2*cosh(2*x+1)*sinh(2*x+1)^2)`
    did not finish within the cheap focal-test budget
- learning:
  - the mathematical capability exists, but the `diff(...) - negative_target`
    residual route is structurally weaker than the equivalent additive route
  - do not promote the affine negative hyperbolic-power representative until
    explicit derivative residual normalization handles this shape directly and
    cheaply
- superseded by:
  - 2026-05-22 observability cycle: the retained follow-up below already
    promoted the bounded affine hyperbolic cubic residual route. Reproduced
    public probes now show
    `integrate(-2*cosh(2*x+1)*sinh(2*x+1)^2, x)` renders as
    `-1/3*sinh(2*x+1)^3`, the explicit subtractive residual
    `diff(-1/3*sinh(2*x+1)^3, x) - (-2*cosh(2*x+1)*sinh(2*x+1)^2)` collapses
    to `0`, and the nested public residual
    `diff(integrate(-2*cosh(2*x+1)*sinh(2*x+1)^2, x), x) - (-2*cosh(2*x+1)*sinh(2*x+1)^2)`
    also collapses to `0`, all under the small budget with no warnings or
    required conditions. The retained value is therefore the already-promoted
    explicit affine `sinh`/`cosh` cubic residual matcher, not a new rule.

## 2026-05-14 - Retained follow-up: negative affine hyperbolic power residual verification

- area:
  - robustness / calculus / explicit derivative residual
- status:
  - `retained`
- retained result:
  - `diff(-1/3*sinh(2*x+1)^3, x) - (-2*cosh(2*x+1)*sinh(2*x+1)^2)`
    now collapses to `0` through the root residual path
  - the public contract for
    `integrate(-2*cosh(2*x+1)*sinh(2*x+1)^2, x)` is promoted with the compact
    primitive `-1/3 * sinh(2 * x + 1)^3`
- implementation note:
  - the retained matcher is limited to explicit affine `sinh`/`cosh` cubic
    primitives and their chain-rule product target
  - it avoids widening generic `diff(...) - target` matching
- validation:
  - focal residual unit, affine hyperbolic integration contract,
    `make engine-fast`, `make engine-scorecard`, and
    `make engine-scorecard-pressure` all passed with `failed=0`

## 2026-05-14 - Discovery observe-only: sign-sensitive reciprocal log-power residual

- area:
  - calculus / integration contract / explicit derivative residual
- status:
  - `discovery/observe-only`
- candidate:
  - promote the non-positive-base representative
    `integrate(2*x/((x^2-1)*ln(x^2-1)^2), x)`
- smoke outcome:
  - standalone integration renders the expected reciprocal-log primitive under
    the real-domain requirement `x^2 - 1 > 0`
  - the explicit residual
    `diff(integrate(2*x/((x^2-1)*ln(x^2-1)^2), x), x) - 2*x/((x^2-1)*ln(x^2-1)^2)`
    did not collapse to `0` in the cheap public probe
- learning:
  - the reciprocal log-power integration rule is usable for the positive
    representative `x^2 + 1`, but the sign-sensitive denominator family still
    needs residual/domain normalization before public promotion
  - keep `x^2-1` as a future robustness candidate rather than widening the
    integration rule or weakening the verification contract
- superseded by:
  - 2026-05-22 observability cycle: the retained follow-up below already
    promoted the sign-sensitive reciprocal log-power residual route.
    Reproduced public probes now show
    `integrate(2*x/((x^2-1)*ln(x^2-1)^2), x)` renders as
    `-1/ln(x^2-1)`, the nested public residual
    `diff(integrate(2*x/((x^2-1)*ln(x^2-1)^2), x), x) - 2*x/((x^2-1)*ln(x^2-1)^2)`
    collapses to `0`, and the explicit primitive residual
    `diff(-1/ln(x^2-1), x) - 2*x/((x^2-1)*ln(x^2-1)^2)` also collapses to
    `0`, all under the small budget with no warnings. The retained domain
    requirements remain explicit as `x^2 - 2 != 0` and `x^2 - 1 > 0`, so this
    discovery is stale scorecard signal rather than an open integration or
    residual blocker.

## 2026-05-14 - Retained follow-up: sign-sensitive reciprocal log-power residual

- area:
  - calculus / integration contract / explicit derivative residual
- status:
  - `retained`
- retained result:
  - `integrate(2*x/((x^2-1)*ln(x^2-1)^2), x)` now renders
    `-1 / ln(x^2 - 1)`
  - the nested residual `diff(integrate(...), x) - ...` collapses to `0`
  - required real-domain conditions stay explicit: `x^2 - 1 > 0`,
    `x^2 - 2 != 0`
- implementation note:
  - retained through existing reciprocal log-power integration and
    constant/rational presentation scaling; no wider integration search
- validation:
  - focal unit/contract plus `make engine-fast` and `make engine-scorecard`
    passed with `failed=0`

## 2026-05-14 - Retained follow-up: shifted reciprocal log-power residual conditions

- area:
  - calculus / integration contract / required-condition normalization
- status:
  - `retained`
- retained result:
  - `integrate((2*x+1)/((x^2+x-1)*ln(x^2+x-1)^2), x)` renders
    `-1 / ln(x^2 + x - 1)`
  - the nested residual `diff(integrate(...), x) - ...` collapses to `0`
  - required real-domain conditions stay compact: `x - 1 != 0`,
    `x + 2 != 0`, `x^2 + x - 1 > 0`
- implementation note:
  - the fix is in display normalization of `NonZero` conditions for n-ary sums
    with a common factor, so the residual no longer surfaces a redundant
    expanded denominator guard containing `ln(...)^2`
  - no wider integration search was added
- validation:
  - focal domain-normalization unit, integration unit/contract,
    `make engine-fast`, `make engine-scorecard`, and
    `make engine-scorecard-pressure` passed with `failed=0`

## 2026-05-14 - Retained follow-up: reciprocal log-power antiderivative verification

- area:
  - calculus / integration contract / explicit derivative residual
- status:
  - `retained`
- retained result:
  - `integrate((2*x+1)/((x^2+x-1)*ln(x^2+x-1)^3), x)` renders
    `-1 / (2 * ln(x^2 + x - 1)^2)`
  - both `diff(integrate(...), x) - ...` and
    `diff(-1/(2*ln(x^2+x-1)^2), x) - ...` collapse to `0`
  - required real-domain conditions stay compact: `x - 1 != 0`,
    `x + 2 != 0`, `x^2 + x - 1 > 0`
- implementation note:
  - added a bounded family predicate for polynomial reciprocal log-derivative
    targets and reused it in the calculus residual verifier
  - no wider integration search was added
- validation:
  - focal integration unit, residual unit, public integration contract,
    `make engine-fast`, `make engine-scorecard`, and
    `make engine-scorecard-pressure` passed with `failed=0`

## 2026-05-14 - Retained coverage: repeated partial-fraction residual under composition

- area:
  - coverage / calculus-integrate / embedded equivalence context
- status:
  - `retained`
- retained result:
  - promoted one live embedded row for the repeated-linear partial-fraction
    representative
    `diff(integrate((3*x+5)/(x^3-x^2-x+1),x),x) - (3*x+5)/(x^3-x^2-x+1)`
    under a `combined_additive_zero` wrapper
  - this covers the expanded cubic denominator with a repeated pole while also
    checking re-entry through independent collect-style additive noise
- why minimal:
  - the public integration contract already covers the standalone integral and
    direct residual
  - the embedded corpus already had a simple linear partial-fraction direct
    diff row, but not the repeated-pole expanded-denominator shape under
    cross-family composition
- validation:
  - focal public probes, embedded corpus, `make engine-fast`,
    `make engine-scorecard`, and `make engine-scorecard-pressure` passed with
    `failed=0`

## 2026-05-14 - Discovery observe-only: reciprocal wrapper re-entry for partial-fraction diff residual

- area:
  - coverage / calculus-integrate / reciprocal shifted wrapper
- status:
  - `discovery/observe-only`
- superseded by:
  - `2026-05-14 - Retained follow-up: reciprocal wrapper re-entry for partial-fraction diff residual`
  - current public probes for the direct residual and the reciprocal shifted
    wrapper collapse to `0`, and the representative is present as the live
    embedded row
    `calculus_integrate_repeated_linear_partial_fraction_reciprocal_shifted_difference_zero`
- candidate:
  - promote
    `1/((diff(integrate((3*x+5)/(x^3-x^2-x+1),x),x))+c) - 1/(((3*x+5)/(x^3-x^2-x+1))+c)`
    as a `reciprocal_shifted_difference_zero` row
- smoke outcome:
  - the direct residual collapses to `0`
  - the equivalent explicit rational reciprocal shape collapses to `0`
  - the candidate with `diff(integrate(...),x)` inside the reciprocal wrapper
    remains as a nonzero-looking difference after one side is re-entered and
    denominator terms are ordered differently
- learning:
  - the calculus capability is correct, but reciprocal shifted wrappers need a
    safer re-entry/canonicalization path for diff-integrate residuals over
    rational partial fractions
  - do not promote reciprocal wrappers for this family until that structural
    gap is fixed directly

## 2026-05-14 - Retained follow-up: reciprocal wrapper re-entry for partial-fraction diff residual

- area:
  - robustness / calculus-integrate / reciprocal shifted wrapper
- status:
  - `retained`
- retained result:
  - `1/((diff(integrate((3*x+5)/(x^3-x^2-x+1),x),x))+c) - 1/(((3*x+5)/(x^3-x^2-x+1))+c)`
    now collapses to `0`
  - the previously observe-only representative is promoted as a live
    `reciprocal_shifted_difference_zero` embedded row
  - required conditions stay bounded to the repeated-pole domain plus the
    shared shifted denominator nonzero condition
- implementation note:
  - added a bounded root matcher for unit reciprocal denominators with one
    shared additive shift whose remaining cores are a supported
    `diff(integrate(T),x)` rational residual pair
  - the matcher does not generalize rational integration or arbitrary
    reciprocal equality; it is gated by the existing rational antiderivative
    verifier
- validation:
  - focal residual unit, public CLI probe, embedded equivalence context,
    `make engine-fast`, `make engine-scorecard`, and
    `make engine-scorecard-pressure` passed with `failed = 0`

## 2026-05-14 - Retained coverage: sine by-parts reciprocal shifted wrapper

- area:
  - coverage / calculus-integrate / reciprocal shifted wrapper
- status:
  - `retained`
- retained result:
  - promoted
    `1/((integrate(x^2*sin(x),x))+c) - 1/((2*x*sin(x)+(2-x^2)*cos(x))+c)`
    as a live `reciprocal_shifted_difference_zero` embedded row
  - this closes the sine half of the earlier integration-by-parts reciprocal
    wrapper discovery without adding new engine logic
- why minimal:
  - the existing live reciprocal row covered `integrate(x^2*cos(x),x)`, while
    this representative covers the distinct public sine primitive
    `2*x*sin(x)+(2-x^2)*cos(x)`
  - no extra noise or deeper shell was promoted
- validation:
  - focused embedded candidate smoke, embedded equivalence context,
    `make engine-fast`, and `make engine-scorecard` passed with `failed = 0`

## 2026-05-14 - Retained robustness: hyperbolic by-parts reciprocal wrapper re-entry

- area:
  - robustness / calculus-integrate / reciprocal shifted wrapper
- status:
  - `retained`
- retained result:
  - `1/((integrate(x^2*sinh(x),x))+c) - 1/((x^2*cosh(x)-2*x*sinh(x)+2*cosh(x))+c)`
    now collapses to `0`
  - the live embedded corpus now includes the representative as
    `calculus_integrate_poly_sinh_by_parts_reciprocal_shifted_difference_zero`
- implementation note:
  - added a bounded reciprocal-shifted matcher for supported polynomial
    hyperbolic by-parts integrals
  - comparison is against the antiderivative generated by the existing
    integrator, with a small polynomial `sinh/cosh` cofactor comparison so
    `(x^2+2)*cosh(x)` and `x^2*cosh(x)+2*cosh(x)` can match
  - no new integration family or arbitrary reciprocal equality was added
- validation:
  - focal unit, public CLI probe, negative-var probe, and focused embedded
    candidate smoke passed before promotion
  - embedded equivalence context, `make engine-fast`, `make engine-scorecard`,
    and `make engine-scorecard-pressure` passed with `failed = 0`

## 2026-05-14 - Retained coverage: atanh-surd integration reciprocal wrapper

- area:
  - coverage / calculus-integrate / reciprocal shifted wrapper
- status:
  - `retained`
- retained result:
  - promoted
    `1/((integrate(2*x/(3-x^4),x))+c) - 1/((atanh(x^2/sqrt(3))/sqrt(3))+c)`
    as a live `reciprocal_shifted_difference_zero` embedded row
  - this extends the existing atanh-surd representative beyond additive
    passthrough without adding a new integration rule
- why minimal:
  - focused smoke also showed common-denominator and shifted-quotient wrappers
    pass, but only the reciprocal shifted wrapper was promoted because it is
    the strongest non-linear wrapper among the cheap candidates
  - the earlier squared integration-by-parts wrapper probe still did not finish
    within a cheap budget, so it remains out of live promotion
- validation:
  - focused embedded candidate smoke, embedded equivalence context,
    `make engine-fast`, and `make engine-scorecard` passed with `failed = 0`

## 2026-05-14 - Retained calculus: inverse-trig sqrt derivative presentation

- area:
  - calculus / post-calculus presentation / differentiation
- status:
  - `retained`
- retained result:
  - `diff(sqrt(arcsin(x)), x)` now renders as
    `1 / (2 * sqrt(1 - x^2) * sqrt(arcsin(x)))`
  - `diff(sqrt(arccos(x)), x)` now renders as
    `-1 / (2 * sqrt(1 - x^2) * sqrt(arccos(x)))`
  - required real-domain conditions stay visible:
    `1 - x^2 > 0` plus the positive outer square-root radicand
- implementation note:
  - extended the existing bounded `sqrt(elementary function)` post-diff
    presentation path with a `sqrt(1 - arg^2)` denominator shape for
    `arcsin`/`arccos`
  - no global simplification preference and no new differentiation rule were
    added
- validation:
  - focal public wire tests, public CLI probes, `calculus_diff_contract`,
    `make engine-fast`, and `make engine-scorecard` passed with `failed = 0`

## 2026-05-14 - Retained calculus: sqrt(tanh) derivative presentation

- area:
  - calculus / post-calculus presentation / differentiation
- status:
  - `retained`
- retained result:
  - `diff(sqrt(tanh(x)), x)` now renders as
    `1 / (2 * cosh(x)^2 * sqrt(tanh(x)))`
  - the public residual
    `diff(sqrt(tanh(x)), x) - 1/(2*cosh(x)^2*sqrt(tanh(x)))`
    collapses to `0`
  - `equiv(diff(sqrt(tanh(x)), x), 1/(2*cosh(x)^2*sqrt(tanh(x))))`
    returns `true`
  - required real-domain conditions stay compact: `tanh(x) > 0`
- implementation note:
  - extended the existing bounded `sqrt(elementary function)` post-diff
    presentation path by mapping `tanh` to the same denominator-square shape
    used by `tan`, with `cosh(arg)^2`
  - no global simplification preference and no new differentiation rule were
    added
- rejected subset:
  - `sqrt(atanh(x))` presentation was rejected because the compact
    `1 - x^2` denominator form degraded the direct result condition from
    `1 - x^2 > 0` to nonzero factors
  - `sqrt(acosh(x))` presentation was rejected because the direct result looked
    compact, but the residual against that presented form no longer collapsed
    to `0`
- validation:
  - focal public wire tests, public CLI residual/equiv probes,
    `calculus_diff_contract`, `make engine-fast`, and `make engine-scorecard`
    passed with `failed = 0`

## 2026-05-14 - Retained coverage: hyperbolic sqrt diff presentation guards

- area:
  - coverage / calculus-diff / post-calculus presentation
- status:
  - `retained`
- retained result:
  - promoted the existing public behavior for `diff(sqrt(sinh(x)), x)`:
    `cosh(x) / (2 * sqrt(sinh(x)))`
  - promoted the existing public behavior for `diff(sqrt(cosh(x)), x)`:
    `sinh(x) / (2 * sqrt(cosh(x)))`
  - both residual checks collapse to `0`, preserving the current real-domain
    conditions (`sinh(x) > 0` for the first case, none for `cosh(x)`)
- why minimal:
  - this guards the direct hyperbolic numerator-function branch of
    `sqrt(elementary function)` derivative presentation
  - nearby contracts already covered trig, log, reciprocal trig, `tanh`, and
    `asinh`; these two rows close the missing direct hyperbolic representatives
    without adding composition or runtime pressure
- observe-only discovery:
  - retrying `sqrt(acosh(x))` presentation as
    `1/(2*sqrt(x-1)*sqrt(x+1)*sqrt(acosh(x)))` was rejected before promotion
  - the direct result looked compact and `equiv(...)` stayed `true`, but the
    residual no longer collapsed to `0` and an extra redundant product-domain
    condition appeared
  - reusable signature: post-calculus presentation that splits a reciprocal
    half-power product into multiple `sqrt` factors can be readable but may
    weaken residual cancellation/domain normalization unless the radical-product
    path is hardened first
- validation:
  - focal public wire tests, public CLI residual/equiv probes,
    `calculus_diff_contract`, `make engine-fast`, and `make engine-scorecard`
    passed with `failed = 0`

## 2026-05-14 - Retained calculus: acosh sqrt affine residual verification

- area:
  - calculus / diff residual verification / inverse hyperbolic presentation
- status:
  - `retained`
- retained result:
  - the public residual
    `diff(sqrt(acosh(2*x+3)), x) - 1/(sqrt(2*x+2)*sqrt(2*x+4)*sqrt(acosh(2*x+3)))`
    now collapses to `0`
  - the existing simple residual
    `diff(sqrt(acosh(x)), x) - 1/(2*sqrt(x-1)*sqrt(x+1)*sqrt(acosh(x)))`
    remains `0`
  - `equiv(diff(sqrt(acosh(2*x+3)), x), 1/(sqrt(2*x+2)*sqrt(2*x+4)*sqrt(acosh(2*x+3))))`
    remains `true`
  - required real-domain conditions stay compact through the public wire path:
    `x + 1 > 0` and `acosh(2*x + 3) > 0`
- implementation note:
  - added a bounded residual matcher for `diff(sqrt(acosh(arg)), x)` against
    the split radical denominator form `sqrt(arg-1)*sqrt(arg+1)*sqrt(acosh(arg))`
  - the matcher builds the `arg-1` and `arg+1` factors through the existing
    polynomial representation and reuses unordered sqrt-denominator matching
  - direct post-calculus presentation for `sqrt(acosh(...))` remains deferred;
    this cycle only hardens residual verification for the compact target form
- validation:
  - focal residual helper unit, public wire residual test, public CLI
    residual/equiv probes, and `calculus_diff_contract` passed

## 2026-05-14 - Retained calculus: acosh sqrt affine diff presentation

- area:
  - calculus / diff / post-calculus presentation
- status:
  - `retained`
- retained result:
  - `diff(sqrt(acosh(2*x+3)), x)` now renders publicly as
    `1/(sqrt(2*x+2)*sqrt(2*x+4)*sqrt(acosh(2*x+3)))`
  - the matching residual
    `diff(sqrt(acosh(2*x+3)), x) - 1/(sqrt(2*x+2)*sqrt(2*x+4)*sqrt(acosh(2*x+3)))`
    still collapses to `0`
  - `equiv(diff(sqrt(acosh(2*x+3)), x), 1/(sqrt(2*x+2)*sqrt(2*x+4)*sqrt(acosh(2*x+3))))`
    still returns `true`
  - required real-domain conditions stay compact through the public wire path:
    `x + 1 > 0` and `acosh(2*x + 3) > 0`
- implementation note:
  - extended the existing bounded `sqrt(elementary function)` diff presentation
    path with an `acosh(arg)` denominator shape based on `sqrt(arg-1)` and
    `sqrt(arg+1)`
  - the `arg-1` and `arg+1` factors are built via the polynomial
    representation, so affine arguments render as simplified public factors
    such as `2*x+2` and `2*x+4`
- rejected subset:
  - direct presentation for the simple derivative `diff(sqrt(acosh(x)), x)`
    remains intentionally deferred
  - the compact split form looked readable, but introduced a redundant
    product-domain condition and weakened the simple residual path; the
    retained rule therefore skips the derivative-`1` case
  - reusable signature: split-radical presentation should be promoted only when
    residual verification and domain normalization are already strong enough for
    that shape
- validation:
  - focal public wire presentation/residual tests, public CLI residual probes,
    `calculus_diff_contract`, `make engine-fast`, and `make engine-scorecard`
    passed with `failed = 0`

## 2026-05-14 - Retained robustness: scaled reciprocal half-power residual

- area:
  - robustness / calculus presentation residual / reciprocal half powers
- status:
  - `retained`
- retained result:
  - the residual shape
    `1/2*(acosh(x)*(x^2-1))^(-1/2) - sqrt(acosh(x)*(x^2-1))/(2*acosh(x)*(x^2-1))`
    now collapses to `0`
  - the public `diff(sqrt(acosh(x)), x)` path remains on the previous internal
    product-of-half-powers presentation, preserving compact required
    conditions: `acosh(x) > 0` and `x - 1 > 0`
  - the public residual
    `diff(sqrt(acosh(x)), x) - 1/(2*sqrt(x-1)*sqrt(x+1)*sqrt(acosh(x)))`
    remains `0`
- implementation note:
  - hardened the existing reciprocal half-power shared-denominator root
    shortcut so it can compare a rationally scaled reciprocal half-power
    against `sqrt(A)/(k*A)` even when the numeric content is inside one product
    denominator factor
  - this is a local residual-verification improvement, not a global radical
    presentation preference
- rejected subset:
  - direct presentation of `diff(sqrt(acosh(x)), x)` as
    `1/(2*sqrt(x-1)*sqrt(x+1)*sqrt(acosh(x)))` was retried and rejected in
    this cycle
  - although the formula and residual can now be verified, the direct public
    route still accumulates the redundant intermediate condition
    `acosh(x)*(x^2-1) >= 0`; presentation should wait for a domain/presentation
    path that avoids that noise
- validation:
  - focal public wire residual tests, public CLI probes, and
    `calculus_diff_contract`, `make engine-fast`, and `make engine-scorecard`
    passed with `failed = 0`

## 2026-05-14 - Discovery observe-only: trig fourth-power integration

- area:
  - calculus / integrate / trig power reduction / antiderivative verification
- status:
  - `observe-only`
- resolved by:
  - `2026-05-14 - Retained robustness: trig fourth-power primitive residual`
  - `2026-05-14 - Retained calculus: trig fourth-power integration`
  - current public probes for `integrate(sin(x)^4, x)`,
    `integrate(cos(2*x+1)^4, x)`, the nested
    `diff(integrate(...), x) - ...` residual, and the explicit primitive
    residual return compact results or `0` without required conditions or
    `depth_overflow`
- candidate:
  - add direct support for `integrate(sin(linear)^4, x)` and
    `integrate(cos(linear)^4, x)` using the standard fourth-power reduction
    primitive
- local signal:
  - the primitive is easy to construct and can render compactly as combinations
    of `x`, `sin(2*u)`, and `sin(4*u)`
  - wrapping the primitive can avoid public result simplification cost for the
    direct `integrate(...)` command
- rejection reason:
  - public antiderivative verification by differentiating the rendered
    primitive is not yet robust enough for this family
  - unprotected `sin(4*x)` triggers angle-doubling simplification and can emit
    debug `depth_overflow` warnings
  - the alternative `sin(2*x)*cos(2*x)` primitive avoids the quadruple-angle
    display, but still makes public verification too slow
- reusable structural signature:
  - before promoting even trig powers beyond square power reduction, harden a
    bounded residual/presentation path that verifies
    `d/dx(3*x/8 +/- sin(2*x)/4 + sin(4*x)/32) = sin(x)^4|cos(x)^4`
    without expanding into a costly double-angle route
- next candidate:
  - add a focused calculus residual verifier for trig fourth-power reduction, or
    a post-calculus presentation guard that preserves quadruple-angle terms only
    when the corresponding derivative residual can be checked cheaply

## 2026-05-14 - Retained robustness: trig fourth-power primitive residual

- area:
  - robustness / calculus residual verification / trig fourth-power reduction
- status:
  - `retained`
- retained result:
  - explicit fourth-power antiderivative residuals now collapse through the
    bounded calculus residual path:
    `diff(3*x/8 - sin(2*x)/4 + sin(4*x)/32, x) - sin(x)^4`
  - the matching cosine residual also collapses:
    `diff(3*x/8 + sin(2*x)/4 + sin(4*x)/32, x) - cos(x)^4`
  - affine variants such as `sin(2*x+1)^4` and `cos(2*x+1)^4` are covered by
    the same matcher
  - public probes return `0` without required conditions or `depth_overflow`
- implementation note:
  - added a narrow primitive-shape matcher for
    `3*x/8 +/- sin(2*u)/(4*a) + sin(4*u)/(32*a)`, where `u` is affine with
    slope `a`
  - the matcher verifies the residual without expanding `sin(4*u)` into a
    costly double-angle route
  - this does not yet promote `integrate(sin(linear)^4, x)` or
    `integrate(cos(linear)^4, x)`; it only hardens the verifier needed before
    that promotion
- validation:
  - focal residual helper unit, public CLI residual contract,
    `calculus_integrate_contract`, `make engine-fast`, and
    `make engine-scorecard` passed with `failed = 0`
- next candidate:
  - retry the conservative integration rule for `sin(linear)^4` and
    `cos(linear)^4`, using this residual verifier as the promotion guard

## 2026-05-14 - Retained calculus: trig fourth-power integration

- area:
  - calculus / integrate / trig power reduction / antiderivative verification
- status:
  - `retained`
- retained result:
  - `integrate(sin(x)^4, x)` now returns
    `1/32*sin(4*x) + 3/8*x - 1/4*sin(2*x)`
  - `integrate(cos(x)^4, x)` now returns
    `1/32*sin(4*x) + 1/4*sin(2*x) + 3/8*x`
  - affine variants such as `integrate(sin(2*x+1)^4, x)` and
    `integrate(cos(2*x+1)^4, x)` are covered by the same rule
  - nested verification residuals such as
    `diff(integrate(sin(x)^4, x), x) - sin(x)^4` collapse to `0` without
    required conditions or `depth_overflow`
- implementation note:
  - promoted the previous observe-only discovery after the bounded residual
    verifier was available
  - the primitive is constructed by fourth-power trig reduction for affine
    arguments and held through public presentation so `sin(4*u)` stays compact
    instead of triggering a costly double-angle expansion
  - the residual verifier also recognizes `diff(integrate(f, x), x) - f`
    directly for this family, avoiding a slow rendered-primitive round trip
- validation:
  - symbolic integration unit, public integration contract, public CLI probes,
    `make engine-fast`, and `make engine-scorecard` passed with `failed = 0`
- next candidate:
  - consider the same conservative promotion pattern for sixth even trig powers
    only if the primitive can be kept compact and antiderivative verification
    stays bounded

## 2026-05-14 - Retained calculus: trig sixth-power integration

- area:
  - calculus / integrate / trig power reduction / antiderivative verification
- status:
  - `retained`
- retained result:
  - `integrate(sin(x)^6, x)` now returns
    `3/64*sin(4*x) + 5/16*x - 15/64*sin(2*x) - 1/192*sin(6*x)`
  - `integrate(cos(x)^6, x)` now returns
    `1/192*sin(6*x) + 3/64*sin(4*x) + 15/64*sin(2*x) + 5/16*x`
  - affine variants such as `integrate(sin(2*x+1)^6, x)` and
    `integrate(cos(2*x+1)^6, x)` are covered by the same rule
  - nested and rendered-primitive residuals collapse to `0` without required
    conditions or `depth_overflow`
- implementation note:
  - a pre-promotion probe showed that the explicit sixth-power residual route
    was not cheap enough through generic public simplification
  - promotion therefore included both the direct sixth-power table rule and a
    bounded residual verifier for
    `5*x/16 +/- 15*sin(2*u)/(64*a) + 3*sin(4*u)/(64*a) +/- sin(6*u)/(192*a)`
  - the primitive is held through public presentation so the multiple-angle
    terms remain compact instead of triggering broad trig expansion
- validation:
  - symbolic integration unit, residual helper unit, public integration
    contract, public CLI probes, `make engine-fast`, and
    `make engine-scorecard` passed with `failed = 0`
- next candidate:
  - before promoting higher even powers, consider extracting a shared bounded
    even-trig-power residual helper so eighth-power support does not duplicate
    the fourth/sixth matcher shape

## 2026-05-14 - Retained robustness: shared even trig residual collector

- area:
  - robustness / calculus residual verification / cohesion
- status:
  - `retained`
- retained result:
  - fourth- and sixth-power trig antiderivative residuals now share one bounded
    collector for the linear term and `sin(k*u)` harmonic terms
  - public behavior is unchanged: existing fourth/sixth integration results and
    residuals still verify without required conditions or `depth_overflow`
- implementation note:
  - replaced separate fourth- and sixth-power term collectors with a single
    harmonic-driven helper
  - kept the family-specific coefficient tables at the call sites so route
    priority and mathematical intent remain explicit
  - this is a behavior-preserving refactor; no new calculus family was promoted
- validation:
  - focal fourth/sixth residual helper tests, full public integration contract,
    `make engine-fast`, and `make engine-scorecard` passed with `failed = 0`
- next candidate:
  - probe eighth even trig powers against the shared residual helper before
    adding any public `integrate` rule, and reject if compact presentation or
    derivative verification becomes slow

## 2026-05-14 - Retained calculus: trig eighth-power integration

- area:
  - calculus / integrate / trig power reduction / antiderivative verification
- status:
  - `retained`
- retained result:
  - `integrate(sin(x)^8, x)` now returns
    `sin(8*x)/1024 + 7*sin(4*x)/128 + 35*x/128 - 7*sin(2*x)/32 - sin(6*x)/96`
  - `integrate(cos(x)^8, x)` now returns
    `sin(8*x)/1024 + sin(6*x)/96 + 7*sin(4*x)/128 + 7*sin(2*x)/32 + 35*x/128`
  - affine variants such as `integrate(sin(2*x+1)^8, x)` and
    `integrate(cos(2*x+1)^8, x)` are covered by the same rule
  - nested and rendered-primitive residuals collapse to `0` without required
    conditions or `depth_overflow`
- implementation note:
  - the pre-promotion probe showed `integrate(sin(x)^8, x)` was unsupported and
    the explicit residual route was not cheap through generic simplification
  - promotion reused the shared even-trig residual collector by adding only the
    eighth-power coefficient table
  - the primitive is held through public presentation so multiple-angle terms
    remain compact and do not trigger broad trig expansion
- validation:
  - symbolic integration unit, residual helper unit, explicit public residual
    contract, full public integration contract, `make engine-fast`, and
    `make engine-scorecard` passed with `failed = 0`
- next candidate:
  - pause even-power promotion and look for a different high-ROI integration
    family, preferably one that reuses antiderivative verification without
    adding another long fixed table

## 2026-05-14 - Observe-only discovery: sextic trig by-parts residual cost

- area:
  - calculus / integrate / antiderivative verification
- status:
  - `discovery/observe-only`
- resolved by:
  - `2026-05-14 - Retained calculus: sextic polynomial-times-trig by-parts`
  - `2026-05-14 - Retained calculus: sparse affine sextic trig by-parts trace`
  - current public probes for `integrate(x^6*sin(x), x)`,
    `diff(integrate(x^6*sin(x), x), x) - x^6*sin(x)`, and the sparse affine
    `(x^6+1)*sin(2*x+1)` variant return compact primitives or `0` without
    required conditions or `depth_overflow`
- observed result:
  - finite derivative-series construction can produce primitives for
    `integrate(x^6*sin(x), x)` and `integrate(x^6*cos(x), x)`
  - the nested public residual probe
    `diff(integrate(x^6*sin(x), x), x) - x^6*sin(x)` did not return in a
    reasonable smoke window
- decision:
  - do not promote degree-6 trig by-parts yet
  - retain only the degree-6 exponential by-parts path, whose public residual
    collapses to `0` cheaply
- reusable weakness:
  - higher-degree polynomial-times-trig primitives need a bounded residual or
    post-calculus grouping path before promotion; simply generating the
    antiderivative is not enough for a retained calculus feature
- next candidate:
  - investigate compact grouping/residual verification for
    polynomial-times-trig by-parts before raising the public trig degree cap

## 2026-05-14 - Retained calculus: sextic polynomial-times-trig by-parts

- area:
  - calculus / integrate / polynomial-times-trig by-parts / antiderivative
    verification
- status:
  - `retained`
- retained result:
  - `integrate(x^6*sin(x), x)` now returns
    `(6*x^5 + 720*x - 120*x^3)*sin(x) + (30*x^4 + 720 - x^6 - 360*x^2)*cos(x)`
  - `integrate(x^6*cos(x), x)` now returns
    `(6*x^5 + 720*x - 120*x^3)*cos(x) + (x^6 + 360*x^2 - 30*x^4 - 720)*sin(x)`
  - nested public residuals such as
    `diff(integrate(x^6*sin(x), x), x) - x^6*sin(x)` collapse to `0`
    without required conditions or `depth_overflow`
- implementation note:
  - the previous observe-only failure was caused by the residual classifier
    still using a degree-5 cap while the primitive builder had become
    degree-cap driven
  - promotion raised the trig by-parts cap to 6 and aligned the classifier and
    additive regrouping path with the same constant
  - the residual matcher uses the existing term-shape comparison after the
    integrand has already been bounded to the polynomial-times-trig family
- validation:
  - symbolic integration unit, calculus residual helper unit, public residual
    CLI probes, public integration contract, `make engine-fast`, and
    `make engine-scorecard` passed with `failed = 0`
- next candidate:
  - before considering degree 7/8 polynomial-times-trig by-parts, probe sparse
    affine degree-6 variants and presentation grouping; reject any promotion
    whose public residual does not stay bounded

## 2026-05-14 - Retained calculus: sparse affine sextic trig by-parts trace

- area:
  - calculus / integrate / didactic presentation / distribution guard
- status:
  - `retained`
- retained result:
  - `integrate((x^6+1)*sin(2*x+1), x)` and
    `integrate((x^6+1)*cos(2*x+1), x)` now keep the compact source product as
    the direct integration target instead of first emitting a distributive
    expansion step
  - both primitives verify through the bounded public residual route, and
    nested residuals collapse to `0` without required conditions or
    `depth_overflow`
- implementation note:
  - the first promotion probe exposed a structural trace-quality weakness: the
    antiderivative was correct and verified, but `DistributeRule` fired before
    the symbolic integration step for sparse degree-6 cofactors with a constant
    term
  - the retained fix adds a narrow distribution guard only when the current
    product is exactly the target of an ancestor `integrate(...)` and the
    bounded polynomial-times-trig classifier already accepts it
  - this preserves general distribution behavior outside integrate-prep and
    avoids adding another integration table
- validation:
  - focal sparse affine sextic contract, existing sparse quartic trace contract,
    full public integration contract, `make engine-fast`,
    `make engine-scorecard`, and `make engine-scorecard-pressure` passed with
    `failed = 0`
- next candidate:
  - probe sparse affine degree-6 hyperbolic or exponential by-parts products
    for the same trace-quality weakness before increasing any degree caps

## 2026-05-14 - Retained calculus: sparse affine sextic exp by-parts trace

- area:
  - calculus / integrate / didactic presentation / distribution guard
- status:
  - `retained`
- retained result:
  - `integrate((x^6+1)*exp(2*x+1), x)` now keeps the compact source product as
    the direct integration target instead of first expanding to a sum
  - the primitive verifies through the bounded public residual route, and the
    nested residual
    `diff(integrate((x^6+1)*exp(2*x+1), x), x) - (x^6+1)*exp(2*x+1)`
    collapses to `0` without required conditions or `depth_overflow`
- implementation note:
  - pre-promotion probes showed the result and residual were already correct,
    but the public trace had a distributive expansion followed by cleanup
  - the retained fix generalizes the existing integrate-target distribution
    guard to also consult the bounded polynomial-times-exp classifier
  - sparse affine degree-6 `sinh/cosh` probes still remain unsupported and were
    not promoted in this iteration
- validation:
  - focal sparse affine sextic exp contract, existing sparse quartic exp
    contract, sparse affine sextic trig contract, full public integration
    contract, `make engine-fast`, `make engine-scorecard`, and
    `make engine-scorecard-pressure` passed with `failed = 0`
- next candidate:
  - investigate whether polynomial-times-hyperbolic degree-6 support should
    reuse the same finite derivative-series pattern, but promote only if
    public residual verification stays bounded

## 2026-05-15 - Retained calculus: sextic polynomial-times-hyperbolic by-parts

- area:
  - calculus / integrate / polynomial-times-hyperbolic by-parts /
    antiderivative verification
- status:
  - `retained`
- retained result:
  - `integrate(x^6*sinh(x), x)` now returns
    `(x^6 + 30*x^4 + 360*x^2 + 720)*cosh(x) - (6*x^5 + 120*x^3 + 720*x)*sinh(x)`
  - `integrate(x^6*cosh(x), x)` now returns the companion primitive with
    `sinh`/`cosh` swapped
  - sparse affine variants such as
    `integrate((x^6+1)*sinh(2*x+1), x)` and
    `integrate((x^6+1)*cosh(2*x+1), x)` resolve directly and their nested
    public residuals collapse to `0`
- implementation note:
  - the previous support was capped at degree 5 because the finite by-parts
    series stopped at the fifth derivative
  - the retained fix adds the missing sixth-derivative contribution to the
    even hyperbolic polynomial and aligns the public classifier with the same
    degree-6 cap
  - no real-domain conditions are introduced; `sinh` and `cosh` remain total
    in the supported real branch
- validation:
  - symbolic integration unit, public residual CLI probes, public integration
    contract, `make engine-fast`, and `make engine-scorecard` passed with
    `failed = 0`
- next candidate:
  - probe whether degree-6 polynomial-times-hyperbolic cases should get the
    same embedded wrapper representative as exp/trig, but promote only a
    minimal non-duplicate row if it exposes wrapper or shell-depth value

## 2026-05-15 - Retained calculus: septic polynomial-times-exp by-parts

- area:
  - calculus / integrate / polynomial-times-exp by-parts / antiderivative
    verification
- status:
  - `retained`
- retained result:
  - `integrate(x^7*exp(x), x)` now returns
    `e^x*(x^7 + 42*x^5 + 840*x^3 + 5040*x - 7*x^6 - 210*x^4 - 2520*x^2 - 5040)`
  - the nested public residual
    `diff(integrate(x^7*exp(x), x), x) - x^7*exp(x)` collapses to `0`
  - sparse affine variants such as
    `integrate((x^7+1)*exp(2*x+1), x)` also verify through the bounded public
    residual route
- implementation note:
  - the exp by-parts builder already used a generic finite derivative series;
    the retained change only raises the exp polynomial cap from 6 to 7 and
    extends the public contracts
  - no real-domain conditions are introduced because `exp` remains total in the
    supported real branch
  - trig and hyperbolic degree-7 support was not promoted in this iteration;
    trig needs a separate residual/presentation probe and hyperbolic still
    requires extending the manual odd-series term
- validation:
  - symbolic integration unit, focal public exp contract, exp public residual
    route contract, full public integration contract, `make engine-fast`, and
    `make engine-scorecard` passed with `failed = 0`
- next candidate:
  - probe degree-7 polynomial-times-trig by-parts separately; promote only if
    the public residual remains bounded and the trace stays teachable without
    expansion cleanup

## 2026-05-15 - Retained calculus: septic polynomial-times-trig by-parts

- area:
  - calculus / integrate / polynomial-times-trig by-parts / antiderivative
    verification
- status:
  - `retained`
- retained result:
  - `integrate(x^7*sin(x), x)` now returns
    `(42*x^5 + 5040*x - x^7 - 840*x^3)*cos(x) + (7*x^6 + 2520*x^2 - 210*x^4 - 5040)*sin(x)`
  - `integrate(x^7*cos(x), x)` now returns the companion primitive with
    `sin`/`cos` swapped
  - sparse affine variants such as
    `integrate((x^7+1)*sin(2*x+1), x)` and
    `integrate((x^7+1)*cos(2*x+1), x)` resolve directly and their nested
    public residuals collapse to `0`
- implementation note:
  - the trig by-parts builder already used a generic finite derivative series;
    the retained change only raises the trig polynomial cap from 6 to 7 and
    extends the public contracts
  - no real-domain conditions are introduced because `sin` and `cos` are total
    in the supported real branch
  - embedded corpus padding was skipped because the scorecard marks embedded
    coverage as balanced/saturated; the public calculus contract is the
    retained representative
- validation:
  - symbolic integration unit, focal public trig contract, trig public residual
    route contract, full public integration contract, `make engine-fast`, and
    `make engine-scorecard` passed with `failed = 0`
- next candidate:
  - probe degree-7 polynomial-times-hyperbolic by-parts separately; promote
    only if the manual hyperbolic series can be extended cleanly and public
    residuals remain bounded

## 2026-05-15 - Retained calculus: septic polynomial-times-hyperbolic by-parts

- area:
  - calculus / integrate / polynomial-times-hyperbolic by-parts /
    antiderivative verification
- status:
  - `retained`
- retained result:
  - `integrate(x^7*sinh(x), x)` now returns
    `(x^7 + 42*x^5 + 840*x^3 + 5040*x)*cosh(x) - (7*x^6 + 210*x^4 + 2520*x^2 + 5040)*sinh(x)`
  - `integrate(x^7*cosh(x), x)` now returns the companion primitive with
    `sinh`/`cosh` swapped
  - sparse affine variants such as
    `integrate((x^7+1)*sinh(2*x+1), x)` and
    `integrate((x^7+1)*cosh(2*x+1), x)` resolve directly and their nested
    public residuals collapse to `0`
- implementation note:
  - the previous hyperbolic support was capped at degree 6 because the manual
    finite by-parts series stopped before the seventh-derivative odd term
  - the retained fix adds that seventh derivative over `slope^8` and aligns the
    public classifier with the same degree-7 cap
  - no real-domain conditions are introduced; `sinh` and `cosh` remain total in
    the supported real branch
  - embedded corpus padding was skipped because the scorecard marks embedded
    coverage as balanced/saturated; the public calculus contract is the
    retained representative
- validation:
  - symbolic integration unit, focal public hyperbolic contract, hyperbolic
    public residual route contract, full public integration contract, direct
    residual probes for simple and sparse affine degree-7 cases,
    `make engine-fast`, and `make engine-scorecard` passed with `failed = 0`
- next candidate:
  - probe nontrivial sparse degree-7 hyperbolic cofactors beyond `x^7+1`; promote
    only if they expose a new representative shape rather than another
    near-duplicate

## 2026-05-15 - Retained calculus: log by-parts didactic trace and compact proportional affine prep

- area:
  - calculus / integrate / logarithmic integration by parts / didactic trace
    quality
- status:
  - `retained`
- retained result:
  - `integrate(x*ln(x), x)` still returns the same verified primitive and
    required condition `x > 0`, but now exposes a concrete
    `Usar integración por partes` substep
  - `integrate((2*x+1)*ln(2*x+1), x)` now keeps the proportional affine-log
    product as the direct integration target in `steps on` mode, instead of
    first emitting `Expandir la expresión`
  - both residuals
    `diff(integrate(x*ln(x), x), x) - x*ln(x)` and
    `diff(integrate((2*x+1)*ln(2*x+1), x), x) - (2*x+1)*ln(2*x+1)` collapse
    to `0` while preserving the same positive-domain requirements
- implementation note:
  - the mathematical antiderivatives already existed; the retained change
    publishes narrow log-by-parts classifiers for didactic/presentation callers
  - the transform preserve gate is intentionally restricted to proportional
    affine-log products so non-proportional cases such as
    `(x+1)*ln(2*x+1)` keep the previous, better-expanded presentation
  - this is a calculus trace improvement, not a new integration algorithm
- validation:
  - focal log-by-parts trace contract, existing monomial-log and affine-log
    contracts, full public integration contract (`237 passed; 0 failed;
    1 ignored`), direct residual probes, `make engine-fast`,
    `make engine-scorecard`, and `make engine-scorecard-pressure` passed with
    `failed = 0`
- derive bridge:
  - no derive row promoted; this is command-specific calculus step quality, not
    a reusable algebraic target-form transition
- next candidate:
  - audit other correct one-step integration-by-parts families, especially
    inverse-trig by-parts, for missing non-decorative substeps before expanding
    more integration surface

## 2026-05-15 - Retained calculus: inverse-trig by-parts didactic trace

- area:
  - calculus / integrate / inverse-trig integration by parts / didactic trace
    quality
- status:
  - `retained`
- retained result:
  - `integrate(arcsin(x), x)` still returns
    `sqrt(1 - x^2) + x*arcsin(x)` with required condition `1 - x^2 > 0`,
    but now exposes a concrete `Usar integración por partes` substep
  - `integrate(arccos(x), x)` still returns
    `x*arccos(x) - sqrt(1 - x^2)` with required condition
    `1 - x^2 > 0`, and now exposes the same didactic substep
  - `integrate(arctan(x), x)` and reciprocal-affine variants such as
    `integrate(arctan(1/(2*x+1)), x)` now also expose the by-parts substep
    while preserving their existing domain requirements
  - residual probes for `arccos(x)` and `arctan(1/(2*x+1))` collapsed to `0`
    with the same required conditions
- implementation note:
  - the mathematical antiderivatives already existed; the retained change only
    publishes a narrow `arctan(linear)` classifier and teaches the didactic
    classifier to recognize existing bounded inverse-trig and arctan
    by-parts targets
  - no antiderivative selection, domain policy, or simplification rule changed
  - embedded corpus padding was skipped because this is command-specific
    calculus trace quality, not a new algebraic equivalence family
- validation:
  - focal inverse-trig by-parts trace contract, full public integration
    contract (`238 passed; 0 failed; 1 ignored`), direct residual probes,
    `make engine-fast`, and `make engine-scorecard` passed with `failed = 0`
- derive bridge:
  - no derive row promoted; this is calculus command trace quality rather than
    a reusable derive target-form transition
- next candidate:
  - inspect correct but opaque substitution-style integration families for the
    same issue: promote only a representative family when the substep exposes a
    real method such as substitution, not just a decorated table lookup

## 2026-05-15 - Retained calculus: polynomial substitution didactic trace

- area:
  - calculus / integrate / polynomial-derivative substitution / didactic trace
    quality
- status:
  - `retained`
- retained result:
  - `integrate(2*x*exp(x^2), x)` still returns `exp(x^2)` with no required
    conditions, but now exposes a concrete `Usar sustitución` substep
  - representative trig and hyperbolic substitution cases such as
    `integrate(2*x*cos(x^2), x)`, `integrate(2*x*sin(x^2), x)`, and
    `integrate(2*x*sinh(x^2), x)` now expose the same method-level substep
  - residual probes for the cosine and hyperbolic representatives collapsed to
    `0` with empty required conditions
  - unsupported cases such as `integrate(sin(x^2), x)` remain unsupported and
    emit no fake substitution trace
- implementation note:
  - the mathematical antiderivatives already existed; the retained change
    publishes a narrow classifier that delegates to the existing
    `f'(x)*kernel(f(x))` substitution recognizer
  - the didactic layer uses that classifier only to add a substep to an already
    successful `Symbolic Integration` step
  - no antiderivative selection, domain policy, or general integration search
    changed
- validation:
  - focal substitution trace contract, full public integration contract
    (`239 passed; 0 failed; 1 ignored`), direct residual probes,
    unsupported-case smoke, `make engine-fast`, and `make engine-scorecard`
    passed with `failed = 0`
- derive bridge:
  - no derive row promoted; this is calculus command trace quality rather than
    a reusable algebraic target-form transition
- next candidate:
  - inspect quotient substitution families with visible pre-normalization, such
    as logarithmic derivative ratios, to decide whether compact-preserve or a
    substitution substep gives better retained didactic value without changing
    domains

## 2026-05-15 - Retained calculus: hyperbolic quotient substitution didactic trace

- area:
  - calculus / integrate / hyperbolic quotient substitution / didactic trace
    quality
- status:
  - `retained`
- retained result:
  - `integrate(2*x*cosh(x^2)/sinh(x^2), x)` still returns
    `ln(|sinh(x^2)|)` with required condition `sinh(x^2) ≠ 0`, but now
    exposes a concrete `Usar sustitución` substep
  - `integrate(2*x/tanh(x^2), x)` now exposes the same substitution substep
    while preserving the same nonzero `sinh(x^2)` requirement
  - `integrate(2*x/cosh(x^2)^2, x)` still returns `tanh(x^2)` with no required
    conditions and now exposes the substitution substep
  - unsupported missing-cofactor cases such as
    `integrate(cosh(x^2)/sinh(x^2), x)` remain unsupported and emit no fake
    substitution trace
- implementation note:
  - the mathematical antiderivatives already existed; the retained change
    publishes a narrow classifier over the existing hyperbolic quotient
    substitution helpers
  - the didactic layer uses that classifier only to add a substep to an already
    successful `Symbolic Integration` step
  - no antiderivative selection, domain policy, pre-normalization, or general
    integration search changed
- validation:
  - focal hyperbolic quotient substitution trace contract, full public
    integration contract (`240 passed; 0 failed; 1 ignored`), direct residual
    probes, unsupported-case smoke, `make engine-fast`, and
    `make engine-scorecard` passed with `failed = 0`
- derive bridge:
  - no derive row promoted; this is calculus command trace quality rather than
    a reusable algebraic target-form transition
- next candidate:
  - consider trig logarithmic-derivative quotient substitution (`sin/cos`,
    `cos/sin`) for the same narrow didactic trace, preserving nonzero
    conditions and avoiding fake traces on missing-cofactor cases

## 2026-05-15 - Retained calculus: trig quotient substitution didactic trace

- area:
  - calculus / integrate / trig quotient substitution / didactic trace quality
- status:
  - `retained`
- retained result:
  - `integrate(2*x*tan(x^2), x)` still returns `-ln(|cos(x^2)|)` with
    required condition `cos(x^2) ≠ 0`, but now exposes a concrete
    `Usar sustitución` substep on the public integration step
  - `integrate(3*x^2*cot(x^3), x)` still returns `ln(|sin(x^3)|)` with
    required condition `sin(x^3) ≠ 0`, and now exposes the same substitution
    substep
  - `integrate(2*x/cos(x^2)^2, x)` still returns `tan(x^2)` with required
    condition `cos(x^2) ≠ 0`, and now exposes the substitution substep
  - unsupported missing-cofactor cases such as `integrate(tan(x^2), x)` remain
    unsupported and emit no fake integration step
- implementation note:
  - the mathematical antiderivatives already existed; the retained change
    publishes a narrow classifier over existing trig quotient substitution
    helpers
  - the didactic layer uses that classifier only to add a substep to an already
    successful `Symbolic Integration` step
  - no antiderivative selection, domain policy, pre-normalization, or general
    integration search changed
- observe-only note:
  - `integrate(2*x*sin(x^2)/cos(x^2)^2, x)` also resolves correctly to
    `sec(x^2)`, but currently reaches the UI through a different public step
    shape rather than the `Calcular la integral` step used by this substep
    classifier
  - it was not promoted in this iteration; treat it as a future didactic-route
    audit candidate rather than widening this patch
- validation:
  - focal trig quotient substitution trace contract, full public integration
    contract (`241 passed; 0 failed; 1 ignored`), direct residual probes,
    unsupported-case smoke, `make engine-fast`, and `make engine-scorecard`
    passed with `failed = 0`
- derive bridge:
  - no derive row promoted; this is calculus command trace quality rather than
    a reusable algebraic target-form transition
- next candidate:
  - inspect successful integration families that bypass `Symbolic Integration`
    in the public trace, starting with sec/csc derivative quotients, and decide
    whether they need a shared didactic hook or should remain separate

## 2026-05-15 - Retained calculus: direct sec/csc derivative quotient trace

- area:
  - calculus / integrate / direct public integration steps / didactic trace
    quality
- status:
  - `retained`
- retained result:
  - `integrate(2*x*sin(x^2)/cos(x^2)^2, x)` still returns `sec(x^2)` with
    required condition `cos(x^2) ≠ 0`, but now exposes a concrete
    `Usar sustitución` substep on its direct public integration step
  - `integrate(3*x^2*cos(x^3)/sin(x^3)^2, x)` still returns `-csc(x^3)` with
    required condition `sin(x^3) ≠ 0`, and now exposes the same substitution
    substep
  - unsupported missing-cofactor cases such as
    `integrate(sin(x^2)/cos(x^2)^2, x)` remain unsupported and emit no fake
    substep
- implementation note:
  - the mathematical antiderivatives and narrow trig quotient classifier
    already existed
  - the retained change removes the didactic dependency on the internal
    `Symbolic Integration` rule name for substitution substeps; the gate is now
    the actual `before` expression being `integrate(...)` plus the existing
    family classifier accepting the integrand
  - no antiderivative selection, domain policy, pre-normalization, or general
    integration search changed
- validation:
  - focal direct sec/csc derivative quotient trace contract, full public
    integration contract (`242 passed; 0 failed; 1 ignored`), direct residual
    probes, unsupported-case smoke, `make engine-fast`, and
    `make engine-scorecard` passed with `failed = 0`
- derive bridge:
  - no derive row promoted; this is calculus command trace quality rather than
    a reusable algebraic target-form transition
- next candidate:
  - audit other direct one-step integration families with correct results but
    no method-level substep; promote only when a narrow existing classifier can
    explain the method without changing math or domains

## 2026-05-15 - Retained calculus: direct trig-log substitution trace

- area:
  - calculus / integrate / direct trig-log substitution / didactic trace
    quality
- status:
  - `retained`
- retained result:
  - `integrate(tan(2*x+1), x)` still returns `-1/2*ln(|cos(2*x+1)|)` with
    required condition `cos(2*x+1) ≠ 0`, and now exposes `Usar sustitución`
    on the integration step
  - `integrate(cot(2*x+1), x)` still returns `1/2*ln(|sin(2*x+1)|)` with
    required condition `sin(2*x+1) ≠ 0`, and now exposes the same substitution
    substep
  - `integrate(sec(2*x+1), x)` and `integrate(csc(2*x+1), x)` expose
    substitution on the actual integration step after reciprocal preparation,
    while prep and cleanup steps do not receive fake substitution substeps
  - unsupported non-linear missing-cofactor cases such as
    `integrate(tan(x^2), x)` remain unsupported and emit no fake substitution
    trace
- implementation note:
  - publishes a narrow classifier over existing `trig_log_antiderivative` and
    `reciprocal_trig_log_antiderivative`
  - the didactic substitution gate now also rejects steps whose `after` is
    still `integrate(...)`, so preparation rewrites do not get method substeps
  - no antiderivative selection, domain policy, pre-normalization, or general
    integration search changed
- validation:
  - focal direct trig-log substitution trace contract, full public integration
    contract (`243 passed; 0 failed; 1 ignored`), direct residual/smoke probes,
    `make engine-fast`, and `make engine-scorecard` passed with `failed = 0`
  - the first focal version was slow because it redundantly verified
    antiderivatives already covered elsewhere; the retained contract was
    reduced to trace assertions and finished in 0.57s
- derive bridge:
  - no derive row promoted; this is calculus command trace quality rather than
    an algebraic target-form transition
- next candidate:
  - audit other direct one-step integration families with correct results but
    no method-level substep; prefer a narrow existing classifier and avoid
    adding duplicate residual-heavy tests

## 2026-05-15 - Retained calculus: polynomial-base substitution trace

- area:
  - calculus / integrate / polynomial-base substitution / didactic trace
    quality
- status:
  - `retained`
- retained result:
  - `integrate((2*x+1)/(x^2+x-1), x)` still returns
    `ln(|x^2 + x - 1|)` with required condition `x^2 + x - 1 ≠ 0`, and now
    exposes `Usar sustitución`
  - `integrate((2*x+1)/(x^2+x-1)^3, x)` still returns
    `-1/(2*(x^2+x-1)^2)` with the same nonzero base condition, and now
    exposes the substitution substep
  - `integrate(x/sqrt(x^2+1), x)`, `integrate(2*x/sqrt(x^2-1), x)`, and
    `integrate(2*x*(x^2-1)^(3/2), x)` expose the same substep while preserving
    their current empty, positive, or nonnegative radicand conditions
  - table-shaped cases without a polynomial cofactor, such as
    `integrate(1/(x^2+1), x)` and `integrate(1/sqrt(x^2+1), x)`, do not emit a
    fake substitution substep
- implementation note:
  - publishes a narrow didactic classifier over existing log-derivative,
    denominator-power, square-root derivative, and polynomial-power
    antiderivative helpers
  - the retained change does not alter antiderivative selection, domain policy,
    pre-normalization, or general integration search
- narrowed scope:
  - early smoke showed that product-form inverse-hyperbolic table cases such
    as `integrate(1/sqrt(x^2+1), x)` would be overclassified if product
    inverse-trig/hyperbolic helpers were included
  - those product-form helpers were removed from this classifier; nested
    `arcsin/asinh` substitutions like `integrate(2*x/sqrt(4-x^4), x)` remain a
    separate future candidate rather than widening this patch
- validation:
  - focal polynomial-base substitution trace contract, full public integration
    contract (`244 passed; 0 failed; 1 ignored`), `make engine-fast`, and
    `make engine-scorecard` passed with `failed = 0`
- derive bridge:
  - no derive row promoted; this is calculus command trace quality rather than
    an algebraic target-form transition
- next candidate:
  - add a separate narrow classifier for nested inverse-trig/hyperbolic
    substitutions only if it can distinguish genuine cofactor substitutions
    from table primitives without domain or trace regressions

## 2026-05-15 - Retained calculus: nested inverse-polynomial substitution trace

- area:
  - calculus / integrate / nested inverse trig-hyperbolic substitution /
    didactic trace quality
- status:
  - `retained`
- retained result:
  - `integrate(2*x/sqrt(4-x^4), x)` still returns `arcsin(x^2/2)` with
    required condition `4 - x^4 > 0`, and now exposes `Usar sustitución`
  - `integrate(2*x/sqrt(1+x^4), x)` still returns `asinh(x^2)` with no
    required conditions, and now exposes the same substep
  - `integrate(2*x/(1+x^4), x)` still returns `arctan(x^2)`, and now exposes
    the substitution substep
  - `integrate(2*x/(4-x^4), x)` keeps `1/2*atanh(x^2/2)` with required
    condition `4 - x^4 > 0`, and keeps the substitution substep through the
    new non-linear detector rather than the broader polynomial-base detector
- false-positive reduction:
  - table or linear primitives such as `integrate(1/(x^2+1), x)`,
    `integrate(1/(1-x^2), x)`, `integrate(1/(4-x^2), x)`,
    `integrate(1/sqrt(x^2+1), x)`, `integrate(1/sqrt(1-x^2), x)`,
    `integrate(1/sqrt(4-(x+1)^2), x)`,
    `integrate(1/sqrt(4+(x+1)^2), x)`, and
    `integrate(2/(1+(2*x+1)^2), x)` do not emit a fake substitution substep
- implementation note:
  - introduces a separate didactic classifier that requires a non-linear
    polynomial substitution argument/radicand before accepting inverse
    trig-hyperbolic families
  - removes inverse trig-hyperbolic matching from the generic polynomial-base
    substitution classifier so direct table primitives stay clean
  - no antiderivative selection, result presentation, domain policy,
    pre-normalization, or general integration search changed
- validation:
  - focal nested inverse-polynomial substitution trace contract, previous
    polynomial-base substitution trace contract, full public integration
    contract (`245 passed; 0 failed; 1 ignored`), `make engine-fast`, and
    `make engine-scorecard` passed with `failed = 0`
- derive bridge:
  - no derive row promoted; this is calculus command trace quality rather than
    an algebraic target-form transition
- next candidate:
  - audit remaining successful one-step integration families for either
    missing method substeps or overly broad method substeps, starting with
    arctan/atanh linear scaled variants and preserving the same no-fake-table
    standard

## 2026-05-17 - Discovery observe-only: 7/2 post-integration sqrt presentation residual

- area:
  - calculus / post-calculus presentation / antiderivative verification
- status:
  - `discovery/observe-only`
- observed while testing:
  - the local presentation candidate for
    `integrate((2*x+1)/(x^2+x+1)^(7/2), x)` rewrote the correct primitive from
    `-2/(5*(x^2+x+1)^(5/2))` to the more educational
    `-2/(5*sqrt(x^2+x+1)*(x^2+x+1)^2)`
  - direct plain CLI probing could simplify the corresponding derivative
    residual to `0`
  - the public JSON contract path did not collapse
    `diff(integrate((2*x+1)/(x^2+x+1)^(7/2), x), x) -
    (2*x+1)/(x^2+x+1)^(7/2)` to `0`; it left a large semienteric rational
    residual
- retained learning:
  - extending post-calculus presentation beyond `3/2` reciprocal powers needs
    residual support for two-term semienteric polynomial quotients under the
    public wire/JSON route, not only a display rewrite
  - the candidate should remain unpromoted until antiderivative verification is
    stable in the promoted contract path
- resolved by:
  - resolved by a bounded residual-normalization extension for exact
    polynomial-power denominators plus a retained post-integration presentation
    contract for the `7/2 -> 5/2` reciprocal-power primitive
  - validation kept `make engine-fast`, `make engine-scorecard`, and
    `make engine-scorecard-pressure` at `failed = 0`

## 2026-05-16 - Rejected broad condition propagation: reciprocal trig canonicalization

- area:
  - calculus / domain safety / trig reciprocal canonicalization
- status:
  - `rejected`
- local win:
  - adding required nonzero conditions directly to `sec(u) -> 1/cos(u)`,
    `csc(u) -> 1/sin(u)`, and `cot(u) -> cos(u)/sin(u)` made
    `integrate(2*x*sec(x^2)^2, x)` and
    `integrate(2*x*csc(x^2)^2, x)` publish the expected source-domain
    conditions after IntegratePrep rewrote the input
- global loss:
  - `make engine-fast` failed in `calculus_diff_contract`
  - the broad rewrite surfaced redundant or reordered conditions in existing
    diff contracts, including `diff(cot(sqrt(2*x)), x)`,
    `diff(cot((2-3*x)/2), x)`, and `diff(tan(x)^2/2, x)`
- retained learning:
  - source-domain preservation for integration should not be solved by making
    all reciprocal-trig canonicalization rewrites emit conditions globally
  - the retained route should stay inside integration condition extraction and
    recognize the supported `du/cos(u)^2`, `du/sin(u)^2`, `du*sec(u)^2`, and
    `du*csc(u)^2` families directly
- follow-up:
  - only revisit global reciprocal-trig condition propagation if diff policy is
    deliberately updated to accept those extra conditions and ordering is
    normalized first

## 2026-05-16 - Discovery observe-only: affine exp-trig antiderivative residual is slow

- area:
  - calculus / integrate / exp-trig by-parts / residual verification
- status:
  - `discovery/observe-only`
- observed while testing:
  - a generated candidate for
    `integrate(exp(2*x+1)*sin(2*x+1), x)` produced the expected compact
    antiderivative `1/4*e^(2*x+1)*(sin(2*x+1)-cos(2*x+1))`
  - the public residual
    `diff(1/4*exp(2*x+1)*(sin(2*x+1)-cos(2*x+1)), x) - exp(2*x+1)*sin(2*x+1)`
    did eventually simplify to `0`, but took about `45s` in the CLI smoke
- retained learning:
  - unit-argument `exp(x)*sin(x)` and `exp(x)*cos(x)` are cheap and stable, but
    affine exp-trig antiderivatives should not be promoted until the
    derivative residual path avoids expanding or repeatedly normalizing the
    shared affine argument
  - this is a reusable simplification/runtime weakness, not a malformed case
    or display-only issue
- resolved by:
  - current public integration contracts verify same-affine exp-trig
    antiderivatives through the bounded public residual route, including
    `integrate(exp(2*x+1)*sin(2*x+1), x)`
  - the original slow residual now collapses to `0` quickly without warnings or
    required conditions

## 2026-05-17 - Refined condition dominance: positive quadratic interval vs exterior

- area:
  - calculus / post-calculus condition presentation / domain normalization
- status:
  - `refined-retained`
- local win:
  - broad dominance of affine `NonZero` guards by a positive quadratic
    condition removed redundant public `Requires` for exterior affine-quotient
    domains such as `diff(arctan(sqrt((2*x+1)/(x+3))), x)`, reducing
    `["x ≠ -4/3", "x < -3 or x > -1/2"]` to
    `["x < -3 or x > -1/2"]`
- global loss:
  - the first version also removed `x ≠ -2` from the bounded interval
    integration residual guard `["-3 < x < 1", "x ≠ -2"]`
  - that was unsound because `-2` lies inside the allowed interval, so the
    nonzero guard is not redundant
- retained learning:
  - a positive quadratic dominates an affine nonzero root between its two roots
    only when the positive region is exterior, i.e. the quadratic leading
    coefficient is positive
  - for factored affine products, the analogous condition is a positive product
    of slopes; opposite-slope products describe a bounded interval and must not
    erase interior nonzero guards
- follow-up:
  - future condition-presentation cleanup should add both exterior and bounded
    negative tests whenever it reasons from roots instead of direct factors

## 2026-05-17 - Rejected broad quotient-gap condition presentation

- area:
  - calculus / post-calculus condition presentation / atanh sqrt quotient
- status:
  - `partial-reject-retained-presentation`
- local win:
  - for `diff(atanh(sqrt(x/(x+1))), x)`, expanding the gap condition
    `1 - x/(x+1) > 0` to `x+1 > 0` plus refining
    `x*(x+1) > 0` with that positive factor reduced the public conditions to
    the ideal `["x > 0"]`
- global loss:
  - the same global expansion changed branch-sensitive contracts for
    `diff(acosh(1-2*x), x)` and
    `diff(ln(sqrt((2*x+1)^2-4)-(2*x+1)), x)`, surfacing the opposite branch
    condition such as `x > 1` or `x > 1/2` where the promoted contract expects
    only the selected branch
  - this failed `make engine-fast` in `calculus_diff_contract`
- retained learning:
  - `1 - N/D > 0` is not just display cleanup in branch-sensitive calculus
    traces; expanding it globally can erase or expose branch-selection policy
    that other families intentionally keep directional
  - quotient-gap domain cleanup should be tied to a family-aware calculus
    condition policy or a more explicit branch regime, not a broad
    condition-normalization rule
- retained part:
  - the derivative presentation for affine positive-gap
    `atanh(sqrt(N/D))` was kept, so supported cases now render compact
    root-denominator forms while preserving existing conservative conditions
- follow-up:
  - revisit minimal `Requires` for `atanh(sqrt(N/D))` only after adding a
    branch-aware condition combiner that has negative tests for `acosh` and
    `ln(sqrt(gap) ± affine)` branch contracts

## 2026-05-17 - Observe-only scaled arctan-sqrt unit-shift-square primitive

- area:
  - calculus / integration / antiderivative residual verification
- status:
  - `superseded`
- observed while testing:
  - the minimal primitive for
    `integrate(1/(sqrt(x)*(x+1)^2), x)` is stable as
    `arctan(sqrt(x)) + sqrt(x)/(x+1)`
  - the scaled candidate
    `integrate(1/(2*sqrt(x)*(x+1)^2), x)` produced a mathematically analogous
    primitive, but the public residual
    `diff(1/2*arctan(sqrt(x)) + (sqrt(x)*1/2)/(x+1), x) - 1/(2*sqrt(x)*(x+1)^2)`
    entered depth-overflow/timeout-shaped simplification during smoke
- retained learning:
  - the unscaled family representative is promotable
  - a later retained cycle promoted the constant-scaled `integrate(...)` path
    with a bounded `diff(integrate(...), x) - integrand` residual verifier
  - this is a reusable residual-verification weakness, not a malformed input
    or duplicate corpus row
- superseded by:
  - a follow-up post-calculus presentation/residual robustness cycle taught the
    arctan-sqrt primitive matcher to recognize the precombined quotient form of
    the manually rendered scaled primitive, so
    `diff(1/2*arctan(sqrt(x)) + (sqrt(x)*1/2)/(x+1), x) - 1/(2*sqrt(x)*(x+1)^2)`
    now collapses to `0` under `x > 0` without entering the depth-overflow
    route

## 2026-05-18 - Discovery observe-only: distinct-slope exp-cos residual is slow

- area:
  - calculus / integration / exp-trig by-parts / residual verification
- status:
  - `discovery/observe-only`
- observed while testing:
  - the distinct-slope sine case
    `integrate(exp(2*x)*sin(3*x), x)` produces the compact primitive
    `1/13*e^(2*x)*(2*sin(3*x)-3*cos(3*x))` and its public residual collapses
    to `0`
  - the analogous cosine primitive
    `1/13*e^(2*x)*(2*cos(3*x)+3*sin(3*x))` triggered repeated
    `depth_overflow` warnings and did not close cheaply during a CLI residual
    probe
- retained learning:
  - distinct-slope `exp(linear)*sin(linear)` can be promoted conservatively,
    but `exp(linear)*cos(linear)` needs a bounded residual simplification route
    before promotion
  - this is a reusable residual-verification weakness, not a malformed input or
    a presentation-only issue
- follow-up:
  - add a direct residual route for `d/dx(e^(a*x+b)*(A*cos(c*x+d)+B*sin(c*x+d)))`
    minus an `exp(linear)*cos(linear)` target before enabling the cosine
    sibling
- resolved by:
  - current public integration contracts now promote
    `integrate(exp(2*x)*cos(3*x), x)` with the compact primitive
    `1/13*e^(2*x)*(2*cos(3*x)+3*sin(3*x))`
  - the nested public residual
    `diff(integrate(exp(2*x)*cos(3*x), x), x) - exp(2*x)*cos(3*x)` and the
    explicit primitive residual both close to `0` through the bounded public
    residual route, with no warnings, `depth_overflow`, or real-domain
    conditions
  - additive constants in the displayed primitive are also accepted by the same
    verification route, so the retained value is the bounded exp-trig
    antiderivative verification path rather than a broad trig expansion rule

## 2026-05-18 - Retained: distinct-slope exp-cos without multiple-angle expansion

- area:
  - calculus / integration / exp-trig by-parts / residual verification
- status:
  - `retained-partial`
- retained:
  - the direct derivative recognizer now handles grouped primitives of the
    form `e^(a*x+b)*(A*sin(c*x+d)+B*cos(c*x+d))`
  - promoted the non-expansion-risk public representative
    `integrate(exp(2*x)*cos((3*x+1)/2), x)` with the compact primitive
    `4/25*e^(2*x)*(3/2*sin((3*x+1)/2)+2*cos((3*x+1)/2))`
  - the public residual collapses to `0` without adding real-domain conditions
- still deferred:
  - `integrate(exp(2*x)*cos(3*x), x)` is deliberately not promoted because the
    public pipeline expands the pure integer multiple-angle argument before the
    residual route can use the compact primitive shape
- retained learning:
  - the useful boundary is not only `sin` vs `cos`; it is whether public
    pre-diff presentation preserves the grouped trig argument long enough for
    the bounded residual route to match
- follow-up:
  - add an anti-expansion or held-presentation route for pure integer
    multiple-angle exp-trig residuals before promoting `cos(3*x)` and similar
    siblings

## 2026-05-18 - Retained coverage: shifted rational exp-sin by-parts guard

- area:
  - calculus / integration / exp-trig by-parts / contract coverage
- status:
  - `retained-coverage`
- retained:
  - promoted `integrate(exp(2*x)*sin((3*x+1)/2), x)` as the sine counterpart
    to the retained shifted rational cosine representative
  - expected primitive:
    `4/25*e^(2*x)*(2*sin((3*x+1)/2)-3/2*cos((3*x+1)/2))`
  - public antiderivative residual collapses to `0` and adds no real-domain
    conditions
- why this representative:
  - it combines distinct exp/trig slopes, a rational shifted trig argument, and
    the sine orientation without triggering pure integer multiple-angle
    expansion
  - it is smaller and safer than promoting the deferred `cos(3*x)` family
- follow-up:
  - still target anti-expansion or held-presentation for pure integer
    multiple-angle exp-trig residuals before widening this family further

## 2026-05-18 - Retained robustness: unsupported integer-multiple exp-cos boundary

- area:
  - calculus / integration / exp-trig by-parts / residual robustness
- status:
  - `retained-robustness`
- retained:
  - added a public integration contract that keeps
    `integrate(exp(2*x)*cos(3*x), x)` as an unsupported, quiet boundary
    until the compact primitive can be verified without expanding the trig
    argument into a deep polynomial route
  - the boundary returns the normalized residual call
    `integrate(cos(3*x)*e^(2*x), x)` with no invented real-domain conditions
    and no blocked hint noise
- why retained:
  - the compact primitive is mathematically known, but public `diff` currently
    expands `sin(3*x)` / `cos(3*x)` before the bounded exp-trig residual route
    can match it
  - forcing this capability now would reintroduce repeated `depth_overflow`
    warnings and a slow residual path
- follow-up:
  - implement a shape-preserving anti-expansion or held-presentation route for
    pure integer multiple-angle exp-trig primitives, then promote the cosine
    sibling with derivative verification

## 2026-05-19 - Retained robustness: trig-root diff with sqrt-variable term

- area:
  - calculus / differentiation / post-calculus presentation / radical
    denominator pressure
- status:
  - `retained-robustness`
- retained:
  - `diff(sqrt(sin(2*x)+cos(x)+sqrt(x)), x)` now stays on the bounded direct
    presentation route instead of entering the generic depth-overflow path
  - public result:
    `(cos(2·x) + 1/4·x^(-1/2) - 1/2·sin(x)) / sqrt(sin(2·x) + cos(x) + sqrt(x))`
  - required conditions preserve both the outer radicand positivity and `x > 0`
    for differentiating `sqrt(x)`
- rejected subcandidate:
  - a prettier common-denominator presentation using explicit
    `2*sqrt(x)*sqrt(radicand)` repeatedly triggered radical rationalization and
    OPQ-style depth-overflow during smoke before promotion
- retained learning:
  - for this family, half-power presentation is currently the safer retained
    public form than forcing a reciprocal `sqrt(x)` denominator
  - future pretty-printing of this result should first add a bounded
    post-calculus radical-denominator presentation route, not rely on global
    rationalization

## 2026-05-19 - Discovery observe-only: tan-root diff elementary-term boundary

- area:
  - calculus / differentiation / post-calculus presentation / mixed
    tan-root elementary terms
- status:
  - `discovery-observe-only`
- observed:
  - `diff(sqrt(tan(x)+sqrt(x)+x), x)` still reaches repeated
    `depth_overflow` warnings and times out during local smoke
  - `diff(sqrt(tan(x)+exp(x)+x), x)` also times out before a retained direct
    presentation route can be promoted
- rejected subcandidates:
  - adding `sqrt(x)` to the existing tan-root direct route via an explicit
    `sqrt(x)` denominator triggered radical rationalization and deeper cleanup
    loops
  - representing the same derivative as `x^(-1/2)` in the numerator still
    allowed the denominator-rationalization path to reopen
  - adding `exp(x)` as a simple derivative term did not intercept the timeout
    early enough to pass smoke
- retained learning:
  - this boundary is not a missing derivative formula; it is a route-ordering
    and post-calculus radical-denominator cleanup issue around mixed tan-root
    expressions
  - future work should add a bounded, held presentation route for these mixed
    tan-root outputs before widening the family to `sqrt(x)` or `exp(x)`
- resolved by:
  - later retained tan-root presentation/residual routes bound both original
    subcases without entering generic cleanup:
    `diff(sqrt(tan(x)+sqrt(x)+x), x)` and
    `diff(sqrt(tan(x)+exp(x)+x), x)`
  - the educational residual
    `diff(sqrt(tan(x)+exp(x)+x), x) -
    (sec(x)^2+e^x+1)/(2*sqrt(tan(x)+exp(x)+x))` now closes to `0` before
    cleanup while preserving `cos(x) != 0` and radicand positivity

## 2026-05-19 - Retained robustness: tan-root diff with sqrt-variable term

- area:
  - calculus / differentiation / post-calculus presentation / mixed
    tan-root elementary terms
- status:
  - `retained-robustness`
- retained:
  - `diff(sqrt(tan(x)+sqrt(x)+x), x)` now uses the bounded direct tan-root
    presentation route and no longer reaches repeated `depth_overflow` during
    smoke
  - the route preserves the tangent pole guard, the `sqrt(x)` derivative
    guard, and the outer radicand positivity
- later resolved:
  - `diff(sqrt(tan(x)+exp(x)+x), x)` was subsequently bounded by the retained
    sec-squared tan-exp root derivative presentation/residual route
- retained learning:
  - adding `sqrt(x)` safely required both a narrow direct derivative term and a
    post-calculus rationalization guard for trig-plus-`sqrt(variable)`
    radicands; either piece alone was not enough to retain the fix

## 2026-05-19 - Discovery observe-only: diff residual after child expansion

- area:
  - calculus / differentiation / exact fraction residual cancellation
- status:
  - `discovery-observe-only`
- observed:
  - the compact algebraic residual
    `-(3*x+1)/(2*sqrt(x)*(x*(x+1)^2+1)) + (6*x+2)/(4*sqrt(x)*(x*(x+1)^2+1))`
    now collapses to `0`
  - the embedded calculus residual
    `diff(arctan(1/(sqrt(x)*(x+1))), x) + (6*x+2)/(4*sqrt(x)*(x*(x+1)^2+1))`
    still fails to collapse after the right-hand denominator is simplified
    independently before the root additive cancellation can see the compact pair
- retained:
  - exact fraction-pair cancellation now recognizes a single rationally scaled
    numerator factor, including distributed negative linear factors
  - a narrow pre-order path lets already-compact exact fraction pairs collapse
    before denominator expansion
- rejected subcandidate:
  - promoting the nested `diff(...) + residual` contract in this iteration; the
    route still needs a shape-preserving residual comparison across one compact
    denominator and one child-expanded denominator
- follow-up:
  - add a bounded denominator-product-vs-expanded-product equivalence probe for
    exact fraction-pair cancellation, then promote the nested diff residual only
    if guardrail and pressure stay at `failed = 0`
- resolved by:
  - the nested residual was promoted as
    `arctan_reciprocal_sqrt_product_diff_residual_collapses_after_child_expansion`
  - the public residual now closes to `0` through `Cancel Opposite Fractions`
    after the derivative step, without warnings or generic fraction cleanup,
    while preserving the required `x > 0` condition
  - the quadratic sibling is also covered by
    `arctan_reciprocal_sqrt_quadratic_product_diff_residual_collapses_after_child_expansion`

## 2026-05-20 - Discovery observe-only: inline reciprocal term in trig-root diff presentation

- area:
  - calculus / differentiation / post-calculus presentation / additive
    trig-root reciprocal terms
- status:
  - `discovery-observe-only`
- observed:
  - trying to present `diff(sqrt(sin(2*x)+cos(x)+1/x), x)` as a chain-rule
    numerator containing an inline `-1/x^2` term reopens deep simplification of
    the radicand and repeatedly hits `depth_overflow` during local smoke
  - the existing common-denominator presentation
    `(2*cos(2*x)*x^2 - sin(x)*x^2 - 1)/(2*x^2*sqrt(sin(2*x)+cos(x)+1/x))`
    remains stable and preserves the required `x != 0` condition
- retained:
  - a focused unit guard now freezes the stable direct presentation for the
    added reciprocal case so a prettier but fragile inline-division route is
    not reintroduced accidentally
- follow-up:
  - improve this presentation only through a bounded held post-calculus display
    route that does not feed nested `Div` terms back into general simplification
- resolved by:
  - the following retained robustness iteration kept the stable
    common-denominator public presentation for
    `diff(sqrt(sin(2*x)+cos(x)+1/x), x)` and added a bounded residual bridge
    for the equivalent inline numerator target
  - the inline educational residual now closes to `0` before general cleanup
    while preserving the outer radicand positivity and `x != 0` conditions
  - no public inline-division presentation was promoted; the retained policy is
    that this family should stay on the stable denominator-cleared display
    unless a future held display route can prove equally bounded

## 2026-05-20 - Retained robustness: inline reciprocal trig-root diff residual

- area:
  - calculus / differentiation / post-calculus residual simplification /
    additive trig-root reciprocal terms
- status:
  - `retained-robustness`
- retained:
  - the direct public result for `diff(sqrt(sin(2*x)+cos(x)+1/x), x)` stays in
    the stable common-denominator presentation
  - the equivalent user-facing residual
    `diff(sqrt(sin(2*x)+cos(x)+1/x), x) -
    (cos(2*x)-sin(x)/2-1/(2*x^2))/sqrt(sin(2*x)+cos(x)+1/x)`
    now closes to `0` before general simplification
  - required conditions remain the outer radicand positivity and `x != 0`
- retained learning:
  - for this family, the safe bridge is not to make the public derivative more
    inline; it is to compare the inline numerator against the stable
    common-denominator numerator by canceling exact extra denominator factors
    locally

## 2026-05-20 - Discovery observe-only: reciprocal-sqrt trig-root residual orientation

- area:
  - calculus / differentiation / post-calculus residual simplification /
    additive trig-root `sqrt(x)` terms
- status:
  - `discovery-observe-only`
- observed:
  - the public derivative
    `diff(sqrt(sin(2*x)+cos(x)+sqrt(x)), x)` remains fast and domain-safe
  - the already-supported residual with numerator term `x^(-1/2)` closes
    before cleanup
  - the mathematically equivalent common-denominator residual
    `diff(sqrt(sin(2*x)+cos(x)+sqrt(x)), x) -
    (4*sqrt(x)*cos(2*x)+1-2*sqrt(x)*sin(x))/
    (4*sqrt(x)*sqrt(sin(2*x)+cos(x)+sqrt(x)))`
    still times out in the public route
- rejected subcandidate:
  - simply making the existing denominator-factor matcher symmetric is not a
    retained fix; the isolated matcher can compare the shapes, but the public
    route still enters a deeper residual/simplification path before producing a
    result
- retained learning:
  - this is a route-ordering/early-exit problem as much as a presentation
    equivalence problem
  - the next retained attempt should add a bounded pre-order residual route for
    the exact `sqrt(x)` common-denominator presentation instead of widening the
    general post-calculus fraction matcher
- resolved by:
  - the immediately following retained robustness iteration added the bounded
    early residual route for the exact `sqrt(x)` common-denominator
    presentation
  - the public residual now closes to `0` before general cleanup, without
    warnings, while preserving the radicand positivity and `x > 0` conditions

## 2026-05-20 - Retained robustness: reciprocal-sqrt trig-root common-denominator residual

- area:
  - calculus / differentiation / post-calculus residual simplification /
    additive trig-root `sqrt(x)` terms
- status:
  - `retained-robustness`
- retained:
  - the public derivative
    `diff(sqrt(sin(2*x)+cos(x)+sqrt(x)), x)` keeps its compact
    common-denominator presentation
  - the common-denominator residual
    `diff(sqrt(sin(2*x)+cos(x)+sqrt(x)), x) -
    (4*sqrt(x)*cos(2*x)+1-2*sqrt(x)*sin(x))/
    (4*sqrt(x)*sqrt(sin(2*x)+cos(x)+sqrt(x)))`
    now closes to `0` before general cleanup
  - required conditions remain the radicand positivity and `x > 0`
- retained learning:
  - the retained fix is an exact early residual route: remove exact denominator
    factors, scale the additive numerator locally, and cancel the bounded
    `sqrt(P) * P^(-1/2)` pair produced by common-denominator presentation
  - this avoids retrying the rejected symmetric broad matcher while closing the
    reusable educational residual shape

## 2026-05-20 - Retained robustness: tan-exp-sqrt inline derivative residual

- area:
  - calculus / differentiation / post-calculus residual simplification /
    cross-family tan + exp + sqrt terms
- status:
  - `retained-robustness`
- retained:
  - the public residual
    `diff(sqrt(tan(x)+exp(x)+sqrt(x)+x), x) -
    (sec(x)^2+e^x+1+1/(2*sqrt(x)))/
    (2*sqrt(tan(x)+exp(x)+sqrt(x)+x))`
    now closes to `0` before general simplification
  - the retained route preserves the existing public derivative presentation
    and the required conditions `cos(x) != 0`, radicand positivity, and `x > 0`
- retained learning:
  - the slow path was not missing calculus knowledge; it was a residual
    comparison gap after scaling an inline target numerator into the engine's
    common-denominator presentation
  - accepting commutative product cores inside the already-bounded additive
    residual matcher is enough to close this family without expanding
    `tan(x)` into `sin(x)/cos(x)` or widening global simplification

## 2026-05-20 - Retained robustness: cot-sqrt root derivative csc presentation

- area:
  - calculus / differentiation / post-calculus presentation /
    additive trig-root `cot(x)` plus `sqrt(x)` terms
- status:
  - `retained-robustness`
- retained:
  - the public derivative `diff(sqrt(cot(x)+sqrt(x)+x), x)` now uses the
    direct `csc(x)^2` presentation instead of expanding `cot(x)` through
    `cos(x)/sin(x)` and entering the cleanup path
  - the matching residual
    `diff(sqrt(cot(x)+sqrt(x)+x), x) -
    (2*sqrt(x)+1-2*sqrt(x)*csc(x)^2)/
    (4*sqrt(x)*sqrt(cot(x)+sqrt(x)+x))`
    now closes to `0` in the public route
  - required conditions remain explicit: radicand positivity, `sin(x) != 0`,
    and `x > 0`
- retained learning:
  - the presentation helper was structurally reusable but still hardcoded
    `sec`; parameterizing the reciprocal trig function lets the cot route
    reuse the same bounded derivative shape with `csc`
  - this is a retained presentation/residual robustness improvement, not new
    calculus semantics and not a global trig rewrite

## 2026-05-20 - Retained robustness: sec/csc root derivative presentation

- area:
  - calculus / differentiation / post-calculus presentation /
    additive reciprocal-trig root terms with `sqrt(x)`
- status:
  - `retained-robustness`
- retained:
  - `diff(sqrt(sec(x)+sqrt(x)+x), x)` no longer enters the general trig
    expansion/cleanup path and now presents directly with `sec(x)*tan(x)`
  - `diff(sqrt(csc(x)+sqrt(x)+x), x)` now presents directly with
    `-csc(x)*cot(x)` instead of expanding through `sin(x)`
  - both matching residuals close to `0` in the public route
  - required conditions remain explicit: radicand positivity, `x > 0`, and
    the relevant reciprocal-trig domain condition (`cos(x) != 0` or
    `sin(x) != 0`)
- retained learning:
  - this was not missing differentiation semantics; it was a missing bounded
    presentation route for elementary reciprocal trig terms inside root
    derivatives
  - recognizing `sec/csc` locally avoids a timeout-prone cleanup route without
    widening global trig simplification or changing canonical internal forms

## 2026-05-20 - Retained robustness: sec/csc exp-root derivative presentation

- area:
  - calculus / differentiation / post-calculus presentation /
    additive reciprocal-trig root terms with `exp(x)` and polynomial noise
- status:
  - `retained-robustness`
- retained:
  - `diff(sqrt(sec(x)+exp(x)+x), x)` no longer times out and now presents as
    `(e^x + tan(x)*sec(x) + 1)/(2*sqrt(sec(x)+e^x+x))`
  - `diff(sqrt(csc(x)+exp(x)+x), x)` no longer times out and now presents as
    `(e^x + 1 - csc(x)*cot(x))/(2*sqrt(csc(x)+e^x+x))`
  - both matching residuals close to `0` in the public route
  - required conditions remain explicit: radicand positivity plus the relevant
    reciprocal-trig domain condition (`cos(x) != 0` or `sin(x) != 0`)
- retained learning:
  - the previous reciprocal-trig presentation route was still too dependent on
    seeing `sqrt(x)` or `1/sqrt(x)` as a special denominator builder
  - once the radicand derivative terms are already known, the reusable bounded
    exit is the ordinary calculus shape `radicand'/(2*sqrt(radicand))`; using
    that avoids the timeout-prone general cleanup route without broadening
    global trig simplification

## 2026-05-20 - Retained robustness: sec/csc log-root derivative presentation

- area:
  - calculus / differentiation / post-calculus presentation /
    additive reciprocal-trig root terms with `ln(x)` and polynomial noise
- status:
  - `retained-robustness`
- retained:
  - `diff(sqrt(sec(x)+ln(x)+x), x)` no longer times out and now presents as
    `(x*tan(x)*sec(x) + x + 1)/(2*x*sqrt(sec(x)+ln(x)+x))`
  - `diff(sqrt(csc(x)+ln(x)+x), x)` no longer times out and now presents as
    `(x + 1 - x*csc(x)*cot(x))/(2*x*sqrt(csc(x)+ln(x)+x))`
  - both matching residuals close to `0` in the public route
  - required conditions remain explicit: radicand positivity, `x > 0`, and
    the relevant reciprocal-trig domain condition (`cos(x) != 0` or
    `sin(x) != 0`)
- retained learning:
  - the previous retained `sec/csc` exit covered numerator-only radicand
    derivatives, but `ln(x)` introduces a common denominator that must be
    represented directly in the post-calculus fraction
  - scaling known derivative terms by the common denominator and appending the
    logarithmic reciprocal contribution closes this family without entering
    the timeout-prone general cleanup route

## 2026-05-20 - Observe-only discovery: hyperbolic double-nested denominator residual timeout

- area:
  - calculus / integration residual smoke / post-calculus presentation /
    shell-depth 3 denominator wrappers
- status:
  - `discovery/observe-only`
- observed:
  - after preserving compact triple binomial denominator products, the
    `double_nested_den` wrapper passes quickly for representative polynomial,
    arctan-root, and fractional-power residuals
  - the same wrapper times out for the hyperbolic by-parts residual families
    `hyperbolic_sinh` and `hyperbolic_cosh` under the 6s smoke timeout
- decision:
  - do not promote the hyperbolic depth-3 wrapper into the default smoke matrix
    in this iteration
  - retain only the minimal passing representatives as discovery pressure
- retained learning:
  - shell-depth 3 denominator presentation is not uniformly cheap across
    integration residual families
  - the hyperbolic by-parts residual path needs a narrower residual/presentation
    exit before it is safe to expand the wrapper globally
- resolved by:
  - 2026-05-20 robustness iteration widened only the hyperbolic
    constant-passthrough quotient residual adapter to depth-2 nested quotients
  - `hyperbolic_sinh:double_nested_den` and `hyperbolic_cosh:double_nested_den`
    now pass the focused smoke under 6s and were promoted as minimal
    representatives
  - 2026-05-22 robustness iteration revalidated the default smoke signature
    after it reproduced locally; the retained fix keeps the depth-2 quotient
    allowance scoped to the hyperbolic adapter and preserves all three
    denominator `NonZero` conditions

## 2026-05-20 - Observe-only discovery: non-corpus rational quadratic wrapper warning

- area:
  - calculus / integration residual smoke / rational quadratic antiderivative
    verification / shell-depth 3 denominator wrappers
- status:
  - `discovery/observe-only`
- observed:
  - a generated rational quadratic wrapper around
    `diff(integrate((3*x + 5)/(x^2 + x + 1),x),x)-((3*x + 5)/(x^2 + x + 1))`
    returns the compact expected quotient, but emits a `depth_overflow` warning
    in the rationalization phase under `forbid_warnings`
  - the retained corpus representative `rational_quad:double_nested_den`
    remains clean: it preserves `x + 1`, `x + 2`, `x + 3`, `x + 4` conditions
    and passes without warnings
- decision:
  - do not promote the non-corpus rational quadratic variant in this iteration
  - promote only the existing stable `rational_quad` representative as the
    minimal shell-depth 3 coverage case
- retained learning:
  - rational quadratic antiderivative residuals can be mathematically closed
    while still leaking a rationalization-depth warning on some non-canonical
    denominators
  - a future iteration should address this as robustness/presentation only if
    the warning is reproduced by a minimal stable representative
- resolved by:
  - 2026-05-20 robustness iteration widened only the rational quadratic
    constant-passthrough quotient residual adapter to depth-2 nested quotients
  - `rational_quad_positive_quadratic:double_nested_den` now passes under
    `forbid_warnings` with only `x + 2`, `x + 3`, and `x + 4` conditions
  - 2026-05-22 robustness iteration revalidated the current default smoke
    signature after it reproduced locally; the retained fix keeps the depth-2
    quotient allowance scoped to supported rational-quadratic residuals

## 2026-05-21 - Discovery observe-only: inline sqrt-variable tan-root presentation residual timeout

- area:
  - calculus / differentiation / post-calculus presentation /
    additive `tan` root terms with `sqrt(x)`
- status:
  - `discovery/observe-only`
- observed:
  - a local presentation candidate rewrote
    `diff(sqrt(tan(x)+sqrt(x)+x), x)` from the stable common-denominator form
    `(2*sqrt(x)+2*sqrt(x)*sec(x)^2+1)/(4*sqrt(x)*sqrt(tan(x)+sqrt(x)+x))`
    to the more chain-rule-like
    `(sec(x)^2 + 1/(2*sqrt(x)) + 1)/(2*sqrt(tan(x)+sqrt(x)+x))`
  - direct public diff probing produced the prettier form with the same domain
    conditions and no warnings
  - the matching residual contract for the cross-family sibling
    `diff(sqrt(tan(x)+exp(x)+sqrt(x)+x), x)` timed out after more than 60s
    when the inline `1/(2*sqrt(x))` numerator term was used
- decision:
  - do not retain or promote the inline numerator presentation in this
    iteration
  - keep the current common-denominator presentation for `tan + sqrt(x)` root
    derivatives because its residual contracts close before cleanup
- retained learning:
  - for this family, prettier chain-rule display is not enough; the displayed
    form must also have a bounded residual comparison route
  - future work should first add an exact early residual route for inline
    `1/(2*sqrt(x))` terms in `tan` root derivatives, then revisit the public
    presentation
- partial follow-up:
  - 2026-05-21 robustness iteration retained an exact, presentation-only
    residual route for the pure inline-vs-common-denominator equality
  - this removes the depth-overflow route for the standalone presentation
    identity
  - later 2026-05-21 robustness iteration wired the same exact matcher into
    the private `diff(...) - target` residual route, so the embedded inline
    target direction is now bounded without changing the public `diff`
    presentation yet
- resolved by:
  - current public `diff` contracts now retain the inline presentation for
    `diff(sqrt(tan(x)+exp(x)+sqrt(x)+x), x)` directly, with the tangent pole,
    radicand positivity, and `x > 0` requirements intact
  - both the inline target residual and the common-denominator target residual
    close to `0` through the bounded post-calculus residual route before
    generic cleanup, with no warnings or `depth_overflow`
  - the retained value is the exact tan/exp/sqrt-root presentation-residual
    bridge already covered by the public differentiation contracts, not a
    broader global rule for arbitrary inline radical numerators

## 2026-05-23 - Rejected runtime: broad direct-chunk candidate gate before small-zero partition enumeration

- area:
  - arithmetic / exact-zero additive composition / mixed log plus hyperbolic
    pressure
- status:
  - `rejected`
- local lane:
  - `simplify_zero_mixed` probes for `sum@700+100 #2865` and `#3021`
- local signal:
  - accepting top-level direct chunks before subset enumeration looked useful
    on the log-square / log-abs plus hyperbolic-cubic probes
  - the same run still left the repeated negated-log `default_simplify` traffic
    visible, so the local win was not causally clean
- global result:
  - pressure profile with the broad gate regressed to `2.19s` for
    `simplify_zero_mixed` versus the retained baseline around `2.00s`
  - after removing the gate and keeping only the narrow negated-log reject,
    pressure returned to `1.99s` with `failed = 0`
- why it regressed globally:
  - invoking `try_build_small_direct_zero_core_rewrite` from the generic
    candidate gate broadens work across unrelated mixed-pressure partitions
    before a cheap enough syntactic proof exists
- what could make it combinable later:
  - move any direct-chunk preacceptance to a narrower orchestrator call-site or
    add a cheaper structural signature that proves both chunks are known
    zero-family members without running rewrite builders
- retained alternative:
  - a scoped reject for nonreciprocal `-ln(a)` versus `ln(b)` pairs removed the
    hot `default_simplify` no-match traffic without broadening the candidate
    gate

## 2026-05-24 - Rejected: pre-core log/trig plus hyperbolic exact-zero shortcut

- area:
  - orchestrator / exact-zero additive composition / pre-core routing
- status:
  - `rejected/local-win-global-loss`
- hypothesis tested:
  - detect hot root pairs where one side is a known log or trig zero identity
    and the other side is a known hyperbolic zero identity, then run the
    existing direct small-zero additive-combination rewrite before the Core
    phase
- local signal:
  - focused unit probes passed and confirmed the route could skip Core for
    `ln(x^3) + ln(y^2) - ln(x^3 * y^2)` plus
    `cosh(x) + sinh(x) - e^x`
  - the implementation had to explicitly reattach implicit input-domain
    requirements to preserve the shortcut step metadata
- global result:
  - `make engine-scorecard` stayed semantically green, but embedded runtime
    moved from `12.82s` to `13.38s`
  - `make engine-scorecard-pressure` regressed `simplify_zero_mixed` from the
    retained `2.03s` baseline to `2.25s`
  - the dominant log/hyperbolic pressure expressions did not improve
    materially; `sum@700+100 #2865` remained around `8.72ms` steady-state
    median simplify time
- why it regressed globally:
  - placing this route before Core adds domain inference and direct-rewrite
    builder work on a path where Core still performs useful normalization for
    these shapes
  - the pressure bottleneck is not just the late compact shortcut; it includes
    the cost and ordering of log-domain/domain-preservation work around the
    exact-zero proof
- what could make it combinable later:
  - add observability around domain inference and direct rewrite-builder cost
    for log/hyperbolic roots before trying another pre-core route
  - prefer a cheaper structural proof for a single concrete family over a
    pre-Core call into the general additive-combination rewrite builder
- retained alternative:
  - reverted the pre-core route and kept the previous post-Core narrow
    log/nested-fraction plus hyperbolic shortcuts

## 2026-05-24 - Discovery observe-only: hyperbolic angle-sum plus telescoping residual remains slow

- area:
  - orchestrator / exact-zero additive composition / hyperbolic angle-sum plus
    telescoping-fraction residuals
- status:
  - `discovery/observe-only`
- observed:
  - during the nested-fraction plus hyperbolic angle-sum runtime iteration, a
    temporary promotion attempt for
    `(sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y))) + (1/(u*(u+1)) - 1/u + 1/(u+1))`
    stayed correct but remained slow in the direct `simplify_pipeline` debug
    probe, finishing around `3.6s`
  - the nested-fraction sibling had a cheap retained syntactic route and is
    covered by an active unit test; the telescoping sibling did not show the
    same bounded route
- decision:
  - do not broaden the retained shortcut from nested fractions to consecutive
    telescoping fractions in this iteration
  - keep the telescoping `simplify_pipeline` regression ignored until a cheap
    telescoping-side proof or a narrower root route exists
- resolved by:
  - later exact-zero additive composition and telescoping-fraction routes now
    make the direct `simplify_pipeline` regression pass in both debug and
    release without a special broad pre-Core shortcut
  - the previously ignored hyperbolic angle-sum plus telescoping residual test
    has been promoted back to an active unit regression guard; sibling
    cosh-cubic plus telescoping coverage remains active
- retained learning:
  - the reusable weakness is not the hyperbolic angle-sum side; it is the lack
    of a cheap, promotion-ready telescoping-fraction zero proof for this
    cross-family additive composition
  - a future runtime candidate should isolate the telescoping side first, then
    re-test this composition before promoting the ignored regression

## 2026-05-25 - Discovery observe-only: limit residual result/step presentation can diverge after structural cleanup

- area:
  - calculus / limit / residual presentation after safe structural cleanup
- status:
  - `discovery/observe-only`
- observed:
  - while auditing whether `limit` needed the same presimplified-residual trace
    policy recently retained for `diff`/`integrate`, cheap CLI probes showed
    that residual limit steps can display the cleaned expression while the
    public result still displays the noisier original residual
  - examples:
    - `limit(sqrt(x + 0), x, 0)` renders result
      `limit(sqrt(x + 0), x, 0)` but the residual step goes
      `sqrt(x) -> limit(sqrt(x), x, 0)`
    - `limit(sign(x + 0), x, 0)` similarly keeps `x + 0` in the result while
      the step shows `sign(x)`
- decision:
  - do not fix this in the current cycle because the retained ROI is better as
    a support-matrix promotion for an already generalized finite-limit family
  - treat this as a future presentation/observability candidate, not as a
    reason to broaden limit simplification policy
- resolved by:
  - the command-level `limit` matrix now retains both minimal residual cleanup
    examples from the discovery: the domain-bearing endpoint case
    `limit(sqrt(x + 0), x, 0)` and the discontinuous no-condition case
    `limit(sign(x + 0), x, 0)`
  - both rows keep the command residual, preserve finite-limit warning policy,
    and require result/step presentation to agree on the structurally cleaned
    residual form
- retained learning:
  - the reusable weakness is result/step presentation consistency for residual
    `limit(...)` after structural cleanup, not mathematical limit evaluation
  - a future candidate should localize whether the mismatch comes from residual
    result construction, display rendering, or step rendering before changing
    public limit policy

## 2026-05-27 - Discovery observe-only: broad sec/csc result holding regresses stable integration presentation

- area:
  - calculus / integration / reciprocal trig derivative product presentation
- status:
  - `discovery/observe-only`
- observed:
  - while retaining shifted-polynomial reciprocal trig products such as
    `integrate(2*x*sec(x^2+b)*tan(x^2+b), x)`, a broad result-side compacting
    flag for any `sec(...)`/`csc(...)` antiderivative fixed the new local
    presentation but changed existing promoted rows
  - examples from the public integration matrix:
    - `sec(2*x+1)*tan(2*x+1)` changed from `sec(2·x + 1) / 2` to
      `1/2·sec(2·x + 1)`
    - negative symbolic external-scale rows changed factor ordering from
      `k·sec(b - a·x)`/`-k·csc(b - a·x)` to trailing-`k` forms
- decision:
  - reject broad result-based holding for reciprocal trig primitives
  - retain the narrower source-side presentation hold only when the exact
    integration route has to ignore an additive term independent of the
    integration variable
- superseded by:
  - the command-level integration matrix now pins the stable public
    presentation for the affected reciprocal-trig derivative-product cells,
    including affine `sec(u)*tan(u)` and negative symbolic external-scale
    `sec`/`csc` rows
  - the rejected broad result-side holding remains documented as retained
    learning, but it is no longer an open promotion candidate because the
    narrower source-side policy and matrix representatives protect the intended
    behavior without changing unrelated rows
- retained learning:
  - compactness should be attached to the specific derivative-cofactor
    recognition path that needs it, not to every `sec`/`csc` result after the
    integrator has already chosen a presentation
  - future reciprocal-trig presentation work should add family-specific
    predicates before the final simplification boundary instead of broad
    result-shape checks

## 2026-05-27 - Discovery observe-only: shifted sech-fourth primitive verification is structurally fragile

- area:
  - calculus / integration / hyperbolic reciprocal fourth verification
- status:
  - `discovery/observe-only`
- observed:
  - while probing a generalization from retained `1/cosh(u)^4` support to
    shifted polynomial arguments with symbolic external scale, the public
    result for `integrate(2*k*x/cosh(x^2+b)^4, x)` was computable and
    `diff(integrate(2*k*x/cosh(x^2+b)^4, x), x) - 2*k*x/cosh(x^2+b)^4`
    reduced to `0`
  - verifying the explicit primitive, for example
    `diff(1/3*(3*k*tanh(x^2+b)-k*tanh(x^2+b)^3), x) - 2*k*x/cosh(x^2+b)^4`,
    repeatedly hit `depth_overflow` and `cycle_detected` paths before the
    integration matrix timeout
- decision:
  - reject the shifted/symbolic `sech^4` capability promotion for now
  - retain only the smaller matrix/didactic promotion for the already-supported
    affine `1/cosh(2*x+1)^4` case
- resolved by:
  - the retained shifted `sech^4` symbolic cofactor follow-up added the bounded
    `R - R*tanh(u)^2 -> R/cosh(u)^2` verification route and promoted
    `integrate(2*k*x/cosh(x^2+b)^4, x)` as a verified matrix row
  - the original depth/cycle failure remains useful as historical context, but
    the reusable blocker it identified is no longer open for this exact
    shifted symbolic-cofactor family
- retained learning:
  - the reusable blocker is explicit verification of
    `tanh(u) - tanh(u)^3/3` against `1/cosh(u)^4` when `u` contains an
    additive symbolic shift; the simplifier expands hyperbolic angle sums
    instead of taking a bounded `1 - tanh(u)^2 -> 1/cosh(u)^2` route
  - the retained affine matrix row is covered by the numeric-coefficient route
    `k - k*tanh(u)^2 -> k/cosh(u)^2`; a follow-up probe still reproduced
    depth/cycle warnings for symbolic common cofactors such as `6*k*x -
    6*k*x*tanh(x^2+b)^2`
  - a future candidate should first add a bounded common-cofactor route for
    `R - R*tanh(u)^2 -> R/cosh(u)^2` without expanding shifted hyperbolic
    sums, then retry the shifted symbolic integration generalization

## 2026-05-27 - Retained follow-up: shifted sech-fourth symbolic cofactor route

- area:
  - calculus / integration / hyperbolic reciprocal fourth verification
- status:
  - `retained`
- observed:
  - the bounded common-cofactor route
    `R - R*tanh(u)^2 -> R/cosh(u)^2` avoids shifted hyperbolic sum expansion
    for symbolic cofactors such as `6*k*x - 6*k*x*tanh(x^2+b)^2`
  - `integrate(2*k*x/cosh(x^2+b)^4, x)` now promotes as an explicit
    `1/cosh(u)^4` substitution case and verifies by differentiating the
    explicit primitive
- retained learning:
  - the prior blocker was the verification/simplification route, not the
    hyperbolic reciprocal fourth integration formula
  - symbolic external-scale `sech^4` cases should remain bounded to exact
    derivative-cofactor substitution before broadening to more general
    hyperbolic powers

## 2026-05-27 - Discovery observe-only: csch-fourth primitive verification lacks bounded route

- area:
  - calculus / integration / hyperbolic reciprocal fourth residual verification
- status:
  - `discovery/observe-only`
- observed:
  - `integrate(1/sinh(2*x+1)^4, x)` and
    `integrate(2*k*x/sinh(x^2+b)^4, x)` remain domain-aware residuals with
    `sinh(...) != 0`
  - explicit `cosh`/`sinh` primitive verification repeatedly entered
    `depth_overflow` and `cycle_detected` routes before completion
  - the tanh-reciprocal primitive avoids the expansion cliff but leaves a
    residual shaped like
    `1/(cosh(u)^2*tanh(u)^4) - 1/sinh(u)^4 - 1/sinh(u)^2`
- decision:
  - do not promote `csch^4` integration in this cycle
  - retain one residual/domain matrix row so the unsupported dual of `sech^4`
    is visible and does not silently become an unverified antiderivative
- resolved by:
  - the retained shifted `csch^4` symbolic cofactor follow-up promoted
    `integrate(2*k*x/sinh(x^2+b)^4, x)` with explicit
    `sinh(x^2 + b) != 0` domain retention and verified the antiderivative by
    differentiation
  - the verifier now recognizes the tanh-square denominator presentation
    variants that originally kept the primitive in residual form
- retained learning:
  - the next reusable engine candidate should add a bounded verifier
    normalization for tanh-denominator powers, or an equivalent
    common-denominator hyperbolic reciprocal identity, before retrying the
    antiderivative

## 2026-05-27 - Retained follow-up: shifted csch-fourth symbolic cofactor route

- area:
  - calculus / integration / hyperbolic reciprocal fourth residual verification
- status:
  - `retained`
- observed:
  - promoting `integrate(2*k*x/sinh(x^2+b)^4, x)` first failed antiderivative
    verification even though the integration result and domain were public and
    stable
  - the residual appeared as a bounded identity with two presentation variants:
    `1/(cosh(u)^2*tanh(u)^2)` instead of `1/sinh(u)^2`, plus external
    cofactor forms such as `2*(k*x/den)` after fraction cleanup
- retained learning:
  - the reusable blocker was not the integration formula; it was the verifier's
    narrow term-shape matcher after normal simplification rewrites
  - `csch^4` verification now recognizes the equivalent tanh-square
    denominator and absorbs a single externally scaled fraction term before
    comparing cofactors
  - the verifier rule is targeted to additive nodes with priority above broad
    exact-zero subset search, preventing the expensive generic route from
    consuming the public simplification budget first

## 2026-05-28 - Partial rejection: atanh open-interval compaction must skip rational-only scales

- area:
  - calculus / differentiation / atanh domain-condition presentation
- status:
  - `retained-with-narrowing`
- observed:
  - a local helper that compacted `(scale*sqrt(radicand))^2` inside the
    `atanh` open-interval condition fixed the symbolic case
    `diff(atanh(a*sqrt(x+1)), x)`
  - applying the same path to rational-only scales changed the established
    display order for `diff(atanh(sqrt(x+1)/3)/sqrt(5), x)` from
    `x < 8, x > -1` to `x > -1, x < 8`, tripping the promoted diff contract
- retained learning:
  - symbolic scale compaction is useful and domain-safe, but rational-only
    scaled-root `atanh` cases already have a stable presentation path
  - future condition-presentation helpers should avoid intercepting
    rational-only scale cases unless they explicitly preserve the promoted
    condition ordering contract

## 2026-05-28 - Discovery observe-only: atanh exact-square denominator scale hits depth overflow

- area:
  - calculus / differentiation / atanh domain-condition presentation
- status:
  - `discovery/observe-only`
- observed:
  - the sibling probe `diff(atanh(sqrt(4*x+4)/a), x)` still emits
    `depth_overflow` on stderr before returning a result
  - the returned domain display remains split between `a != 0`,
    `x > -1`, `a^2 - 4*x - 4 != 0`, and the uncompact
    `1 - (((x + 1)^(1/2)*2)/a)^2 > 0`
- decision:
  - do not promote the exact-square scaled-radicand variant in this cycle
  - keep the retained case to the minimal symbolic denominator scale
    `sqrt(x+1)/a`, which adds a distinct public domain presentation regime
    without the overflow route
- retained learning:
  - exact-square/rational-factor extraction before atanh open-interval
    compaction needs a bounded route; otherwise it can expose depth overflow
    and redundant nonzero conditions even when the final inequality should
    collapse to a polynomial gap
- follow-up resolution:
  - retained command-matrix and public CLI contract coverage for
    `diff(atanh(sqrt(4*x+4)/a), x)`
  - the public result stays compact, the required display is now the bounded
    open-interval policy `a != 0`, `a^2 - 4*x - 4 > 0`, `x > -1`, and the
    CLI trace keeps the perfect-square extraction, constant factoring,
    chain-rule, and `u/du` evidence without `depth_overflow` warnings

