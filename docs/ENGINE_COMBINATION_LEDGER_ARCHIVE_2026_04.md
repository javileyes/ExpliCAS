# Engine Combination Ledger - Archive 2026-04

Entries rotated from [ENGINE_COMBINATION_LEDGER.md](ENGINE_COMBINATION_LEDGER.md).
Scorecard discovery metrics still read this file; treat it as
read-only history and do not add new entries here.

## 2026-04-29 - Auto-improvement cycle: derive cofunction symmetry promotion

- candidate:
  - promote `cos(pi/2 - x) -> sin(x)` as the complementary sine/cosine
    cofunction representative in `derive_pairs.csv`
- files_changed:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [derive_command.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: `derive cos(pi/2 - x), sin(x)` is promoted to the
    primary derive contract under `expand trig`, both cofunction
    representatives keep the visible `Cofunction Identity` route with no
    redundant substeps, no generic simplify expectation appears, and scorecard
    guardrails stay green
  - `primary_dimension`: engine-to-derive bridge coverage for complementary
    sine/cosine cofunction symmetry
  - `secondary_dimension`: didactic route-quality regression for a direct,
    self-explanatory identity
  - `hypothesis`: target-aware cofunction support already handles both
    directions, but the primary derive corpus only covered `sin(pi/2 - x)`;
    adding the complementary representative improves contract symmetry without
    changing runtime search
  - `relevant_lanes`: CLI cofunction probe, target-aware trig unit, direct
    derive-command regression, derive generic-simplify guard, release derive
    contract, release derive didactic audit, `make engine-fast`,
    `make engine-scorecard`
  - `promotion_target`: `derive_pairs.csv`
  - `derive_bridge_check`: promoted because `identity_pairs.csv` contains both
    sine/cosine cofunction directions while the derive contract had only the
    sine source representative
  - `engine_feedback_check`: classified as derive/corpus path-quality coverage;
    no reusable simplification runtime rule was missing
  - `retain_if`: the new row derives with `expand trig`, emits exactly
    `Cofunction Identity`, remains accepted as a no-padding direct step in the
    didactic audit, and global guardrails remain green
  - `reject_if`: the row triggers shape-budget pressure, falls back to generic
    simplify, or creates noisy didactic substeps/flags
- structural_axis:
  - trig cofunction target-family symmetry
- why_this_is_not_a_duplicate:
  - the existing row covered only `sin(pi/2 - x) -> cos(x)`; this adds the
    complementary `cos(pi/2 - x) -> sin(x)` representative already supported by
    the runtime and by identity corpus symmetry
- discovery_or_promotion:
  - promotion of an existing stable runtime route into the primary derive
    contract
- if_promoted_why_minimal_representative:
  - exactly one complementary no-passthrough row was added; plus-sign variants
    remain covered by the lower-level target-aware trig unit
- local_result:
  - added `expand_trig_cofunction_cosine_minus` to the derive contract
  - added a direct derive regression covering both sine and cosine cofunction
    minus forms
  - expanded the didactic audit regression to require a visible, direct
    cofunction step for both representatives
  - CLI probe for `derive cos(pi/2 - x), sin(x)` reported `Strategy: expand
    trig` and `[Cofunction Identity]`
- guardrails:
  - `cargo fmt`
  - `cargo test -q -p cas_solver derive_command::tests::direct_derive_rewrites_cofunction_sine_cosine_pair_with_specific_rule -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_cofunction_identity_uses_visible_rule_without_padding -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive::trig::tests::rewrites_cofunction_phase_shift_targets_before_generic_simplify -- --exact --nocapture`
  - CLI probe: `derive cos(pi/2 - x), sin(x)` reported `Strategy: expand trig`
    and `[Cofunction Identity]`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=346`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `expand trig=86`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=430`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=457`
  - `make engine-fast`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=346 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `430 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as coverage plus route-quality symmetry: the engine already knew
    the identity, and the improvement makes `derive` measure and preserve the
    complementary route explicitly

## 2026-04-29 - Auto-improvement cycle: hyperbolic parity derive route

- candidate:
  - fix `derive tanh(-x), -tanh(x)` so hyperbolic odd/even parity is routed and
    taught as hyperbolic parity instead of being captured by the trig parity
    route
- files_changed:
  - [hyperbolic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/hyperbolic.rs)
  - [trig.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/trig.rs)
  - [derive_command.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs)
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [visible_rule_names.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/visible_rule_names.rs)
  - [focused_rule_substeps.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: robustness
  - `success_condition`: `derive tanh(-x), -tanh(x)` reports `Strategy:
    rewrite hyperbolics`, emits `[Hyperbolic Parity (Odd/Even)]`, has a
    didactic parity substep, promotes one primary derive row, and keeps global
    scorecard guardrails green
  - `primary_dimension`: route quality and family classification for
    hyperbolic odd/even parity
  - `secondary_dimension`: engine-to-derive bridge coverage for
    sign/orientation robustness
  - `hypothesis`: the shared parity rewriter already covered `sinh/cosh/tanh`,
    but the derive trig expansion path consumed those cases first; adding a
    hyperbolic parity kind and restricting the trig normalizer to circular trig
    functions corrects the route without broad search
  - `relevant_lanes`: hyperbolic target-aware unit, trig parity regression,
    direct derive-command regression, focused derive didactic regression, CLI
    derive probe, derive generic-simplify guard, release derive contract,
    release derive didactic audit, `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: `derive_pairs.csv`
  - `derive_bridge_check`: promoted `tanh(-x) -> -tanh(x)` as the minimal
    primary representative for hyperbolic parity identities already present in
    `identity_pairs.csv`; `sinh/cosh` variants stay in target-aware and direct
    unit tests
  - `engine_feedback_check`: derive route/classification bug; no missing
    simplification runtime capability
  - `retain_if`: CLI/direct tests report `Hyperbolic Parity (Odd/Even)` under
    `rewrite hyperbolics`, trig parity still passes for `tan(-x)`, the derive
    row does not add generic simplify pressure, the didactic audit stays
    unflagged, and embedded/simplify guardrails remain green
  - `reject_if`: the new route steals circular trig parity, regresses existing
    hyperbolic identities, emits noisy didactic substeps, or causes guardrail
    failures/timeouts
- structural_axis:
  - sign/orientation robustness for odd/even hyperbolic functions
- why_this_is_not_a_duplicate:
  - the engine already had the algebraic parity rewrite, but derive mislabeled
    hyperbolic functions as trig parity and selected `expand trig`; this fixes
    a route-quality bug and then promotes the smallest guardrail row
- discovery_or_promotion:
  - discovery from a CLI probe plus promotion of the minimal stable row
- if_promoted_why_minimal_representative:
  - `tanh(-x) -> -tanh(x)` is the smallest user-visible representative that
    reproduced the wrong trig route; `sinh(-x)` and `cosh(-x)` validate breadth
    in unit tests without inflating the primary derive corpus
- local_result:
  - added `HyperbolicOddEvenParity` and a target-aware hyperbolic parity route
  - constrained trig parity normalization to circular trig functions only
  - added visible Spanish title `Aplicar paridad hiperbólica` and reused the
    parity formula substep
  - added `hyperbolic_negative_tanh_parity` to the derive contract
  - CLI probe for `derive tanh(-x), -tanh(x)` now reports `Strategy: rewrite
    hyperbolics` and `[Hyperbolic Parity (Odd/Even)]`
- guardrails:
  - `cargo fmt`
  - `cargo test -q -p cas_solver derive::hyperbolic::tests::target_aware_hyperbolic_rewrite_rewrites_negative_parity_variants -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive::trig::tests::rewrites_negative_trig_parity_variants_target_aware -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive_command::tests::direct_derive_rewrites_negative_hyperbolic_parity_with_specific_rule -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_hyperbolic_negative_parity_uses_specific_identity -- --exact --nocapture`
  - CLI probe: `derive tanh(-x), -tanh(x)` reported `Strategy: rewrite
    hyperbolics` and `[Hyperbolic Parity (Odd/Even)]`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=347`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `rewrite hyperbolics=28`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=431`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=458`
  - `make engine-fast`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=347 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `431 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as a robustness/path-quality fix: the algebraic capability already
    existed, but `derive` was teaching the wrong family; now hyperbolic parity
    is classified, measured, and explained through its own route

## 2026-04-29 - Auto-improvement cycle: hyperbolic quotient derive bridge

- candidate:
  - promote `derive sinh(x)/cosh(x), tanh(x)` as a minimal engine-to-derive
    bridge for the hyperbolic quotient identity, with visible didactic wording
    and a formula substep
- files_changed:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [visible_rule_names.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/visible_rule_names.rs)
  - [focused_rule_substeps.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: `derive sinh(x)/cosh(x), tanh(x)` remains under
    `Strategy: rewrite hyperbolics`, the derive corpus expects `rewrite
    hyperbolics`, the visible didactic rule is `Aplicar identidad hiperbólica
    de cociente`, and the audit emits a non-generic quotient formula substep
  - `primary_dimension`: engine-to-derive bridge coverage for a hyperbolic
    quotient identity already present in `identity_pairs.csv`
  - `secondary_dimension`: didactic path quality for direct hyperbolic
    identities that should not look magical or generic
  - `hypothesis`: the route already existed and was semantically correct, but
    the case was absent from `derive_pairs.csv` and the rule had no visible
    Spanish title/substep; promoting one minimal representative improves
    coverage without adding search or runtime traffic
  - `relevant_lanes`: CLI derive probe, derive generic-simplify guard, release
    derive contract, focused derive didactic regression, release derive
    didactic audit, `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: `derive_pairs.csv`
  - `derive_bridge_check`: promoted only `sinh(x)/cosh(x) -> tanh(x)` as the
    minimal contraction representative; the inverse expansion and passthrough
    variants are deferred until they add separate path or domain value
  - `engine_feedback_check`: no missing simplification runtime capability; this
    was corpus/didactic bridge coverage over an existing reusable engine
    transition
  - `retain_if`: the new row derives without generic `simplify`, audit flags
    stay at zero, and embedded/simplify guardrails remain green
  - `reject_if`: the row needs a generic strategy expectation, the substep is
    pruned or flagged as opaque/noisy, or global scorecard guardrails regress
- structural_axis:
  - quotient-to-function transition coverage inside hyperbolic identities
- why_this_is_not_a_duplicate:
  - existing hyperbolic rows covered exponential definitions, pythagorean
    identities, angle identities, parity, and double/triple/product routes, but
    not the basic `sinh(u)/cosh(u) = tanh(u)` quotient identity as a derive
    contract row
- discovery_or_promotion:
  - promotion from an already-supported CLI route and identity-pair seed
- if_promoted_why_minimal_representative:
  - one direct contraction row is the smallest durable case; it avoids adding
    inverse/domain variants until they expose a distinct derive or didactic
    requirement
- local_result:
  - added `hyperbolic_contract_tanh_quotient` to the derive corpus
  - mapped `Hyperbolic Quotient Identity` to `Aplicar identidad hiperbólica de
    cociente`
  - added a focused formula substep `Usar sinh(u) / cosh(u) = tanh(u)`
  - added a didactic audit regression for the visible rule and substep
- guardrails:
  - `cargo fmt`
  - CLI probe: `derive sinh(x)/cosh(x), tanh(x)` reports `Strategy: rewrite
    hyperbolics` and `[Hyperbolic Quotient Identity]`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_hyperbolic_quotient_uses_specific_identity -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=348`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `rewrite hyperbolics=29`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=432`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=459`
  - `make engine-fast`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=348 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `432 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as a conservative coverage/didactic bridge: the engine already had
    the identity, and now `derive` measures and teaches the minimal
    target-form transition explicitly

## 2026-04-29 - Auto-improvement cycle: reciprocal trig expansion derive promotion

- candidate:
  - promote `derive sec(x), 1/cos(x)` from an inline didactic probe into the
    main derive contract as the minimal expansion-direction representative for
    reciprocal trig identities
- files_changed:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: `derive sec(x), 1/cos(x)` reports `Strategy: expand
    trig`, the derive corpus expects `expand trig`, generic simplify pressure
    stays zero, and the derive didactic audit stays unflagged
  - `primary_dimension`: derive contract coverage for the expansion orientation
    of reciprocal trig identities
  - `secondary_dimension`: engine-to-derive bridge coverage for a route already
    supported by the reciprocal trig transition provider
  - `hypothesis`: the route was already correct and had a focused inline
    didactic regression, but it did not count in `derive_pairs.csv` or the
    scorecard; promoting one representative converts local proof into a durable
    guardrail without adding runtime traffic
  - `relevant_lanes`: CLI derive probe, focused derive didactic regression,
    derive generic-simplify guard, release derive contract, release derive
    didactic audit, `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: `derive_pairs.csv`
  - `derive_bridge_check`: promoted `sec(x) -> 1/cos(x)` because it exercises a
    target-form expansion path, not only the existing contraction
    `1/cos(x) -> sec(x)`
  - `engine_feedback_check`: no missing reusable engine capability; this was a
    corpus/guardrail promotion over an already-stable transition
  - `retain_if`: `derive_contract` increases by one under `expand trig`,
    `generic_simplify_expected` remains `0`, audit flags remain `0`, and
    embedded/simplify guardrails remain green
  - `reject_if`: the promoted row falls through a generic strategy, causes a
    didactic flag, or changes global guardrail pass/fail/runtime materially
- structural_axis:
  - expansion orientation for reciprocal trig functions
- why_this_is_not_a_duplicate:
  - the corpus already covered contraction `1/cos(x) -> sec(x)`, but not the
    inverse expansion route with its explicit nonzero cosine condition
- discovery_or_promotion:
  - promotion of an already-supported inline didactic case into the main derive
    contract
- if_promoted_why_minimal_representative:
  - `sec(x)` is the smallest representative for the reciprocal expansion
    family; `csc` and `cot` are deferred until they add distinct coverage or
    domain/path value
- local_result:
  - added `expand_trig_sec_reciprocal` to the derive corpus with expected
    strategy `expand trig`
  - changed the focused didactic regression to load the promoted corpus row
    instead of constructing an inline case
  - CLI confirms the route emits `[Reciprocal Trig Identity]` and requires
    `cos(x) != 0`
- guardrails:
  - `cargo fmt`
  - CLI probe: `derive sec(x), 1/cos(x)` reported `Strategy: expand trig` and
    `[Reciprocal Trig Identity]`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_secant_reciprocal_expansion_uses_direct_identity_language -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=349`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `expand trig=87`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=433`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`
  - `make engine-fast`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=349 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `433 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as a low-risk coverage promotion: no engine runtime code changed,
    but a real target-form expansion path is now measured by the primary derive
    guardrail instead of only by an inline didactic probe

## 2026-04-29 - Auto-improvement cycle: hyperbolic tanh pythagorean reverse derive promotion

- candidate:
  - promote `derive 1/cosh(x)^2, 1 - tanh(x)^2` as the missing expansion
    orientation for the hyperbolic tanh Pythagorean identity
- files_changed:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: `derive 1/cosh(x)^2, 1 - tanh(x)^2` reports
    `Strategy: rewrite hyperbolics`, the derive corpus expects `rewrite
    hyperbolics`, generic simplify pressure stays zero, and the derive
    didactic audit stays unflagged
  - `primary_dimension`: derive contract coverage for the inverse/expansive
    orientation of the hyperbolic tanh Pythagorean identity
  - `secondary_dimension`: engine-to-derive bridge coverage over an existing
    equivalence and target-aware hyperbolic transition provider
  - `hypothesis`: the engine and derive route already handled the transition,
    and the forward contraction row already existed, but the reverse direction
    was not measured by `derive_pairs.csv`; promoting one representative turns
    an existing capability into a durable guardrail without adding runtime code
  - `relevant_lanes`: CLI derive probe, existing target-aware unit coverage,
    derive generic-simplify guard, release derive contract, focused derive
    didactic regression, release derive didactic audit, `make engine-fast`,
    `make engine-scorecard`
  - `promotion_target`: `derive_pairs.csv`
  - `derive_bridge_check`: promoted only `1/cosh(x)^2 -> 1 - tanh(x)^2`; the
    existing forward `1 - tanh(x)^2 -> 1/cosh(x)^2` row already covers the
    contraction orientation
  - `engine_feedback_check`: no missing engine runtime capability was found;
    this is corpus coverage over an already-supported route
  - `retain_if`: the new row derives without generic `simplify`, audit flags
    stay at zero, and embedded/simplify guardrails remain green
  - `reject_if`: the row needs a generic strategy expectation, the didactic
    title becomes opaque or padded, or global scorecard guardrails regress
- structural_axis:
  - inverse/expansive orientation for a hyperbolic Pythagorean identity
- why_this_is_not_a_duplicate:
  - the corpus already covered `1 - tanh(x)^2 -> 1/cosh(x)^2`, but not the
    reverse target-form expansion `1/cosh(x)^2 -> 1 - tanh(x)^2`
- discovery_or_promotion:
  - promotion from an already-supported identity-pair route and existing
    target-aware unit coverage
- if_promoted_why_minimal_representative:
  - one direct reverse row is the smallest durable case; passthrough and
    negated/contextual variants are deferred until they expose separate path or
    domain value
- local_result:
  - added `hyperbolic_tanh_pythagorean_reverse` to the derive corpus with
    expected strategy `rewrite hyperbolics`
  - extended the focused direct hyperbolic didactic audit regression to cover
    the promoted corpus row and visible Spanish title
  - CLI confirms the route emits `[Hyperbolic Pythagorean Identity]` and the
    local change `1 / cosh(x)^2 -> 1 - tanh(x)^2`
- guardrails:
  - `cargo fmt`
  - CLI probe: `derive 1/cosh(x)^2, 1 - tanh(x)^2` reported `Strategy:
    rewrite hyperbolics` and `[Hyperbolic Pythagorean Identity]`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_direct_hyperbolic_identity_rules_use_visible_titles_without_padding -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=350`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `rewrite hyperbolics=30`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=434`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=459`
  - `make engine-fast`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=350 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `434 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as a low-risk derive coverage promotion: no runtime engine code
    changed, and the reverse hyperbolic Pythagorean transition is now measured
    by the primary derive contract

## 2026-04-29 - Auto-improvement cycle: hyperbolic tanh angle-sum contraction derive promotion

- candidate:
  - promote `derive (tanh(x)+tanh(y))/(1+tanh(x)*tanh(y)), tanh(x+y)` as the
    missing contraction orientation for the hyperbolic tanh angle-sum identity
- files_changed:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: `derive (tanh(x)+tanh(y))/(1+tanh(x)*tanh(y)),
    tanh(x+y)` reports `Strategy: rewrite hyperbolics`, the derive corpus
    expects `rewrite hyperbolics`, generic simplify pressure stays zero, and
    the derive didactic audit stays unflagged
  - `primary_dimension`: derive contract coverage for target-form contraction
    of the hyperbolic tanh angle-sum identity
  - `secondary_dimension`: engine-to-derive symmetry with the already-covered
    expansion orientation and with the analogous trig tangent angle-sum rows
  - `hypothesis`: the target-aware route and didactic substep already existed,
    but only the expansion `tanh(x+y) -> ...` was measured by `derive_pairs.csv`;
    promoting one contraction representative increases bridgeability coverage
    without adding runtime code or search
  - `relevant_lanes`: CLI derive probe, existing target-aware unit coverage,
    focused derive didactic regression, derive generic-simplify guard, release
    derive contract, release derive didactic audit, `make engine-fast`, `make
    engine-scorecard`
  - `promotion_target`: `derive_pairs.csv`
  - `derive_bridge_check`: promoted only the sum contraction
    `(tanh(x)+tanh(y))/(1+tanh(x)*tanh(y)) -> tanh(x+y)`; the difference,
    negated, and passthrough forms are deferred as same-family variants
  - `engine_feedback_check`: no missing reusable engine capability was found;
    this was a corpus/guardrail promotion over a stable target-aware transition
  - `retain_if`: `derive_contract` increases by one under `rewrite
    hyperbolics`, `generic_simplify_expected` remains `0`, audit flags remain
    `0`, and embedded/simplify guardrails remain green
  - `reject_if`: the row falls through a generic strategy, loses the specific
    hyperbolic angle-sum substep, or changes global guardrail pass/fail/runtime
    materially
- structural_axis:
  - inverse/contractive orientation for hyperbolic tanh angle-sum identities
- why_this_is_not_a_duplicate:
  - the corpus already covered the expansion `tanh(x+y) ->
    (tanh(x)+tanh(y))/(1+tanh(x)*tanh(y))`, but not the reverse target-form
    contraction toward `tanh(x+y)`
- discovery_or_promotion:
  - promotion from an already-supported CLI route and existing target-aware unit
    coverage
- if_promoted_why_minimal_representative:
  - one direct sum row is the smallest durable representative; the `x-y`
    difference and wrapper variants are intentionally deferred until they add
    distinct path, domain, or didactic value
- local_result:
  - added `contract_hyperbolic_tanh_sum` to the derive corpus with expected
    strategy `rewrite hyperbolics`
  - extended the focused hyperbolic angle-sum didactic audit regression to
    cover the promoted contraction row and the formula substep
  - CLI confirms the route emits `[Hyperbolic Angle Sum/Difference Identity]`
    and changes the quotient directly to `tanh(x + y)`
- guardrails:
  - `cargo fmt`
  - CLI probe: `derive (tanh(x)+tanh(y))/(1+tanh(x)*tanh(y)), tanh(x+y)`
    reported `Strategy: rewrite hyperbolics` and `[Hyperbolic Angle
    Sum/Difference Identity]`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_hyperbolic_angle_sum_diff_explains_direct_identities -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=351`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `rewrite hyperbolics=31`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=435`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=460`
  - `make engine-fast`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=351 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `435 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as a conservative derive bridge promotion: no runtime engine code
    changed, and the hyperbolic tanh angle-sum contraction is now measured by
    the primary derive contract with a specific didactic formula substep

## 2026-04-29 - Auto-improvement cycle: exponential sum expansion derive promotion

- candidate:
  - promote `derive exp(x+y), exp(x)*exp(y)` and make the exponential
    sum/difference didactic substep work in the expansion orientation
- files_changed:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [focused_rule_substeps.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: `derive exp(x+y), exp(x)*exp(y)` reports `Strategy:
    rewrite exponentials`, the derive corpus expects `rewrite exponentials`,
    web/audit output includes a concrete expansion formula substep, generic
    simplify pressure stays zero, and the scorecard remains green
  - `primary_dimension`: derive contract coverage for the expansion orientation
    of the exponential sum law
  - `secondary_dimension`: didactic path quality for an existing target-aware
    transition whose web substep generator only handled the contraction side
  - `hypothesis`: the engine and derive provider already know
    `exp(A+B) -> exp(A)exp(B)`, but the contract measured only the reverse
    contraction and the didactic substep generator looked only at the
    post-step `exp(A+B)` form; promoting one row plus making the substep
    orientation-aware converts an existing capability into an explicit,
    teachable guardrail
  - `relevant_lanes`: CLI derive probe, existing target-aware unit coverage,
    focused derive didactic regression, derive generic-simplify guard, release
    derive contract, release derive didactic audit, `make engine-fast`, `make
    engine-scorecard`
  - `promotion_target`: `derive_pairs.csv` plus the exponential
    sum/difference didactic substep generator
  - `derive_bridge_check`: promoted only `exp(x+y) -> exp(x)*exp(y)` because it
    adds target-form expansion coverage and didactic pressure; `exp(x-y)`,
    passthrough, and longer sums remain same-family variants
  - `engine_feedback_check`: no missing reusable engine runtime capability was
    found; the retained fix is coverage and didactic trace quality over an
    existing target-aware transition
  - `retain_if`: `derive_contract` increases by one under `rewrite
    exponentials`, `generic_simplify_expected` remains `0`, didactic audit
    flags remain `0`, and embedded/simplify guardrails remain green
  - `reject_if`: the row falls through a generic strategy, the expansion
    orientation has no concrete web substep, or global guardrails regress
- structural_axis:
  - expansion orientation for exponential sum/difference laws
- why_this_is_not_a_duplicate:
  - the corpus already covered `exp(x)*exp(y) -> exp(x+y)`, but not the target
    expansion `exp(x+y) -> exp(x)*exp(y)`; the prior substep generator also
    only recognized the contraction output shape
- discovery_or_promotion:
  - promotion from a stable CLI route and existing target-aware unit coverage,
    with a small didactic generator correction discovered during the probe
- if_promoted_why_minimal_representative:
  - one direct sum row is the smallest durable case; the difference,
    passthrough, and multi-factor variants are deferred until they expose
    separate path, domain, or didactic value
- local_result:
  - added `expand_exponential_sum` to the derive corpus with expected strategy
    `rewrite exponentials`
  - made `generate_exponential_sum_diff_identity_substeps` inspect the
    pre-step expression for expansion routes and emit reverse formulas such as
    `Usar e^(A+B) = e^A · e^B`
  - extended the focused exponential didactic regression to cover the expansion
    row
- guardrails:
  - `cargo fmt`
  - CLI probe: `derive exp(x+y), exp(x)*exp(y)` reported `Strategy: rewrite
    exponentials` and `[Exponential Sum/Difference Identity]`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_basic_exponential_laws_show_concrete_identities -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=352`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `rewrite exponentials=10`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=436`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=461`
  - `make engine-fast`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=352 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `436 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as a coverage plus didactic-quality bridge: no engine runtime code
    changed, and an existing exponential expansion route is now measured and
    explained in the derive audit path

## 2026-04-29 - Auto-improvement cycle: exponential power expansion derive promotion

- candidate:
  - promote `derive exp(3*x), exp(x)^3` and make the exponential power
    didactic substep work in the expansion orientation
- files_changed:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [focused_rule_substeps.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: `derive exp(3*x), exp(x)^3` reports `Strategy:
    rewrite exponentials`, the derive corpus expects `rewrite exponentials`,
    web/audit output includes the expansion formula substep, generic simplify
    pressure stays zero, and the scorecard remains green
  - `primary_dimension`: derive contract coverage for the expansion orientation
    of the exponential power law
  - `secondary_dimension`: didactic path quality for an existing target-aware
    transition whose substep generator only handled the contraction side
  - `hypothesis`: the provider already knows `exp(n*u) -> exp(u)^n`, but the
    contract measured only `exp(u)^n -> exp(n*u)` and the didactic helper only
    looked at the post-step contracted shape; one row plus an orientation-aware
    substep turns the existing route into a durable bridge without runtime code
  - `relevant_lanes`: CLI derive probe, existing target-aware unit coverage,
    focused derive didactic regression, derive generic-simplify guard, release
    derive contract, release derive didactic audit, `make engine-fast`, `make
    engine-scorecard`
  - `promotion_target`: `derive_pairs.csv` plus the exponential power didactic
    substep generator
  - `derive_bridge_check`: promoted only `exp(3*x) -> exp(x)^3`; `2*x`,
    symbolic powers, and passthrough variants are deferred as same-family cases
  - `engine_feedback_check`: no missing engine runtime capability was found;
    the retained value is coverage and didactic trace quality over an existing
    target-aware transition
  - `retain_if`: `derive_contract` increases by one under `rewrite
    exponentials`, `generic_simplify_expected` remains `0`, didactic audit
    flags remain `0`, and embedded/simplify guardrails remain green
  - `reject_if`: the promoted row loses its concrete web substep, falls through
    a generic strategy, or changes global guardrail pass/fail/runtime
    materially
- structural_axis:
  - expansion orientation for exponential power laws
- why_this_is_not_a_duplicate:
  - the corpus already covered `exp(x)^3 -> exp(3*x)`, but not the target
    expansion `exp(3*x) -> exp(x)^3`; the previous exponential sum cycle did
    not exercise power-law orientation
- discovery_or_promotion:
  - promotion from a stable CLI route and existing target-aware unit coverage,
    with a small didactic generator correction discovered during the probe
- if_promoted_why_minimal_representative:
  - `3*x` is the smallest durable representative aligned with the existing
    contraction row and unit test; `2*x`, symbolic exponents, and wrappers are
    deferred until they expose distinct path, domain, or didactic value
- local_result:
  - added `expand_exponential_power` to the derive corpus with expected
    strategy `rewrite exponentials`
  - made `generate_exponential_power_identity_substeps` inspect the pre-step
    expression for expansion routes and emit `Usar e^(n·A) = (e^A)^n`
  - extended the focused exponential didactic regression to cover the expansion
    row
- guardrails:
  - `cargo fmt`
  - CLI probe: `derive exp(3*x), exp(x)^3` reported `Strategy: rewrite
    exponentials` and `[Exponential Power Identity]`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_basic_exponential_laws_show_concrete_identities -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=353`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `rewrite exponentials=11`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=437`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=462`
  - `make engine-fast`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=353 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `437 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as coverage plus didactic-quality work: no engine runtime code
    changed, and an existing exponential power expansion route is now measured
    and explained by the derive audit path

## 2026-04-29 - Auto-improvement cycle: exponential reciprocal expansion derive promotion

- candidate:
  - promote `derive exp(-x), 1/exp(x)` and make the reciprocal exponential
    didactic substep work in the expansion orientation
- files_changed:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [focused_rule_substeps.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: `derive exp(-x), 1/exp(x)` reports `Strategy:
    rewrite exponentials`, the derive corpus expects `rewrite exponentials`,
    web/audit output includes the expansion formula substep, generic simplify
    pressure stays zero, and the scorecard remains green
  - `primary_dimension`: derive contract coverage for the expansion orientation
    of the exponential reciprocal identity
  - `secondary_dimension`: didactic path quality for a target-aware transition
    whose substep generator only handled the contraction side
  - `hypothesis`: the provider already knows `exp(-u) -> 1/exp(u)`, but the
    contract measured only `1/exp(u) -> exp(-u)` and the didactic helper only
    looked at the post-step contracted shape; one row plus an orientation-aware
    substep turns the existing route into a durable bridge without runtime code
  - `relevant_lanes`: CLI derive probe, existing target-aware unit coverage,
    focused derive didactic regression, derive generic-simplify guard, release
    derive contract, release derive didactic audit, `make engine-fast`, `make
    engine-scorecard`
  - `promotion_target`: `derive_pairs.csv` plus the exponential reciprocal
    didactic substep generator
  - `derive_bridge_check`: promoted only `exp(-x) -> 1/exp(x)`; signed outer
    variants, passthroughs, and wrappers are deferred as same-family cases
  - `engine_feedback_check`: no missing engine runtime capability was found;
    the retained value is coverage and didactic trace quality over an existing
    target-aware transition
  - `retain_if`: `derive_contract` increases by one under `rewrite
    exponentials`, `generic_simplify_expected` remains `0`, didactic audit
    flags remain `0`, and embedded/simplify guardrails remain green
  - `reject_if`: the promoted row loses its concrete web substep, falls through
    a generic strategy, or changes global guardrail pass/fail/runtime
    materially
- structural_axis:
  - expansion orientation for exponential reciprocal identities
- why_this_is_not_a_duplicate:
  - the corpus already covered `1/exp(x) -> exp(-x)`, but not the target
    expansion `exp(-x) -> 1/exp(x)`; the previous exponential power cycle did
    not exercise reciprocal/sign orientation
- discovery_or_promotion:
  - promotion from a stable CLI route and existing target-aware unit coverage,
    with a small didactic generator correction discovered during the probe
- if_promoted_why_minimal_representative:
  - `x` is the smallest durable representative aligned with the existing
    contraction row and unit test; signed outer variants, wrappers, and
    passthroughs are deferred until they expose distinct path, domain, or
    didactic value
- local_result:
  - added `expand_exponential_reciprocal` to the derive corpus with expected
    strategy `rewrite exponentials`
  - made `generate_exponential_reciprocal_identity_substeps` inspect the
    pre-step expression for expansion routes and emit `Usar e^(-A) = 1/e^A`
  - extended the focused exponential didactic regression to cover the expansion
    row
- guardrails:
  - `cargo fmt`
  - CLI probe: `derive exp(-x), 1/exp(x)` reported `Strategy: rewrite
    exponentials` and `[Exponential Reciprocal Identity]`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_basic_exponential_laws_show_concrete_identities -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=354`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `rewrite exponentials=12`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=438`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=463`
  - `make engine-fast`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=354 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `438 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as coverage plus didactic-quality work: no engine runtime code
    changed, and an existing exponential reciprocal expansion route is now
    measured and explained by the derive audit path

## 2026-04-29 - Auto-improvement cycle: hyperbolic tanh difference contraction derive promotion

- candidate:
  - promote `derive (tanh(x)-tanh(y))/(1-tanh(x)*tanh(y)), tanh(x-y)` as the
    missing contraction orientation for the hyperbolic tanh difference identity
- files_changed:
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: the new derive row reports `Strategy: rewrite
    hyperbolics`, the corpus expects `rewrite hyperbolics`, the web/audit
    output keeps the `tanh(A-B)` identity substep, generic simplify pressure
    stays zero, and the scorecard remains green
  - `primary_dimension`: derive contract coverage for the contraction
    orientation of the hyperbolic tanh difference identity
  - `secondary_dimension`: didactic pressure for sign/orientation robustness in
    hyperbolic angle sum/difference rewrites
  - `hypothesis`: the target-aware provider already knows how to contract the
    difference quotient, but the corpus measured only tanh-sum contraction and
    tanh-difference expansion; one row exposes the existing bridge without
    runtime code
  - `relevant_lanes`: CLI derive probe, focused derive didactic regression,
    derive generic-simplify guard, release derive contract, release derive
    didactic audit, `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: `derive_pairs.csv` plus the focused hyperbolic
    didactic identity expectation
  - `derive_bridge_check`: promoted only the minimal tanh-difference
    contraction; negated variants, wrappers, passthroughs, and triple-angle
    contraction are deferred as separate dimensions
  - `engine_feedback_check`: no reusable engine runtime gap was found; the
    retained value is measuring and auditing an existing target-aware route
  - `retain_if`: `derive_contract` increases by one under `rewrite
    hyperbolics`, `generic_simplify_expected` remains `0`, didactic audit flags
    remain `0`, and embedded/simplify guardrails remain green
  - `reject_if`: the row falls through a generic strategy, loses the visible
    tanh-difference identity substep, or materially changes global guardrail
    pass/fail/runtime
- structural_axis:
  - contractive orientation plus negative sign path for `tanh(A-B)`
- why_this_is_not_a_duplicate:
  - existing rows covered `tanh(A+B)` in both directions and `tanh(A-B)` only
    in expansion; the missing row exercises the quotient-to-compact target for
    the difference case
- discovery_or_promotion:
  - promotion from a stable CLI route and existing didactic substep support
- if_promoted_why_minimal_representative:
  - `x,y` is the smallest representative that preserves the negative numerator
    and denominator sign; hotter wrappers or negated variants do not add a new
    retained dimension yet
- local_result:
  - added `contract_hyperbolic_tanh_difference` to the derive corpus with
    expected strategy `rewrite hyperbolics`
  - extended the focused hyperbolic angle sum/difference didactic regression
    to assert the `tanh(A-B)` identity remains visible for the contraction row
- guardrails:
  - `cargo fmt`
  - CLI probe: `derive (tanh(x)-tanh(y))/(1-tanh(x)*tanh(y)), tanh(x-y)`
    reported `Strategy: rewrite hyperbolics` and `[Hyperbolic Angle
    Sum/Difference Identity]`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_hyperbolic_angle_sum_diff_explains_direct_identities -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=355`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `rewrite hyperbolics=32`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=439`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=464`
  - `make engine-fast`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=355 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `439 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as coverage plus didactic-quality pressure: no engine runtime code
    changed, and an existing hyperbolic tanh difference contraction route is
    now measured and kept explainable

## 2026-04-29 - Auto-improvement cycle: log change-of-base chain expansion derive promotion

- candidate:
  - promote `derive log(b,c), log(b,a)*log(a,c)` from a generic simplify
    fallback to the explicit `expand_log` change-of-base route
- files_changed:
  - [logs.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/logs.rs)
  - [derive_command.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs)
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [visible_rule_names.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/visible_rule_names.rs)
  - [focused_rule_substeps.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: the CLI route changes from `Strategy: simplify` to
    `Strategy: expand_log`, the primary derive corpus expects `expand_log`,
    `generic_simplify_expected` remains `0`, didactic output keeps
    `Expandir cambio de base` without redundant substeps, and scorecard
    guardrails remain green
  - `primary_dimension`: derive bridgeability for logarithmic
    change-of-base chain expansion
  - `secondary_dimension`: remove a generic fallback for an identity the
    engine could already prove semantically
  - `hypothesis`: the simplifier/runtime already proves
    `log_b(c) = log_b(a) * log_a(c)`, but `derive` only had target-aware
    change-of-base support for the quotient form; recognizing the two-factor
    chain gives a specific and explainable route
  - `relevant_lanes`: CLI derive probe, log target-aware unit regression,
    direct derive unit regression, focused derive didactic regression, derive
    generic-simplify guard, release derive contract, release derive didactic
    audit, `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: log target-aware provider, direct derive stage, primary
    `derive_pairs.csv`, and didactic visible-rule preservation
  - `derive_bridge_check`: promoted only the two-factor chain
    `log(base, mid) * log(mid, arg)`; longer chains, passthrough contexts, and
    wrapped products remain separate dimensions
  - `engine_feedback_check`: no simplification runtime gap was found; the
    retained change is a `derive` classification/routing improvement over an
    existing semantic capability
  - `retain_if`: `derive_contract` increases by one under `expand_log`,
    `generic_simplify_expected` remains `0`, didactic audit flags remain `0`,
    and embedded/simplify guardrails remain green
  - `reject_if`: the row still falls through `Strategy: simplify`, requires a
    broad search, breaks direct quotient change-of-base substeps, or changes
    global guardrail pass/fail/runtime materially
- structural_axis:
  - expansion from a base-log call to a two-link change-of-base product chain
- why_this_is_not_a_duplicate:
  - existing corpus rows covered direct quotient expansion and product-chain
    contraction; the missing row exercised the opposite chain-expansion
    orientation
- discovery_or_promotion:
  - promotion after a reproducible CLI fallback: before the change,
    `derive log(b,c), log(b,a)*log(a,c)` succeeded only as `Strategy: simplify`
- if_promoted_why_minimal_representative:
  - `b,a,c` is the smallest symbolic representative that preserves distinct
    base, intermediate base, and argument without adding wrappers or
    passthrough noise
- local_result:
  - added `BaseLogToChain` to the log change-of-base target-aware rewrite kind
  - routed `BaseLogToChain` through the direct `expand_log` stage
  - added `expand_log_change_of_base_chain` to the primary derive corpus with
    expected strategy `expand_log`
  - mapped the chain expansion to visible rule `Expandir cambio de base` while
    preserving no-substep behavior for this already-direct didactic step
- guardrails:
  - `cargo fmt`
  - `cargo test -q -p cas_solver expands_general_base_log_to_change_of_base_chain -- --nocapture`
  - `cargo test -q -p cas_solver direct_derive_expands_change_of_base_chain_without_planner -- --nocapture`
  - CLI probe: `derive log(b,c), log(b,a)*log(a,c)` reported `Strategy:
    expand_log` and `[Change of Base]`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_log_change_of_base_cases_stay_direct -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_direct_log_change_of_base_cases_expose_components -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=356`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `expand_log=16`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=439`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=464`
  - `make engine-fast`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=356 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `439 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as a `derive` routing and didactic-quality improvement: no
    simplifier runtime code changed, and a previously generic logarithmic
    identity now has a specific, contracted guardrail in the primary corpus

## 2026-04-29 - Auto-improvement cycle: reverse nested-fraction derive promotion

- candidate:
  - promote `derive z/(x*z+y), 1/(x + y/z)` from a generic simplify fallback
    to the explicit `nested fraction` route
- files_changed:
  - [fractions.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/fractions.rs)
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [analysis_command_eval_tests.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/analysis_command_eval_tests.rs)
  - [focused_rule_substeps.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: the CLI route changes from `Strategy: simplify` to
    `Strategy: nested fraction`, the primary derive corpus expects `nested
    fraction`, `generic_simplify_expected` remains `0`, reverse nested-fraction
    didactic output keeps a concrete factor-extraction substep, and scorecard
    guardrails remain green
  - `primary_dimension`: derive bridgeability for reverse nested-fraction
    targets
  - `secondary_dimension`: didactic quality for reconstructing a nested
    denominator/numerator without a generic simplify strategy label
  - `hypothesis`: the runtime already proves and renders the reverse form, but
    the target-aware nested-fraction provider only checked source-to-simplified
    direction; accepting the target when simplifying it returns the source
    creates a bounded and reusable reverse route
  - `relevant_lanes`: CLI derive probe, nested-fraction target-aware unit,
    derive command nested-fraction route tests, focused derive didactic
    regression, derive generic-simplify guard, release derive contract, release
    derive didactic audit, `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: nested-fraction target-aware provider and primary
    `derive_pairs.csv`
  - `derive_bridge_check`: promoted only the minimal reverse case; broader
    general and compound reverse nested-fraction cases remain covered by
    focused tests and didactic audit, not inflated into the primary corpus
  - `engine_feedback_check`: no simplification runtime gap was found; this is
    a strategy/classification gap in `derive` over an existing engine
    capability
  - `retain_if`: `derive_contract` increases by one under `nested fraction`,
    `generic_simplify_expected` remains `0`, didactic audit flags remain `0`,
    and embedded/simplify guardrails remain green
  - `reject_if`: the row still falls through `Strategy: simplify`, direct
    nested-fraction orientation regresses, reverse didactic substeps become
    generic, or global guardrails change materially
- structural_axis:
  - reverse orientation from a flat factored fraction into a denominator or
    numerator that contains an inner fraction
- why_this_is_not_a_duplicate:
  - the primary corpus covered direct nested-fraction simplification; generated
    audit cases exposed the reverse orientation but only through a generic
    strategy
- discovery_or_promotion:
  - promotion from generated didactic audit discovery plus reproducible CLI
    fallback
- if_promoted_why_minimal_representative:
  - `z/(x*z+y)` is the smallest symbolic representative that reconstructs
    `1/(x + y/z)` by factoring a common denominator out of the flat
    denominator
- local_result:
  - split nested-fraction target matching into forward and reverse checks
  - added a reverse check that accepts the target only when the target's real
    nested-fraction simplification matches the source
  - promoted `nested_fraction_one_over_sum_with_fraction_reverse` to the
    primary derive corpus with expected strategy `nested fraction`
  - updated command-line route tests to expect `nested fraction` for reverse
    structural nested-fraction targets
  - preserved reverse nested-fraction web substeps for `Simplify Nested
    Fraction` steps
- guardrails:
  - `cargo fmt`
  - CLI probe: `derive z/(x*z+y), 1/(x + y/z)` reported `Strategy: nested
    fraction`
  - `cargo test -q -p cas_solver rewrites_nested_fraction_targets_aware -- --nocapture`
  - `cargo test -q -p cas_solver evaluate_derive_command_lines_reaches_tabulated_reverse_nested_fraction_targets -- --nocapture`
  - `cargo test -q -p cas_solver evaluate_derive_command_lines_reaches_tabulated_reverse_structural_nested_fraction_targets -- --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_reverse_structural_nested_fraction_cases_keep_trace_direct -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_structural_nested_fraction_cases_keep_single_denominator_sum_substep -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=357`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `nested fraction=5`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=439`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=464`
  - `make engine-fast`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=357 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `439 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as `derive` routing plus didactic-quality coverage: no runtime
    simplifier code changed, and a known reverse nested-fraction capability now
    has an explicit strategy and primary corpus guardrail

## 2026-04-29 - Auto-improvement cycle: inverse hyperbolic log derive promotion

- candidate:
  - promote `derive atanh((x^2 - 1)/(x^2 + 1)), ln(x)` from
    equivalent-but-unsupported to the explicit `rewrite hyperbolics` route
- files_changed:
  - [hyperbolic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/hyperbolic.rs)
  - [target_classifier.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/target_classifier.rs)
  - [derive_command.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs)
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [focused_rule_substeps.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs)
  - [visible_rule_names.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/visible_rule_names.rs)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: the CLI route changes from
    equivalent-but-unsupported to `Strategy: rewrite hyperbolics`, the primary
    derive corpus gains one minimal row, `generic_simplify_expected` remains
    `0`, the didactic audit has `0` flags, and scorecard guardrails remain
    green
  - `primary_dimension`: derive bridgeability for inverse-hyperbolic-to-log
    identities
  - `secondary_dimension`: engine-to-derive reuse of an existing conditional
    simplification capability
  - `hypothesis`: `cas_math::hyperbolic_core_support` already exposes a
    bounded `atanh((u^2 - 1)/(u^2 + 1)) -> ln(u)` rewrite and the engine can
    prove the pair conditionally, but `derive` did not classify inverse
    hyperbolic calls as a hyperbolic target family or expose that core rewrite
    as a target-aware transition
  - `relevant_lanes`: CLI derive probe, hyperbolic target-aware unit, direct
    derive unit, focused derive didactic regression, derive generic-simplify
    guard, release derive contract, release derive didactic audit,
    `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: primary `derive_pairs.csv` minimal row plus
    family-local unit and didactic coverage
  - `derive_bridge_check`: promoted only the forward conditional simplification
    `atanh((x^2 - 1)/(x^2 + 1)) -> ln(x)`; reverse and generic atanh-log
    definition forms are deferred because branch/domain proof is not symmetric
    yet
  - `engine_feedback_check`: no runtime simplifier gap was found; the retained
    change is derive classifier/provider exposure over an existing reusable
    engine capability
  - `retain_if`: the route reports `rewrite hyperbolics`, reaches `ln(x)`,
    keeps `x > 0` visible, derived count increases by one,
    `generic_simplify_expected` remains `0`, and guardrails pass
  - `reject_if`: the row still reports unsupported or generic `simplify`,
    requires broad search, weakens branch/domain behavior, or regresses global
    guardrails
- structural_axis:
  - inverse hyperbolic rational square argument collapsing to a logarithm under
    a positive-argument regime
- why_this_is_not_a_duplicate:
  - existing derive coverage handled `sinh`/`cosh`/`tanh` identities and
    hyperbolic exponential bridges, but not this inverse-hyperbolic logarithmic
    reduction; the probe was unsupported rather than already categorized
- discovery_or_promotion:
  - promotion from a reproducible CLI unsupported-equivalent case backed by an
    existing engine helper
- if_promoted_why_minimal_representative:
  - one variable `x` with `(x^2 - 1)/(x^2 + 1)` is the smallest shape that
    exercises the rule and its `x > 0` condition without passthrough or
    wrappers
- local_result:
  - added `AtanhSquareRatioToLn` as a hyperbolic derive rewrite kind
  - classified inverse hyperbolic calls as part of the hyperbolic target family
    for derive routing
  - reused `try_rewrite_atanh_square_ratio_to_ln` in the target-aware
    hyperbolic provider
  - promoted `inverse_hyperbolic_atanh_square_ratio_log` into the primary
    derive corpus with expected strategy `rewrite hyperbolics`
  - added a visible rule name and a concrete web substep identifying the
    `(u^2 - 1)/(u^2 + 1)` argument pattern
- guardrails:
  - `cargo fmt`
  - CLI probe reported `Strategy: rewrite hyperbolics` and `Requires: x > 0`
  - `cargo test -q -p cas_solver target_aware_hyperbolic_rewrite_recognizes_atanh_square_ratio_log_identity -- --nocapture`
  - `cargo test -q -p cas_solver direct_derive_labels_atanh_square_ratio_log_without_generic_simplify -- --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_inverse_hyperbolic_log_identity_has_concrete_substeps -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=358`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `rewrite hyperbolics=33`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=440`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=465`
  - `make engine-fast`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=358 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `440 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as derive coverage and didactic-quality improvement: no runtime
    simplifier code changed, and a previously unsupported conditional
    hyperbolic/log identity now has a specific route, condition display, and
    primary corpus guardrail

## 2026-04-29 - Auto-improvement cycle: hyperbolic composition derive promotion

- candidate:
  - promote `derive sinh(asinh(x)), x` from generic `Strategy: simplify` to
    the explicit `rewrite hyperbolics` route
- files_changed:
  - [hyperbolic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/hyperbolic.rs)
  - [derive_command.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs)
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [focused_rule_substeps.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs)
  - [visible_rule_names.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/visible_rule_names.rs)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: `derive sinh(asinh(x)), x` reports `Strategy:
    rewrite hyperbolics`; the primary derive corpus gains one minimal row;
    `generic_simplify_expected` remains `0`; the didactic audit keeps `0`
    flags; `make engine-fast` and `make engine-scorecard` remain green
  - `primary_dimension`: derive bridgeability for direct hyperbolic
    function/inverse compositions
  - `secondary_dimension`: didactic quality by replacing a generic simplify
    fallback with a named hyperbolic composition transition
  - `hypothesis`: `cas_math::hyperbolic_core_support` already exposes
    `try_rewrite_hyperbolic_composition`, but the target-aware hyperbolic
    derive provider did not expose it as a `rewrite hyperbolics` transition
  - `relevant_lanes`: CLI derive probe, hyperbolic target-aware unit, direct
    derive unit, focused derive didactic regression, derive generic-simplify
    guard, release derive contract, release derive didactic audit,
    `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: primary `derive_pairs.csv` minimal row plus
    family-local unit coverage over all six direct compositions
  - `derive_bridge_check`: promoted only `sinh(asinh(x)) -> x` as the minimal
    corpus representative; the other five direct compositions remain in
    family-local tests to avoid near-duplicate corpus growth
  - `engine_feedback_check`: no runtime simplifier gap was found; this is a
    derive provider exposure gap over an existing reusable engine capability
  - `retain_if`: route reports `rewrite hyperbolics`, rule is `Hyperbolic
    Composition`, derived count increases by one, `generic_simplify_expected`
    remains `0`, didactic flags remain `0`, and guardrails pass
  - `reject_if`: the route still falls through generic `simplify`, needs broad
    search, emits opaque didactic output, or regresses global guardrails
- structural_axis:
  - direct composition of a hyperbolic function with its inverse, in both
    outer-direct and outer-inverse orientations
- why_this_is_not_a_duplicate:
  - existing hyperbolic derive coverage handled pythagorean, quotient,
    exponential, angle, parity, and inverse-log reductions; the basic
    function/inverse composition family was still reaching `derive` through a
    generic simplify fallback
- discovery_or_promotion:
  - promotion from a reproducible CLI generic-simplify case backed by an
    existing engine helper
- if_promoted_why_minimal_representative:
  - `sinh(asinh(x)) -> x` is the smallest composition that exercises the family
    without branch-heavy logarithmic definitions, passthrough, or wrappers
- local_result:
  - added `HyperbolicComposition` as a hyperbolic derive rewrite kind
  - reused `try_rewrite_hyperbolic_composition` in the target-aware hyperbolic
    provider
  - promoted `hyperbolic_composition_sinh_asinh` into the primary derive corpus
    with expected strategy `rewrite hyperbolics`
  - added a visible rule name and two focused substeps that state the inverse
    function identity and bind `u = x`
  - covered all six direct compositions in a family-local provider unit
- guardrails:
  - `cargo fmt`
  - CLI probe reported `Strategy: rewrite hyperbolics` and `[Hyperbolic
    Composition]`
  - `cargo test -q -p cas_solver target_aware_hyperbolic_rewrite_recognizes_compositions -- --nocapture`
  - `cargo test -q -p cas_solver direct_derive_labels_hyperbolic_composition_without_generic_simplify -- --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_hyperbolic_composition_has_concrete_substeps -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_simplify_log_representatives -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=359`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `rewrite hyperbolics=34`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=441`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=467`
  - `make engine-fast`: `simplify_add_small 435/435`, `contextual_strict_fast
    64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=359 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `441 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as a derive coverage and didactic-quality improvement: no runtime
    simplifier code changed, and a basic reusable hyperbolic engine capability
    now has a specific derive route and visible substeps

## 2026-04-29 - Auto-improvement cycle: hyperbolic special value derive promotion

- files_changed:
  - [hyperbolic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/hyperbolic.rs)
  - [derive_command.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs)
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [visible_rule_names.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/visible_rule_names.rs)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
  - [DERIVE_DIDACTIC_AUDIT.md](/Users/javiergimenezmoya/developer/math/docs/generated/DERIVE_DIDACTIC_AUDIT.md)
  - [DIDACTIC_STEP_QUALITY_AUDIT_REPORT.md](/Users/javiergimenezmoya/developer/math/docs/generated/DIDACTIC_STEP_QUALITY_AUDIT_REPORT.md)
  - [engine_improvement_scorecard.json](/Users/javiergimenezmoya/developer/math/docs/generated/engine_improvement_scorecard.json)
  - [engine_improvement_scorecard.md](/Users/javiergimenezmoya/developer/math/docs/generated/engine_improvement_scorecard.md)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: `derive sinh(0), 0` changes from `Strategy:
    simplify` to `Strategy: rewrite hyperbolics`; the primary derive corpus
    gains one minimal row; `generic_simplify_expected` remains `0`; the
    didactic audit keeps `0` flags; `make engine-fast` and
    `make engine-scorecard` remain green
  - `primary_dimension`: derive bridgeability for hyperbolic special values
  - `secondary_dimension`: engine-to-derive reuse of
    `try_eval_hyperbolic_special_value` without adding runtime simplifier rules
  - `hypothesis`: `cas_math::hyperbolic_core_support` already evaluates
    `sinh(0)`, `cosh(0)`, `tanh(0)`, `asinh(0)`, `atanh(0)`, and `acosh(1)`,
    but the target-aware derive provider did not expose the family as
    `rewrite hyperbolics`
  - `relevant_lanes`: CLI derive probe, hyperbolic target-aware unit, direct
    derive unit, focused derive didactic regression, derive generic-simplify
    guard, release derive contract, release derive didactic audit,
    `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: primary `derive_pairs.csv` minimal row plus
    family-local unit coverage over all six special values
  - `derive_bridge_check`: promoted only `sinh(0) -> 0` as the minimal corpus
    representative; the other five special values remain in family-local tests
    to avoid near-duplicate corpus growth
  - `engine_feedback_check`: no runtime simplifier gap was found; this is a
    derive provider exposure gap over an existing reusable engine capability
  - `retain_if`: route reports `rewrite hyperbolics`, rule is `Evaluate
    Hyperbolic Functions`, derived count increases by one,
    `generic_simplify_expected` remains `0`, didactic flags remain `0`, and
    guardrails pass
  - `reject_if`: the route still falls through generic `simplify`, needs broad
    search, emits opaque didactic output, or regresses global guardrails
- structural_axis:
  - evaluation of special values for direct and inverse hyperbolic functions
- why_this_is_not_a_duplicate:
  - existing hyperbolic derive coverage handled pythagorean, quotient,
    exponential, angle, parity, inverse-log, and direct composition reductions;
    basic special-value evaluation was still reaching `derive` through a
    generic simplify fallback
- discovery_or_promotion:
  - promotion from reproducible CLI generic-simplify probes backed by an
    existing engine helper
- if_promoted_why_minimal_representative:
  - `sinh(0) -> 0` is the smallest direct hyperbolic special value and avoids
    branch-heavy logarithmic definitions, wrappers, or domain assumptions
- local_result:
  - added `HyperbolicSpecialValue` as a hyperbolic derive rewrite kind
  - reused `try_eval_hyperbolic_special_value` in the target-aware hyperbolic
    provider
  - expanded direct derive hyperbolic function detection to inverse
    hyperbolics so `asinh`, `acosh`, and `atanh` special values use the same
    named strategy
  - promoted `hyperbolic_special_value_sinh_zero` into the primary derive
    corpus with expected strategy `rewrite hyperbolics`
  - added a visible didactic rule name and a focused didactic audit regression
    that accepts this direct value step without requiring artificial substeps
  - covered all six supported special values in family-local provider and
    direct derive tests
- guardrails:
  - `cargo fmt`
  - CLI probes reported `Strategy: rewrite hyperbolics` and `[Evaluate
    Hyperbolic Functions]` for `sinh(0)`, `asinh(0)`, and `acosh(1)`
  - `cargo test -q -p cas_solver target_aware_hyperbolic_rewrite_recognizes_special_values -- --nocapture`
  - `cargo test -q -p cas_solver direct_derive_labels_hyperbolic_special_values_without_generic_simplify -- --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_hyperbolic_special_value_is_direct_and_unflagged -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_simplify_log_representatives -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=360`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `rewrite hyperbolics=35`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=442`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=467`, `mean_step_count=1.06`
  - `make engine-fast`: `simplify_add_small 435/435`, `contextual_strict_fast
    64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=360 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `442 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as a derive coverage improvement: no runtime simplifier code
    changed, and a pre-existing hyperbolic engine capability now has a named
    target-aware derive route plus didactic visibility

## 2026-04-29 - Auto-improvement cycle: direct inverse-trig composition derive promotion

- files_changed:
  - [target_classifier.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/target_classifier.rs)
  - [derive_command.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs)
  - [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [focused_rule_substeps.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs)
  - [derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
  - [DERIVE_DIDACTIC_AUDIT.md](/Users/javiergimenezmoya/developer/math/docs/generated/DERIVE_DIDACTIC_AUDIT.md)
  - [DIDACTIC_STEP_QUALITY_AUDIT_REPORT.md](/Users/javiergimenezmoya/developer/math/docs/generated/DIDACTIC_STEP_QUALITY_AUDIT_REPORT.md)
  - [engine_improvement_scorecard.json](/Users/javiergimenezmoya/developer/math/docs/generated/engine_improvement_scorecard.json)
  - [engine_improvement_scorecard.md](/Users/javiergimenezmoya/developer/math/docs/generated/engine_improvement_scorecard.md)
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `success_condition`: `derive(sin(arcsin(x)), x)` changes from `Strategy:
    simplify` to `Strategy: rewrite inverse trigs`; the primary derive corpus
    gains one minimal row; `generic_simplify_expected` remains `0`; the
    didactic audit keeps `0` flags; `make engine-fast` and
    `make engine-scorecard` remain green
  - `primary_dimension`: derive bridgeability for direct inverse-trig
    compositions
  - `secondary_dimension`: didactic quality by replacing a generic simplify
    fallback with a named inverse-trig composition transition and visible
    substeps
  - `hypothesis`: `cas_math::inverse_trig_composition_support` already plans
    `sin(arcsin(x))` and `cos(arccos(x))`, but derive queried the planner only
    in strict mode, so symbolic direct compositions fell through to generic
    simplify even though the runtime simplifier already handled them
  - `relevant_lanes`: CLI derive probes, target classifier unit, direct derive
    unit, focused derive didactic regression, derive generic-simplify guard,
    release derive contract, release derive didactic audit, `make
    engine-fast`, `make engine-scorecard`
  - `promotion_target`: primary `derive_pairs.csv` minimal row plus
    family-local coverage for `sin(arcsin(x))`, `cos(arccos(x))`, and the
    existing `tan(arctan(x))` route
  - `derive_bridge_check`: promoted only `sin(arcsin(x)) -> x` as the minimal
    corpus representative; `cos(arccos(x)) -> x` remains in unit coverage to
    avoid near-duplicate corpus growth
  - `engine_feedback_check`: no runtime simplifier gap was found; this is a
    derive provider exposure gap over an existing reusable engine capability
  - `retain_if`: route reports `rewrite inverse trigs`, rule is `Inverse Trig
    Composition`, derived count increases by one, `generic_simplify_expected`
    remains `0`, didactic flags remain `0`, and guardrails pass
  - `reject_if`: the route still falls through generic `simplify`, changes
    branch/domain semantics, emits opaque didactic output, or regresses global
    guardrails
- structural_axis:
  - direct function/inverse composition for inverse trigonometric functions
    under symbolic targets
- why_this_is_not_a_duplicate:
  - existing inverse-trig derive rows handled arctangent reciprocal sums,
    `arcsin(x/sqrt(1+x^2)) -> arctan(x)`, right-triangle projections, and
    complement projections; the direct `sin(arcsin(x)) -> x` bridge was still
    reaching derive through a generic simplify fallback
- discovery_or_promotion:
  - promotion from reproducible CLI generic-simplify probes backed by an
    existing engine helper
- if_promoted_why_minimal_representative:
  - `sin(arcsin(x)) -> x` is the smallest direct inverse-trig composition,
    already present in the engine identity corpus, and exercises the symbolic
    generic-planner fallback without wrappers or extra algebra
- local_result:
  - derive inverse-trig composition planning now falls back from strict to
    generic mode for target-aware direct compositions, matching existing
    simplifier behavior without adding runtime rules
  - the target classifier now recognizes direct inverse-trig composition
    targets before the simplified fallback
  - promoted `inverse_trig_composition_sin_arcsin` into the primary derive
    corpus with expected strategy `rewrite inverse trigs`
  - added focused didactic substeps for direct inverse-trig compositions:
    the template inverse-function identity and the concrete `u = x` binding
  - covered `sin(arcsin(x))`, `cos(arccos(x))`, and `tan(arctan(x))` in the
    direct derive unit
- guardrails:
  - `cargo fmt`
  - CLI probes for `derive(sin(arcsin(x)), x)`,
    `derive(cos(arccos(x)), x)`, and `derive(tan(arctan(x)), x)` reported
    `Strategy: rewrite inverse trigs` and `[Inverse Trig Composition]`
  - `cargo test -q -p cas_solver classifies_tabulated_inverse_trig_rewritten_targets -- --nocapture`
  - `cargo test -q -p cas_solver direct_derive_rewrites_direct_inverse_trig_compositions_without_generic_simplify -- --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_direct_inverse_trig_composition_uses_inverse_language -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_simplify_log_representatives -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=361`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `rewrite inverse trigs=8`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=443`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=469`, `mean_step_count=1.06`
  - `make engine-fast`: `simplify_add_small 435/435`, `contextual_strict_fast
    64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=361 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `443 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as a derive coverage and didactic-quality improvement: no runtime
    simplifier code changed, and a stable inverse-trig engine capability now
    has a named target-aware derive route with explicit substeps

## 2026-04-29 - Auto-improvement cycle: trig special value derive promotion

- capture:
  - `investment_class`: coverage
  - `success_condition`: `derive(sin(0), 0)` changes from generic
    `Strategy: simplify` to `Strategy: rewrite trigs`; one minimal
    `derive_pairs.csv` row is promoted; `generic_simplify_expected` remains
    `0`; didactic audit stays at `0` flags; `make engine-fast` and
    `make engine-scorecard` pass
  - `primary_dimension`: derive bridgeability for direct and inverse
    trigonometric special values
  - `secondary_dimension`: engine-to-derive reuse of the existing exact trig
    value table without duplicating runtime rules
  - `hypothesis`: `cas_math::trig_core_identity_support` already evaluates
    direct and inverse trig special values, but derive did not expose that path
    target-aware, so simple values such as `sin(0)` and `asin(0)` fell through
    to generic simplify
  - `relevant_lanes`: CLI derive probes, target-aware trig unit, direct derive
    unit, focused didactic audit, generic simplify guard, release derive
    contract, release didactic audit, `make engine-fast`, `make
    engine-scorecard`
  - `promotion_target`: corpus row `trig_special_value_sin_zero`; local unit
    coverage for `sin(0)`, `cos(0)`, `tan(0)`, `asin(0)`, `acos(1)`, and
    `atan(0)`
  - `derive_bridge_check`: promoted because this replaces a magical generic
    simplify route with a named trig rewrite over an existing engine helper
  - `engine_feedback_check`: no runtime simplifier gap was found; this is a
    derive exposure gap and confirms the existing table is reusable
  - `retain_if`: route reports `rewrite trigs`, visible rule is
    `Evaluar valor trigonométrico especial`, derived count increases by one,
    `generic_simplify_expected` remains `0`, didactic flags remain `0`, and
    guardrails pass
  - `reject_if`: the route stays generic `simplify`, duplicates the trig table,
    changes branch/domain semantics, captures parity rewrites as special
    values, or regresses guardrails
- structural_axis:
  - exact special values for direct and inverse trigonometric functions
- why_this_is_not_a_duplicate:
  - existing inverse-trig rows covered compositions/projections and existing
    trig rows covered identities; basic trig value evaluation still reached
    derive through generic simplify
- discovery_or_promotion:
  - promotion from reproducible CLI generic-simplify probes backed by an
    existing `cas_math` helper
- rejected_local_candidate_detail:
  - the first implementation reused the legacy helper too broadly and captured
    `tan(-x) -> -tan(x)` as `Evaluate Trigonometric Functions`, regressing the
    existing parity case `expand_trig_negative_tangent_parity`; the retained
    implementation filters legacy odd/even parity rewrites and keeps them on
    their prior strategy
- if_promoted_why_minimal_representative:
  - `sin(0) -> 0` is the smallest stable no-domain-extra row and exercises the
    new route without adding near-duplicate corpus cases
- local_result:
  - added `TrigSpecialValue` as a derive trig rewrite kind and target-aware
    bridge to `try_rewrite_legacy_evaluate_trig_expr`
  - routed direct trig special values through `try_fast_direct_trig_derive`
    before generic simplify, while excluding legacy negative-parity rewrites
  - included inverse trig aliases in the direct trig gate so `asin(0)`,
    `acos(1)`, and `atan(0)` use the same visible derive path
  - added Spanish visible rule text and didactic audit allowance for the direct
    special-value step
  - promoted `trig_special_value_sin_zero` in `derive_pairs.csv`
- guardrails:
  - `cargo fmt`
  - CLI probes for `derive(sin(0), 0)`, `derive(asin(0), 0)`, and
    `derive(acos(1), 0)` reported `Strategy: rewrite trigs` and visible rule
    `Evaluar valor trigonométrico especial`
  - `cargo test -q -p cas_solver target_aware_trig_rewrite_recognizes_special_values -- --nocapture`
  - `cargo test -q -p cas_solver direct_derive_labels_trig_special_values_without_generic_simplify -- --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_trig_special_value_is_direct_and_unflagged -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_simplify_log_representatives -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=362`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `rewrite trigs=7`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=444`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=469`, `mean_step_count=1.06`
  - `make engine-fast`: `simplify_add_small 435/435`,
    `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=362 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `444 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as a derive coverage improvement: it exposes existing runtime trig
    value knowledge through a named, didactic, target-aware derive route while
    preserving existing parity and global guardrails

## 2026-04-29 - Auto-improvement cycle: reciprocal trig special value derive promotion

- candidate_capture:
  - `investment_class`: coverage
  - `success_condition`: `derive(sec(pi/4), sqrt(2))` changes from generic
    `simplify` to `rewrite trigs`; one minimal corpus row is promoted for
    reciprocal trig special values; `generic_simplify_expected` remains `0`;
    didactic audit remains at `0` flags; `make engine-fast` and `make
    engine-scorecard` pass
  - `primary_dimension`: derive bridgeability for exact reciprocal trig values
  - `secondary_dimension`: reuse of the modern trig evaluation table in
    `cas_math::trig_eval_table_support` instead of legacy-only derive lookup
  - `hypothesis`: the engine already knew exact `sec`/`csc`/`cot` table values,
    but target-aware derive only consulted the older direct/inverse trig helper,
    so reciprocal values reached the right result through a generic simplify
    path with a misleading reciprocal rule
  - `relevant_lanes`: CLI derive probes, trig table unit, target-aware trig
    unit, direct derive unit, focused didactic audit, derive contract, release
    didactic audit, `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: `reciprocal_trig_special_value_sec_pi_fourth`
    (`sec(pi/4) -> sqrt(2)`)
  - `derive_bridge_check`: promoted because this exposes existing engine table
    knowledge as a named, target-aware, teachable derive route
  - `engine_feedback_check`: no runtime math gap was found; the useful change is
    in the derive bridge consuming the broader table API
  - `retain_if`: route reports `rewrite trigs`, visible rule is
    `Evaluar valor trigonométrico especial`, derived count increases by one,
    `generic_simplify_expected` remains `0`, didactic flags remain `0`, and
    guardrails pass
  - `reject_if`: the change captures parity as table evaluation, duplicates
    trig tables, changes undefined-value semantics, or regresses any guardrail
- structural_axis:
  - exact reciprocal trigonometric table values (`sec`, `csc`, `cot`)
- why_this_is_not_a_duplicate:
  - direct/inverse trig special values were already promoted, but reciprocal
    trig functions were still missing from target-aware derive despite being
    present in the runtime table evaluator and identity corpus
- discovery_or_promotion:
  - promotion from reproducible CLI probes over existing runtime knowledge
- rejected_local_candidate_detail:
  - no mathematical candidate was rejected in this cycle; one mistyped Cargo
    invocation supplied two test filters and was rerun correctly. A parity
    guard was added explicitly so `tan(-x) -> -tan(x)` remains an `expand trig`
    parity rewrite rather than a special-value table rewrite
- if_promoted_why_minimal_representative:
  - `sec(pi/4) -> sqrt(2)` is the smallest nontrivial reciprocal-table case
    that exercises a radical result without adding near-duplicate variants
- local_result:
  - `trig_eval_table_support` now canonicalizes inverse aliases (`asin`,
    `acos`, `atan`) before table lookup and has unit coverage for inverse alias
    and reciprocal trig values
  - target-aware derive tries `try_rewrite_trig_eval_table_expr` for
    `TrigEvalRewriteKind::Table` before the legacy helper, while rejecting
    `NegativeParity`
  - direct derive and didactic audit coverage now include
    `sec(pi/4) -> sqrt(2)`, with nearby local tests for `csc(pi/6) -> 2` and
    `cot(pi/4) -> 1`
  - promoted one corpus row:
    `reciprocal_trig_special_value_sec_pi_fourth`
- guardrails:
  - `cargo fmt`
  - CLI probes for `derive(sec(pi/4), sqrt(2))` and
    `derive(csc(pi/6), 2)` reported `Strategy: rewrite trigs` and visible rule
    `Evaluar valor trigonométrico especial`
  - CLI parity probe `derive(tan(-x), -tan(x))` stayed on `Strategy: expand
    trig` with rule `Aplicar paridad trigonométrica`
  - `cargo test -q -p cas_math trig_eval_table_support -- --nocapture`
  - `cargo test -q -p cas_solver direct_derive_labels_trig_special_values_without_generic_simplify -- --nocapture`
  - `cargo test -q -p cas_solver target_aware_trig_special_value -- --nocapture`
  - `cargo test -q -p cas_solver target_aware_trig_rewrite_recognizes_special_values -- --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_trig_special_value_is_direct_and_unflagged -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_simplify_log_representatives -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=363`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `rewrite trigs=8`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=445`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=469`, `mean_step_count=1.06`
  - `make engine-fast`: `simplify_add_small 435/435`,
    `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=363 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `445 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as a derive coverage improvement: the cycle converts existing
    runtime reciprocal trig value knowledge into an explicit, didactic derive
    route, while preserving parity behavior and all engine guardrails

## 2026-04-29 - Auto-improvement cycle: inverse trig direct radical value promotion

- candidate_capture:
  - `investment_class`: coverage
  - `success_condition`: `derive(arctan(sqrt(3)), pi/3)` changes from generic
    `simplify` with a misleading visible rule to `rewrite trigs` with
    `Evaluar valor trigonométrico especial`; one minimal corpus row is
    promoted; `generic_simplify_expected` remains `0`; didactic audit remains
    at `0` flags; `make engine-fast` and `make engine-scorecard` pass
  - `primary_dimension`: derive bridgeability for exact inverse-trig inputs
    with direct radical form
  - `secondary_dimension`: reuse of the existing inverse trig table in
    `cas_math` by making special-input detection robust to parser surface form
  - `hypothesis`: `INVERSE_TRIG_TABLE` already contained
    `arctan(√3) = π/3`, but `detect_inverse_trig_input` only recognized `√3`
    through the `Pow` arm; when the input arrived as function-form `sqrt(3)`,
    derive missed the table bridge and fell back to generic simplify
  - `relevant_lanes`: CLI derive probe, inverse-trig input detection unit,
    trig table unit, target-aware trig unit, direct derive unit, focused
    didactic audit, derive contract, release didactic audit, `make
    engine-fast`, `make engine-scorecard`
  - `promotion_target`: `inverse_trig_special_value_arctan_sqrt_three`
    (`arctan(sqrt(3)) -> pi/3`)
  - `derive_bridge_check`: promoted because this exposes existing engine table
    knowledge as a named, target-aware, teachable derive route
  - `engine_feedback_check`: reusable helper gap in `cas_math`, not a
    planner-only miss and not a derive-only special case
  - `retain_if`: route reports `rewrite trigs`, visible rule is
    `Evaluar valor trigonométrico especial`, derived count increases by one,
    `generic_simplify_expected` remains `0`, didactic flags remain `0`, and
    guardrails pass
  - `reject_if`: the change duplicates inverse trig tables, patches only
    `derive`, changes branch/domain semantics, or regresses any guardrail
- structural_axis:
  - direct radical special inputs for inverse trigonometric exact values
- why_this_is_not_a_duplicate:
  - `arctan(1)`, `arcsin(1/2)`, `arccos(1/2)`,
    `arcsin(sqrt(3)/2)`, `arccos(sqrt(3)/2)`, and
    `arctan(1/sqrt(3))` already reached the table path; the direct
    `sqrt(3)` input was the missing parser-shape variant and is present in
    `identity_pairs.csv`
- discovery_or_promotion:
  - promotion from reproducible CLI generic-simplify probe backed by an
    existing identity-pair row and an existing inverse trig table entry
- rejected_local_candidate_detail:
  - no mathematical candidate was rejected in this cycle
- if_promoted_why_minimal_representative:
  - `arctan(sqrt(3)) -> pi/3` is the smallest stable direct-radical inverse
    trig value that was still using generic simplify; nearby inverse values
    were already covered and were not promoted as duplicates
- local_result:
  - `detect_inverse_trig_input` now falls back to
    `extract_numeric_sqrt_radicand` for non-number/div/mul shapes, so
    function-form `sqrt(3)` and power-form `3^(1/2)` both classify as `Sqrt3`
  - `trig_eval_table_support` has unit coverage for `arctan(sqrt(3))`
  - target-aware trig and direct derive tests include the direct radical case
  - promoted one corpus row:
    `inverse_trig_special_value_arctan_sqrt_three`
- guardrails:
  - `cargo fmt`
  - CLI probe for `derive(arctan(sqrt(3)), pi/3)` reported `Strategy: rewrite
    trigs` and visible rule `Evaluar valor trigonométrico especial`
  - `cargo test -q -p cas_math trig_value_detection_support::tests::detect_inverse_trig_input_forms -- --exact --nocapture`
  - `cargo test -q -p cas_math trig_eval_table_support -- --nocapture`
  - `cargo test -q -p cas_solver target_aware_trig_rewrite_recognizes_special_values -- --nocapture`
  - `cargo test -q -p cas_solver direct_derive_labels_trig_special_values_without_generic_simplify -- --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_trig_special_value_is_direct_and_unflagged -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_simplify_log_representatives -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=364`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `rewrite trigs=9`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=446`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=469`, `mean_step_count=1.06`
  - `make engine-fast`: `simplify_add_small 435/435`,
    `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=364 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `446 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as a targeted coverage and bridgeability improvement: an existing
    exact inverse-trig table value now reaches `derive` through the correct
    visible family route without adding runtime risk or duplicate table logic

## 2026-04-29 - Auto-improvement cycle: derive negative quotient target matching

- candidate_capture:
  - `investment_class`: robustness
  - `success_condition`: `derive(cos(2*pi/3), -1/2)` and nearby negative
    exact quotient targets route through `rewrite trigs` instead of falling
    back to generic `simplify`; one minimal corpus row is promoted;
    `generic_simplify_expected` remains `0`; didactic audit remains at `0`
    flags; `make engine-fast` and `make engine-scorecard` pass
  - `primary_dimension`: derive target matching for syntactically different
    negative quotient forms
  - `secondary_dimension`: exact trigonometric value bridgeability for
    second-quadrant cosine values already known by the runtime evaluator
  - `hypothesis`: the runtime evaluator can produce values such as
    `-(sqrt(2)/2)`, but user/corpus targets like `-sqrt(2)/2` parse as a
    negative numerator quotient; `strong_target_match` strips global signs from
    negations and products, but not from quotients, so target-aware derive
    misses an otherwise valid exact-value rewrite
  - `relevant_lanes`: CLI derive probes, matcher unit, target-aware trig unit,
    direct derive unit, focused didactic audit, derive contract, release
    didactic audit, `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: `trig_special_value_cos_two_pi_thirds_negative_half`
    (`cos(2*pi/3) -> -1/2`)
  - `derive_bridge_check`: promoted because the engine already reaches the
    correct exact value, but derive cannot certify it through the intended
    target-aware trig route when the target surface form carries the sign in a
    quotient numerator
  - `engine_feedback_check`: reusable derive matcher gap, not a new
    mathematical identity table and not a one-off cosine case
  - `retain_if`: route reports `rewrite trigs`, visible rule is
    `Evaluar valor trigonométrico especial`, derived count increases by one,
    `generic_simplify_expected` remains `0`, didactic flags remain `0`, and
    guardrails pass
  - `reject_if`: the change hides target mismatch through generic simplify,
    broadens matching to mathematically unsafe sign movement, changes
    simplification output forms, or regresses any guardrail
- structural_axis:
  - negative quotient sign normalization in derive target matching
- why_this_is_not_a_duplicate:
  - previous trig special-value cycles exposed table entries and parser-shape
    variants; this candidate addresses a shared matcher weakness that blocked
    already-known negative exact quotient results
- discovery_or_promotion:
  - promotion from reproducible CLI generic-simplify probes:
    `cos(2*pi/3) -> -1/2`, `cos(3*pi/4) -> -sqrt(2)/2`, and
    `cos(5*pi/6) -> -sqrt(3)/2`
- rejected_local_candidate_detail:
  - adding second-quadrant cosine rows to the special-angle table was rejected
    as the primary fix after probes showed the runtime evaluator already knew
    the values; the blocking weakness was target-shape matching for negative
    quotients
- if_promoted_why_minimal_representative:
  - `cos(2*pi/3) -> -1/2` is the smallest stable negative quotient
    representative; nearby radical quotient forms are covered by focused tests
    rather than separate corpus promotions
- local_result:
  - `strong_target_match` now extracts a syntactic global sign through
    quotient numerators and denominators, so `-(a/b)`, `(-a)/b`, and
    `a/(-b)` share the same signed core for target comparison
  - matcher unit coverage now includes `-(sqrt(2)/2)` vs `-sqrt(2)/2` and
    `-(1/2)` vs `-1/2`
  - target-aware trig and direct derive tests now cover
    `cos(2*pi/3) -> -1/2`, `cos(3*pi/4) -> -sqrt(2)/2`, and
    `cos(5*pi/6) -> -sqrt(3)/2`
  - promoted one corpus row:
    `trig_special_value_cos_two_pi_thirds_negative_half`
- guardrails:
  - `cargo fmt`
  - CLI probes for `derive(cos(2*pi/3), -1/2)`,
    `derive(cos(3*pi/4), -sqrt(2)/2)`, and
    `derive(cos(5*pi/6), -sqrt(3)/2)` reported `Strategy: rewrite trigs` and
    visible rule `Evaluar valor trigonométrico especial`
  - `cargo test -q -p cas_solver derive::match_support::tests::matches_global_negation_against_negative_quotient_numerator -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive::match_support::tests::matches_negative_rational_quotient_surface_forms -- --exact --nocapture`
  - `cargo test -q -p cas_solver target_aware_trig_rewrite_recognizes_special_values -- --nocapture`
  - `cargo test -q -p cas_solver direct_derive_labels_trig_special_values_without_generic_simplify -- --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_trig_special_value_is_direct_and_unflagged -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_trig_representatives -- --exact --nocapture`
  - `make engine-fast`: `simplify_add_small 435/435`,
    `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=365 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `447 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as a derive robustness improvement: an existing exact-value engine
    result now reaches the intended target-aware `rewrite trigs` route across
    equivalent negative quotient surface forms, without adding duplicate table
    logic or changing simplification output

## 2026-04-29 - Auto-improvement cycle: derive half-scaled sine double-angle contraction

- candidate_capture:
  - `investment_class`: coverage
  - `success_condition`: `derive(sin(x)*cos(x), sin(2*x)/2)` routes through
    the specific `contract trig` stage instead of generic `simplify`; one
    minimal corpus row is promoted; `generic_simplify_expected` remains `0`;
    didactic audit remains at `0` flags; `make engine-fast` and
    `make engine-scorecard` pass
  - `primary_dimension`: derive bridgeability for scaled double-angle sine
    contraction targets
  - `secondary_dimension`: target-aware trig contraction can represent the
    unit-coefficient product `sin(u)*cos(u)` as half of the standard
    `sin(2u)` identity
  - `hypothesis`: the runtime simplifier already proves
    `sin(x)*cos(x) == sin(2*x)/2`, and derive already has the full
    `2*sin(x)*cos(x) -> sin(2*x)` contraction; the missing bridge is the
    scaled representative where the factor `2` is carried by the target as
    division by `2`
  - `relevant_lanes`: CLI derive probes, target-aware trig unit, direct derive
    strategy label, derive contract representative, focused didactic audit,
    `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: `contract_trig_half_scaled_double_sin`
    (`sin(x)*cos(x) -> sin(2*x)/2`)
  - `derive_bridge_check`: promoted only if the route becomes a visible
    trig-contract step rather than a generic simplifier proof
  - `engine_feedback_check`: no runtime simplification gap is required here;
    the value of the change is exposing an already-known identity through the
    didactic derive layer
  - `retain_if`: route reports `contract trig`, visible rule is the double-angle
    identity, derived count increases by one, `generic_simplify_expected`
    remains `0`, didactic flags remain `0`, and guardrails pass
  - `reject_if`: the implementation hard-codes only the exact printed target,
    interferes with existing full double-angle contractions, creates extra
    generic-simplify expectations, or regresses any guardrail
- structural_axis:
  - unit-coefficient sine/cosine product contraction to half-scaled
    double-angle sine
- why_this_is_not_a_duplicate:
  - existing corpus rows cover `2*sin(x)*cos(x) -> sin(2*x)` and the reverse
    expansion; they do not cover the equivalent half-scaled target form that
    appeared as a generic-simplify discovery
- discovery_or_promotion:
  - promoted from an identity feeder and CLI probe where
    `derive(sin(x)*cos(x), sin(2*x)/2)` succeeded only through generic
    `simplify`
- if_promoted_why_minimal_representative:
  - `sin(x)*cos(x) -> sin(2*x)/2` is the smallest representative of the scaled
    contraction gap; broader product-to-sum and mixed double-angle products
    already have separate coverage
- local_result:
  - `try_rewrite_trig_contraction_target_aware` now recognizes a two-factor
    `sin(u)*cos(u)` product and builds the target-aware candidate
    `sin(2u)/2` through the double-angle sine identity
  - direct derive now reports `Strategy: contract trig` for
    `derive(sin(x)*cos(x), sin(2*x)/2)` with visible rule
    `Expandir ángulo doble`, instead of generic `simplify`
  - promoted one corpus row:
    `contract_trig_half_scaled_double_sin`
  - focused unit and didactic tests cover the target-aware rewrite, direct
    strategy label, and no-audit-flag trace behavior
- guardrails:
  - `cargo fmt`
  - CLI probe for `derive(sin(x)*cos(x), sin(2*x)/2)` reported
    `Strategy: contract trig` and visible rule `Expandir ángulo doble`
  - `cargo test -q -p cas_solver derive::trig::tests::contracts_half_scaled_sine_double_angle_product_target_aware -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive_command::tests::direct_derive_contracts_half_scaled_sine_double_angle_without_generic_simplify -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_half_scaled_sine_double_angle_contracts_directly -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_trig_representatives -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=366`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `contract trig=44`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=448`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=469`, `mean_step_count=1.06`
  - `make engine-fast`: `simplify_add_small 435/435`,
    `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=366 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `448 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as a derive coverage improvement: an identity already proved by
    the runtime simplifier now has a specific, didactic `contract trig` bridge
    for the half-scaled double-angle form, without changing broad simplify
    runtime behavior

## 2026-04-29 - Auto-improvement cycle: derive fractional binomial expansion classification

- candidate_capture:
  - `investment_class`: coverage
  - `success_condition`: `derive((x+1/2)^2, x^2 + x + 1/4)` routes through
    `expand` with the visible binomial-expansion rule instead of generic
    `simplify`; one minimal corpus row is promoted; `generic_simplify_expected`
    remains `0`; didactic audit remains at `0` flags; `make engine-fast` and
    `make engine-scorecard` pass
  - `primary_dimension`: derive target classification for binomial expansions
    with rational constants inside the source
  - `secondary_dimension`: didactic highlight quality for expanded polynomial
    targets that the engine already proves but currently explains as a generic
    simplification
  - `hypothesis`: `detect_expanded_target` rejects every source containing a
    division-like subterm before it asks the expansion provider for a
    classification; that protects fraction-specific routes, but it also blocks
    safe binomial powers such as `(x+1/2)^2` from reaching the existing expand
    strategy
  - `relevant_lanes`: CLI derive probe, target-classifier unit, expand unit,
    direct derive strategy label, derive contract representative, focused
    didactic audit, `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: `expand_fractional_binomial_square`
    (`(x+1/2)^2 -> x^2 + x + 1/4`)
  - `derive_bridge_check`: promoted because the engine can already expand and
    prove the target, but derive classifies it as generic simplification rather
    than a teachable binomial expansion
  - `engine_feedback_check`: no new runtime simplifier capability is required;
    the reusable gap is a target-classifier over-filter for expanded binomial
    powers with fractional coefficients
  - `retain_if`: route reports `expand`, visible rule is `Expandir binomio`,
    derived count increases by one, `generic_simplify_expected` remains `0`,
    didactic flags remain `0`, and guardrails pass
  - `reject_if`: the relaxation causes fraction expansion/cancellation targets
    to be misclassified as general expand, adds generic simplify expectations,
    changes simplification output forms, or regresses any guardrail
- structural_axis:
  - rational-coefficient binomial power expansion in derive target
    classification
- why_this_is_not_a_duplicate:
  - existing derive rows cover symbolic/integer binomial powers and fraction
    expansion as separate families; no row covers a binomial source with a
    rational constant that previously tripped the division-like source filter
- discovery_or_promotion:
  - promoted from an identity-pair feeder and CLI probe where
    `(x+1/2)^2 -> x^2 + x + 1/4` succeeded only through generic `simplify`
    with a misleading `Combine Constants` step
- if_promoted_why_minimal_representative:
  - `(x+1/2)^2` is the smallest stable rational-coefficient binomial square;
    nearby `(x-1/2)^2`, `(x+1/3)^2`, and symbolic fractional variants are
    useful focused coverage, but not all need corpus promotion
- local_result:
  - `detect_expanded_target` now allows division-like source terms only after
    the expansion provider has identified the rewrite as `BinomialPower`; other
    division-like sources remain blocked from generic expand classification so
    fraction-specific routes keep ownership
  - direct derive now reports `Strategy: expand` for
    `derive((x+1/2)^2, x^2 + x + 1/4)` and renders the visible rule
    `Expandir binomio`, instead of generic `simplify` with `Combine Constants`
  - promoted one corpus row:
    `expand_fractional_binomial_square`
  - focused unit and didactic tests cover target classification, target-aware
    expand, direct strategy label, fraction-expansion separation, and
    no-audit-flag trace behavior
- guardrails:
  - `cargo fmt`
  - CLI probe for `derive((x+1/2)^2, x^2 + x + 1/4)` reported
    `Strategy: expand` and visible rule `Expandir binomio`
  - `cargo test -q -p cas_solver derive::target_classifier::tests::classifies_fractional_binomial_square_as_expanded_target -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive::expand::tests::expands_fractional_binomial_target_aware -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive_command::tests::direct_derive_expands_fractional_binomial_square_without_generic_simplify -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive::target_classifier::tests::classifies_tabulated_fraction_expanded_targets -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_direct_binomial_expansions_are_audit_clean_without_padding -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_expand_representatives -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=367`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `expand=31`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=449`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=469`, `mean_step_count=1.06`
  - `make engine-fast`: `simplify_add_small 435/435`,
    `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=367 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `449 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
- decision:
  - retained as a derive coverage/classification improvement: a rational
    binomial expansion that the engine already proves now uses the intended
    `expand` route and didactic binomial step, while preserving fraction-family
    classification and all global guardrails

## 2026-04-29 - Auto-improvement cycle: derive tangent half-angle substitution for sine

- candidate_capture:
  - `investment_class`: coverage
  - `success_condition`: `derive(sin(x), 2*tan(x/2)/(1+tan(x/2)^2))`
    routes through `expand trig` with the visible half-angle tangent identity
    instead of generic `simplify`; one minimal corpus row is promoted;
    `generic_simplify_expected` remains `0`; didactic audit remains at `0`
    flags; `make engine-fast` and `make engine-scorecard` pass
  - `primary_dimension`: derive bridgeability for the tangent half-angle
    substitution from a direct sine source to a rational `tan(x/2)` target
  - `secondary_dimension`: engine-to-derive feedback, because the simplifier
    already proves this identity while derive currently hides it behind a
    generic simplification step
  - `hypothesis`: the trig derive provider has half-angle tangent routes for
    `tan(u)` to sine/cosine quotients, but it lacks the inverse Weierstrass
    substitution forms for `sin(u)` and `cos(u)` expressed with `tan(u/2)`;
    adding the sine form should remove a generic simplify fallback without
    changing broad simplification behavior
  - `relevant_lanes`: CLI derive probe, trig target-aware unit, direct derive
    strategy label, derive contract representative, focused didactic audit,
    `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: `expand_trig_tangent_half_angle_substitution_sine`
    (`sin(x) -> 2*tan(x/2)/(1+tan(x/2)^2)`)
  - `derive_bridge_check`: promoted because an engine-known identity from the
    identity feeder is reachable only through generic simplify today, so the
    missing piece is a teachable target-aware derive bridge
  - `engine_feedback_check`: no new simplifier runtime capability is required;
    the runtime feedback is used to identify a missing derive route and to
    keep `derive` aligned with identities already proven elsewhere
  - `retain_if`: route reports `expand trig`, visible rule is
    `Aplicar identidad de tangente de ángulo mitad`, derived count increases
    by one, `generic_simplify_expected` remains `0`, didactic flags remain
    `0`, and guardrails pass
  - `reject_if`: the rewrite overgeneralizes unrelated tangent targets,
    introduces unsupported domain conditions, adds generic simplify
    expectations, changes simplification output forms, or regresses any
    guardrail
- structural_axis:
  - tangent half-angle substitution / Weierstrass rational trig form
- why_this_is_not_a_duplicate:
  - existing derive rows cover `tan(x/2)` expanding to
    `sin(x)/(1+cos(x))` and `(1-cos(x))/sin(x)`, plus the corresponding
    contraction; they do not cover direct `sin(x)` or `cos(x)` sources
    rewritten into rational functions of `tan(x/2)`
- discovery_or_promotion:
  - promoted from identity-feeder and CLI probes where
    `sin(x) -> 2*tan(x/2)/(1+tan(x/2)^2)` succeeds semantically but is
    classified as generic `simplify`
- if_promoted_why_minimal_representative:
  - the sine form is the smallest stable representative: it has a simple
    numerator `2*tan(x/2)` and exercises the missing bridge without committing
    the corpus to every tangent half-angle rationalization variant in one
    cycle
- local_result:
  - `try_rewrite_trig_expansion` now recognizes direct `sin(u)` and `cos(u)`
    sources whose target is the tangent half-angle rational form in
    `tan(u/2)`; only the sine form is promoted to the derive corpus, while the
    cosine form is covered by focused unit/direct derive tests
  - direct derive now reports `Strategy: expand trig` for
    `derive(sin(x), 2*tan(x/2)/(1+tan(x/2)^2))` with rule
    `Half-Angle Tangent Identity`, instead of generic `simplify`
  - promoted one corpus row:
    `expand_trig_tangent_half_angle_substitution_sine`
  - focused unit, direct derive, representative contract, and didactic tests
    cover the route without introducing padded substeps or generic simplify
    expectations
- guardrails:
  - `cargo fmt`
  - CLI probe for `derive(sin(x), 2*tan(x/2)/(1+tan(x/2)^2)` changed from
    `Strategy: simplify` to `Strategy: expand trig` with rule
    `Half-Angle Tangent Identity`
  - `cargo test -q -p cas_solver derive::trig::tests::rewrites_tangent_half_angle_substitution_variants_target_aware -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive_command::tests::direct_derive_expands_tangent_half_angle_substitution_without_generic_simplify -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_half_angle_tangent_simplified_argument_uses_specific_identity -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_trig_representatives -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=368`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `expand trig=88`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=450`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=469`, `mean_step_count=1.06`
  - `make engine-fast`: `simplify_add_small 435/435`,
    `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=368 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `450 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
  - `git diff --check`
- decision:
  - retained as a derive bridgeability improvement: an identity already proven
    by the simplifier now has a specific target-aware `expand trig` route and
    didactic half-angle tangent rule, with no new simplify runtime behavior and
    no guardrail regression

## 2026-04-29 - Auto-improvement cycle: derive arctan double-angle projection substeps

- candidate_capture:
  - `investment_class`: coverage
  - `success_condition`: `derive(sin(2*arctan(x)), 2*x/(1+x^2))` is promoted
    as a minimal `expand trig` corpus row and renders the double-angle move
    with concrete inverse-trig projection substeps instead of a single opaque
    local jump; `generic_simplify_expected` remains `0`; didactic audit remains
    at `0` flags; `make engine-fast` and `make engine-scorecard` pass
  - `primary_dimension`: derive reachability coverage for substitution-based
    inverse-trig double-angle identities
  - `secondary_dimension`: didactic quality for a composed transition that
    combines double-angle expansion with `sin(arctan(x))` and
    `cos(arctan(x))` projections
  - `hypothesis`: runtime derive already recognizes arctangent double-angle
    rational targets, but the focused didactic substep generator only handles
    `arcsin`/`arccos` as inverse-trig double-angle arguments; adding `arctan`
    to that focused path makes the existing engine capability teachable and
    suitable for durable corpus promotion
  - `relevant_lanes`: CLI derive probe, direct derive strategy regression,
    focused derive didactic audit, derive contract representative, release
    derive contract, `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: `expand_trig_double_sin_arctan_projection`
    (`sin(2*arctan(x)) -> 2*x/(1+x^2)`)
  - `derive_bridge_check`: promoted because the identity is already present in
    the substitution/metamorphic feeder and the engine can derive it with a
    specific strategy; the missing retained artifact is a corpus row plus
    didactic substeps that expose the hidden projection step
  - `engine_feedback_check`: no new simplifier runtime rule is required; this
    consumes a reusable engine capability and documents that the companion
    cosine arctangent projection can be promoted later if needed
  - `retain_if`: route reports `expand trig`, visible rule is
    `Expandir ángulo doble`, web substeps include both the double-angle
    intermediate and inverse-trig projection substitution, derived count
    increases by one, didactic flags remain `0`, and guardrails pass
  - `reject_if`: the row falls back to generic `simplify`, substeps duplicate
    the parent before/after without an intermediate, arctan support breaks the
    existing `arcsin`/`arccos` double-angle cases, or any guardrail regresses
- structural_axis:
  - inverse-trig argument inside double-angle trig expansion, arctangent
    right-triangle projection to rational target
- why_this_is_not_a_duplicate:
  - existing derive rows cover direct `sin(arctan(x))` and `cos(arctan(x))`
    projections, plus double-angle expansions for ordinary and selected
    inverse-trig arguments; no corpus row covers the substitution identity
    `sin(2*arctan(x)) -> 2*x/(1+x^2)` from the metamorphic feeder
- discovery_or_promotion:
  - promoted from a cheap CLI probe and `substitution_identities.csv`, where
    the runtime already reports `Strategy: expand trig` but the durable derive
    corpus did not yet exercise the composed arctangent projection path
- if_promoted_why_minimal_representative:
  - the sine form is the smallest stable representative: it uses the standard
    `2*sin(u)*cos(u)` double-angle identity and both arctangent right-triangle
    projections; the companion cosine rational form is useful, but can stay as
    focused/direct coverage unless a later cycle needs the second corpus row
- local_result:
  - `generate_inverse_trig_double_angle_expansion_substeps` now recognizes
    `arctan`/`atan` arguments in addition to `arcsin`/`arccos`, so
    `sin(2*arctan(x))` renders a concrete double-angle intermediate before
    substituting the arctangent right-triangle projections
  - promoted one derive corpus row:
    `expand_trig_double_sin_arctan_projection`
  - extended the direct derive regression for inverse-trig double-angle
    projections to include both `sin(2*arctan(x))` and
    `cos(2*arctan(x))`, while only promoting the sine form to durable corpus
  - focused didactic coverage now requires the promoted arctangent case to show
    the same two substeps as the existing `arcsin`/`arccos` double-angle cases:
    first the double-angle identity, then the inverse-trig projection
    substitution
- guardrails:
  - `cargo fmt`
  - CLI probe for `derive(sin(2*arctan(x)), 2*x/(1+x^2))` reported
    `Strategy: expand trig` and rule `Double Angle Expansion`
  - `cargo test -q -p cas_solver derive_command::tests::direct_derive_expands_inverse_trig_double_angle_projections -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_inverse_trig_double_angle_expansions_show_projection -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_trig_representatives -- --exact --nocapture`
  - `cargo test --release -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes -- --exact --nocapture`: `derived=369`, `unsupported=0`, `not_equivalent=1`, `generic_simplify_expected=0`, `expand trig=89`
  - `cargo test --release -q -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`: `cases=451`, `flagged=0`, `no_web_substeps=0`, `no_web_steps=0`, `total_web_substeps=471`, `mean_step_count=1.06`
  - `make engine-fast`: `simplify_add_small 435/435`,
    `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=369 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `451 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
  - `git diff --check`
- decision:
  - retained as a derive coverage and didactic-quality improvement: an
    existing engine-supported substitution identity is now represented in the
    durable derive corpus and its composed double-angle/projection transition
    is explained through concrete substeps, without changing broad simplify
    runtime behavior or regressing any guardrail

## 2026-04-29 - Auto-improvement cycle: derive csc/cot reciprocal Pythagorean corpus promotion

- candidate_capture:
  - `investment_class`: coverage
  - `success_condition`: `derive(csc(x)^2 - cot(x)^2, 1)` is promoted from
    focused/inline coverage into the durable derive corpus with strategy
    `rewrite trigs`; representative derive tests, focused didactic audit,
    `make engine-fast`, and `make engine-scorecard` pass; `generic_simplify`
    expectations remain `0`
  - `primary_dimension`: derive coverage for the cosecant/cotangent branch of
    reciprocal Pythagorean identities
  - `secondary_dimension`: corpus reuse of an engine-supported identity already
    present in `identity_pairs.csv`, keeping derive aligned with the engine
    feeder instead of only testing the secant/tangent sibling
  - `hypothesis`: the runtime and didactic layers already support the
    `csc²(u) - cot²(u) = 1` transition, but the durable derive corpus only
    includes the `sec²(u) - tan²(u) = 1` sibling; promoting the missing
    csc/cot row should increase stable branch coverage without changing
    simplification behavior
  - `relevant_lanes`: CLI derive probe, trig target-aware unit, simplify/log
    representative derive contract, focused derive didactic audit, release
    derive contract, `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: `csc_cot_pythagorean_to_one`
    (`csc(x)^2 - cot(x)^2 -> 1`)
  - `derive_bridge_check`: promoted because this is the smallest durable
    csc/cot representative of an engine-known identity; the existing inline
    didactic test proves the path is meaningful but not yet part of the main
    corpus metrics
  - `engine_feedback_check`: no missing engine capability was found; the
    feedback is that a reusable engine transition had a derive corpus gap
  - `retain_if`: direct derive reports `rewrite trigs`, the promoted row passes
    representative and release derive contracts, didactic audit remains at `0`
    flags, and engine guardrails keep `failed=0` and `timeouts=0`
  - `reject_if`: the promotion duplicates an existing durable row exactly,
    routes through generic `simplify`, breaks the existing sec/tan reciprocal
    identity path, or regresses any guardrail
- structural_axis:
  - reciprocal Pythagorean identity branch coverage for `csc/cot`
- why_this_is_not_a_duplicate:
  - the corpus already has `sec(x)^2 - tan(x)^2 -> 1`, but not the
    cosecant/cotangent sibling; this exercises a separate reciprocal function
    pair and a separate runtime rewrite kind while keeping the case minimal
- discovery_or_promotion:
  - promoted from existing `identity_pairs.csv` coverage and an inline
    didactic audit case; no generated candidate failed in this cycle
- if_promoted_why_minimal_representative:
  - the naked `csc²-cot²` identity is the minimal stable representative; larger
    passthrough or contextual variants would add corpus weight without changing
    the transition family
- local_result:
  - promoted one derive corpus row:
    `csc_cot_pythagorean_to_one`
  - converted the focused didactic audit from an inline ad-hoc case to the
    durable corpus row, so the same case now contributes to corpus metrics and
    didactic regression coverage
  - added a direct derive regression proving the command routes through
    `DeriveStrategy::TrigRewrite` with rule `Reciprocal Pythagorean Identity`
    instead of depending on a generic simplification fallback
  - added the row to the simplify/log representative derive contract slice
    because the branch belongs to the `rewrite trigs` simplify family rather
    than trig expansion/contraction
- guardrails:
  - `cargo fmt`
  - CLI probe for `derive(csc(x)^2 - cot(x)^2, 1)` reported
    `Strategy: rewrite trigs` and rule `Reciprocal Pythagorean Identity`
  - `cargo test -q -p cas_solver derive_command::tests::direct_derive_rewrites_csc_cot_pythagorean_to_one_without_simplify -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive::trig::tests::rewrites_csc_cot_pythagorean_to_one_target_aware -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_csc_cot_pythagorean_to_one_uses_direct_identity_language -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_simplify_log_representatives -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_partition_covers_corpus -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_perf_slices_cover_corpus -- --exact --nocapture`
  - `make engine-fast`: `simplify_add_small 435/435`,
    `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=370 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `452 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
  - `git diff --check`
- decision:
  - retained as a conservative derive coverage improvement: a real
    engine-supported reciprocal Pythagorean branch is now represented in the
    durable derive corpus and didactic audit without changing broad simplify
    runtime behavior or regressing guardrails

## 2026-04-29 - Auto-improvement cycle: derive tangent quotient expansion corpus promotion

- candidate_capture:
  - `investment_class`: coverage
  - `success_condition`: `derive(tan(x), sin(x)/cos(x))` is promoted from
    shadow/embedded diagnostic coverage into the durable derive corpus with
    strategy `expand trig`; direct derive, representative contract, focused
    didactic audit, `make engine-fast`, and `make engine-scorecard` pass;
    `generic_simplify_expected` remains `0`
  - `primary_dimension`: derive bridgeability for the root tangent quotient
    expansion direction
  - `secondary_dimension`: engine-to-derive corpus reuse, because the same
    root identity is already sampled by `derive_shadow_pressure` from
    `embedded_equivalence_context_corpus.csv`
  - `hypothesis`: derive already has a target-aware `TanToSinCos` provider and
    didactic substep generator, but the durable corpus covers only richer
    tangent quotient contractions such as `sin(2*x)/cos(2*x) -> tan(2*x)`;
    promoting the root expansion closes the missing source-to-target direction
    without changing simplifier runtime behavior
  - `relevant_lanes`: CLI derive probe, direct derive strategy regression,
    trig representative derive contract, focused derive didactic audit,
    release derive contract, `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: `expand_trig_tan_to_sin_cos`
    (`tan(x) -> sin(x)/cos(x)`)
  - `derive_bridge_check`: promoted because the case is currently only a
    diagnostic shadow row; it should be a durable target-form transition since
    it exposes a real domain condition `cos(x) != 0` and a direct visible rule
  - `engine_feedback_check`: no new engine capability is required; the feedback
    is that an engine-known identity and shadow case had no durable derive
    corpus row in the expansion direction
  - `retain_if`: direct derive reports `expand trig`, internal rule is
    `Trig Expansion`, web visible rule is
    `Expandir tangente como seno entre coseno`, the web audit shows the
    tangent quotient identity with no flags, derived count increases by one,
    and all guardrails pass
  - `reject_if`: the row routes through generic `simplify`, loses the domain
    condition, duplicates an existing durable root expansion, or any guardrail
    regresses
- structural_axis:
  - reciprocal/quotient trigonometric expansion direction for tangent
- why_this_is_not_a_duplicate:
  - existing durable rows cover contraction from sine/cosine quotients to
    tangent and reciprocal expansions for sec/csc/cot; they do not cover the
    root tangent source expanding to `sin(x)/cos(x)`, which is also the exact
    identity sampled by the shadow lane
- discovery_or_promotion:
  - promoted from `derive_shadow_pressure` and embedded-equivalence rows after
    a CLI probe confirmed the specific `expand trig` route
- if_promoted_why_minimal_representative:
  - the naked root identity is the smallest stable representative; scaled,
    negated, or contextual variants stay in focused/embedded coverage because
    they do not add a separate transition family for this cycle
- local_result:
  - promoted `expand_trig_tan_to_sin_cos` to `derive_pairs.csv`
    (`tan(x) -> sin(x)/cos(x)`, `expected_strategy=expand trig`)
  - added it to the trig representative derive contract slice and added a
    direct derive regression proving it routes through `DeriveStrategy::TrigExpand`
    with internal rule `Trig Expansion`
  - made the didactic web rule description-specific:
    `Expandir tangente como seno entre coseno`
  - encoded that direct one-step tangent expansion is audit-clean without
    padding substeps, because the only possible substep duplicates the parent
    rewrite
- guardrails:
  - `cargo fmt`
  - CLI JSON probe for `derive(tan(x), sin(x)/cos(x))` reported
    `Strategy: expand trig`, one step, web rule
    `Expandir tangente como seno entre coseno`, and required condition
    `cos(x) != 0`
  - `cargo test -q -p cas_solver derive_command::tests::direct_derive_expands_tangent_to_sine_over_cosine_without_simplify -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive::trig::tests::expands_tabulated_scaled_trig_targets_aware -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_tangent_quotient_expansion_uses_direct_identity_language -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_trig_representatives -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_partition_covers_corpus -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_perf_slices_cover_corpus -- --exact --nocapture`
  - `make engine-fast`: `simplify_add_small 435/435`,
    `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=371 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `453 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
  - `git diff --check`
- decision:
  - retained as a conservative derive/didactic coverage improvement: a
    shadow-only root tangent expansion is now a durable derive corpus row with
    a direct strategy regression, clean web audit, explicit domain condition,
    and no simplifier runtime change

## 2026-04-29 - Auto-improvement cycle: derive cosecant reciprocal expansion corpus promotion

- candidate_capture:
  - `investment_class`: coverage
  - `success_condition`: `derive(csc(x), 1/sin(x))` is promoted into the
    durable derive corpus with strategy `expand trig`; direct derive, trig
    representative contract, focused didactic audit, `make engine-fast`, and
    `make engine-scorecard` pass; `generic_simplify_expected` remains `0`
  - `primary_dimension`: derive bridgeability for the root reciprocal
    trigonometric expansion direction of cosecant
  - `secondary_dimension`: didactic specificity for reciprocal trig expansion
    steps that currently render under a generic reciprocal identity label
  - `hypothesis`: derive already has a target-aware
    `ExpandCscToRecipSin` provider and the engine reports the required
    condition `sin(x) != 0`, but the durable corpus covers only secant
    reciprocal expansion plus csc/cot contraction; promoting the csc expansion
    closes a missing reciprocal-function direction without changing broad
    simplifier runtime behavior
  - `relevant_lanes`: CLI derive probe, direct derive strategy regression,
    trig representative derive contract, focused derive didactic audit,
    release derive contract, `make engine-fast`, `make engine-scorecard`
  - `promotion_target`: `expand_trig_csc_reciprocal`
    (`csc(x) -> 1/sin(x)`)
  - `derive_bridge_check`: promoted because the engine and derive strategy
    already expose a real one-step target-form transition with an explicit
    nonzero condition; the missing piece is durable reachability and audit
    coverage for this root direction
  - `engine_feedback_check`: no new engine capability is required; the feedback
    is that an engine-supported reciprocal trig expansion had no durable derive
    corpus row for cosecant
  - `retain_if`: direct derive reports `expand trig`, internal rule is
    `Reciprocal Trig Identity`, web visible rule is cosecant-specific, the web
    audit has no flags, derived count increases by one, and all guardrails pass
  - `reject_if`: the row routes through generic `simplify`, loses the
    `sin(x) != 0` condition, duplicates an existing durable csc expansion, or
    any guardrail regresses
- structural_axis:
  - reciprocal trigonometric expansion direction for cosecant
- why_this_is_not_a_duplicate:
  - existing durable rows cover `sec(x) -> 1/cos(x)` and the contraction
    `1/sin(x) -> csc(x)`, but not the root source-form expansion
    `csc(x) -> 1/sin(x)`
- discovery_or_promotion:
  - promoted from an already-supported derive provider after CLI probes
    confirmed the specific `expand trig` route and domain condition
- if_promoted_why_minimal_representative:
  - the naked root identity is the smallest stable representative; the sibling
    cotangent quotient expansion is a plausible later cycle, but adding it here
    would bundle a separate quotient shape into the same promotion
- local_result:
  - promoted `expand_trig_csc_reciprocal` to `derive_pairs.csv`
    (`csc(x) -> 1/sin(x)`, `expected_strategy=expand trig`)
  - added it to the trig representative derive contract slice and added a
    direct derive regression proving it routes through
    `DeriveStrategy::TrigExpand` with internal rule
    `Reciprocal Trig Identity`
  - made this expansion's web rule description-specific:
    `Reescribir cosecante como recíproco del seno`
  - encoded that the direct one-step cosecant expansion is audit-clean without
    padding substeps because the possible substep duplicates the parent rewrite
- guardrails:
  - `cargo fmt`
  - CLI JSON probe for `derive(csc(x), 1/sin(x))` reported
    `Strategy: expand trig`, one step, web rule
    `Reescribir cosecante como recíproco del seno`, and required condition
    `sin(x) != 0`
  - `cargo test -q -p cas_solver derive_command::tests::direct_derive_expands_cosecant_to_reciprocal_sine_without_simplify -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive::trig::tests::expands_tabulated_scaled_trig_targets_aware -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_cosecant_reciprocal_expansion_uses_direct_identity_language -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_trig_representatives -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_partition_covers_corpus -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_perf_slices_cover_corpus -- --exact --nocapture`
  - `make engine-fast`: `simplify_add_small 435/435`,
    `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=372 unsupported=0 not_equivalent=1`, derive shadow pressure
    `50/50`, derive audit `454 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, simplify strict `16518/16518 proved-symbolic`,
    `0 failed`, `0 timeouts`
  - `git diff --check`
- decision:
  - retained as a conservative derive/didactic coverage improvement: a
    previously engine-supported but corpus-missing reciprocal cosecant
    expansion is now durable, teachable, guarded by a direct strategy
    regression, and does not change broad simplifier runtime behavior

## 2026-04-29 - Auto-improvement cycle: derive cotangent quotient expansion

- investment_class: coverage
- success_condition:
  - `derive(cot(x), cos(x)/sin(x))` is promoted as
    `expand_trig_cot_quotient`; direct derive regression, trig representative
    contract, focused didactic audit, `make engine-fast`, and
    `make engine-scorecard` pass; `generic_simplify_expected` remains `0`
- primary_dimension:
  - derive bridgeability for the root quotient-form trigonometric expansion of
    cotangent
- secondary_dimension:
  - didactic specificity for reciprocal/quotient trig expansion steps that
    currently share a generic reciprocal identity label
- hypothesis:
  - derive already has a target-aware `ExpandCotToCosSin` provider and the
    engine reports the required condition `sin(x) != 0`, but the durable corpus
    covers tangent, secant, and cosecant expansions plus cotangent contraction
    without the root cotangent expansion direction; promoting the minimal root
    case closes that quotient-shape gap without changing broad simplifier
    runtime behavior
- relevant_lanes:
  - CLI derive probe, direct derive strategy regression, trig representative
    derive contract, focused derive didactic audit, release derive contract,
    `make engine-fast`, `make engine-scorecard`
- promotion_target:
  - `expand_trig_cot_quotient` (`cot(x) -> cos(x)/sin(x)`)
- derive_bridge_check:
  - promoted because the existing provider reaches the exact target in one
    `expand trig` step with a real domain condition; the missing piece is
    durable reachability plus specific didactic rendering for this quotient
    expansion direction
- engine_feedback_check:
  - no new engine runtime capability is required; the feedback is that an
    engine-supported quotient trig expansion had no durable derive corpus row
    even though its inverse contraction was already covered
- retain_if:
  - direct derive reports `expand trig`, internal rule is
    `Reciprocal Trig Identity`, description is
    `Expand cot(u) as cos(u) / sin(u)`, web visible rule is cotangent-specific,
    the web audit has no flags, derived count increases by one, and all
    guardrails pass
- reject_if:
  - the row routes through generic `simplify`, loses the `sin(x) != 0`
    condition, duplicates an existing durable cotangent expansion, or any
    guardrail regresses
- structural_axis:
  - reciprocal/quotient trigonometric expansion direction for cotangent
- why_this_is_not_a_duplicate:
  - existing durable rows cover `tan(x) -> sin(x)/cos(x)`, reciprocal root
    expansions for secant/cosecant, and `cos(x)/sin(x) -> cot(x)`, but not the
    root source-form expansion `cot(x) -> cos(x)/sin(x)`; this adds the missing
    opposite direction and a numerator/denominator quotient shape
- discovery_or_promotion:
  - promoted from an already-supported derive provider after a CLI probe
    confirmed the specific `expand trig` route and domain condition
- if_promoted_why_minimal_representative:
  - the naked root identity is the smallest stable representative; scaled or
    nested cotangent wrappers would be near-duplicates because the target-aware
    provider already has argument-aware coverage in unit tests
- local_result:
  - promoted `expand_trig_cot_quotient` to `derive_pairs.csv`
    (`cot(x) -> cos(x)/sin(x)`, `expected_strategy=expand trig`)
  - added it to the trig representative derive contract slice and added a
    direct derive regression proving it routes through
    `DeriveStrategy::TrigExpand` with internal rule
    `Reciprocal Trig Identity`
  - made this expansion's web rule description-specific:
    `Reescribir cotangente como coseno entre seno`
  - encoded that the direct one-step cotangent quotient expansion is
    audit-clean without padding substeps because the available substep would
    duplicate the parent rewrite
- guardrails:
  - `cargo fmt`
  - CLI probe for `derive(cot(x), cos(x)/sin(x))` reported
    `Strategy: expand trig`, one step, internal rule
    `Reciprocal Trig Identity`, and required condition `sin(x) != 0`
  - `cargo test -q -p cas_solver derive_command::tests::direct_derive_expands_cotangent_to_cosine_over_sine_without_simplify -- --exact --nocapture`
  - `cargo test -q -p cas_solver derive::trig::tests::expands_tabulated_scaled_trig_targets_aware -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_cotangent_quotient_expansion_uses_direct_identity_language -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_trig_representatives -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_partition_covers_corpus -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_perf_slices_cover_corpus -- --exact --nocapture`
  - `make engine-fast`: `simplify_add_small 435/435`,
    `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=373 unsupported=0 not_equivalent=1`, `generic_simplify_expected=0`,
    derive shadow pressure `50/50`, derive audit `455 cases / 0 flags`,
    simplify audit `14 cases / 0 flags`, simplify strict
    `16518/16518 proved-symbolic`, `0 failed`, `0 timeouts`
  - `git diff --check`
- decision:
  - retained as a conservative derive/didactic coverage improvement: a
    previously engine-supported but corpus-missing cotangent quotient expansion
    is now durable, teachable, guarded by a direct strategy regression, and
    does not change broad simplifier runtime behavior

## 2026-04-29 - Auto-improvement cycle: derive secant reciprocal wording

- investment_class: coverage
- success_condition:
  - existing `expand_trig_sec_reciprocal` keeps deriving
    `sec(x) -> 1/cos(x)` through `expand trig`, gains a direct strategy
    regression, renders with the specific web rule
    `Reescribir secante como recíproco del coseno`, and focused didactic
    audit, `make engine-fast`, and `make engine-scorecard` pass
- primary_dimension:
  - didactic quality for an already-covered derive reciprocal trig expansion
- secondary_dimension:
  - consistency of the sec/csc/cot reciprocal expansion triad after the previous
    csc and cot promotions
- hypothesis:
  - the engine and derive provider already reach `sec(x) -> 1/cos(x)` in one
    target-aware `expand trig` step with condition `cos(x) != 0`, but the web
    render still exposes the generic reciprocal-trig label while csc/cot now
    use direct source-specific wording; adding the missing secant-specific
    visible-rule case improves teachability without changing runtime behavior
- relevant_lanes:
  - CLI derive probe, direct derive strategy regression, focused derive
    didactic audit, release derive didactic audit, `make engine-fast`,
    `make engine-scorecard`
- promotion_target:
  - existing row `expand_trig_sec_reciprocal` (`sec(x) -> 1/cos(x)`) gains
    direct strategy and didactic-specificity guardrails
- derive_bridge_check:
  - retained as a derive didactic-quality improvement because bridgeability is
    already present; the gap is visible wording, not target classification,
    planner reachability, or reusable engine capability
- engine_feedback_check:
  - no engine runtime change is required; this cycle confirms the engine
    provider is reusable and only the public didactic label was less specific
    than its sibling identities
- retain_if:
  - direct derive reports `DeriveStrategy::TrigExpand`, internal rule is
    `Reciprocal Trig Identity`, description is
    `Expand sec(u) as 1 / cos(u)`, web visible rule is secant-specific, audit
    flags stay at `0`, and all guardrails pass
- reject_if:
  - the route falls back to generic `simplify`, the domain condition
    `cos(x) != 0` disappears, the web audit flags the direct step as padded or
    opaque, or any global guardrail regresses
- structural_axis:
  - reciprocal trigonometric expansion wording for secant
- why_this_is_not_a_duplicate:
  - csc and cot expansion wording was already made specific, while secant
    remains the only root reciprocal expansion in the trio rendering under the
    generic reciprocal-trig label
- discovery_or_promotion:
  - promoted as a didactic-quality guardrail on an existing durable derive row,
    not as a new corpus row
- if_promoted_why_minimal_representative:
  - the existing naked root row is the minimal representative; adding wrapper
    variants would not test a new transition or didactic concept
- local_result:
  - added a direct derive regression for `sec(x) -> 1/cos(x)` proving it routes
    through `DeriveStrategy::TrigExpand` with internal rule
    `Reciprocal Trig Identity` and description
    `Expand sec(u) as 1 / cos(u)`
  - made the web-visible rule description-specific:
    `Reescribir secante como recíproco del coseno`
  - updated the derive didactic audit assertion so
    `expand_trig_sec_reciprocal` must remain audit-clean with zero padded
    substeps and the secant-specific visible label
- guardrails:
  - `cargo fmt`
  - CLI probe for `derive(sec(x), 1/cos(x))` reported
    `Strategy: expand trig`, one step, internal rule
    `Reciprocal Trig Identity`, and required condition `cos(x) != 0`
  - `cargo test -q -p cas_solver derive_command::tests::direct_derive_expands_secant_to_reciprocal_cosine_without_simplify -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_secant_reciprocal_expansion_uses_direct_identity_language -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_trig_representatives -- --exact --nocapture`
  - `make engine-fast`: `simplify_add_small 435/435`,
    `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=373 unsupported=0 not_equivalent=1`, `generic_simplify_expected=0`,
    derive shadow pressure `50/50`, derive audit `455 cases / 0 flags`,
    simplify audit `14 cases / 0 flags`, simplify strict
    `16518/16518 proved-symbolic`, `0 failed`, `0 timeouts`
  - `git diff --check`
- decision:
  - retained as a low-risk derive didactic-quality improvement: the secant
    reciprocal expansion already had engine/derive bridgeability, and now its
    public web trace is as explicit as the sibling csc/cot reciprocal expansion
    steps without adding runtime logic or corpus noise

## 2026-04-29 - Auto-improvement cycle: derive reciprocal trig contraction wording

- investment_class: coverage
- success_condition:
  - existing `contract_trig_sec_reciprocal`,
    `contract_trig_csc_reciprocal`, and `contract_trig_cot_quotient` keep
    deriving through `contract trig`, gain direct strategy/didactic guardrails,
    render with source-specific web rules, and focused tests,
    `make engine-fast`, and `make engine-scorecard` pass
- primary_dimension:
  - didactic quality for already-covered derive reciprocal trig contractions
- secondary_dimension:
  - consistency between reciprocal trig expansion and contraction directions
- hypothesis:
  - the engine and derive planner already reach all three root contraction
    targets in one `contract trig` step, but the web render still collapses
    secant, cosecant, and cotangent contractions into the generic reciprocal
    trig label; using the existing concrete descriptions as visible-rule
    discriminators improves teachability without changing engine runtime
    behavior or adding corpus rows
- relevant_lanes:
  - CLI derive probes for the three root contractions, direct derive strategy
    regression, focused derive didactic audit, trig representative derive
    contract, release derive didactic audit, `make engine-fast`,
    `make engine-scorecard`
- promotion_target:
  - existing rows `contract_trig_sec_reciprocal`,
    `contract_trig_csc_reciprocal`, and `contract_trig_cot_quotient` gain
    direct strategy and didactic-specificity guardrails
- derive_bridge_check:
  - retained as a derive didactic-quality improvement because bridgeability is
    already present; the gap is visible wording, not target classification,
    planner reachability, or reusable engine capability
- engine_feedback_check:
  - no engine runtime change is required; this cycle confirms the contraction
    providers are reusable and only the public labels were less specific than
    their concrete internal descriptions
- retain_if:
  - each direct derive reports `DeriveStrategy::TrigContract`, internal rule is
    `Reciprocal Trig Identity`, each description matches the concrete
    sec/csc/cot recognition, web visible rules are source-specific, audit flags
    stay at `0`, and all guardrails pass
- reject_if:
  - any route falls back to generic `simplify`, any direct contraction loses
    its exact target, the web audit flags padded/opaque steps, or any global
    guardrail regresses
- structural_axis:
  - reciprocal trigonometric contraction wording for secant/cosecant/cotangent
- why_this_is_not_a_duplicate:
  - expansion wording for sec/csc/cot is now specific, but the inverse
    contraction direction still exposes a generic label for three distinct
    source shapes; this closes the opposite-direction didactic gap without
    adding near-duplicate corpus rows
- discovery_or_promotion:
  - promoted as a didactic-quality guardrail on existing durable derive rows,
    not as new corpus coverage
- if_promoted_why_minimal_representative:
  - the existing naked root rows are the minimal representatives; wrapper
    variants would not test a new transition or didactic concept
- local_result:
  - added a direct derive regression covering the three root contraction forms:
    `1/cos(x) -> sec(x)`, `1/sin(x) -> csc(x)`, and
    `cos(x)/sin(x) -> cot(x)`; all route through
    `DeriveStrategy::TrigContract` with internal rule
    `Reciprocal Trig Identity`
  - made the web-visible rules description-specific:
    `Reconocer secante desde un recíproco`,
    `Reconocer cosecante desde un recíproco`, and
    `Reconocer cotangente desde un cociente`
  - updated focused derive didactic audit assertions so the three existing rows
    must remain audit-clean with zero padded substeps and specific visible
    labels
  - added the three existing contraction rows to the trig representative
    derive contract slice; no new corpus rows were added
- guardrails:
  - `cargo fmt`
  - CLI probes for the three root contractions reported `Strategy: contract trig`
    and one `Reciprocal Trig Identity` step each
  - `cargo test -q -p cas_solver derive_command::tests::direct_derive_contracts_reciprocal_trig_root_forms_without_simplify -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_reciprocal_cosine_contraction_uses_direct_identity_language -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_reciprocal_sine_contraction_uses_direct_identity_language -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_cotangent_quotient_contraction_uses_direct_identity_language -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_trig_representatives -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `make engine-fast`: `simplify_add_small 435/435`,
    `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=373 unsupported=0 not_equivalent=1`, `generic_simplify_expected=0`,
    derive shadow pressure `50/50`, derive audit `455 cases / 0 flags`,
    simplify audit `14 cases / 0 flags`, simplify strict
    `16518/16518 proved-symbolic`, `0 failed`, `0 timeouts`
  - `git diff --check`
- decision:
  - retained as a low-risk derive didactic-quality improvement: the reciprocal
    contraction providers already had engine/derive bridgeability, and now the
    public web traces explain the concrete source shape instead of collapsing
    three distinct contractions into one generic reciprocal label

## 2026-04-29 - Auto-improvement cycle: derive hyperbolic quotient wording

- investment_class: coverage
- success_condition:
  - existing `hyperbolic_contract_tanh_quotient` keeps deriving through
    `rewrite hyperbolics`, gains direct strategy/didactic guardrails, renders
    with a source-specific web rule, removes the template-only `u` substep, and
    focused tests, `make engine-fast`, and `make engine-scorecard` pass
- primary_dimension:
  - didactic quality for already-covered derive hyperbolic quotient contraction
- secondary_dimension:
  - derive bridge coverage for a stable engine hyperbolic rewrite already used
    by the simplify/log representative lane
- hypothesis:
  - the engine and derive planner already reach `sinh(x)/cosh(x) -> tanh(x)` in
    one `rewrite hyperbolics` step, but the web trace still uses a generic
    hyperbolic quotient label plus a placeholder formula substep; using the
    concrete internal description as the visible-rule discriminator and treating
    the direct quotient rewrite as self-explanatory improves teachability
    without changing runtime behavior or adding corpus rows
- relevant_lanes:
  - CLI derive probe for the hyperbolic quotient contraction, direct derive
    strategy regression, focused derive didactic audit, simplify/log
    representative derive contract, release derive didactic audit,
    `make engine-fast`, `make engine-scorecard`
- promotion_target:
  - existing row `hyperbolic_contract_tanh_quotient` gains direct strategy and
    didactic-specificity guardrails
- derive_bridge_check:
  - retained as a derive didactic-quality improvement because bridgeability is
    already present; the gap is visible wording/substep quality, not target
    classification, planner reachability, or reusable engine capability
- engine_feedback_check:
  - no engine runtime change is required; this cycle confirms the hyperbolic
    quotient provider is reusable and only the public trace was less specific
    than its internal description
- retain_if:
  - direct derive reports `DeriveStrategy::HyperbolicRewrite`, internal rule is
    `Hyperbolic Quotient Identity`, description is
    `Recognize sinh(u) / cosh(u) as tanh(u)`, the web visible rule is
    source-specific with no padded substeps, audit flags stay at `0`, and all
    guardrails pass
- reject_if:
  - the route falls back to generic `simplify`, the exact target is lost, the
    web audit flags missing/noisy steps, or any global guardrail regresses
- structural_axis:
  - hyperbolic quotient contraction wording and substep suppression
- why_this_is_not_a_duplicate:
  - reciprocal trig direct quotient contractions were made specific in the prior
    cycle, while the analogous hyperbolic quotient contraction still emits a
    generic label plus a placeholder formula substep; this closes a neighboring
    family gap without adding near-duplicate corpus rows
- discovery_or_promotion:
  - promoted as a didactic-quality guardrail on an existing durable derive row,
    not as new corpus coverage
- if_promoted_why_minimal_representative:
  - the existing naked quotient row is the minimal representative; wrapper
    variants would not test a new transition or didactic concept
- local_result:
  - added a direct derive regression for
    `sinh(x)/cosh(x) -> tanh(x)` proving it routes through
    `DeriveStrategy::HyperbolicRewrite` with internal rule
    `Hyperbolic Quotient Identity` and description
    `Recognize sinh(u) / cosh(u) as tanh(u)`
  - made the web-visible rule description-specific:
    `Reconocer tangente hiperbólica desde un cociente`
  - suppressed the placeholder-only formula substep for that exact direct
    quotient contraction; the generated audit now shows one web step and zero
    substeps for `hyperbolic_contract_tanh_quotient`
  - added the existing row to the simplify/log representative derive contract
    slice; no new corpus rows were added
- guardrails:
  - `cargo fmt`
  - CLI probe for `derive sinh(x)/cosh(x), tanh(x)` reported
    `Strategy: rewrite hyperbolics` and one
    `Hyperbolic Quotient Identity` step
  - `cargo test -q -p cas_solver derive_command::tests::direct_derive_contracts_hyperbolic_quotient_without_simplify -- --exact --nocapture`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_hyperbolic_quotient_uses_specific_identity -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_simplify_log_representatives -- --exact --nocapture`
  - `make engine-fast`: `simplify_add_small 435/435`,
    `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=373 unsupported=0 not_equivalent=1`, `generic_simplify_expected=0`,
    derive shadow pressure `50/50`, derive audit `455 cases / 0 flags`,
    `total_web_substeps=470`, simplify audit `14 cases / 0 flags`, simplify
    strict `16518/16518 proved-symbolic`, `0 failed`, `0 timeouts`
- decision:
  - retained as a low-risk derive didactic-quality improvement: the hyperbolic
    quotient provider already had engine/derive bridgeability, and now the
    public web trace explains the concrete source shape without a generic label
    or decorative placeholder substep

## 2026-04-29 - Auto-improvement cycle: derive log-exp power inverse bridge

- investment_class: coverage
- success_condition:
  - `derive ln(exp(x)^2), 2*x` routes through `rewrite exponentials`, emits a
    visible power-normalization step followed by `Log-Exp Inverse`, stays out of
    generic `simplify`, and focused tests plus the engine guardrails pass
- primary_dimension:
  - derive bridge coverage for log-exp inverse identities that require one
    exponential power normalization before cancellation
- secondary_dimension:
  - didactic path quality for the newly exposed two-step route
- hypothesis:
  - the engine already simplifies `ln(exp(x)^2)` by multiplying exponents and
    then canceling `ln(exp(u))`, but `derive` lacked a bounded target-aware plan
    for that composed route and therefore fell back to generic simplify
- relevant_lanes:
  - CLI derive probe, exponential derive unit, direct derive strategy regression,
    derive contract representative slice, generic-simplify guard, focused derive
    didactic audit, `make engine-fast`, `make engine-scorecard`
- promotion_target:
  - new minimal row `log_exp_inverse_ln_exp_power`
- derive_bridge_check:
  - retained as a planner/coverage bridge, not a semantic engine change; the
    route reuses existing exponential power and log-exp inverse capabilities as
    two explicit steps
- engine_feedback_check:
  - no runtime change was required; this confirms a known simplification path
    can be exposed as a stable target-aware derive path with better trace
    quality
- retain_if:
  - direct derive reports `DeriveStrategy::ExponentialRewrite`, the CLI shows
    `Power of a Power` then `Log-Exp Inverse`, the derive audit stays at zero
    flags, and all guardrails pass
- reject_if:
  - the route collapses into one magical step, falls back to `simplify`, emits
    opaque didactic output, or regresses embedded/derive/simplify guardrails
- structural_axis:
  - target-aware bridge from `ln((exp(u))^n)` to `n*u`
- why_this_is_not_a_duplicate:
  - existing `ln(exp(x)) -> x` covers only direct inverse cancellation; this
    representative covers the missing normalization-before-cancellation shape
- discovery_or_promotion:
  - promoted as a new curated derive row and direct planner path
- if_promoted_why_minimal_representative:
  - exponent `2` over one symbol is the smallest case that requires both the
    power-normalization and log-exp cancellation steps
- local_result:
  - added `try_plan_log_exp_power_inverse_target_aware` and a direct derive
    bridge that emits two retained stages
  - added the row `log_exp_inverse_ln_exp_power` to the derive contract and
    simplify/log representative slice
  - reused the focused exponential-power didactic substep generator for
    `Power of a Power`, with visible label `Multiplicar exponentes`
  - CLI probe:
    `derive ln(exp(x)^2), 2*x` reports `Strategy: rewrite exponentials` with
    steps `Power of a Power` and `Log-Exp Inverse`
- guardrails:
  - `cargo fmt`
  - `cargo test -q -p cas_solver log_exp_power -- --nocapture`
  - `cargo test -q -p cas_didactic derive_didactic_log_exp_power_inverse_has_visible_normalization_substep -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_simplify_log_representatives -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `make engine-fast`: `simplify_add_small 435/435`,
    `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=382 unsupported=0 not_equivalent=1`, `generic_simplify_expected=0`,
    derive shadow pressure `50/50`, derive audit `464 cases / 0 flags`,
    `total_web_substeps=477`, simplify audit `14 cases / 0 flags`, simplify
    strict `16518/16518 proved-symbolic`, `0 failed`, `0 timeouts`
  - `git diff --check`
- decision:
  - retained as a focused derive bridge improvement: it increases target-aware
    derive completeness for an existing engine identity family while improving
    the visible step-by-step trace instead of hiding the work in generic
    simplify

## 2026-04-29 - Auto-improvement cycle: derive log-exp product didactic bridge

- investment_class: coverage
- success_condition:
  - `derive ln(exp(x)*exp(y)), x+y` remains on a specific `expand_log`
    strategy, gains a concrete web/CLI substep for the hidden post-expansion
    `ln(exp(u))` cancellations, enters the curated derive corpus, and focused
    tests plus engine guardrails pass
- primary_dimension:
  - didactic quality and coverage for the composed route
    `log expansion -> log-exp inverse`
- secondary_dimension:
  - engine/derive bridge quality for a simplification the engine already knew
    but previously rendered as a single opaque derive move
- hypothesis:
  - the engine expands `ln(e^x*e^y)` to `ln(e^x)+ln(e^y)` and then simplifies
    to `x+y`, but the public derive trace exposed only the parent expand-log
    step; adding a focused substep for the second transition makes the route
    teachable without changing runtime behavior
- relevant_lanes:
  - CLI JSON probe, focused derive didactic audit, log-expanded command eval
    test, derive contract representative slice, generic-simplify guard,
    `make engine-fast`, `make engine-scorecard`
- promotion_target:
  - new minimal row `log_exp_inverse_ln_exp_product`
- derive_bridge_check:
  - retained as path-quality coverage rather than reachability work; the route
    already derived, but now the hidden engine transition is visible and
    guarded
- engine_feedback_check:
  - no reusable runtime change was needed; this confirms the existing engine
    path can support a didactic bridge with a small substep recognizer
- retain_if:
  - strategy remains `expand_log`, web JSON contains one concrete cancellation
    substep after the parent expansion, audit flags stay at `0`, and guardrails
    pass
- reject_if:
  - the substep is pruned as duplicate/noisy, the case falls to generic
    simplify, or global embedded/derive/simplify guardrails regress
- structural_axis:
  - product-log expansion followed by log-exp cancellation
- why_this_is_not_a_duplicate:
  - direct `ln(exp(x)) -> x` and power `ln(exp(x)^2) -> 2*x` do not cover the
    product expansion shape; two exponential factors are the smallest case that
    requires the composed route
- discovery_or_promotion:
  - promoted from a reproducible didactic gap discovered by probing an
    engine-known equivalent pair
- if_promoted_why_minimal_representative:
  - `ln(exp(x)*exp(y)) -> x+y` is the minimal product requiring both log
    expansion and cancellation, without wrapper/noise complexity
- local_result:
  - added a focused `expand_log` didactic substep when the local expansion
    produces a sum/difference of `ln(exp(...))` terms and the final derive
    result cancels them
  - added `log_exp_inverse_ln_exp_product` to `derive_pairs.csv` and the
    simplify/log representative derive contract slice
  - added focused didactic and command-eval regressions
  - JSON probe now shows parent rule `Expandir logaritmos` and substep
    `Cancelar cada logaritmo natural con su exponencial` with
    `ln(e^x)+ln(e^y) -> x+y`
- guardrails:
  - `cargo fmt`
  - `cargo test -q -p cas_didactic --test derive_didactic_audit derive_didactic_log_exp_product_inverse_shows_post_expansion_cancellation -- --exact --nocapture`
  - `cargo test -q -p cas_solver analysis_command_eval_tests::tests::evaluate_derive_command_lines_reaches_tabulated_log_expanded_targets -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_follow_expected_outcomes_simplify_log_representatives -- --exact --nocapture`
  - `cargo test -q -p cas_solver --test derive_contract_tests derive_pairs_do_not_expect_generic_simplify_for_derived_cases -- --exact --nocapture`
  - `make engine-fast`: rerun passed after one transient harness flake;
    `simplify_add_small 435/435`, `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1417/1417`, derive contract
    `derived=383 unsupported=0 not_equivalent=1`, `generic_simplify_expected=0`,
    derive shadow pressure `50/50`, derive audit `465 cases / 0 flags`,
    `total_web_substeps=478`, simplify audit `14 cases / 0 flags`, simplify
    strict `16518/16518 proved-symbolic`, `0 failed`, `0 timeouts`
  - `git diff --check`
- decision:
  - retained as a low-risk didactic bridge improvement: it promotes a minimal
    composed log-exp route and makes an existing engine simplification visible
    without adding runtime search or broad matcher cost

## 2026-04-29 - Auto-improvement cycle: finite aggregate embedded coverage

- investment_class: coverage
- success_condition:
  - promote a minimal `finite_aggregate` slice into
    `embedded_equivalence_context_corpus.csv`, keep the new family balanced at
    the combined-additive target count, and preserve all fast/guardrail lanes
- primary_dimension:
  - embedded contextual equivalence family breadth
- secondary_dimension:
  - reuse of an already-audited engine/derive bridge for finite sums/products
- hypothesis:
  - finite aggregate closed forms already work and are teachable in `derive`,
    but embedded coverage counted only `finite_telescoping`; adding a small
    live slice should increase family breadth without runtime code or search
    risk
- relevant_lanes:
  - candidate smoke for each proposed row, `--family finite_aggregate`
    embedded slice, CLI derive probe, `make engine-fast`, `make
    engine-scorecard`
- promotion_target:
  - `live` embedded equivalence corpus
- derive_bridge_check:
  - no new derive row was needed: `finite_aggregate_sum_first_integers_symbolic`
    and sibling finite aggregate rows already exist in `derive_pairs.csv` and
    the didactic audit remains clean
- engine_feedback_check:
  - no reusable runtime gap was found; this cycle promotes contextual pressure
    for an existing engine capability rather than adding a new rule
- retain_if:
  - `finite_aggregate` passes in isolation, embedded global remains
    `failed=0`, derive guardrails stay green, and combined-additive family
    balance does not regress
- reject_if:
  - any new wrapper fails, the new family remains under the combined-additive
    target, or embedded elapsed regresses materially
- structural_axis:
  - new embedded family plus wrappers
    `additive_passthrough_zero/scaled_difference_zero/common_denominator_zero/shifted_quotient_one/reciprocal_shifted_difference_zero`
    and six `combined_additive_zero` representatives
- why_this_is_not_a_duplicate:
  - previous embedded coverage had finite telescoping sums/products, not
    direct finite aggregate closed forms such as `sum(k,k,1,n)` and
    `product(k,k,1,n)`
- discovery_or_promotion:
  - promoted; the initial six-wrapper probe exposed that the scorecard's
    combined-additive balance would otherwise leave `finite_aggregate` under
    target, so five additional minimal combined-additive rows were added
- if_promoted_why_minimal_representative:
  - `sum(k,k,1,n) -> n*(n+1)/2` is the smallest finite aggregate bridge; the
    sibling combined rows use already-curated closed forms to satisfy the
    existing family-balance metric without introducing a new runtime family
- local_result:
  - added 11 embedded corpus rows for `finite_aggregate`
  - isolated embedded slice: `11/11` passed, `0` failed, `13.12ms`
  - CLI derive probe for `derive(sum(k,k,1,n), n*(n+1)/2)` kept
    `Strategy: finite sums/products` with concrete web substeps
- guardrails:
  - `make engine-fast`: harness unit tests passed, `simplify_add_small
    435/435`, `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1428/1428`, families `24`, elapsed
    `3.88s`, avg `2.717ms/case`, combined-additive
    `collapse_families=24/24`, `families_under_target=0/24`, derive contract
    `derived=383 unsupported=0 not_equivalent=1`, derive shadow `50/50`,
    derive audit `465 cases / 0 flags`, simplify audit `14 cases / 0 flags`,
    simplify strict `16518/16518 proved-symbolic`, `0` failures/timeouts
  - `git diff --check`
- decision:
  - retained as coverage: it expands embedded family breadth from `23` to `24`
    with no semantic failures, keeps the family-balance guardrail green, and
    ties the engine corpus to an already strong derive bridge instead of
    adding duplicate derive cases

## 2026-04-29 - Auto-improvement cycle: number theory embedded coverage

- investment_class: coverage
- success_condition:
  - promote `number_theory` into embedded equivalence corpus with wrapper
    spread plus balanced combined-additive rows; keep fast/guardrail lanes
    green
- primary_dimension:
  - embedded contextual equivalence family breadth
- secondary_dimension:
  - engine/derive bridge reuse for binomial-coefficient identities
- hypothesis:
  - `choose` identities are stable and already didactic in `derive`, but
    embedded coverage did not count them as a contextual family; adding a
    minimal live slice increases family breadth without runtime code
- relevant_lanes:
  - candidate smokes, embedded `--family number_theory`, CLI derive probes,
    `make engine-fast`, `make engine-scorecard`
- promotion_target:
  - `live` embedded equivalence corpus
- derive_bridge_check:
  - no derive row was added because `choose_numeric_binomial_coefficient`,
    `choose_numeric_pascal_identity`, and `choose_numeric_symmetry` already
    exist and audit clean
- engine_feedback_check:
  - no reusable runtime gap was found; smokes confirm existing number-theory
    rewrites survive wrappers
- retain_if:
  - `number_theory` passes in isolation, embedded global remains `failed=0`,
    combined-additive count reaches `6`, and guardrails stay green
- reject_if:
  - any wrapper fails/times out, the new family leaves `families_under_target`
    nonzero, or embedded runtime regresses materially
- structural_axis:
  - new embedded family; wrapper spread; orientation via reversed
    Pascal/symmetry; depth4 via nested-fraction companion
- why_this_is_not_a_duplicate:
  - existing embedded coverage had finite aggregates/telescoping but no
    binomial-coefficient number theory family
- discovery_or_promotion:
  - promoted; all candidate smokes passed before edit
- if_promoted_why_minimal_representative:
  - Pascal is the smallest non-pure numeric `choose` bridge; symmetry and
    numeric coefficient complete the three existing derive-audited number
    theory subcases
- local_result:
  - added 11 embedded rows for `number_theory`
  - isolated slice: `11/11` passed, `8.95ms`
  - CLI derive probes kept `Strategy: number theory` with Pascal/symmetry
    substeps
- guardrails:
  - `make engine-fast`: harness unit tests passed, `simplify_add_small
    435/435`, `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1439/1439`, families `25`, elapsed
    `3.64s`, avg `2.530ms/case`, combined `collapse_families=25/25`,
    `families_under_target=0/25`, derive contract
    `derived=383 unsupported=0 not_equivalent=1`, shadow `50/50`, derive
    audit `465 cases / 0 flags`, simplify audit `14 cases / 0 flags`, strict
    `16518/16518 proved-symbolic`, `0` failures/timeouts
  - `git diff --check`
- decision:
  - retained as coverage: expands embedded breadth from `24` to `25` families
    and reuses existing derive bridges without adding runtime search or
    duplicate derive rows

## 2026-04-29 - Auto-improvement cycle: squared passthrough new-family closure

- investment_class: coverage
- success_condition:
  - close the `squared_passthrough_zero` family-breadth gap introduced by the
    recent `finite_aggregate` and `number_theory` embedded promotions while
    preserving `failed=0`
- primary_dimension:
  - sparse wrapper family breadth in embedded contextual equivalence
- secondary_dimension:
  - reuse existing `derive` bridges for finite sums/products and binomial
    coefficients without adding duplicate derive rows
- hypothesis:
  - both newly promoted families already pass simpler wrappers and combined
    additives; a single minimal squared wrapper per missing family should close
    the structural wrapper gap with negligible runtime cost
- relevant_lanes:
  - one-row candidate smokes, isolated embedded family slices,
    `make engine-fast`, `make engine-scorecard`
- promotion_target:
  - `live` embedded equivalence corpus
- derive_bridge_check:
  - no derive row was added: `finite_aggregate_sum_first_integers_symbolic` and
    `choose_numeric_pascal_identity` already have audited target-aware routes,
    and this cycle only adds contextual squared-wrapper pressure
- engine_feedback_check:
  - no runtime gap found; both smokes passed, so this is retained as coverage
    rather than a rule change
- retain_if:
  - both candidate rows pass, affected families pass in isolation, global
    embedded remains `failed=0`, and `squared_passthrough_zero` no longer has
    missing families
- reject_if:
  - either candidate fails/times out, the global scorecard regresses, or the
    wrapper remains listed as a sparse family gap
- structural_axis:
  - sparse wrapper breadth closure for `squared_passthrough_zero`
- why_this_is_not_a_duplicate:
  - `finite_aggregate` and `number_theory` were covered by other wrappers, but
    had no squared-passthrough representative
- discovery_or_promotion:
  - promoted; both candidates passed one-row smoke before edit
- if_promoted_why_minimal_representative:
  - exactly one row per missing family closes the scorecard gap without adding
    near-duplicate variants
- local_result:
  - added `finite_aggregate_sum_first_integers_symbolic_squared`
  - added `choose_numeric_pascal_identity_squared`
  - candidate smokes: finite aggregate `1/1` passed in `14.80ms` runner,
    number theory `1/1` passed in `4.25ms` runner
  - affected slices: `finite_aggregate 12/12`, `number_theory 12/12`
- guardrails:
  - `make engine-fast`: harness unit tests passed, `simplify_add_small
    435/435`, `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1441/1441`, families `25`, elapsed
    `3.66s`, avg `2.540ms/case`, `squared_passthrough_zero` total `74`
    passed and no longer listed in sparse wrapper gaps, derive contract
    `derived=383 unsupported=0 not_equivalent=1`, shadow `50/50`, derive
    audit `465 cases / 0 flags`, simplify audit `14 cases / 0 flags`, strict
    `16518/16518 proved-symbolic`, `0` failures/timeouts
- decision:
  - retained as coverage: closes a concrete wrapper-family breadth gap with two
    minimal corpus rows and no runtime or derive-code changes

## 2026-04-29 - Auto-improvement cycle: combined additive structural closure

- investment_class: coverage
- success_condition:
  - close the remaining combined-additive structural gaps:
    `depth4_missing=finite_aggregate`,
    `orientation_missing=finite_aggregate`, and
    `multi_core_missing=finite_aggregate,number_theory`
- primary_dimension:
  - `combined_additive_zero` structural coverage across depth, orientation, and
    cross-family composition
- secondary_dimension:
  - contextual pressure for existing engine/derive bridges in finite
    sums/products and binomial-coefficient identities
- hypothesis:
  - the newly promoted `finite_aggregate` and `number_theory` families already
    pass their isolated wrappers; one `three_core` row per missing family can
    close the scorecard axes without new runtime search
- relevant_lanes:
  - one-row candidate smokes, affected embedded family slices,
    `make engine-fast`, `make engine-scorecard`, `make
    engine-scorecard-pressure`
- promotion_target:
  - `live` embedded equivalence corpus
- derive_bridge_check:
  - no derive rows were added: both source families already have audited
    target-aware routes, and this cycle exercises contextual composition rather
    than new source-to-target reachability
- engine_feedback_check:
  - no reusable runtime gap was found; both `three_core` candidates passed
    smoke before edit
- retain_if:
  - affected slices pass, embedded global remains `failed=0`, combined-additive
    `depth4_families`, `orientation_families`, and `multi_core_families` all
    reach `25/25`
- reject_if:
  - either candidate fails/times out, global guardrail regresses, or any of the
    targeted structural axes remains under-covered
- structural_axis:
  - `three_core` combined additive composition, finite-aggregate reversed
    orientation, and depth4 nested-fraction companion
- why_this_is_not_a_duplicate:
  - prior rows covered the two families under wrappers and simple combined
    additives, but did not cover the remaining composition axes
- discovery_or_promotion:
  - promoted; both candidates passed one-row smoke
- if_promoted_why_minimal_representative:
  - exactly two rows close all remaining combined-additive structural gaps:
    one finite aggregate row covers depth4 + orientation + multi-core, and one
    number theory row covers its missing multi-core axis
- local_result:
  - added
    `finite_aggregate_sum_reversed_nested_fraction_depth4_factor_three_core_combined_zero`
  - added
    `choose_numeric_pascal_identity_nested_fraction_factor_three_core_combined_zero`
  - candidate smokes: finite aggregate `1/1` passed in `30.51ms` runner;
    number theory `1/1` passed in `52.99ms` runner
  - affected slices: `finite_aggregate 13/13`, `number_theory 13/13`
- guardrails:
  - `make engine-fast`: harness unit tests passed, `simplify_add_small
    435/435`, `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1443/1443`, families `25`, elapsed
    `3.74s`, avg `2.592ms/case`, combined `depth4_families=25/25`,
    `orientation_families=25/25`, `multi_core_families=25/25`, derive contract
    `derived=383 unsupported=0 not_equivalent=1`, shadow `50/50`, derive
    audit `465 cases / 0 flags`, simplify audit `14 cases / 0 flags`, strict
    `16518/16518 proved-symbolic`, `0` failures/timeouts
  - `make engine-scorecard-pressure`: `simplify_zero_mixed 450/450`,
    `failed=0`, elapsed `205.90ms`
- decision:
  - retained as coverage: all combined-additive structural family axes now
    report `25/25` with no runtime code changes and no derive duplication

## 2026-04-29 - Auto-improvement cycle: reciprocal depth4 new-family closure

- investment_class: coverage
- success_condition:
  - close `reciprocal_shifted_difference_zero` depth4 family breadth from
    `23/25` to `25/25` without failures
- primary_dimension:
  - sparse wrapper shell-depth coverage
- secondary_dimension:
  - contextual pressure for existing finite aggregate and number theory
    engine/derive bridges
- hypothesis:
  - recently promoted `finite_aggregate` and `number_theory` have reciprocal
    rows at depth 3 but not depth 4; adding double-noise reciprocal rows should
    close the depth gap without runtime code
- relevant_lanes:
  - candidate smokes, affected embedded family slices, `make engine-fast`,
    `make engine-scorecard`
- promotion_target:
  - `live` embedded equivalence corpus
- derive_bridge_check:
  - no derive rows added: `finite_aggregate_sum_first_integers_symbolic` and
    `choose_numeric_pascal_identity` already have audited routes; this cycle
    only deepens contextual wrappers
- engine_feedback_check:
  - no runtime gap found; simple-noise candidates passed but remained depth3,
    while double-noise candidates passed at depth4
- retain_if:
  - affected slices pass; embedded global failed=0; reciprocal depth4 family
    breadth reaches `25/25`
- reject_if:
  - candidate fails/times out, depth4 stays under-covered, or embedded runtime
    regresses materially
- structural_axis:
  - `reciprocal_shifted_difference_zero` shell_depth 4 via double fractional
    noise
- why_this_is_not_a_duplicate:
  - prior rows covered these families at reciprocal depth 3, not reciprocal
    depth 4
- discovery_or_promotion:
  - promoted; non-moving simple-noise probes discarded without ledger, depth4
    probes passed
- if_promoted_why_minimal_representative:
  - one depth4 reciprocal row per missing family closes the exact scorecard
    gap
- local_result:
  - added `finite_aggregate_sum_first_integers_symbolic_double_noise_fraction`
  - added `choose_numeric_pascal_identity_double_noise_fraction`
  - depth3 probes passed but were not promoted because they did not move the
    target metric
  - depth4 candidate smokes: finite aggregate `1/1` passed in `4.77ms`
    runner; number theory `1/1` passed in `16.56ms` runner
  - affected slices: `finite_aggregate 14/14`, `number_theory 14/14`
- guardrails:
  - `make engine-fast`: harness unit tests passed, `simplify_add_small
    435/435`, `contextual_strict_fast 64/64`
  - `make engine-scorecard`: embedded `1445/1445`, families `25`, elapsed
    `3.92s`, avg `2.713ms/case`, reciprocal
    `depth4_families=25/25`, derive contract
    `derived=383 unsupported=0 not_equivalent=1`, shadow `50/50`, derive
    audit `465 cases / 0 flags`, simplify audit `14 cases / 0 flags`, strict
    `16518/16518 proved-symbolic`, `0` failures/timeouts
- decision:
  - retained as coverage: closes the last reciprocal depth4 family gap with two
    minimal corpus rows and no runtime/derive-code changes

## 2026-04-29 - Auto-improvement cycle: shell-depth scorecard visibility

- investment_class: observability
- success_condition:
  - the generated scorecard must show all global shell-depth buckets when the
    bucket list is small, including `depth 4` next to `depth 5`
- primary_dimension:
  - embedded scorecard structural-depth visibility
- secondary_dimension:
  - retain clear feedback for future shell-depth promotions without relying only
    on sparse-wrapper subreports
- hypothesis:
  - the previous `first four + max` summary policy hid `depth 4` in a six-depth
    corpus; returning all rows for small bucket lists preserves the important
    signal without making the Markdown noisy
- relevant_lanes:
  - `python3 -m unittest scripts/test_engine_improvement_scorecard.py`
  - `make engine-scorecard`
- promotion_target:
  - none; harness/reporting change only
- derive_bridge_check:
  - not applicable: no mathematical family, runtime rule, or derive route was
    changed
- engine_feedback_check:
  - no engine capability gap found; this was a scorecard reporting blind spot
- retain_if:
  - scorecard unit tests pass; generated Markdown includes `depth 4`; guardrail
    profile remains green
- reject_if:
  - Markdown becomes too noisy, scorecard parsing regresses, or any guardrail
    suite fails
- cohesion_scope:
  - `scripts/engine_improvement_scorecard.py` and its unit test
- behavior_change_expected:
  - output-only observability change; no engine semantic or runtime behavior
    change
- local_result:
  - changed `shell_depth_summary_rows` to return all shell-depth buckets when
    there are at most eight rows
  - added a regression test for a six-bucket `0..5` shape so `depth 4` is not
    hidden
- guardrails:
  - scorecard unit tests: `25/25` passed
  - `make engine-scorecard`: embedded `1445/1445`, families `25`, elapsed
    `4.01s`, avg `2.775ms/case`, shell-depth mix now reports
    `depth 0`, `depth 1`, `depth 2`, `depth 3`, `depth 4`, and `depth 5`,
    derive contract `derived=383 unsupported=0 not_equivalent=1`, shadow
    `50/50`, derive audit `465 cases / 0 flags`, simplify audit
    `14 cases / 0 flags`, strict `16518/16518 proved-symbolic`, `0`
    failures/timeouts
- decision:
  - retained as observability: the scorecard now exposes the structural-depth
    bucket that recent coverage work optimized, with all guardrails green and no
    runtime code touched

