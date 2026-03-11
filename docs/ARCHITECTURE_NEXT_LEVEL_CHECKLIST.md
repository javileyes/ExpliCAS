# Architecture Next-Level Checklist

## Purpose

This checklist converts

- `/Users/javiergimenezmoya/developer/math/docs/ARCHITECTURE_NEXT_LEVEL_PLAN.md`
- `/Users/javiergimenezmoya/developer/math/docs/SOLVER_EVENT_OBSERVER.md`

into an execution list that can be used PR by PR.

It is intentionally strict:

- one architectural goal per PR
- no mixed tracks
- no speculative refactors
- validation gates on every step

## Priority Legend

- `P0`: do now, high value, low ambiguity
- `P1`: do next if `P0` is stable
- `P2`: only after reevaluation
- `R&D`: isolated experiment, not part of the main migration stream

## Global Rules

Premise for this migration stage:

- The software is still pre-release.
- Public API / CLI compatibility is not a constraint by itself.
- Prefer the cleanest boundary and API shape, then update in-repo consumers,
  tests, docs, web, and FFI to match.

Before starting any item:

- [ ] Confirm the item removes real coupling, not just visual disorder.
- [ ] Confirm the item does not require a wide rename-only churn.
- [ ] Confirm the item can be validated independently.

After completing any item:

- [ ] `cargo fmt --all`
- [ ] `cargo check -p cas_solver_core -p cas_engine -p cas_solver -p cas_session -p cas_cli -p cas_didactic`
- [ ] `make ci` if the change is broad enough to justify full validation
- [ ] If behavior changed in simplify/solve: rerun the relevant metamorphic check

## Track 1: High-Value Boundary Cleanup

### P0.1 Review remaining transport-specific naming

Goal:
- identify internal modules whose names still encode transport (`json`) when
  they really build typed Rust models

Checklist:
- [x] Audit `cas_solver/src/json/*` (now `cas_solver/src/wire/*`)
- [x] Audit `cas_didactic/src/eval_json_*`
- [x] Mark each case as:
  - `transport boundary`
  - `typed model builder`
  - `mixed`
- [x] Rename only the clearly mixed cases
- [x] Do not rename boundary modules that truly own wire/DTO assembly

Done when:
- internal naming no longer implies JSON when the module is not actually doing
  JSON transport work

Current progress:
- audited `cas_solver/src/json/*` as transport boundary
  - current tree now lives at `cas_solver/src/wire/*`
  - keep:
    - stateless JSON entry points
    - envelope assembly
    - API mappers
    - substitute/eval command JSON wrappers
- audited `cas_didactic/src/eval_json_*` as mixed typed-model builders
  - renamed to:
    - `step_payloads`
    - `step_payload_render`
  - rationale:
    - they build `StepJson` DTOs and highlighted latex snippets
    - they do not serialize JSON
    - they are internal didactic presentation helpers, not outer transport
- left `cas_solver/src/eval_json_*` open for narrower follow-up review
  - many of those modules are still tightly tied to the `eval-json` command
    boundary and should not be bulk-renamed without splitting by responsibility
- renamed typed finalization helpers in `cas_solver`:
  - `eval_json_finalize_expr` -> `eval_output_finalize_expr`
  - `eval_json_finalize_nonexpr` -> `eval_output_finalize_nonexpr`
  - `eval_json_finalize_input` -> `eval_output_finalize_input`
  - `eval_json_finalize` -> `eval_output_finalize`
  - `eval_json_finalize_wire` -> `eval_output_finalize_wire`
  - internal finalize/build runtime now aliases the wire DTO as
    `EvalOutputWire` / `EvalOutputWireBuild` so `cas_solver` no longer drags
    `EvalJsonOutput*` through local builder/finalizer signatures
  - `eval_command_runtime` and `cas_session::eval_command::session` now use the
    same local `EvalOutputWire` / `EvalCommandResult` aliases instead of
    spelling `EvalJsonOutput` directly in runtime signatures
  - rationale:
    - they assemble typed `EvalJsonOutput` payload/context state
    - they do not perform wire serialization
    - they still build the current `EvalJsonOutput` DTO contract, but their own
      naming is now output-oriented rather than transport-oriented
- narrowed the remaining `cas_solver` mixed area to presentation-only helpers
  - renamed:
    - `eval_json_presentation*` -> `eval_output_presentation*`
  - rationale:
    - they format strings/latex and build typed output DTO fragments
    - they do not own request parsing, option decoding, wire assembly or
      serialization
- also renamed output metadata helpers:
  - `eval_json_stats*` -> `eval_output_stats*`
  - rationale:
    - they compute typed `ExprStatsWire`, truncation metadata and stable hashes
    - they do not perform transport serialization
- renamed the former `eval_json_input` subtree into a real neutral `eval_input`
  module
  - renamed internal request models/builders to:
    - module `eval_json_input` -> `eval_input`
    - test module `eval_json_input_tests` -> `eval_input_tests`
    - `PreparedEvalRequest`
    - `EvalNonSolveAction`
    - `build_prepared_eval_request_for_input(...)`
    - `detect_solve_variable_for_eval_request(...)`
    - module `eval_json_input_variable` -> `eval_input_variable`
    - module `eval_json_input_special` -> `eval_input_special`
    - `parse_solve_input_as_equation_expr(...)` -> `parse_solve_input_for_eval_request(...)`
    - module `eval_json_request_runtime` -> `eval_request_runtime`
  - rationale:
    - the outer `eval-json` command boundary stays in command/finalize modules
    - input/request preparation itself now uses fully neutral naming because it
      only parses/builds typed solver requests
- keep as `eval-json` boundary modules for now:
  - the former `eval_json_command_runtime` was renamed to neutral
    `eval_command_runtime`
    - public entrypoint renamed to `evaluate_eval_with_session(...)`
    - rationale:
      - it prepares and executes typed eval requests against a session
      - it does not perform serialization or wire assembly itself
      - it still consumes/returns `EvalJson*` API model DTOs because those are
        the current outer contract, but the runtime orchestration no longer
        encodes transport in its own naming
  - inside `eval_command_runtime`, private orchestration structs/helpers use
    neutral names (`PreparedEvalRun`, `CollectedEvalArtifacts`,
    `prepare_eval_run`, `finalize_eval_run`, `collect_eval_artifacts`) because
    they do not own wire concerns
  - the former `eval_json_finalize*` was renamed to neutral
    `eval_output_finalize*`
    - private typed payload/build helpers use neutral names
      (`EvalOutputResultPayload`, `build_eval_output`,
      `build_eval_output_wire`, `build_eval_output_wire_value`,
      `finalize_expr_like_eval_output`)
    - the typed build input in `cas_api_models` was also neutralized from
      `EvalJsonOutputBuild` to `EvalOutputBuild`; only the actual wire DTO keeps
      the `EvalJsonOutput` name
    - rationale:
      - they assemble native `EvalJsonOutput` data and wire reply fragments, but
        they do not own JSON transport as a conceptual boundary anymore
  - option-axis mapping moved out of `eval_json_options*` into neutral
    `eval_option_axes*`, because it only maps string axes into typed
    `EvalOptions` and does not own transport/wire concerns
  - the typed DTO crate itself was normalized from `json_types.rs` to
    `wire_types.rs`
    - internal typed models now consistently use `*Wire` naming
      (`EvalWireOutput`, `EvalRunOptions`, `SubstituteWireResponse`,
      `StepWire`, `WarningWire`, `LimitWireResponse`, etc.)
    - rationale:
      - these are stable typed wire models, not transport implementation code
  - CLI/session/android adapters were also narrowed to `wire` naming wherever
    the code is only bridging typed payloads
    - `eval_json`/`envelope_json` internal command bridges -> `eval` /
      `envelope`
    - `eval_json_command*` and envelope contract tests in `cas_session` ->
      `eval_command*` / `envelope_wire_command_tests`
    - internal FFI helpers now use wire-oriented names for fallback payloads
      and parsers
  - `cas_didactic` no longer keeps the old `eval_json_steps` compatibility
    layer
    - only `step_payloads` / `step_payload_render` remain, which matches their
      actual role as typed didactic builders
  - the former root boundary module `cas_solver::json` now also matches that
    naming physically and logically as `cas_solver::wire`
    - directory `cas_solver/src/json` -> `cas_solver/src/wire`
    - stale empty `eval_json_finalize_input/` directory removed
    - solver exports now re-export from `crate::wire::*`
    - rationale:
      - the subtree still owns wire/DTO assembly
      - it no longer implies JSON-specific transport implementation in the
        crate root
  - `eval_request_runtime`
  - `cas_session` also dropped transport naming for the session-backed eval
    runner:
    - module `eval_json_command` -> `eval_command`
    - `EvalJsonCommandConfig` -> `EvalCommandConfig`
    - `evaluate_eval_json_command_with_session(...)` ->
      `evaluate_eval_command_with_session(...)`
    - `evaluate_eval_json_command_pretty_with_session(...)` ->
      `evaluate_eval_command_pretty_with_session(...)`
    - rationale:
      - it only orchestrates session load/run/save around typed eval requests
      - the actual wire boundary remains in the CLI/web entrypoints and API DTOs
      - after the pre-release CLI cleanup, the public CLI path is
        `eval ... --format json`, not a separate `eval-json` command
    - the session runtime now also aliases the wire DTO locally as
      `EvalOutputWire`, instead of spelling `EvalJsonOutput` directly through
      `eval_command::session`
  - `cas_android_ffi` also neutralized its JNI-internal helper names:
    - `eval_json_core(...)` -> `eval_core(...)`
    - `substitute_json_core(...)` -> `substitute_core(...)`
    - rationale:
      - the exported JNI entry points were aligned too:
        `evalJson` / `substituteJson` -> `evalWire` / `substituteWire`
      - the inner Rust helpers are direct bridge cores around solver-level
        stateless wire entrypoints and do not own transport naming
    - boundary-facing tests/helpers now also prefer `wire` naming when they
      validate output contracts rather than the transport mechanism itself
  - residual helper/test naming was also cleaned where the responsibility is
    already neutral:
    - `eval_input_tests` now uses `build_prepared_eval_request_*` test names
    - CLI local helper `eval_json_command_config(...)` ->
      `eval_command_config(...)`
    - the envelope wire helper was neutralized:
      - `evaluate_envelope_json_command(...)` ->
        `evaluate_envelope_wire_command(...)`
      - session-side test module renamed consistently:
        `envelope_json_command_tests` -> `envelope_wire_command_tests`
      - solver-side bridge contract tests were aligned too:
        `json_bridge_tests` -> `wire_bridge_tests`
    - substitute canonical wrappers/tests were aligned too:
      - module `substitute_subcommand_json` -> `substitute_subcommand_wire`
      - `evaluate_substitute_subcommand_json_canonical(...)` ->
        `evaluate_substitute_subcommand_wire(...)`
      - `parse_substitute_json_text_lines(...)` ->
        `parse_substitute_wire_text_lines(...)`
      - integration test `substitute_json_contract_tests` ->
        `substitute_wire_contract_tests`
      - substitute wire DTOs were aligned too:
        `EngineJsonSubstep` -> `EngineWireSubstep`,
        `SubstituteOptionsJson` -> `SubstituteWireOptions`,
        `SubstituteJsonResponse` -> `SubstituteWireResponse`
    - eval/substitute internal renderers were aligned too:
      - `build_engine_json_steps(...)` -> `build_engine_wire_steps(...)`
      - `substitute_str_to_json_impl(...)` ->
        `substitute_str_to_wire_impl(...)`
      - `SubstituteParseIssue::to_json_error(...)` ->
        `to_wire_error(...)`
    - CLI/public command surface was simplified too:
      - module `eval_json` -> `eval`, with wire rendering folded into the main
        `eval` command module
      - hidden legacy alias `eval-json` was removed
      - public wire path is now `eval ... --format json`
      - hidden legacy alias `envelope-json` was removed
      - public envelope command is now `envelope`
    - the envelope/CLI bridge and subcommand toggles were aligned too:
      - module `envelope_json` -> `envelope`
      - `EnvelopeJsonArgs` -> `EnvelopeArgs`
      - internal `json_output` flags in limit/substitute subcommand bridges ->
        `wire_output`
    - `cas_session` no longer fronts stateless wire passthroughs:
      - `eval_str_to_wire`, `substitute_str_to_wire`,
        `evaluate_envelope_wire_command`, `evaluate_limit_subcommand`,
        `evaluate_substitute_subcommand` are now consumed directly from
        `cas_solver` by CLI/FFI
      - rationale:
        - `cas_session` should own session/repl/stateful orchestration
        - stateless wire entrypoints belong to the solver facade
    - internal parity/contract tests were aligned too where they verify the
      wire layer rather than JSON serialization itself:
      - former `step_count_matches_between_text_and_json_renderers` ->
        `step_count_matches_between_text_and_wire_renderers`
      - `evaluate_limit_subcommand_output_json_mode_returns_payload` ->
        `evaluate_limit_subcommand_output_wire_mode_returns_payload`
      - integration test `json_contract_tests` ->
        `wire_contract_tests`
      - CLI wire/domain contract suites now use `parse_wire(...)` and `wire`
        locals instead of `parse_json(...)` / `json`
      - stale witness-survival expectations in `cli_wire_contract_tests` were
        updated to the current wire contract, which now surfaces denominator
        and nonnegative requirements for `(x-y)/(sqrt(x)-sqrt(y))`
    - residual `eval-json` naming now only remains in historical notes/docs or
      in true transport names like `to_json*` serializers and `opts_json`
    - engine wire DTO naming was aligned too:
      - `EngineJsonWarning` -> `EngineWireWarning`
      - `EngineJsonStep` -> `EngineWireStep`
      - `EngineJsonError` -> `EngineWireError`
      - `EngineJsonResponse` -> `EngineWireResponse`
    - remaining leaf wire DTOs were aligned too:
      - `ErrorJsonOutput` -> `ErrorWireOutput`
      - `LimitJsonResponse` -> `LimitWireResponse`
    - the main typed eval output model was aligned too:
      - `EvalJsonOutput` -> `EvalWireOutput`
    - canonical wire entrypoints and CLI-facing payload variants were aligned
      too:
      - `eval_str_to_json(...)` -> `eval_str_to_wire(...)`
      - `substitute_str_to_json(...)` -> `substitute_str_to_wire(...)`
      - `limit_str_to_json(...)` -> `limit_str_to_wire(...)`
      - `SubstituteSubcommandOutput::Json` ->
        `SubstituteSubcommandOutput::Wire`
      - `LimitSubcommandEvalOutput::Json` ->
        `LimitSubcommandEvalOutput::Wire`
      - `LimitSubcommandOutput::Json` -> `LimitSubcommandOutput::Wire`
      - lint/tooling script names were aligned too:
        `lint_eval_json_canonical.sh` ->
        `lint_eval_wire_entrypoint.sh`
        `lint_substitute_json_canonical.sh` ->
        `lint_substitute_wire_entrypoint.sh`
    - eval metadata/support DTOs were aligned too:
      - `ExprStatsJson` -> `ExprStatsWire`
      - `TimingsJson` -> `TimingsWire`
      - `DomainJson` -> `DomainWire`
      - `OptionsJson` -> `OptionsWire`
      - `SemanticsJson` -> `SemanticsWire`
      - `BudgetJsonInfo` / `BudgetExceededJson` ->
        `BudgetWireInfo` / `BudgetExceededWire`
      - `SpanJson` -> `SpanWire`
    - script/benchmark wire models were aligned too:
      - `ScriptJsonOutput` -> `ScriptWireOutput`
      - `MmGcdModpJsonOutput` -> `MmGcdModpWireOutput`
    - the `cas_api_models` root file was aligned too:
      - `src/json_types.rs` -> `src/wire_types.rs`
    - eval/solve step DTOs were aligned too:
      - `WarningJson` -> `WarningWire`
      - `RequiredConditionJson` -> `RequiredConditionWire`
      - `StepJson` / `SubStepJson` -> `StepWire` / `SubStepWire`
      - `SolveStepJson` / `SolveSubStepJson` ->
        `SolveStepWire` / `SolveSubStepWire`
    - `cas_didactic` no longer carries the transitional `eval_json_steps`
      wrapper module or the former `collect_eval_json_steps*` compatibility exports;
      callers now use `collect_step_payloads*` directly
    - inside the `cas_solver::json::eval` boundary, the private prep helper now
      uses neutral naming too:
      - `PreparedEvalRequestState` -> `PreparedStatelessEvalState`
      - `prepare_eval_json_request(...)` ->
        `prepare_stateless_eval_request(...)`
      - local test `eval_json_session_ref_returns_invalid_input` (former name) ->
        `eval_session_ref_returns_invalid_input`
    - inside `cas_solver::eval_command_runtime`, DTO leaks are now aliased to
      runtime-neutral names:
      - `EvalSessionRunConfig` -> `EvalCommandRunConfig`
      - `EvalJsonOutput` -> `EvalCommandOutput`
      - rationale:
        - the runtime still consumes API DTOs at the boundary
        - but its own orchestration signatures now read as session/eval runtime,
          not transport-specific code
    - typed command DTOs in `cas_api_models` also dropped the transport prefix
      where they are not wire payloads:
      - `EvalJsonLimitApproach` -> `EvalLimitApproach`
      - `EvalJsonSpecialCommand` -> `EvalSpecialCommand`
      - `parse_eval_json_special_command(...)` ->
        `parse_eval_special_command(...)`
      - `EvalJsonSessionRunConfig` -> `EvalSessionRunConfig`
      - `JsonRunOptions` -> `EvalRunOptions`
      - `SubstituteJsonOptions` -> `SubstituteRunOptions`

### P0.2 Review `cas_session_core`

Goal:
- decide whether `cas_session_core` should remain or merge back into
  `cas_session`

Checklist:
- [x] Map current responsibilities of `cas_session_core`
- [x] Check whether merging reintroduces dependency cycles
- [x] Check whether merge reduces indirection
- [x] If no, explicitly document why it stays

Done when:
- `cas_session_core` is either justified as permanent or removed

Decision:
- `cas_session_core` stays.

Why:
- `cas_engine` uses its stateless/session-agnostic eval contracts.
- `cas_solver` uses its low-level store, snapshot and entry-id types.
- `cas_session` already depends on both `cas_engine` and `cas_solver`.

Merging it back into `cas_session` would worsen the graph, not simplify it.

### P0.3 Review remaining `cas_math` leaks

Goal:
- only continue if a remaining candidate is clearly not math-pure

Checklist:
- [x] Audit only modules still suspected to be rule-facing or user-facing
- [x] For each candidate, answer:
  - is this deterministic reusable math?
  - does it mainly exist for one engine rule?
  - does it contain `desc`/policy/runtime packaging?
- [x] Move or clean up only the clearly engine-facing candidates
- [x] Stop as soon as remaining candidates become debatable

Done when:
- no obvious engine-facing helper remains in `cas_math`

Current progress:
- moved shared runtime `step_*` helpers (`step_rules`, `step_optimize`,
  `step_absorption`, `step_productivity`, `step_semantic`) out of `cas_math`
  into `cas_solver_core`
- rationale:
  - they operate on step/rule heuristics, not reusable algebra
  - they already served runtime orchestration in `cas_engine` /
    `cas_solver_core`
  - leaving them in `cas_math` blurred the boundary between pure math and
    didactic/runtime cleanup
- moved `rationalize_policy` out of `cas_math` into `cas_solver_core`
  - rationale:
    - it is shared simplification configuration and outcome reporting
    - it is consumed by `cas_solver_core`, `cas_engine` and `cas_solver`
    - it does not implement algebraic transformation logic itself
- moved `undefined_risk_policy_support` out of `cas_math` into `cas_solver_core`
  - rationale:
    - it is domain-policy gating for cancellation rewrites
    - it does not perform symbolic algebra
    - it was only serving runtime rule decisions in `cas_engine`
- spot-checked the remaining suspicious families and stopped there:
  - `fraction_univar_gcd_support` remains structural polynomial reduction with
    domain policy intentionally delegated upward
  - `trig_dyadic_policy_support` remains a math/domain gate, not runtime
    packaging
  - `poly_gcd_dispatch` remains a shared algorithm/mode selector with injected
    callbacks, not solver-session wiring
  - `undefined_risk_support` remains pure structural detection with caller
    provided proof oracle
- remaining suspects in `cas_math` are now the more debatable
  `*_support` families with mixed policy/description concerns; review should
  stop again before broad churn

## Track 2: Solve Observer/Event Phase 1

### P1.1 Add minimal `SolverEvent` model

Goal:
- create the smallest event model that is equation-aware

Checklist:
- [x] Add `cas_solver_core/src/solver_events.rs`
- [x] Add `SolverEvent<Equation, Importance>`
- [x] Add `SolveEventListener`
- [x] Add `cas_solver_core/src/solver_event_collector.rs`
- [x] Register both in `cas_solver_core/src/lib.rs`

Done when:
- solve events exist as a stable core abstraction with no solver runtime
  rewiring yet

### P1.2 Add `SolveStep <-> SolverEvent` adapters

Goal:
- prove the event model is lossless before changing any consumer

Checklist:
- [x] Add adapter module in `cas_solver`
- [x] Convert `Vec<SolveStep>` to `Vec<SolverEvent>`
- [x] Convert `Vec<SolverEvent>` back to `Vec<SolveStep>`
- [x] Add roundtrip tests

Done when:
- `SolveStep -> events -> SolveStep` preserves:
  - description
  - equation
  - importance
  - substeps

### P1.3 Integrate one consumer: `timeline_solve`

Goal:
- test whether solve events simplify a real consumer

Checklist:
- [x] Integrate event stream into `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/timeline_solve_eval.rs`
- [x] Keep public API unchanged
- [x] Keep old path available while verifying parity
- [x] Validate HTML/output parity in `cas_didactic`

Done when:
- `timeline_solve` can consume solve events without changing user-visible output

Current status:
- implemented as a roundtrip `DisplaySolveSteps -> SolverEvent -> DisplaySolveSteps`
- guarded by shape-preserving fallback to the original `DisplaySolveSteps`
- HTML + CLI parity coverage added in `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/solve_timeline_parity_tests.rs`
- validated with `cargo test -p cas_didactic --test solve_timeline_parity_tests`

### P1.4 Stop-and-evaluate checkpoint

Checklist:
- [x] Confirm the event consumer did not become meaningfully simpler
- [x] Confirm the coupling reduction is only an adapter seam, not a new boundary
- [x] Confirm the experiment avoided fragile solve runtime paths

Decision:
- [ ] Open a design step for native emission
- [x] Stop here and keep the adapter layer only

Current decision:
- keep the derived `SolveStep <-> SolverEvent` adapter layer
- do not open native solve event emission yet
- revisit only if a concrete consumer shows clear payoff beyond `timeline_solve`

Why:
- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/timeline_solve_eval.rs`
  still solves into `DisplaySolveSteps` first and only then roundtrips through
  `SolverEvent`
- `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/timeline/render_api.rs`
  still renders `&[SolveStep]`, so the didactic boundary did not get narrower
- the event model is useful as a stable core contract and test seam, but not
  enough to justify native solve-runtime emission work

## Track 3: Optional Solve Observer/Event Phase 2

### P2.1 Native emission design only

Goal:
- design, not implement blindly

Checklist:
- [ ] Identify stable emission boundaries in:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver_core/src/strategy_kernels.rs`
- [ ] Define which events are emitted:
  - main step
  - substep
- [ ] Explicitly exclude:
  - failed attempts
  - branch-internal noise
  - diagnostics deltas

Done when:
- there is a short design note for native solve event emission

### P2.2 Native emission implementation

Goal:
- only if P2.1 proves it is worth it

Checklist:
- [ ] Emit events at strategy boundaries
- [ ] Keep `SolveStep` compatibility
- [ ] Validate parity against current `timeline_solve`

Done when:
- native events can replace derived ones without regressions

## Track 4: Measured Memory Work

### R&D.1 Symbol storage benchmark

Goal:
- measure whether string storage is worth changing

Checklist:
- [ ] Choose benchmark workloads
- [ ] Measure current allocation behavior
- [ ] Prototype `smol_str` or equivalent
- [ ] Compare memory + runtime

Done when:
- there is benchmark data and a go/no-go recommendation

### R&D.2 AST sharing experiment

Goal:
- test `Rc`/arena ideas without contaminating mainline work

Checklist:
- [ ] Isolate prototype branch
- [ ] Measure clone-heavy workloads
- [ ] Compare ergonomics and regressions

Done when:
- there is a data-backed recommendation, not an intuition

## Track 5: E-Graph Exploration

### R&D.3 `egg` prototype

Goal:
- assess whether one rule family benefits from equality saturation

Checklist:
- [ ] Choose one narrow target:
  - trig identities
  - fraction normalization
  - rational canonicalization
- [ ] Prototype outside the main migration stream
- [ ] Define extraction cost function
- [ ] Compare:
  - correctness
  - complexity
  - maintainability

Done when:
- there is a prototype note and a decision whether to proceed

## Suggested PR Order

### PR 1
- [x] `P0.1` transport naming audit/fixes, if any clear wins remain

### PR 2
- [x] `P0.2` `cas_session_core` review and decision

### PR 3
- [x] `P0.3` final `cas_math` leak review, only for clearly engine-facing cases

### PR 4
- [x] `P1.1` add `SolverEvent` core model

### PR 5
- [x] `P1.2` add `SolveStep <-> SolverEvent` adapters

### PR 6
- [x] `P1.3` integrate `timeline_solve` and validate renderer parity

### PR 7
- [x] `P1.4` stop-and-evaluate checkpoint

Only after PR 7:
- [x] decide whether to open `P2`
- [ ] or shift to memory / `egg` exploration

## Explicit Do-Not-Do List

- [ ] Do not rename `cas_solver`/`cas_solver_core` in the same stream as event work.
- [ ] Do not adopt `egg` in the mainline without a prototype branch.
- [ ] Do not introduce `Rc<Expr>` or arena ownership changes without benchmarks.
- [ ] Do not keep moving tiny helpers one by one once the next candidate is debatable.
- [ ] Do not expand the solve event model beyond step/substep in Phase 1.

## Exit Criteria

We stop this next-level architecture effort when:

- [x] no obvious domain leaks remain
- [x] solve-event Phase 1 has a clear yes/no outcome
- [x] any further work would require speculative redesign rather than bounded improvement

At that point, the next step is not "more migration".
It is either:

- a dedicated performance program
- a dedicated solver-event architecture program
- or no further architecture change at all
