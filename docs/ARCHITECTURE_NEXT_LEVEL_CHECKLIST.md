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
- `cas_solver` root also stopped owning stateless CLI-oriented command
  entrypoints implicitly
  - `limit` and `substitute` subcommand APIs now live under
    `cas_solver::command_api::{limit, substitute}`
  - CLI consumers were migrated to those owners instead of the crate root
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
    - source-internal didactic/timeline runtime now resolves `Step`,
      `DisplayEvalSteps`, `Engine`, `EvalSession` and related eval types through
      a local bridge backed by `cas_engine` / `cas_solver_core`
      - `cas_solver` remains only for solver-specific helpers such as
        `reconstruct_global_expr(...)` and grouped assumption-line formatting
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
  - `cas_session` root no longer acts as an implicit façade for solver-facing
    stateless helpers
  - removed the redundant `simplifier_setup_types` bridge in `cas_solver`
    - `SimplifierRuleConfig` / `SimplifierToggleConfig` now come directly from
      `cas_solver_core::simplifier_config`
    - `set_simplifier_toggle_rule(...)` now belongs to
      `simplifier_setup_toggle`, which is the actual owner of toggle
      application/update behavior
    - rationale:
      - the old module only duplicated core types plus a single local helper
      - deleting it removes another fake ownership layer from `cas_solver`
  - removed the redundant `substitute_subcommand_types` bridge in `cas_solver`
    - `SubstituteCommandMode` / `SubstituteSubcommandOutput` now come directly
      from `cas_solver_core::substitute_command_types`
    - `command_api::substitute` reexports them from the owner real instead of a
      local pass-through module
  - removed the redundant `limit_command_types` bridge in `cas_solver`
    - `LimitCommandInput` / `Limit*Eval*` types now come directly from
      `cas_solver_core::limit_command_types`
    - `command_api::limit`, `limit_command_core`, and `limit_command_parse`
      now point to the owner real instead of a local pass-through module
  - removed the redundant `analysis_command_types` bridge in `cas_solver`
    - analysis/equiv/visualize-facing error and output types now come directly
      from `cas_solver_core::analysis_command_types`
    - the command façade now reexports them from the owner real in
      [`lib.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/lib.rs)
      instead of a local pass-through module
  - removed the redundant `health_command_types` bridge in `cas_solver`
    - `HealthCommandInput` / `HealthCommandEvalOutput` / `HealthStatusInput`
      now come directly from `cas_solver_core::health_runtime`
    - the command façade in
      [`lib.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/lib.rs)
      and the local health runtime/format/parse modules now point to the owner
      real instead of a local pass-through
  - removed the redundant `steps_command_types` bridge in `cas_solver`
    - `StepsCommand*` / `StepsDisplayMode` now come directly from
      `cas_solver_core::steps_command_types`
    - both `session_api::settings` and `exports_base::settings::steps` now
      reexport them from the owner real instead of a local pass-through
  - removed the redundant `set_command_types` bridge in `cas_solver`
    - `SetCommand*` / `SetDisplayMode` now come directly from
      `cas_solver_core::set_command_types`
    - both `session_api::settings` and `exports_base::settings::set_command`
      now reexport them from the owner real instead of a local pass-through
  - removed the redundant `history_types` / `inspect_types` bridges in
    `cas_solver`
    - history/inspect-facing models now come directly from
      `cas_solver_core::history_models`
    - the command façade in
      [`lib.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/lib.rs)
      and local history formatting now point to the owner real instead of
      local pass-through modules
    - `pub use solver_exports::*;` was removed from the crate root
    - `cas_session::solver_exports` was removed entirely
    - consumers now use the explicit owner directly:
      - `cas_solver::session_api::runtime::*`
      - `cas_solver::session_api::settings::*`
      - `cas_solver::session_api::simplifier::*`
      - `cas_solver::session_api::assumptions::*`
      - `cas_solver::session_api::solve::*`
      - `cas_solver::session_api::eval::*`
    - `cas_solver::session_api::formatting` was removed entirely
      - analysis-facing parsing/formatting moved to
        `cas_solver::session_api::analysis::*`
      - semantics-facing view/help formatting moved to
        `cas_solver::session_api::settings::*`
      - parse-error rendering moved to `cas_solver::session_api::repl::*`
    - internal and integration tests were migrated from
      `crate::solver_exports::...` to `cas_solver::session_api::...`
    - solver-facing stateless eval helpers/types now live under
      `cas_solver::session_api::*`
    - `cas_didactic` no longer consumes stateless session-facing APIs through
      `cas_session`, and its production dependency on `cas_session` was dropped
      in favor of direct `cas_solver::session_api::*` consumption
    - rationale:
      - `cas_session` root stays focused on session/repl/stateful flows
      - solver session-facing passthroughs remain available, but only from the
        crate that actually owns them
  - `cas_session` root also stopped serving as a grab-bag namespace for
    cache/resolution internals
    - lower-level surfaces now live under explicit namespaces:
      - `cas_session::cache::{...}`
      - `cas_session::resolve_refs::{...}`
    - integration tests were migrated away from `use cas_session::*`
      toward `cas_session::cache`, `cas_session::resolve_refs`, and
      `cas_session_core::{store,types}` where those types truly belong
    - rationale:
      - the root keeps session-facing concepts (`SessionState`, `ReplCore`,
        session eval flows, config)
      - lower-level store/cache/ref-resolution APIs remain available, but
        behind names that reflect ownership instead of leaking through the
        crate root
  - `cas_solver` root also stopped re-exporting direct stateless wire entrypoints
    from the crate root
    - `cas_solver::wire` is now the explicit namespace for:
      - `eval_str_to_wire(...)`
      - `substitute_str_to_wire(...)`
      - `evaluate_envelope_wire_command(...)`
    - CLI, Android FFI, and contract tests were migrated to consume
      `cas_solver::wire::*` directly
    - rationale:
      - the solver root keeps broad math/runtime ownership
      - transport/wire entrypoints are still public, but now live behind a
        boundary namespace instead of appearing as generic root APIs
  - `cas_solver` root also stopped acting as the implicit public namespace for
    engine/runtime-facing types
    - `cas_solver::runtime` is now the explicit public namespace for:
      - `Engine`, `Simplifier`, `EvalOptions`, `EvalRequest`, `EvalOutput`,
        `EvalResult`, `EvalAction`
      - `DisplayEvalSteps`, `Step`, `ImportanceLevel`
      - `DisplaySolveSteps`, `SolveDiagnostics`, `SolveStep`, `SolveSubStep`,
        `SolverOptions`, `StatelessEvalSession`
      - semantic/runtime config types such as `EvalConfig`,
        `SharedSemanticConfig`, `SimplifyOptions`, `BranchMode`,
        `ContextMode`, `ComplexMode`, `StepsMode`, `ExpandPolicy`,
        `InverseTrigPolicy`
      - runtime rule namespace via `cas_solver::runtime::rules::*`
    - workspace consumers/tests were migrated from `cas_solver::*` to
      `cas_solver::runtime::*`
    - full integration test compilation now also passes with the new ownership:
      `cargo test -p cas_solver --tests --no-run`
    - the remaining legitimate root-facing items are façade/domain APIs such as
      `solve(...)`, `solve_with_display_steps(...)`, `Proof`, and
      `VerifySummary`, not technical runtime plumbing
    - the old duplicate internal layer `exports_base::runtime` was removed;
      internal solver code now also uses the same `crate::runtime::*` seam
    - rationale:
      - the solver root stays focused on façade-level math/wire entrypoints
      - runtime types still exist publicly, but now live behind a namespace
        that makes ownership explicit instead of leaking through the crate root
  - `cas_solver` root also stopped re-exporting session/stateful runtime
    adapters that conceptually belong to `session_api::runtime` or `runtime`
    - `cas_solver::session_api::runtime` is now the explicit owner for:
      - `evaluate_eval_with_session(...)`
      - `build_runtime_with_config(...)`
      - `reset_runtime_with_config(...)`
      - `reset_runtime_full_with_config(...)`
      - `evaluate_and_apply_config_command_on_runtime(...)`
      - `evaluate_context_command_with_config_sync_on_runtime(...)`
      - `evaluate_autoexpand_command_with_config_sync_on_runtime(...)`
      - `evaluate_semantics_command_with_config_sync_on_runtime(...)`
      - runtime context traits such as
        `ReplConfiguredRuntimeContext`,
        `ReplEvalRuntimeContext`,
        `ReplRuntimeStateContext`,
        `ReplSimplifierRuntimeContext`
    - `cas_solver::runtime` is now the explicit owner for eval-output/runtime
      adapters such as:
      - `required_conditions_from_eval_output(...)`
      - `to_display_steps(...)`
    - `cas_session` and `cas_didactic` consumers were migrated accordingly,
      and a workspace grep no longer finds these helpers used via
      `cas_solver::*` root
    - rationale:
      - the root keeps façade/domain APIs (`solve`, `verify`, etc.)
      - stateful/session runtime helpers and eval-output adapters remain
        public, but only from the namespaces that truly own them
  - remaining root leaks around session/config/history ownership were also
    closed
    - `cas_solver::session_api::simplifier` is now the owner for
      `evaluate_unary_command_message_on_runtime(...)`
    - `cas_solver::session_api::session_support` is now the owner for
      `InspectHistoryContext`
    - `cas_solver::session_api::simplifier` is the owner consumed by `cas_session`
      for `apply_simplifier_toggle_config(...)`
    - after these migrations, a workspace grep outside `cas_solver` no longer
      finds non-legitimate `cas_solver::*` root imports beyond:
      - `api::*`
      - `runtime::*`
      - `wire::*`
      - `session_api::*`
      - `command_api::*`
      - `math::*`
      - façade/domain APIs such as `solve(...)`, `verify_*`, `expand(...)`
    - rationale:
      - the root is no longer acting as a technical grab-bag namespace
      - the remaining root-facing APIs are intentional façade/domain entrypoints
  - façade/domain solver APIs also now have an explicit owner namespace
    - `cas_solver::api` is now the intended owner for:
      - `solve(...)`
      - `solve_with_display_steps(...)`
      - `expand(...)`
      - `telescope(...)`
      - `prepare_timeline_solve_equation(...)`
      - domain/proof-facing types such as
        `ImplicitCondition`, `RequireOrigin`, `Proof`,
        `VerifyResult`, `VerifyStatus`, `VerifySummary`
    - workspace consumers and `cas_solver` integration tests were migrated to
      `cas_solver::api::*`
    - the root no longer re-exports `solve`, `solve_with_display_steps`,
      `expand`, or `telescope`
    - rationale:
      - façade/domain entrypoints stay public, but behind a namespace that
        makes their ownership explicit instead of mixing them with every other
        root-level export
  - remaining session-facing runtime/option traits were also moved behind the
    explicit owner namespaces
    - `cas_solver::session_api::health` now owns the health-facing surface:
      - `ReplHealthRuntimeContext`
      - `evaluate_health_command_message_on_repl_core(...)`
      - `update_health_report_on_repl_core(...)`
      - `HealthCommandInput`
      - `HealthStatusInput`
      - `HealthCommandEvalOutput`
      - `HealthSuiteCategory`
      - health formatting/messages/catalog/report helpers
    - `cas_solver::session_api::history` now owns the history/inspect-facing surface:
      - `evaluate_history_command_message_on_repl_core(...)`
      - `evaluate_show_command_lines_on_repl_core(...)`
      - `evaluate_delete_history_command_message_on_repl_core(...)`
      - `evaluate_history_command_lines(...)`
      - `evaluate_history_command_lines_with_context(...)`
      - `delete_history_entries(...)`
      - `history_overview_entries(...)`
      - `InspectHistoryContext`
      - history/inspect parse/format helpers
      - `DeleteHistory*`, `HistoryOverview*`, `HistoryEntry*`,
        `InspectHistoryEntryInputError`, `ParseHistoryEntryIdError`
    - `cas_solver::session_api::config` now owns the config-command surface:
      - `evaluate_and_apply_config_command(...)`
      - `evaluate_and_apply_config_command_on_runtime(...)`
      - `evaluate_config_command(...)`
      - `parse_config_command_input(...)`
      - `config_*_usage_message(...)`
      - `ConfigCommandInput`, `ConfigCommandResult`, `ConfigCommandApplyOutput`
    - `cas_solver::session_api::bindings` now owns the assignment/bindings-facing
      surface:
      - `apply_assignment(...)`
      - `evaluate_assignment_command_message_on_repl_core(...)`
      - `evaluate_let_assignment_command_message_on_repl_core(...)`
      - `evaluate_vars_command_message_on_repl_core(...)`
      - `evaluate_clear_command_lines_on_repl_core(...)`
      - `evaluate_assignment_command(...)`
      - `evaluate_let_assignment_command(...)`
      - `evaluate_vars_command_lines(...)`
      - `evaluate_vars_command_lines_with_context(...)`
      - `clear_bindings_command(...)`
      - assignment/bindings parse/format helpers
      - `AssignmentCommandOutput`, `AssignmentError`, `LetAssignmentParseError`,
        `ParsedLetAssignment`, `BindingOverviewEntry`, `ClearBindingsResult`
    - `cas_solver::session_api::budget` now owns the solve-budget-facing
      surface:
      - `apply_solve_budget_command(...)`
      - `evaluate_solve_budget_command_message(...)`
      - `evaluate_solve_budget_command_message_on_repl_core(...)`
      - `format_solve_budget_command_message(...)`
      - `SolveBudgetCommandResult`
    - `cas_solver::session_api::set` now owns the `set`-facing surface:
      - `evaluate_set_command_on_repl_core(...)`
      - `set_command_state_for_repl_core(...)`
      - `apply_set_command_plan_on_repl_core(...)`
      - `ReplSetRuntimeContext`
      - `ReplSetCommandOutput`, `ReplSetMessageKind`
      - `Set*` command/state/apply types
    - `cas_solver::session_api::settings` now owns the settings-facing
      surface:
      - `evaluate_steps_command_input(...)`
      - `steps_command_state_for_repl_core(...)`
      - `apply_steps_command_update_on_repl_core(...)`
      - `evaluate_context_*`
      - `evaluate_autoexpand_*`
      - `evaluate_semantics_*`
      - `evaluate_*_with_config_sync_on_runtime(...)`
      - `ReplStepsRuntimeContext`, `ReplSemanticsRuntimeContext`
      - `Steps*`, `Context*`, `Autoexpand*`, `SemanticsPreset*`,
        `SemanticsSet*` command/state/apply types
    - `cas_solver::session_api::eval` now owns the eval-facing session
      surface:
      - `evaluate_eval_with_session(...)`
      - `evaluate_eval_command_output(...)`
      - `evaluate_eval_text_simplify_with_session(...)`
      - `evaluate_eval_command_render_plan_on_repl_core(...)`
      - `evaluate_expand_command_render_plan_on_repl_core(...)`
      - `EvalCommandError`, `EvalCommandOutput`, `EvalCommandRenderPlan`
    - `cas_solver::session_api::lifecycle` now owns the REPL/session
      lifecycle-facing surface:
      - `build_runtime_with_config(...)`
      - `reset_runtime_with_config(...)`
      - `reset_runtime_full_with_config(...)`
      - `build_repl_prompt(...)`
      - `eval_options_from_repl_core(...)`
      - `clear_repl_profile_cache(...)`
      - `reset_repl_runtime_state(...)`
    - `cas_solver::session_api::profile` now owns the profile/profile-cache
      surface:
      - `evaluate_profile_cache_command_lines_on_repl_core(...)`
      - `evaluate_profile_command_message_on_repl_core(...)`
      - `apply_profile_command_on_repl_core(...)`
      - profile/profile-cache parse/apply/format helpers
      - `ProfileCacheCommandResult`, `ProfileCommandInput`, `ProfileCommandResult`
    - `cas_solver::session_api::assumptions` now owns the assumptions-facing
      surface:
      - `format_assumption_records_summary(...)`
      - `filter_blocked_hints_for_eval(...)`
      - `format_eval_blocked_hints_lines(...)`
      - `format_solve_assumption_and_blocked_sections(...)`
      - `format_blocked_hint_lines(...)`
      - `format_diagnostics_requires_lines(...)`
      - `format_domain_warning_lines(...)`
      - `format_normalized_condition_lines(...)`
      - `format_required_condition_lines(...)`
      - `collect_assumed_conditions_from_steps(...)`
      - `format_assumed_conditions_report_lines(...)`
      - `group_assumed_conditions_by_rule(...)`
      - `SolveAssumptionSectionConfig`
    - `cas_solver::session_api::analysis` now owns the analysis-facing
      surface:
      - `evaluate_equiv_*`
      - `evaluate_explain_*`
      - `evaluate_visualize_*`
      - `format_explain_command_error_message(...)`
      - `format_visualize_command_error_message(...)`
      - `ExplainCommandEvalError`, `ExplainGcdEvalOutput`,
        `VisualizeCommandOutput`, `VisualizeEvalError`
    - `cas_solver::session_api::linear_algebra` now owns the
      linear-algebra-facing surface:
      - `evaluate_det_command_message_on_repl_core(...)`
      - `evaluate_trace_command_message_on_repl_core(...)`
      - `evaluate_transpose_command_message_on_repl_core(...)`
      - `evaluate_linear_system_command_message_on_repl_core(...)`
      - `evaluate_linear_system_command_message(...)`
    - `cas_solver::session_api::algebra` now owns the algebra-facing
      surface:
      - `evaluate_expand_log_*`
      - `evaluate_telescope_*`
      - `evaluate_weierstrass_*`
      - `evaluate_rationalize_command_lines*`
      - algebra parse/usage helpers for expand/telescope/weierstrass
    - `cas_solver::session_api::substitute` now owns the substitute-facing
      surface:
      - `evaluate_substitute_*`
      - `evaluate_substitute_invocation_user_message_on_repl_core(...)`
      - `format_substitute_eval_lines(...)`
      - `format_substitute_parse_error_message(...)`
      - `substitute_render_mode_from_display_mode(...)`
      - `SubstituteRenderMode`
    - `cas_solver::session_api::repl` now owns the REPL-facing parse/preprocess
      surface:
      - `build_prompt_from_eval_options(...)`
      - `parse_repl_command_input(...)`
      - `preprocess_repl_function_syntax(...)`
      - `split_repl_statements(...)`
      - `ReplCommandInput`
    - `cas_session::repl` now owns the public session-side REPL façade
      directly:
      - `build_repl_core_with_config(...)`
      - `evaluate_and_apply_config_command_on_repl(...)`
      - `reset_repl_core_with_config(...)`
      - `reset_repl_core_full_with_config(...)`
      - `evaluate_autoexpand_command_on_repl(...)`
      - `evaluate_context_command_on_repl(...)`
      - `evaluate_semantics_command_on_repl(...)`
    - `cas_session::eval` now owns the public session-side eval façade
      directly:
      - `evaluate_eval_command_with_session(...)`
      - `evaluate_eval_command_pretty_with_session(...)`
      - `evaluate_eval_text_command_with_session(...)`
      - `EvalCommandConfig`
    - `cas_solver::session_api::timeline` now owns the timeline-facing session
      surface:
      - `evaluate_timeline_command_with_session(...)`
      - `format_timeline_command_error_message(...)`
      - `TimelineCommandInput`
      - `TimelineCommandEvalError`
    - `cas_solver::session_api::solve` now owns the solve/full-simplify-facing
      REPL surface:
      - `evaluate_solve_command_message_on_repl_core(...)`
      - `evaluate_full_simplify_command_lines_on_repl_core(...)`
      - `format_solve_command_eval_lines(...)`
      - `TimelineSimplifyEvalError`
      - `TimelineSolveEvalError`
      - `TimelineCommandEvalOutput`
      - `TimelineSimplifyEvalOutput`
      - `TimelineSolveEvalOutput`
    - `cas_solver::session_api::runtime` now re-exports:
      - `ReplSemanticsRuntimeContext`
      - `ReplSolveRuntimeContext`
      - `ReplEngineRuntimeContext`
      - `ReplSessionRuntimeContext`
      - `ReplSessionViewRuntimeContext`
      - `ReplSessionStateMutRuntimeContext`
      - `ReplSessionSimplifierRuntimeContext`
      - `ReplSessionEngineRuntimeContext`
      - `EvalCommandError`
      - `EvalCommandOutput`
    - `cas_solver::session_api::set` now re-exports:
      - `ReplSetRuntimeContext`
      - `SetCommandApplyEffects`
      - `SetCommandPlan`
      - `SetCommandState`
      - `SetDisplayMode`
    - `cas_solver::session_api::settings` now re-exports only settings
      families:
      - `Context*`
      - `Autoexpand*`
      - `SemanticsPreset*`
      - `SemanticsSet*`
    - `cas_solver::session_api::steps` now re-exports:
      - `ReplStepsRuntimeContext`
      - `StepsCommandApplyEffects`
      - `StepsDisplayMode`
    - `cas_solver::session_api::simplifier` now re-exports:
      - `ReplSimplifierRuntimeContext`
      - `SimplifierRuleConfig`
      - `SimplifierToggleConfig`
      - `build_simplifier_with_rule_config(...)`
      - `apply_simplifier_toggle_config(...)`
      - `set_simplifier_toggle_rule(...)`
    - `cas_session` runtime impls were migrated to consume those owners
      directly instead of pulling them from `cas_solver::*` root
    - rationale:
      - the root of `cas_solver` no longer acts as the implicit namespace for
        REPL/session plumbing
  - the last remaining root imports in external crates were also removed
    - `cas_cli` now consumes math façade helpers such as `eval_f64(...)` and
      `is_zero(...)` from `cas_solver::api::*`
    - a workspace grep across `cas_cli`, `cas_session`, `cas_didactic`, and
      `cas_android_ffi` no longer finds `use cas_solver::{...}` root imports
    - rationale:
      - external consumers now rely only on explicit owner namespaces
        (`api`, `runtime`, `wire`, `session_api`, `command_api`, `math`)
  - math-facing helpers also now have an explicit owner namespace
    - `cas_solver::math` is now the intended owner for:
      - `canonical_forms::*`
      - `pattern_marks::*`
    - the last remaining external consumers were migrated away from
      `cas_solver::canonical_forms::*` / `cas_solver::pattern_marks::*`
    - rationale:
      - math utility surfaces stay public when useful, but behind a namespace
        that makes their ownership explicit instead of leaking through the root
  - ownership is now guarded by CI
    - `scripts/lint_solver_namespace_ownership.sh` rejects non-legitimate
      `cas_solver::*` root imports in consumer crates
    - the allowed public ownership surfaces are:
      - `cas_solver::api::*`
      - `cas_solver::runtime::*`
      - `cas_solver::wire::*`
      - `cas_solver::session_api::*`
      - `cas_solver::command_api::*`
      - removed root-level `cas_solver::math::*`
      - removed root-level `cas_solver::strategies::*`
    - rationale:
      - the ownership split is no longer only a convention or grep habit; it
        is enforced automatically in CI
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
      - `cas_android_ffi` now depends directly on `cas_solver::wire` for eval
        and substitute, and no longer depends on `cas_session`
      - rationale:
        - `cas_session` should own session/repl/stateful orchestration
        - stateless wire entrypoints belong to the solver facade
    - `cas_didactic` runtime no longer depends on the `cas_solver` facade just
      for neutral helpers:
      - assumption display line/grouped formatting moved to `cas_solver_core`
      - `reconstruct_global_expr(...)` moved to `cas_solver_core`
      - `cas_didactic` keeps `cas_solver` only in `dev-dependencies` for
        solver-backed tests
      - those helpers were also dropped from the public `cas_solver` facade;
        internal solver code now uses explicit module paths and contracts live
        at `cas_solver_core`
      - internal `cas_didactic/src/**` code now uses a local `crate::runtime`
        seam instead of the misleading alias `crate::cas_solver`
    - the same cleanup now applies to neutral assumption/domain reporting:
      - required-condition, normalized-condition, domain-warning, blocked-hint
        and diagnostics line formatters moved to `cas_solver_core`
      - "assumptions used" collect/group/format helpers moved to
        `cas_solver_core`
      - `cas_session` now reexports these from core instead of `cas_solver`
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
    - stateless eval command entrypoints/types now live under
      `cas_solver::command_api::eval`
      - `evaluate_eval_command_output(...)`
      - `build_eval_command_render_plan(...)`
      - `evaluate_eval_text_simplify_with_session(...)`
      - `EvalCommandError`, `EvalCommandOutput`, `EvalCommandRenderPlan`
      - `EvalDisplayMessage*`, `EvalMetadataLines`, `EvalResultLine`
      - `session_api::{runtime,types}` now re-export from that owner instead of
        reaching into `eval_command_*` directly
      - the root `cas_solver` facade no longer re-exports those eval command
        types/entrypoints through `exports_commands/eval/core.rs`
    - the `substitute` command path also stopped duplicating its internal
      contracts:
      - `SubstituteSimplifyEvalOutput` now has a single owner in
        `cas_solver::substitute::types`
      - `SubstituteParseError` now has a single owner in
        `cas_solver_core::substitute_command_types`
      - `substitute_command_eval` now reuses the real
        `evaluate_substitute_and_simplify(...)` evaluator instead of keeping a
        second copy of the same parse/substitute/simplify flow
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
- `cas_session` public ownership is now explicit: REPL/session entrypoints live under `cas_session::repl`, eval/session rendering under `cas_session::eval`, and mutable session state under `cas_session::state`; the crate root no longer acts as a catch-all public facade for those flows.
- inside `cas_session` itself, the crate root no longer acts as an internal hub
  for `CasConfig`, `ReplCore`, or `SessionState`; module code now imports those
  owners directly from `config`, `repl_core`, and `state_core`.
- inside `cas_session`, the crate root also no longer reexports internal
  support types like `SimplifyCacheKey`, `SimplifiedCache`, `EntryKind`,
  `EntryId`, `ResolveError`, or `CacheConfig`; module code now points directly
  to `cache::*` or `cas_session_core::types::*`.
- `cas_cli::EvalArgs` is no longer stringly-typed for `steps/context/branch/complex/autoexpand`; those axes are now explicit `ValueEnum`s, so CLI parsing and downstream mapping no longer depend on ad-hoc string contracts.
- `cas_api_models::EvalSessionRunConfig` now carries typed enums for
  `steps/context/branch/expand/complex/budget/domain/const_fold/value_domain/complex_branch/inv_trig/assume_scope`,
  so the `cas_cli -> cas_session -> cas_solver` eval path no longer reintroduces
  stringly-typed axes immediately after parse.
- `cas_api_models::EnvelopeEvalOptions` now uses the same typed domain/value-domain
  enums, so the stateless `cas_cli -> cas_solver::wire` envelope path no longer
  reintroduces stringly-typed axes either.
- `cas_solver::eval_option_axes` now also consumes those shared enums directly;
  the intermediate `enum -> &str -> parse -> enum` bridge was removed, so the
  eval path stays typed until the final wire-label boundary.
- `cas_solver/src/exports.rs` has been removed; the crate root now wires
  directly to `exports_base`, `exports_commands`, and `exports_repl` instead of
  keeping an extra internal hub layer.
- `cas_solver/src/repl_command_types.rs` has been removed; REPL parsing now
  uses `cas_solver_core::repl_command_types::ReplCommandInput` directly instead
  of a local pass-through wrapper.
- `cas_solver/src/exports_repl/session.rs` has been removed; the root no longer
  reexports REPL session/runtime helpers through an extra wrapper layer, and
  internal users now point to `repl_session_runtime` or `session_api::runtime`
  directly.
- `cas_solver/src/repl_set_types.rs` has been removed; `ReplSetCommandOutput`
  and `ReplSetMessageKind` now come straight from
  `cas_solver_core::repl_set_types` or `cas_solver::session_api::settings`
  instead of a local pass-through module.
- `cas_solver/src/session_api/symbolic_commands.rs` and
  `cas_solver/src/session_api/types.rs` have been removed; their residual
  surface now lives under the thematic owners `session_api::solve`,
  `session_api::settings`, `session_api::steps`,
  `session_api::simplifier`, and `session_api::eval`
  instead of two mixed grab-bags.
- `cas_solver/src/bindings_types.rs`, `config_command_types.rs`,
  `options_budget_types.rs`, and `semantics_command_types.rs` have been
  removed; those types now come directly from `cas_solver_core` or the local
  owner modules instead of trivial pass-through wrappers.
- `cas_solver/src/substitute_command_types.rs` has also been removed; the
  substitute command now uses `cas_solver_core::substitute_command_types` and
  `cas_solver::substitute::SubstituteSimplifyEvalOutput` directly instead of a
  local pass-through types wrapper.
- `cas_solver/src/rationalize_command_types.rs` has also been removed; the
  `rationalize` command now owns its usage string, typed error, outcome, and
  eval output directly in `rationalize_command.rs` instead of a dedicated
  pass-through types file.
- `cas_solver/src/autoexpand_command_types.rs`,
  `context_command_types.rs`, `semantics_preset_types.rs`, and
  `semantics_set_types.rs` have also been removed; the autoexpand/context/
  semantics flows now import those models directly from `cas_solver_core`
  instead of going through local pass-through wrappers.
- `cas_solver/src/assignment_command/types.rs` and
  `profile_command/types.rs` have also been removed; assignment/profile command
  flows now import those models directly from `cas_solver_core` instead of
  local pass-through wrappers.
- `cas_solver/src/profile_cache_command/types.rs` has also been removed; the
  cache command now owns its local `ProfileCacheCommandInput` in
  `profile_cache_command.rs` and reexports `ProfileCacheCommandResult`
  directly from `cas_solver_core`.
- `cas_solver/src/full_simplify_eval/types.rs` and
  `full_simplify_display/types.rs` have also been removed; those tiny local
  owners now live directly in `full_simplify_eval.rs` and
  `full_simplify_display.rs` instead of an extra `types.rs` layer.
- `cas_solver/src/eval_output_finalize_input/types.rs` has also been removed;
  the root of `eval_output_finalize_input` now reexports `context`, `input`,
  and `shared` directly instead of through another wrapper layer.
- `cas_solver/src/semantics_set_parse.rs` has also been removed; callers now
  use `semantics_set_parse_apply::evaluate_semantics_set_args` directly instead
  of a one-line forwarding wrapper.
- `cas_solver/src/solve_backend.rs` and `solve_backend_active.rs` have also
  been removed; solve dispatch now points directly to
  `solve_backend_contract` and `solve_backend_local`.
- `cas_solver/src/assignment_apply/context.rs`,
  `bindings_command/context.rs`, `inspect_runtime/raw.rs`, and
  `path_rewrite.rs` have also been removed; those parent modules now reexport
  or consume their `cas_solver_core` owners directly instead of keeping
  single-item wrapper leaves.
- `cas_solver/src/assignment_types.rs`, `semantics_view_types.rs`, and
  `solve_input_types.rs` have also been removed; assignment parsing/formatting,
  semantics view formatting, and solve command façades now import those models
  directly from `cas_solver_core` instead of trivial local wrappers.
- `cas_solver/src/health_suite_types.rs` and its submodules have also been
  removed; the health suite catalog/runner/reporting path now imports
  `Category`, `HealthCase`, `HealthLimits`, and `HealthCaseResult` directly
  from `cas_solver_core::{health_category, health_suite_models}` instead of a
  local pass-through wrapper tree.
- `cas_solver/src/exports_base/settings/*` has also been folded directly into
  [`lib.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/lib.rs),
  so the internal root no longer bounces `semantics`, `show`, `set_command`,
  `steps`, and `simplifier` through another layer of tiny wrapper modules.
- `cas_solver/src/exports_base/solve/*` has also been folded directly into
  [`lib.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/lib.rs),
  so `solve`, `substitute`, `timeline`, and transform-facing exports now hang
  from the root owner instead of another wrapper tree.
- the root-level `exports_base.rs` hub has also been removed; `lib.rs` now
  wires directly to its folded `settings` surface, its folded `solve`
  surface, and
  its folded `solver_core` surface
  without another intermediary hub, and that solve/solver-core bridge now
  exports only the root surface still used by the crate.
- local `types.rs` wrappers in
  [`solve_render_config.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/solve_render_config.rs),
  [`eval_input.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/eval_input.rs),
  and
  [`eval_option_axes.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/eval_option_axes.rs)
  have also been folded into their root modules, so those seams now own their
  tiny request/config types directly instead of bouncing through sibling
  `types.rs` files.
- [`solve_command_eval_core/eval.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/solve_command_eval_core/eval.rs)
  now owns `SolveSessionExecution` directly; the local
  `solve_command_eval_core/eval/types.rs` wrapper has been removed.
- [`substitute.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/substitute.rs)
  now owns `SubstituteSimplifyEvalOutput`, `SubstituteStep`,
  `SubstituteResult`, and `SubstituteStrategy` directly; the local
  `substitute/types.rs` wrapper has been removed.
- [`linear_system.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/linear_system.rs)
  now owns `LinearSystemError`, `LinSolveResult`, and
  `with_equation_index(...)` directly; the local `linear_system/types.rs`
  wrapper has been removed.
- [`snapshot.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_session/src/snapshot.rs)
  now owns `SessionSnapshot`, `SessionSnapshotHeader`, and
  `SessionStoreSnapshot` directly; the local `snapshot/types.rs` wrapper has
  been removed.
- [`fraction_sum_analysis.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/fraction_sum_analysis.rs)
  now owns `FractionSumInfo` directly; the local
  `fraction_sum_analysis/types.rs` wrapper has been removed.
- [`timeline/mod.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/timeline/mod.rs)
  now owns `TimelineCliAction`, `TimelineCliRender`, and the timeline command
  output types directly; the local `timeline/types.rs` wrapper has been
  removed.
- [`step_visibility.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/step_visibility.rs)
  now owns `StepVisibility` directly, and the local
  `step_visibility/types.rs` wrapper has been removed.
- [`plans.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/display_policy/plans.rs)
  now owns `CliSubstepsRenderPlan` and `TimelineSubstepsRenderPlan` directly;
  the local `display_policy/plans/types.rs` wrapper has been removed.
- [`timeline/simplify.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/timeline/simplify.rs)
  now owns `TimelineHtml` directly, and the local
  `timeline/simplify/types.rs` wrapper has been removed.
- [`timeline/solve.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/timeline/solve.rs)
  now owns `SolveTimelineHtml` directly, and the local
  `timeline/solve/types.rs` wrapper has been removed.
- [`didactic/mod.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/mod.rs)
  now owns `EnrichedStep` and `SubStep` directly via its leaf modules, and the
  local `didactic/types.rs` wrapper has been removed.
- `cas_solver/src/timeline_types.rs` has been removed; timeline eval output
  types now live under [`session_api/timeline.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/session_api/timeline.rs),
  which is their natural session-facing owner.
- `cas_solver/src/eval_command_types.rs` has been removed; eval command types
  now live directly under
  [`command_api/eval.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/command_api/eval.rs),
  which is the natural stateless owner of `EvalCommand*`.
- `cas_solver/src/eval_command_format.rs`,
  `cas_solver/src/eval_output_presentation_solution.rs`, and
  `cas_solver/src/eval_output_presentation_solve.rs` have been removed; their
  tiny reexport-only surfaces now point directly at the real owners
  ([`eval_command_format_metadata.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/eval_command_format_metadata.rs),
  [`eval_command_format_result.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/eval_command_format_result.rs),
  [`eval_output_presentation_solution_display.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/eval_output_presentation_solution_display.rs),
  [`eval_output_presentation_solution_latex.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/eval_output_presentation_solution_latex.rs),
  [`eval_output_presentation_input.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/eval_output_presentation_input.rs),
  and
  [`eval_output_presentation_solve_steps.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/eval_output_presentation_solve_steps.rs))
  instead of bouncing through wrapper modules.
- `cas_solver/src/linear_system_command_types.rs` has been removed; parse-facing
  linear-system types now live under
  [`linear_system_command_parse.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/linear_system_command_parse.rs)
  and eval-facing command output/error now live under
  [`linear_system_command_eval.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/linear_system_command_eval.rs),
  which are their natural owners.
- `cas_solver/src/types.rs` has been removed; its façade aliases are now wired
  directly from [`api.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/api.rs),
  [`runtime.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/runtime.rs),
  and the leaf modules under
  [`src/types/`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/types),
  instead of going through another internal hub.
- dead passthrough wrappers in `exports_base/solver_core/` (`domain.rs`,
  `solve.rs`) have been removed; those items are now owned only by
  `cas_solver::api` / `cas_solver_core`, without orphan compatibility files.
- the remaining pure passthroughs under the former `exports_base/solver_core`
  surface have also been folded directly into
  [`lib.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/lib.rs);
  the subwrappers
  `solver_core/runtime.rs` and `solver_core/assumptions.rs` are gone, so the
  bridge no longer hides those core-owned items behind another nested layer.
- `cas_cli` non-REPL commands now have a single render owner in
  `commands::dispatch`; `app.rs` and the `frontend_cli` benchmark both route
  through the same dispatch seam instead of maintaining duplicate command
  matches.
- `cas_solver/src/exports_commands/output.rs` has been folded into
  [`lib.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/lib.rs);
  façade-level output helpers (`clean_result_output_line`, parse error render,
  and pipeline display) now hang directly from the root internal façade owner
  instead of another passthrough submodule.
- `cas_solver/src/exports_commands/assumptions.rs` has also been folded into
  [`lib.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/lib.rs);
  assumption-summary and blocked-hint façade helpers now hang directly from
  the root internal façade owner instead of another tiny passthrough file.
- the remaining thematic passthroughs under
  [`lib.rs`](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/lib.rs)
  have also been folded into that root internal owner; the wrapper files
  `exports_commands/{assignment,analysis,health,history}.rs` are gone, so the
  command façade no longer bounces those surfaces through another layer of
  purely mechanical reexports.
- `cas_solver/src/engine_bridge.rs` has been removed; `cas_solver::runtime`
  now reexports `Engine`, `Simplifier`, `Rule`, `Rewrite`, `Orchestrator`,
  `ParentContext` and related runtime types directly from
  `cas_engine`, instead of hiding that ownership behind a one-file passthrough
  module.
- `cas_session::{state_api,eval_api,repl_api}` simplified to `cas_session::{state,eval,repl}` to remove legacy `_api` surface names.
- `cas_session::eval` now owns the eval command surface directly; the internal `eval_command.rs` hub was removed.
- `cas_didactic` now reexports the canonical `cas_solver_core::engine_event_collector::EngineEventCollector` directly from its crate root; the old local `events` wrapper and dead collector/listener leftovers were removed.
