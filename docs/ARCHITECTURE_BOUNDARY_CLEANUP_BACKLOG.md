# Architecture Boundary Cleanup Backlog

## Purpose

This file turns the current architectural review into a low-risk execution
checklist.

It is intentionally narrower than a full crate-renaming effort:

- preserve the value of the migration already completed
- seal the few remaining real boundary leaks
- avoid large churn with weak payoff
- keep every step easy to validate and easy to revert

## Current Position

The codebase is already in a good post-migration state:

- `cas_ast`, `cas_parser`, `cas_formatter`, and `cas_api_models` are split
- `cas_session_core` is now a justified shared kernel crate
- `cas_solver_core` is a real low-level semantic layer
- `cas_solver` is effectively the application/facade layer

The remaining work is not a broad redesign.
It is a bounded cleanup of a few misplaced models and presentation seams.

## Rules For This Backlog

1. Do not rename crates in this track.
2. Do not move modules just because their names sound heuristic or impure.
3. Only move modules when the destination layer is clearly better.
4. Keep performance/behavior stable and verify after each move.
5. One move set per PR.

## In Scope

### 1. Move history-facing models out of `cas_solver_core`

- [x] Move `crates/cas_solver_core/src/history_models.rs`
      to `crates/cas_solver/src/history_models.rs` or equivalent facade/session
      boundary module.

Reason:
- these types describe history entry inspection, deletion, and overview
- they belong to command/session/application boundaries
- they are not equation-solving kernel logic

### 2. Move health-suite models out of `cas_solver_core`

- [x] Move `crates/cas_solver_core/src/health_suite_models.rs`
      to `crates/cas_solver/src/health_suite_models.rs`
- [x] Move `crates/cas_solver_core/src/health_category.rs`
      alongside it

Reason:
- health suites are application-level regression harnesses
- the runner, catalogs, and report formatting already live in `cas_solver`
- keeping these models in `cas_solver_core` creates a fake dependency upward

### 3. Move owned command DTOs from `cas_solver_core` to `cas_api_models`

- [x] Create a `commands/` area in `crates/cas_api_models/src/`
- [x] Move owned command input/error DTOs from `cas_solver_core` into
      `cas_api_models`

Start with:

- [x] `analysis_command_types.rs`
- [x] `assignment_command_types.rs`
- [x] `autoexpand_command_types.rs`
- [x] `config_command_types.rs`
- [x] `context_command_types.rs`
- [x] `limit_command_types.rs`
- [x] `limit_subcommand_types.rs`
- [x] `profile_command_types.rs`
- [x] `profile_cache_command_types.rs`
- [x] `semantics_command_types.rs`
- [x] `set_command_types.rs`
- [x] `solve_command_types.rs`
- [x] `steps_command_types.rs`
- [x] `substitute_command_types.rs`

Reason:
- these are typed application payloads and result/error DTOs
- they are not transport-specific, so `cas_api_models` is the right home
- they are also not low-level solver kernel logic

Note:
- `semantics_preset_types.rs`, `semantics_set_types.rs`, and
  `semantics_view_types.rs` were intentionally moved out of
  `cas_solver_core`, but into `cas_solver` rather than `cas_api_models`.
- they are REPL/session-facing semantics state models, not neutral external API
  DTOs
- that still satisfies the boundary cleanup goal for `cas_solver_core`

### 4. Move borrowed REPL parse types to `cas_solver`

- [x] Move `crates/cas_solver_core/src/repl_command_types.rs`
      to `crates/cas_solver/src/repl_command_types.rs`

Reason:
- this enum is parser/runtime-facing and borrowed
- it is REPL infrastructure, not a shared stable API model
- it does not belong in `cas_api_models`

### 5. Externalize didactic HTML/CSS assets from Rust source

- [x] Extract the CSS/theme payloads from:
      `crates/cas_didactic/src/timeline/page_theme_css/`
- [x] Extract page CSS from:
      `crates/cas_didactic/src/timeline/simplify_page/css/`
- [x] Extract page CSS from:
      `crates/cas_didactic/src/timeline/solve_page/css/`
- [x] Keep thin Rust render adapters around the generated document/string path

Reason:
- current didactic HTML/CSS is embedded as Rust string constants
- that hurts maintainability and visual iteration speed
- this is a presentation concern, not core symbolic logic

Current status:
- `page_theme_css` already loads its payload from `crates/cas_didactic/assets/`
  via `include_str!`
- `simplify_page/css` and `solve_page/css` also load from
  `crates/cas_didactic/assets/` via `include_str!`
- the shared page shell (`head`, `body` intro, footer) and the static timeline
  scripts also load from `crates/cas_didactic/assets/` via `include_str!`
- `solve_render` static fragments (`timeline`, step shell, substep shell, final
  result) also load from `crates/cas_didactic/assets/` via `include_str!`
- `simplify_render` static fragments (`timeline`, step shell, before/rule/after
  sections, domain wrapper, substep wrappers, global requires, final-result
  wrappers) also load from
  `crates/cas_didactic/assets/` via `include_str!`
- the Rust modules in these trees are now only thin composition/adaptation
  layers

## Explicitly Out Of Scope

- [ ] Do not rename `cas_solver` to `cas_orchestrator` in this track
- [ ] Do not rename `cas_solver_core` to `cas_solver` in this track
- [ ] Do not merge `cas_session_core` back into `cas_session`
- [ ] Do not move `budget_model.rs` out of `cas_solver_core`
- [ ] Do not move verification kernel modules out of `cas_solver_core`
- [ ] Do not move `quadratic_formula.rs` / `isolation_power.rs` out of
      `cas_solver_core`
- [ ] Do not move `fraction_add_heuristics_support.rs` out of `cas_math`
- [ ] Do not move `trig_pattern_detection.rs` out of `cas_math`
- [ ] Do not start `egg` / e-graph adoption here
- [ ] Do not reopen AST ownership redesign (`Rc`, `Arc`, arena rewrite) here

## Why These Modules Stay Put

### Keep in `cas_solver_core`

- `budget_model.rs`
- `verification.rs`
- `verification_runtime_flow.rs`
- `quadratic_formula.rs`
- `isolation_power.rs`

Reason:
- these are core runtime semantics, proof, isolation, or anti-explosion policy

### Keep in `cas_math` for now

- `fraction_add_heuristics_support.rs`
- `trig_pattern_detection.rs`

Reason:
- they are still low-level AST math support
- they do not know about `Engine`, sessions, wire, or didactic rendering
- moving them only by naming instinct would likely worsen layering

## Recommended Execution Order

1. `history_models`
2. `health_suite_models` + `health_category`
3. owned command DTOs -> `cas_api_models`
4. `repl_command_types` -> `cas_solver`
5. didactic asset extraction
6. only then reassess whether crate renaming still matters

## Validation Gates

Every move set should keep this green:

1. `cargo fmt --all`
2. `make ci`
3. relevant focused tests for the moved subsystem
4. if solve/eval boundaries move, re-run:
   - `cargo test --release -p cas_solver --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --exact --nocapture`

## Done Criteria

This backlog is complete when:

1. `cas_solver_core` no longer carries history-facing models
2. `cas_solver_core` no longer carries health-suite models
3. owned command DTOs live in `cas_api_models`
4. borrowed REPL parse types live in `cas_solver`
5. `cas_didactic` no longer stores large CSS/HTML payloads as Rust source
6. no crate rename was needed to achieve those outcomes
