# Didactic Rendering Phase 2 Plan

## Purpose

This document turns the optional `cas_didactic` follow-up into a bounded
execution plan.

Phase 1 is already done:

- large static HTML/CSS payloads left Rust source
- assets now live under `crates/cas_didactic/assets/timeline`
- Rust renderers already act as thin adapters around `include_str!`

Phase 2 exists only to finish the cleanup pragmatically.
It is **not** a frontend rewrite and **not** a template-engine migration.

## Goal

Reduce repetitive renderer boilerplate while keeping the current rendering
model:

- static assets stay in `assets/`
- Rust stays responsible for typed composition and escaping
- output parity remains stable

## Non-Goals

Do not use this track to:

- redesign the timeline UI
- introduce Askama, Tera, or another template engine
- move more static payloads just for the sake of movement
- expand `cas_didactic` into a frontend framework

## Scope

### In Scope

1. Add tiny template helpers/macros in `timeline/render_template.rs`.
2. Use those helpers in the highest-duplication renderers:
   - `page_shell`
   - `solve_render`
   - `solve_timeline_render`
   - `simplify_step_html`
   - `simplify_substeps`
   - `simplify_summary`
3. Keep placeholder usage consistent across the touched renderers.
4. Strengthen validation using existing render/parity suites.

### Out of Scope

1. CSS/theme redesign.
2. Timeline interaction redesign.
3. Generic component system beyond tiny helpers/macros.
4. New serialization or transport concerns.

## Execution Plan

### Phase 2A. Shared Render Helper Surface

Introduce a small helper surface in `render_template.rs`:

- load timeline assets without repeating full `include_str!(concat!(...))`
- render asset-backed templates without repeating the same wrapper shape
- optionally append rendered fragments into a `String` without temporary
  boilerplate

Success criteria:

- helper is tiny and obvious
- no new abstraction layer around business logic
- renderers become shorter without becoming magical

### Phase 2B. Solve Renderer Cleanup

Apply the helper to:

- `solve_render/*`
- `solve_timeline_render.rs`
- `page_shell/*`

Success criteria:

- solve timeline output stays unchanged
- solve render callsites no longer repeat full asset include paths

### Phase 2C. Simplify Renderer Cleanup

Apply the helper to:

- `simplify_step_html/*`
- `simplify_substeps/*`
- `simplify_summary/*`
- `simplify_render/shell.rs`

Success criteria:

- simplify timeline output stays unchanged
- section/substep/final-result renderers follow the same asset-render pattern

### Phase 2D. Close the Track

Update backlog/docs so the optional item is no longer a vague future task.

Success criteria:

- follow-up backlog marks this track as done
- the retained approach is explicit: static assets + thin typed adapters

## Validation

Required validation for closing the track:

- `cargo test -p cas_didactic --lib -- --nocapture`
- `cargo test -p cas_didactic --test solve_timeline_parity_tests -- --nocapture`
- `cargo test -p cas_didactic --test timeline_render_test -- --nocapture`
- `cargo test -p cas_didactic --tests --no-run`
- `cargo fmt --all`
- `make ci`

## Stop Condition

Stop after the helper + renderer cleanup if:

- output parity is preserved
- the remaining duplication is small
- any further work would become speculative frontend redesign

That is the intended end state for Phase 2.
