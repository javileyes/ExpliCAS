//! Tests for REPL prompt indicator format
//!
//! These tests verify that the prompt indicator format is stable and predictable,
//! helping prevent UX regressions when multiple modes are active.

use cas_engine::options::{BranchMode, ComplexMode, ContextMode, EvalOptions, StepsMode};

/// Build prompt indicators string (pure function for testing).
/// Mirrors the logic in Repl::build_prompt() without requiring a full Repl instance.
fn build_prompt_indicators(opts: &EvalOptions) -> String {
    let mut indicators = Vec::new();

    // Show steps mode if not On (default)
    match opts.steps_mode {
        StepsMode::Off => indicators.push("[steps:off]"),
        StepsMode::Compact => indicators.push("[steps:compact]"),
        StepsMode::On => {} // Default, no indicator
    }

    // Show context mode if not Auto (default)
    match opts.context_mode {
        ContextMode::IntegratePrep => indicators.push("[ctx:integrate]"),
        ContextMode::Solve => indicators.push("[ctx:solve]"),
        ContextMode::Standard => indicators.push("[ctx:standard]"),
        ContextMode::Auto => {} // Default, no indicator
    }

    // Show branch mode if not Strict (default)
    match opts.branch_mode {
        BranchMode::PrincipalBranch => indicators.push("[branch:principal]"),
        BranchMode::Strict => {} // Default, no indicator
    }

    // Show complex mode if not Auto (default)
    match opts.complex_mode {
        ComplexMode::On => indicators.push("[cx:on]"),
        ComplexMode::Off => indicators.push("[cx:off]"),
        ComplexMode::Auto => {} // Default, no indicator
    }

    if indicators.is_empty() {
        "> ".to_string()
    } else {
        format!("{} > ", indicators.join(""))
    }
}

#[test]
fn default_options_empty_prompt() {
    let opts = EvalOptions::default();
    assert_eq!(build_prompt_indicators(&opts), "> ");
}

#[test]
fn steps_off_only() {
    let opts = EvalOptions {
        steps_mode: StepsMode::Off, ..Default::default()
        ..Default::default()
    };
    assert_eq!(build_prompt_indicators(&opts), "[steps:off] > ");
}

#[test]
fn steps_compact_only() {
    let opts = EvalOptions {
        steps_mode: StepsMode::Compact, ..Default::default()
        ..Default::default()
    };
    assert_eq!(build_prompt_indicators(&opts), "[steps:compact] > ");
}

#[test]
fn context_integrate_only() {
    let opts = EvalOptions {
        context_mode: ContextMode::IntegratePrep,
        ..Default::default()
    };
    assert_eq!(build_prompt_indicators(&opts), "[ctx:integrate] > ");
}

#[test]
fn branch_principal_only() {
    let opts = EvalOptions {
        branch_mode: BranchMode::PrincipalBranch,
        ..Default::default()
    };
    assert_eq!(build_prompt_indicators(&opts), "[branch:principal] > ");
}

#[test]
fn complex_on_only() {
    let opts = EvalOptions {
        complex_mode: ComplexMode::On,
        ..Default::default()
    };
    assert_eq!(build_prompt_indicators(&opts), "[cx:on] > ");
}

#[test]
fn combined_steps_and_context() {
    let opts = EvalOptions {
        steps_mode: StepsMode::Off, ..Default::default()
        context_mode: ContextMode::IntegratePrep,
        ..Default::default()
    };
    assert_eq!(
        build_prompt_indicators(&opts),
        "[steps:off][ctx:integrate] > "
    );
}

#[test]
fn combined_all_non_default() {
    let opts = EvalOptions {
        steps_mode: StepsMode::Off, ..Default::default()
        context_mode: ContextMode::IntegratePrep,
        branch_mode: BranchMode::PrincipalBranch,
        complex_mode: ComplexMode::On,
    };
    assert_eq!(
        build_prompt_indicators(&opts),
        "[steps:off][ctx:integrate][branch:principal][cx:on] > "
    );
}

#[test]
fn order_is_deterministic() {
    // Order: steps, context, branch, complex (alphabetical except steps first)
    let opts = EvalOptions {
        steps_mode: StepsMode::Compact, ..Default::default()
        context_mode: ContextMode::Solve,
        branch_mode: BranchMode::PrincipalBranch,
        complex_mode: ComplexMode::Off,
    };
    // Run multiple times to ensure determinism
    for _ in 0..10 {
        assert_eq!(
            build_prompt_indicators(&opts),
            "[steps:compact][ctx:solve][branch:principal][cx:off] > "
        );
    }
}
