//! Golden Corpus Tests
//!
//! Runs a corpus of commands through the engine to detect regressions.
//! Primary goal: **no panic** on any input.
//! Secondary goal: stable output (kind, code).
//!
//! # Usage
//! ```bash
//! cargo test --test golden_corpus_tests
//! ```

use cas_engine::{Engine, Simplifier};
use cas_parser::parse;
use std::panic::catch_unwind;

/// Load corpus file and return lines (excluding comments and empty lines)
fn load_corpus(filename: &str) -> Vec<String> {
    let path = format!("{}/tests/corpus/{}", env!("CARGO_MANIFEST_DIR"), filename);
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to load corpus {}: {}", path, e));

    content
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && !trimmed.starts_with('#')
        })
        .map(|s| s.to_string())
        .collect()
}

/// Test that all corpus commands execute without panic
#[test]
fn corpus_basic_no_panic() {
    let commands = load_corpus("basic.txt");
    let mut failures: Vec<(String, String)> = Vec::new();

    for cmd in &commands {
        // Each command gets a fresh engine to avoid state leakage
        let result = catch_unwind(|| {
            let mut simplifier = Simplifier::with_default_rules();

            // Try to parse and simplify
            if let Ok(expr) = parse(cmd, &mut simplifier.context) {
                let _ = simplifier.simplify(expr);
            }
            // Parse errors are OK - we just want no panics
        });

        if let Err(panic_info) = result {
            let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            failures.push((cmd.clone(), panic_msg));
        }
    }

    if !failures.is_empty() {
        let report: Vec<String> = failures
            .iter()
            .map(|(cmd, msg)| format!("  '{}' -> {}", cmd, msg))
            .collect();
        panic!(
            "Corpus test failed! {} of {} commands panicked:\n{}",
            failures.len(),
            commands.len(),
            report.join("\n")
        );
    }

    // Success summary
    eprintln!(
        "✓ Corpus: {} commands executed without panic",
        commands.len()
    );
}

/// Test that solve commands don't panic
#[test]
fn corpus_solve_commands_no_panic() {
    let commands = load_corpus("basic.txt");
    let solve_commands: Vec<_> = commands
        .iter()
        .filter(|c| c.starts_with("solve "))
        .collect();

    let mut failures: Vec<(String, String)> = Vec::new();

    for cmd in &solve_commands {
        let result = catch_unwind(|| {
            let mut engine = Engine::new();

            // Parse the solve command
            // Format: "solve <equation>, <var>"
            let rest = cmd.strip_prefix("solve ").unwrap();
            if let Some((eq_str, var)) = rest.rsplit_once(',') {
                let var = var.trim();
                if let Ok(cas_parser::Statement::Equation(eq)) =
                    cas_parser::parse_statement(eq_str.trim(), &mut engine.simplifier.context)
                {
                    let _ = cas_engine::solver::solve(&eq, var, &mut engine.simplifier);
                }
            }
        });

        if let Err(panic_info) = result {
            let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            failures.push((cmd.to_string(), panic_msg));
        }
    }

    if !failures.is_empty() {
        let report: Vec<String> = failures
            .iter()
            .map(|(cmd, msg)| format!("  '{}' -> {}", cmd, msg))
            .collect();
        panic!(
            "Solve corpus test failed! {} of {} commands panicked:\n{}",
            failures.len(),
            solve_commands.len(),
            report.join("\n")
        );
    }

    eprintln!(
        "✓ Solve corpus: {} commands executed without panic",
        solve_commands.len()
    );
}

/// Helper to run a corpus file through the simplifier
fn run_corpus_no_panic(corpus_file: &str) {
    let commands = load_corpus(corpus_file);
    let mut failures: Vec<(String, String)> = Vec::new();

    for cmd in &commands {
        let result = catch_unwind(|| {
            let mut simplifier = Simplifier::with_default_rules();

            // Try to parse and simplify
            if let Ok(expr) = parse(cmd, &mut simplifier.context) {
                let _ = simplifier.simplify(expr);
            }
        });

        if let Err(panic_info) = result {
            let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            failures.push((cmd.clone(), panic_msg));
        }
    }

    if !failures.is_empty() {
        let report: Vec<String> = failures
            .iter()
            .map(|(cmd, msg)| format!("  '{}' -> {}", cmd, msg))
            .collect();
        panic!(
            "Corpus '{}' test failed! {} of {} commands panicked:\n{}",
            corpus_file,
            failures.len(),
            commands.len(),
            report.join("\n")
        );
    }

    eprintln!(
        "✓ Corpus '{}': {} commands executed without panic",
        corpus_file,
        commands.len()
    );
}

/// Test polynomial operations corpus (expand, factor, gcd, multivariate)
#[test]
fn corpus_polynomial_no_panic() {
    run_corpus_no_panic("polynomial.txt");
}

/// Test limits and asymptotics corpus
#[test]
fn corpus_limits_no_panic() {
    run_corpus_no_panic("limits.txt");
}
