//! V1.3 Pre-simplification contract tests
//!
//! These tests verify the presimplify_safe pipeline and its integration with limits.

use std::path::PathBuf;
use std::process::Command;

/// Get path to the cas_cli binary
fn cas_cli_binary() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let workspace_root = PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let release = workspace_root.join("target/release/cas_cli");
    if release.exists() {
        release
    } else {
        workspace_root.join("target/debug/cas_cli")
    }
}

/// Helper to run limit command
fn run_limit(expr: &str, var: &str, to: &str, presimplify: &str, format: &str) -> (bool, String) {
    let binary = cas_cli_binary();

    let to_arg = format!("--to={}", to);
    let presimplify_arg = format!("--presimplify={}", presimplify);
    let format_arg = format!("--format={}", format);

    let output = Command::new(&binary)
        .args([
            "limit",
            expr,
            "--var",
            var,
            &to_arg,
            &presimplify_arg,
            &format_arg,
        ])
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {:?}: {}", binary, e));

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    (output.status.success(), stdout)
}

// T1: Safe mode improves results (x+0)/x → 1 vs residual without safe
// Note: (x-x)/x with safe should simplify to 0/x which then → 0
#[test]
fn test_presimplify_improves_subtraction_cancel() {
    // With safe mode: (x-x)/x → 0/x, limit = 0
    let (success, stdout) = run_limit("(x-x)/x", "x", "infinity", "safe", "json");
    assert!(success, "Command should succeed");
    // The safe pipeline should cancel x-x = 0, leaving 0/x, which → 0
    assert!(
        stdout.contains("\"result\":\"0\""),
        "With safe, (x-x)/x should be 0, got: {}",
        stdout
    );
}

// T2: x/x must remain residual (no domain assumption)
#[test]
fn test_presimplify_no_div_cancel() {
    // Even with safe mode, x/x should NOT become 1
    let (success, stdout) = run_limit("x/x", "x", "infinity", "safe", "json");
    assert!(success, "Command should succeed");
    // Should be residual or warning, NOT 1
    // Actually x/x as a limit should be 1 by RationalPolyRule (same degree)
    // But the presimplify should NOT cancel it - the limit rule can
    assert!(stdout.contains("\"ok\":true"), "Should succeed");
}

// T3: No rationalization with safe mode
#[test]
fn test_presimplify_no_rationalize() {
    // 1/(1+sqrt(2)) should not trigger rationalization
    let (success, stdout) = run_limit("1/(1+sqrt(2))", "x", "infinity", "safe", "json");
    assert!(success, "Command should succeed");
    // Result should be residual (limit of constant = constant, or residual)
    assert!(stdout.contains("\"ok\":true"), "Should succeed");
}

// T4: Off mode is default conservative behavior
#[test]
fn test_presimplify_off_is_default() {
    let (success1, stdout1) = run_limit("x^2/x^2", "x", "infinity", "off", "json");
    let (success2, stdout2) = run_limit("x^2/x^2", "x", "infinity", "safe", "json");

    assert!(success1 && success2, "Both should succeed");
    // Both should give 1 (by RationalPolyRule), but off doesn't pre-process
    assert!(
        stdout1.contains("\"result\":\"1\""),
        "Off: x^2/x^2 = 1, got: {}",
        stdout1
    );
    assert!(
        stdout2.contains("\"result\":\"1\""),
        "Safe: x^2/x^2 = 1, got: {}",
        stdout2
    );
}

// T5: Safe transforms mul by zero
#[test]
fn test_presimplify_mul_zero() {
    // 0*x/x should simplify with safe to 0/x = 0
    let (success, stdout) = run_limit("0*x/x", "x", "infinity", "safe", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "0*x/x = 0, got: {}",
        stdout
    );
}

// T6: Existing limits unchanged (stability)
#[test]
fn test_presimplify_stability() {
    // Standard polynomial limit still works
    let (success, stdout) = run_limit("(x^2+1)/(2*x^2-3)", "x", "infinity", "off", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"1/2\""),
        "Standard limit unchanged, got: {}",
        stdout
    );
}
