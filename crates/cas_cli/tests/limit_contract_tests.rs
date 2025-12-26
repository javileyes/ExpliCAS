//! Non-regression tests for CLI limit command
//!
//! These tests verify the limit command works correctly for various cases.

use std::path::PathBuf;
use std::process::Command;

/// Get path to the cas_cli binary
fn cas_cli_binary() -> PathBuf {
    // CARGO_MANIFEST_DIR points to crates/cas_cli
    // Binary is at workspace_root/target/release/cas_cli or target/debug/cas_cli
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let workspace_root = PathBuf::from(manifest_dir)
        .parent()
        .unwrap() // crates/
        .parent()
        .unwrap() // workspace root
        .to_path_buf();

    let release = workspace_root.join("target/release/cas_cli");
    if release.exists() {
        release
    } else {
        workspace_root.join("target/debug/cas_cli")
    }
}

/// Helper to run limit command and get stdout
fn run_limit(expr: &str, var: &str, to: &str, format: &str) -> (bool, String) {
    let binary = cas_cli_binary();

    // Use = format for --to to handle negative values like -infinity
    let to_arg = format!("--to={}", to);
    let format_arg = format!("--format={}", format);

    let output = Command::new(&binary)
        .args(["limit", expr, "--var", var, &to_arg, &format_arg])
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {:?}: {}", binary, e));

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    (output.status.success(), stdout)
}

#[test]
fn test_limit_x_to_infinity_text() {
    let (success, stdout) = run_limit("x", "x", "infinity", "text");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("infinity"),
        "Result should be infinity, got: {}",
        stdout
    );
}

#[test]
fn test_limit_rational_poly_json() {
    let (success, stdout) = run_limit("(x^2+1)/(2*x^2-3)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(stdout.contains("\"ok\":true"), "JSON should have ok:true");
    assert!(
        stdout.contains("\"result\":\"1/2\""),
        "Result should be 1/2, got: {}",
        stdout
    );
}

#[test]
fn test_limit_neg_infinity_parity() {
    let (success, stdout) = run_limit("x^3/x^2", "x", "-infinity", "text");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("-infinity"),
        "Result should be -infinity, got: {}",
        stdout
    );
}

#[test]
fn test_limit_residual_with_warning_json() {
    let (success, stdout) = run_limit("sin(x)/x", "x", "infinity", "json");
    assert!(success, "Command should succeed even for residual");
    assert!(stdout.contains("\"ok\":true"), "JSON should have ok:true");
    assert!(
        stdout.contains("\"warning\""),
        "Should have warning for unresolved limit"
    );
    assert!(
        stdout.contains("limit("),
        "Result should contain residual limit(...)"
    );
}

#[test]
fn test_limit_deg_num_less_than_deg_den() {
    let (success, stdout) = run_limit("x^2/x^3", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "Result should be 0, got: {}",
        stdout
    );
}

#[test]
fn test_limit_higher_num_degree_infinity() {
    let (success, stdout) = run_limit("x^3/x^2", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("infinity"),
        "Result should contain infinity, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("-infinity"),
        "Should be positive infinity at +∞"
    );
}
