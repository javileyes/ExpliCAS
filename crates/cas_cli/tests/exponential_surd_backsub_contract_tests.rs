//! Contracts for universality cycle U3 (scout backlog #5): exponential
//! quadratics whose u-roots are IRRATIONAL now back-substitute exactly.
//!
//! `e^(2x) − e^x − 1 < 0` reduces to `u² − u − 1 < 0` with u = e^x, whose
//! roots (1±√5)/2 are surds the rational bound-mapper could not read — the
//! whole map declined. The mapper now classifies symbolic bounds with the
//! EXACT sign oracles (linear-surd, then const-value-bounds; never f64):
//! provably-negative endpoints clamp away, provably-positive ones map
//! through the boundary equation (`e^x = (1+√5)/2 → ln(φ)`). Verified by
//! independent numeric sampling (~6000 points per set) at review time.

use assert_cmd::cargo;
use assert_cmd::Command;
use serde_json::Value;

fn solve(input: &str) -> String {
    let out = Command::new(cargo::cargo_bin!("cas_cli"))
        .args(["eval", &format!("solve({input}, x)"), "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
    wire["result"].as_str().unwrap_or("").to_string()
}

#[test]
fn golden_ratio_roots_back_substitute() {
    assert_eq!(solve("e^(2*x)-e^x-1<0"), "(-infinity, ln(phi))");
    assert_eq!(solve("e^(2*x)-e^x-1>0"), "(ln(phi), infinity)");
    assert_eq!(solve("e^(2*x)-e^x-1>=0"), "[ln(phi), infinity)");
}

#[test]
fn non_e_bases_map_through_their_logs() {
    assert_eq!(solve("2^(2*x)-2^x-1<0"), "(-infinity, ln(phi) / ln(2))");
}

#[test]
fn rational_root_paths_and_guards_are_untouched() {
    assert_eq!(solve("e^(2*x)-3*e^x+2<0"), "(0, ln(2))");
    assert_eq!(solve("e^(2*x)+3*e^x+2<0"), "No solution");
    assert_eq!(solve("e^(2*x)+e^x-2<0"), "(-infinity, 0)");
    // Fractional (decreasing) bases still decline honestly.
    assert_eq!(
        solve("(1/2)^(2*x)-3*(1/2)^x+2<0"),
        "solve((1/2)^(2·x) + 2 - 3·(1/2)^x < 0, x)"
    );
}
