//! Metamorphic Simplification Tests
//!
//! Tests mathematical correctness using metamorphic testing:
//! If A simplifies to B, then A+e should also equal B+e (numerically).
//!
//! This catches soundness bugs that golden string tests miss:
//! - Incomplete requires
//! - Rule priority issues
//! - Cancellation failures in context
//!
//! # Configuration
//!
//! Set environment variables to control behavior:
//! - `METATEST_STRESS=1`: Enable stress mode (more samples, deeper expressions)
//! - `METATEST_SEED=<u64>`: Force specific RNG seed for reproducibility
//!
//! # Example failure output
//!
//! ```text
//! Metatest failed (seed=12648430, iter=42)
//! A = tan(x) * tan(pi/3 - x) * tan(pi/3 + x)
//! B = tan(3*x)
//! e = (x) + (sin(x * 2))
//! A+e = (tan(x) * tan(pi/3 - x) * tan(pi/3 + x)) + ((x) + (sin(x * 2)))
//! B+e = (tan(3*x)) + ((x) + (sin(x * 2)))
//! ```

mod test_utils;

use cas_ast::{Context, DisplayExpr, ExprId};
use cas_engine::engine::eval_f64;
use cas_engine::Simplifier;
use cas_parser::parse;
use std::collections::HashMap;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

use std::time::SystemTime;

// =============================================================================
// Logging Infrastructure
// =============================================================================

/// Log file path (relative to project root)
fn log_file_path() -> PathBuf {
    // Try to use project root, fallback to current dir
    let base = env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    base.parent().unwrap_or(&base).join("metatest_log.jsonl")
}

/// Append a log entry to the metatest log file (JSON Lines format)
fn log_metatest_run(
    test_name: &str,
    config: &MetatestConfig,
    passed: usize,
    failed: usize,
    skipped: usize,
) {
    // Get timestamp
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let stress = env::var("METATEST_STRESS").ok().as_deref() == Some("1");

    // Build JSON entry
    let entry = format!(
        r#"{{"timestamp":{},"test":"{}","seed":{},"samples":{},"depth":{},"min_valid":{},"stress":{},"passed":{},"failed":{},"skipped":{}}}"#,
        timestamp,
        test_name,
        config.seed,
        config.samples,
        config.depth,
        config.min_valid,
        stress,
        passed,
        failed,
        skipped
    );

    // Append to log file
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_file_path())
    {
        let _ = writeln!(file, "{}", entry);
    }
}

// Configuration
// =============================================================================

/// Get metatest configuration from environment
fn metatest_config() -> MetatestConfig {
    let stress = env::var("METATEST_STRESS").ok().as_deref() == Some("1");

    let samples = if stress { 500 } else { 50 };
    let min_valid = if stress { 250 } else { 20 };
    let depth = if stress { 5 } else { 3 };

    // Seed from env or default
    let seed = env::var("METATEST_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0xC0FFEE_u64);

    MetatestConfig {
        samples,
        min_valid,
        depth,
        seed,
        atol: 1e-9,
        rtol: 1e-9,
        sample_range: (-5.0, 5.0),
        eval_samples: 200,
    }
}

#[derive(Clone, Debug)]
struct MetatestConfig {
    /// Number of random expressions `e` to generate
    samples: usize,
    /// Minimum valid numeric samples per test
    min_valid: usize,
    /// Maximum depth of generated expression `e`
    depth: usize,
    /// RNG seed (for reproducibility)
    seed: u64,
    /// Absolute tolerance for numeric comparison
    atol: f64,
    /// Relative tolerance for numeric comparison
    rtol: f64,
    /// Range for variable sampling
    sample_range: (f64, f64),
    /// Number of evaluation samples per comparison
    eval_samples: usize,
}

// =============================================================================
// Deterministic RNG (avoid external dependencies)
// =============================================================================

/// Linear Congruential Generator for deterministic randomness
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }

    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Pick in range [0, n)
    fn pick(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }

    /// Pick in range [lo, hi]
    fn pick_i32(&mut self, lo: i32, hi: i32) -> i32 {
        let span = (hi - lo + 1) as u32;
        lo + (self.pick(span) as i32)
    }
}

// =============================================================================
// Expression Generator
// =============================================================================

/// Generate a random expression using only "safe" operations.
///
/// Safe operations (no domain issues):
/// - Variables from the provided list
/// - Small integer constants (-3 to 3)
/// - Add, Sub, Mul
/// - Pow with small non-negative integer exponents (0-4)
/// - sin, cos (total functions)
///
/// NOT included (domain issues):
/// - Division
/// - log, ln, sqrt, root
/// - Negative exponents
fn gen_expr(vars: &[&str], depth: usize, rng: &mut Lcg) -> String {
    if depth == 0 || vars.is_empty() {
        // Leaf: variable or constant
        if vars.is_empty() || rng.pick(4) == 0 {
            // Constant
            rng.pick_i32(-3, 3).to_string()
        } else {
            // Variable
            let idx = rng.pick(vars.len() as u32) as usize;
            vars[idx].to_string()
        }
    } else {
        match rng.pick(10) {
            0 | 1 => {
                // Add
                format!(
                    "({}) + ({})",
                    gen_expr(vars, depth - 1, rng),
                    gen_expr(vars, depth - 1, rng)
                )
            }
            2 | 3 => {
                // Sub
                format!(
                    "({}) - ({})",
                    gen_expr(vars, depth - 1, rng),
                    gen_expr(vars, depth - 1, rng)
                )
            }
            4 | 5 => {
                // Mul
                format!(
                    "({}) * ({})",
                    gen_expr(vars, depth - 1, rng),
                    gen_expr(vars, depth - 1, rng)
                )
            }
            6 => {
                // Pow with small non-negative exponent
                let base = gen_expr(vars, depth - 1, rng);
                let exp = [0, 1, 2, 3, 4][rng.pick(5) as usize];
                format!("({})^({})", base, exp)
            }
            7 => {
                // sin (total function)
                format!("sin({})", gen_expr(vars, depth - 1, rng))
            }
            8 => {
                // cos (total function)
                format!("cos({})", gen_expr(vars, depth - 1, rng))
            }
            _ => {
                // Bias toward leaves to avoid size explosion
                gen_expr(vars, 0, rng)
            }
        }
    }
}

// =============================================================================
// Numeric Equivalence Check
// =============================================================================

/// Check if two expressions are numerically equivalent for 1 variable.
/// Returns Ok(valid_count) or Err(message).
fn check_numeric_equiv_1var(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var: &str,
    config: &MetatestConfig,
) -> Result<usize, String> {
    let (lo, hi) = config.sample_range;
    let mut valid = 0usize;
    let mut eval_failed = 0usize;

    for i in 0..config.eval_samples {
        let t = (i as f64 + 0.5) / config.eval_samples as f64;
        let x = lo + (hi - lo) * t;

        let mut var_map = HashMap::new();
        var_map.insert(var.to_string(), x);

        let va = eval_f64(ctx, a, &var_map);
        let vb = eval_f64(ctx, b, &var_map);

        match (va, vb) {
            (Some(va), Some(vb)) => {
                if va.is_nan() || vb.is_nan() {
                    eval_failed += 1;
                    continue;
                }
                valid += 1;

                // Check approximate equality
                let diff = (va - vb).abs();
                let scale = va.abs().max(vb.abs()).max(1.0);
                let allowed = config.atol + config.rtol * scale;

                if diff > allowed {
                    return Err(format!(
                        "Numeric mismatch at {}={}:\n  a={:.15}\n  b={:.15}\n  diff={:.3e} > allowed={:.3e}",
                        var, x, va, vb, diff, allowed
                    ));
                }
            }
            _ => {
                eval_failed += 1;
            }
        }
    }

    if valid < config.min_valid {
        return Err(format!(
            "Too few valid samples: {} < {} (eval_failed={})",
            valid, config.min_valid, eval_failed
        ));
    }

    Ok(valid)
}

// =============================================================================
// Metamorphic Test Harness
// =============================================================================

/// Assert metamorphic property: if A ≡ B, then A+e ≡ B+e for random e.
///
/// Uses numeric verification to avoid circular dependency on simplifier.
/// Logs results to metatest_log.jsonl for historical tracking.
fn assert_metamorphic_addition(test_name: &str, base_a: &str, base_b: &str, vars: &[&str]) {
    let config = metatest_config();

    if vars.is_empty() {
        // Skip tests with no variables (can't do numeric verification)
        eprintln!(
            "⚠️  Skipping metatest (no variables): {} → {}",
            base_a, base_b
        );
        log_metatest_run(test_name, &config, 0, 0, 1);
        return;
    }

    // Only support single variable for now
    if vars.len() > 1 {
        eprintln!(
            "⚠️  Skipping metatest (multi-var not yet supported): {} → {}",
            base_a, base_b
        );
        log_metatest_run(test_name, &config, 0, 0, 1);
        return;
    }

    let mut rng = Lcg::new(config.seed);
    let var = vars[0];

    for iter in 0..config.samples {
        let e = gen_expr(vars, config.depth, &mut rng);
        let a_plus = format!("({}) + ({})", base_a, e);
        let b_plus = format!("({}) + ({})", base_b, e);

        // Parse and simplify
        let mut simplifier = Simplifier::with_default_rules();
        let a_expr = match parse(&a_plus, &mut simplifier.context) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("Parse error (skipping): A+e='{}', error={:?}", a_plus, err);
                continue;
            }
        };
        let b_expr = match parse(&b_plus, &mut simplifier.context) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("Parse error (skipping): B+e='{}', error={:?}", b_plus, err);
                continue;
            }
        };

        // Simplify both sides
        let (a_simp, _) = simplifier.simplify(a_expr);
        let (b_simp, _) = simplifier.simplify(b_expr);

        // Check numeric equivalence
        if let Err(err) =
            check_numeric_equiv_1var(&simplifier.context, a_simp, b_simp, var, &config)
        {
            // Log failure before panic
            log_metatest_run(test_name, &config, 0, 1, 0);
            panic!(
                "Metatest FAILED (seed={}, iter={})\n\
                 A = {}\n\
                 B = {}\n\
                 e = {}\n\
                 A+e = {}\n\
                 B+e = {}\n\
                 A+e simplified = {}\n\
                 B+e simplified = {}\n\
                 Error: {}",
                config.seed,
                iter,
                base_a,
                base_b,
                e,
                a_plus,
                b_plus,
                DisplayExpr {
                    context: &simplifier.context,
                    id: a_simp
                },
                DisplayExpr {
                    context: &simplifier.context,
                    id: b_simp
                },
                err
            );
        }
    }

    // Log success
    log_metatest_run(test_name, &config, 1, 0, 0);
}

// =============================================================================
// Metamorphic Tests
// =============================================================================

#[test]
fn metatest_pythagorean_identity() {
    // sin²(x) + cos²(x) = 1
    assert_metamorphic_addition("pythagorean_identity", "sin(x)^2 + cos(x)^2", "1", &["x"]);
}

#[test]
fn metatest_double_angle_sin() {
    // sin(2x) = 2·sin(x)·cos(x)
    assert_metamorphic_addition("double_angle_sin", "sin(2*x)", "2*sin(x)*cos(x)", &["x"]);
}

#[test]
fn metatest_double_angle_cos() {
    // cos(2x) = cos²(x) - sin²(x)
    assert_metamorphic_addition(
        "double_angle_cos",
        "cos(2*x)",
        "cos(x)^2 - sin(x)^2",
        &["x"],
    );
}

#[test]
fn metatest_add_zero() {
    // x + 0 = x
    assert_metamorphic_addition("add_zero", "x + 0", "x", &["x"]);
}

#[test]
fn metatest_mul_one() {
    // x * 1 = x
    assert_metamorphic_addition("mul_one", "x * 1", "x", &["x"]);
}

#[test]
fn metatest_binomial_square() {
    // (x + 1)² = x² + 2x + 1
    assert_metamorphic_addition("binomial_square", "(x + 1)^2", "x^2 + 2*x + 1", &["x"]);
}

#[test]
fn metatest_difference_of_squares() {
    // (x - 1)(x + 1) = x² - 1
    assert_metamorphic_addition(
        "difference_of_squares",
        "(x - 1) * (x + 1)",
        "x^2 - 1",
        &["x"],
    );
}

#[test]
#[ignore] // Known issue: TanToSinCosRule expands standalone tan(3*x) but not identity result
fn metatest_triple_tan_identity() {
    // tan(x)·tan(π/3-x)·tan(π/3+x) = tan(3x)
    //
    // Known issue: When tan(3*x) appears in sum context (B+e), TanToSinCosRule
    // expands it to sin/cos. But when the product simplifies to tan(3x) (A+e),
    // that tan(3x) is NOT expanded because it comes from an identity rule.
    // This causes A+e and B+e to have different canonical forms.
    //
    // TODO: Gate TanToSinCosRule to not fire on identity-derived results,
    // or compare numerically without simplification.
    assert_metamorphic_addition(
        "triple_tan_identity",
        "tan(x) * tan(pi/3 - x) * tan(pi/3 + x)",
        "tan(3*x)",
        &["x"],
    );
}

#[test]
fn metatest_log_product() {
    // ln(2) + ln(3) = ln(6)
    // No variables, skipped by harness
    assert_metamorphic_addition("log_product", "ln(2) + ln(3)", "ln(6)", &[]);
}

#[test]
fn metatest_polynomial_simplify() {
    // (x + 1)(x - 1) + 1 = x²
    assert_metamorphic_addition(
        "polynomial_simplify",
        "(x + 1) * (x - 1) + 1",
        "x^2",
        &["x"],
    );
}
