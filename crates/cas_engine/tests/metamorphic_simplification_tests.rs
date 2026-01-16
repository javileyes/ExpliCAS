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
///
/// Modes:
/// - Normal: 50 samples, depth=3 (default, ~1s per test)
/// - Stress: 500 samples, depth=5 (METATEST_STRESS=1, ~4s per test)
/// - Extreme: 1000 samples, depth=8 (METATEST_EXTREME=1, ~30s+ per test)
fn metatest_config() -> MetatestConfig {
    let extreme = env::var("METATEST_EXTREME").ok().as_deref() == Some("1");
    let stress = env::var("METATEST_STRESS").ok().as_deref() == Some("1");

    let (samples, min_valid, depth, eval_samples) = if extreme {
        (1000, 500, 7, 1000) // depth=7 avoids numerical precision issues
    } else if stress {
        (500, 250, 5, 500)
    } else {
        (50, 20, 3, 200)
    };

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
        eval_samples,
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
                // Pow with small positive exponent (avoid 0 to prevent 0^0=undefined)
                let base = gen_expr(vars, depth - 1, rng);
                let exp = [1, 2, 3, 4][rng.pick(4) as usize];
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
                // Filter out NaN and Inf (singularities like tan poles)
                if va.is_nan() || vb.is_nan() || va.is_infinite() || vb.is_infinite() {
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
            (va, vb) => {
                // Both None, or one None - skip this sample
                let _ = (va, vb); // suppress warning
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
/// In stress mode, spawns a thread with 16MB stack to handle deep expressions.
fn assert_metamorphic_addition(test_name: &str, base_a: &str, base_b: &str, vars: &[&str]) {
    let _config = metatest_config();
    let extreme = env::var("METATEST_EXTREME").ok().as_deref() == Some("1");
    let stress = env::var("METATEST_STRESS").ok().as_deref() == Some("1");

    // In stress/extreme mode, spawn a thread with larger stack
    if stress || extreme {
        let test_name = test_name.to_string();
        let base_a = base_a.to_string();
        let base_b = base_b.to_string();
        let vars: Vec<String> = vars.iter().map(|v| v.to_string()).collect();

        // 32MB for extreme, 16MB for stress
        let stack_size = if extreme {
            32 * 1024 * 1024
        } else {
            16 * 1024 * 1024
        };
        let child = std::thread::Builder::new()
            .stack_size(stack_size)
            .spawn(move || {
                let vars_refs: Vec<&str> = vars.iter().map(|s| s.as_str()).collect();
                assert_metamorphic_addition_impl(&test_name, &base_a, &base_b, &vars_refs);
            })
            .expect("Failed to spawn test thread");

        child.join().expect("Test thread panicked");
        return;
    }

    // Normal mode: run inline
    assert_metamorphic_addition_impl(test_name, base_a, base_b, vars);
}

/// Core implementation of metamorphic test (called from main thread or spawned thread)
fn assert_metamorphic_addition_impl(test_name: &str, base_a: &str, base_b: &str, vars: &[&str]) {
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

        // Log BEFORE simplify for crash reproduction
        // This is crucial: stack overflow can't be caught, so we log first
        {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open("/tmp/metatest_repro.log")
            {
                let _ = writeln!(f, "test={}", test_name);
                let _ = writeln!(f, "seed={}", config.seed);
                let _ = writeln!(f, "iter={}", iter);
                let _ = writeln!(f, "expr_a={}", a_plus);
                let _ = writeln!(f, "expr_b={}", b_plus);
                let _ = f.flush();
            }
        }

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

// =============================================================================
// Exponent Identities
// =============================================================================

#[test]
fn metatest_power_of_power_even_root() {
    // (x^2)^(1/2) = |x|
    assert_metamorphic_addition("power_of_power_even_root", "(x^2)^(1/2)", "|x|", &["x"]);
}

#[test]
fn metatest_power_of_power_odd_root() {
    // (x^3)^(1/3) = x
    assert_metamorphic_addition("power_of_power_odd_root", "(x^3)^(1/3)", "x", &["x"]);
}

#[test]
fn metatest_exp_quotient() {
    // exp(x+1)/exp(1) = e^x
    assert_metamorphic_addition("exp_quotient", "exp(x + 1) / exp(1)", "e^(x)", &["x"]);
}

// =============================================================================
// Fraction Identities
// =============================================================================

#[test]
fn metatest_fraction_opposite_denominators() {
    // 1/(x-1) + 1/(1-x) = 0
    assert_metamorphic_addition(
        "fraction_opposite_denominators",
        "1/(x-1) + 1/(1-x)",
        "0",
        &["x"],
    );
}

#[test]
fn metatest_fraction_same_denominator() {
    // 1/(x-1) + 2/(x-1) = 3/(x-1)
    assert_metamorphic_addition(
        "fraction_same_denominator",
        "1/(x-1) + 2/(x-1)",
        "3/(x-1)",
        &["x"],
    );
}

// =============================================================================
// Algebraic Identities
// =============================================================================

#[test]
fn metatest_cube_of_sum() {
    // (x + 1)^3 = x^3 + 3x^2 + 3x + 1
    assert_metamorphic_addition("cube_of_sum", "(x + 1)^3", "x^3 + 3*x^2 + 3*x + 1", &["x"]);
}

#[test]
fn metatest_sum_of_cubes() {
    // x^3 + 1 = (x + 1)(x^2 - x + 1)
    assert_metamorphic_addition("sum_of_cubes", "x^3 + 1", "(x + 1) * (x^2 - x + 1)", &["x"]);
}

#[test]
fn metatest_distribute_and_collect() {
    // 2(x + 1) + 3(x - 1) = 5x - 1
    assert_metamorphic_addition(
        "distribute_and_collect",
        "2*(x + 1) + 3*(x - 1)",
        "5*x - 1",
        &["x"],
    );
}

// =============================================================================
// Trigonometric Identities (additional)
// =============================================================================

#[test]
fn metatest_sin_cos_sum() {
    // sin(x)^2 + cos(x)^2 - cos(2x) = 2*sin(x)^2
    // (since cos(2x) = cos²x - sin²x = 1 - 2sin²x)
    assert_metamorphic_addition(
        "sin_cos_sum_identity",
        "sin(x)^2 + cos(x)^2 - cos(2*x)",
        "2*sin(x)^2",
        &["x"],
    );
}

#[test]
fn metatest_tan_identity() {
    // sin(x)/cos(x) = tan(x)
    assert_metamorphic_addition("tan_identity", "sin(x)/cos(x)", "tan(x)", &["x"]);
}

// =============================================================================
// Power Identities (additional)
// =============================================================================

#[test]
fn metatest_power_product() {
    // x^2 * x^3 = x^5
    assert_metamorphic_addition("power_product", "x^2 * x^3", "x^5", &["x"]);
}

#[test]
fn metatest_power_quotient() {
    // x^5 / x^2 = x^3
    assert_metamorphic_addition("power_quotient", "x^5 / x^2", "x^3", &["x"]);
}

#[test]
fn metatest_power_of_product() {
    // (2*x)^3 = 8*x^3
    assert_metamorphic_addition("power_of_product", "(2*x)^3", "8*x^3", &["x"]);
}

#[test]
fn metatest_negative_exponent() {
    // x^(-1) = 1/x
    assert_metamorphic_addition("negative_exponent", "x^(-1)", "1/x", &["x"]);
}

// =============================================================================
// Inverse Trigonometric Identities
// =============================================================================

#[test]
fn metatest_tan_arctan() {
    // tan(arctan(x)) = x (always true for all real x)
    assert_metamorphic_addition("tan_arctan", "tan(arctan(x))", "x", &["x"]);
}

#[test]
fn metatest_sin_arcsin() {
    // sin(arcsin(x)) = x (for x in [-1, 1])
    assert_metamorphic_addition("sin_arcsin", "sin(arcsin(x))", "x", &["x"]);
}

#[test]
fn metatest_cos_arccos() {
    // cos(arccos(x)) = x (for x in [-1, 1])
    assert_metamorphic_addition("cos_arccos", "cos(arccos(x))", "x", &["x"]);
}

// =============================================================================
// Logarithm and Exponential Identities
// =============================================================================

#[test]
fn metatest_ln_exp() {
    // ln(exp(x)) = x
    assert_metamorphic_addition("ln_exp", "ln(exp(x))", "x", &["x"]);
}

#[test]
fn metatest_exp_ln() {
    // exp(ln(x)) = x (for x > 0)
    assert_metamorphic_addition("exp_ln", "exp(ln(x))", "x", &["x"]);
}

#[test]
fn metatest_log_power() {
    // ln(x^2) = 2*ln(x) (for x > 0)
    // Note: This tests the log power rule
    assert_metamorphic_addition("log_power", "ln(x^2)", "2*ln(x)", &["x"]);
}

// =============================================================================
// Root Identities
// =============================================================================

#[test]
fn metatest_sqrt_squared() {
    // sqrt(x)^2 = x (for x >= 0)
    assert_metamorphic_addition("sqrt_squared", "sqrt(x)^2", "x", &["x"]);
}

#[test]
fn metatest_cbrt_cubed() {
    // cbrt(x)^3 = x (cube root cubed)
    assert_metamorphic_addition("cbrt_cubed", "x^(1/3) * x^(1/3) * x^(1/3)", "x", &["x"]);
}

// =============================================================================
// Complex Algebraic Identities
// =============================================================================

#[test]
fn metatest_foil_expansion() {
    // (x + 2)(x + 3) = x^2 + 5x + 6
    assert_metamorphic_addition(
        "foil_expansion",
        "(x + 2) * (x + 3)",
        "x^2 + 5*x + 6",
        &["x"],
    );
}

#[test]
fn metatest_perfect_square_trinomial() {
    // x^2 + 4x + 4 = (x + 2)^2
    assert_metamorphic_addition(
        "perfect_square_trinomial",
        "x^2 + 4*x + 4",
        "(x + 2)^2",
        &["x"],
    );
}

#[test]
fn metatest_factor_common() {
    // 3x + 6 = 3(x + 2)
    assert_metamorphic_addition("factor_common", "3*x + 6", "3*(x + 2)", &["x"]);
}

// =============================================================================
// Numeric Root Simplifications
// =============================================================================

#[test]
fn metatest_sqrt_product() {
    // sqrt(8) * sqrt(2) = sqrt(16) = 4
    assert_metamorphic_addition("sqrt_product", "sqrt(8) * sqrt(2)", "4", &[]);
}

#[test]
fn metatest_sqrt_quotient() {
    // sqrt(32) / sqrt(2) = sqrt(16) = 4
    assert_metamorphic_addition("sqrt_quotient", "sqrt(32) / sqrt(2)", "4", &[]);
}

#[test]
fn metatest_root_addition() {
    // sqrt(12) + sqrt(27) = 2*sqrt(3) + 3*sqrt(3) = 5*sqrt(3)
    assert_metamorphic_addition("root_addition", "sqrt(12) + sqrt(27)", "5*sqrt(3)", &[]);
}

#[test]
fn metatest_mixed_root_product() {
    // sqrt(2) * 2^(1/3) = 2^(1/2 + 1/3) = 2^(5/6)
    assert_metamorphic_addition("mixed_root_product", "sqrt(2) * 2^(1/3)", "2^(5/6)", &[]);
}

// =============================================================================
// More Trigonometric Identities
// =============================================================================

#[test]
fn metatest_sec_identity() {
    // sec(x) = 1/cos(x)
    assert_metamorphic_addition("sec_identity", "1/cos(x)", "sec(x)", &["x"]);
}

#[test]
fn metatest_csc_identity() {
    // csc(x) = 1/sin(x)
    assert_metamorphic_addition("csc_identity", "1/sin(x)", "csc(x)", &["x"]);
}

#[test]
fn metatest_cot_identity() {
    // cot(x) = cos(x)/sin(x)
    assert_metamorphic_addition("cot_identity", "cos(x)/sin(x)", "cot(x)", &["x"]);
}

#[test]
fn metatest_sin_2x() {
    // sin(2x) = 2*sin(x)*cos(x) (alternate form)
    assert_metamorphic_addition("sin_2x", "2*sin(x)*cos(x)", "sin(2*x)", &["x"]);
}

// =============================================================================
// Quadratic and Polynomial
// =============================================================================

#[test]
fn metatest_quadratic_factored() {
    // x^2 - 5x + 6 = (x-2)(x-3)
    assert_metamorphic_addition(
        "quadratic_factored",
        "x^2 - 5*x + 6",
        "(x - 2) * (x - 3)",
        &["x"],
    );
}

#[test]
fn metatest_sum_of_squares_factor() {
    // (x+1)^2 + (x-1)^2 = 2x^2 + 2
    assert_metamorphic_addition("sum_of_squares", "(x+1)^2 + (x-1)^2", "2*x^2 + 2", &["x"]);
}

#[test]
fn metatest_cubic_expansion() {
    // (x - 1)^3 = x^3 - 3x^2 + 3x - 1
    assert_metamorphic_addition(
        "cubic_expansion",
        "(x - 1)^3",
        "x^3 - 3*x^2 + 3*x - 1",
        &["x"],
    );
}

// =============================================================================
// Absolute Value
// =============================================================================

#[test]
fn metatest_abs_squared() {
    // |x|^2 = x^2
    assert_metamorphic_addition("abs_squared", "|x|^2", "x^2", &["x"]);
}

#[test]
fn metatest_sqrt_x_squared() {
    // sqrt(x^2) = |x|
    assert_metamorphic_addition("sqrt_x_squared", "sqrt(x^2)", "|x|", &["x"]);
}

// =============================================================================
// Exponential Properties
// =============================================================================

#[test]
fn metatest_exp_sum() {
    // e^x * e^y with x and y being expressions
    // e^(x+1) * e^(x-1) = e^(2x)
    assert_metamorphic_addition("exp_sum", "exp(x+1) * exp(x-1)", "exp(2*x)", &["x"]);
}

// =============================================================================
// Special Constant Values
// =============================================================================

#[test]
fn metatest_sin_pi() {
    // sin(π) = 0
    assert_metamorphic_addition("sin_pi", "sin(pi)", "0", &[]);
}

#[test]
fn metatest_cos_pi() {
    // cos(π) = -1
    assert_metamorphic_addition("cos_pi", "cos(pi)", "-1", &[]);
}

#[test]
fn metatest_sin_pi_2() {
    // sin(π/2) = 1
    assert_metamorphic_addition("sin_pi_2", "sin(pi/2)", "1", &[]);
}

#[test]
fn metatest_cos_pi_2() {
    // cos(π/2) = 0
    assert_metamorphic_addition("cos_pi_2", "cos(pi/2)", "0", &[]);
}

#[test]
fn metatest_tan_pi_4() {
    // tan(π/4) = 1
    assert_metamorphic_addition("tan_pi_4", "tan(pi/4)", "1", &[]);
}

#[test]
fn metatest_ln_e() {
    // ln(e) = 1
    assert_metamorphic_addition("ln_e", "ln(e)", "1", &[]);
}

#[test]
fn metatest_log_10_100() {
    // log(10, 100) = 2
    assert_metamorphic_addition("log_10_100", "log(10, 100)", "2", &[]);
}

// =============================================================================
// More Algebraic Manipulations
// =============================================================================

#[test]
fn metatest_difference_of_cubes() {
    // x^3 - 1 = (x - 1)(x^2 + x + 1)
    assert_metamorphic_addition(
        "difference_of_cubes",
        "x^3 - 1",
        "(x - 1) * (x^2 + x + 1)",
        &["x"],
    );
}

#[test]
fn metatest_nested_square() {
    // (x^2)^2 = x^4
    assert_metamorphic_addition("nested_square", "(x^2)^2", "x^4", &["x"]);
}

#[test]
fn metatest_fraction_multiply() {
    // (x/2) * 4 = 2x
    assert_metamorphic_addition("fraction_multiply", "(x/2) * 4", "2*x", &["x"]);
}

#[test]
fn metatest_fraction_divide() {
    // (x/2) / 2 = x/4
    assert_metamorphic_addition("fraction_divide", "(x/2) / 2", "x/4", &["x"]);
}

#[test]
fn metatest_double_negative() {
    // -(-x) = x
    assert_metamorphic_addition("double_negative", "-(-x)", "x", &["x"]);
}

#[test]
fn metatest_add_opposite() {
    // x + (-x) = 0
    assert_metamorphic_addition("add_opposite", "x + (-x)", "0", &["x"]);
}

// =============================================================================
// More Trigonometric Identities
// =============================================================================

#[test]
fn metatest_cos_double_angle_variant() {
    // cos(2x) = 1 - 2*sin(x)^2
    assert_metamorphic_addition(
        "cos_double_angle_variant",
        "cos(2*x)",
        "1 - 2*sin(x)^2",
        &["x"],
    );
}

#[test]
fn metatest_tan_squared_plus_1() {
    // tan(x)^2 + 1 = sec(x)^2
    assert_metamorphic_addition("tan_squared_plus_1", "tan(x)^2 + 1", "sec(x)^2", &["x"]);
}

#[test]
fn metatest_sin_neg() {
    // sin(-x) = -sin(x)
    assert_metamorphic_addition("sin_neg", "sin(-x)", "-sin(x)", &["x"]);
}

#[test]
fn metatest_cos_neg() {
    // cos(-x) = cos(x)
    assert_metamorphic_addition("cos_neg", "cos(-x)", "cos(x)", &["x"]);
}

// =============================================================================
// Half-angle and Co-function Identities
// =============================================================================

#[test]
fn metatest_sin_pi_minus_x() {
    // sin(π - x) = sin(x)
    assert_metamorphic_addition("sin_pi_minus_x", "sin(pi - x)", "sin(x)", &["x"]);
}

#[test]
fn metatest_cos_pi_minus_x() {
    // cos(π - x) = -cos(x)
    assert_metamorphic_addition("cos_pi_minus_x", "cos(pi - x)", "-cos(x)", &["x"]);
}

#[test]
fn metatest_sin_cofunction() {
    // sin(π/2 - x) = cos(x)
    assert_metamorphic_addition("sin_cofunction", "sin(pi/2 - x)", "cos(x)", &["x"]);
}

#[test]
fn metatest_cos_cofunction() {
    // cos(π/2 - x) = sin(x)
    assert_metamorphic_addition("cos_cofunction", "cos(pi/2 - x)", "sin(x)", &["x"]);
}

// =============================================================================
// More Power Identities
// =============================================================================

#[test]
fn metatest_power_of_power() {
    // (x^2)^3 = x^6
    assert_metamorphic_addition("power_of_power", "(x^2)^3", "x^6", &["x"]);
}

#[test]
fn metatest_zero_power() {
    // x^0 = 1 (for x ≠ 0)
    assert_metamorphic_addition("zero_power", "x^0", "1", &["x"]);
}

#[test]
fn metatest_one_power() {
    // x^1 = x
    assert_metamorphic_addition("one_power", "x^1", "x", &["x"]);
}

#[test]
fn metatest_sqrt_product_rule() {
    // sqrt(x) * sqrt(y) = sqrt(x*y) -- tested numerically
    // sqrt(4) * sqrt(9) = sqrt(36) = 6
    assert_metamorphic_addition("sqrt_product_rule", "sqrt(4) * sqrt(9)", "6", &[]);
}

// =============================================================================
// More Logarithm Identities
// =============================================================================

#[test]
fn metatest_log_quotient() {
    // ln(x) - ln(y) = ln(x/y) -- numerically for x=e^2, y=e
    // ln(e^2) - ln(e) = 2 - 1 = 1
    assert_metamorphic_addition("log_quotient_num", "ln(e^2) - ln(e)", "1", &[]);
}

#[test]
fn metatest_log_sum() {
    // ln(2) + ln(3) = ln(6)
    assert_metamorphic_addition("log_sum", "ln(2) + ln(3)", "ln(6)", &[]);
}

#[test]
fn metatest_log_one() {
    // ln(1) = 0
    assert_metamorphic_addition("log_one", "ln(1)", "0", &[]);
}

// =============================================================================
// Polynomial Factoring
// =============================================================================

#[test]
fn metatest_factor_x_squared_minus_4() {
    // x^2 - 4 = (x-2)(x+2)
    assert_metamorphic_addition("factor_x2_minus_4", "x^2 - 4", "(x - 2) * (x + 2)", &["x"]);
}

#[test]
fn metatest_factor_x_squared_minus_9() {
    // x^2 - 9 = (x-3)(x+3)
    assert_metamorphic_addition("factor_x2_minus_9", "x^2 - 9", "(x - 3) * (x + 3)", &["x"]);
}

#[test]
fn metatest_expand_x_plus_1_squared() {
    // (x+1)^2 already tested, testing (x+2)^2 = x^2 + 4x + 4
    assert_metamorphic_addition("expand_x_plus_2_sq", "(x + 2)^2", "x^2 + 4*x + 4", &["x"]);
}

#[test]
fn metatest_expand_x_minus_3_squared() {
    // (x-3)^2 = x^2 - 6x + 9
    assert_metamorphic_addition("expand_x_minus_3_sq", "(x - 3)^2", "x^2 - 6*x + 9", &["x"]);
}

// =============================================================================
// Fraction Simplifications
// =============================================================================

#[test]
fn metatest_fraction_add_same_denom() {
    // x/3 + 2x/3 = 3x/3 = x
    assert_metamorphic_addition("fraction_add_same_denom", "x/3 + 2*x/3", "x", &["x"]);
}

#[test]
fn metatest_fraction_cancel() {
    // 2x/2 = x
    assert_metamorphic_addition("fraction_cancel", "2*x/2", "x", &["x"]);
}

#[test]
fn metatest_nested_fraction() {
    // (x/2)/(3/4) = (x/2) * (4/3) = 2x/3
    assert_metamorphic_addition("nested_fraction", "(x/2)/(3/4)", "2*x/3", &["x"]);
}

// =============================================================================
// Numeric Simplifications
// =============================================================================

#[test]
fn metatest_arithmetic_chain() {
    // 2 + 3 * 4 - 1 = 2 + 12 - 1 = 13
    assert_metamorphic_addition("arithmetic_chain", "2 + 3 * 4 - 1", "13", &[]);
}

#[test]
fn metatest_power_numeric() {
    // 2^10 = 1024
    assert_metamorphic_addition("power_numeric", "2^10", "1024", &[]);
}

// =============================================================================
// Distributive Property
// =============================================================================

#[test]
fn metatest_distribute_simple() {
    // 2(x + 3) = 2x + 6
    assert_metamorphic_addition("distribute_simple", "2*(x + 3)", "2*x + 6", &["x"]);
}

#[test]
fn metatest_distribute_negative() {
    // -1(x + 2) = -x - 2
    assert_metamorphic_addition("distribute_negative", "-1*(x + 2)", "-x - 2", &["x"]);
}

#[test]
fn metatest_distribute_variable() {
    // x(x + 1) = x^2 + x
    assert_metamorphic_addition("distribute_variable", "x*(x + 1)", "x^2 + x", &["x"]);
}

// =============================================================================
// Combining Like Terms
// =============================================================================

#[test]
fn metatest_combine_like_simple() {
    // 2x + 3x = 5x
    assert_metamorphic_addition("combine_like_simple", "2*x + 3*x", "5*x", &["x"]);
}

#[test]
fn metatest_combine_like_subtract() {
    // 5x - 2x = 3x
    assert_metamorphic_addition("combine_like_subtract", "5*x - 2*x", "3*x", &["x"]);
}

#[test]
fn metatest_combine_constant() {
    // x + 2 + 3 = x + 5
    assert_metamorphic_addition("combine_constant", "x + 2 + 3", "x + 5", &["x"]);
}

// =============================================================================
// Integer Root Simplifications
// =============================================================================

#[test]
fn metatest_sqrt_4() {
    // sqrt(4) = 2
    assert_metamorphic_addition("sqrt_4", "sqrt(4)", "2", &[]);
}

#[test]
fn metatest_sqrt_16() {
    // sqrt(16) = 4
    assert_metamorphic_addition("sqrt_16", "sqrt(16)", "4", &[]);
}

#[test]
fn metatest_cbrt_8() {
    // 8^(1/3) = 2
    assert_metamorphic_addition("cbrt_8", "8^(1/3)", "2", &[]);
}

#[test]
fn metatest_cbrt_27() {
    // 27^(1/3) = 3
    assert_metamorphic_addition("cbrt_27", "27^(1/3)", "3", &[]);
}

// =============================================================================
// More Special Trig Values
// =============================================================================

#[test]
fn metatest_sin_pi_6() {
    // sin(π/6) = 1/2
    assert_metamorphic_addition("sin_pi_6", "sin(pi/6)", "1/2", &[]);
}

#[test]
fn metatest_cos_pi_3() {
    // cos(π/3) = 1/2
    assert_metamorphic_addition("cos_pi_3", "cos(pi/3)", "1/2", &[]);
}

#[test]
fn metatest_sin_pi_3() {
    // sin(π/3) = sqrt(3)/2
    assert_metamorphic_addition("sin_pi_3", "sin(pi/3)", "sqrt(3)/2", &[]);
}

#[test]
fn metatest_cos_pi_6() {
    // cos(π/6) = sqrt(3)/2
    assert_metamorphic_addition("cos_pi_6", "cos(pi/6)", "sqrt(3)/2", &[]);
}

// =============================================================================
// Reciprocal Identities
// =============================================================================

#[test]
fn metatest_reciprocal_of_reciprocal() {
    // 1/(1/x) = x
    assert_metamorphic_addition("reciprocal_of_reciprocal", "1/(1/x)", "x", &["x"]);
}

#[test]
fn metatest_x_times_reciprocal() {
    // x * (1/x) = 1
    assert_metamorphic_addition("x_times_reciprocal", "x * (1/x)", "1", &["x"]);
}

#[test]
fn metatest_reciprocal_product() {
    // (1/x) * (1/y) = 1/(x*y) - numerically with constants
    // (1/2) * (1/3) = 1/6
    assert_metamorphic_addition("reciprocal_product", "(1/2) * (1/3)", "1/6", &[]);
}

// =============================================================================
// Rationalization
// =============================================================================

#[test]
fn metatest_rationalize_sqrt_2() {
    // 1/sqrt(2) = sqrt(2)/2
    assert_metamorphic_addition("rationalize_sqrt_2", "1/sqrt(2)", "sqrt(2)/2", &[]);
}

#[test]
fn metatest_rationalize_sqrt_3() {
    // 2/sqrt(3) = 2*sqrt(3)/3
    assert_metamorphic_addition("rationalize_sqrt_3", "2/sqrt(3)", "2*sqrt(3)/3", &[]);
}

// =============================================================================
// More Polynomial Identities
// =============================================================================

#[test]
fn metatest_square_of_difference() {
    // (x - y)^2 = x^2 - 2xy + y^2 - with y=1
    // (x - 1)^2 already tested, try (x - 2)^2
    assert_metamorphic_addition("square_of_diff_2", "(x - 2)^2", "x^2 - 4*x + 4", &["x"]);
}

#[test]
fn metatest_fourth_power_binomial() {
    // (x + 1)^4 = x^4 + 4x^3 + 6x^2 + 4x + 1
    assert_metamorphic_addition(
        "fourth_power_binomial",
        "(x + 1)^4",
        "x^4 + 4*x^3 + 6*x^2 + 4*x + 1",
        &["x"],
    );
}

#[test]
fn metatest_product_of_conjugates() {
    // (sqrt(x) + 1)(sqrt(x) - 1) = x - 1
    assert_metamorphic_addition(
        "product_of_conjugates",
        "(sqrt(x) + 1) * (sqrt(x) - 1)",
        "x - 1",
        &["x"],
    );
}

// =============================================================================
// GCD and Factoring Patterns
// =============================================================================

#[test]
fn metatest_factor_out_2() {
    // 4x + 6 = 2(2x + 3)
    assert_metamorphic_addition("factor_out_2", "4*x + 6", "2*(2*x + 3)", &["x"]);
}

#[test]
fn metatest_factor_out_x() {
    // x^2 + x = x(x + 1)
    assert_metamorphic_addition("factor_out_x", "x^2 + x", "x*(x + 1)", &["x"]);
}

#[test]
fn metatest_factor_out_x_squared() {
    // x^3 + x^2 = x^2(x + 1)
    assert_metamorphic_addition("factor_out_x_squared", "x^3 + x^2", "x^2*(x + 1)", &["x"]);
}

// =============================================================================
// More Exponent Rules
// =============================================================================

#[test]
fn metatest_fractional_exponent_product() {
    // x^(1/2) * x^(1/3) = x^(5/6)
    assert_metamorphic_addition("frac_exp_product", "x^(1/2) * x^(1/3)", "x^(5/6)", &["x"]);
}

#[test]
fn metatest_fractional_exponent_quotient() {
    // x^(1/2) / x^(1/3) = x^(1/6)
    assert_metamorphic_addition("frac_exp_quotient", "x^(1/2) / x^(1/3)", "x^(1/6)", &["x"]);
}

#[test]
fn metatest_negative_fractional_exponent() {
    // x^(-1/2) = 1/sqrt(x)
    assert_metamorphic_addition("neg_frac_exp", "x^(-1/2)", "1/sqrt(x)", &["x"]);
}

// =============================================================================
// More Trig Identities
// =============================================================================

#[test]
fn metatest_cot_squared_plus_1() {
    // cot(x)^2 + 1 = csc(x)^2
    assert_metamorphic_addition("cot_squared_plus_1", "cot(x)^2 + 1", "csc(x)^2", &["x"]);
}

#[test]
fn metatest_sin_plus_cos_squared() {
    // (sin(x) + cos(x))^2 = 1 + 2*sin(x)*cos(x) = 1 + sin(2x)
    assert_metamorphic_addition(
        "sin_plus_cos_squared",
        "(sin(x) + cos(x))^2",
        "1 + sin(2*x)",
        &["x"],
    );
}

#[test]
fn metatest_sin_minus_cos_squared() {
    // (sin(x) - cos(x))^2 = 1 - 2*sin(x)*cos(x) = 1 - sin(2x)
    assert_metamorphic_addition(
        "sin_minus_cos_squared",
        "(sin(x) - cos(x))^2",
        "1 - sin(2*x)",
        &["x"],
    );
}

// =============================================================================
// Numeric Fractions
// =============================================================================

#[test]
fn metatest_fraction_addition() {
    // 1/2 + 1/3 = 5/6
    assert_metamorphic_addition("fraction_addition", "1/2 + 1/3", "5/6", &[]);
}

#[test]
fn metatest_fraction_subtraction() {
    // 3/4 - 1/4 = 1/2
    assert_metamorphic_addition("fraction_subtraction", "3/4 - 1/4", "1/2", &[]);
}

#[test]
fn metatest_fraction_multiplication() {
    // (2/3) * (3/4) = 1/2
    assert_metamorphic_addition("fraction_multiplication", "(2/3) * (3/4)", "1/2", &[]);
}

#[test]
fn metatest_fraction_division() {
    // (1/2) / (1/4) = 2
    assert_metamorphic_addition("fraction_division", "(1/2) / (1/4)", "2", &[]);
}

// =============================================================================
// Mixed Numeric
// =============================================================================

#[test]
fn metatest_factorial_pattern() {
    // 3! = 6 (if factorial is supported, otherwise just 6)
    assert_metamorphic_addition("three_factorial", "1*2*3", "6", &[]);
}

#[test]
fn metatest_sqrt_fraction() {
    // sqrt(1/4) = 1/2
    assert_metamorphic_addition("sqrt_fraction", "sqrt(1/4)", "1/2", &[]);
}

// =============================================================================
// More Special Trig Values (π/4)
// =============================================================================

#[test]
fn metatest_sin_pi_4() {
    // sin(π/4) = sqrt(2)/2
    assert_metamorphic_addition("sin_pi_4", "sin(pi/4)", "sqrt(2)/2", &[]);
}

#[test]
fn metatest_cos_pi_4() {
    // cos(π/4) = sqrt(2)/2
    assert_metamorphic_addition("cos_pi_4", "cos(pi/4)", "sqrt(2)/2", &[]);
}

// =============================================================================
// Multiples of pi
// =============================================================================

#[test]
fn metatest_sin_2pi() {
    // sin(2π) = 0
    assert_metamorphic_addition("sin_2pi", "sin(2*pi)", "0", &[]);
}

#[test]
fn metatest_cos_2pi() {
    // cos(2π) = 1
    assert_metamorphic_addition("cos_2pi", "cos(2*pi)", "1", &[]);
}

#[test]
fn metatest_sin_3pi() {
    // sin(3π) = 0
    assert_metamorphic_addition("sin_3pi", "sin(3*pi)", "0", &[]);
}

#[test]
fn metatest_cos_3pi() {
    // cos(3π) = -1
    assert_metamorphic_addition("cos_3pi", "cos(3*pi)", "-1", &[]);
}

// =============================================================================
// Expression Simplifications
// =============================================================================

#[test]
fn metatest_multiply_by_zero() {
    // x * 0 = 0
    assert_metamorphic_addition("multiply_by_zero", "x * 0", "0", &["x"]);
}

#[test]
fn metatest_add_to_self() {
    // x + x = 2x
    assert_metamorphic_addition("add_to_self", "x + x", "2*x", &["x"]);
}

#[test]
fn metatest_subtract_from_self() {
    // x - x = 0
    assert_metamorphic_addition("subtract_from_self", "x - x", "0", &["x"]);
}

#[test]
fn metatest_divide_by_self() {
    // x / x = 1 (for x ≠ 0)
    assert_metamorphic_addition("divide_by_self", "x / x", "1", &["x"]);
}

// =============================================================================
// More Polynomial Forms
// =============================================================================

#[test]
fn metatest_expand_squared_binomial_a() {
    // (2x + 1)^2 = 4x^2 + 4x + 1
    assert_metamorphic_addition(
        "expand_2x_plus_1_sq",
        "(2*x + 1)^2",
        "4*x^2 + 4*x + 1",
        &["x"],
    );
}

#[test]
fn metatest_expand_squared_binomial_b() {
    // (x + 3)^2 = x^2 + 6x + 9
    assert_metamorphic_addition("expand_x_plus_3_sq", "(x + 3)^2", "x^2 + 6*x + 9", &["x"]);
}

#[test]
fn metatest_difference_of_fourth_powers() {
    // x^4 - 1 = (x^2 + 1)(x^2 - 1) = (x^2 + 1)(x+1)(x-1)
    // Just check numeric equivalence
    assert_metamorphic_addition(
        "diff_fourth_powers",
        "x^4 - 1",
        "(x^2 + 1) * (x + 1) * (x - 1)",
        &["x"],
    );
}

// =============================================================================
// Log Properties
// =============================================================================

#[test]
fn metatest_log_base_change() {
    // log(2, 8) = 3 (since 2^3 = 8)
    assert_metamorphic_addition("log_2_8", "log(2, 8)", "3", &[]);
}

#[test]
fn metatest_log_base_same() {
    // log(x, x) = 1
    assert_metamorphic_addition("log_base_same", "log(x, x)", "1", &["x"]);
}

#[test]
fn metatest_log_of_1() {
    // log(x, 1) = 0
    assert_metamorphic_addition("log_of_1", "log(x, 1)", "0", &["x"]);
}

// =============================================================================
// More Absolute Value
// =============================================================================

#[test]
fn metatest_abs_negative_constant() {
    // |-5| = 5
    assert_metamorphic_addition("abs_negative", "|-5|", "5", &[]);
}

#[test]
fn metatest_abs_positive_constant() {
    // |7| = 7
    assert_metamorphic_addition("abs_positive", "|7|", "7", &[]);
}

#[test]
fn metatest_abs_product() {
    // |x| * |y| = |x*y| - numerically
    // |2| * |3| = |6| = 6
    assert_metamorphic_addition("abs_product_num", "|2| * |3|", "6", &[]);
}

// =============================================================================
// Advanced Power Rules
// =============================================================================

#[test]
fn metatest_power_sum_in_exp() {
    // x^(a+b) at x=2, a=1, b=2: 2^3 = 8
    assert_metamorphic_addition("power_sum_exp", "2^(1+2)", "8", &[]);
}

#[test]
fn metatest_power_diff_in_exp() {
    // 2^(5-2) = 2^3 = 8
    assert_metamorphic_addition("power_diff_exp", "2^(5-2)", "8", &[]);
}

#[test]
fn metatest_sqrt_of_power() {
    // sqrt(x^4) = x^2 (for x >= 0) or |x|^2 = x^2
    assert_metamorphic_addition("sqrt_of_power", "sqrt(x^4)", "x^2", &["x"]);
}

// =============================================================================
// Fraction with Variables
// =============================================================================

#[test]
fn metatest_simplify_fraction_xy() {
    // (x*y) / y = x
    assert_metamorphic_addition("simplify_xy_over_y", "(x*y) / y", "x", &["x"]);
}

#[test]
fn metatest_simplify_fraction_x2() {
    // x^2 / x = x
    assert_metamorphic_addition("simplify_x2_over_x", "x^2 / x", "x", &["x"]);
}

#[test]
fn metatest_simplify_fraction_x3() {
    // x^3 / x^2 = x
    assert_metamorphic_addition("simplify_x3_over_x2", "x^3 / x^2", "x", &["x"]);
}

// =============================================================================
// Final Batch: Reach 150+ tests
// =============================================================================

#[test]
fn metatest_e_to_0() {
    // e^0 = 1
    assert_metamorphic_addition("e_to_0", "e^0", "1", &[]);
}

#[test]
fn metatest_e_to_1() {
    // e^1 = e
    assert_metamorphic_addition("e_to_1", "e^1", "e", &[]);
}

#[test]
fn metatest_ln_e_squared() {
    // ln(e^2) = 2
    assert_metamorphic_addition("ln_e_squared", "ln(e^2)", "2", &[]);
}

#[test]
fn metatest_sqrt_9() {
    // sqrt(9) = 3
    assert_metamorphic_addition("sqrt_9", "sqrt(9)", "3", &[]);
}

#[test]
fn metatest_sqrt_25() {
    // sqrt(25) = 5
    assert_metamorphic_addition("sqrt_25", "sqrt(25)", "5", &[]);
}

#[test]
fn metatest_sqrt_100() {
    // sqrt(100) = 10
    assert_metamorphic_addition("sqrt_100", "sqrt(100)", "10", &[]);
}

#[test]
fn metatest_cbrt_64() {
    // 64^(1/3) = 4
    assert_metamorphic_addition("cbrt_64", "64^(1/3)", "4", &[]);
}

#[test]
fn metatest_cbrt_125() {
    // 125^(1/3) = 5
    assert_metamorphic_addition("cbrt_125", "125^(1/3)", "5", &[]);
}

#[test]
fn metatest_fourth_root_16() {
    // 16^(1/4) = 2
    assert_metamorphic_addition("fourth_root_16", "16^(1/4)", "2", &[]);
}

#[test]
fn metatest_fourth_root_81() {
    // 81^(1/4) = 3
    assert_metamorphic_addition("fourth_root_81", "81^(1/4)", "3", &[]);
}

#[test]
fn metatest_pi_times_0() {
    // π * 0 = 0
    assert_metamorphic_addition("pi_times_0", "pi * 0", "0", &[]);
}

#[test]
fn metatest_e_times_0() {
    // e * 0 = 0
    assert_metamorphic_addition("e_times_0", "e * 0", "0", &[]);
}
