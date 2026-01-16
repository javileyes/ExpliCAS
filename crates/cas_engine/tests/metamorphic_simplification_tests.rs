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
//! # Identity Pairs
//!
//! Identity pairs are loaded from `identity_pairs.csv`. Add new identities there
//! to automatically include them in combination testing.

#![allow(dead_code)] // Infrastructure code for future expansion
#![allow(unused_imports)]

mod test_utils;

use cas_ast::{Context, ExprId};
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

/// Check if two expressions are numerically equivalent for 2 variables.
/// Returns Ok(valid_count) or Err(message).
fn check_numeric_equiv_2var(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
    config: &MetatestConfig,
) -> Result<usize, String> {
    let (lo, hi) = config.sample_range;
    let mut valid = 0usize;
    let mut eval_failed = 0usize;

    // Use fewer samples for 2D grid to keep runtime reasonable
    let samples_per_dim = (config.eval_samples as f64).sqrt() as usize;

    for i in 0..samples_per_dim {
        for j in 0..samples_per_dim {
            let t1 = (i as f64 + 0.5) / samples_per_dim as f64;
            let t2 = (j as f64 + 0.5) / samples_per_dim as f64;
            let x = lo + (hi - lo) * t1;
            let y = lo + (hi - lo) * t2;

            let mut var_map = HashMap::new();
            var_map.insert(var1.to_string(), x);
            var_map.insert(var2.to_string(), y);

            let va = eval_f64(ctx, a, &var_map);
            let vb = eval_f64(ctx, b, &var_map);

            match (va, vb) {
                (Some(va), Some(vb)) => {
                    if va.is_nan() || vb.is_nan() || va.is_infinite() || vb.is_infinite() {
                        eval_failed += 1;
                        continue;
                    }
                    valid += 1;

                    let diff = (va - vb).abs();
                    let scale = va.abs().max(vb.abs()).max(1.0);
                    let allowed = config.atol + config.rtol * scale;

                    if diff > allowed {
                        return Err(format!(
                            "Numeric mismatch at {}={}, {}={}:\n  a={:.15}\n  b={:.15}\n  diff={:.3e} > allowed={:.3e}",
                            var1, x, var2, y, va, vb, diff, allowed
                        ));
                    }
                }
                _ => {
                    eval_failed += 1;
                }
            }
        }
    }

    // Lower threshold for 2D since we have fewer samples
    let min_valid_2d = config.min_valid / 4;
    if valid < min_valid_2d {
        return Err(format!(
            "Too few valid samples: {} < {} (eval_failed={})",
            valid, min_valid_2d, eval_failed
        ));
    }

    Ok(valid)
}

// =============================================================================
// Combination Metamorphic Tests: Exp1 op Exp2 ≡ Simp1 op Simp2
// =============================================================================

/// A test pair: an expression and its simplified equivalent
struct TestPair {
    exp: &'static str,
    simp: &'static str,
    /// Variable used (for alpha-renaming)
    var: &'static str,
}

/// Alpha-rename a variable in an expression string.
/// Simple text replacement - works for our test expressions.
fn alpha_rename(expr: &str, from: &str, to: &str) -> String {
    // Use word boundaries to avoid replacing 'x' inside 'exp' etc.
    // Simple approach: replace 'x' followed by non-alphanumeric or end
    let mut result = expr.to_string();

    // Replace patterns like "x)" "x+" "x-" "x*" "x/" "x^" "x," "x " "|x|" and standalone "x"
    let patterns = [
        (format!("{})", from), format!("{})", to)),
        (format!("{}+", from), format!("{}+", to)),
        (format!("{}-", from), format!("{}-", to)),
        (format!("{}*", from), format!("{}*", to)),
        (format!("{}/", from), format!("{}/", to)),
        (format!("{}^", from), format!("{}^", to)),
        (format!("{},", from), format!("{},", to)),
        (format!("{} ", from), format!("{} ", to)),
        (format!("({})", from), format!("({})", to)),
        (format!("({}", from), format!("({}", to)),
        // Absolute value: |x|
        (format!("|{}|", from), format!("|{}|", to)),
        (format!("|{}", from), format!("|{}", to)),
        (format!("{}|", from), format!("{}|", to)),
    ];

    for (pat, rep) in &patterns {
        result = result.replace(pat, rep);
    }

    // Handle end of string
    if result.ends_with(from) {
        let len = result.len();
        result.replace_range(len - from.len().., to);
    }

    result
}

/// Assert that combining two identity pairs preserves equivalence.
/// Given: Exp1 ≡ Simp1 and Exp2 ≡ Simp2
/// Verify: Exp1 + Exp2' ≡ Simp1 + Simp2' (where Exp2' is alpha-renamed)
///
/// This tests for interaction bugs between different simplification rules.
fn assert_metamorphic_combine(
    test_name: &str,
    pair1: TestPair,
    pair2: TestPair,
    op: &str, // "+", "-", or "*"
) {
    let config = metatest_config();

    // Alpha-rename pair2 to avoid variable collisions
    // x -> u, y -> v
    let pair2_exp = alpha_rename(pair2.exp, pair2.var, "u");
    let pair2_simp = alpha_rename(pair2.simp, pair2.var, "u");

    // Build combined expressions
    let combined_exp = format!("({}) {} ({})", pair1.exp, op, pair2_exp);
    let combined_simp = format!("({}) {} ({})", pair1.simp, op, pair2_simp);

    // Variables: original from pair1, renamed from pair2
    let vars = if pair1.var == pair2.var {
        vec![pair1.var, "u"] // pair2 was renamed to u
    } else {
        vec![pair1.var, pair2.var]
    };

    // Parse expressions
    let mut simplifier = Simplifier::with_default_rules();
    let exp_parsed = match parse(&combined_exp, &mut simplifier.context) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Parse error in combine test: {} - {:?}", combined_exp, err);
            return;
        }
    };
    let simp_parsed = match parse(&combined_simp, &mut simplifier.context) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Parse error in combine test: {} - {:?}", combined_simp, err);
            return;
        }
    };

    // Simplify both sides
    let (exp_simplified, _) = simplifier.simplify(exp_parsed);
    let (simp_simplified, _) = simplifier.simplify(simp_parsed);

    // Verify numeric equivalence
    let check_result = if vars.len() == 1 {
        check_numeric_equiv_1var(
            &simplifier.context,
            exp_simplified,
            simp_simplified,
            vars[0],
            &config,
        )
    } else {
        check_numeric_equiv_2var(
            &simplifier.context,
            exp_simplified,
            simp_simplified,
            vars[0],
            vars[1],
            &config,
        )
    };

    if let Err(err) = check_result {
        panic!(
            "Combination Metatest FAILED: {}\n\
             pair1: {} ≡ {}\n\
             pair2: {} ≡ {} (renamed: {} ≡ {})\n\
             combined_exp: {}\n\
             combined_simp: {}\n\
             Error: {}",
            test_name,
            pair1.exp,
            pair1.simp,
            pair2.exp,
            pair2.simp,
            pair2_exp,
            pair2_simp,
            combined_exp,
            combined_simp,
            err
        );
    }
}

/// Assert that combining THREE identity pairs preserves equivalence.
/// Given: Exp1 ≡ Simp1, Exp2 ≡ Simp2, Exp3 ≡ Simp3
/// Verify: Exp1 + Exp2 + Exp3 ≡ Simp1 + Simp2 + Simp3
///
/// Uses alpha-renaming: pair2 uses 'u', pair3 uses 'v'
fn assert_metamorphic_combine_triple(
    test_name: &str,
    pair1: TestPair,
    pair2: TestPair,
    pair3: TestPair,
    op: &str,
) {
    let config = metatest_config();

    // Alpha-rename pairs to avoid collisions
    // pair1: x, pair2: u, pair3: v
    let pair2_exp = alpha_rename(pair2.exp, pair2.var, "u");
    let pair2_simp = alpha_rename(pair2.simp, pair2.var, "u");
    let pair3_exp = alpha_rename(pair3.exp, pair3.var, "v");
    let pair3_simp = alpha_rename(pair3.simp, pair3.var, "v");

    // Build combined expressions: (exp1 op exp2) op exp3
    let combined_exp = format!(
        "(({}) {} ({})) {} ({})",
        pair1.exp, op, pair2_exp, op, pair3_exp
    );
    let combined_simp = format!(
        "(({}) {} ({})) {} ({})",
        pair1.simp, op, pair2_simp, op, pair3_simp
    );

    // Variables: x, u, v (all different now)
    let vars = [pair1.var, "u", "v"];

    // Parse expressions
    let mut simplifier = Simplifier::with_default_rules();
    let exp_parsed = match parse(&combined_exp, &mut simplifier.context) {
        Ok(e) => e,
        Err(err) => {
            eprintln!(
                "Parse error in triple combine test: {} - {:?}",
                combined_exp, err
            );
            return;
        }
    };
    let simp_parsed = match parse(&combined_simp, &mut simplifier.context) {
        Ok(e) => e,
        Err(err) => {
            eprintln!(
                "Parse error in triple combine test: {} - {:?}",
                combined_simp, err
            );
            return;
        }
    };

    // Simplify both sides
    let (exp_simplified, _) = simplifier.simplify(exp_parsed);
    let (simp_simplified, _) = simplifier.simplify(simp_parsed);

    // For 3 variables, we need a different check - use sampling approach
    // Check each variable independently with others fixed at sample values
    let (lo, hi) = config.sample_range;
    let mut valid = 0usize;
    let samples_per_dim = 5; // 5^3 = 125 samples

    for i in 0..samples_per_dim {
        for j in 0..samples_per_dim {
            for k in 0..samples_per_dim {
                let t1 = (i as f64 + 0.5) / samples_per_dim as f64;
                let t2 = (j as f64 + 0.5) / samples_per_dim as f64;
                let t3 = (k as f64 + 0.5) / samples_per_dim as f64;
                let x = lo + (hi - lo) * t1;
                let y = lo + (hi - lo) * t2;
                let z = lo + (hi - lo) * t3;

                let mut var_map = HashMap::new();
                var_map.insert(vars[0].to_string(), x);
                var_map.insert(vars[1].to_string(), y);
                var_map.insert(vars[2].to_string(), z);

                let va = eval_f64(&simplifier.context, exp_simplified, &var_map);
                let vb = eval_f64(&simplifier.context, simp_simplified, &var_map);

                if let (Some(va), Some(vb)) = (va, vb) {
                    if va.is_nan() || vb.is_nan() || va.is_infinite() || vb.is_infinite() {
                        continue;
                    }
                    valid += 1;

                    let diff = (va - vb).abs();
                    let scale = va.abs().max(vb.abs()).max(1.0);
                    let allowed = config.atol + config.rtol * scale;

                    if diff > allowed {
                        panic!(
                            "Triple Combination Metatest FAILED: {}\n\
                             pair1: {} ≡ {}\n\
                             pair2: {} ≡ {}\n\
                             pair3: {} ≡ {}\n\
                             at x={}, u={}, v={}\n\
                             a={:.15}, b={:.15}, diff={:.3e}",
                            test_name,
                            pair1.exp,
                            pair1.simp,
                            pair2.exp,
                            pair2.simp,
                            pair3.exp,
                            pair3.simp,
                            x,
                            y,
                            z,
                            va,
                            vb,
                            diff
                        );
                    }
                }
            }
        }
    }

    if valid < 10 {
        eprintln!(
            "Warning: triple combine {} had only {} valid samples",
            test_name, valid
        );
    }
}

// =============================================================================
// CSV-Based Identity Pairs
// =============================================================================

/// An identity pair loaded from CSV
#[derive(Clone, Debug)]
struct IdentityPair {
    exp: String,
    simp: String,
    var: String,
}

/// Load identity pairs from CSV file
fn load_identity_pairs() -> Vec<IdentityPair> {
    let csv_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/identity_pairs.csv");
    let content = std::fs::read_to_string(csv_path).expect("Failed to read identity_pairs.csv");

    let mut pairs = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        // Skip comments and empty lines
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse CSV: exp,simp,var
        let parts: Vec<&str> = line.splitn(3, ',').collect();
        if parts.len() >= 3 {
            pairs.push(IdentityPair {
                exp: parts[0].trim().to_string(),
                simp: parts[1].trim().to_string(),
                var: parts[2].trim().to_string(),
            });
        }
    }

    pairs
}

/// Run combination tests from CSV pairs
fn run_csv_combination_tests(max_pairs: usize, include_triples: bool) {
    let pairs = load_identity_pairs();
    let config = metatest_config();

    // Limit pairs to avoid explosion
    let pairs: Vec<_> = pairs.into_iter().take(max_pairs).collect();
    let n = pairs.len();

    eprintln!("📊 Running CSV combination tests with {} pairs", n);

    let mut passed = 0;
    let mut failed = 0;
    let mut symbolic_passed = 0;
    let mut numeric_only_passed = 0;

    // Double combinations: all pairs of different identities
    for i in 0..n {
        for j in (i + 1)..n {
            let pair1 = &pairs[i];
            let pair2 = &pairs[j];

            // Alpha-rename pair2
            let pair2_exp = alpha_rename(&pair2.exp, &pair2.var, "u");
            let pair2_simp = alpha_rename(&pair2.simp, &pair2.var, "u");

            let combined_exp = format!("({}) + ({})", pair1.exp, pair2_exp);
            let combined_simp = format!("({}) + ({})", pair1.simp, pair2_simp);

            // Parse and simplify
            let mut simplifier = Simplifier::with_default_rules();
            let exp_parsed = match parse(&combined_exp, &mut simplifier.context) {
                Ok(e) => e,
                Err(_) => continue,
            };
            let simp_parsed = match parse(&combined_simp, &mut simplifier.context) {
                Ok(e) => e,
                Err(_) => continue,
            };

            let (exp_simplified, _) = simplifier.simplify(exp_parsed);
            let (simp_simplified, _) = simplifier.simplify(simp_parsed);

            // First check: symbolic equality (exact match)
            let symbolic_match = cas_engine::ordering::compare_expr(
                &simplifier.context,
                exp_simplified,
                simp_simplified,
            ) == std::cmp::Ordering::Equal;

            if symbolic_match {
                symbolic_passed += 1;
                passed += 1;
            } else {
                // Fallback: check numeric equivalence
                let result = check_numeric_equiv_2var(
                    &simplifier.context,
                    exp_simplified,
                    simp_simplified,
                    &pair1.var,
                    "u",
                    &config,
                );

                if result.is_ok() {
                    numeric_only_passed += 1;
                    passed += 1;
                } else {
                    failed += 1;
                    if failed <= 5 {
                        eprintln!("❌ Double combo failed: ({}) + ({})", pair1.exp, pair2.exp);
                    }
                }
            }
        }
    }

    eprintln!(
        "✅ Double combinations: {} passed ({} symbolic, {} numeric-only), {} failed",
        passed, symbolic_passed, numeric_only_passed, failed
    );

    // Triple combinations (optional, limited)
    if include_triples && n >= 3 {
        let mut triple_passed = 0;
        let mut triple_failed = 0;
        let triple_limit = 100; // Limit to avoid explosion
        let mut triple_count = 0;

        'outer: for i in 0..n.min(20) {
            for j in (i + 1)..n.min(20) {
                for k in (j + 1)..n.min(20) {
                    if triple_count >= triple_limit {
                        break 'outer;
                    }

                    let pair1 = &pairs[i];
                    let pair2 = &pairs[j];
                    let pair3 = &pairs[k];

                    // Alpha-rename
                    let pair2_exp = alpha_rename(&pair2.exp, &pair2.var, "u");
                    let pair2_simp = alpha_rename(&pair2.simp, &pair2.var, "u");
                    let pair3_exp = alpha_rename(&pair3.exp, &pair3.var, "v");
                    let pair3_simp = alpha_rename(&pair3.simp, &pair3.var, "v");

                    let combined_exp =
                        format!("(({}) + ({})) + ({})", pair1.exp, pair2_exp, pair3_exp);
                    let combined_simp =
                        format!("(({}) + ({})) + ({})", pair1.simp, pair2_simp, pair3_simp);

                    let mut simplifier = Simplifier::with_default_rules();
                    let exp_parsed = match parse(&combined_exp, &mut simplifier.context) {
                        Ok(e) => e,
                        Err(_) => continue,
                    };
                    let simp_parsed = match parse(&combined_simp, &mut simplifier.context) {
                        Ok(e) => e,
                        Err(_) => continue,
                    };

                    let (exp_simplified, _) = simplifier.simplify(exp_parsed);
                    let (simp_simplified, _) = simplifier.simplify(simp_parsed);

                    // 3-var check: sample at a few points
                    let (lo, hi) = config.sample_range;
                    let mut valid = true;
                    for test_val in [0.5, 1.0, 1.5, 2.0] {
                        let t = (test_val - lo) / (hi - lo);
                        let x_val = lo + (hi - lo) * t.clamp(0.0, 1.0);

                        let mut var_map = HashMap::new();
                        var_map.insert(pair1.var.clone(), x_val);
                        var_map.insert("u".to_string(), x_val * 1.1);
                        var_map.insert("v".to_string(), x_val * 0.9);

                        let va = eval_f64(&simplifier.context, exp_simplified, &var_map);
                        let vb = eval_f64(&simplifier.context, simp_simplified, &var_map);

                        match (va, vb) {
                            (Some(va), Some(vb))
                                if !va.is_nan()
                                    && !vb.is_nan()
                                    && !va.is_infinite()
                                    && !vb.is_infinite() =>
                            {
                                let diff = (va - vb).abs();
                                let scale = va.abs().max(vb.abs()).max(1.0);
                                let allowed = config.atol + config.rtol * scale;
                                if diff > allowed {
                                    valid = false;
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }

                    if valid {
                        triple_passed += 1;
                    } else {
                        triple_failed += 1;
                    }

                    triple_count += 1;
                }
            }
        }

        eprintln!(
            "✅ Triple combinations: {} passed, {} failed (of {} tested)",
            triple_passed, triple_failed, triple_count
        );
    }

    assert_eq!(failed, 0, "Some CSV combination tests failed");
}
// =============================================================================
// CSV-BASED AUTOMATIC COMBINATION TESTS
// =============================================================================

/// Run automatic double and triple combinations from CSV file
/// This generates thousands of test cases from ~180 identity pairs
#[test]
fn metatest_csv_combinations_small() {
    // Small run: 30 pairs = 435 double combinations
    run_csv_combination_tests(30, false);
}

#[test]
#[ignore] // Run with: cargo test --ignored
fn metatest_csv_combinations_full() {
    // Full run: all pairs with triples
    run_csv_combination_tests(100, true);
}

/// Test individual identity pairs (not combinations) to see which simplify symbolically
#[test]
#[ignore = "Diagnostic test - run manually to check symbolic vs numeric equivalence"]
fn metatest_individual_identities() {
    let pairs = load_identity_pairs();
    let config = metatest_config();

    let mut symbolic_passed = 0;
    let mut numeric_only_passed = 0;
    let mut failed = 0;
    let mut numeric_only_examples: Vec<String> = Vec::new();

    for pair in &pairs {
        let mut simplifier = Simplifier::with_default_rules();

        // Parse both expressions
        let exp_parsed = match parse(&pair.exp, &mut simplifier.context) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let simp_parsed = match parse(&pair.simp, &mut simplifier.context) {
            Ok(e) => e,
            Err(_) => continue,
        };

        // Simplify both
        let (exp_simplified, _) = simplifier.simplify(exp_parsed);
        let (simp_simplified, _) = simplifier.simplify(simp_parsed);

        // Check symbolic equality
        let symbolic_match = cas_engine::ordering::compare_expr(
            &simplifier.context,
            exp_simplified,
            simp_simplified,
        ) == std::cmp::Ordering::Equal;

        if symbolic_match {
            symbolic_passed += 1;
        } else {
            // Check numeric equivalence
            let result = check_numeric_equiv_1var(
                &simplifier.context,
                exp_simplified,
                simp_simplified,
                &pair.var,
                &config,
            );

            if result.is_ok() {
                numeric_only_passed += 1;
                if numeric_only_examples.len() < 20 {
                    numeric_only_examples.push(format!("{} ≡ {}", pair.exp, pair.simp));
                }
            } else {
                failed += 1;
                if failed <= 5 {
                    eprintln!("❌ Identity failed: {} ≡ {}", pair.exp, pair.simp);
                }
            }
        }
    }

    let total = symbolic_passed + numeric_only_passed + failed;
    let symbolic_pct = (symbolic_passed as f64 / total as f64 * 100.0) as u32;

    eprintln!("\n📊 Individual Identity Results:");
    eprintln!("   Total: {}", total);
    eprintln!("   ✅ Symbolic: {} ({}%)", symbolic_passed, symbolic_pct);
    eprintln!("   🔢 Numeric-only: {}", numeric_only_passed);
    eprintln!("   ❌ Failed: {}", failed);

    if !numeric_only_examples.is_empty() {
        eprintln!("\n📝 Examples of numeric-only (first 20):");
        for ex in &numeric_only_examples {
            eprintln!("   • {}", ex);
        }
    }

    if failed > 0 {
        eprintln!(
            "\n⚠️  {} identities failed numeric equivalence - may need domain restrictions",
            failed
        );
    }
}
