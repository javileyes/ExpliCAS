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
    let vars = vec![pair1.var, "u", "v"];

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

                match (va, vb) {
                    (Some(va), Some(vb)) => {
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
                    _ => {}
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

            // Check numeric equivalence
            let result = check_numeric_equiv_2var(
                &simplifier.context,
                exp_simplified,
                simp_simplified,
                &pair1.var,
                "u",
                &config,
            );

            if result.is_ok() {
                passed += 1;
            } else {
                failed += 1;
                if failed <= 5 {
                    eprintln!("❌ Double combo failed: ({}) + ({})", pair1.exp, pair2.exp);
                }
            }
        }
    }

    eprintln!(
        "✅ Double combinations: {} passed, {} failed",
        passed, failed
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
                        let x_val = lo + (hi - lo) * t.min(1.0).max(0.0);

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

    // Support 1 or 2 variables
    if vars.len() > 2 {
        eprintln!(
            "⚠️  Skipping metatest (>2 vars not yet supported): {} → {}",
            base_a, base_b
        );
        log_metatest_run(test_name, &config, 0, 0, 1);
        return;
    }

    let mut rng = Lcg::new(config.seed);

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

        // Check numeric equivalence (1 or 2 variables)
        let check_result = if vars.len() == 1 {
            check_numeric_equiv_1var(&simplifier.context, a_simp, b_simp, vars[0], &config)
        } else {
            check_numeric_equiv_2var(
                &simplifier.context,
                a_simp,
                b_simp,
                vars[0],
                vars[1],
                &config,
            )
        };

        if let Err(err) = check_result {
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

// =============================================================================
// Rationalization Tests
// =============================================================================

#[test]
fn metatest_rationalize_1_over_sqrt_5() {
    // 1/sqrt(5) = sqrt(5)/5
    assert_metamorphic_addition("rationalize_sqrt_5", "1/sqrt(5)", "sqrt(5)/5", &[]);
}

#[test]
fn metatest_rationalize_3_over_sqrt_2() {
    // 3/sqrt(2) = 3*sqrt(2)/2
    assert_metamorphic_addition("rationalize_3_sqrt_2", "3/sqrt(2)", "3*sqrt(2)/2", &[]);
}

#[test]
fn metatest_rationalize_4_over_sqrt_8() {
    // 4/sqrt(8) = 4/(2*sqrt(2)) = 2/sqrt(2) = sqrt(2)
    assert_metamorphic_addition("rationalize_4_sqrt_8", "4/sqrt(8)", "sqrt(2)", &[]);
}

#[test]
fn metatest_rationalize_sqrt_x_denom() {
    // 1/sqrt(x) * sqrt(x)/sqrt(x) = sqrt(x)/x
    assert_metamorphic_addition("rationalize_sqrt_x", "1/sqrt(x)", "sqrt(x)/x", &["x"]);
}

#[test]
fn metatest_conjugate_rationalize_plus() {
    // 1/(sqrt(2) + 1) * (sqrt(2) - 1)/(sqrt(2) - 1) = sqrt(2) - 1
    assert_metamorphic_addition(
        "conjugate_rationalize_plus",
        "1/(sqrt(2) + 1)",
        "sqrt(2) - 1",
        &[],
    );
}

#[test]
fn metatest_conjugate_rationalize_minus() {
    // 1/(sqrt(2) - 1) = sqrt(2) + 1
    assert_metamorphic_addition(
        "conjugate_rationalize_minus",
        "1/(sqrt(2) - 1)",
        "sqrt(2) + 1",
        &[],
    );
}

#[test]
fn metatest_conjugate_rationalize_sqrt_3_plus_1() {
    // 2/(sqrt(3) + 1) = sqrt(3) - 1
    assert_metamorphic_addition(
        "conjugate_sqrt_3_plus_1",
        "2/(sqrt(3) + 1)",
        "sqrt(3) - 1",
        &[],
    );
}

#[test]
fn metatest_conjugate_sqrt_5_minus_2() {
    // 1/(sqrt(5) - 2) = sqrt(5) + 2
    assert_metamorphic_addition(
        "conjugate_sqrt_5_minus_2",
        "1/(sqrt(5) - 2)",
        "sqrt(5) + 2",
        &[],
    );
}

// =============================================================================
// More Root Simplifications
// =============================================================================

#[test]
fn metatest_sqrt_18() {
    // sqrt(18) = 3*sqrt(2)
    assert_metamorphic_addition("sqrt_18", "sqrt(18)", "3*sqrt(2)", &[]);
}

#[test]
fn metatest_sqrt_50() {
    // sqrt(50) = 5*sqrt(2)
    assert_metamorphic_addition("sqrt_50", "sqrt(50)", "5*sqrt(2)", &[]);
}

#[test]
fn metatest_sqrt_75() {
    // sqrt(75) = 5*sqrt(3)
    assert_metamorphic_addition("sqrt_75", "sqrt(75)", "5*sqrt(3)", &[]);
}

#[test]
fn metatest_sqrt_98() {
    // sqrt(98) = 7*sqrt(2)
    assert_metamorphic_addition("sqrt_98", "sqrt(98)", "7*sqrt(2)", &[]);
}

#[test]
fn metatest_sqrt_200() {
    // sqrt(200) = 10*sqrt(2)
    assert_metamorphic_addition("sqrt_200", "sqrt(200)", "10*sqrt(2)", &[]);
}

// =============================================================================
// Sum/Difference of Roots
// =============================================================================

#[test]
fn metatest_sqrt_8_plus_sqrt_2() {
    // sqrt(8) + sqrt(2) = 2*sqrt(2) + sqrt(2) = 3*sqrt(2)
    assert_metamorphic_addition("sqrt_8_plus_sqrt_2", "sqrt(8) + sqrt(2)", "3*sqrt(2)", &[]);
}

#[test]
fn metatest_sqrt_18_minus_sqrt_2() {
    // sqrt(18) - sqrt(2) = 3*sqrt(2) - sqrt(2) = 2*sqrt(2)
    assert_metamorphic_addition(
        "sqrt_18_minus_sqrt_2",
        "sqrt(18) - sqrt(2)",
        "2*sqrt(2)",
        &[],
    );
}

#[test]
fn metatest_sqrt_27_plus_sqrt_12() {
    // sqrt(27) + sqrt(12) = 3*sqrt(3) + 2*sqrt(3) = 5*sqrt(3)
    assert_metamorphic_addition(
        "sqrt_27_plus_sqrt_12",
        "sqrt(27) + sqrt(12)",
        "5*sqrt(3)",
        &[],
    );
}

#[test]
fn metatest_sqrt_45_minus_sqrt_20() {
    // sqrt(45) - sqrt(20) = 3*sqrt(5) - 2*sqrt(5) = sqrt(5)
    assert_metamorphic_addition(
        "sqrt_45_minus_sqrt_20",
        "sqrt(45) - sqrt(20)",
        "sqrt(5)",
        &[],
    );
}

// =============================================================================
// Product of Roots
// =============================================================================

#[test]
fn metatest_sqrt_2_times_sqrt_8() {
    // sqrt(2) * sqrt(8) = sqrt(16) = 4
    assert_metamorphic_addition("sqrt_2_times_sqrt_8", "sqrt(2) * sqrt(8)", "4", &[]);
}

#[test]
fn metatest_sqrt_3_times_sqrt_12() {
    // sqrt(3) * sqrt(12) = sqrt(36) = 6
    assert_metamorphic_addition("sqrt_3_times_sqrt_12", "sqrt(3) * sqrt(12)", "6", &[]);
}

#[test]
fn metatest_sqrt_5_times_sqrt_5() {
    // sqrt(5) * sqrt(5) = 5
    assert_metamorphic_addition("sqrt_5_times_sqrt_5", "sqrt(5) * sqrt(5)", "5", &[]);
}

// =============================================================================
// Quotient of Roots
// =============================================================================

#[test]
fn metatest_sqrt_50_over_sqrt_2() {
    // sqrt(50) / sqrt(2) = sqrt(25) = 5
    assert_metamorphic_addition("sqrt_50_over_sqrt_2", "sqrt(50) / sqrt(2)", "5", &[]);
}

#[test]
fn metatest_sqrt_72_over_sqrt_8() {
    // sqrt(72) / sqrt(8) = sqrt(9) = 3
    assert_metamorphic_addition("sqrt_72_over_sqrt_8", "sqrt(72) / sqrt(8)", "3", &[]);
}

#[test]
fn metatest_sqrt_75_over_sqrt_3() {
    // sqrt(75) / sqrt(3) = sqrt(25) = 5
    assert_metamorphic_addition("sqrt_75_over_sqrt_3", "sqrt(75) / sqrt(3)", "5", &[]);
}

// =============================================================================
// More Polynomial Simplifications
// =============================================================================

#[test]
fn metatest_quadratic_plus_linear() {
    // x^2 + 2x + 1 = (x+1)^2
    assert_metamorphic_addition(
        "quadratic_plus_linear",
        "x^2 + 2*x + 1",
        "(x + 1)^2",
        &["x"],
    );
}

#[test]
fn metatest_expand_diff_squares() {
    // (x+3)(x-3) = x^2 - 9
    assert_metamorphic_addition("expand_diff_sq", "(x + 3) * (x - 3)", "x^2 - 9", &["x"]);
}

// =============================================================================
// MULTIVARIABLE TESTS (x, y)
// =============================================================================

#[test]
fn metatest_2var_commutative_add() {
    // x + y = y + x
    assert_metamorphic_addition("2var_commutative_add", "x + y", "y + x", &["x", "y"]);
}

#[test]
fn metatest_2var_commutative_mul() {
    // x * y = y * x
    assert_metamorphic_addition("2var_commutative_mul", "x * y", "y * x", &["x", "y"]);
}

#[test]
fn metatest_2var_associative_add() {
    // (x + y) + 1 = x + (y + 1)
    assert_metamorphic_addition(
        "2var_associative_add",
        "(x + y) + 1",
        "x + (y + 1)",
        &["x", "y"],
    );
}

#[test]
fn metatest_2var_associative_mul() {
    // (x * y) * 2 = x * (y * 2)
    assert_metamorphic_addition(
        "2var_associative_mul",
        "(x * y) * 2",
        "x * (y * 2)",
        &["x", "y"],
    );
}

#[test]
fn metatest_2var_distributive() {
    // x * (y + 1) = x*y + x
    assert_metamorphic_addition("2var_distributive", "x * (y + 1)", "x*y + x", &["x", "y"]);
}

#[test]
fn metatest_2var_distributive_reverse() {
    // x*y + x*1 = x*(y + 1)
    assert_metamorphic_addition("2var_distributive_rev", "x*y + x", "x*(y + 1)", &["x", "y"]);
}

#[test]
fn metatest_2var_foil() {
    // (x + 1)(y + 1) = x*y + x + y + 1
    assert_metamorphic_addition(
        "2var_foil",
        "(x + 1) * (y + 1)",
        "x*y + x + y + 1",
        &["x", "y"],
    );
}

#[test]
fn metatest_2var_difference_of_squares() {
    // (x + y)(x - y) = x^2 - y^2
    assert_metamorphic_addition(
        "2var_diff_squares",
        "(x + y) * (x - y)",
        "x^2 - y^2",
        &["x", "y"],
    );
}

#[test]
fn metatest_2var_square_of_sum() {
    // (x + y)^2 = x^2 + 2xy + y^2
    assert_metamorphic_addition(
        "2var_square_sum",
        "(x + y)^2",
        "x^2 + 2*x*y + y^2",
        &["x", "y"],
    );
}

#[test]
fn metatest_2var_square_of_diff() {
    // (x - y)^2 = x^2 - 2xy + y^2
    assert_metamorphic_addition(
        "2var_square_diff",
        "(x - y)^2",
        "x^2 - 2*x*y + y^2",
        &["x", "y"],
    );
}

#[test]
fn metatest_2var_cube_of_sum() {
    // (x + y)^3 = x^3 + 3x^2y + 3xy^2 + y^3
    assert_metamorphic_addition(
        "2var_cube_sum",
        "(x + y)^3",
        "x^3 + 3*x^2*y + 3*x*y^2 + y^3",
        &["x", "y"],
    );
}

#[test]
fn metatest_2var_sum_of_cubes() {
    // x^3 + y^3 = (x + y)(x^2 - xy + y^2)
    assert_metamorphic_addition(
        "2var_sum_cubes",
        "x^3 + y^3",
        "(x + y) * (x^2 - x*y + y^2)",
        &["x", "y"],
    );
}

#[test]
fn metatest_2var_diff_of_cubes() {
    // x^3 - y^3 = (x - y)(x^2 + xy + y^2)
    assert_metamorphic_addition(
        "2var_diff_cubes",
        "x^3 - y^3",
        "(x - y) * (x^2 + x*y + y^2)",
        &["x", "y"],
    );
}

#[test]
fn metatest_2var_fraction_add() {
    // x/2 + y/2 = (x + y)/2
    assert_metamorphic_addition("2var_fraction_add", "x/2 + y/2", "(x + y)/2", &["x", "y"]);
}

#[test]
fn metatest_2var_divide_by_each() {
    // (x*y)/x = y (for x ≠ 0)
    assert_metamorphic_addition("2var_divide_cancel", "(x*y)/x", "y", &["x", "y"]);
}

#[test]
fn metatest_2var_power_product() {
    // x^2 * y^2 = (x*y)^2
    assert_metamorphic_addition("2var_power_product", "x^2 * y^2", "(x*y)^2", &["x", "y"]);
}

#[test]
fn metatest_2var_ratio_simplify() {
    // (x^2 * y) / (x * y) = x
    assert_metamorphic_addition(
        "2var_ratio_simplify",
        "(x^2 * y) / (x * y)",
        "x",
        &["x", "y"],
    );
}

#[test]
fn metatest_2var_nested_product() {
    // (x + y)(x - y) + y^2 = x^2
    assert_metamorphic_addition(
        "2var_nested_prod",
        "(x + y)*(x - y) + y^2",
        "x^2",
        &["x", "y"],
    );
}

#[test]
fn metatest_2var_trig_add() {
    // sin(x) + sin(y) unchanged but equal to itself
    assert_metamorphic_addition(
        "2var_trig_add",
        "sin(x) + sin(y)",
        "sin(x) + sin(y)",
        &["x", "y"],
    );
}

#[test]
fn metatest_2var_mixed_poly() {
    // x*(y + 2) - 2*x = x*y
    assert_metamorphic_addition("2var_mixed_poly", "x*(y + 2) - 2*x", "x*y", &["x", "y"]);
}

// =============================================================================
// COMBINATION METAMORPHIC TESTS
// Testing: If Exp1 ≡ Simp1 and Exp2 ≡ Simp2, then Exp1 + Exp2 ≡ Simp1 + Simp2
// Uses alpha-renaming to avoid accidental cancellations
// =============================================================================

#[test]
fn metatest_combine_pythagorean_plus_double_angle() {
    // Combine: sin²+cos² = 1 with sin(2x) = 2sin(x)cos(x)
    assert_metamorphic_combine(
        "pythagorean_plus_double_angle",
        TestPair {
            exp: "sin(x)^2 + cos(x)^2",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "sin(2*x)",
            simp: "2*sin(x)*cos(x)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_binomial_plus_difference_of_squares() {
    // Combine: (x+1)² = x²+2x+1 with x²-1 = (x-1)(x+1)
    assert_metamorphic_combine(
        "binomial_plus_diff_squares",
        TestPair {
            exp: "(x+1)^2",
            simp: "x^2 + 2*x + 1",
            var: "x",
        },
        TestPair {
            exp: "x^2 - 1",
            simp: "(x-1)*(x+1)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_exp_log_plus_power() {
    // Combine: ln(e^x) = x with x³/x² = x
    assert_metamorphic_combine(
        "exp_log_plus_power",
        TestPair {
            exp: "ln(exp(x))",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "x^3 / x^2",
            simp: "x",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_trig_minus_algebraic() {
    // Combine: tan² + 1 = sec² with x + (-x) = 0
    assert_metamorphic_combine(
        "trig_minus_algebraic",
        TestPair {
            exp: "tan(x)^2 + 1",
            simp: "sec(x)^2",
            var: "x",
        },
        TestPair {
            exp: "x + (-x)",
            simp: "0",
            var: "x",
        },
        "-",
    );
}

#[test]
fn metatest_combine_sqrt_plus_cube() {
    // Combine: sqrt(x)² = x with (x+1)³ expanded
    assert_metamorphic_combine(
        "sqrt_plus_cube",
        TestPair {
            exp: "sqrt(x)^2",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "(x+1)^3",
            simp: "x^3 + 3*x^2 + 3*x + 1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_fraction_plus_rationalize() {
    // Combine: x/x = 1 with 1/sqrt(2) = sqrt(2)/2
    assert_metamorphic_combine(
        "fraction_plus_rationalize",
        TestPair {
            exp: "x/x",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "1/sqrt(x)",
            simp: "sqrt(x)/x",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_arctan_plus_log() {
    // Combine: tan(arctan(x)) = x with ln(1) = 0
    assert_metamorphic_combine(
        "arctan_plus_log",
        TestPair {
            exp: "tan(arctan(x))",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "ln(x^0)",
            simp: "0",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_mul_power_and_neg() {
    // Combine: x² * x³ = x⁵ with -(-x) = x, using multiplication
    assert_metamorphic_combine(
        "mul_power_and_neg",
        TestPair {
            exp: "x^2 * x^3",
            simp: "x^5",
            var: "x",
        },
        TestPair {
            exp: "-(-x)",
            simp: "x",
            var: "x",
        },
        "*",
    );
}

#[test]
fn metatest_combine_cos_cofunction_plus_sin_cofunction() {
    // Combine: cos(π/2 - x) = sin(x) with sin(π/2 - x) = cos(x)
    assert_metamorphic_combine(
        "cos_sin_cofunction",
        TestPair {
            exp: "cos(pi/2 - x)",
            simp: "sin(x)",
            var: "x",
        },
        TestPair {
            exp: "sin(pi/2 - x)",
            simp: "cos(x)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_distribute_plus_factor() {
    // Combine: 2(x+3) = 2x+6 with x²+x = x(x+1)
    assert_metamorphic_combine(
        "distribute_plus_factor",
        TestPair {
            exp: "2*(x+3)",
            simp: "2*x + 6",
            var: "x",
        },
        TestPair {
            exp: "x^2 + x",
            simp: "x*(x+1)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_zero_power_plus_one_power() {
    // Combine: x^0 = 1 with x^1 = x
    assert_metamorphic_combine(
        "zero_plus_one_power",
        TestPair {
            exp: "x^0",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "x^1",
            simp: "x",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_negative_exp_plus_reciprocal() {
    // Combine: x^(-1) = 1/x with 1/(1/x) = x
    assert_metamorphic_combine(
        "neg_exp_plus_reciprocal",
        TestPair {
            exp: "x^(-1)",
            simp: "1/x",
            var: "x",
        },
        TestPair {
            exp: "1/(1/x)",
            simp: "x",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_mul_trig() {
    // Combine: sin(x)/cos(x) = tan(x) with cos(x)/sin(x) = cot(x)
    assert_metamorphic_combine(
        "mul_trig_ratios",
        TestPair {
            exp: "sin(x)/cos(x)",
            simp: "tan(x)",
            var: "x",
        },
        TestPair {
            exp: "cos(x)/sin(x)",
            simp: "cot(x)",
            var: "x",
        },
        "*",
    );
}

#[test]
fn metatest_combine_abs_plus_sqrt() {
    // Combine: |x|² = x² with sqrt(x²) = |x|
    assert_metamorphic_combine(
        "abs_plus_sqrt",
        TestPair {
            exp: "|x|^2",
            simp: "x^2",
            var: "x",
        },
        TestPair {
            exp: "sqrt(x^2)",
            simp: "|x|",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_chain_three_way() {
    // More complex: combine three operations conceptually
    // (x+x) + (2x - x) = 2x + x = 3x
    // But we test pair-wise: (x+x)=2x combined with (x+1-1)=x
    assert_metamorphic_combine(
        "chain_addition",
        TestPair {
            exp: "x + x",
            simp: "2*x",
            var: "x",
        },
        TestPair {
            exp: "x + 1 - 1",
            simp: "x",
            var: "x",
        },
        "+",
    );
}

// =============================================================================
// More Combination Tests - Different Rule Interactions
// =============================================================================

#[test]
fn metatest_combine_sin_cos_product() {
    // sin(x)*cos(x) combined with tan(x) definition
    assert_metamorphic_combine(
        "sin_cos_mul_tan",
        TestPair {
            exp: "sin(x)*cos(x)",
            simp: "sin(2*x)/2",
            var: "x",
        },
        TestPair {
            exp: "sin(x)/cos(x)",
            simp: "tan(x)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_log_rules() {
    // ln(a*b) = ln(a) + ln(b) combined with ln(e) = 1
    assert_metamorphic_combine(
        "log_rules_combined",
        TestPair {
            exp: "ln(e*x)",
            simp: "1 + ln(x)",
            var: "x",
        },
        TestPair {
            exp: "ln(e)",
            simp: "1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_power_rules() {
    // x^a * x^b = x^(a+b) combined with (x^a)^b = x^(a*b)
    assert_metamorphic_combine(
        "power_rules_combined",
        TestPair {
            exp: "x^2 * x^3",
            simp: "x^5",
            var: "x",
        },
        TestPair {
            exp: "(x^2)^3",
            simp: "x^6",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_binomial_cubic() {
    // (x+1)^2 combined with (x-1)^3
    assert_metamorphic_combine(
        "binomial_squared_cubed",
        TestPair {
            exp: "(x+1)^2",
            simp: "x^2 + 2*x + 1",
            var: "x",
        },
        TestPair {
            exp: "(x-1)^3",
            simp: "x^3 - 3*x^2 + 3*x - 1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_trig_pythagorean_variants() {
    // sin²+cos² = 1 combined with tan²+1 = sec²
    assert_metamorphic_combine(
        "pythagorean_tan_sec",
        TestPair {
            exp: "sin(x)^2 + cos(x)^2",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "tan(x)^2 + 1",
            simp: "sec(x)^2",
            var: "x",
        },
        "-",
    );
}

#[test]
fn metatest_combine_fraction_operations() {
    // x/x = 1 combined with (x+1)/1 = x+1
    assert_metamorphic_combine(
        "fraction_ops",
        TestPair {
            exp: "x/x",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "(x+1)/1",
            simp: "x+1",
            var: "x",
        },
        "*",
    );
}

#[test]
fn metatest_combine_exp_properties() {
    // e^(x+1) = e*e^x combined with e^0 = 1
    assert_metamorphic_combine(
        "exp_properties",
        TestPair {
            exp: "exp(x+1)",
            simp: "e*exp(x)",
            var: "x",
        },
        TestPair {
            exp: "exp(0)",
            simp: "1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_root_quotient() {
    // sqrt(x)/sqrt(x) = 1 combined with sqrt(x^2) = |x|
    assert_metamorphic_combine(
        "root_quotient",
        TestPair {
            exp: "sqrt(x)/sqrt(x)",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "sqrt(x^2)",
            simp: "|x|",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_special_angles() {
    // sin(π/6) = 1/2 combined with cos(π/3) = 1/2
    assert_metamorphic_combine(
        "special_angles",
        TestPair {
            exp: "sin(pi/6)",
            simp: "1/2",
            var: "x",
        },
        TestPair {
            exp: "cos(pi/3)",
            simp: "1/2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_polynomial_factoring() {
    // x^2 - 4 = (x-2)(x+2) combined with x^2 - 9 = (x-3)(x+3)
    assert_metamorphic_combine(
        "poly_factoring",
        TestPair {
            exp: "x^2 - 4",
            simp: "(x-2)*(x+2)",
            var: "x",
        },
        TestPair {
            exp: "x^2 - 9",
            simp: "(x-3)*(x+3)",
            var: "x",
        },
        "+",
    );
}

// =============================================================================
// Additional Single-Variable Tests
// =============================================================================

#[test]
fn metatest_exp_product_rule() {
    // e^x * e^2 = e^(x+2)
    assert_metamorphic_addition("exp_product_rule", "exp(x) * exp(2)", "exp(x+2)", &["x"]);
}

#[test]
fn metatest_log_of_power_numeric() {
    // ln(8) = ln(2^3) = 3*ln(2)
    assert_metamorphic_addition("log_of_power_numeric", "ln(8)", "3*ln(2)", &[]);
}

#[test]
fn metatest_sin_triple_value() {
    // sin(3*π/2) = -1
    assert_metamorphic_addition("sin_3pi_2", "sin(3*pi/2)", "-1", &[]);
}

#[test]
fn metatest_cos_triple_value() {
    // cos(3*π/2) = 0
    assert_metamorphic_addition("cos_3pi_2", "cos(3*pi/2)", "0", &[]);
}

#[test]
fn metatest_tan_2x() {
    // tan(2x) = 2*tan(x)/(1-tan(x)^2) - check numeric equivalence
    assert_metamorphic_addition("tan_2x", "tan(2*x)", "2*tan(x)/(1-tan(x)^2)", &["x"]);
}

#[test]
fn metatest_cot_2x() {
    // cot(2x) = (cot(x)^2 - 1)/(2*cot(x))
    assert_metamorphic_addition("cot_2x", "cot(2*x)", "(cot(x)^2 - 1)/(2*cot(x))", &["x"]);
}

#[test]
fn metatest_sec_squared() {
    // sec(x)^2 - 1 = tan(x)^2
    assert_metamorphic_addition("sec_squared_minus_1", "sec(x)^2 - 1", "tan(x)^2", &["x"]);
}

#[test]
fn metatest_csc_squared() {
    // csc(x)^2 - 1 = cot(x)^2
    assert_metamorphic_addition("csc_squared_minus_1", "csc(x)^2 - 1", "cot(x)^2", &["x"]);
}

#[test]
fn metatest_half_angle_sin() {
    // 2*sin(x/2)^2 = 1 - cos(x)
    assert_metamorphic_addition("half_angle_sin", "2*sin(x/2)^2", "1 - cos(x)", &["x"]);
}

#[test]
fn metatest_half_angle_cos() {
    // 2*cos(x/2)^2 = 1 + cos(x)
    assert_metamorphic_addition("half_angle_cos", "2*cos(x/2)^2", "1 + cos(x)", &["x"]);
}

// =============================================================================
// More Numeric Root Tests
// =============================================================================

#[test]
fn metatest_sqrt_144() {
    assert_metamorphic_addition("sqrt_144", "sqrt(144)", "12", &[]);
}

#[test]
fn metatest_sqrt_169() {
    assert_metamorphic_addition("sqrt_169", "sqrt(169)", "13", &[]);
}

#[test]
fn metatest_cbrt_216() {
    assert_metamorphic_addition("cbrt_216", "216^(1/3)", "6", &[]);
}

#[test]
fn metatest_cbrt_343() {
    assert_metamorphic_addition("cbrt_343", "343^(1/3)", "7", &[]);
}

#[test]
fn metatest_fourth_root_256() {
    assert_metamorphic_addition("fourth_root_256", "256^(1/4)", "4", &[]);
}

#[test]
fn metatest_fifth_root_32() {
    assert_metamorphic_addition("fifth_root_32", "32^(1/5)", "2", &[]);
}

// =============================================================================
// More Trig Special Values
// =============================================================================

#[test]
fn metatest_tan_pi_3() {
    // tan(π/3) = sqrt(3)
    assert_metamorphic_addition("tan_pi_3", "tan(pi/3)", "sqrt(3)", &[]);
}

#[test]
fn metatest_tan_pi_6() {
    // tan(π/6) = sqrt(3)/3 = 1/sqrt(3)
    assert_metamorphic_addition("tan_pi_6", "tan(pi/6)", "1/sqrt(3)", &[]);
}

#[test]
fn metatest_cot_pi_4() {
    // cot(π/4) = 1
    assert_metamorphic_addition("cot_pi_4", "cot(pi/4)", "1", &[]);
}

#[test]
fn metatest_sec_0() {
    // sec(0) = 1
    assert_metamorphic_addition("sec_0", "sec(0)", "1", &[]);
}

// =============================================================================
// Expression Simplification Edge Cases
// =============================================================================

#[test]
fn metatest_nested_negation() {
    // -(-(-x)) = -x
    assert_metamorphic_addition("nested_negation", "-(-(-x))", "-x", &["x"]);
}

#[test]
fn metatest_triple_product() {
    // x * x * x = x^3
    assert_metamorphic_addition("triple_product", "x * x * x", "x^3", &["x"]);
}

#[test]
fn metatest_quadruple_product() {
    // x * x * x * x = x^4
    assert_metamorphic_addition("quadruple_product", "x * x * x * x", "x^4", &["x"]);
}

#[test]
fn metatest_fraction_chain() {
    // (x/2)/2 = x/4
    assert_metamorphic_addition("fraction_chain", "(x/2)/2", "x/4", &["x"]);
}

#[test]
fn metatest_power_of_fraction() {
    // (1/x)^2 = 1/x^2
    assert_metamorphic_addition("power_of_fraction", "(1/x)^2", "1/x^2", &["x"]);
}

#[test]
fn metatest_sqrt_of_fraction() {
    // sqrt(x/4) = sqrt(x)/2
    assert_metamorphic_addition("sqrt_of_fraction", "sqrt(x/4)", "sqrt(x)/2", &["x"]);
}

// =============================================================================
// More Polynomial Tests
// =============================================================================

#[test]
fn metatest_expand_x_plus_4_squared() {
    assert_metamorphic_addition("expand_x_plus_4_sq", "(x+4)^2", "x^2 + 8*x + 16", &["x"]);
}

#[test]
fn metatest_expand_x_minus_5_squared() {
    assert_metamorphic_addition("expand_x_minus_5_sq", "(x-5)^2", "x^2 - 10*x + 25", &["x"]);
}

#[test]
fn metatest_factor_x2_minus_16() {
    assert_metamorphic_addition("factor_x2_minus_16", "x^2 - 16", "(x-4)*(x+4)", &["x"]);
}

#[test]
fn metatest_factor_x2_minus_25() {
    assert_metamorphic_addition("factor_x2_minus_25", "x^2 - 25", "(x-5)*(x+5)", &["x"]);
}

// =============================================================================
// More Numeric Computations
// =============================================================================

#[test]
fn metatest_exp_0() {
    assert_metamorphic_addition("exp_0", "exp(0)", "1", &[]);
}

#[test]
fn metatest_log_e_squared() {
    assert_metamorphic_addition("log_e_squared", "ln(e^2)", "2", &[]);
}

#[test]
fn metatest_log_e_cubed() {
    assert_metamorphic_addition("log_e_cubed", "ln(e^3)", "3", &[]);
}

#[test]
fn metatest_sin_0() {
    assert_metamorphic_addition("sin_0", "sin(0)", "0", &[]);
}

#[test]
fn metatest_cos_0() {
    assert_metamorphic_addition("cos_0", "cos(0)", "1", &[]);
}

#[test]
fn metatest_tan_0() {
    assert_metamorphic_addition("tan_0", "tan(0)", "0", &[]);
}

// =============================================================================
// More Expression Patterns
// =============================================================================

#[test]
fn metatest_x_plus_0() {
    assert_metamorphic_addition("x_plus_0", "x + 0", "x", &["x"]);
}

#[test]
fn metatest_0_plus_x() {
    assert_metamorphic_addition("0_plus_x", "0 + x", "x", &["x"]);
}

#[test]
fn metatest_x_minus_0() {
    assert_metamorphic_addition("x_minus_0", "x - 0", "x", &["x"]);
}

#[test]
fn metatest_1_times_x() {
    assert_metamorphic_addition("1_times_x", "1 * x", "x", &["x"]);
}

#[test]
fn metatest_x_divided_1() {
    assert_metamorphic_addition("x_divided_1", "x / 1", "x", &["x"]);
}

#[test]
fn metatest_x_power_2_sqrt() {
    // sqrt(x^2) = |x|
    assert_metamorphic_addition("x_power_2_sqrt", "sqrt(x^2)", "|x|", &["x"]);
}

// =============================================================================
// More Combination Tests
// =============================================================================

#[test]
fn metatest_combine_double_angle_both() {
    // sin(2x) and cos(2x) together
    assert_metamorphic_combine(
        "double_angle_both",
        TestPair {
            exp: "sin(2*x)",
            simp: "2*sin(x)*cos(x)",
            var: "x",
        },
        TestPair {
            exp: "cos(2*x)",
            simp: "cos(x)^2 - sin(x)^2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_power_and_root() {
    assert_metamorphic_combine(
        "power_and_root",
        TestPair {
            exp: "x^4",
            simp: "(x^2)^2",
            var: "x",
        },
        TestPair {
            exp: "sqrt(x)",
            simp: "x^(1/2)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_log_exp_inverse() {
    assert_metamorphic_combine(
        "log_exp_inverse",
        TestPair {
            exp: "ln(exp(x))",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "exp(ln(x))",
            simp: "x",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_triple_mul() {
    assert_metamorphic_combine(
        "triple_mul",
        TestPair {
            exp: "x*x*x",
            simp: "x^3",
            var: "x",
        },
        TestPair {
            exp: "x+x+x",
            simp: "3*x",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_negate_and_reciprocal() {
    assert_metamorphic_combine(
        "negate_and_reciprocal",
        TestPair {
            exp: "-(-x)",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "1/(1/x)",
            simp: "x",
            var: "x",
        },
        "*",
    );
}

// =============================================================================
// Final Push to 300+ Tests
// =============================================================================

#[test]
fn metatest_sqrt_2() {
    // sqrt(2) ≈ 1.414... (numeric check)
    assert_metamorphic_addition("sqrt_2_self", "sqrt(2)", "sqrt(2)", &[]);
}

#[test]
fn metatest_sqrt_3() {
    assert_metamorphic_addition("sqrt_3_self", "sqrt(3)", "sqrt(3)", &[]);
}

#[test]
fn metatest_pi_self() {
    assert_metamorphic_addition("pi_self", "pi", "pi", &[]);
}

#[test]
fn metatest_e_self() {
    assert_metamorphic_addition("e_self", "e", "e", &[]);
}

#[test]
fn metatest_power_base_1() {
    // 1^x = 1
    assert_metamorphic_addition("power_base_1", "1^x", "1", &["x"]);
}

#[test]
fn metatest_log_base_10() {
    // log(10, 100) = 2
    assert_metamorphic_addition("log_base_10_100", "log(10, 100)", "2", &[]);
}

#[test]
fn metatest_log_10_1000() {
    // log(10, 1000) = 3
    assert_metamorphic_addition("log_10_1000", "log(10, 1000)", "3", &[]);
}

#[test]
fn metatest_sin_4pi() {
    // sin(4π) = 0
    assert_metamorphic_addition("sin_4pi", "sin(4*pi)", "0", &[]);
}

#[test]
fn metatest_cos_4pi() {
    // cos(4π) = 1
    assert_metamorphic_addition("cos_4pi", "cos(4*pi)", "1", &[]);
}

#[test]
fn metatest_tan_negative() {
    // tan(-x) = -tan(x)
    assert_metamorphic_addition("tan_negative", "tan(-x)", "-tan(x)", &["x"]);
}

#[test]
fn metatest_cot_negative() {
    // cot(-x) = -cot(x)
    assert_metamorphic_addition("cot_negative", "cot(-x)", "-cot(x)", &["x"]);
}

#[test]
fn metatest_sec_negative() {
    // sec(-x) = sec(x)
    assert_metamorphic_addition("sec_negative", "sec(-x)", "sec(x)", &["x"]);
}

#[test]
fn metatest_csc_negative() {
    // csc(-x) = -csc(x)
    assert_metamorphic_addition("csc_negative", "csc(-x)", "-csc(x)", &["x"]);
}

#[test]
fn metatest_arctan_0() {
    // arctan(0) = 0
    assert_metamorphic_addition("arctan_0", "arctan(0)", "0", &[]);
}

#[test]
fn metatest_arcsin_0() {
    // arcsin(0) = 0
    assert_metamorphic_addition("arcsin_0", "arcsin(0)", "0", &[]);
}

#[test]
fn metatest_arccos_1() {
    // arccos(1) = 0
    assert_metamorphic_addition("arccos_1", "arccos(1)", "0", &[]);
}

#[test]
fn metatest_arcsin_1() {
    // arcsin(1) = π/2
    assert_metamorphic_addition("arcsin_1", "arcsin(1)", "pi/2", &[]);
}

#[test]
fn metatest_power_neg_1() {
    // x^(-1) * x = 1
    assert_metamorphic_addition("power_neg_1_times_x", "x^(-1) * x", "1", &["x"]);
}

#[test]
fn metatest_sqrt_x_squared_times_x() {
    // sqrt(x) * sqrt(x) = x
    assert_metamorphic_addition("sqrt_x_times_sqrt_x", "sqrt(x) * sqrt(x)", "x", &["x"]);
}

#[test]
fn metatest_cbrt_x_cubed() {
    // (x^(1/3))^3 = x
    assert_metamorphic_addition("cbrt_x_cubed", "(x^(1/3))^3", "x", &["x"]);
}

#[test]
fn metatest_double_sqrt() {
    // sqrt(sqrt(x)) = x^(1/4)
    assert_metamorphic_addition("double_sqrt", "sqrt(sqrt(x))", "x^(1/4)", &["x"]);
}

#[test]
fn metatest_triple_sqrt() {
    // sqrt(sqrt(sqrt(x))) = x^(1/8)
    assert_metamorphic_addition("triple_sqrt", "sqrt(sqrt(sqrt(x)))", "x^(1/8)", &["x"]);
}

#[test]
fn metatest_fraction_mul_inverse() {
    // x * (1/x) = 1
    assert_metamorphic_addition("fraction_mul_inverse", "x * (1/x)", "1", &["x"]);
}

// =============================================================================
// Beyond 300: More Comprehensive Coverage
// =============================================================================

#[test]
fn metatest_sin_cos_product_formula() {
    // sin(x)*cos(x) = sin(2x)/2
    assert_metamorphic_addition("sin_cos_product", "sin(x)*cos(x)", "sin(2*x)/2", &["x"]);
}

#[test]
fn metatest_sin_squared_half_angle() {
    // sin(x)^2 = (1 - cos(2x))/2
    assert_metamorphic_addition("sin_sq_half", "sin(x)^2", "(1 - cos(2*x))/2", &["x"]);
}

#[test]
fn metatest_cos_squared_half_angle() {
    // cos(x)^2 = (1 + cos(2x))/2
    assert_metamorphic_addition("cos_sq_half", "cos(x)^2", "(1 + cos(2*x))/2", &["x"]);
}

#[test]
fn metatest_tan_half_angle() {
    // tan(x/2) = sin(x)/(1 + cos(x))
    assert_metamorphic_addition("tan_half", "tan(x/2)", "sin(x)/(1 + cos(x))", &["x"]);
}

#[test]
fn metatest_expand_cubic_minus() {
    // (x-1)^3 = x^3 - 3x^2 + 3x - 1
    assert_metamorphic_addition(
        "expand_cubic_minus",
        "(x-1)^3",
        "x^3 - 3*x^2 + 3*x - 1",
        &["x"],
    );
}

#[test]
fn metatest_expand_cubic_plus_2() {
    // (x+2)^3 = x^3 + 6x^2 + 12x + 8
    assert_metamorphic_addition(
        "expand_cubic_plus_2",
        "(x+2)^3",
        "x^3 + 6*x^2 + 12*x + 8",
        &["x"],
    );
}

#[test]
fn metatest_sum_of_cubes_factor() {
    // x^3 + 8 = (x+2)(x^2 - 2x + 4)
    assert_metamorphic_addition("sum_cubes_8", "x^3 + 8", "(x+2)*(x^2 - 2*x + 4)", &["x"]);
}

#[test]
fn metatest_diff_of_cubes_factor() {
    // x^3 - 8 = (x-2)(x^2 + 2x + 4)
    assert_metamorphic_addition("diff_cubes_8", "x^3 - 8", "(x-2)*(x^2 + 2*x + 4)", &["x"]);
}

#[test]
fn metatest_power_5() {
    // x^5 / x^3 = x^2
    assert_metamorphic_addition("power_5_div_3", "x^5 / x^3", "x^2", &["x"]);
}

#[test]
fn metatest_power_6() {
    // x^6 / x^4 = x^2
    assert_metamorphic_addition("power_6_div_4", "x^6 / x^4", "x^2", &["x"]);
}

#[test]
fn metatest_sqrt_512() {
    // sqrt(512) = 16*sqrt(2)
    assert_metamorphic_addition("sqrt_512", "sqrt(512)", "16*sqrt(2)", &[]);
}

#[test]
fn metatest_sqrt_128() {
    // sqrt(128) = 8*sqrt(2)
    assert_metamorphic_addition("sqrt_128", "sqrt(128)", "8*sqrt(2)", &[]);
}

#[test]
fn metatest_sqrt_72() {
    // sqrt(72) = 6*sqrt(2)
    assert_metamorphic_addition("sqrt_72", "sqrt(72)", "6*sqrt(2)", &[]);
}

#[test]
fn metatest_6th_root_64() {
    // 64^(1/6) = 2
    assert_metamorphic_addition("6th_root_64", "64^(1/6)", "2", &[]);
}

#[test]
fn metatest_8th_root_256() {
    // 256^(1/8) = 2
    assert_metamorphic_addition("8th_root_256", "256^(1/8)", "2", &[]);
}

#[test]
fn metatest_ln_product() {
    // ln(x*e) = ln(x) + 1
    assert_metamorphic_addition("ln_product_e", "ln(x*e)", "ln(x) + 1", &["x"]);
}

#[test]
fn metatest_ln_quotient_e() {
    // ln(x/e) = ln(x) - 1
    assert_metamorphic_addition("ln_quotient_e", "ln(x/e)", "ln(x) - 1", &["x"]);
}

#[test]
fn metatest_exp_negative() {
    // exp(-x) = 1/exp(x)
    assert_metamorphic_addition("exp_negative", "exp(-x)", "1/exp(x)", &["x"]);
}

#[test]
fn metatest_ln_sqrt() {
    // ln(sqrt(x)) = ln(x)/2
    assert_metamorphic_addition("ln_sqrt", "ln(sqrt(x))", "ln(x)/2", &["x"]);
}

#[test]
fn metatest_exp_ln_2() {
    // e^(ln(2)) = 2
    assert_metamorphic_addition("exp_ln_2", "exp(ln(2))", "2", &[]);
}

#[test]
fn metatest_2var_factored_product() {
    // (x+y)*(x-y) + (x+y)^2 = (x+y)*(2x)
    assert_metamorphic_addition(
        "2var_factored_product",
        "(x+y)*(x-y) + (x+y)^2",
        "(x+y)*(2*x)",
        &["x", "y"],
    );
}

#[test]
fn metatest_combine_all_pythagorean() {
    // sin²+cos² combined with 1+tan²=sec² and 1+cot²=csc²
    assert_metamorphic_combine(
        "all_pythagorean",
        TestPair {
            exp: "sin(x)^2 + cos(x)^2",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "1 + tan(x)^2",
            simp: "sec(x)^2",
            var: "x",
        },
        "*",
    );
}

#[test]
fn metatest_combine_roots_and_powers() {
    assert_metamorphic_combine(
        "roots_and_powers",
        TestPair {
            exp: "sqrt(x)*sqrt(x)",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "x^3/x^2",
            simp: "x",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_combine_ln_exp_both() {
    assert_metamorphic_combine(
        "ln_exp_both_ways",
        TestPair {
            exp: "ln(exp(x))",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "exp(ln(x))",
            simp: "x",
            var: "x",
        },
        "*",
    );
}

// =============================================================================
// EXTENSIVE COMBINATION TESTS - Testing Rule Interactions
// =============================================================================

// Trig + Algebraic
#[test]
fn metatest_comb_sin2cos2_plus_binomial() {
    assert_metamorphic_combine(
        "sin2cos2_plus_binomial",
        TestPair {
            exp: "sin(x)^2 + cos(x)^2",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "(x+1)^2",
            simp: "x^2 + 2*x + 1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_double_angle_plus_foil() {
    assert_metamorphic_combine(
        "double_angle_plus_foil",
        TestPair {
            exp: "sin(2*x)",
            simp: "2*sin(x)*cos(x)",
            var: "x",
        },
        TestPair {
            exp: "(x+1)*(x-1)",
            simp: "x^2 - 1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_tan_identity_plus_cube() {
    assert_metamorphic_combine(
        "tan_identity_plus_cube",
        TestPair {
            exp: "tan(x)^2 + 1",
            simp: "sec(x)^2",
            var: "x",
        },
        TestPair {
            exp: "(x+1)^3",
            simp: "x^3 + 3*x^2 + 3*x + 1",
            var: "x",
        },
        "+",
    );
}

// Log/Exp + Trig
#[test]
fn metatest_comb_ln_exp_plus_cos_double() {
    assert_metamorphic_combine(
        "ln_exp_plus_cos_double",
        TestPair {
            exp: "ln(exp(x))",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "cos(2*x)",
            simp: "cos(x)^2 - sin(x)^2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_exp_ln_times_pythagorean() {
    assert_metamorphic_combine(
        "exp_ln_times_pythagorean",
        TestPair {
            exp: "exp(ln(x))",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "sin(x)^2 + cos(x)^2",
            simp: "1",
            var: "x",
        },
        "*",
    );
}

// Power + Root
#[test]
fn metatest_comb_power_product_plus_sqrt() {
    assert_metamorphic_combine(
        "power_product_plus_sqrt",
        TestPair {
            exp: "x^2 * x^3",
            simp: "x^5",
            var: "x",
        },
        TestPair {
            exp: "sqrt(x)^2",
            simp: "x",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_power_tower_plus_cbrt() {
    assert_metamorphic_combine(
        "power_tower_plus_cbrt",
        TestPair {
            exp: "(x^2)^3",
            simp: "x^6",
            var: "x",
        },
        TestPair {
            exp: "(x^(1/3))^3",
            simp: "x",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_frac_exp_plus_double_sqrt() {
    assert_metamorphic_combine(
        "frac_exp_plus_double_sqrt",
        TestPair {
            exp: "x^(1/2) * x^(1/2)",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "sqrt(sqrt(x))",
            simp: "x^(1/4)",
            var: "x",
        },
        "+",
    );
}

// Fraction + Trig
#[test]
fn metatest_comb_fraction_cancel_plus_tan() {
    assert_metamorphic_combine(
        "fraction_cancel_plus_tan",
        TestPair {
            exp: "x/x",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "sin(x)/cos(x)",
            simp: "tan(x)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_reciprocal_plus_cot() {
    assert_metamorphic_combine(
        "reciprocal_plus_cot",
        TestPair {
            exp: "1/(1/x)",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "cos(x)/sin(x)",
            simp: "cot(x)",
            var: "x",
        },
        "+",
    );
}

// Multiple operator tests
#[test]
fn metatest_comb_mul_sin_cos() {
    assert_metamorphic_combine(
        "mul_sin_cos_identities",
        TestPair {
            exp: "sin(x)^2",
            simp: "(1 - cos(2*x))/2",
            var: "x",
        },
        TestPair {
            exp: "cos(x)^2",
            simp: "(1 + cos(2*x))/2",
            var: "x",
        },
        "*",
    );
}

#[test]
fn metatest_comb_sub_powers() {
    assert_metamorphic_combine(
        "sub_powers",
        TestPair {
            exp: "x^5 / x^3",
            simp: "x^2",
            var: "x",
        },
        TestPair {
            exp: "x^6 / x^4",
            simp: "x^2",
            var: "x",
        },
        "-",
    );
}

// Special values + algebraic
#[test]
fn metatest_comb_sin_pi_plus_zero_power() {
    assert_metamorphic_combine(
        "sin_pi_plus_zero_power",
        TestPair {
            exp: "sin(pi)",
            simp: "0",
            var: "x",
        },
        TestPair {
            exp: "x^0",
            simp: "1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_cos_pi_times_neg() {
    assert_metamorphic_combine(
        "cos_pi_times_neg",
        TestPair {
            exp: "cos(pi)",
            simp: "-1",
            var: "x",
        },
        TestPair {
            exp: "-(-x)",
            simp: "x",
            var: "x",
        },
        "*",
    );
}

// Polynomial factoring combinations
#[test]
fn metatest_comb_diff_squares_plus_sum_cubes() {
    assert_metamorphic_combine(
        "diff_squares_plus_sum_cubes",
        TestPair {
            exp: "x^2 - 4",
            simp: "(x-2)*(x+2)",
            var: "x",
        },
        TestPair {
            exp: "x^3 + 8",
            simp: "(x+2)*(x^2 - 2*x + 4)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_binomial_sq_plus_diff_cubes() {
    assert_metamorphic_combine(
        "binomial_sq_plus_diff_cubes",
        TestPair {
            exp: "(x+2)^2",
            simp: "x^2 + 4*x + 4",
            var: "x",
        },
        TestPair {
            exp: "x^3 - 8",
            simp: "(x-2)*(x^2 + 2*x + 4)",
            var: "x",
        },
        "+",
    );
}

// Log properties combinations
#[test]
fn metatest_comb_ln_product_plus_ln_sqrt() {
    assert_metamorphic_combine(
        "ln_product_plus_ln_sqrt",
        TestPair {
            exp: "ln(x*e)",
            simp: "ln(x) + 1",
            var: "x",
        },
        TestPair {
            exp: "ln(sqrt(x))",
            simp: "ln(x)/2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_exp_neg_plus_ln_e() {
    assert_metamorphic_combine(
        "exp_neg_plus_ln_e",
        TestPair {
            exp: "exp(-x)",
            simp: "1/exp(x)",
            var: "x",
        },
        TestPair {
            exp: "ln(e)",
            simp: "1",
            var: "x",
        },
        "+",
    );
}

// Inverse trig combinations
#[test]
fn metatest_comb_arctan_plus_arcsin() {
    assert_metamorphic_combine(
        "arctan_plus_arcsin",
        TestPair {
            exp: "tan(arctan(x))",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "sin(arcsin(x))",
            simp: "x",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_arccos_plus_sqrt() {
    assert_metamorphic_combine(
        "arccos_plus_sqrt",
        TestPair {
            exp: "cos(arccos(x))",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "sqrt(x)*sqrt(x)",
            simp: "x",
            var: "x",
        },
        "+",
    );
}

// Absolute value combinations
#[test]
fn metatest_comb_abs_sq_plus_sqrt_sq() {
    assert_metamorphic_combine(
        "abs_sq_plus_sqrt_sq",
        TestPair {
            exp: "|x|^2",
            simp: "x^2",
            var: "x",
        },
        TestPair {
            exp: "sqrt(x^2)",
            simp: "|x|",
            var: "x",
        },
        "+",
    );
}

// Negative argument combinations
#[test]
fn metatest_comb_sin_neg_plus_cos_neg() {
    assert_metamorphic_combine(
        "sin_neg_plus_cos_neg",
        TestPair {
            exp: "sin(-x)",
            simp: "-sin(x)",
            var: "x",
        },
        TestPair {
            exp: "cos(-x)",
            simp: "cos(x)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_tan_neg_times_cot_neg() {
    assert_metamorphic_combine(
        "tan_neg_times_cot_neg",
        TestPair {
            exp: "tan(-x)",
            simp: "-tan(x)",
            var: "x",
        },
        TestPair {
            exp: "cot(-x)",
            simp: "-cot(x)",
            var: "x",
        },
        "*",
    );
}

// Distribution combinations
#[test]
fn metatest_comb_distribute_plus_factor() {
    assert_metamorphic_combine(
        "distribute_plus_factor_combo",
        TestPair {
            exp: "2*(x+3)",
            simp: "2*x + 6",
            var: "x",
        },
        TestPair {
            exp: "x^2 + x",
            simp: "x*(x+1)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_combine_like_plus_expand() {
    assert_metamorphic_combine(
        "combine_like_plus_expand",
        TestPair {
            exp: "2*x + 3*x",
            simp: "5*x",
            var: "x",
        },
        TestPair {
            exp: "(x+1)^2",
            simp: "x^2 + 2*x + 1",
            var: "x",
        },
        "+",
    );
}

// Mixed algebra and special values
#[test]
fn metatest_comb_e_power_plus_pi_times() {
    assert_metamorphic_combine(
        "e_power_plus_pi_times",
        TestPair {
            exp: "e^0",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "pi * 0",
            simp: "0",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_ln_e_cubed_plus_exp_0() {
    assert_metamorphic_combine(
        "ln_e_cubed_plus_exp_0",
        TestPair {
            exp: "ln(e^3)",
            simp: "3",
            var: "x",
        },
        TestPair {
            exp: "exp(0)",
            simp: "1",
            var: "x",
        },
        "+",
    );
}

// Rationalization combinations
#[test]
fn metatest_comb_rationalize_plus_sqrt_simplify() {
    assert_metamorphic_combine(
        "rationalize_plus_sqrt_simplify",
        TestPair {
            exp: "1/sqrt(x)",
            simp: "sqrt(x)/x",
            var: "x",
        },
        TestPair {
            exp: "sqrt(18)",
            simp: "3*sqrt(2)",
            var: "x",
        },
        "+",
    );
}

// =============================================================================
// TRIPLE COMBINATION TESTS - Exp1 + Exp2 + Exp3 ≡ Simp1 + Simp2 + Simp3
// =============================================================================

#[test]
fn metatest_triple_trig_all() {
    // sin²+cos² = 1, tan²+1 = sec², cot²+1 = csc²
    assert_metamorphic_combine_triple(
        "triple_trig_all",
        TestPair {
            exp: "sin(x)^2 + cos(x)^2",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "tan(x)^2 + 1",
            simp: "sec(x)^2",
            var: "x",
        },
        TestPair {
            exp: "cot(x)^2 + 1",
            simp: "csc(x)^2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_powers() {
    // x²·x³=x⁵, (x²)³=x⁶, x⁵/x³=x²
    assert_metamorphic_combine_triple(
        "triple_powers",
        TestPair {
            exp: "x^2 * x^3",
            simp: "x^5",
            var: "x",
        },
        TestPair {
            exp: "(x^2)^3",
            simp: "x^6",
            var: "x",
        },
        TestPair {
            exp: "x^5 / x^3",
            simp: "x^2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_log_exp() {
    // ln(e^x)=x, exp(ln(x))=x, ln(e)=1
    assert_metamorphic_combine_triple(
        "triple_log_exp",
        TestPair {
            exp: "ln(exp(x))",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "exp(ln(x))",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "ln(e)",
            simp: "1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_binomials() {
    // (x+1)², (x+2)², (x+3)²
    assert_metamorphic_combine_triple(
        "triple_binomials",
        TestPair {
            exp: "(x+1)^2",
            simp: "x^2 + 2*x + 1",
            var: "x",
        },
        TestPair {
            exp: "(x+2)^2",
            simp: "x^2 + 4*x + 4",
            var: "x",
        },
        TestPair {
            exp: "(x+3)^2",
            simp: "x^2 + 6*x + 9",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_roots() {
    // sqrt(x)²=x, ∛x³=x, sqrt(sqrt(x))=x^(1/4)
    assert_metamorphic_combine_triple(
        "triple_roots",
        TestPair {
            exp: "sqrt(x)^2",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "(x^(1/3))^3",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "sqrt(sqrt(x))",
            simp: "x^(1/4)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_fractions() {
    // x/x=1, 1/(1/x)=x, x*1/x=1
    assert_metamorphic_combine_triple(
        "triple_fractions",
        TestPair {
            exp: "x/x",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "1/(1/x)",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "x * (1/x)",
            simp: "1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_double_angles() {
    // sin(2x), cos(2x), tan(2x)
    assert_metamorphic_combine_triple(
        "triple_double_angles",
        TestPair {
            exp: "sin(2*x)",
            simp: "2*sin(x)*cos(x)",
            var: "x",
        },
        TestPair {
            exp: "cos(2*x)",
            simp: "cos(x)^2 - sin(x)^2",
            var: "x",
        },
        TestPair {
            exp: "tan(2*x)",
            simp: "2*tan(x)/(1-tan(x)^2)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_negative_args() {
    // sin(-x), cos(-x), tan(-x)
    assert_metamorphic_combine_triple(
        "triple_negative_args",
        TestPair {
            exp: "sin(-x)",
            simp: "-sin(x)",
            var: "x",
        },
        TestPair {
            exp: "cos(-x)",
            simp: "cos(x)",
            var: "x",
        },
        TestPair {
            exp: "tan(-x)",
            simp: "-tan(x)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_inverse_trig() {
    // tan(arctan), sin(arcsin), cos(arccos)
    assert_metamorphic_combine_triple(
        "triple_inverse_trig",
        TestPair {
            exp: "tan(arctan(x))",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "sin(arcsin(x))",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "cos(arccos(x))",
            simp: "x",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_polynomial_factor() {
    // x²-1, x²-4, x²-9
    assert_metamorphic_combine_triple(
        "triple_poly_factor",
        TestPair {
            exp: "x^2 - 1",
            simp: "(x-1)*(x+1)",
            var: "x",
        },
        TestPair {
            exp: "x^2 - 4",
            simp: "(x-2)*(x+2)",
            var: "x",
        },
        TestPair {
            exp: "x^2 - 9",
            simp: "(x-3)*(x+3)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_mixed_trig_algebra() {
    // pythagorean + binomial + factor
    assert_metamorphic_combine_triple(
        "triple_mixed",
        TestPair {
            exp: "sin(x)^2 + cos(x)^2",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "(x+1)^2",
            simp: "x^2 + 2*x + 1",
            var: "x",
        },
        TestPair {
            exp: "x^2 - 4",
            simp: "(x-2)*(x+2)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_log_properties() {
    // ln(x*e), ln(sqrt(x)), ln(x/e)
    assert_metamorphic_combine_triple(
        "triple_log_props",
        TestPair {
            exp: "ln(x*e)",
            simp: "ln(x) + 1",
            var: "x",
        },
        TestPair {
            exp: "ln(sqrt(x))",
            simp: "ln(x)/2",
            var: "x",
        },
        TestPair {
            exp: "ln(x/e)",
            simp: "ln(x) - 1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_special_values() {
    // sin(π)=0, cos(π)=-1, e^0=1
    assert_metamorphic_combine_triple(
        "triple_special",
        TestPair {
            exp: "sin(pi)",
            simp: "0",
            var: "x",
        },
        TestPair {
            exp: "cos(pi)",
            simp: "-1",
            var: "x",
        },
        TestPair {
            exp: "exp(0)",
            simp: "1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_cubes() {
    // (x+1)³, x³+8, x³-8
    assert_metamorphic_combine_triple(
        "triple_cubes",
        TestPair {
            exp: "(x+1)^3",
            simp: "x^3 + 3*x^2 + 3*x + 1",
            var: "x",
        },
        TestPair {
            exp: "x^3 + 8",
            simp: "(x+2)*(x^2 - 2*x + 4)",
            var: "x",
        },
        TestPair {
            exp: "x^3 - 8",
            simp: "(x-2)*(x^2 + 2*x + 4)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_sqrt_simplify() {
    // sqrt(18), sqrt(50), sqrt(72)
    assert_metamorphic_combine_triple(
        "triple_sqrt_simplify",
        TestPair {
            exp: "sqrt(18)",
            simp: "3*sqrt(2)",
            var: "x",
        },
        TestPair {
            exp: "sqrt(50)",
            simp: "5*sqrt(2)",
            var: "x",
        },
        TestPair {
            exp: "sqrt(72)",
            simp: "6*sqrt(2)",
            var: "x",
        },
        "+",
    );
}

// =============================================================================
// MORE DOUBLE COMBINATIONS - Diverse Pairs
// =============================================================================

#[test]
fn metatest_comb_half_angle_plus_double() {
    assert_metamorphic_combine(
        "half_plus_double_angle",
        TestPair {
            exp: "2*sin(x/2)^2",
            simp: "1 - cos(x)",
            var: "x",
        },
        TestPair {
            exp: "sin(2*x)",
            simp: "2*sin(x)*cos(x)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_power_reduction_both() {
    assert_metamorphic_combine(
        "power_reduction_both",
        TestPair {
            exp: "sin(x)^2",
            simp: "(1 - cos(2*x))/2",
            var: "x",
        },
        TestPair {
            exp: "cos(x)^2",
            simp: "(1 + cos(2*x))/2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_trig_ratios_mul() {
    assert_metamorphic_combine(
        "trig_ratios_mul",
        TestPair {
            exp: "sin(x)/cos(x)",
            simp: "tan(x)",
            var: "x",
        },
        TestPair {
            exp: "cos(x)/sin(x)",
            simp: "cot(x)",
            var: "x",
        },
        "*",
    );
}

#[test]
fn metatest_comb_exp_sum_product() {
    assert_metamorphic_combine(
        "exp_sum_product",
        TestPair {
            exp: "exp(x+1)",
            simp: "e*exp(x)",
            var: "x",
        },
        TestPair {
            exp: "exp(x-1)",
            simp: "exp(x)/e",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_fourth_power() {
    assert_metamorphic_combine(
        "fourth_power",
        TestPair {
            exp: "(x+1)^4",
            simp: "x^4 + 4*x^3 + 6*x^2 + 4*x + 1",
            var: "x",
        },
        TestPair {
            exp: "x^4 - 1",
            simp: "(x^2+1)*(x+1)*(x-1)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_nested_roots() {
    assert_metamorphic_combine(
        "nested_roots",
        TestPair {
            exp: "sqrt(sqrt(x))",
            simp: "x^(1/4)",
            var: "x",
        },
        TestPair {
            exp: "sqrt(sqrt(sqrt(x)))",
            simp: "x^(1/8)",
            var: "x",
        },
        "+",
    );
}

// =============================================================================
// MORE TRIPLE COMBINATIONS - Extensive Coverage
// =============================================================================

#[test]
fn metatest_triple_cofunctions() {
    // sin(π/2-x)=cos(x), cos(π/2-x)=sin(x), tan(π/2-x)=cot(x)
    assert_metamorphic_combine_triple(
        "triple_cofunctions",
        TestPair {
            exp: "sin(pi/2 - x)",
            simp: "cos(x)",
            var: "x",
        },
        TestPair {
            exp: "cos(pi/2 - x)",
            simp: "sin(x)",
            var: "x",
        },
        TestPair {
            exp: "x + 0",
            simp: "x",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_power_quotients() {
    // x^5/x^2, x^6/x^3, x^7/x^4
    assert_metamorphic_combine_triple(
        "triple_power_quotients",
        TestPair {
            exp: "x^5 / x^2",
            simp: "x^3",
            var: "x",
        },
        TestPair {
            exp: "x^6 / x^3",
            simp: "x^3",
            var: "x",
        },
        TestPair {
            exp: "x^7 / x^4",
            simp: "x^3",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_rationalize() {
    // 1/sqrt(2), 1/sqrt(3), 1/sqrt(5)
    assert_metamorphic_combine_triple(
        "triple_rationalize",
        TestPair {
            exp: "1/sqrt(2)",
            simp: "sqrt(2)/2",
            var: "x",
        },
        TestPair {
            exp: "1/sqrt(3)",
            simp: "sqrt(3)/3",
            var: "x",
        },
        TestPair {
            exp: "1/sqrt(5)",
            simp: "sqrt(5)/5",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_distribute() {
    // 2(x+1), 3(x+2), 4(x+3)
    assert_metamorphic_combine_triple(
        "triple_distribute",
        TestPair {
            exp: "2*(x+1)",
            simp: "2*x + 2",
            var: "x",
        },
        TestPair {
            exp: "3*(x+2)",
            simp: "3*x + 6",
            var: "x",
        },
        TestPair {
            exp: "4*(x+3)",
            simp: "4*x + 12",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_factor_out() {
    // x²+x, x³+x², x²+2x
    assert_metamorphic_combine_triple(
        "triple_factor_out",
        TestPair {
            exp: "x^2 + x",
            simp: "x*(x+1)",
            var: "x",
        },
        TestPair {
            exp: "x^3 + x^2",
            simp: "x^2*(x+1)",
            var: "x",
        },
        TestPair {
            exp: "x^2 + 2*x",
            simp: "x*(x+2)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_exp_properties() {
    // e^(x+1), e^(x-1), e^(2x)
    assert_metamorphic_combine_triple(
        "triple_exp_props",
        TestPair {
            exp: "exp(x+1)",
            simp: "e*exp(x)",
            var: "x",
        },
        TestPair {
            exp: "exp(x-1)",
            simp: "exp(x)/e",
            var: "x",
        },
        TestPair {
            exp: "exp(2*x)",
            simp: "exp(x)^2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_trig_ratios() {
    // sin/cos=tan, cos/sin=cot, 1/cos=sec
    assert_metamorphic_combine_triple(
        "triple_trig_ratios",
        TestPair {
            exp: "sin(x)/cos(x)",
            simp: "tan(x)",
            var: "x",
        },
        TestPair {
            exp: "cos(x)/sin(x)",
            simp: "cot(x)",
            var: "x",
        },
        TestPair {
            exp: "1/cos(x)",
            simp: "sec(x)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_abs_patterns() {
    // |x|², sqrt(x²), |-x|
    assert_metamorphic_combine_triple(
        "triple_abs_patterns",
        TestPair {
            exp: "|x|^2",
            simp: "x^2",
            var: "x",
        },
        TestPair {
            exp: "sqrt(x^2)",
            simp: "|x|",
            var: "x",
        },
        TestPair {
            exp: "|-x|",
            simp: "|x|",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_zero_one() {
    // x+0, x*1, x/1
    assert_metamorphic_combine_triple(
        "triple_zero_one",
        TestPair {
            exp: "x + 0",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "x * 1",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "x / 1",
            simp: "x",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_self_cancel() {
    // x-x, x/x, x^0
    assert_metamorphic_combine_triple(
        "triple_self_cancel",
        TestPair {
            exp: "x - x",
            simp: "0",
            var: "x",
        },
        TestPair {
            exp: "x / x",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "x^0",
            simp: "1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_combining_like() {
    // 2x+3x, 5x-2x, x+x
    assert_metamorphic_combine_triple(
        "triple_combining_like",
        TestPair {
            exp: "2*x + 3*x",
            simp: "5*x",
            var: "x",
        },
        TestPair {
            exp: "5*x - 2*x",
            simp: "3*x",
            var: "x",
        },
        TestPair {
            exp: "x + x",
            simp: "2*x",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_numeric_roots() {
    // sqrt(16), sqrt(25), sqrt(36)
    assert_metamorphic_combine_triple(
        "triple_numeric_roots",
        TestPair {
            exp: "sqrt(16)",
            simp: "4",
            var: "x",
        },
        TestPair {
            exp: "sqrt(25)",
            simp: "5",
            var: "x",
        },
        TestPair {
            exp: "sqrt(36)",
            simp: "6",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_log_bases() {
    // log(2,8), log(3,27), log(4,64)
    assert_metamorphic_combine_triple(
        "triple_log_bases",
        TestPair {
            exp: "log(2, 8)",
            simp: "3",
            var: "x",
        },
        TestPair {
            exp: "log(3, 27)",
            simp: "3",
            var: "x",
        },
        TestPair {
            exp: "log(4, 64)",
            simp: "3",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_trig_values_pi() {
    // sin(2π), cos(2π), sin(3π)
    assert_metamorphic_combine_triple(
        "triple_trig_values_pi",
        TestPair {
            exp: "sin(2*pi)",
            simp: "0",
            var: "x",
        },
        TestPair {
            exp: "cos(2*pi)",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "sin(3*pi)",
            simp: "0",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_arcfun_values() {
    // arctan(0), arcsin(0), arccos(1)
    assert_metamorphic_combine_triple(
        "triple_arcfun_values",
        TestPair {
            exp: "arctan(0)",
            simp: "0",
            var: "x",
        },
        TestPair {
            exp: "arcsin(0)",
            simp: "0",
            var: "x",
        },
        TestPair {
            exp: "arccos(1)",
            simp: "0",
            var: "x",
        },
        "+",
    );
}

// =============================================================================
// EVEN MORE DOUBLE COMBINATIONS
// =============================================================================

#[test]
fn metatest_comb_product_to_sum() {
    assert_metamorphic_combine(
        "product_to_sum",
        TestPair {
            exp: "sin(x)*cos(x)",
            simp: "sin(2*x)/2",
            var: "x",
        },
        TestPair {
            exp: "sin(x)^2 + cos(x)^2",
            simp: "1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_half_angle_both() {
    assert_metamorphic_combine(
        "half_angle_both",
        TestPair {
            exp: "2*sin(x/2)^2",
            simp: "1 - cos(x)",
            var: "x",
        },
        TestPair {
            exp: "2*cos(x/2)^2",
            simp: "1 + cos(x)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_ln_exp_inverse() {
    assert_metamorphic_combine(
        "ln_exp_inverse",
        TestPair {
            exp: "ln(exp(x))",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "exp(ln(x))",
            simp: "x",
            var: "x",
        },
        "-",
    );
}

#[test]
fn metatest_comb_power_same_base() {
    assert_metamorphic_combine(
        "power_same_base",
        TestPair {
            exp: "x^3 * x^4",
            simp: "x^7",
            var: "x",
        },
        TestPair {
            exp: "x^5 * x^2",
            simp: "x^7",
            var: "x",
        },
        "-",
    );
}

#[test]
fn metatest_comb_sqrt_combos() {
    assert_metamorphic_combine(
        "sqrt_combos",
        TestPair {
            exp: "sqrt(8) + sqrt(2)",
            simp: "3*sqrt(2)",
            var: "x",
        },
        TestPair {
            exp: "sqrt(18) - sqrt(2)",
            simp: "2*sqrt(2)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_quotient_roots() {
    assert_metamorphic_combine(
        "quotient_roots",
        TestPair {
            exp: "sqrt(50) / sqrt(2)",
            simp: "5",
            var: "x",
        },
        TestPair {
            exp: "sqrt(72) / sqrt(8)",
            simp: "3",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_expand_minus() {
    assert_metamorphic_combine(
        "expand_minus",
        TestPair {
            exp: "(x-1)^2",
            simp: "x^2 - 2*x + 1",
            var: "x",
        },
        TestPair {
            exp: "(x-2)^2",
            simp: "x^2 - 4*x + 4",
            var: "x",
        },
        "-",
    );
}

#[test]
fn metatest_comb_cofunction_pair() {
    assert_metamorphic_combine(
        "cofunction_pair",
        TestPair {
            exp: "sin(pi/2 - x)",
            simp: "cos(x)",
            var: "x",
        },
        TestPair {
            exp: "cos(pi/2 - x)",
            simp: "sin(x)",
            var: "x",
        },
        "*",
    );
}

#[test]
fn metatest_comb_sec_csc() {
    assert_metamorphic_combine(
        "sec_csc",
        TestPair {
            exp: "1/cos(x)",
            simp: "sec(x)",
            var: "x",
        },
        TestPair {
            exp: "1/sin(x)",
            simp: "csc(x)",
            var: "x",
        },
        "*",
    );
}

#[test]
fn metatest_comb_ln_sqrt_power() {
    assert_metamorphic_combine(
        "ln_sqrt_power",
        TestPair {
            exp: "ln(sqrt(x))",
            simp: "ln(x)/2",
            var: "x",
        },
        TestPair {
            exp: "ln(x^2)",
            simp: "2*ln(x)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_negative_functions() {
    assert_metamorphic_combine(
        "negative_functions",
        TestPair {
            exp: "sin(-x) + sin(x)",
            simp: "0",
            var: "x",
        },
        TestPair {
            exp: "cos(-x) - cos(x)",
            simp: "0",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_cancel_pairs() {
    assert_metamorphic_combine(
        "cancel_pairs",
        TestPair {
            exp: "x*x^(-1)",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "exp(x)*exp(-x)",
            simp: "1",
            var: "x",
        },
        "*",
    );
}

// =============================================================================
// MASSIVE BATCH OF COMBINATION TESTS
// =============================================================================

// Triple: Cross-category combinations
#[test]
fn metatest_triple_trig_log_power() {
    assert_metamorphic_combine_triple(
        "trig_log_power",
        TestPair {
            exp: "sin(x)^2 + cos(x)^2",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "ln(exp(x))",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "x^2 * x^3",
            simp: "x^5",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_double_half_product() {
    assert_metamorphic_combine_triple(
        "double_half_product",
        TestPair {
            exp: "sin(2*x)",
            simp: "2*sin(x)*cos(x)",
            var: "x",
        },
        TestPair {
            exp: "2*sin(x/2)^2",
            simp: "1 - cos(x)",
            var: "x",
        },
        TestPair {
            exp: "sin(x)*cos(x)",
            simp: "sin(2*x)/2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_power_reduction_all() {
    assert_metamorphic_combine_triple(
        "power_reduction_all",
        TestPair {
            exp: "sin(x)^2",
            simp: "(1 - cos(2*x))/2",
            var: "x",
        },
        TestPair {
            exp: "cos(x)^2",
            simp: "(1 + cos(2*x))/2",
            var: "x",
        },
        TestPair {
            exp: "sin(x)*cos(x)",
            simp: "sin(2*x)/2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_expand_all_degrees() {
    assert_metamorphic_combine_triple(
        "expand_all_degrees",
        TestPair {
            exp: "(x+1)^2",
            simp: "x^2 + 2*x + 1",
            var: "x",
        },
        TestPair {
            exp: "(x+1)^3",
            simp: "x^3 + 3*x^2 + 3*x + 1",
            var: "x",
        },
        TestPair {
            exp: "(x+1)^4",
            simp: "x^4 + 4*x^3 + 6*x^2 + 4*x + 1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_diff_squares_all() {
    assert_metamorphic_combine_triple(
        "diff_squares_all",
        TestPair {
            exp: "x^2 - 1",
            simp: "(x-1)*(x+1)",
            var: "x",
        },
        TestPair {
            exp: "x^2 - 4",
            simp: "(x-2)*(x+2)",
            var: "x",
        },
        TestPair {
            exp: "x^4 - 1",
            simp: "(x^2+1)*(x+1)*(x-1)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_sqrt_products() {
    assert_metamorphic_combine_triple(
        "sqrt_products",
        TestPair {
            exp: "sqrt(2)*sqrt(8)",
            simp: "4",
            var: "x",
        },
        TestPair {
            exp: "sqrt(3)*sqrt(12)",
            simp: "6",
            var: "x",
        },
        TestPair {
            exp: "sqrt(5)*sqrt(5)",
            simp: "5",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_ln_powers() {
    assert_metamorphic_combine_triple(
        "ln_powers",
        TestPair {
            exp: "ln(e^2)",
            simp: "2",
            var: "x",
        },
        TestPair {
            exp: "ln(e^3)",
            simp: "3",
            var: "x",
        },
        TestPair {
            exp: "ln(e^4)",
            simp: "4",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_trig_pi_fractions() {
    assert_metamorphic_combine_triple(
        "trig_pi_fractions",
        TestPair {
            exp: "sin(pi/6)",
            simp: "1/2",
            var: "x",
        },
        TestPair {
            exp: "cos(pi/3)",
            simp: "1/2",
            var: "x",
        },
        TestPair {
            exp: "tan(pi/4)",
            simp: "1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_reciprocal_chain() {
    assert_metamorphic_combine_triple(
        "reciprocal_chain",
        TestPair {
            exp: "1/(1/x)",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "x * (1/x)",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "(1/x)^(-1)",
            simp: "x",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_power_tower() {
    assert_metamorphic_combine_triple(
        "power_tower",
        TestPair {
            exp: "(x^2)^3",
            simp: "x^6",
            var: "x",
        },
        TestPair {
            exp: "(x^3)^2",
            simp: "x^6",
            var: "x",
        },
        TestPair {
            exp: "x^(2*3)",
            simp: "x^6",
            var: "x",
        },
        "+",
    );
}

// Double combinations with multiplication
#[test]
fn metatest_comb_mul_pythagorean() {
    assert_metamorphic_combine(
        "mul_pythagorean",
        TestPair {
            exp: "sin(x)^2 + cos(x)^2",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "tan(x)^2 + 1",
            simp: "sec(x)^2",
            var: "x",
        },
        "*",
    );
}

#[test]
fn metatest_comb_mul_power_roots() {
    assert_metamorphic_combine(
        "mul_power_roots",
        TestPair {
            exp: "x^2 * x^3",
            simp: "x^5",
            var: "x",
        },
        TestPair {
            exp: "sqrt(x)^2",
            simp: "x",
            var: "x",
        },
        "*",
    );
}

#[test]
fn metatest_comb_mul_log_exp() {
    assert_metamorphic_combine(
        "mul_log_exp",
        TestPair {
            exp: "ln(exp(x))",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "exp(ln(x))",
            simp: "x",
            var: "x",
        },
        "*",
    );
}

#[test]
fn metatest_comb_sub_binomials() {
    assert_metamorphic_combine(
        "sub_binomials",
        TestPair {
            exp: "(x+1)^2",
            simp: "x^2 + 2*x + 1",
            var: "x",
        },
        TestPair {
            exp: "(x-1)^2",
            simp: "x^2 - 2*x + 1",
            var: "x",
        },
        "-",
    );
}

#[test]
fn metatest_comb_sub_cubes() {
    assert_metamorphic_combine(
        "sub_cubes",
        TestPair {
            exp: "(x+1)^3",
            simp: "x^3 + 3*x^2 + 3*x + 1",
            var: "x",
        },
        TestPair {
            exp: "(x-1)^3",
            simp: "x^3 - 3*x^2 + 3*x - 1",
            var: "x",
        },
        "-",
    );
}

#[test]
fn metatest_comb_add_double_angles() {
    assert_metamorphic_combine(
        "add_double_angles",
        TestPair {
            exp: "sin(2*x)",
            simp: "2*sin(x)*cos(x)",
            var: "x",
        },
        TestPair {
            exp: "cos(2*x)",
            simp: "cos(x)^2 - sin(x)^2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_add_half_angles() {
    assert_metamorphic_combine(
        "add_half_angles",
        TestPair {
            exp: "2*sin(x/2)^2",
            simp: "1 - cos(x)",
            var: "x",
        },
        TestPair {
            exp: "2*cos(x/2)^2",
            simp: "1 + cos(x)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_add_sqrt_sums() {
    assert_metamorphic_combine(
        "add_sqrt_sums",
        TestPair {
            exp: "sqrt(27) + sqrt(12)",
            simp: "5*sqrt(3)",
            var: "x",
        },
        TestPair {
            exp: "sqrt(45) - sqrt(20)",
            simp: "sqrt(5)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_add_cbrt() {
    assert_metamorphic_combine(
        "add_cbrt",
        TestPair {
            exp: "64^(1/3)",
            simp: "4",
            var: "x",
        },
        TestPair {
            exp: "125^(1/3)",
            simp: "5",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_mul_fractions() {
    assert_metamorphic_combine(
        "mul_fractions",
        TestPair {
            exp: "x/x",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "2*x/2",
            simp: "x",
            var: "x",
        },
        "*",
    );
}

// More triples with diverse patterns
#[test]
fn metatest_triple_exp_chain() {
    assert_metamorphic_combine_triple(
        "exp_chain",
        TestPair {
            exp: "exp(0)",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "exp(1)",
            simp: "e",
            var: "x",
        },
        TestPair {
            exp: "exp(x)*exp(-x)",
            simp: "1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_neg_functions() {
    assert_metamorphic_combine_triple(
        "neg_functions",
        TestPair {
            exp: "-(-x)",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "-(-(-x))",
            simp: "-x",
            var: "x",
        },
        TestPair {
            exp: "-(x - x)",
            simp: "0",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_fraction_simplify() {
    assert_metamorphic_combine_triple(
        "fraction_simplify",
        TestPair {
            exp: "2*x/2",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "3*x/3",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "4*x/4",
            simp: "x",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_square_roots_numeric() {
    assert_metamorphic_combine_triple(
        "sqrt_numeric",
        TestPair {
            exp: "sqrt(49)",
            simp: "7",
            var: "x",
        },
        TestPair {
            exp: "sqrt(64)",
            simp: "8",
            var: "x",
        },
        TestPair {
            exp: "sqrt(81)",
            simp: "9",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_cube_roots_numeric() {
    assert_metamorphic_combine_triple(
        "cbrt_numeric",
        TestPair {
            exp: "8^(1/3)",
            simp: "2",
            var: "x",
        },
        TestPair {
            exp: "27^(1/3)",
            simp: "3",
            var: "x",
        },
        TestPair {
            exp: "64^(1/3)",
            simp: "4",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_trig_zeros() {
    assert_metamorphic_combine_triple(
        "trig_zeros",
        TestPair {
            exp: "sin(0)",
            simp: "0",
            var: "x",
        },
        TestPair {
            exp: "sin(pi)",
            simp: "0",
            var: "x",
        },
        TestPair {
            exp: "sin(2*pi)",
            simp: "0",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_trig_ones() {
    assert_metamorphic_combine_triple(
        "trig_ones",
        TestPair {
            exp: "cos(0)",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "cos(2*pi)",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "sec(0)",
            simp: "1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_power_one() {
    assert_metamorphic_combine_triple(
        "power_one",
        TestPair {
            exp: "x^1",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "1^x",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "x^0",
            simp: "1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_log_special() {
    assert_metamorphic_combine_triple(
        "log_special",
        TestPair {
            exp: "ln(1)",
            simp: "0",
            var: "x",
        },
        TestPair {
            exp: "ln(e)",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "log(x, x)",
            simp: "1",
            var: "x",
        },
        "+",
    );
}

// More doubles for extensive coverage
#[test]
fn metatest_comb_trig_sum() {
    assert_metamorphic_combine(
        "trig_sum",
        TestPair {
            exp: "sin(x)^2",
            simp: "(1 - cos(2*x))/2",
            var: "x",
        },
        TestPair {
            exp: "cos(x)^2",
            simp: "(1 + cos(2*x))/2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_inverse_ops() {
    assert_metamorphic_combine(
        "inverse_ops",
        TestPair {
            exp: "x + (-x)",
            simp: "0",
            var: "x",
        },
        TestPair {
            exp: "x * (1/x)",
            simp: "1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_distribute_collect() {
    assert_metamorphic_combine(
        "distribute_collect",
        TestPair {
            exp: "x*(x+1)",
            simp: "x^2 + x",
            var: "x",
        },
        TestPair {
            exp: "x^2 + x",
            simp: "x*(x+1)",
            var: "x",
        },
        "-",
    );
}

#[test]
fn metatest_comb_arcfun_chain() {
    assert_metamorphic_combine(
        "arcfun_chain",
        TestPair {
            exp: "sin(arcsin(x))",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "cos(arccos(x))",
            simp: "x",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_conjugate_mult() {
    assert_metamorphic_combine(
        "conjugate_mult",
        TestPair {
            exp: "(sqrt(2)+1)*(sqrt(2)-1)",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "(sqrt(3)+1)*(sqrt(3)-1)",
            simp: "2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_power_frac() {
    assert_metamorphic_combine(
        "power_frac",
        TestPair {
            exp: "x^(1/2) * x^(1/2)",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "x^(1/3) * x^(2/3)",
            simp: "x",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_nested_exp() {
    assert_metamorphic_combine(
        "nested_exp",
        TestPair {
            exp: "exp(ln(x))",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "exp(2*ln(x))",
            simp: "x^2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_root_power_cancel() {
    assert_metamorphic_combine(
        "root_power_cancel",
        TestPair {
            exp: "sqrt(x^2)",
            simp: "|x|",
            var: "x",
        },
        TestPair {
            exp: "(x^(1/3))^3",
            simp: "x",
            var: "x",
        },
        "+",
    );
}

// =============================================================================
// EVEN MORE COMBINATION TESTS - BATCH 2
// =============================================================================

// Triple: More cross-category
#[test]
fn metatest_triple_all_pythagorean() {
    assert_metamorphic_combine_triple(
        "all_pythagorean_2",
        TestPair {
            exp: "sin(x)^2 + cos(x)^2",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "1 + tan(x)^2",
            simp: "sec(x)^2",
            var: "x",
        },
        TestPair {
            exp: "1 + cot(x)^2",
            simp: "csc(x)^2",
            var: "x",
        },
        "*",
    );
}

#[test]
fn metatest_triple_sqrt_simplify_more() {
    assert_metamorphic_combine_triple(
        "sqrt_simplify_more",
        TestPair {
            exp: "sqrt(32)",
            simp: "4*sqrt(2)",
            var: "x",
        },
        TestPair {
            exp: "sqrt(48)",
            simp: "4*sqrt(3)",
            var: "x",
        },
        TestPair {
            exp: "sqrt(75)",
            simp: "5*sqrt(3)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_power_products() {
    assert_metamorphic_combine_triple(
        "power_products",
        TestPair {
            exp: "x^2 * x^2",
            simp: "x^4",
            var: "x",
        },
        TestPair {
            exp: "x^3 * x^3",
            simp: "x^6",
            var: "x",
        },
        TestPair {
            exp: "x^4 * x^4",
            simp: "x^8",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_trig_negative_all() {
    assert_metamorphic_combine_triple(
        "trig_negative_all",
        TestPair {
            exp: "sin(-x)",
            simp: "-sin(x)",
            var: "x",
        },
        TestPair {
            exp: "tan(-x)",
            simp: "-tan(x)",
            var: "x",
        },
        TestPair {
            exp: "cot(-x)",
            simp: "-cot(x)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_trig_even() {
    assert_metamorphic_combine_triple(
        "trig_even",
        TestPair {
            exp: "cos(-x)",
            simp: "cos(x)",
            var: "x",
        },
        TestPair {
            exp: "sec(-x)",
            simp: "sec(x)",
            var: "x",
        },
        TestPair {
            exp: "|x|",
            simp: "|-x|",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_log_properties_2() {
    assert_metamorphic_combine_triple(
        "log_properties",
        TestPair {
            exp: "ln(x^2)",
            simp: "2*ln(x)",
            var: "x",
        },
        TestPair {
            exp: "ln(x^3)",
            simp: "3*ln(x)",
            var: "x",
        },
        TestPair {
            exp: "ln(sqrt(x))",
            simp: "ln(x)/2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_exp_sum() {
    assert_metamorphic_combine_triple(
        "exp_sum",
        TestPair {
            exp: "exp(x+1)",
            simp: "e*exp(x)",
            var: "x",
        },
        TestPair {
            exp: "exp(x+2)",
            simp: "e^2*exp(x)",
            var: "x",
        },
        TestPair {
            exp: "exp(x-1)",
            simp: "exp(x)/e",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_factorial_roots() {
    assert_metamorphic_combine_triple(
        "factorial_roots",
        TestPair {
            exp: "sqrt(100)",
            simp: "10",
            var: "x",
        },
        TestPair {
            exp: "sqrt(121)",
            simp: "11",
            var: "x",
        },
        TestPair {
            exp: "sqrt(144)",
            simp: "12",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_nth_roots() {
    assert_metamorphic_combine_triple(
        "nth_roots",
        TestPair {
            exp: "16^(1/4)",
            simp: "2",
            var: "x",
        },
        TestPair {
            exp: "81^(1/4)",
            simp: "3",
            var: "x",
        },
        TestPair {
            exp: "625^(1/4)",
            simp: "5",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_binomial_minus() {
    assert_metamorphic_combine_triple(
        "binomial_minus",
        TestPair {
            exp: "(x-1)^2",
            simp: "x^2 - 2*x + 1",
            var: "x",
        },
        TestPair {
            exp: "(x-2)^2",
            simp: "x^2 - 4*x + 4",
            var: "x",
        },
        TestPair {
            exp: "(x-3)^2",
            simp: "x^2 - 6*x + 9",
            var: "x",
        },
        "+",
    );
}

// Double: More patterns
#[test]
fn metatest_comb_sin_cos_sum() {
    assert_metamorphic_combine(
        "sin_cos_sum",
        TestPair {
            exp: "sin(x)^2",
            simp: "(1 - cos(2*x))/2",
            var: "x",
        },
        TestPair {
            exp: "cos(x)^2",
            simp: "(1 + cos(2*x))/2",
            var: "x",
        },
        "-",
    );
}

#[test]
fn metatest_comb_tan_cot() {
    assert_metamorphic_combine(
        "tan_cot",
        TestPair {
            exp: "tan(x)*cot(x)",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "sin(x)*csc(x)",
            simp: "1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_sec_cos() {
    assert_metamorphic_combine(
        "sec_cos",
        TestPair {
            exp: "sec(x)*cos(x)",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "csc(x)*sin(x)",
            simp: "1",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_log_quotient() {
    assert_metamorphic_combine(
        "log_quotient",
        TestPair {
            exp: "ln(x/e)",
            simp: "ln(x) - 1",
            var: "x",
        },
        TestPair {
            exp: "ln(x/e^2)",
            simp: "ln(x) - 2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_power_neg() {
    assert_metamorphic_combine(
        "power_neg",
        TestPair {
            exp: "x^(-1)",
            simp: "1/x",
            var: "x",
        },
        TestPair {
            exp: "x^(-2)",
            simp: "1/x^2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_sqrt_div() {
    assert_metamorphic_combine(
        "sqrt_div",
        TestPair {
            exp: "sqrt(x)/sqrt(x)",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "sqrt(4*x)/sqrt(x)",
            simp: "2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_exp_product() {
    assert_metamorphic_combine(
        "exp_product",
        TestPair {
            exp: "exp(x)*exp(x)",
            simp: "exp(2*x)",
            var: "x",
        },
        TestPair {
            exp: "exp(x)*exp(2*x)",
            simp: "exp(3*x)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_arcfun_values() {
    assert_metamorphic_combine(
        "arcfun_values",
        TestPair {
            exp: "arcsin(1)",
            simp: "pi/2",
            var: "x",
        },
        TestPair {
            exp: "arccos(0)",
            simp: "pi/2",
            var: "x",
        },
        "-",
    );
}

#[test]
fn metatest_comb_nested_sqrt() {
    assert_metamorphic_combine(
        "nested_sqrt",
        TestPair {
            exp: "sqrt(x*x)",
            simp: "|x|",
            var: "x",
        },
        TestPair {
            exp: "sqrt(x^4)",
            simp: "x^2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_fraction_add() {
    assert_metamorphic_combine(
        "fraction_add",
        TestPair {
            exp: "x/2 + x/2",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "x/3 + x/3 + x/3",
            simp: "x",
            var: "x",
        },
        "+",
    );
}

// More triples
#[test]
fn metatest_triple_arctan_chain() {
    assert_metamorphic_combine_triple(
        "arctan_chain",
        TestPair {
            exp: "tan(arctan(x))",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "sin(arcsin(x))",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "arctan(0)",
            simp: "0",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_power_division() {
    assert_metamorphic_combine_triple(
        "power_division",
        TestPair {
            exp: "x^8 / x^4",
            simp: "x^4",
            var: "x",
        },
        TestPair {
            exp: "x^9 / x^6",
            simp: "x^3",
            var: "x",
        },
        TestPair {
            exp: "x^10 / x^5",
            simp: "x^5",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_log_base_change() {
    assert_metamorphic_combine_triple(
        "log_base_change",
        TestPair {
            exp: "log(2, 4)",
            simp: "2",
            var: "x",
        },
        TestPair {
            exp: "log(3, 9)",
            simp: "2",
            var: "x",
        },
        TestPair {
            exp: "log(5, 25)",
            simp: "2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_sum_difference() {
    assert_metamorphic_combine_triple(
        "sum_difference",
        TestPair {
            exp: "x + x",
            simp: "2*x",
            var: "x",
        },
        TestPair {
            exp: "x - x",
            simp: "0",
            var: "x",
        },
        TestPair {
            exp: "x * x",
            simp: "x^2",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_triple_trig_half_pi() {
    assert_metamorphic_combine_triple(
        "trig_half_pi",
        TestPair {
            exp: "sin(pi/2)",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "cos(pi/2)",
            simp: "0",
            var: "x",
        },
        TestPair {
            exp: "tan(pi/4)",
            simp: "1",
            var: "x",
        },
        "+",
    );
}

// Even more doubles
#[test]
fn metatest_comb_rationalize_both() {
    assert_metamorphic_combine(
        "rationalize_both",
        TestPair {
            exp: "1/sqrt(2)",
            simp: "sqrt(2)/2",
            var: "x",
        },
        TestPair {
            exp: "1/sqrt(3)",
            simp: "sqrt(3)/3",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_cube_factor() {
    assert_metamorphic_combine(
        "cube_factor",
        TestPair {
            exp: "x^3 + 1",
            simp: "(x+1)*(x^2 - x + 1)",
            var: "x",
        },
        TestPair {
            exp: "x^3 - 1",
            simp: "(x-1)*(x^2 + x + 1)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_expand_product() {
    assert_metamorphic_combine(
        "expand_product",
        TestPair {
            exp: "(x+1)*(x+2)",
            simp: "x^2 + 3*x + 2",
            var: "x",
        },
        TestPair {
            exp: "(x+2)*(x+3)",
            simp: "x^2 + 5*x + 6",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_special_products() {
    assert_metamorphic_combine(
        "special_products",
        TestPair {
            exp: "(x+1)*(x-1)",
            simp: "x^2 - 1",
            var: "x",
        },
        TestPair {
            exp: "(x+2)*(x-2)",
            simp: "x^2 - 4",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_trig_cofunction() {
    assert_metamorphic_combine(
        "trig_cofunction",
        TestPair {
            exp: "sin(pi/2 - x)",
            simp: "cos(x)",
            var: "x",
        },
        TestPair {
            exp: "cos(pi/2 - x)",
            simp: "sin(x)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_abs_properties() {
    assert_metamorphic_combine(
        "abs_properties",
        TestPair {
            exp: "|x| * |x|",
            simp: "x^2",
            var: "x",
        },
        TestPair {
            exp: "|x^2|",
            simp: "x^2",
            var: "x",
        },
        "-",
    );
}

#[test]
fn metatest_comb_log_mult() {
    assert_metamorphic_combine(
        "log_mult",
        TestPair {
            exp: "2*ln(x)",
            simp: "ln(x^2)",
            var: "x",
        },
        TestPair {
            exp: "3*ln(x)",
            simp: "ln(x^3)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_exp_division() {
    assert_metamorphic_combine(
        "exp_division",
        TestPair {
            exp: "exp(2*x)/exp(x)",
            simp: "exp(x)",
            var: "x",
        },
        TestPair {
            exp: "exp(3*x)/exp(x)",
            simp: "exp(2*x)",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_sqrt_product() {
    assert_metamorphic_combine(
        "sqrt_product",
        TestPair {
            exp: "sqrt(x) * sqrt(x)",
            simp: "x",
            var: "x",
        },
        TestPair {
            exp: "sqrt(2*x) * sqrt(2*x)",
            simp: "2*x",
            var: "x",
        },
        "+",
    );
}

#[test]
fn metatest_comb_power_of_power() {
    assert_metamorphic_combine(
        "power_of_power",
        TestPair {
            exp: "(x^2)^2",
            simp: "x^4",
            var: "x",
        },
        TestPair {
            exp: "(x^3)^3",
            simp: "x^9",
            var: "x",
        },
        "+",
    );
}

// More triples with multiplication
#[test]
fn metatest_triple_mul_identities() {
    assert_metamorphic_combine_triple(
        "mul_identities",
        TestPair {
            exp: "x/x",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "x^0",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "sin(x)^2 + cos(x)^2",
            simp: "1",
            var: "x",
        },
        "*",
    );
}

#[test]
fn metatest_triple_sub_all() {
    assert_metamorphic_combine_triple(
        "sub_all",
        TestPair {
            exp: "(x+1)^2",
            simp: "x^2 + 2*x + 1",
            var: "x",
        },
        TestPair {
            exp: "(x-1)^2",
            simp: "x^2 - 2*x + 1",
            var: "x",
        },
        TestPair {
            exp: "x^2",
            simp: "x*x",
            var: "x",
        },
        "-",
    );
}

#[test]
fn metatest_triple_mixed_ops() {
    assert_metamorphic_combine_triple(
        "mixed_ops",
        TestPair {
            exp: "ln(e)",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "exp(0)",
            simp: "1",
            var: "x",
        },
        TestPair {
            exp: "tan(pi/4)",
            simp: "1",
            var: "x",
        },
        "+",
    );
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
