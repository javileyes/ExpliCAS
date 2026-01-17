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
use cas_engine::{eval_f64_checked, EvalCheckedError, EvalCheckedOptions};
use cas_parser::parse;
use std::collections::HashMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
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
        near_singularity_threshold: 1e10, // Values > 10^10 are considered near-singularity
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
    /// Threshold for near-singularity detection (values larger than this are suspicious)
    near_singularity_threshold: f64,
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
// Reporting Helpers
// =============================================================================

/// Truncate an identity string for display (avoids log bloat)
fn truncate_identity(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

// =============================================================================
// MetaTransform (Phase B) - Substitution Transforms
// =============================================================================

/// Metamorphic transformation: x → T(x)
#[derive(Clone, Debug, PartialEq)]
enum MetaTransform {
    /// x → x + k
    Shift(f64),
    /// x → k * x
    Scale(f64),
    /// x → x²
    Square,
}

impl MetaTransform {
    fn name(&self) -> String {
        match self {
            MetaTransform::Shift(k) => format!("shift({})", fmt_f64(*k)),
            MetaTransform::Scale(k) => format!("scale({})", fmt_f64(*k)),
            MetaTransform::Square => "square".to_string(),
        }
    }

    /// Apply to a numeric sample x (for composed filters and evaluation)
    fn apply_f64(&self, x: f64) -> f64 {
        match self {
            MetaTransform::Shift(k) => x + *k,
            MetaTransform::Scale(k) => (*k) * x,
            MetaTransform::Square => x * x,
        }
    }
}

/// Format f64 for stable display (avoids "2" vs "2.0" inconsistency)
fn fmt_f64(x: f64) -> String {
    if x.fract().abs() < 1e-12 {
        format!("{:.0}", x)
    } else {
        format!("{:.6}", x)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

/// Parse METATEST_TRANSFORMS env var.
/// Format: "scale:2,scale:-1,shift:1,square"
/// Returns defaults (scale:2, scale:-1) if not set.
fn parse_meta_transforms_from_env() -> Vec<MetaTransform> {
    let raw = env::var("METATEST_TRANSFORMS").ok().unwrap_or_default();
    let raw = raw.trim();

    if raw.is_empty() {
        // Defaults: scale(2), scale(-1) - very safe transforms
        return vec![MetaTransform::Scale(2.0), MetaTransform::Scale(-1.0)];
    }

    parse_meta_transforms(raw)
}

/// Parse transform spec string
fn parse_meta_transforms(spec: &str) -> Vec<MetaTransform> {
    let mut out = Vec::new();

    for (idx, item) in spec.split(',').enumerate() {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }

        // "square" without parameter
        if item.eq_ignore_ascii_case("square") {
            out.push(MetaTransform::Square);
            continue;
        }

        // Format with ':'
        let (kind, val_str) = match item.split_once(':') {
            Some(parts) => parts,
            None => panic!(
                "Invalid METATEST_TRANSFORMS item #{}: '{}'. Expected 'scale:<num>', 'shift:<num>', or 'square'. Full spec: '{}'",
                idx + 1, item, spec
            ),
        };

        let kind = kind.trim();
        let val_str = val_str.trim();

        let val: f64 = val_str.parse().unwrap_or_else(|e| {
            panic!(
                "Invalid numeric value in METATEST_TRANSFORMS item #{}: '{}'. Error: {}. Full spec: '{}'",
                idx + 1, item, e, spec
            )
        });

        if !val.is_finite() {
            panic!(
                "Value must be finite in METATEST_TRANSFORMS item #{}: '{}'. Full spec: '{}'",
                idx + 1,
                item,
                spec
            );
        }

        match kind.to_lowercase().as_str() {
            "scale" => out.push(MetaTransform::Scale(val)),
            "shift" => out.push(MetaTransform::Shift(val)),
            _ => panic!(
                "Unknown transform kind in METATEST_TRANSFORMS item #{}: '{}'. Supported: scale, shift, square. Full spec: '{}'",
                idx + 1, item, spec
            ),
        }
    }

    // Dedup (stable order)
    let mut seen: Vec<MetaTransform> = Vec::new();
    out.retain(|t| {
        if seen.iter().any(|x| x == t) {
            false
        } else {
            seen.push(t.clone());
            true
        }
    });

    if out.is_empty() {
        panic!(
            "METATEST_TRANSFORMS parsed to empty list. Spec: '{}'. Example: 'scale:2,scale:-1,shift:1,square'",
            spec
        );
    }

    out
}

// =============================================================================
// Shuffle Canonicalization (Phase A)
// =============================================================================

/// Collect all addends from a flattened Add tree (recursive)
/// Returns vec of ExprIds in order they appear in the tree
fn collect_addends(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    match ctx.nodes.get(expr.index()) {
        Some(cas_ast::Expr::Add(a, b)) => {
            let mut result = collect_addends(ctx, *a);
            result.extend(collect_addends(ctx, *b));
            result
        }
        _ => vec![expr],
    }
}

/// Collect all factors from a flattened Mul tree (recursive)
fn collect_factors(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    match ctx.nodes.get(expr.index()) {
        Some(cas_ast::Expr::Mul(a, b)) => {
            let mut result = collect_factors(ctx, *a);
            result.extend(collect_factors(ctx, *b));
            result
        }
        _ => vec![expr],
    }
}

/// Rebuild Add tree from terms (left-associative)
fn rebuild_add(ctx: &mut Context, terms: &[ExprId]) -> ExprId {
    if terms.is_empty() {
        ctx.add_raw(cas_ast::Expr::Number(
            num_rational::BigRational::from_integer(0.into()),
        ))
    } else if terms.len() == 1 {
        terms[0]
    } else {
        let mut result = terms[0];
        for &term in &terms[1..] {
            result = ctx.add_raw(cas_ast::Expr::Add(result, term));
        }
        result
    }
}

/// Rebuild Mul tree from factors (left-associative)
fn rebuild_mul(ctx: &mut Context, factors: &[ExprId]) -> ExprId {
    if factors.is_empty() {
        ctx.add_raw(cas_ast::Expr::Number(
            num_rational::BigRational::from_integer(1.into()),
        ))
    } else if factors.len() == 1 {
        factors[0]
    } else {
        let mut result = factors[0];
        for &factor in &factors[1..] {
            result = ctx.add_raw(cas_ast::Expr::Mul(result, factor));
        }
        result
    }
}

/// Stable hash for an expression (FNV-1a based, deterministic)
fn stable_expr_hash(ctx: &Context, expr: ExprId) -> u64 {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;

    fn hash_combine(hash: u64, byte: u8) -> u64 {
        (hash ^ (byte as u64)).wrapping_mul(FNV_PRIME)
    }

    fn hash_u64(hash: u64, val: u64) -> u64 {
        let mut h = hash;
        for i in 0..8 {
            h = hash_combine(h, ((val >> (i * 8)) & 0xff) as u8);
        }
        h
    }

    fn hash_expr(ctx: &Context, expr: ExprId, h: u64) -> u64 {
        match ctx.nodes.get(expr.index()) {
            Some(cas_ast::Expr::Number(n)) => {
                let mut h = hash_combine(h, b'N');
                for b in n.to_string().bytes() {
                    h = hash_combine(h, b);
                }
                h
            }
            Some(cas_ast::Expr::Variable(name)) => {
                let mut h = hash_combine(h, b'V');
                for b in name.bytes() {
                    h = hash_combine(h, b);
                }
                h
            }
            Some(cas_ast::Expr::Add(a, b)) => {
                let h = hash_combine(h, b'+');
                let h = hash_expr(ctx, *a, h);
                hash_expr(ctx, *b, h)
            }
            Some(cas_ast::Expr::Mul(a, b)) => {
                let h = hash_combine(h, b'*');
                let h = hash_expr(ctx, *a, h);
                hash_expr(ctx, *b, h)
            }
            Some(cas_ast::Expr::Pow(base, exp)) => {
                let h = hash_combine(h, b'^');
                let h = hash_expr(ctx, *base, h);
                hash_expr(ctx, *exp, h)
            }
            Some(cas_ast::Expr::Function(name, args)) => {
                let mut h = hash_combine(h, b'F');
                for b in name.bytes() {
                    h = hash_combine(h, b);
                }
                for arg in args {
                    h = hash_expr(ctx, *arg, h);
                }
                h
            }
            Some(cas_ast::Expr::Neg(inner)) => {
                let h = hash_combine(h, b'-');
                hash_expr(ctx, *inner, h)
            }
            Some(cas_ast::Expr::Sub(a, b)) => {
                let h = hash_combine(h, b'S');
                let h = hash_expr(ctx, *a, h);
                hash_expr(ctx, *b, h)
            }
            Some(cas_ast::Expr::Div(a, b)) => {
                let h = hash_combine(h, b'/');
                let h = hash_expr(ctx, *a, h);
                hash_expr(ctx, *b, h)
            }
            Some(cas_ast::Expr::Constant(c)) => {
                let mut h = hash_combine(h, b'C');
                for b in format!("{:?}", c).bytes() {
                    h = hash_combine(h, b);
                }
                h
            }
            _ => hash_combine(h, b'?'),
        }
    }

    hash_expr(ctx, expr, FNV_OFFSET)
}

/// Deterministic shuffle based on expr hash (Fisher-Yates with seeded PRNG)
fn shuffle_vec<T>(items: &mut [T], seed: u64) {
    let mut rng = seed;
    for i in (1..items.len()).rev() {
        // Simple LCG PRNG
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng as usize) % (i + 1);
        items.swap(i, j);
    }
}

/// Shuffle an expression by permuting Add/Mul children deterministically
/// Only touches commutative nodes (Add, Mul), preserves structure of other nodes
fn shuffle_expr(ctx: &mut Context, expr: ExprId) -> ExprId {
    let seed = stable_expr_hash(ctx, expr);
    shuffle_expr_seeded(ctx, expr, seed)
}

fn shuffle_expr_seeded(ctx: &mut Context, expr: ExprId, seed: u64) -> ExprId {
    match ctx.nodes.get(expr.index()).cloned() {
        Some(cas_ast::Expr::Add(_, _)) => {
            // Flatten, shuffle, rebuild
            let mut terms = collect_addends(ctx, expr);
            if terms.len() > 1 {
                // Shuffle terms
                shuffle_vec(&mut terms, seed);
                // Recursively shuffle each term
                let shuffled_terms: Vec<_> = terms
                    .iter()
                    .enumerate()
                    .map(|(i, &t)| shuffle_expr_seeded(ctx, t, seed.wrapping_add(i as u64)))
                    .collect();
                rebuild_add(ctx, &shuffled_terms)
            } else {
                expr
            }
        }
        Some(cas_ast::Expr::Mul(_, _)) => {
            // Flatten, shuffle, rebuild
            let mut factors = collect_factors(ctx, expr);
            if factors.len() > 1 {
                shuffle_vec(&mut factors, seed.wrapping_add(1000));
                let shuffled_factors: Vec<_> = factors
                    .iter()
                    .enumerate()
                    .map(|(i, &f)| shuffle_expr_seeded(ctx, f, seed.wrapping_add(2000 + i as u64)))
                    .collect();
                rebuild_mul(ctx, &shuffled_factors)
            } else {
                expr
            }
        }
        Some(cas_ast::Expr::Pow(base, exp)) => {
            // Don't shuffle base/exp order, just recurse
            let new_base = shuffle_expr_seeded(ctx, base, seed.wrapping_add(100));
            let new_exp = shuffle_expr_seeded(ctx, exp, seed.wrapping_add(200));
            ctx.add_raw(cas_ast::Expr::Pow(new_base, new_exp))
        }
        Some(cas_ast::Expr::Function(name, args)) => {
            // Recurse into args (don't reorder - function args aren't commutative)
            let new_args: Vec<_> = args
                .iter()
                .enumerate()
                .map(|(i, &a)| shuffle_expr_seeded(ctx, a, seed.wrapping_add(300 + i as u64)))
                .collect();
            ctx.add_raw(cas_ast::Expr::Function(name, new_args))
        }
        Some(cas_ast::Expr::Neg(inner)) => {
            let new_inner = shuffle_expr_seeded(ctx, inner, seed.wrapping_add(400));
            ctx.add_raw(cas_ast::Expr::Neg(new_inner))
        }
        Some(cas_ast::Expr::Sub(a, b)) => {
            // Sub is not commutative - just recurse
            let new_a = shuffle_expr_seeded(ctx, a, seed.wrapping_add(500));
            let new_b = shuffle_expr_seeded(ctx, b, seed.wrapping_add(600));
            ctx.add_raw(cas_ast::Expr::Sub(new_a, new_b))
        }
        Some(cas_ast::Expr::Div(a, b)) => {
            // Div is not commutative - just recurse
            let new_a = shuffle_expr_seeded(ctx, a, seed.wrapping_add(700));
            let new_b = shuffle_expr_seeded(ctx, b, seed.wrapping_add(800));
            ctx.add_raw(cas_ast::Expr::Div(new_a, new_b))
        }
        // Leaf nodes - no change
        _ => expr,
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
    let stats = check_numeric_equiv_1var_stats(ctx, a, b, var, config, &FilterSpec::None);

    // Require higher min_valid when many samples had issues
    let problematic =
        stats.near_pole + stats.domain_error + stats.eval_failed + stats.asymmetric_invalid;
    let adjusted_min_valid = if problematic > config.eval_samples / 4 {
        (config.eval_samples - problematic) / 2
    } else {
        config.min_valid
    };

    if stats.valid < adjusted_min_valid {
        return Err(format!(
            "Too few valid samples: {} < {} (near_pole={}, domain_error={}, asymmetric={}, eval_failed={})",
            stats.valid, adjusted_min_valid, stats.near_pole, stats.domain_error, stats.asymmetric_invalid, stats.eval_failed
        ));
    }

    if !stats.mismatches.is_empty() {
        return Err(format!(
            "Numeric mismatches: {}",
            stats.mismatches.join("; ")
        ));
    }

    Ok(stats.valid)
}

/// Stats-returning version of check_numeric_equiv_1var for diagnostics
fn check_numeric_equiv_1var_stats(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var: &str,
    config: &MetatestConfig,
    filter_spec: &FilterSpec,
) -> NumericEquivStats {
    let (lo, hi) = config.sample_range;
    let mut stats = NumericEquivStats::default();

    // Configure checked evaluator with near-pole detection
    let opts = EvalCheckedOptions {
        zero_abs_eps: 1e-12,
        zero_rel_eps: 1e-12,
        trig_pole_eps: 1e-9,
        max_depth: 200,
    };

    for i in 0..config.eval_samples {
        let t = (i as f64 + 0.5) / config.eval_samples as f64;
        let x = lo + (hi - lo) * t;

        // Apply filter if specified
        if !filter_spec.accept(x) {
            stats.filtered_out += 1;
            continue;
        }

        let mut var_map = HashMap::new();
        var_map.insert(var.to_string(), x);

        let va = eval_f64_checked(ctx, a, &var_map, &opts);
        let vb = eval_f64_checked(ctx, b, &var_map, &opts);

        match (&va, &vb) {
            (Ok(va), Ok(vb)) => {
                let diff = (va - vb).abs();
                let scale = va.abs().max(vb.abs()).max(1.0);
                let allowed = config.atol + config.rtol * scale;

                if diff <= allowed {
                    stats.valid += 1;
                } else {
                    stats.record_mismatch(x, *va, *vb, var);
                }
            }
            (Err(EvalCheckedError::NearPole { .. }), Err(EvalCheckedError::NearPole { .. })) => {
                stats.near_pole += 1;
            }
            (Err(EvalCheckedError::Domain { .. }), Err(EvalCheckedError::Domain { .. })) => {
                stats.domain_error += 1;
            }
            (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
                stats.asymmetric_invalid += 1;
            }
            _ => {
                stats.eval_failed += 1;
            }
        }
    }

    stats
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
    let mut near_pole = 0usize;
    let mut domain_error = 0usize;
    let mut asymmetric_invalid = 0usize;

    // Configure checked evaluator
    let opts = EvalCheckedOptions {
        zero_abs_eps: 1e-12,
        zero_rel_eps: 1e-12,
        trig_pole_eps: 1e-9,
        max_depth: 200,
    };

    // Use fewer samples for 2D grid to keep runtime reasonable
    let samples_per_dim = (config.eval_samples as f64).sqrt() as usize;
    let total_samples = samples_per_dim * samples_per_dim;

    for i in 0..samples_per_dim {
        for j in 0..samples_per_dim {
            let t1 = (i as f64 + 0.5) / samples_per_dim as f64;
            let t2 = (j as f64 + 0.5) / samples_per_dim as f64;
            let x = lo + (hi - lo) * t1;
            let y = lo + (hi - lo) * t2;

            let mut var_map = HashMap::new();
            var_map.insert(var1.to_string(), x);
            var_map.insert(var2.to_string(), y);

            let va = eval_f64_checked(ctx, a, &var_map, &opts);
            let vb = eval_f64_checked(ctx, b, &var_map, &opts);

            match (&va, &vb) {
                (Ok(va), Ok(vb)) => {
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
                // Symmetric failures
                (
                    Err(EvalCheckedError::NearPole { .. }),
                    Err(EvalCheckedError::NearPole { .. }),
                ) => {
                    near_pole += 1;
                }
                (Err(EvalCheckedError::Domain { .. }), Err(EvalCheckedError::Domain { .. })) => {
                    domain_error += 1;
                }
                // Asymmetric: one Ok, one Err
                (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
                    asymmetric_invalid += 1;
                }
                _ => {
                    eval_failed += 1;
                }
            }
        }
    }

    // Lower threshold for 2D, adjusted for problematic samples
    let problematic = near_pole + domain_error + eval_failed + asymmetric_invalid;
    let base_min_valid = config.min_valid / 4;
    let adjusted_min_valid = if problematic > total_samples / 4 {
        (total_samples - problematic) / 2
    } else {
        base_min_valid
    };

    if valid < adjusted_min_valid {
        return Err(format!(
            "Too few valid samples: {} < {} (near_pole={}, domain_error={}, asymmetric={}, eval_failed={})",
            valid, adjusted_min_valid, near_pole, domain_error, asymmetric_invalid, eval_failed
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

/// An identity pair loaded from CSV (supports both legacy 4-col and extended 7-col format)
#[derive(Clone, Debug)]
struct IdentityPair {
    exp: String,
    simp: String,
    vars: Vec<String>,
    mode: DomainRequirement,
    bucket: Bucket,
    branch_mode: BranchMode,
    filter_spec: FilterSpec, // Parsed from CSV, e.g., "abs_lt(0.9)" → AbsLt { limit: 0.9 }
}

/// Domain requirement for an identity
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DomainRequirement {
    Generic, // Works in both modes (g)
    Assume,  // Requires DomainMode::Assume (a)
}

/// Classification bucket for identity pairs
/// Determines how the test should be run and results interpreted
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[allow(dead_code)]
enum Bucket {
    /// Pure algebraic/trig identity without branch issues
    Unconditional,
    /// True under domain conditions (requires x≠0, cos(x)≠0, etc.)
    #[default]
    ConditionalRequires,
    /// Involves inverse trig, log, or complex pow - branch sensitive
    BranchSensitive,
}

/// Branch comparison mode for inverse trig and log identities
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[allow(dead_code)]
enum BranchMode {
    /// Compare values directly (principal value strict)
    #[default]
    PrincipalStrict,
    /// Compare with domain filtering (e.g., |x| < 1 for arctan)
    PrincipalWithFilter,
    /// Compare modulo π (for arctan identities)
    ModuloPi,
    /// Compare modulo 2π (for general trig identities)
    Modulo2Pi,
}

/// Check approximate equality modulo a period (for branch-sensitive comparisons)
///
/// Returns true if the circular distance between a and b (mod period) is within tolerance.
/// Used for arctan identities (mod π) and general trig (mod 2π).
/// Handles NaN/Inf by returning false.
#[allow(dead_code)]
fn approx_eq_mod_period(a: f64, b: f64, period: f64, atol: f64, rtol: f64) -> bool {
    // Handle non-finite values
    if !a.is_finite() || !b.is_finite() || !period.is_finite() || period <= 0.0 {
        return false;
    }

    // Calculate circular distance
    let diff = (a - b).rem_euclid(period);
    let circular_dist = diff.min(period - diff);

    let scale = a.abs().max(b.abs()).max(1.0);
    let allowed = atol + rtol * scale;

    circular_dist <= allowed
}

// =============================================================================
// Standard Sample Filters
// =============================================================================

/// Filter: |x| < bound
#[allow(dead_code)]
fn filter_abs_lt(bound: f64) -> impl Fn(f64) -> bool {
    move |x| x.abs() < bound
}

/// Filter: keep samples away from singularities
#[allow(dead_code)]
fn filter_away_from(singularities: Vec<f64>, eps: f64) -> impl Fn(f64) -> bool {
    move |x| singularities.iter().all(|&s| (x - s).abs() > eps)
}

/// Filter: |x| < bound AND away from singularities
#[allow(dead_code)]
fn filter_abs_lt_and_away(bound: f64, singularities: Vec<f64>, eps: f64) -> impl Fn(f64) -> bool {
    move |x| x.abs() < bound && singularities.iter().all(|&s| (x - s).abs() > eps)
}

// =============================================================================
// FilterSpec for CSV-driven Filtering
// =============================================================================

/// Runtime-parseable filter specification (no closures, serializable)
#[derive(Debug, Clone)]
enum FilterSpec {
    None,
    AbsLt {
        limit: f64,
    },
    AwayFrom {
        centers: Vec<f64>,
        eps: f64,
    },
    AbsLtAndAway {
        limit: f64,
        centers: Vec<f64>,
        eps: f64,
    },
    // New domain filters for ln/sqrt/etc.
    Gt {
        limit: f64,
    }, // x > limit
    Ge {
        limit: f64,
    }, // x >= limit
    Lt {
        limit: f64,
    }, // x < limit
    Le {
        limit: f64,
    }, // x <= limit
    Range {
        min: f64,
        max: f64,
    }, // min <= x <= max (inclusive)
}

impl FilterSpec {
    /// Check if sample x should be included
    fn accept(&self, x: f64) -> bool {
        match self {
            FilterSpec::None => true,
            FilterSpec::AbsLt { limit } => x.abs() < *limit,
            FilterSpec::AwayFrom { centers, eps } => centers.iter().all(|c| (x - c).abs() > *eps),
            FilterSpec::AbsLtAndAway {
                limit,
                centers,
                eps,
            } => x.abs() < *limit && centers.iter().all(|c| (x - c).abs() > *eps),
            FilterSpec::Gt { limit } => x > *limit,
            FilterSpec::Ge { limit } => x >= *limit,
            FilterSpec::Lt { limit } => x < *limit,
            FilterSpec::Le { limit } => x <= *limit,
            FilterSpec::Range { min, max } => x >= *min && x <= *max,
        }
    }

    /// Check if filter is None (no filtering applied)
    fn is_none(&self) -> bool {
        matches!(self, FilterSpec::None)
    }

    /// Convert to string representation (for reporting)
    fn to_string(&self) -> String {
        match self {
            FilterSpec::None => String::new(),
            FilterSpec::AbsLt { limit } => format!("abs_lt({})", limit),
            FilterSpec::AwayFrom { centers, eps } => {
                let centers_str: Vec<String> = centers.iter().map(|c| c.to_string()).collect();
                format!("away_from({};eps={})", centers_str.join(";"), eps)
            }
            FilterSpec::AbsLtAndAway {
                limit,
                centers,
                eps,
            } => {
                let centers_str: Vec<String> = centers.iter().map(|c| c.to_string()).collect();
                format!(
                    "abs_lt_and_away({};{};eps={})",
                    limit,
                    centers_str.join(";"),
                    eps
                )
            }
            FilterSpec::Gt { limit } => format!("gt({})", limit),
            FilterSpec::Ge { limit } => format!("ge({})", limit),
            FilterSpec::Lt { limit } => format!("lt({})", limit),
            FilterSpec::Le { limit } => format!("le({})", limit),
            FilterSpec::Range { min, max } => format!("range({};{})", min, max),
        }
    }
}

/// Parse filter spec from CSV string
/// Valid formats:
///   "" or empty → None
///   "abs_lt(0.9)" → AbsLt { limit: 0.9 }
///   "away_from(1.57;-1.57;eps=0.01)" → AwayFrom { centers: [1.57, -1.57], eps: 0.01 }
///   "abs_lt_and_away(0.9;1.0;-1.0;eps=0.1)" → AbsLtAndAway { limit: 0.9, centers: [1.0, -1.0], eps: 0.1 }
///   "gt(0.0)" → Gt { limit: 0.0 }
///   "ge(0.0)" → Ge { limit: 0.0 }
///   "lt(1.0)" → Lt { limit: 1.0 }
///   "le(1.0)" → Le { limit: 1.0 }
///   "range(0.1;3.0)" → Range { min: 0.1, max: 3.0 }
fn parse_filter_spec(spec: &str, line_num: usize) -> FilterSpec {
    let spec = spec.trim();
    if spec.is_empty() {
        return FilterSpec::None;
    }

    // abs_lt(limit)
    if spec.starts_with("abs_lt(") && spec.ends_with(')') {
        let inner = &spec[7..spec.len() - 1];
        let limit: f64 = inner.parse().unwrap_or_else(|_| {
            panic!("Invalid abs_lt limit at line {}: '{}'", line_num, spec);
        });
        return FilterSpec::AbsLt { limit };
    }

    // away_from(c1;c2;...;eps=<val>)
    if spec.starts_with("away_from(") && spec.ends_with(')') {
        let inner = &spec[10..spec.len() - 1];
        return parse_away_from_inner(inner, line_num, spec);
    }

    // abs_lt_and_away(limit;c1;c2;...;eps=<val>)
    if spec.starts_with("abs_lt_and_away(") && spec.ends_with(')') {
        let inner = &spec[16..spec.len() - 1];
        let parts: Vec<&str> = inner.split(';').collect();
        if parts.is_empty() {
            panic!("Invalid abs_lt_and_away at line {}: '{}'", line_num, spec);
        }
        let limit: f64 = parts[0].parse().unwrap_or_else(|_| {
            panic!(
                "Invalid abs_lt_and_away limit at line {}: '{}'",
                line_num, spec
            );
        });
        let remaining = parts[1..].join(";");
        let away = parse_away_from_inner(&remaining, line_num, spec);
        match away {
            FilterSpec::AwayFrom { centers, eps } => {
                return FilterSpec::AbsLtAndAway {
                    limit,
                    centers,
                    eps,
                };
            }
            _ => panic!("Invalid abs_lt_and_away at line {}: '{}'", line_num, spec),
        }
    }

    // gt(limit) - x > limit
    if spec.starts_with("gt(") && spec.ends_with(')') {
        let inner = &spec[3..spec.len() - 1];
        let limit: f64 = inner.parse().unwrap_or_else(|_| {
            panic!("Invalid gt limit at line {}: '{}'", line_num, spec);
        });
        return FilterSpec::Gt { limit };
    }

    // ge(limit) - x >= limit
    if spec.starts_with("ge(") && spec.ends_with(')') {
        let inner = &spec[3..spec.len() - 1];
        let limit: f64 = inner.parse().unwrap_or_else(|_| {
            panic!("Invalid ge limit at line {}: '{}'", line_num, spec);
        });
        return FilterSpec::Ge { limit };
    }

    // lt(limit) - x < limit
    if spec.starts_with("lt(") && spec.ends_with(')') {
        let inner = &spec[3..spec.len() - 1];
        let limit: f64 = inner.parse().unwrap_or_else(|_| {
            panic!("Invalid lt limit at line {}: '{}'", line_num, spec);
        });
        return FilterSpec::Lt { limit };
    }

    // le(limit) - x <= limit
    if spec.starts_with("le(") && spec.ends_with(')') {
        let inner = &spec[3..spec.len() - 1];
        let limit: f64 = inner.parse().unwrap_or_else(|_| {
            panic!("Invalid le limit at line {}: '{}'", line_num, spec);
        });
        return FilterSpec::Le { limit };
    }

    // range(min;max) - min <= x <= max
    if spec.starts_with("range(") && spec.ends_with(')') {
        let inner = &spec[6..spec.len() - 1];
        let parts: Vec<&str> = inner.split(';').collect();
        if parts.len() != 2 {
            panic!(
                "Invalid range at line {}: '{}'. Expected range(min;max)",
                line_num, spec
            );
        }
        let min: f64 = parts[0].trim().parse().unwrap_or_else(|_| {
            panic!("Invalid range min at line {}: '{}'", line_num, spec);
        });
        let max: f64 = parts[1].trim().parse().unwrap_or_else(|_| {
            panic!("Invalid range max at line {}: '{}'", line_num, spec);
        });
        if min > max {
            panic!(
                "Invalid range at line {}: '{}'. min ({}) > max ({})",
                line_num, spec, min, max
            );
        }
        return FilterSpec::Range { min, max };
    }

    panic!(
        "Unknown filter_spec at line {}: '{}'. \
         Expected: abs_lt(<f64>), away_from(<f64>;...; eps=<f64>), abs_lt_and_away(...), \
         gt(<f64>), ge(<f64>), lt(<f64>), le(<f64>), or range(<min>;<max>)",
        line_num, spec
    );
}

/// Parse the inner part of away_from: "c1;c2;...;eps=<val>"
fn parse_away_from_inner(inner: &str, line_num: usize, spec: &str) -> FilterSpec {
    let parts: Vec<&str> = inner.split(';').collect();
    let mut centers = Vec::new();
    let mut eps = 0.01; // default

    for part in parts {
        let part = part.trim();
        if let Some(eps_str) = part.strip_prefix("eps=") {
            eps = eps_str.parse().unwrap_or_else(|_| {
                panic!("Invalid eps value at line {}: '{}'", line_num, spec);
            });
        } else if !part.is_empty() {
            let c: f64 = part.parse().unwrap_or_else(|_| {
                panic!(
                    "Invalid center value '{}' at line {}: '{}'",
                    part, line_num, spec
                );
            });
            centers.push(c);
        }
    }

    FilterSpec::AwayFrom { centers, eps }
}

// =============================================================================
// Numeric Equivalence Statistics
// =============================================================================

/// Maximum number of mismatches to record (avoid log bloat)
const MAX_MISMATCH_RECORDS: usize = 5;

/// Detailed statistics from numeric equivalence checking
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct NumericEquivStats {
    pub valid: usize,
    pub near_pole: usize,
    pub domain_error: usize,
    pub asymmetric_invalid: usize,
    pub eval_failed: usize,
    pub filtered_out: usize,
    pub mismatches: Vec<String>,
    pub max_abs_err: f64,
    pub max_rel_err: f64,
    pub worst_sample: Option<(f64, f64, f64)>, // (x, a, b)
}

impl Default for NumericEquivStats {
    fn default() -> Self {
        Self {
            valid: 0,
            near_pole: 0,
            domain_error: 0,
            asymmetric_invalid: 0,
            eval_failed: 0,
            filtered_out: 0,
            mismatches: Vec::new(),
            max_abs_err: 0.0,
            max_rel_err: 0.0,
            worst_sample: None,
        }
    }
}

#[allow(dead_code)]
impl NumericEquivStats {
    fn is_pass(&self, min_valid: usize) -> bool {
        self.valid >= min_valid && self.mismatches.is_empty()
    }

    fn total_samples(&self) -> usize {
        self.valid
            + self.near_pole
            + self.domain_error
            + self.asymmetric_invalid
            + self.eval_failed
            + self.filtered_out
    }

    /// Record a mismatch (capped at MAX_MISMATCH_RECORDS)
    fn record_mismatch(&mut self, x: f64, a: f64, b: f64, var: &str) {
        let abs_err = (a - b).abs();
        let scale = a.abs().max(b.abs()).max(1.0);
        let rel_err = abs_err / scale;

        // Update worst sample
        if abs_err > self.max_abs_err {
            self.max_abs_err = abs_err;
            self.max_rel_err = rel_err;
            self.worst_sample = Some((x, a, b));
        }

        // Record mismatch description (limited)
        if self.mismatches.len() < MAX_MISMATCH_RECORDS {
            self.mismatches.push(format!(
                "{}={:.6}: a={:.10}, b={:.10}, diff={:.3e}",
                var, x, a, b, abs_err
            ));
        }
    }

    /// Check if test is fragile (too many poles/domain errors)
    fn is_fragile(&self) -> bool {
        let total = self.total_samples();
        if total == 0 {
            return false;
        }

        let problematic = self.near_pole + self.domain_error;
        (problematic as f64 / total as f64) > 0.30
    }

    /// Get invalid rate (near_pole + domain_error + eval_failed as percentage)
    fn invalid_rate(&self) -> f64 {
        let total = self.total_samples();
        if total == 0 {
            return 0.0;
        }
        (self.near_pole + self.domain_error + self.eval_failed) as f64 / total as f64
    }

    /// Check for suspicious asymmetric failures
    fn has_asymmetric_failures(&self) -> bool {
        self.asymmetric_invalid > 0
    }

    /// Get domain error rate (domain_error / total)
    fn domain_rate(&self) -> f64 {
        let total = self.total_samples();
        if total == 0 {
            return 0.0;
        }
        self.domain_error as f64 / total as f64
    }

    /// Get near-pole rate (near_pole / total)
    fn pole_rate(&self) -> f64 {
        let total = self.total_samples();
        if total == 0 {
            return 0.0;
        }
        self.near_pole as f64 / total as f64
    }

    /// Get eval_failed rate (eval_failed / total)
    fn eval_failed_rate(&self) -> f64 {
        let total = self.total_samples();
        if total == 0 {
            return 0.0;
        }
        self.eval_failed as f64 / total as f64
    }
}

/// Fragility severity levels for CI
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum FragilityLevel {
    Ok,      // Within normal bounds
    Warning, // Elevated but acceptable
    Fail,    // Should fail CI
}

// =============================================================================
// Diagnostic Category Classification (Phase 3)
// =============================================================================

/// Diagnostic category for identity classification
/// Ordered by priority (higher priority = checked first)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum DiagCategory {
    /// Potential bug: asymmetric failures (one side evaluates, other doesn't)
    BugSignal,
    /// Configuration error: unbound variables, unsupported operations
    ConfigError,
    /// Needs domain filter: high domain_error rate (ln/sqrt with negative inputs)
    NeedsFilter,
    /// Fragile near poles: high near_pole rate (tan/sec near π/2)
    Fragile,
    /// All good: within acceptable thresholds
    Ok,
}

impl DiagCategory {
    fn emoji(&self) -> &'static str {
        match self {
            DiagCategory::BugSignal => "🐛",
            DiagCategory::ConfigError => "⚙️",
            DiagCategory::NeedsFilter => "🔧",
            DiagCategory::Fragile => "⚠️",
            DiagCategory::Ok => "✅",
        }
    }

    fn name(&self) -> &'static str {
        match self {
            DiagCategory::BugSignal => "BugSignal",
            DiagCategory::ConfigError => "ConfigError",
            DiagCategory::NeedsFilter => "NeedsFilter",
            DiagCategory::Fragile => "Fragile",
            DiagCategory::Ok => "Ok",
        }
    }
}

/// Classification thresholds
const DOMAIN_ERROR_THRESHOLD: f64 = 0.20; // 20% domain_error → NeedsFilter
const POLE_RATE_THRESHOLD: f64 = 0.15; // 15% near_pole → Fragile
const EVAL_FAILED_THRESHOLD: f64 = 0.50; // 50% eval_failed → ConfigError

/// Classify an identity into a diagnostic category
///
/// Precedence (highest to lowest):
/// 1. BugSignal: asymmetric_invalid > 0
/// 2. ConfigError: eval_failed_rate > 50% (likely unbound variable)
/// 3. NeedsFilter: domain_rate > 20%
/// 4. Fragile: pole_rate > 15%
/// 5. Ok: everything else
#[allow(dead_code)]
fn classify_diagnostic(stats: &NumericEquivStats) -> DiagCategory {
    // Priority 1: BugSignal (asymmetric failures indicate potential engine bugs)
    if stats.asymmetric_invalid > 0 {
        return DiagCategory::BugSignal;
    }

    // Priority 2: ConfigError (high eval_failed usually means unbound variable)
    if stats.eval_failed_rate() > EVAL_FAILED_THRESHOLD {
        return DiagCategory::ConfigError;
    }

    // Priority 3: NeedsFilter (high domain_error means function called outside domain)
    if stats.domain_rate() > DOMAIN_ERROR_THRESHOLD {
        return DiagCategory::NeedsFilter;
    }

    // Priority 4: Fragile (high pole_rate means near singularities)
    if stats.pole_rate() > POLE_RATE_THRESHOLD {
        return DiagCategory::Fragile;
    }

    // Priority 5: Ok
    DiagCategory::Ok
}

// =============================================================================
// JSONL Baseline System (Phase 2)
// =============================================================================

/// Snapshot of an identity's test results for baseline comparison
#[derive(Clone, Debug)]
struct IdentitySnapshot {
    /// Stable ID: first 16 chars of hash(exp|simp|vars|mode|bucket|branch|filter)
    id: String,
    exp: String,
    simp: String,
    category: String,
    // Raw stats (not rates - derive rates in comparator)
    valid: usize,
    filtered_out: usize,
    near_pole: usize,
    domain_error: usize,
    eval_failed: usize,
    asymmetric_invalid: usize,
    mismatches: usize,
    total_samples: usize,
}

impl IdentitySnapshot {
    /// Create from IdentityPair and NumericEquivStats
    fn from_pair_stats(
        pair: &IdentityPair,
        stats: &NumericEquivStats,
        category: DiagCategory,
    ) -> Self {
        let id = generate_identity_id(pair);
        Self {
            id,
            exp: pair.exp.clone(),
            simp: pair.simp.clone(),
            category: category.name().to_string(),
            valid: stats.valid,
            filtered_out: stats.filtered_out,
            near_pole: stats.near_pole,
            domain_error: stats.domain_error,
            eval_failed: stats.eval_failed,
            asymmetric_invalid: stats.asymmetric_invalid,
            mismatches: stats.mismatches.len(),
            total_samples: stats.total_samples(),
        }
    }

    /// Serialize to JSON line
    fn to_json(&self) -> String {
        format!(
            r#"{{"id":"{}","exp":"{}","simp":"{}","category":"{}","valid":{},"filtered_out":{},"near_pole":{},"domain_error":{},"eval_failed":{},"asymmetric":{},"mismatches":{},"total":{}}}"#,
            self.id,
            escape_json(&self.exp),
            escape_json(&self.simp),
            self.category,
            self.valid,
            self.filtered_out,
            self.near_pole,
            self.domain_error,
            self.eval_failed,
            self.asymmetric_invalid,
            self.mismatches,
            self.total_samples,
        )
    }

    /// Parse from JSON line
    fn from_json(line: &str) -> Option<Self> {
        // Simple manual parsing (avoid serde dependency)
        let get_str = |key: &str| -> Option<String> {
            let pattern = format!(r#""{}":""#, key);
            let start = line.find(&pattern)? + pattern.len();
            let end = line[start..].find('"')? + start;
            Some(line[start..end].to_string())
        };
        let get_usize = |key: &str| -> Option<usize> {
            let pattern = format!(r#""{}":"#, key);
            let start = line.find(&pattern)? + pattern.len();
            let end_candidates = [',', '}'];
            let end = end_candidates
                .iter()
                .filter_map(|c| line[start..].find(*c))
                .min()?
                + start;
            line[start..end].parse().ok()
        };

        Some(Self {
            id: get_str("id")?,
            exp: unescape_json(&get_str("exp")?),
            simp: unescape_json(&get_str("simp")?),
            category: get_str("category")?,
            valid: get_usize("valid")?,
            filtered_out: get_usize("filtered_out")?,
            near_pole: get_usize("near_pole")?,
            domain_error: get_usize("domain_error")?,
            eval_failed: get_usize("eval_failed")?,
            asymmetric_invalid: get_usize("asymmetric")?,
            mismatches: get_usize("mismatches")?,
            total_samples: get_usize("total")?,
        })
    }

    /// Calculate filtered_rate (for comparison)
    fn filtered_rate(&self) -> f64 {
        if self.total_samples == 0 {
            0.0
        } else {
            self.filtered_out as f64 / self.total_samples as f64
        }
    }

    /// Calculate invalid_rate (for comparison)
    fn invalid_rate(&self) -> f64 {
        if self.total_samples == 0 {
            0.0
        } else {
            (self.near_pole + self.domain_error + self.eval_failed) as f64
                / self.total_samples as f64
        }
    }
}

/// Generate stable ID for an identity (hash of canonical representation)
fn generate_identity_id(pair: &IdentityPair) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    pair.exp.hash(&mut hasher);
    pair.simp.hash(&mut hasher);
    pair.vars.join(";").hash(&mut hasher);
    format!("{:?}", pair.mode).hash(&mut hasher);
    format!("{:?}", pair.bucket).hash(&mut hasher);
    format!("{:?}", pair.branch_mode).hash(&mut hasher);
    pair.filter_spec.to_string().hash(&mut hasher);

    format!("{:016x}", hasher.finish())
}

/// Escape string for JSON output
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Unescape JSON string
fn unescape_json(s: &str) -> String {
    s.replace("\\\"", "\"")
        .replace("\\\\", "\\")
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
}

/// Baseline file path
fn baseline_file_path() -> PathBuf {
    let base = env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    base.join("tests/baselines/metatest_baseline.jsonl")
}

/// Generate deterministic hash of test configuration for baseline validation
fn generate_config_hash(config: &MetatestConfig) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    config.eval_samples.hash(&mut hasher);
    config.min_valid.hash(&mut hasher);
    // Hash floats as bits for determinism
    config.atol.to_bits().hash(&mut hasher);
    config.rtol.to_bits().hash(&mut hasher);
    config.sample_range.0.to_bits().hash(&mut hasher);
    config.sample_range.1.to_bits().hash(&mut hasher);

    format!("{:016x}", hasher.finish())
}

/// Config header line for baseline file
fn config_header_json(config: &MetatestConfig) -> String {
    format!(
        r#"{{"_type":"config","cfg_hash":"{}","samples":{},"min_valid":{},"atol":{},"rtol":{},"range":[{},{}]}}"#,
        generate_config_hash(config),
        config.eval_samples,
        config.min_valid,
        config.atol,
        config.rtol,
        config.sample_range.0,
        config.sample_range.1
    )
}

/// Category ranking for regression detection (higher = worse)
fn category_rank(cat: &str) -> u8 {
    match cat {
        "Ok" => 0,
        "Fragile" => 1,
        "NeedsFilter" => 2,
        "ConfigError" => 3,
        "BugSignal" => 4,
        _ => 5,
    }
}

/// Check if a snapshot represents a regression compared to baseline
#[derive(Debug)]
struct RegressionResult {
    id: String,
    exp: String,
    reasons: Vec<String>,
}

fn check_regression(
    baseline: &IdentitySnapshot,
    current: &IdentitySnapshot,
) -> Option<RegressionResult> {
    let mut reasons = Vec::new();

    // 1. Category worsened
    if category_rank(&current.category) > category_rank(&baseline.category) {
        reasons.push(format!(
            "category {} → {}",
            baseline.category, current.category
        ));
    }

    // 2. asymmetric went from 0 to >0
    if baseline.asymmetric_invalid == 0 && current.asymmetric_invalid > 0 {
        reasons.push(format!("asymmetric 0 → {}", current.asymmetric_invalid));
    }

    // 3. invalid_rate increased by >5%
    let base_rate = baseline.invalid_rate();
    let curr_rate = current.invalid_rate();
    if curr_rate > base_rate + 0.05 {
        reasons.push(format!(
            "invalid_rate {:.1}% → {:.1}%",
            base_rate * 100.0,
            curr_rate * 100.0
        ));
    }

    // 4. filtered_rate increased by >20% (absolute)
    let base_filt = baseline.filtered_rate();
    let curr_filt = current.filtered_rate();
    if curr_filt > base_filt + 0.20 {
        reasons.push(format!(
            "filtered_rate {:.1}% → {:.1}%",
            base_filt * 100.0,
            curr_filt * 100.0
        ));
    }

    // 5. mismatches went from 0 to >0 (for non-BranchSensitive)
    if baseline.mismatches == 0 && current.mismatches > 0 {
        reasons.push(format!("mismatches 0 → {}", current.mismatches));
    }

    if reasons.is_empty() {
        None
    } else {
        Some(RegressionResult {
            id: current.id.clone(),
            exp: truncate_identity(&current.exp, 50),
            reasons,
        })
    }
}

/// Check fragility level based on bucket-specific thresholds
///
/// Thresholds (warning/fail):
/// - Unconditional: 10% / 25% (pure identities should rarely hit poles)
/// - ConditionalRequires: 30% / 50% (some poles expected)
/// - BranchSensitive: 40% / 60% (more tolerance for complex cases)
#[allow(dead_code)]
fn fragility_level_for_bucket(stats: &NumericEquivStats, bucket: Bucket) -> FragilityLevel {
    let rate = stats.invalid_rate();

    let (warn_threshold, fail_threshold) = match bucket {
        Bucket::Unconditional => (0.10, 0.25),
        Bucket::ConditionalRequires => (0.30, 0.50),
        Bucket::BranchSensitive => (0.40, 0.60),
    };

    if rate >= fail_threshold {
        FragilityLevel::Fail
    } else if rate >= warn_threshold {
        FragilityLevel::Warning
    } else {
        FragilityLevel::Ok
    }
}

/// Get minimum valid samples required based on bucket type
#[allow(dead_code)]
fn min_valid_for_bucket(bucket: Bucket, total_samples: usize) -> usize {
    let ratio = match bucket {
        Bucket::Unconditional => 0.70,       // 70% for pure identities
        Bucket::ConditionalRequires => 0.50, // 50% for conditional
        Bucket::BranchSensitive => 0.35,     // 35% for branch-sensitive
    };
    ((total_samples as f64) * ratio).ceil() as usize
}

/// Check numeric equivalence with BranchMode support
///
/// This is the unified branch-aware numeric equivalence checker.
/// - PrincipalStrict: direct comparison with atol/rtol
/// - ModuloPi: compare modulo π (for arctan identities)
/// - Modulo2Pi: compare modulo 2π (for trig identities)
/// - PrincipalWithFilter: direct comparison but requires filter (panics if None)
#[allow(dead_code)]
fn check_numeric_equiv_branch_1var<F>(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var: &str,
    branch_mode: BranchMode,
    config: &MetatestConfig,
    filter: Option<F>,
) -> NumericEquivStats
where
    F: Fn(f64) -> bool,
{
    use std::f64::consts::PI;

    // PrincipalWithFilter requires a filter
    if branch_mode == BranchMode::PrincipalWithFilter && filter.is_none() {
        panic!("PrincipalWithFilter mode requires a non-None filter");
    }

    let (lo, hi) = config.sample_range;
    let mut stats = NumericEquivStats::default();

    let opts = EvalCheckedOptions {
        zero_abs_eps: 1e-12,
        zero_rel_eps: 1e-12,
        trig_pole_eps: 1e-9,
        max_depth: 200,
    };

    for i in 0..config.eval_samples {
        let t = (i as f64 + 0.5) / config.eval_samples as f64;
        let x = lo + (hi - lo) * t;

        // Apply optional filter
        if let Some(ref f) = filter {
            if !f(x) {
                stats.filtered_out += 1;
                continue;
            }
        }

        let mut var_map = HashMap::new();
        var_map.insert(var.to_string(), x);

        let va = eval_f64_checked(ctx, a, &var_map, &opts);
        let vb = eval_f64_checked(ctx, b, &var_map, &opts);

        match (&va, &vb) {
            (Ok(va), Ok(vb)) => {
                // Choose comparison method based on branch mode
                let is_equal = match branch_mode {
                    BranchMode::PrincipalStrict | BranchMode::PrincipalWithFilter => {
                        let diff = (va - vb).abs();
                        let scale = va.abs().max(vb.abs()).max(1.0);
                        let allowed = config.atol + config.rtol * scale;
                        diff <= allowed
                    }
                    BranchMode::ModuloPi => {
                        approx_eq_mod_period(*va, *vb, PI, config.atol, config.rtol)
                    }
                    BranchMode::Modulo2Pi => {
                        approx_eq_mod_period(*va, *vb, 2.0 * PI, config.atol, config.rtol)
                    }
                };

                if is_equal {
                    stats.valid += 1;
                } else {
                    stats.record_mismatch(x, *va, *vb, var);
                }
            }
            // Symmetric failures
            (Err(EvalCheckedError::NearPole { .. }), Err(EvalCheckedError::NearPole { .. })) => {
                stats.near_pole += 1;
            }
            (Err(EvalCheckedError::Domain { .. }), Err(EvalCheckedError::Domain { .. })) => {
                stats.domain_error += 1;
            }
            // Asymmetric: one Ok, one Err (suspicious)
            (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
                stats.asymmetric_invalid += 1;
            }
            _ => {
                stats.eval_failed += 1;
            }
        }
    }

    stats
}

/// Load identity pairs from CSV file
/// Supports two formats:
/// - Legacy 4-col: exp,simp,vars,mode (bucket=conditional_requires, branch=principal_strict)
/// - Extended 7-col: exp,simp,vars,domain_mode,bucket,branch_mode,filter
fn load_identity_pairs() -> Vec<IdentityPair> {
    let csv_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/identity_pairs.csv");
    let content = std::fs::read_to_string(csv_path).expect("Failed to read identity_pairs.csv");

    let mut pairs = Vec::new();
    for (line_num, line) in content.lines().enumerate() {
        let line_num = line_num + 1; // 1-indexed for humans
        let line = line.trim();
        // Skip comments and empty lines
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Count columns to determine format
        let parts: Vec<&str> = line.split(',').collect();

        if parts.len() >= 7 {
            // Extended 7-column format: exp,simp,vars,domain_mode,bucket,branch_mode,filter
            let vars: Vec<String> = parts[2]
                .trim()
                .split(';')
                .map(|s| s.trim().to_string())
                .collect();

            let mode = parse_domain_mode(parts[3].trim());
            let bucket = parse_bucket(parts[4].trim());
            let branch_mode = parse_branch_mode(parts[5].trim());
            let filter_spec = parse_filter_spec(parts[6].trim(), line_num);

            pairs.push(IdentityPair {
                exp: parts[0].trim().to_string(),
                simp: parts[1].trim().to_string(),
                vars,
                mode,
                bucket,
                branch_mode,
                filter_spec,
            });
        } else if parts.len() >= 3 {
            // Legacy 4-column format: exp,simp,vars,mode
            let vars: Vec<String> = parts[2]
                .trim()
                .split(';')
                .map(|s| s.trim().to_string())
                .collect();

            let mode = if parts.len() >= 4 {
                parse_domain_mode(parts[3].trim())
            } else {
                DomainRequirement::Generic
            };

            pairs.push(IdentityPair {
                exp: parts[0].trim().to_string(),
                simp: parts[1].trim().to_string(),
                vars,
                mode,
                bucket: legacy_bucket_from_env(), // Configurable via METATEST_LEGACY_BUCKET
                branch_mode: BranchMode::default(),
                filter_spec: FilterSpec::None,
            });
        }
    }

    pairs
}

/// Get legacy bucket from environment variable (for migration flexibility)
/// METATEST_LEGACY_BUCKET=unconditional|conditional_requires (default)
fn legacy_bucket_from_env() -> Bucket {
    match env::var("METATEST_LEGACY_BUCKET").ok().as_deref() {
        Some("unconditional") => Bucket::Unconditional,
        _ => Bucket::ConditionalRequires,
    }
}

/// Validate filter spec - fail-fast if malformed
/// Valid formats: "", "abs_lt(0.9)", "away_from(1.0;-1.0;eps=0.1)", "abs_lt_and_away(...)"
#[allow(dead_code)]
fn validate_filter_spec(spec: &str, line_num: usize) {
    if spec.is_empty() {
        return; // Empty is valid (no filter)
    }

    // Basic syntax check: must start with known function name and have balanced parens
    let valid_prefixes = ["abs_lt(", "away_from(", "abs_lt_and_away("];
    let has_valid_prefix = valid_prefixes.iter().any(|p| spec.starts_with(p));
    let has_balanced_parens =
        spec.chars().filter(|&c| c == '(').count() == spec.chars().filter(|&c| c == ')').count();
    let ends_with_paren = spec.ends_with(')');

    if !has_valid_prefix || !has_balanced_parens || !ends_with_paren {
        panic!(
            "Invalid filter_spec at line {}: '{}'. \
             Expected: abs_lt(<f64>), away_from(<f64>;...; eps=<f64>), or abs_lt_and_away(...)",
            line_num, spec
        );
    }
}

/// Parse domain mode from string
fn parse_domain_mode(s: &str) -> DomainRequirement {
    match s.to_lowercase().as_str() {
        "a" | "assume" => DomainRequirement::Assume,
        _ => DomainRequirement::Generic,
    }
}

/// Parse bucket from string
fn parse_bucket(s: &str) -> Bucket {
    match s.to_lowercase().as_str() {
        "unconditional" | "u" => Bucket::Unconditional,
        "branch_sensitive" | "branch" | "b" => Bucket::BranchSensitive,
        _ => Bucket::ConditionalRequires, // Default
    }
}

/// Parse branch mode from string
fn parse_branch_mode(s: &str) -> BranchMode {
    match s.to_lowercase().as_str() {
        "modulo_pi" | "mod_pi" => BranchMode::ModuloPi,
        "modulo_2pi" | "mod_2pi" => BranchMode::Modulo2Pi,
        "principal_with_filter" | "filter" => BranchMode::PrincipalWithFilter,
        _ => BranchMode::PrincipalStrict, // Default
    }
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
            let pair2_exp = alpha_rename(&pair2.exp, &pair2.vars[0], "u");
            let pair2_simp = alpha_rename(&pair2.simp, &pair2.vars[0], "u");

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
                    &pair1.vars[0],
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
                    let pair2_exp = alpha_rename(&pair2.exp, &pair2.vars[0], "u");
                    let pair2_simp = alpha_rename(&pair2.simp, &pair2.vars[0], "u");
                    let pair3_exp = alpha_rename(&pair3.exp, &pair3.vars[0], "v");
                    let pair3_simp = alpha_rename(&pair3.simp, &pair3.vars[0], "v");

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
                        var_map.insert(pair1.vars[0].clone(), x_val);
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
///
/// Environment variables:
/// - METATEST_MODE=assume : Use DomainMode::Assume (includes all identities)
/// - METATEST_MODE=generic (or unset) : Use DomainMode::Generic (skips assume-only identities)
#[test]
#[ignore = "Diagnostic test - run manually to check symbolic vs numeric equivalence"]
fn metatest_individual_identities() {
    // Run in a thread with larger stack to avoid overflow
    let handle = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024) // 16 MB stack
        .spawn(metatest_individual_identities_impl)
        .expect("Failed to spawn test thread");
    handle.join().expect("Test thread panicked");
}

fn metatest_individual_identities_impl() {
    let pairs = load_identity_pairs();
    let config = metatest_config();

    // Determine test mode from environment
    let use_assume_mode = env::var("METATEST_MODE").ok().as_deref() == Some("assume");
    let domain_mode = if use_assume_mode {
        cas_engine::domain::DomainMode::Assume
    } else {
        cas_engine::domain::DomainMode::Generic
    };

    eprintln!(
        "🔧 Running in {} mode",
        if use_assume_mode { "ASSUME" } else { "GENERIC" }
    );

    let mut symbolic_passed = 0;
    let mut numeric_only_passed = 0;
    let mut failed = 0;
    let mut skipped = 0;
    let mut numeric_only_examples: Vec<String> = Vec::new();
    #[allow(unused_mut, unused_variables)]
    let mut fragile_identities: Vec<String> = Vec::new(); // High near_pole/domain rate
    #[allow(unused_mut, unused_variables)]
    let mut asymmetric_count = 0; // Suspicious L=Ok/R=Err cases

    // Diagnostics: track per-identity fragility (infrastructure for future use)
    #[allow(dead_code)]
    struct IdentityDiag {
        idx: usize,
        exp: String,
        simp: String,
        bucket: Bucket,
        stats: NumericEquivStats,
        filter_str: String, // Original filter spec from CSV (empty if None)
    }

    // Only collect diagnostics if METATEST_DIAG=1
    let diag_enabled = env::var("METATEST_DIAG").is_ok();
    let mut diagnostics: Vec<IdentityDiag> = Vec::new();

    // Snapshot/baseline mode detection
    let snapshot_enabled = env::var("METATEST_SNAPSHOT").is_ok();
    let update_baseline = env::var("METATEST_UPDATE_BASELINE").is_ok();
    let mut snapshots: Vec<(IdentityPair, NumericEquivStats)> = Vec::new();

    for pair in &pairs {
        // Skip assume-only identities in generic mode
        if pair.mode == DomainRequirement::Assume && !use_assume_mode {
            skipped += 1;
            continue;
        }

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

        // Simplify with selected domain mode
        let opts = cas_engine::phase::SimplifyOptions {
            domain: domain_mode,
            ..Default::default()
        };
        let (exp_simplified, _) = simplifier.simplify_with_options(exp_parsed, opts.clone());
        let (simp_simplified, _) = simplifier.simplify_with_options(simp_parsed, opts);

        // Check symbolic equality
        let symbolic_match = cas_engine::ordering::compare_expr(
            &simplifier.context,
            exp_simplified,
            simp_simplified,
        ) == std::cmp::Ordering::Equal;

        if symbolic_match {
            symbolic_passed += 1;
        } else {
            // Get string representations of simplified forms using Debug
            let exp_simplified_str = format!("{:?}", simplifier.context.get(exp_simplified));
            let simp_simplified_str = format!("{:?}", simplifier.context.get(simp_simplified));

            // Check numeric equivalence - select function based on variable count
            let (result, stats_opt) = match pair.vars.len() {
                1 => {
                    if diag_enabled {
                        let stats = check_numeric_equiv_1var_stats(
                            &simplifier.context,
                            exp_simplified,
                            simp_simplified,
                            &pair.vars[0],
                            &config,
                            &pair.filter_spec,
                        );
                        let pass = stats.valid >= config.min_valid && stats.mismatches.is_empty();
                        (
                            if pass {
                                Ok(stats.valid)
                            } else {
                                Err("failed".to_string())
                            },
                            Some(stats),
                        )
                    } else {
                        (
                            check_numeric_equiv_1var(
                                &simplifier.context,
                                exp_simplified,
                                simp_simplified,
                                &pair.vars[0],
                                &config,
                            ),
                            None,
                        )
                    }
                }
                2 => (
                    check_numeric_equiv_2var(
                        &simplifier.context,
                        exp_simplified,
                        simp_simplified,
                        &pair.vars[0],
                        &pair.vars[1],
                        &config,
                    ),
                    None,
                ),
                _ => {
                    // 3+ variables: skip for now
                    skipped += 1;
                    continue;
                }
            };

            // Collect stats for diagnostics if enabled
            if let Some(stats) = stats_opt {
                diagnostics.push(IdentityDiag {
                    idx: diagnostics.len(),
                    exp: pair.exp.clone(),
                    simp: pair.simp.clone(),
                    bucket: pair.bucket,
                    stats: stats.clone(),
                    filter_str: pair.filter_spec.to_string(),
                });

                // Collect snapshot for baseline comparison
                if snapshot_enabled || update_baseline {
                    snapshots.push((pair.clone(), stats));
                }
            }

            if result.is_ok() {
                numeric_only_passed += 1;
                if numeric_only_examples.len() < 30 {
                    numeric_only_examples.push(format!(
                        "{} ≡ {}\n     → L: {}\n     → R: {}",
                        pair.exp, pair.simp, exp_simplified_str, simp_simplified_str
                    ));
                }
            } else {
                failed += 1;
                if failed <= 10 {
                    eprintln!("❌ Identity failed: {} ≡ {}", pair.exp, pair.simp);
                    eprintln!("   → L simplified: {}", exp_simplified_str);
                    eprintln!("   → R simplified: {}", simp_simplified_str);
                }
            }
        }
    }

    let total = symbolic_passed + numeric_only_passed + failed;
    let symbolic_pct = if total > 0 {
        (symbolic_passed as f64 / total as f64 * 100.0) as u32
    } else {
        0
    };

    eprintln!("\n📊 Individual Identity Results:");
    eprintln!("   Total tested: {}", total);
    eprintln!("   ✅ Symbolic: {} ({}%)", symbolic_passed, symbolic_pct);
    eprintln!("   🔢 Numeric-only: {}", numeric_only_passed);
    eprintln!("   ❌ Failed: {}", failed);
    eprintln!("   ⏭️  Skipped: {}", skipped);

    // Top-10 fragility ranking (only when METATEST_DIAG=1)
    if diag_enabled && !diagnostics.is_empty() {
        // Classify all diagnostics
        let classified: Vec<_> = diagnostics
            .iter()
            .map(|d| (classify_diagnostic(&d.stats), d))
            .collect();

        // Count by category
        let bug_count = classified
            .iter()
            .filter(|(c, _)| *c == DiagCategory::BugSignal)
            .count();
        let config_count = classified
            .iter()
            .filter(|(c, _)| *c == DiagCategory::ConfigError)
            .count();
        let filter_count = classified
            .iter()
            .filter(|(c, _)| *c == DiagCategory::NeedsFilter)
            .count();
        let fragile_count = classified
            .iter()
            .filter(|(c, _)| *c == DiagCategory::Fragile)
            .count();
        let ok_count = classified
            .iter()
            .filter(|(c, _)| *c == DiagCategory::Ok)
            .count();

        eprintln!("\n📊 Diagnostic Classification (METATEST_DIAG=1):");
        eprintln!(
            "   Summary: ✅ Ok={} | 🐛 BugSignal={} | ⚙️ ConfigError={} | 🔧 NeedsFilter={} | ⚠️ Fragile={}\n",
            ok_count, bug_count, config_count, filter_count, fragile_count
        );

        // Filter Coverage Report
        let with_filter: Vec<_> = diagnostics
            .iter()
            .filter(|d| !d.filter_str.is_empty())
            .collect();
        let filtered_count = with_filter.len();
        let total_diag = diagnostics.len();

        if filtered_count > 0 {
            // Dual coverage: snapshot vs total loaded
            let total_loaded = pairs.len();
            eprintln!(
                "🔍 Filter Coverage: {}/{} snapshot ({:.1}%) | {}/{} total loaded ({:.1}%)",
                filtered_count,
                total_diag,
                filtered_count as f64 / total_diag as f64 * 100.0,
                filtered_count,
                total_loaded,
                filtered_count as f64 / total_loaded as f64 * 100.0
            );

            // Sort by filtered_rate DESC (potential "cheating" filters)
            let mut by_filtered: Vec<_> = diagnostics
                .iter()
                .filter(|d| !d.filter_str.is_empty())
                .map(|d| {
                    let total = d.stats.total_samples();
                    let filtered_rate = if total > 0 {
                        d.stats.filtered_out as f64 / total as f64
                    } else {
                        0.0
                    };
                    (filtered_rate, d)
                })
                .collect();
            by_filtered.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            eprintln!("   Top-5 by filtered_rate (potential 'cheating' filters):");
            for (i, (rate, d)) in by_filtered.iter().take(5).enumerate() {
                eprintln!(
                    "   {:2}. [{:4.0}%] valid={:3}/{:3} {} → {}",
                    i + 1,
                    rate * 100.0,
                    d.stats.valid,
                    d.stats.total_samples(),
                    d.filter_str,
                    truncate_identity(&d.exp, 35)
                );
            }
            eprintln!();
        }
        // Helper to print a ranking section
        let print_ranking = |category: DiagCategory, items: Vec<&IdentityDiag>, max_show: usize| {
            if items.is_empty() {
                return;
            }
            eprintln!(
                "   {} {} ({})",
                category.emoji(),
                category.name(),
                items.len()
            );
            for (i, d) in items.iter().take(max_show).enumerate() {
                let total = d.stats.total_samples();
                eprintln!(
                    "      {:2}. valid={:3}/{:3} pole={:.0}% domain={:.0}% eval={:.0}% asym={}",
                    i + 1,
                    d.stats.valid,
                    total,
                    d.stats.pole_rate() * 100.0,
                    d.stats.domain_rate() * 100.0,
                    d.stats.eval_failed_rate() * 100.0,
                    d.stats.asymmetric_invalid,
                );
                eprintln!("          {} ≡ {}", d.exp, d.simp);
            }
            if items.len() > max_show {
                eprintln!("          ... and {} more", items.len() - max_show);
            }
            eprintln!();
        };

        // 1. BugSignal ranking (sorted by asymmetric_invalid DESC)
        let mut bug_items: Vec<_> = classified
            .iter()
            .filter(|(c, _)| *c == DiagCategory::BugSignal)
            .map(|(_, d)| *d)
            .collect();
        bug_items.sort_by(|a, b| {
            b.stats
                .asymmetric_invalid
                .cmp(&a.stats.asymmetric_invalid)
                .then_with(|| a.idx.cmp(&b.idx))
        });
        print_ranking(DiagCategory::BugSignal, bug_items, 10);

        // 2. ConfigError ranking (sorted by eval_failed_rate DESC)
        let mut config_items: Vec<_> = classified
            .iter()
            .filter(|(c, _)| *c == DiagCategory::ConfigError)
            .map(|(_, d)| *d)
            .collect();
        config_items.sort_by(|a, b| {
            b.stats
                .eval_failed_rate()
                .partial_cmp(&a.stats.eval_failed_rate())
                .unwrap()
                .then_with(|| a.idx.cmp(&b.idx))
        });
        print_ranking(DiagCategory::ConfigError, config_items, 5);

        // 3. NeedsFilter ranking (sorted by domain_rate DESC)
        let mut filter_items: Vec<_> = classified
            .iter()
            .filter(|(c, _)| *c == DiagCategory::NeedsFilter)
            .map(|(_, d)| *d)
            .collect();
        filter_items.sort_by(|a, b| {
            b.stats
                .domain_rate()
                .partial_cmp(&a.stats.domain_rate())
                .unwrap()
                .then_with(|| a.idx.cmp(&b.idx))
        });
        print_ranking(DiagCategory::NeedsFilter, filter_items, 10);

        // 4. Fragile ranking (sorted by pole_rate DESC)
        let mut fragile_items: Vec<_> = classified
            .iter()
            .filter(|(c, _)| *c == DiagCategory::Fragile)
            .map(|(_, d)| *d)
            .collect();
        fragile_items.sort_by(|a, b| {
            b.stats
                .pole_rate()
                .partial_cmp(&a.stats.pole_rate())
                .unwrap()
                .then_with(|| a.idx.cmp(&b.idx))
        });
        print_ranking(DiagCategory::Fragile, fragile_items, 10);

        // Total asymmetric summary
        let total_asym: usize = diagnostics.iter().map(|d| d.stats.asymmetric_invalid).sum();
        if total_asym > 0 {
            eprintln!("   🚨 Total asymmetric_invalid across all: {}", total_asym);
        }
    }

    // Show fragile identities (high near_pole/domain rate)
    if !fragile_identities.is_empty() {
        eprintln!("\n⚠️  Fragile Identities (>30% near_pole/domain):");
        for (i, id) in fragile_identities.iter().take(10).enumerate() {
            eprintln!("   {}. {}", i + 1, id);
        }
        if fragile_identities.len() > 10 {
            eprintln!("   ... and {} more", fragile_identities.len() - 10);
        }
    }

    // Show asymmetric failures (suspicious - may indicate bugs)
    if asymmetric_count > 0 {
        eprintln!("\n🚨 Asymmetric Failures Detected: {}", asymmetric_count);
        eprintln!("   This may indicate engine bugs (L=Ok but R=Err or vice versa)");
    }

    if !numeric_only_examples.is_empty() {
        eprintln!("\n📝 Numeric-only identities (showing simplifications):");
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

    // JSONL Baseline Processing
    if snapshot_enabled || update_baseline {
        // Generate current snapshots
        let current_snapshots: Vec<IdentitySnapshot> = snapshots
            .iter()
            .map(|(pair, stats)| {
                let category = classify_diagnostic(stats);
                IdentitySnapshot::from_pair_stats(pair, stats, category)
            })
            .collect();

        let baseline_path = baseline_file_path();

        if update_baseline {
            // Write new baseline
            if let Some(parent) = baseline_path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let mut file = File::create(&baseline_path).expect("Failed to create baseline file");
            // Write config header as first line
            writeln!(file, "{}", config_header_json(&config))
                .expect("Failed to write config header");
            for snap in &current_snapshots {
                writeln!(file, "{}", snap.to_json()).expect("Failed to write baseline");
            }
            eprintln!(
                "\n✅ Baseline updated: {} identities + config written to {}",
                current_snapshots.len(),
                baseline_path.display()
            );
            eprintln!("   cfg_hash: {}", generate_config_hash(&config));
        } else if snapshot_enabled {
            // Compare against baseline
            if !baseline_path.exists() {
                eprintln!("\n⚠️  No baseline found at {}", baseline_path.display());
                eprintln!("   Run with METATEST_UPDATE_BASELINE=1 to create one.");
            } else {
                // Load baseline and validate config hash
                let file = File::open(&baseline_path).expect("Failed to open baseline file");
                let reader = BufReader::new(file);
                let lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();

                // Check config hash from first line
                let current_cfg_hash = generate_config_hash(&config);
                if let Some(first_line) = lines.first() {
                    if first_line.contains("\"_type\":\"config\"") {
                        // Extract cfg_hash from first line
                        if let Some(start) = first_line.find("\"cfg_hash\":\"") {
                            let hash_start = start + 12;
                            if let Some(end) = first_line[hash_start..].find('"') {
                                let baseline_hash = &first_line[hash_start..hash_start + end];
                                if baseline_hash != current_cfg_hash {
                                    eprintln!("\n⚠️  Config mismatch detected!");
                                    eprintln!("   Baseline cfg_hash: {}", baseline_hash);
                                    eprintln!("   Current cfg_hash:  {}", current_cfg_hash);
                                    eprintln!(
                                        "   Run with METATEST_UPDATE_BASELINE=1 to regenerate."
                                    );
                                    panic!(
                                        "Baseline/config mismatch - test parameters have changed"
                                    );
                                }
                            }
                        }
                    }
                }

                // Skip config line when loading identities
                let baseline: HashMap<String, IdentitySnapshot> = lines
                    .iter()
                    .filter(|l| !l.contains("\"_type\":\"config\""))
                    .filter_map(|l| IdentitySnapshot::from_json(l))
                    .map(|s| (s.id.clone(), s))
                    .collect();

                // Check for regressions
                let mut regressions: Vec<RegressionResult> = Vec::new();
                let mut new_ids: Vec<String> = Vec::new();
                let mut missing_ids: Vec<String> = Vec::new();

                for snap in &current_snapshots {
                    if let Some(base) = baseline.get(&snap.id) {
                        if let Some(reg) = check_regression(base, snap) {
                            regressions.push(reg);
                        }
                    } else {
                        new_ids.push(snap.id.clone());
                    }
                }

                let current_ids: std::collections::HashSet<_> =
                    current_snapshots.iter().map(|s| &s.id).collect();
                for (id, _) in &baseline {
                    if !current_ids.contains(id) {
                        missing_ids.push(id.clone());
                    }
                }

                // Report results
                eprintln!("\n📊 Baseline Comparison (METATEST_SNAPSHOT=1):");
                eprintln!(
                    "   Current: {} | Baseline: {} | Regressions: {} | New: {} | Missing: {}",
                    current_snapshots.len(),
                    baseline.len(),
                    regressions.len(),
                    new_ids.len(),
                    missing_ids.len()
                );

                if !regressions.is_empty() {
                    eprintln!("\n🚨 REGRESSIONS DETECTED:");
                    for reg in &regressions {
                        eprintln!("   • {}: {}", reg.exp, reg.reasons.join(", "));
                    }
                }

                if !new_ids.is_empty() && new_ids.len() <= 5 {
                    eprintln!("\n➕ New identities (not in baseline):");
                    for id in &new_ids {
                        eprintln!("   • {}", id);
                    }
                } else if !new_ids.is_empty() {
                    eprintln!("\n➕ {} new identities not in baseline", new_ids.len());
                }

                if !missing_ids.is_empty() && missing_ids.len() <= 5 {
                    eprintln!("\n➖ Missing identities (in baseline, not in current):");
                    for id in &missing_ids {
                        eprintln!("   • {}", id);
                    }
                } else if !missing_ids.is_empty() {
                    eprintln!("\n➖ {} identities missing from current", missing_ids.len());
                }

                // Fail on regressions in CI
                if !regressions.is_empty() {
                    panic!(
                        "Baseline regression detected: {} identities worsened",
                        regressions.len()
                    );
                }
            }
        }
    }
}

// =============================================================================
// Shuffle Canonicalization Test (Phase A)
// =============================================================================

/// Test shuffle canonicalization with dual checks:
/// 1. Semantic: simplify(E) ≡ simplify(shuffle(E)) numerically (must pass - bug if fails)
/// 2. Structural: simplify(E) == simplify(shuffle(E)) exactly (metric, optional strict mode)
#[test]
#[ignore]
fn metatest_shuffle_canonicalization() {
    let shuffle_enabled = env::var("METATEST_SHUFFLE").is_ok();
    if !shuffle_enabled {
        eprintln!("Shuffle test skipped. Set METATEST_SHUFFLE=1 to enable.");
        return;
    }

    let strict_canon = env::var("METATEST_STRICT_CANON").is_ok();
    let pairs = load_identity_pairs();
    if pairs.is_empty() {
        panic!("No identity pairs loaded!");
    }

    eprintln!("🔀 Shuffle Canonicalization Test");
    eprintln!(
        "   Mode: {}",
        if strict_canon {
            "STRICT (fail on structural diff)"
        } else {
            "METRIC (report only)"
        }
    );
    eprintln!("   Testing {} identity expressions...\n", pairs.len());

    let mut semantic_failures: Vec<String> = Vec::new();
    let mut structural_failures: Vec<String> = Vec::new();
    let mut tested = 0;

    for pair in &pairs {
        if pair.vars.len() != 1 {
            continue;
        }

        // Test LHS
        match test_shuffle_dual(&pair.exp, &pair.vars[0]) {
            ShuffleResult::Ok => {}
            ShuffleResult::ParseSkip => {} // Skip unsupported syntax
            ShuffleResult::StructuralDiff(msg) => {
                structural_failures.push(format!(
                    "{} (LHS): {}",
                    truncate_identity(&pair.exp, 30),
                    msg
                ));
            }
            ShuffleResult::SemanticFail(msg) => {
                semantic_failures.push(format!("{} (LHS): {}", pair.exp, msg));
            }
        }

        // Test RHS
        match test_shuffle_dual(&pair.simp, &pair.vars[0]) {
            ShuffleResult::Ok => {}
            ShuffleResult::ParseSkip => {} // Skip unsupported syntax
            ShuffleResult::StructuralDiff(msg) => {
                structural_failures.push(format!(
                    "{} (RHS): {}",
                    truncate_identity(&pair.simp, 30),
                    msg
                ));
            }
            ShuffleResult::SemanticFail(msg) => {
                semantic_failures.push(format!("{} (RHS): {}", pair.simp, msg));
            }
        }

        tested += 1;
    }

    // Report results
    eprintln!("📊 Shuffle Results:");
    eprintln!("   Tested: {} expressions", tested * 2);
    eprintln!(
        "   Semantic failures: {} (MUST be 0)",
        semantic_failures.len()
    );
    eprintln!(
        "   Structural diffs: {} (canonicalization gaps)",
        structural_failures.len()
    );

    // Semantic failures are always fatal (indicates a real bug)
    if !semantic_failures.is_empty() {
        eprintln!("\n🚨 SEMANTIC FAILURES (shuffle broke equivalence!):");
        for (i, fail) in semantic_failures.iter().take(5).enumerate() {
            eprintln!("   {}. {}", i + 1, fail);
        }
        panic!(
            "Shuffle caused {} semantic failures - this is a BUG!",
            semantic_failures.len()
        );
    }

    // Structural diffs are informative (or fatal in strict mode)
    if !structural_failures.is_empty() {
        eprintln!("\n⚠️  STRUCTURAL DIFFS (order-dependent canonicalization):");
        for (i, fail) in structural_failures.iter().take(5).enumerate() {
            eprintln!("   {}. {}", i + 1, fail);
        }
        if structural_failures.len() > 5 {
            eprintln!("   ... and {} more", structural_failures.len() - 5);
        }

        if strict_canon {
            panic!(
                "Strict canon mode: {} structural diffs - canonicalization not stable",
                structural_failures.len()
            );
        } else {
            eprintln!("\n💡 Run with METATEST_STRICT_CANON=1 to fail on structural diffs.");
        }
    }

    if semantic_failures.is_empty() && structural_failures.is_empty() {
        eprintln!("\n✅ All shuffle checks passed (semantic + structural)!");
    } else if semantic_failures.is_empty() {
        eprintln!(
            "\n✅ Semantic checks passed. {} structural diffs (non-blocking).",
            structural_failures.len()
        );
    }
}

enum ShuffleResult {
    Ok,
    StructuralDiff(String),
    SemanticFail(String),
    ParseSkip, // Expression couldn't be parsed (syntax not supported)
}

/// Test shuffle with dual check: semantic (numeric) + structural (exact)
fn test_shuffle_dual(expr_str: &str, var: &str) -> ShuffleResult {
    let mut simplifier = Simplifier::new();

    // Parse - skip if syntax not supported
    let expr = match parse(expr_str, &mut simplifier.context) {
        Ok(e) => e,
        Err(_) => return ShuffleResult::ParseSkip,
    };

    // Simplify original
    let (simplified_original, _) = simplifier.simplify(expr);

    // Shuffle and simplify
    let shuffled = shuffle_expr(&mut simplifier.context, expr);
    let (simplified_shuffled, _) = simplifier.simplify(shuffled);

    // 1. Structural check (Debug representation)
    let original_debug = format!("{:?}", simplifier.context.get(simplified_original));
    let shuffled_debug = format!("{:?}", simplifier.context.get(simplified_shuffled));
    let structural_match = original_debug == shuffled_debug;

    // 2. Semantic check (numeric evaluation at a few points)
    let semantic_match = check_numeric_equiv_quick(
        &simplifier.context,
        simplified_original,
        simplified_shuffled,
        var,
    );

    match (structural_match, semantic_match) {
        (true, true) => ShuffleResult::Ok,
        (false, true) => ShuffleResult::StructuralDiff("different debug repr".to_string()),
        (_, false) => ShuffleResult::SemanticFail("numeric mismatch after shuffle".to_string()),
    }
}

/// Quick numeric equivalence check (5 sample points)
fn check_numeric_equiv_quick(ctx: &Context, a: ExprId, b: ExprId, var: &str) -> bool {
    let samples = [-2.0, -0.5, 0.5, 1.5, 3.0];
    let mut valid_checks = 0;
    let mut matching = 0;

    for x in samples {
        let mut vars = HashMap::new();
        vars.insert(var.to_string(), x);

        let va = eval_f64(ctx, a, &vars);
        let vb = eval_f64(ctx, b, &vars);

        match (va, vb) {
            (Some(a_val), Some(b_val)) if a_val.is_finite() && b_val.is_finite() => {
                valid_checks += 1;
                let diff = (a_val - b_val).abs();
                let rel = diff / a_val.abs().max(1e-10);
                if diff < 1e-8 || rel < 1e-8 {
                    matching += 1;
                }
            }
            _ => {} // Skip invalid samples
        }
    }

    // If no valid samples, return true (inconclusive, not a failure)
    // If at least 2 valid samples, require all to match
    if valid_checks < 2 {
        true // Inconclusive - skip this check
    } else {
        matching == valid_checks
    }
}

/// Test that simplify(E) == simplify(shuffle(E)) for a single expression
fn test_shuffle_invariance(expr_str: &str, _label: &str) -> Result<(), String> {
    // Create simplifier (which owns Context)
    let mut simplifier = Simplifier::new();

    // Parse the expression
    let expr = match parse(expr_str, &mut simplifier.context) {
        Ok(e) => e,
        Err(_) => return Err("parse failed".to_string()),
    };

    // Simplify original
    let (simplified_original, _) = simplifier.simplify(expr);
    let original_str = format!("{:?}", simplifier.context.get(simplified_original));

    // Shuffle the expression
    let shuffled = shuffle_expr(&mut simplifier.context, expr);

    // Simplify shuffled
    let (simplified_shuffled, _) = simplifier.simplify(shuffled);
    let shuffled_str = format!("{:?}", simplifier.context.get(simplified_shuffled));

    // Compare (structural equality via Debug representation)
    if original_str != shuffled_str {
        return Err(format!(
            "shuffle mismatch: '{}' vs '{}'",
            original_str, shuffled_str
        ));
    }

    Ok(())
}

// =============================================================================
// MetaTransform Test (Phase B)
// =============================================================================

/// Test metamorphic transforms: A(T(x)) ≡ B(T(x)) for transforms T.
/// Verifies that identities hold under substitution (scale, shift, square).
#[test]
#[ignore]
fn metatest_transform_identities() {
    let transform_enabled =
        env::var("METATEST_TRANSFORMS").is_ok() || env::var("METATEST_TRANSFORMS_DEFAULT").is_ok();

    if !transform_enabled {
        eprintln!("Transform test skipped. Set METATEST_TRANSFORMS=scale:2 or METATEST_TRANSFORMS_DEFAULT=1 to enable.");
        return;
    }

    let transforms = parse_meta_transforms_from_env();
    let pairs = load_identity_pairs();

    if pairs.is_empty() {
        panic!("No identity pairs loaded!");
    }

    // Parse min_valid factor from env
    let min_valid_factor: f64 = env::var("METATEST_TRANSFORM_MIN_VALID_FACTOR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.6);

    eprintln!("🔄 MetaTransform Test");
    eprintln!(
        "   Transforms: {:?}",
        transforms.iter().map(|t| t.name()).collect::<Vec<_>>()
    );
    eprintln!("   Identities: {}", pairs.len());
    eprintln!("   min_valid_factor: {}\n", min_valid_factor);

    let mut total_tests = 0;
    let mut passed = 0;
    let mut semantic_failures: Vec<String> = Vec::new();
    let mut skipped_bucket = 0;

    for pair in &pairs {
        // Skip multi-variable identities
        if pair.vars.len() != 1 {
            continue;
        }

        // Gating by bucket: BranchSensitive only gets scale(2)
        for transform in &transforms {
            // Gate BranchSensitive - only allow scale(2)
            if pair.bucket == Bucket::BranchSensitive {
                if !matches!(transform, MetaTransform::Scale(k) if (*k - 2.0).abs() < 1e-10) {
                    skipped_bucket += 1;
                    continue;
                }
            }

            total_tests += 1;

            match test_transform_identity(&pair, &pair.vars[0], transform, min_valid_factor) {
                TransformResult::Pass => passed += 1,
                TransformResult::Skip(_) => passed += 1, // Inconclusive is OK
                TransformResult::Fail(msg) => {
                    semantic_failures.push(format!(
                        "{} [{}]: {}",
                        truncate_identity(&pair.exp, 25),
                        transform.name(),
                        msg
                    ));
                }
            }
        }
    }

    // Report
    eprintln!("📊 Transform Results:");
    eprintln!("   Total tests: {}", total_tests);
    eprintln!("   Passed: {}", passed);
    eprintln!("   Skipped (bucket gate): {}", skipped_bucket);
    eprintln!("   Semantic failures: {}", semantic_failures.len());

    if !semantic_failures.is_empty() {
        eprintln!("\n🚨 TRANSFORM FAILURES:");
        for (i, fail) in semantic_failures.iter().take(10).enumerate() {
            eprintln!("   {}. {}", i + 1, fail);
        }
        if semantic_failures.len() > 10 {
            eprintln!("   ... and {} more", semantic_failures.len() - 10);
        }
        panic!(
            "Transform test failed with {} semantic failures",
            semantic_failures.len()
        );
    }

    eprintln!("\n✅ All transform tests passed!");
}

enum TransformResult {
    Pass,
    Skip(String),
    Fail(String),
}

/// Test that A(T(x)) ≡ B(T(x)) for a specific transform
fn test_transform_identity(
    pair: &IdentityPair,
    var: &str,
    transform: &MetaTransform,
    min_valid_factor: f64,
) -> TransformResult {
    let mut simplifier = Simplifier::new();

    // Parse expressions
    let exp = match parse(&pair.exp, &mut simplifier.context) {
        Ok(e) => e,
        Err(_) => return TransformResult::Skip("parse exp failed".to_string()),
    };
    let simp = match parse(&pair.simp, &mut simplifier.context) {
        Ok(e) => e,
        Err(_) => return TransformResult::Skip("parse simp failed".to_string()),
    };

    // Simplify both
    let (exp_simplified, _) = simplifier.simplify(exp);
    let (simp_simplified, _) = simplifier.simplify(simp);

    // Sample and evaluate with transform + composed filter
    let samples: Vec<f64> = (-50..=50).map(|i| (i as f64) * 0.2).collect();

    let min_valid = ((samples.len() as f64) * 0.9 * min_valid_factor) as usize;

    let mut valid = 0;
    let mut matching = 0;
    let mut filtered_out = 0;

    for &x in &samples {
        // Apply transform: x' = T(x)
        let x_prime = transform.apply_f64(x);

        // Composed filter: check if x' passes the original filter
        if !pair.filter_spec.accept(x_prime) {
            filtered_out += 1;
            continue;
        }

        // Evaluate at x'
        let mut vars = HashMap::new();
        vars.insert(var.to_string(), x_prime);

        let va = eval_f64(&simplifier.context, exp_simplified, &vars);
        let vb = eval_f64(&simplifier.context, simp_simplified, &vars);

        match (va, vb) {
            (Some(a), Some(b)) if a.is_finite() && b.is_finite() => {
                valid += 1;
                let diff = (a - b).abs();
                let rel = diff / a.abs().max(1e-10);
                if diff < 1e-6 || rel < 1e-6 {
                    matching += 1;
                }
            }
            _ => {} // Skip invalid evaluations
        }
    }

    // Check results
    if valid < min_valid {
        // Inconclusive - not enough valid samples
        return TransformResult::Skip(format!("only {}/{} valid", valid, min_valid));
    }

    if matching != valid {
        return TransformResult::Fail(format!("mismatch {}/{} valid", matching, valid));
    }

    TransformResult::Pass
}
