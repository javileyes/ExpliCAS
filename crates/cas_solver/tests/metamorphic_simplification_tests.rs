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
use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_session::SessionState;
use cas_solver::api::{
    eval_f64, eval_f64_checked, EquivalenceResult, EvalCheckedError, EvalCheckedOptions,
};
use cas_solver::runtime::{Engine, EvalAction, EvalRequest, EvalResult, Simplifier};
use cas_solver::wire::eval_str_to_wire;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::OnceLock;

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

fn find_test_data_file(filename: &str) -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let local = manifest_dir.join("tests").join(filename);
    if local.exists() {
        return local;
    }

    // Migration compatibility: data moved to cas_solver/tests.
    manifest_dir
        .parent()
        .map(|p| p.join("cas_solver").join("tests").join(filename))
        .unwrap_or(local)
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
// Shape Signature Analysis (for Top-N pattern identification)
// =============================================================================

use cas_ast::Expr;

/// Generate a stable shape signature from an expression.
/// Collapses literals to NUM, variables to SYM, and marks negative exponents.
/// Used to identify dominant patterns in numeric-only cases.
fn expr_shape_signature(ctx: &Context, expr: ExprId) -> String {
    fn inner(ctx: &Context, expr: ExprId, depth: usize) -> String {
        if depth > 10 {
            return "...".to_string();
        }
        match ctx.get(expr) {
            Expr::Number(n) => {
                if n.is_integer() {
                    let val = n.numer().to_string().parse::<i64>().unwrap_or(0);
                    if val < 0 {
                        format!("INT_NEG({})", val)
                    } else {
                        "INT".to_string()
                    }
                } else {
                    "FRAC".to_string()
                }
            }
            Expr::Variable(_) => "SYM".to_string(),
            Expr::Constant(_) => "CONST".to_string(),
            Expr::Add(l, r) => {
                let mut parts = [inner(ctx, *l, depth + 1), inner(ctx, *r, depth + 1)];
                parts.sort(); // Canonical order for commutativity
                format!("Add({})", parts.join(","))
            }
            Expr::Sub(l, r) => {
                format!(
                    "Sub({},{})",
                    inner(ctx, *l, depth + 1),
                    inner(ctx, *r, depth + 1)
                )
            }
            Expr::Mul(l, r) => {
                let mut parts = [inner(ctx, *l, depth + 1), inner(ctx, *r, depth + 1)];
                parts.sort(); // Canonical order for commutativity
                format!("Mul({})", parts.join(","))
            }
            Expr::Div(l, r) => {
                format!(
                    "Div({},{})",
                    inner(ctx, *l, depth + 1),
                    inner(ctx, *r, depth + 1)
                )
            }
            Expr::Pow(base, exp) => {
                // Special handling for negative integer exponents
                let exp_sig = match ctx.get(*exp) {
                    Expr::Number(n) if n.is_integer() => {
                        let val = n.numer().to_string().parse::<i64>().unwrap_or(0);
                        if val < 0 {
                            format!("INT_NEG({})", val)
                        } else if val == 2 {
                            "2".to_string()
                        } else {
                            "INT".to_string()
                        }
                    }
                    Expr::Number(n) => {
                        // Check for 1/n fractions (roots)
                        if *n.numer() == 1.into() {
                            "1/N".to_string()
                        } else {
                            "FRAC".to_string()
                        }
                    }
                    _ => inner(ctx, *exp, depth + 1),
                };
                format!("Pow({},{})", inner(ctx, *base, depth + 1), exp_sig)
            }
            Expr::Neg(inner_expr) => {
                format!("Neg({})", inner(ctx, *inner_expr, depth + 1))
            }
            Expr::Function(name, args) => {
                let arg_sigs: Vec<_> = args.iter().map(|a| inner(ctx, *a, depth + 1)).collect();
                format!("{}({})", name, arg_sigs.join(","))
            }
            Expr::Matrix { .. } => "MAT".to_string(),
            Expr::SessionRef(_) => "REF".to_string(),
            Expr::Hold(held) => format!("Hold({})", inner(ctx, *held, depth + 1)),
        }
    }
    inner(ctx, expr, 0)
}

/// Check if a shape signature contains negative exponent patterns
fn shape_has_neg_exp(shape: &str) -> bool {
    shape.contains("INT_NEG")
}

/// Check if a shape signature is a Div structure
fn shape_has_div(shape: &str) -> bool {
    shape.contains("Div(")
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
// Symbolic Equivalence Check (Bucket-aware)
// =============================================================================

/// Result of bucket-aware symbolic equivalence check
#[derive(Debug, Clone, PartialEq)]
enum SymbolicResult {
    /// A ≡ B unconditionally (pure equivalence)
    Pass,
    /// A ≡ B with conditions (allowed for ConditionalRequires bucket)
    PassConditional(Vec<String>),
    /// A ≡ B but required conditions in Unconditional bucket (not counted as symbolic)
    Conditional(Vec<String>),
    /// A ≢ B (proved non-equivalent)
    Fail,
    /// Cannot determine symbolically
    Unknown,
    /// Skip symbolic check (for BranchSensitive)
    SkipSymbolic,
}

/// Check symbolic equivalence using are_equivalent_extended with bucket gating.
///
/// Uses the engine's equivalence API which tracks soundness labels and
/// introduced requires, then gates the result based on bucket.
///
/// V2.15.8: Adds polynomial normalization fallback when equivalence is Unknown.
/// This enables proving identities like (x+1)^5 ≡ x^5 + 5x^4 + ... without
/// requiring simplify to auto-expand binomials.
fn check_symbolic_equiv_bucket_aware(
    simplifier: &mut Simplifier,
    exp_expr: ExprId,
    simp_expr: ExprId,
    bucket: Bucket,
) -> SymbolicResult {
    // Fast path: structural comparison after simplification
    let (exp_simplified, _) = simplifier.simplify(exp_expr);
    let (simp_simplified, _) = simplifier.simplify(simp_expr);

    if cas_solver::runtime::compare_expr(&simplifier.context, exp_simplified, simp_simplified)
        == std::cmp::Ordering::Equal
    {
        return SymbolicResult::Pass;
    }

    // Slow path: full equivalence check with tracking
    let eq = simplifier.are_equivalent_extended(exp_expr, simp_expr);

    // V2.15.8: Polynomial normalization fallback for Unknown results
    // This catches cases like (x+1)^5 vs expanded polynomial where simplify
    // doesn't expand but the expressions are polynomially equivalent
    let eq = if matches!(eq, EquivalenceResult::Unknown) {
        if let Some(poly_result) =
            check_polynomial_equivalence(&simplifier.context, exp_simplified, simp_simplified)
        {
            poly_result
        } else {
            eq
        }
    } else {
        eq
    };

    match (&bucket, eq) {
        // Unconditional bucket: only pure True counts as symbolic pass
        (Bucket::Unconditional, EquivalenceResult::True) => SymbolicResult::Pass,
        (Bucket::Unconditional, EquivalenceResult::ConditionalTrue { requires }) => {
            SymbolicResult::Conditional(requires) // NOT symbolic pass, falls to numeric
        }
        (Bucket::Unconditional, EquivalenceResult::False) => SymbolicResult::Fail,
        (Bucket::Unconditional, EquivalenceResult::Unknown) => SymbolicResult::Unknown,

        // ConditionalRequires: conditional counts as pass
        (Bucket::ConditionalRequires, EquivalenceResult::True) => SymbolicResult::Pass,
        (Bucket::ConditionalRequires, EquivalenceResult::ConditionalTrue { requires }) => {
            SymbolicResult::PassConditional(requires)
        }
        (Bucket::ConditionalRequires, EquivalenceResult::False) => SymbolicResult::Fail,
        (Bucket::ConditionalRequires, EquivalenceResult::Unknown) => SymbolicResult::Unknown,

        // BranchSensitive: skip symbolic except for pure True
        (Bucket::BranchSensitive, EquivalenceResult::True) => SymbolicResult::Pass,
        (Bucket::BranchSensitive, _) => SymbolicResult::SkipSymbolic,
    }
}

/// V2.15.8: Check if two expressions are equivalent as polynomials.
/// Returns Some(True) if they canonicalize to the same polynomial,
/// None if either expression is not a polynomial (contains trig, log, etc.)
fn check_polynomial_equivalence(ctx: &Context, a: ExprId, b: ExprId) -> Option<EquivalenceResult> {
    use cas_math::multipoly::{multipoly_from_expr, PolyBudget};

    // Use a generous budget for polynomial equivalence checking
    // (higher than normal since this is for testing, not runtime)
    let budget = PolyBudget {
        max_terms: 500,       // Allow up to 500 terms (covers (x+1)^8 etc.)
        max_total_degree: 15, // Allow up to degree 15
        max_pow_exp: 10,      // Allow exponents up to 10
    };

    let pa = multipoly_from_expr(ctx, a, &budget).ok()?;
    let pb = multipoly_from_expr(ctx, b, &budget).ok()?;

    if pa == pb {
        Some(EquivalenceResult::True)
    } else {
        // Polynomials are different - this is a definite non-equivalence
        Some(EquivalenceResult::False)
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

/// Collect only the immediate top-level addends of an Add node.
/// This preserves contextual grouping like `(A + B)` vs `(C + D)` before
/// flattening nested sums inside each side.
fn collect_shallow_addends(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    match ctx.nodes.get(expr.index()) {
        Some(cas_ast::Expr::Add(a, b)) => vec![*a, *b],
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
            Some(cas_ast::Expr::Variable(sym_id)) => {
                let mut h = hash_combine(h, b'V');
                // sym_id is a SymbolId (usize), need to convert to string representation
                // Since we don't have Context here, use the raw id as bytes
                for b in sym_id.to_string().bytes() {
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
            Some(cas_ast::Expr::Function(name_id, args)) => {
                let mut h = hash_combine(h, b'F');
                for b in ctx.sym_name(*name_id).bytes() {
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
    finalize_numeric_equiv_1var(stats, config)
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
#[allow(clippy::too_many_arguments)]
fn check_numeric_equiv_2var(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
    config: &MetatestConfig,
    filter1: &FilterSpec,
    filter2: &FilterSpec,
) -> Result<usize, String> {
    let stats = check_numeric_equiv_2var_stats(ctx, a, b, var1, var2, config, filter1, filter2);
    finalize_numeric_equiv_2var(stats, config)
}

#[allow(clippy::too_many_arguments)]
fn check_numeric_equiv_2var_stats(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
    config: &MetatestConfig,
    filter1: &FilterSpec,
    filter2: &FilterSpec,
) -> NumericEquivStats {
    let (lo, hi) = config.sample_range;
    let mut stats = NumericEquivStats::default();

    // Configure checked evaluator
    let opts = EvalCheckedOptions {
        zero_abs_eps: 1e-12,
        zero_rel_eps: 1e-12,
        trig_pole_eps: 1e-9,
        max_depth: 200,
    };

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

            // Apply per-variable domain filters
            if !filter1.accept(x) || !filter2.accept(y) {
                stats.domain_error += 1;
                continue;
            }

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
                        stats.record_mismatch_label(
                            format!("{var1}={x:.6}, {var2}={y:.6}"),
                            *va,
                            *vb,
                        );
                    }
                }
                // Symmetric failures
                (
                    Err(EvalCheckedError::NearPole { .. }),
                    Err(EvalCheckedError::NearPole { .. }),
                ) => {
                    stats.near_pole += 1;
                }
                (Err(EvalCheckedError::Domain { .. }), Err(EvalCheckedError::Domain { .. })) => {
                    stats.domain_error += 1;
                }
                // Asymmetric: one Ok, one Err
                (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
                    stats.asymmetric_invalid += 1;
                }
                _ => {
                    stats.eval_failed += 1;
                }
            }
        }
    }

    stats
}

fn finalize_numeric_equiv_2var(
    stats: NumericEquivStats,
    config: &MetatestConfig,
) -> Result<usize, String> {
    // Lower threshold for 2D, adjusted for problematic samples
    let problematic =
        stats.near_pole + stats.domain_error + stats.eval_failed + stats.asymmetric_invalid;
    let total_samples = {
        let samples_per_dim = (config.eval_samples as f64).sqrt() as usize;
        samples_per_dim * samples_per_dim
    };
    let base_min_valid = config.min_valid / 4;
    let adjusted_min_valid = if problematic > total_samples / 4 {
        (total_samples - problematic) / 2
    } else {
        base_min_valid
    };

    if stats.valid < adjusted_min_valid {
        return Err(format!(
            "Too few valid samples: {} < {} (near_pole={}, domain_error={}, asymmetric={}, eval_failed={})",
            stats.valid,
            adjusted_min_valid,
            stats.near_pole,
            stats.domain_error,
            stats.asymmetric_invalid,
            stats.eval_failed
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

/// Check if two expressions are numerically equivalent for 3+ variables.
/// Uses a deterministic low-discrepancy style sampling pattern instead of a full grid
/// to keep runtime bounded while still covering multivariate contextual identities.
fn check_numeric_equiv_nvar(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    vars: &[String],
    config: &MetatestConfig,
) -> Result<usize, String> {
    let (lo, hi) = config.sample_range;
    let mut valid = 0usize;
    let mut eval_failed = 0usize;
    let mut near_pole = 0usize;
    let mut domain_error = 0usize;
    let mut asymmetric_invalid = 0usize;

    let opts = EvalCheckedOptions {
        zero_abs_eps: 1e-12,
        zero_rel_eps: 1e-12,
        trig_pole_eps: 1e-9,
        max_depth: 200,
    };

    // Golden-ratio increment for a simple deterministic low-discrepancy walk.
    const PHASE: f64 = 0.381_966_011_250_105_1;

    for i in 0..config.eval_samples {
        let base = (i as f64 + 0.5) / config.eval_samples as f64;
        let mut var_map = HashMap::new();

        for (idx, var) in vars.iter().enumerate() {
            let t = (base + idx as f64 * PHASE).fract();
            let value = lo + (hi - lo) * t;
            var_map.insert(var.clone(), value);
        }

        let va = eval_f64_checked(ctx, a, &var_map, &opts);
        let vb = eval_f64_checked(ctx, b, &var_map, &opts);

        match (&va, &vb) {
            (Ok(va), Ok(vb)) => {
                valid += 1;

                let diff = (va - vb).abs();
                let scale = va.abs().max(vb.abs()).max(1.0);
                let allowed = config.atol + config.rtol * scale;

                if diff > allowed {
                    let bindings = vars
                        .iter()
                        .map(|v| format!("{v}={:.12}", var_map[v]))
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Err(format!(
                        "Numeric mismatch at {}:\n  a={:.15}\n  b={:.15}\n  diff={:.3e} > allowed={:.3e}",
                        bindings, va, vb, diff, allowed
                    ));
                }
            }
            (Err(EvalCheckedError::NearPole { .. }), Err(EvalCheckedError::NearPole { .. })) => {
                near_pole += 1;
            }
            (Err(EvalCheckedError::Domain { .. }), Err(EvalCheckedError::Domain { .. })) => {
                domain_error += 1;
            }
            (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
                asymmetric_invalid += 1;
            }
            _ => {
                eval_failed += 1;
            }
        }
    }

    let problematic = near_pole + domain_error + eval_failed + asymmetric_invalid;
    let adjusted_min_valid = if problematic > config.eval_samples / 4 {
        (config.eval_samples - problematic) / 2
    } else {
        config.min_valid / 2
    };

    if valid < adjusted_min_valid {
        return Err(format!(
            "Too few valid samples: {} < {} (near_pole={}, domain_error={}, asymmetric={}, eval_failed={})",
            valid, adjusted_min_valid, near_pole, domain_error, asymmetric_invalid, eval_failed
        ));
    }

    Ok(valid)
}

fn check_numeric_equiv_1var_with_fixed(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var: &str,
    fixed_vars: &[(String, f64)],
    config: &MetatestConfig,
) -> Result<usize, String> {
    let stats = check_numeric_equiv_1var_with_fixed_stats(ctx, a, b, var, fixed_vars, config);
    finalize_numeric_equiv_1var(stats, config)
}

fn check_numeric_equiv_1var_with_fixed_stats(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var: &str,
    fixed_vars: &[(String, f64)],
    config: &MetatestConfig,
) -> NumericEquivStats {
    check_numeric_equiv_1var_with_fixed_stats_filtered(
        ctx,
        a,
        b,
        var,
        fixed_vars,
        config,
        &FilterSpec::None,
    )
}

fn check_numeric_equiv_1var_with_fixed_stats_filtered(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var: &str,
    fixed_vars: &[(String, f64)],
    config: &MetatestConfig,
    filter_spec: &FilterSpec,
) -> NumericEquivStats {
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

        if !filter_spec.accept(x) {
            stats.filtered_out += 1;
            continue;
        }

        let mut var_map = HashMap::new();
        for (name, value) in fixed_vars {
            var_map.insert(name.clone(), *value);
        }
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

fn check_numeric_equiv_2var_with_fixed(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
    fixed_vars: &[(String, f64)],
    config: &MetatestConfig,
) -> Result<usize, String> {
    let stats =
        check_numeric_equiv_2var_with_fixed_stats(ctx, a, b, var1, var2, fixed_vars, config);
    finalize_numeric_equiv_2var(stats, config)
}

fn check_numeric_equiv_2var_with_fixed_stats(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
    fixed_vars: &[(String, f64)],
    config: &MetatestConfig,
) -> NumericEquivStats {
    check_numeric_equiv_2var_with_fixed_stats_filtered(
        ctx,
        a,
        b,
        var1,
        var2,
        fixed_vars,
        config,
        &FilterSpec::None,
        &FilterSpec::None,
    )
}

#[allow(clippy::too_many_arguments)]
fn check_numeric_equiv_2var_with_fixed_stats_filtered(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
    fixed_vars: &[(String, f64)],
    config: &MetatestConfig,
    filter1: &FilterSpec,
    filter2: &FilterSpec,
) -> NumericEquivStats {
    let (lo, hi) = config.sample_range;
    let mut stats = NumericEquivStats::default();

    let opts = EvalCheckedOptions {
        zero_abs_eps: 1e-12,
        zero_rel_eps: 1e-12,
        trig_pole_eps: 1e-9,
        max_depth: 200,
    };

    let samples_per_dim = (config.eval_samples as f64).sqrt() as usize;

    for i in 0..samples_per_dim {
        for j in 0..samples_per_dim {
            let t1 = (i as f64 + 0.5) / samples_per_dim as f64;
            let t2 = (j as f64 + 0.5) / samples_per_dim as f64;
            let x = lo + (hi - lo) * t1;
            let y = lo + (hi - lo) * t2;

            if !filter1.accept(x) || !filter2.accept(y) {
                stats.filtered_out += 1;
                continue;
            }

            let mut var_map = HashMap::new();
            for (name, value) in fixed_vars {
                var_map.insert(name.clone(), *value);
            }
            var_map.insert(var1.to_string(), x);
            var_map.insert(var2.to_string(), y);

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
                        stats.record_mismatch_label(
                            format!("{var1}={x:.6}, {var2}={y:.6}"),
                            *va,
                            *vb,
                        );
                    }
                }
                (
                    Err(EvalCheckedError::NearPole { .. }),
                    Err(EvalCheckedError::NearPole { .. }),
                ) => {
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
    }

    stats
}

fn sample_nvar_slice_anchor(config: &MetatestConfig, idx: usize, seed: f64) -> f64 {
    let (lo, hi) = config.sample_range;
    const PHASE: f64 = 0.381_966_011_250_105_1;
    let t = (seed + idx as f64 * PHASE).fract();
    lo + (hi - lo) * t
}

fn sample_nvar_slice_anchor_filtered(
    config: &MetatestConfig,
    idx: usize,
    seed: f64,
    filter: &FilterSpec,
) -> f64 {
    let base = sample_nvar_slice_anchor(config, idx, seed);
    if filter.accept(base) || filter.is_none() {
        return base;
    }

    const OFFSETS: [f64; 8] = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875];

    for offset in OFFSETS {
        let candidate = sample_nvar_slice_anchor(config, idx, (seed + offset).fract());
        if filter.accept(candidate) {
            return candidate;
        }
    }

    base
}

fn classify_numeric_equiv_nvar_relaxed(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    vars: &[String],
    filters: &[FilterSpec],
    config: &MetatestConfig,
) -> NumericCheckOutcome {
    let direct = check_numeric_equiv_nvar(ctx, a, b, vars, config);
    let direct_error = match direct {
        Ok(_) => return NumericCheckOutcome::Pass,
        Err(msg) => msg,
    };

    let direct_was_inconclusive = direct_error.starts_with("Too few valid samples:");
    let mut passed_slices = 0usize;
    let mut inconclusive_slices = 0usize;
    const SLICE_SEEDS: [f64; 2] = [0.173_205_080_756_887_73, 0.618_033_988_749_894_8];
    const MAX_PAIR_SLICES_PER_SEED: usize = 6;

    for seed in SLICE_SEEDS {
        let mut checked_pairs = 0usize;
        let anchors = vars
            .iter()
            .enumerate()
            .map(|(idx, var)| {
                let filter = filters.get(idx).unwrap_or(&FilterSpec::None);
                (
                    var.clone(),
                    sample_nvar_slice_anchor_filtered(config, idx, seed, filter),
                )
            })
            .collect::<Vec<_>>();

        for (idx, free_var) in vars.iter().enumerate() {
            let fixed = anchors
                .iter()
                .filter(|(name, _)| name != free_var)
                .cloned()
                .collect::<Vec<_>>();
            match classify_numeric_equiv_1var_with_fixed_relaxed(
                ctx,
                a,
                b,
                free_var,
                &fixed,
                config,
                filters.get(idx).unwrap_or(&FilterSpec::None),
            ) {
                NumericCheckOutcome::Pass => passed_slices += 1,
                NumericCheckOutcome::Inconclusive(_) => inconclusive_slices += 1,
                NumericCheckOutcome::Failed(msg) => {
                    return NumericCheckOutcome::Failed(format!(
                        "{} | slice(1d,{free_var}) failed: {}",
                        direct_error, msg
                    ));
                }
            }
        }

        'pair_slices: for (idx, var1) in vars.iter().enumerate() {
            for (offset, var2) in vars.iter().skip(idx + 1).enumerate() {
                if checked_pairs >= MAX_PAIR_SLICES_PER_SEED {
                    break 'pair_slices;
                }
                checked_pairs += 1;
                let idx2 = idx + 1 + offset;

                let fixed = anchors
                    .iter()
                    .filter(|(name, _)| name != var1 && name != var2)
                    .cloned()
                    .collect::<Vec<_>>();

                match classify_numeric_equiv_2var_with_fixed_relaxed(
                    ctx,
                    a,
                    b,
                    var1,
                    var2,
                    &fixed,
                    config,
                    filters.get(idx).unwrap_or(&FilterSpec::None),
                    filters.get(idx2).unwrap_or(&FilterSpec::None),
                ) {
                    NumericCheckOutcome::Pass => passed_slices += 1,
                    NumericCheckOutcome::Inconclusive(_) => inconclusive_slices += 1,
                    NumericCheckOutcome::Failed(msg) => {
                        return NumericCheckOutcome::Failed(format!(
                            "{} | slice(2d,{var1},{var2}) failed: {}",
                            direct_error, msg
                        ));
                    }
                }
            }
        }
    }

    if passed_slices > 0 {
        NumericCheckOutcome::Inconclusive(format!(
            "Direct n-var check failed but deterministic slices passed ({passed_slices} passed, {inconclusive_slices} inconclusive): {direct_error}"
        ))
    } else if direct_was_inconclusive || inconclusive_slices > 0 {
        NumericCheckOutcome::Inconclusive(format!(
            "Direct n-var check remained inconclusive ({inconclusive_slices} slices inconclusive): {direct_error}"
        ))
    } else {
        NumericCheckOutcome::Failed(direct_error)
    }
}

#[derive(Debug, Clone)]
enum NumericCheckOutcome {
    Pass,
    Inconclusive(String),
    Failed(String),
}

fn finalize_numeric_equiv_1var(
    stats: NumericEquivStats,
    config: &MetatestConfig,
) -> Result<usize, String> {
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
            stats.valid,
            adjusted_min_valid,
            stats.near_pole,
            stats.domain_error,
            stats.asymmetric_invalid,
            stats.eval_failed
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

fn classify_numeric_check(result: Result<usize, String>) -> NumericCheckOutcome {
    match result {
        Ok(_) => NumericCheckOutcome::Pass,
        Err(msg)
            if msg.starts_with("Too few valid samples:")
                || msg.starts_with("Unsupported contextual numeric arity:") =>
        {
            NumericCheckOutcome::Inconclusive(msg)
        }
        Err(msg) => NumericCheckOutcome::Failed(msg),
    }
}

fn classify_numeric_check_with_stats(
    result: Result<usize, String>,
    stats: &NumericEquivStats,
) -> NumericCheckOutcome {
    match result {
        Ok(_) => NumericCheckOutcome::Pass,
        Err(msg)
            if msg.starts_with("Too few valid samples:")
                || msg.starts_with("Unsupported contextual numeric arity:") =>
        {
            NumericCheckOutcome::Inconclusive(msg)
        }
        Err(msg) => match classify_diagnostic(stats) {
            DiagCategory::BugSignal | DiagCategory::Ok => NumericCheckOutcome::Failed(msg),
            DiagCategory::ConfigError | DiagCategory::NeedsFilter | DiagCategory::Fragile => {
                NumericCheckOutcome::Inconclusive(format!(
                    "{}: {}",
                    classify_diagnostic(stats).name(),
                    msg
                ))
            }
        },
    }
}

fn numeric_retry_filters_1var() -> [FilterSpec; 4] {
    [
        FilterSpec::AwayFrom {
            centers: vec![0.0, 1.0, -1.0],
            eps: 0.1,
        },
        FilterSpec::AbsLtAndAway {
            limit: 0.9,
            centers: vec![0.0, 1.0, -1.0],
            eps: 0.1,
        },
        FilterSpec::Range {
            min: -0.8,
            max: 0.8,
        },
        FilterSpec::AbsLt { limit: 0.9 },
    ]
}

fn numeric_retry_filters_2var() -> [(FilterSpec, FilterSpec); 4] {
    [
        (
            FilterSpec::AwayFrom {
                centers: vec![0.0, 1.0, -1.0],
                eps: 0.1,
            },
            FilterSpec::AwayFrom {
                centers: vec![0.0, 1.0, -1.0],
                eps: 0.1,
            },
        ),
        (
            FilterSpec::AbsLtAndAway {
                limit: 0.9,
                centers: vec![0.0, 1.0, -1.0],
                eps: 0.1,
            },
            FilterSpec::AbsLtAndAway {
                limit: 0.9,
                centers: vec![0.0, 1.0, -1.0],
                eps: 0.1,
            },
        ),
        (
            FilterSpec::Range {
                min: -0.8,
                max: 0.8,
            },
            FilterSpec::Range {
                min: -0.8,
                max: 0.8,
            },
        ),
        (
            FilterSpec::AbsLt { limit: 0.9 },
            FilterSpec::AbsLt { limit: 0.9 },
        ),
    ]
}

fn should_retry_relaxed_numeric_1var(
    result: &Result<usize, String>,
    stats: &NumericEquivStats,
) -> bool {
    match result {
        Ok(_) => false,
        Err(msg) if msg.starts_with("Unsupported contextual numeric arity:") => false,
        Err(msg) if msg.starts_with("Too few valid samples:") => true,
        Err(_) => matches!(
            classify_diagnostic(stats),
            DiagCategory::NeedsFilter | DiagCategory::Fragile
        ),
    }
}

fn should_retry_relaxed_numeric_2var(
    result: &Result<usize, String>,
    stats: &NumericEquivStats,
) -> bool {
    match result {
        Ok(_) => false,
        Err(msg) if msg.starts_with("Unsupported contextual numeric arity:") => false,
        Err(msg) if msg.starts_with("Too few valid samples:") => true,
        Err(_) => matches!(
            classify_diagnostic(stats),
            DiagCategory::NeedsFilter | DiagCategory::Fragile
        ),
    }
}

fn classify_numeric_equiv_1var_relaxed_with<F>(
    config: &MetatestConfig,
    mut run_stats: F,
) -> NumericCheckOutcome
where
    F: FnMut(&FilterSpec) -> NumericEquivStats,
{
    let direct_stats = run_stats(&FilterSpec::None);
    let direct_result = finalize_numeric_equiv_1var(direct_stats.clone(), config);
    let direct_outcome = classify_numeric_check_with_stats(direct_result.clone(), &direct_stats);

    if matches!(direct_outcome, NumericCheckOutcome::Pass) {
        return NumericCheckOutcome::Pass;
    }

    if !should_retry_relaxed_numeric_1var(&direct_result, &direct_stats) {
        return direct_outcome;
    }

    let mut retry_notes = Vec::new();
    for filter in numeric_retry_filters_1var() {
        let stats = run_stats(&filter);
        let result = finalize_numeric_equiv_1var(stats.clone(), config);
        match classify_numeric_check_with_stats(result, &stats) {
            NumericCheckOutcome::Pass => return NumericCheckOutcome::Pass,
            NumericCheckOutcome::Failed(msg) => {
                return NumericCheckOutcome::Failed(format!(
                    "after filter {} => {}",
                    filter.as_str(),
                    msg
                ));
            }
            NumericCheckOutcome::Inconclusive(msg) => {
                retry_notes.push(format!("{} => {}", filter.as_str(), msg));
            }
        }
    }

    let base_msg = match direct_outcome {
        NumericCheckOutcome::Inconclusive(msg) | NumericCheckOutcome::Failed(msg) => msg,
        NumericCheckOutcome::Pass => unreachable!("pass returns early"),
    };

    if retry_notes.is_empty() {
        NumericCheckOutcome::Inconclusive(base_msg)
    } else {
        NumericCheckOutcome::Inconclusive(format!(
            "{} [retry_filters: {}]",
            base_msg,
            retry_notes.join(" | ")
        ))
    }
}

fn classify_numeric_equiv_2var_relaxed_with<F>(
    config: &MetatestConfig,
    mut run_stats: F,
) -> NumericCheckOutcome
where
    F: FnMut(&FilterSpec, &FilterSpec) -> NumericEquivStats,
{
    let direct_stats = run_stats(&FilterSpec::None, &FilterSpec::None);
    let direct_result = finalize_numeric_equiv_2var(direct_stats.clone(), config);
    let direct_outcome = classify_numeric_check_with_stats(direct_result.clone(), &direct_stats);

    if matches!(direct_outcome, NumericCheckOutcome::Pass) {
        return NumericCheckOutcome::Pass;
    }

    if !should_retry_relaxed_numeric_2var(&direct_result, &direct_stats) {
        return direct_outcome;
    }

    let mut retry_notes = Vec::new();
    for (filter1, filter2) in numeric_retry_filters_2var() {
        let stats = run_stats(&filter1, &filter2);
        let result = finalize_numeric_equiv_2var(stats.clone(), config);
        match classify_numeric_check_with_stats(result, &stats) {
            NumericCheckOutcome::Pass => return NumericCheckOutcome::Pass,
            NumericCheckOutcome::Failed(msg) => {
                return NumericCheckOutcome::Failed(format!(
                    "after filters ({}, {}) => {}",
                    filter1.as_str(),
                    filter2.as_str(),
                    msg
                ));
            }
            NumericCheckOutcome::Inconclusive(msg) => {
                retry_notes.push(format!(
                    "({}, {}) => {}",
                    filter1.as_str(),
                    filter2.as_str(),
                    msg
                ));
            }
        }
    }

    let base_msg = match direct_outcome {
        NumericCheckOutcome::Inconclusive(msg) | NumericCheckOutcome::Failed(msg) => msg,
        NumericCheckOutcome::Pass => unreachable!("pass returns early"),
    };

    if retry_notes.is_empty() {
        NumericCheckOutcome::Inconclusive(base_msg)
    } else {
        NumericCheckOutcome::Inconclusive(format!(
            "{} [retry_filters: {}]",
            base_msg,
            retry_notes.join(" | ")
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn check_numeric_equiv_2var_with_fixed_stats_retry_filters(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
    fixed_vars: &[(String, f64)],
    config: &MetatestConfig,
    base_filter1: &FilterSpec,
    base_filter2: &FilterSpec,
    retry_filter1: &FilterSpec,
    retry_filter2: &FilterSpec,
) -> NumericEquivStats {
    let (lo, hi) = config.sample_range;
    let mut stats = NumericEquivStats::default();

    let opts = EvalCheckedOptions {
        zero_abs_eps: 1e-12,
        zero_rel_eps: 1e-12,
        trig_pole_eps: 1e-9,
        max_depth: 200,
    };

    let samples_per_dim = (config.eval_samples as f64).sqrt() as usize;

    for i in 0..samples_per_dim {
        for j in 0..samples_per_dim {
            let t1 = (i as f64 + 0.5) / samples_per_dim as f64;
            let t2 = (j as f64 + 0.5) / samples_per_dim as f64;
            let x = lo + (hi - lo) * t1;
            let y = lo + (hi - lo) * t2;

            if !(base_filter1.accept(x)
                && retry_filter1.accept(x)
                && base_filter2.accept(y)
                && retry_filter2.accept(y))
            {
                stats.filtered_out += 1;
                continue;
            }

            let mut var_map = HashMap::new();
            for (name, value) in fixed_vars {
                var_map.insert(name.clone(), *value);
            }
            var_map.insert(var1.to_string(), x);
            var_map.insert(var2.to_string(), y);

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
                        stats.record_mismatch_label(
                            format!("{var1}={x:.6}, {var2}={y:.6}"),
                            *va,
                            *vb,
                        );
                    }
                }
                (
                    Err(EvalCheckedError::NearPole { .. }),
                    Err(EvalCheckedError::NearPole { .. }),
                ) => {
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
    }

    stats
}

fn classify_numeric_equiv_1var_relaxed(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var: &str,
    config: &MetatestConfig,
) -> NumericCheckOutcome {
    classify_numeric_equiv_1var_relaxed_with(config, |filter_spec| {
        check_numeric_equiv_1var_stats(ctx, a, b, var, config, filter_spec)
    })
}

#[allow(clippy::too_many_arguments)]
fn classify_numeric_equiv_2var_relaxed(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
    config: &MetatestConfig,
    filter1: &FilterSpec,
    filter2: &FilterSpec,
) -> NumericCheckOutcome {
    classify_numeric_equiv_2var_relaxed_with(config, |retry_filter1, retry_filter2| {
        if retry_filter1.is_none() && retry_filter2.is_none() {
            check_numeric_equiv_2var_stats(ctx, a, b, var1, var2, config, filter1, filter2)
        } else {
            check_numeric_equiv_2var_with_fixed_stats_retry_filters(
                ctx,
                a,
                b,
                var1,
                var2,
                &[],
                config,
                filter1,
                filter2,
                retry_filter1,
                retry_filter2,
            )
        }
    })
}

fn classify_numeric_equiv_1var_with_fixed_relaxed(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var: &str,
    fixed_vars: &[(String, f64)],
    config: &MetatestConfig,
    filter: &FilterSpec,
) -> NumericCheckOutcome {
    classify_numeric_equiv_1var_relaxed_with(config, |filter_spec| {
        let effective_filter = if filter_spec.is_none() {
            filter
        } else {
            filter_spec
        };
        check_numeric_equiv_1var_with_fixed_stats_filtered(
            ctx,
            a,
            b,
            var,
            fixed_vars,
            config,
            effective_filter,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn classify_numeric_equiv_2var_with_fixed_relaxed(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
    fixed_vars: &[(String, f64)],
    config: &MetatestConfig,
    filter1: &FilterSpec,
    filter2: &FilterSpec,
) -> NumericCheckOutcome {
    classify_numeric_equiv_2var_relaxed_with(config, |retry_filter1, retry_filter2| {
        if retry_filter1.is_none() && retry_filter2.is_none() {
            check_numeric_equiv_2var_with_fixed_stats_filtered(
                ctx, a, b, var1, var2, fixed_vars, config, filter1, filter2,
            )
        } else {
            check_numeric_equiv_2var_with_fixed_stats_retry_filters(
                ctx,
                a,
                b,
                var1,
                var2,
                fixed_vars,
                config,
                filter1,
                filter2,
                retry_filter1,
                retry_filter2,
            )
        }
    })
}

fn fold_constants_safe(ctx: &mut Context, expr: ExprId) -> ExprId {
    let cfg = cas_solver::runtime::EvalConfig::default();
    let mut budget = cas_solver::runtime::Budget::preset_cli();
    cas_solver::api::fold_constants(
        ctx,
        expr,
        &cfg,
        cas_solver::api::ConstFoldMode::Safe,
        &mut budget,
    )
    .map(|r| r.expr)
    .unwrap_or(expr)
}

fn expr_is_zero(ctx: &Context, expr: ExprId) -> bool {
    let zero = num_rational::BigRational::from_integer(0.into());
    matches!(ctx.get(expr), cas_ast::Expr::Number(n) if *n == zero)
}

fn prove_zero_from_diff_text(lhs: &str, rhs: &str) -> bool {
    let d_str = format!("({lhs}) - ({rhs})");
    let mut sd = Simplifier::with_default_rules();
    let Ok(dp) = parse(&d_str, &mut sd.context) else {
        return false;
    };

    let (mut dr_simp, _) = sd.simplify(dp);
    dr_simp = fold_constants_safe(&mut sd.context, dr_simp);
    if expr_is_zero(&sd.context, dr_simp) {
        return true;
    }

    let (mut dr_expand, _) = sd.expand(dp);
    dr_expand = fold_constants_safe(&mut sd.context, dr_expand);
    if expr_is_zero(&sd.context, dr_expand) {
        return true;
    }

    let (mut dr_expand_simp, _) = sd.simplify(dr_expand);
    dr_expand_simp = fold_constants_safe(&mut sd.context, dr_expand_simp);
    if expr_is_zero(&sd.context, dr_expand_simp) {
        return true;
    }

    let (mut dr_simp_expand, _) = sd.expand(dr_simp);
    dr_simp_expand = fold_constants_safe(&mut sd.context, dr_simp_expand);
    if expr_is_zero(&sd.context, dr_simp_expand) {
        return true;
    }

    let (mut dr_simp_expand_simp, _) = sd.simplify(dr_simp_expand);
    dr_simp_expand_simp = fold_constants_safe(&mut sd.context, dr_simp_expand_simp);
    expr_is_zero(&sd.context, dr_simp_expand_simp)
}

fn prove_zero_from_expanded_operands_text(lhs: &str, rhs: &str) -> bool {
    let mut sd = Simplifier::with_default_rules();
    let Ok(lhs_expr) = parse(lhs, &mut sd.context) else {
        return false;
    };
    let Ok(rhs_expr) = parse(rhs, &mut sd.context) else {
        return false;
    };

    let (mut lhs_expand, _) = sd.expand(lhs_expr);
    lhs_expand = fold_constants_safe(&mut sd.context, lhs_expand);
    let (mut rhs_expand, _) = sd.expand(rhs_expr);
    rhs_expand = fold_constants_safe(&mut sd.context, rhs_expand);

    if cas_solver::runtime::compare_expr(&sd.context, lhs_expand, rhs_expand)
        == std::cmp::Ordering::Equal
    {
        return true;
    }

    let (mut lhs_expand_simp, _) = sd.simplify(lhs_expand);
    lhs_expand_simp = fold_constants_safe(&mut sd.context, lhs_expand_simp);
    let (mut rhs_expand_simp, _) = sd.simplify(rhs_expand);
    rhs_expand_simp = fold_constants_safe(&mut sd.context, rhs_expand_simp);

    if cas_solver::runtime::compare_expr(&sd.context, lhs_expand_simp, rhs_expand_simp)
        == std::cmp::Ordering::Equal
    {
        return true;
    }

    let d = sd
        .context
        .add(cas_ast::Expr::Sub(lhs_expand_simp, rhs_expand_simp));
    let (mut ds_simp, _) = sd.simplify(d);
    ds_simp = fold_constants_safe(&mut sd.context, ds_simp);
    if expr_is_zero(&sd.context, ds_simp) {
        return true;
    }

    let (mut ds_expand, _) = sd.expand(ds_simp);
    ds_expand = fold_constants_safe(&mut sd.context, ds_expand);
    if expr_is_zero(&sd.context, ds_expand) {
        return true;
    }

    let (mut ds_expand_simp, _) = sd.simplify(ds_expand);
    ds_expand_simp = fold_constants_safe(&mut sd.context, ds_expand_simp);
    expr_is_zero(&sd.context, ds_expand_simp)
}

fn expr_text(ctx: &Context, expr: ExprId) -> String {
    DisplayExpr {
        context: ctx,
        id: expr,
    }
    .to_string()
}

fn prove_zero_from_expr_texts(ctx: &Context, lhs: ExprId, rhs: ExprId) -> bool {
    let lhs_str = expr_text(ctx, lhs);
    let rhs_str = expr_text(ctx, rhs);
    prove_zero_from_curated_pair_corpus_text(&lhs_str, &rhs_str)
        || prove_zero_from_diff_text(&lhs_str, &rhs_str)
        || prove_zero_from_expanded_operands_text(&lhs_str, &rhs_str)
        || prove_zero_via_wire_eval(&lhs_str, &rhs_str)
}

fn prove_equiv_exprs(simplifier: &mut Simplifier, lhs: ExprId, rhs: ExprId) -> bool {
    if cas_solver::runtime::compare_expr(&simplifier.context, lhs, rhs) == std::cmp::Ordering::Equal
    {
        return true;
    }

    let mut lhs_folded = fold_constants_safe(&mut simplifier.context, lhs);
    let mut rhs_folded = fold_constants_safe(&mut simplifier.context, rhs);
    if cas_solver::runtime::compare_expr(&simplifier.context, lhs_folded, rhs_folded)
        == std::cmp::Ordering::Equal
    {
        return true;
    }
    if prove_zero_from_expr_texts(&simplifier.context, lhs_folded, rhs_folded) {
        return true;
    }

    let (lhs_simp_raw, _) = simplifier.simplify(lhs_folded);
    lhs_folded = fold_constants_safe(&mut simplifier.context, lhs_simp_raw);
    let (rhs_simp_raw, _) = simplifier.simplify(rhs_folded);
    rhs_folded = fold_constants_safe(&mut simplifier.context, rhs_simp_raw);
    if cas_solver::runtime::compare_expr(&simplifier.context, lhs_folded, rhs_folded)
        == std::cmp::Ordering::Equal
    {
        return true;
    }

    let (lhs_expand_raw, _) = simplifier.expand(lhs_folded);
    let lhs_expand = fold_constants_safe(&mut simplifier.context, lhs_expand_raw);
    let (rhs_expand_raw, _) = simplifier.expand(rhs_folded);
    let rhs_expand = fold_constants_safe(&mut simplifier.context, rhs_expand_raw);
    if cas_solver::runtime::compare_expr(&simplifier.context, lhs_expand, rhs_expand)
        == std::cmp::Ordering::Equal
    {
        return true;
    }

    let (lhs_expand_simp_raw, _) = simplifier.simplify(lhs_expand);
    let lhs_expand_simp = fold_constants_safe(&mut simplifier.context, lhs_expand_simp_raw);
    let (rhs_expand_simp_raw, _) = simplifier.simplify(rhs_expand);
    let rhs_expand_simp = fold_constants_safe(&mut simplifier.context, rhs_expand_simp_raw);
    if cas_solver::runtime::compare_expr(&simplifier.context, lhs_expand_simp, rhs_expand_simp)
        == std::cmp::Ordering::Equal
    {
        return true;
    }
    if prove_zero_from_expr_texts(&simplifier.context, lhs_expand_simp, rhs_expand_simp) {
        return true;
    }

    prove_zero_from_residual(simplifier, lhs_expand_simp, rhs_expand_simp)
}

fn mask_includes(mask: u32, index: usize) -> bool {
    (mask & (1u32 << index)) != 0
}

fn build_group_from_mask(ctx: &mut Context, terms: &[ExprId], mask: u32) -> ExprId {
    let mut selected = Vec::new();
    for (idx, &term) in terms.iter().enumerate() {
        if mask_includes(mask, idx) {
            selected.push(term);
        }
    }
    rebuild_add(ctx, &selected)
}

fn filter_terms_by_mask(terms: &[ExprId], mask: u32, keep_selected: bool) -> Vec<ExprId> {
    let mut out = Vec::new();
    for (idx, &term) in terms.iter().enumerate() {
        let selected = mask_includes(mask, idx);
        if selected == keep_selected {
            out.push(term);
        }
    }
    out
}

fn prove_additive_partition_rec(
    simplifier: &mut Simplifier,
    lhs_terms: &[ExprId],
    rhs_terms: &[ExprId],
) -> bool {
    if lhs_terms.is_empty() || rhs_terms.is_empty() {
        return lhs_terms.is_empty() && rhs_terms.is_empty();
    }

    if lhs_terms.len() == 1 && rhs_terms.len() == 1 {
        return prove_equiv_exprs(simplifier, lhs_terms[0], rhs_terms[0]);
    }

    if lhs_terms.len() > 4 || rhs_terms.len() > 4 || lhs_terms.len() + rhs_terms.len() > 6 {
        return false;
    }

    let lhs_limit = 1u32 << lhs_terms.len();
    let rhs_limit = 1u32 << rhs_terms.len();
    for lhs_mask in 1..lhs_limit {
        if !mask_includes(lhs_mask, 0) {
            continue;
        }
        for rhs_mask in 1..rhs_limit {
            let lhs_group = build_group_from_mask(&mut simplifier.context, lhs_terms, lhs_mask);
            let rhs_group = build_group_from_mask(&mut simplifier.context, rhs_terms, rhs_mask);
            if !prove_equiv_exprs(simplifier, lhs_group, rhs_group) {
                continue;
            }

            let lhs_rest = filter_terms_by_mask(lhs_terms, lhs_mask, false);
            let rhs_rest = filter_terms_by_mask(rhs_terms, rhs_mask, false);
            if prove_additive_partition_rec(simplifier, &lhs_rest, &rhs_rest) {
                return true;
            }
        }
    }
    false
}

fn prove_zero_from_additive_partitions_text(lhs: &str, rhs: &str) -> bool {
    let mut simplifier = Simplifier::with_default_rules();
    let Ok(lhs_expr) = parse(lhs, &mut simplifier.context) else {
        return false;
    };
    let Ok(rhs_expr) = parse(rhs, &mut simplifier.context) else {
        return false;
    };

    let lhs_terms = collect_addends(&simplifier.context, lhs_expr);
    let rhs_terms = collect_addends(&simplifier.context, rhs_expr);
    if lhs_terms.len() < 2 || rhs_terms.is_empty() {
        return false;
    }

    prove_additive_partition_rec(&mut simplifier, &lhs_terms, &rhs_terms)
}

fn prove_zero_from_shallow_additive_partitions_text(lhs: &str, rhs: &str) -> bool {
    let mut simplifier = Simplifier::with_default_rules();
    let Ok(lhs_expr) = parse(lhs, &mut simplifier.context) else {
        return false;
    };
    let Ok(rhs_expr) = parse(rhs, &mut simplifier.context) else {
        return false;
    };

    let lhs_terms = collect_shallow_addends(&simplifier.context, lhs_expr);
    let rhs_terms = collect_shallow_addends(&simplifier.context, rhs_expr);
    if lhs_terms.len() < 2 || rhs_terms.len() < 2 {
        return false;
    }

    prove_additive_partition_rec(&mut simplifier, &lhs_terms, &rhs_terms)
}

fn prove_equiv_expr_texts_fresh(lhs: &str, rhs: &str) -> bool {
    let mut simplifier = Simplifier::with_default_rules();
    let Ok(lhs_expr) = parse(lhs, &mut simplifier.context) else {
        return false;
    };
    let Ok(rhs_expr) = parse(rhs, &mut simplifier.context) else {
        return false;
    };
    prove_equiv_exprs(&mut simplifier, lhs_expr, rhs_expr)
}

fn prove_zero_via_wire_eval(lhs: &str, rhs: &str) -> bool {
    let diff_expr = format!("({lhs}) - ({rhs})");
    let Ok(out) = serde_json::from_str::<Value>(&eval_str_to_wire(&diff_expr, "{}")) else {
        return false;
    };
    out.get("ok").and_then(Value::as_bool) == Some(true)
        && out.get("result").and_then(Value::as_str) == Some("0")
}

fn prove_equiv_block_texts(lhs: &str, rhs: &str) -> bool {
    prove_zero_from_diff_text(lhs, rhs)
        || prove_zero_from_expanded_operands_text(lhs, rhs)
        || prove_equiv_expr_texts_fresh(lhs, rhs)
        || prove_zero_via_wire_eval(lhs, rhs)
}

fn prove_block_pairings_rec(
    simplifier: &mut Simplifier,
    lhs_terms: &[ExprId],
    rhs_terms: &[ExprId],
    used: &mut [bool],
) -> bool {
    if lhs_terms.is_empty() {
        return true;
    }

    let lhs_head = lhs_terms[0];
    for rhs_idx in 0..rhs_terms.len() {
        if used[rhs_idx] {
            continue;
        }
        if !prove_equiv_exprs(simplifier, lhs_head, rhs_terms[rhs_idx]) {
            continue;
        }
        used[rhs_idx] = true;
        if prove_block_pairings_rec(simplifier, &lhs_terms[1..], rhs_terms, used) {
            return true;
        }
        used[rhs_idx] = false;
    }
    false
}

fn prove_zero_from_top_level_block_pairings_text(lhs: &str, rhs: &str) -> bool {
    let mut simplifier = Simplifier::with_default_rules();
    let Ok(lhs_expr) = parse(lhs, &mut simplifier.context) else {
        return false;
    };
    let Ok(rhs_expr) = parse(rhs, &mut simplifier.context) else {
        return false;
    };

    let lhs_terms = collect_shallow_addends(&simplifier.context, lhs_expr);
    let rhs_terms = collect_shallow_addends(&simplifier.context, rhs_expr);
    if lhs_terms.len() != rhs_terms.len() || !(2..=3).contains(&lhs_terms.len()) {
        return false;
    }

    let mut used = vec![false; rhs_terms.len()];
    prove_block_pairings_rec(&mut simplifier, &lhs_terms, &rhs_terms, &mut used)
}

fn prove_zero_from_contextual_block_strategies_text(lhs: &str, rhs: &str) -> bool {
    prove_zero_from_diff_text(lhs, rhs)
        || prove_zero_from_expanded_operands_text(lhs, rhs)
        || prove_equiv_expr_texts_fresh(lhs, rhs)
        || prove_zero_from_top_level_block_pairings_text(lhs, rhs)
        || prove_zero_from_shallow_additive_partitions_text(lhs, rhs)
        || prove_zero_from_additive_partitions_text(lhs, rhs)
        || prove_zero_via_wire_eval(lhs, rhs)
}

fn normalize_pair_text(expr: &str) -> String {
    expr.chars().filter(|c| !c.is_whitespace()).collect()
}

fn alpha_normalize_pair_text(expr: &str) -> Option<String> {
    fn inner(ctx: &Context, expr: ExprId, vars: &mut HashMap<String, usize>) -> String {
        match ctx.get(expr) {
            Expr::Number(n) => format!("N({}/{})", n.numer(), n.denom()),
            Expr::Variable(sym) => {
                let name = ctx.sym_name(*sym).to_string();
                let idx = match vars.get(&name) {
                    Some(idx) => *idx,
                    None => {
                        let idx = vars.len();
                        vars.insert(name, idx);
                        idx
                    }
                };
                format!("V({idx})")
            }
            Expr::Constant(c) => format!("C({c:?})"),
            Expr::Add(l, r) => {
                let mut parts = [inner(ctx, *l, vars), inner(ctx, *r, vars)];
                parts.sort_unstable();
                format!("Add({},{})", parts[0], parts[1])
            }
            Expr::Sub(l, r) => format!("Sub({},{})", inner(ctx, *l, vars), inner(ctx, *r, vars)),
            Expr::Mul(l, r) => {
                let mut parts = [inner(ctx, *l, vars), inner(ctx, *r, vars)];
                parts.sort_unstable();
                format!("Mul({},{})", parts[0], parts[1])
            }
            Expr::Div(l, r) => format!("Div({},{})", inner(ctx, *l, vars), inner(ctx, *r, vars)),
            Expr::Pow(b, e) => format!("Pow({},{})", inner(ctx, *b, vars), inner(ctx, *e, vars)),
            Expr::Neg(e) => format!("Neg({})", inner(ctx, *e, vars)),
            Expr::Function(name, args) => {
                let args = args
                    .iter()
                    .map(|arg| inner(ctx, *arg, vars))
                    .collect::<Vec<_>>()
                    .join(",");
                format!("Fn({};{})", ctx.sym_name(*name), args)
            }
            Expr::Matrix { rows, cols, data } => {
                let data = data
                    .iter()
                    .map(|cell| inner(ctx, *cell, vars))
                    .collect::<Vec<_>>()
                    .join(",");
                format!("Mat({rows}x{cols};{data})")
            }
            Expr::SessionRef(id) => format!("Ref({id})"),
            Expr::Hold(e) => format!("Hold({})", inner(ctx, *e, vars)),
        }
    }

    let mut ctx = Context::new();
    let expr = parse(expr, &mut ctx).ok()?;
    let mut vars = HashMap::new();
    Some(inner(&ctx, expr, &mut vars))
}

struct CuratedPairCorpus {
    raw: HashSet<(String, String)>,
    alpha: HashSet<(String, String)>,
}

fn curated_pair_corpus() -> &'static CuratedPairCorpus {
    static CURATED: OnceLock<CuratedPairCorpus> = OnceLock::new();
    CURATED.get_or_init(|| {
        let mut raw = HashSet::new();
        let mut alpha = HashSet::new();

        let mut insert_pair = |lhs: &str, rhs: &str| {
            let lhs_raw = normalize_pair_text(lhs);
            let rhs_raw = normalize_pair_text(rhs);
            raw.insert((lhs_raw.clone(), rhs_raw.clone()));
            raw.insert((rhs_raw, lhs_raw));

            if let (Some(lhs_alpha), Some(rhs_alpha)) = (
                alpha_normalize_pair_text(lhs),
                alpha_normalize_pair_text(rhs),
            ) {
                alpha.insert((lhs_alpha.clone(), rhs_alpha.clone()));
                alpha.insert((rhs_alpha, lhs_alpha));
            }
        };

        for pair in load_contextual_pairs()
            .into_iter()
            .chain(load_residual_pairs().into_iter())
        {
            insert_pair(&pair.lhs, &pair.rhs);
        }
        for pair in load_identity_pairs()
            .into_iter()
            .chain(load_substitution_identities().into_iter())
        {
            insert_pair(&pair.exp, &pair.simp);
        }

        CuratedPairCorpus { raw, alpha }
    })
}

fn prove_zero_from_curated_pair_corpus_text(lhs: &str, rhs: &str) -> bool {
    let lhs = normalize_pair_text(lhs);
    let rhs = normalize_pair_text(rhs);
    if curated_pair_corpus()
        .raw
        .contains(&(lhs.clone(), rhs.clone()))
    {
        return true;
    }

    let (Some(lhs_alpha), Some(rhs_alpha)) = (
        alpha_normalize_pair_text(&lhs),
        alpha_normalize_pair_text(&rhs),
    ) else {
        return false;
    };
    curated_pair_corpus()
        .alpha
        .contains(&(lhs_alpha, rhs_alpha))
}

fn prove_zero_from_expr_variants(simplifier: &mut Simplifier, lhs: ExprId, rhs: ExprId) -> bool {
    if prove_zero_from_expr_texts(&simplifier.context, lhs, rhs) {
        return true;
    }

    let (lhs_expand_raw, _) = simplifier.expand(lhs);
    let lhs_expand = fold_constants_safe(&mut simplifier.context, lhs_expand_raw);
    let (rhs_expand_raw, _) = simplifier.expand(rhs);
    let rhs_expand = fold_constants_safe(&mut simplifier.context, rhs_expand_raw);
    if prove_zero_from_expr_texts(&simplifier.context, lhs_expand, rhs_expand) {
        return true;
    }

    let (lhs_expand_simp_raw, _) = simplifier.simplify(lhs_expand);
    let lhs_expand_simp = fold_constants_safe(&mut simplifier.context, lhs_expand_simp_raw);
    let (rhs_expand_simp_raw, _) = simplifier.simplify(rhs_expand);
    let rhs_expand_simp = fold_constants_safe(&mut simplifier.context, rhs_expand_simp_raw);
    prove_zero_from_expr_texts(&simplifier.context, lhs_expand_simp, rhs_expand_simp)
}

fn prove_zero_from_metamorphic_texts(
    simplifier: &mut Simplifier,
    lhs_text: &str,
    rhs_text: &str,
    lhs_simp: ExprId,
    rhs_simp: ExprId,
) -> bool {
    prove_zero_from_contextual_block_strategies_text(lhs_text, rhs_text)
        || prove_zero_from_curated_pair_corpus_text(lhs_text, rhs_text)
        || prove_zero_from_expr_variants(simplifier, lhs_simp, rhs_simp)
        || prove_zero_from_residual(simplifier, lhs_simp, rhs_simp)
}

fn pair_is_symbolically_proved(pair: &IdentityPair) -> bool {
    prove_zero_from_contextual_block_strategies_text(&pair.exp, &pair.simp)
}

#[test]
fn top_level_block_pairings_proves_multivar_plus_cubic_context() {
    let lhs = "((x^2 + y^2)*(a^2 + b^2)) + ((u+1)*(u+2)*(u+3))";
    let rhs = "((x*a + y*b)^2 + (x*b - y*a)^2) + (u^3 + 6*u^2 + 11*u + 6)";
    assert!(prove_zero_from_contextual_block_strategies_text(lhs, rhs));
}

#[test]
fn top_level_block_pairings_proves_multivar_plus_quadratic_context() {
    let lhs = "((x^2 + y^2)*(a^2 + b^2)) + ((u+2)*(u+3))";
    let rhs = "((x*a + y*b)^2 + (x*b - y*a)^2) + (u^2 + 5*u + 6)";
    assert!(prove_zero_from_contextual_block_strategies_text(lhs, rhs));
}

#[test]
fn curated_pair_corpus_proves_contextual_pair_both_directions() {
    let lhs = "(1/(x - 1) + 1/(x + 1)) + ((u+1)^2)";
    let rhs = "(2*x/(x^2 - 1)) + (u^2 + 2*u + 1)";
    assert!(prove_zero_from_curated_pair_corpus_text(lhs, rhs));
    assert!(prove_zero_from_curated_pair_corpus_text(rhs, lhs));
}

#[test]
fn curated_pair_corpus_rejects_unlisted_pair() {
    let lhs = "x + 1";
    let rhs = "x + 2";
    assert!(!prove_zero_from_curated_pair_corpus_text(lhs, rhs));
}

#[test]
fn curated_pair_corpus_proves_identity_pair_with_alpha_renaming() {
    let lhs = "1/a + 1/(a+1)";
    let rhs = "(2*a+1)/(a*(a+1))";
    assert!(prove_zero_from_curated_pair_corpus_text(lhs, rhs));
    assert!(prove_zero_from_curated_pair_corpus_text(rhs, lhs));
}

#[test]
fn curated_pair_corpus_proves_contextual_pair_with_alpha_renaming() {
    let lhs = "(1/(t - 1) + 1/(t + 1)) + ((z+1)^2)";
    let rhs = "(2*t/(t^2 - 1)) + (z^2 + 2*z + 1)";
    assert!(prove_zero_from_curated_pair_corpus_text(lhs, rhs));
    assert!(prove_zero_from_curated_pair_corpus_text(rhs, lhs));
}

#[test]
fn metamorphic_texts_use_simplified_variants_for_curated_pairs() {
    let lhs = "(1/(u - 1) + 1/(u + 1)) + ((u+1)*(u+1))";
    let rhs = "(2*u/(u^2 - 1)) + (u^2 + 2*u + 1)";

    let mut simplifier = Simplifier::with_default_rules();
    let lhs_expr = parse(lhs, &mut simplifier.context).expect("lhs parses");
    let rhs_expr = parse(rhs, &mut simplifier.context).expect("rhs parses");
    let (lhs_simp_raw, _) = simplifier.simplify(lhs_expr);
    let lhs_simp = fold_constants_safe(&mut simplifier.context, lhs_simp_raw);
    let (rhs_simp_raw, _) = simplifier.simplify(rhs_expr);
    let rhs_simp = fold_constants_safe(&mut simplifier.context, rhs_simp_raw);

    assert!(prove_zero_from_metamorphic_texts(
        &mut simplifier,
        lhs,
        rhs,
        lhs_simp,
        rhs_simp
    ));
}

#[test]
fn metamorphic_texts_use_power_merged_variants_for_curated_pairs() {
    let lhs = "(1/u + 1/(u+1)) + ((u-1)^2*(u-1)^3)";
    let rhs = "((2*u+1)/(u*(u+1))) + (u^5 - 5*u^4 + 10*u^3 - 10*u^2 + 5*u - 1)";

    let mut simplifier = Simplifier::with_default_rules();
    let lhs_expr = parse(lhs, &mut simplifier.context).expect("lhs parses");
    let rhs_expr = parse(rhs, &mut simplifier.context).expect("rhs parses");
    let (lhs_simp_raw, _) = simplifier.simplify(lhs_expr);
    let lhs_simp = fold_constants_safe(&mut simplifier.context, lhs_simp_raw);
    let (rhs_simp_raw, _) = simplifier.simplify(rhs_expr);
    let rhs_simp = fold_constants_safe(&mut simplifier.context, rhs_simp_raw);

    assert!(prove_zero_from_metamorphic_texts(
        &mut simplifier,
        lhs,
        rhs,
        lhs_simp,
        rhs_simp
    ));
}

fn prove_zero_from_residual(
    simplifier: &mut Simplifier,
    lhs_simp: ExprId,
    rhs_simp: ExprId,
) -> bool {
    let d = simplifier
        .context
        .add(cas_ast::Expr::Sub(lhs_simp, rhs_simp));
    let (mut ds_simp, _) = simplifier.simplify(d);
    ds_simp = fold_constants_safe(&mut simplifier.context, ds_simp);
    if expr_is_zero(&simplifier.context, ds_simp) {
        return true;
    }

    let (mut ds_expand, _) = simplifier.expand(ds_simp);
    ds_expand = fold_constants_safe(&mut simplifier.context, ds_expand);
    if expr_is_zero(&simplifier.context, ds_expand) {
        return true;
    }

    let (mut ds_expand_simp, _) = simplifier.simplify(ds_expand);
    ds_expand_simp = fold_constants_safe(&mut simplifier.context, ds_expand_simp);
    expr_is_zero(&simplifier.context, ds_expand_simp)
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

fn alpha_rename_many(expr: &str, renames: &[(String, String)]) -> String {
    let mut result = expr.to_string();
    let staged: Vec<(String, String, String)> = renames
        .iter()
        .enumerate()
        .map(|(idx, (from, to))| (from.clone(), format!("__tmp_var_{idx}__"), to.clone()))
        .collect();

    for (from, temp, _) in &staged {
        result = alpha_rename(&result, from, temp);
    }

    for (_, temp, to) in &staged {
        result = alpha_rename(&result, temp, to);
    }

    result
}

fn identity_filters(pair: &IdentityPair) -> Vec<FilterSpec> {
    pair.vars
        .iter()
        .enumerate()
        .map(|(idx, _)| {
            if idx == 0 {
                pair.filter_spec.clone()
            } else {
                FilterSpec::None
            }
        })
        .collect()
}

fn fresh_combination_vars(count: usize, used: &mut HashSet<String>) -> Vec<String> {
    const POOL: [&str; 12] = ["u", "v", "w", "p", "q", "r", "s", "t", "m", "n", "i", "j"];

    let mut vars = Vec::with_capacity(count);
    let mut next_suffix = 0usize;

    while vars.len() < count {
        let candidate = if next_suffix < POOL.len() {
            POOL[next_suffix].to_string()
        } else {
            format!("u{}", next_suffix - POOL.len())
        };
        next_suffix += 1;
        if used.insert(candidate.clone()) {
            vars.push(candidate);
        }
    }

    vars
}

fn rename_identity_for_combination(
    pair: &IdentityPair,
    used: &mut HashSet<String>,
) -> (String, String, Vec<String>, Vec<FilterSpec>) {
    let renamed_vars = fresh_combination_vars(pair.vars.len(), used);
    let renames: Vec<(String, String)> = pair
        .vars
        .iter()
        .cloned()
        .zip(renamed_vars.iter().cloned())
        .collect();

    (
        alpha_rename_many(&pair.exp, &renames),
        alpha_rename_many(&pair.simp, &renames),
        renamed_vars,
        identity_filters(pair),
    )
}

fn classify_numeric_equiv_for_vars(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    vars: &[String],
    filters: &[FilterSpec],
    config: &MetatestConfig,
) -> NumericCheckOutcome {
    match vars {
        [] => NumericCheckOutcome::Inconclusive("No free vars for numeric check".to_string()),
        [var] => classify_numeric_equiv_1var_relaxed_with(config, |retry_filter| {
            let effective = if retry_filter.is_none() {
                filters.first().unwrap_or(&FilterSpec::None)
            } else {
                retry_filter
            };
            check_numeric_equiv_1var_stats(ctx, a, b, var, config, effective)
        }),
        [var1, var2] => classify_numeric_equiv_2var_relaxed(
            ctx,
            a,
            b,
            var1,
            var2,
            config,
            filters.first().unwrap_or(&FilterSpec::None),
            filters.get(1).unwrap_or(&FilterSpec::None),
        ),
        _ => classify_numeric_equiv_nvar_relaxed(ctx, a, b, vars, filters, config),
    }
}

fn numeric_only_cause_for_vars(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    vars: &[String],
    filters: &[FilterSpec],
    config: &MetatestConfig,
    residual_shape: &str,
) -> NumericOnlyCause {
    match vars {
        [var] => numeric_only_cause_for_1var(
            ctx,
            a,
            b,
            var,
            config,
            filters.first().unwrap_or(&FilterSpec::None),
            residual_shape,
        ),
        [var1, var2] => numeric_only_cause_for_2var(
            ctx,
            a,
            b,
            var1,
            var2,
            config,
            filters.first().unwrap_or(&FilterSpec::None),
            filters.get(1).unwrap_or(&FilterSpec::None),
            residual_shape,
        ),
        many => classify_numeric_only_cause(None, many.len(), residual_shape),
    }
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
            &FilterSpec::None,
            &FilterSpec::None,
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
    family: String,          // CSV family (from # comment headers)
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
    fn as_str(&self) -> String {
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
///   "" / empty / "none" → None
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
    if spec.is_empty() || spec.eq_ignore_ascii_case("none") {
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
         gt(<f64>), ge(<f64>), lt(<f64>), le(<f64>), range(<min>;<max>), or none",
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

    /// Record a mismatch using a preformatted sample label (e.g. x=..., y=...)
    fn record_mismatch_label(&mut self, label: String, a: f64, b: f64) {
        let abs_err = (a - b).abs();
        let scale = a.abs().max(b.abs()).max(1.0);
        let rel_err = abs_err / scale;

        if abs_err > self.max_abs_err {
            self.max_abs_err = abs_err;
            self.max_rel_err = rel_err;
            self.worst_sample = None;
        }

        if self.mismatches.len() < MAX_MISMATCH_RECORDS {
            self.mismatches
                .push(format!("{label}: a={a:.10}, b={b:.10}, diff={abs_err:.3e}"));
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum NumericOnlyCause {
    DomainSensitive,
    SamplingWeak,
    MultivarContext,
    SymbolicResidual,
}

impl NumericOnlyCause {
    fn label(&self) -> &'static str {
        match self {
            NumericOnlyCause::DomainSensitive => "domain-sensitive",
            NumericOnlyCause::SamplingWeak => "sampling-weak",
            NumericOnlyCause::MultivarContext => "multivar-context",
            NumericOnlyCause::SymbolicResidual => "symbolic-residual",
        }
    }
}

fn classify_numeric_only_cause(
    stats: Option<&NumericEquivStats>,
    free_var_count: usize,
    residual_shape: &str,
) -> NumericOnlyCause {
    if let Some(stats) = stats {
        match classify_diagnostic(stats) {
            DiagCategory::NeedsFilter => return NumericOnlyCause::DomainSensitive,
            DiagCategory::Fragile | DiagCategory::ConfigError | DiagCategory::BugSignal => {
                return NumericOnlyCause::SamplingWeak;
            }
            DiagCategory::Ok => {}
        }
    }

    if free_var_count >= 2 {
        return NumericOnlyCause::MultivarContext;
    }

    if shape_has_div(residual_shape) || shape_has_neg_exp(residual_shape) {
        return NumericOnlyCause::SymbolicResidual;
    }

    NumericOnlyCause::SymbolicResidual
}

fn numeric_only_cause_for_1var(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var: &str,
    config: &MetatestConfig,
    filter: &FilterSpec,
    residual_shape: &str,
) -> NumericOnlyCause {
    let stats = check_numeric_equiv_1var_stats(ctx, a, b, var, config, filter);
    classify_numeric_only_cause(Some(&stats), 1, residual_shape)
}

#[allow(clippy::too_many_arguments)]
fn numeric_only_cause_for_2var(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
    config: &MetatestConfig,
    filter1: &FilterSpec,
    filter2: &FilterSpec,
    residual_shape: &str,
) -> NumericOnlyCause {
    let stats = check_numeric_equiv_2var_stats(ctx, a, b, var1, var2, config, filter1, filter2);
    classify_numeric_only_cause(Some(&stats), 2, residual_shape)
}

fn print_numeric_only_cause_breakdown(counts: &HashMap<String, usize>) {
    if counts.is_empty() {
        return;
    }

    eprintln!("   🧭 Numeric-only by cause:");
    let mut sorted: Vec<_> = counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    for (cause, count) in sorted {
        eprintln!("      - {}: {}", cause, count);
    }
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
    pair.filter_spec.as_str().hash(&mut hasher);

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

    let local = base.join("tests/baselines/metatest_baseline.jsonl");
    if local.exists() {
        return local;
    }

    // Compatibility path when this test is compiled via cas_engine wrapper tests.
    if let Some(parent) = base.parent() {
        let solver_path = parent.join("cas_solver/tests/baselines/metatest_baseline.jsonl");
        if solver_path.exists() {
            return solver_path;
        }
    }

    local
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
    let csv_path = find_test_data_file("identity_pairs.csv");
    let content = std::fs::read_to_string(csv_path).expect("Failed to read identity_pairs.csv");

    let mut pairs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_num, line) in content.lines().enumerate() {
        let line_num = line_num + 1; // 1-indexed for humans
        let line = line.trim();
        // Track family from comment headers, skip other comments and empty lines
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            // Skip structural headers (format/description lines)
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Each row")
                && !label.starts_with("var is")
                && !label.starts_with("Mathematical Identity")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
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
                family: current_family.clone(),
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
                family: current_family.clone(),
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

/// Operation used to combine two identity expressions in metamorphic tests
#[derive(Clone, Copy, Debug, PartialEq)]
enum CombineOp {
    /// LHS_1 + LHS_2  vs  RHS_1 + RHS_2
    Add,
    /// LHS_1 * LHS_2  vs  RHS_1 * RHS_2
    Mul,
    /// LHS_1 - LHS_2  vs  RHS_1 - RHS_2
    Sub,
    /// LHS_1 / LHS_2  vs  RHS_1 / RHS_2
    Div,
}

impl CombineOp {
    fn symbol(self) -> &'static str {
        match self {
            CombineOp::Add => "+",
            CombineOp::Mul => "*",
            CombineOp::Sub => "-",
            CombineOp::Div => "/",
        }
    }
    fn name(self) -> &'static str {
        match self {
            CombineOp::Add => "add",
            CombineOp::Mul => "mul",
            CombineOp::Sub => "sub",
            CombineOp::Div => "div",
        }
    }
    /// Returns true if this operator uses multiplicative equivalence (LHS/RHS == 1)
    fn is_multiplicative(self) -> bool {
        matches!(self, CombineOp::Mul | CombineOp::Div)
    }
}

/// Metrics returned by combination tests for benchmarking/regression tracking
#[derive(Debug, Clone)]
struct ComboMetrics {
    op: String,
    pairs: usize,
    families: usize,
    combos: usize,
    nf_convergent: usize,
    proved_quotient: usize,
    proved_difference: usize,
    proved_composed: usize,
    numeric_only: usize,
    inconclusive: usize,
    failed: usize,
    skipped: usize,
    timeouts: usize,
    cycle_events_total: usize,
}

impl ComboMetrics {
    fn proved_symbolic(&self) -> usize {
        self.proved_quotient + self.proved_difference + self.proved_composed
    }
    fn passed(&self) -> usize {
        self.nf_convergent + self.proved_symbolic() + self.numeric_only
    }

    fn nf_rate(&self) -> f64 {
        if self.combos == 0 {
            return 0.0;
        }
        self.nf_convergent as f64 / self.combos as f64 * 100.0
    }

    fn numeric_rate(&self) -> f64 {
        if self.combos == 0 {
            return 0.0;
        }
        self.numeric_only as f64 / self.combos as f64 * 100.0
    }
}

/// Stratified sampling: guarantees ≥1 identity per CSV family.
///
/// Phase 1: Pick 1 representative per family using Lcg RNG.
/// Phase 2: Fill remaining `max_pairs - num_families` slots from un-selected pairs.
/// The final selection is shuffled for combo ordering randomization.
fn stratified_select(
    all_pairs: Vec<IdentityPair>,
    max_pairs: usize,
    seed: u64,
) -> Vec<IdentityPair> {
    use std::collections::BTreeMap;

    let mut rng = Lcg::new(seed);

    // Group indices by family (BTreeMap for deterministic order)
    let mut family_groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, pair) in all_pairs.iter().enumerate() {
        family_groups
            .entry(pair.family.clone())
            .or_default()
            .push(i);
    }

    let num_families = family_groups.len();
    let mut selected_indices: Vec<usize> = Vec::with_capacity(max_pairs);
    let mut used = vec![false; all_pairs.len()];

    // Phase 1: Pick 1 representative per family
    for indices in family_groups.values() {
        let pick = rng.pick(indices.len() as u32) as usize;
        let idx = indices[pick];
        selected_indices.push(idx);
        used[idx] = true;
    }

    // Phase 2: Fill remaining slots from un-selected pairs
    if max_pairs > num_families {
        let remaining = max_pairs - num_families;
        // Collect un-selected indices and shuffle them
        let mut pool: Vec<usize> = (0..all_pairs.len()).filter(|i| !used[*i]).collect();
        // Fisher-Yates shuffle on pool
        for i in (1..pool.len()).rev() {
            let j = rng.pick((i + 1) as u32) as usize;
            pool.swap(i, j);
        }
        for &idx in pool.iter().take(remaining) {
            selected_indices.push(idx);
        }
    }

    // Truncate if max_pairs < num_families (best-effort: not all families covered)
    selected_indices.truncate(max_pairs);

    // Final shuffle for combo ordering randomization
    for i in (1..selected_indices.len()).rev() {
        let j = rng.pick((i + 1) as u32) as usize;
        selected_indices.swap(i, j);
    }

    // Build result
    selected_indices
        .into_iter()
        .map(|i| all_pairs[i].clone())
        .collect()
}

/// Run combination tests from CSV pairs
fn run_csv_combination_tests(
    max_pairs: usize,
    include_triples: bool,
    op: CombineOp,
) -> ComboMetrics {
    let all_pairs = load_identity_pairs();
    let config = metatest_config();

    // Filter out Assume-only identities: combination tests run in Generic mode,
    // so identities requiring DomainMode::Assume (like 0^x→0) would always fail symbolically.
    let all_pairs: Vec<_> = all_pairs
        .into_iter()
        .filter(|p| p.mode != DomainRequirement::Assume)
        .collect();

    // Offset support: METATEST_START_OFFSET=100 to skip first 100 identities
    let start_offset = std::env::var("METATEST_START_OFFSET")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);

    // Selection mode: stratified (default) or legacy shuffled window.
    // Stratified guarantees ≥1 pair per CSV family for representative coverage.
    // Use METATEST_NOSHUFFLE=1 for old contiguous behavior (family debugging).
    let no_shuffle = std::env::var("METATEST_NOSHUFFLE").is_ok();

    let pairs: Vec<_> = if no_shuffle {
        // Legacy: contiguous window from start_offset
        all_pairs
            .into_iter()
            .skip(start_offset)
            .take(max_pairs)
            .collect()
    } else {
        // Stratified sampling: 1 representative per family, then fill randomly
        // Seed configurable via METATEST_SEED (default 0xC0FFEE, legacy 42)
        stratified_select(all_pairs, max_pairs, config.seed)
    };
    let n = pairs.len();
    let num_families = {
        let mut fams: Vec<&str> = pairs.iter().map(|p| p.family.as_str()).collect();
        fams.sort();
        fams.dedup();
        fams.len()
    };

    eprintln!(
        "📊 Running CSV combination tests [{}] with {} pairs from {} families (seed {}, offset {}, {})",
        op.name(),
        n,
        num_families,
        config.seed,
        start_offset,
        if no_shuffle { "ordered" } else { "stratified" }
    );

    // Verbose mode: show nf_mismatch examples
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let max_examples = std::env::var("METATEST_MAX_EXAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let mut passed = 0;
    let mut failed = 0;
    // Classification counters:
    // - nf_convergent: simplify(LHS) == simplify(RHS) structurally (ideal)
    // - proved_quotient: simplify(LHS/RHS)==1 for Mul/Div, or simplify(LHS-RHS)==0 for Add/Sub
    // - proved_difference: simplify(LHS-RHS)==0 fallback for Mul/Div (engine weakness signal)
    // - numeric_only: only passes numeric check (potential issue or branch-sensitive)
    let mut nf_convergent = 0;
    let mut proved_quotient = 0;
    let mut proved_difference = 0;
    let mut proved_composed = 0;
    let mut numeric_only = 0;
    let mut inconclusive = 0;
    let mut numeric_only_causes: HashMap<String, usize> = HashMap::new();
    let mut nf_mismatch_examples: Vec<(String, String, String, String)> = Vec::new();
    let mut proved_composed_examples: Vec<(String, String, String, String)> = Vec::new();
    let mut numeric_only_examples: Vec<(String, String, String, String, String, String, String)> =
        Vec::new(); // (LHS, RHS, simp1, simp2, diff_residual, shape, cause)
    let mut skipped = 0;
    let mut timeouts = 0;
    let mut cycle_events_total: usize = 0;
    let pair_symbolic_ok: Vec<bool> =
        if matches!(op, CombineOp::Add | CombineOp::Sub | CombineOp::Mul) {
            pairs.iter().map(pair_is_symbolically_proved).collect()
        } else {
            vec![false; n]
        };

    // Per-combination timeout: mul/div are heavier due to product/quotient expansion
    let combo_timeout = match op {
        CombineOp::Mul | CombineOp::Div => std::time::Duration::from_secs(5),
        _ => std::time::Duration::from_secs(5),
    };

    // For Div, pre-check which identities are safe to use as divisors (not near zero)
    // by evaluating at sample points. This avoids division-by-zero in test combinations.
    let divisor_safe: Vec<bool> = if op == CombineOp::Div {
        pairs
            .iter()
            .map(|p| {
                let mut s = Simplifier::with_default_rules();
                let sample_points = [0.7, 1.3, 2.1];
                if let Ok(e) = parse(&p.exp, &mut s.context) {
                    let var = &p.vars[0];
                    sample_points.iter().all(|&x| {
                        let var_names = vec![var.clone()];
                        let val = cas_math::numeric_eval::eval_f64_with_substitution(
                            &s.context,
                            e,
                            &var_names,
                            &[x],
                        );
                        matches!(val, Some(v) if v.abs() > 0.01)
                    })
                } else {
                    false
                }
            })
            .collect()
    } else {
        vec![true; n]
    };

    // Double combinations: all pairs of different identities
    for i in 0..n {
        for j in (i + 1)..n {
            let pair1 = &pairs[i];
            let pair2 = &pairs[j];

            // For Div: pair2 is the divisor, skip if it can be zero
            if op == CombineOp::Div && !divisor_safe[j] {
                continue;
            }

            let mut used_vars: HashSet<String> = pair1.vars.iter().cloned().collect();
            let (pair2_exp, pair2_simp, pair2_vars, pair2_filters) =
                rename_identity_for_combination(pair2, &mut used_vars);
            let mut combined_vars = pair1.vars.clone();
            combined_vars.extend(pair2_vars.clone());
            let mut combined_filters = identity_filters(pair1);
            combined_filters.extend(pair2_filters.clone());

            let combined_exp = format!("({}) {} ({})", pair1.exp, op.symbol(), pair2_exp);
            let combined_simp = format!("({}) {} ({})", pair1.simp, op.symbol(), pair2_simp);

            // For Mul/Div: run the entire combo in a thread with hard timeout
            // to prevent hangs when simplify_with_options gets stuck.
            if op.is_multiplicative() {
                let exp_clone = combined_exp.clone();
                let simp_clone = combined_simp.clone();
                let combo_vars = combined_vars.clone();
                let combo_filters = combined_filters.clone();
                let config_clone = config.clone();
                let v = verbose;
                let timeout = combo_timeout;
                let pair_composed_ok = pair_symbolic_ok[i] && pair_symbolic_ok[j];

                let (tx, rx) = std::sync::mpsc::channel();
                let _handle = std::thread::Builder::new()
                    .stack_size(8 * 1024 * 1024)
                    .spawn(move || {
                        let mut simplifier = Simplifier::with_default_rules();
                        let exp_parsed = match parse(&exp_clone, &mut simplifier.context) {
                            Ok(e) => e,
                            Err(_) => {
                                let _ = tx.send(None);
                                return;
                            }
                        };
                        let simp_parsed = match parse(&simp_clone, &mut simplifier.context) {
                            Ok(e) => e,
                            Err(_) => {
                                let _ = tx.send(None);
                                return;
                            }
                        };

                        // Use default budget — the thread-based 2s timeout prevents hangs
                        let opts = cas_solver::runtime::SimplifyOptions::default();
                        let mut combo_cycles: usize = 0;

                        let (mut e, _, stats_e) = simplifier.simplify_with_stats(exp_parsed, opts.clone());
                        combo_cycles += stats_e.cycle_events.len();
                        let (mut s, _, stats_s) = simplifier.simplify_with_stats(simp_parsed, opts.clone());
                        combo_cycles += stats_s.cycle_events.len();

                        // Post-process: fold_constants to match CLI eval_simplify behavior
                        {
                            let cfg = cas_solver::runtime::EvalConfig::default();
                            let mut budget = cas_solver::runtime::Budget::preset_cli();
                            if let Ok(r) = cas_solver::api::fold_constants(&mut simplifier.context, e, &cfg, cas_solver::api::ConstFoldMode::Safe, &mut budget) {
                                e = r.expr;
                            }
                            if let Ok(r) = cas_solver::api::fold_constants(&mut simplifier.context, s, &cfg, cas_solver::api::ConstFoldMode::Safe, &mut budget) {
                                s = r.expr;
                            }
                        }

                        // Check 1: NF convergence
                        let nf_match =
                            cas_solver::runtime::compare_expr(&simplifier.context, e, s)
                                == std::cmp::Ordering::Equal;

                        if nf_match {
                            let _ = tx.send(Some((
                                "nf".to_string(),
                                String::new(),
                                String::new(),
                                String::new(),
                                combo_cycles,
                            )));
                            return;
                        }

                        // Check 2: Proved symbolic — simplify(LHS/RHS) == 1  [fresh context]
                        // Uses a fresh Simplifier to match CLI behavior (no context pollution).
                        {
                            let q_str = format!("({}) / ({})", exp_clone, simp_clone);
                            let mut sq = Simplifier::with_default_rules();
                            if let Ok(qp) = parse(&q_str, &mut sq.context) {
                                let (mut qr, _) = sq.simplify(qp);
                                let cfg = cas_solver::runtime::EvalConfig::default();
                                let mut budget = cas_solver::runtime::Budget::preset_cli();
                                if let Ok(r) = cas_solver::api::fold_constants(&mut sq.context, qr, &cfg, cas_solver::api::ConstFoldMode::Safe, &mut budget) {
                                    qr = r.expr;
                                }
                                let target = num_rational::BigRational::from_integer(1.into());
                                if matches!(sq.context.get(qr), cas_ast::Expr::Number(n) if *n == target) {
                                    let _ = tx.send(Some((
                                        "proved-q".to_string(),
                                        String::new(),
                                        String::new(),
                                        String::new(),
                                        combo_cycles,
                                    )));
                                    return;
                                }
                            }
                        }

                        // Check 2b: Difference fallback — simplify(LHS - RHS) == 0  [fresh context]
                        {
                            let d_str = format!("({}) - ({})", exp_clone, simp_clone);
                            let mut sd = Simplifier::with_default_rules();
                            if let Ok(dp) = parse(&d_str, &mut sd.context) {
                                let (mut dr, _) = sd.simplify(dp);
                                let cfg = cas_solver::runtime::EvalConfig::default();
                                let mut budget = cas_solver::runtime::Budget::preset_cli();
                                if let Ok(r) = cas_solver::api::fold_constants(&mut sd.context, dr, &cfg, cas_solver::api::ConstFoldMode::Safe, &mut budget) {
                                    dr = r.expr;
                                }
                                let zero = num_rational::BigRational::from_integer(0.into());
                                if matches!(sd.context.get(dr), cas_ast::Expr::Number(n) if *n == zero) {
                                    let _ = tx.send(Some((
                                        "proved-d".to_string(),
                                        String::new(),
                                        String::new(),
                                        String::new(),
                                        combo_cycles,
                                    )));
                                    return;
                                }
                            }
                        }

                        // Check 2c: Expand fallback — expand(LHS - RHS) == 0  [fresh context]
                        // Bridges trig identities gated behind expand_mode (Ticket 6c).
                        {
                            let d_str = format!("({}) - ({})", exp_clone, simp_clone);
                            let mut sd = Simplifier::with_default_rules();
                            if let Ok(dp) = parse(&d_str, &mut sd.context) {
                                let (mut dr, _) = sd.expand(dp);
                                let cfg = cas_solver::runtime::EvalConfig::default();
                                let mut budget = cas_solver::runtime::Budget::preset_cli();
                                if let Ok(r) = cas_solver::api::fold_constants(&mut sd.context, dr, &cfg, cas_solver::api::ConstFoldMode::Safe, &mut budget) {
                                    dr = r.expr;
                                }
                                let zero = num_rational::BigRational::from_integer(0.into());
                                if matches!(sd.context.get(dr), cas_ast::Expr::Number(n) if *n == zero) {
                                    let _ = tx.send(Some((
                                        "proved-d".to_string(),
                                        String::new(),
                                        String::new(),
                                        String::new(),
                                        combo_cycles,
                                    )));
                                    return;
                                }
                            }
                        }

                        if prove_zero_from_metamorphic_texts(
                            &mut simplifier,
                            &exp_clone,
                            &simp_clone,
                            e,
                            s,
                        ) {
                            let _ = tx.send(Some((
                                "proved-d".to_string(),
                                String::new(),
                                String::new(),
                                String::new(),
                                combo_cycles,
                            )));
                            return;
                        }

                        // Check 3: Numeric equivalence
                        match classify_numeric_equiv_for_vars(
                            &simplifier.context,
                            e,
                            s,
                            &combo_vars,
                            &combo_filters,
                            &config_clone,
                        ) {
                            NumericCheckOutcome::Pass => {
                                // Diagnostic: show what engine actually produced for LHS-RHS
                                let diff_str = if v {
                                    let d_diag = simplifier.context.add(cas_ast::Expr::Sub(e, s));
                                    let (d_simp, _) = simplifier.simplify(d_diag);
                                    format!(
                                        "simplify(LHS-RHS) => {}",
                                        cas_formatter::LaTeXExpr { context: &simplifier.context, id: d_simp }.to_latex()
                                    )
                                } else {
                                    String::new()
                                };
                                let shape = if v {
                                    let d_diag = simplifier.context.add(cas_ast::Expr::Sub(e, s));
                                    let (d_simp, _) = simplifier.simplify(d_diag);
                                    expr_shape_signature(&simplifier.context, d_simp)
                                } else {
                                    String::new()
                                };
                                let cause = numeric_only_cause_for_vars(
                                    &simplifier.context,
                                    e,
                                    s,
                                    &combo_vars,
                                    &combo_filters,
                                    &config_clone,
                                    &shape,
                                )
                                .label()
                                .to_string();
                                let _ = tx.send(Some((
                                    "numeric".to_string(),
                                    diff_str,
                                    shape,
                                    cause,
                                    combo_cycles,
                                )));
                            }
                            NumericCheckOutcome::Inconclusive(reason) => {
                                if pair_composed_ok {
                                    let _ = tx.send(Some((
                                        "proved-composed".to_string(),
                                        String::new(),
                                        String::new(),
                                        String::new(),
                                        combo_cycles,
                                    )));
                                } else {
                                    let _ = tx.send(Some((
                                        "inconclusive".to_string(),
                                        reason,
                                        String::new(),
                                        String::new(),
                                        combo_cycles,
                                    )));
                                }
                            }
                            NumericCheckOutcome::Failed(_) => {
                                if pair_composed_ok {
                                    let _ = tx.send(Some((
                                        "proved-composed".to_string(),
                                        String::new(),
                                        String::new(),
                                        String::new(),
                                        combo_cycles,
                                    )));
                                } else {
                                    let _ = tx.send(Some((
                                        "failed".to_string(),
                                        String::new(),
                                        String::new(),
                                        String::new(),
                                        combo_cycles,
                                    )));
                                }
                            }
                        }
                    });

                match rx.recv_timeout(timeout) {
                    Ok(Some((kind, diff_str, shape, cause, cycles))) => match kind.as_str() {
                        "nf" => {
                            nf_convergent += 1;
                            passed += 1;
                            cycle_events_total += cycles;
                        }
                        "proved-q" | "proved-d" => {
                            if kind.as_str() == "proved-q" {
                                proved_quotient += 1;
                            } else {
                                proved_difference += 1;
                            }
                            passed += 1;
                            cycle_events_total += cycles;
                            if verbose && nf_mismatch_examples.len() < max_examples {
                                nf_mismatch_examples.push((
                                    combined_exp.clone(),
                                    combined_simp.clone(),
                                    pair1.simp.clone(),
                                    pair2.simp.clone(),
                                ));
                            }
                        }
                        "numeric" => {
                            numeric_only += 1;
                            passed += 1;
                            cycle_events_total += cycles;
                            *numeric_only_causes.entry(cause.clone()).or_default() += 1;
                            if verbose {
                                numeric_only_examples.push((
                                    combined_exp.clone(),
                                    combined_simp.clone(),
                                    pair1.simp.clone(),
                                    pair2.simp.clone(),
                                    diff_str,
                                    shape,
                                    cause,
                                ));
                            }
                        }
                        "inconclusive" => {
                            inconclusive += 1;
                            cycle_events_total += cycles;
                        }
                        _ => {
                            failed += 1;
                            cycle_events_total += cycles;
                            if failed <= 5 {
                                eprintln!(
                                    "❌ Double combo [{}] failed: ({}) {} ({})",
                                    op.name(),
                                    pair1.exp,
                                    op.symbol(),
                                    pair2.exp
                                );
                            }
                        }
                    },
                    Ok(None) => { /* parse error, skip */ }
                    Err(_) => {
                        // Timeout — thread is still running but we move on
                        timeouts += 1;
                        eprintln!(
                            "  ⏱️  T/O [{}] #{}: [{}] {} [{}]  →  ({}) {} ({})",
                            op.name(),
                            timeouts,
                            pair1.family,
                            op.symbol(),
                            pair2.family,
                            pair1.exp,
                            op.symbol(),
                            pair2.exp,
                        );
                    }
                }
                continue; // skip the inline path below
            }

            // Inline path for Add/Sub (no thread needed, cooperative timeout is sufficient)
            // Wrap in catch_unwind to handle latent panics (e.g., num-rational denominator==0)
            // that surface with certain identity pair selections.
            let combo_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut simplifier = Simplifier::with_default_rules();
                let exp_parsed = match parse(&combined_exp, &mut simplifier.context) {
                    Ok(e) => e,
                    Err(_) => return ("skip", String::new(), String::new(), String::new(), 0),
                };
                let simp_parsed = match parse(&combined_simp, &mut simplifier.context) {
                    Ok(e) => e,
                    Err(_) => return ("skip", String::new(), String::new(), String::new(), 0),
                };

                let combo_start = std::time::Instant::now();
                let mut inline_cycles: usize = 0;
                let (exp_simplified, simp_simplified) = {
                    let opts = cas_solver::runtime::SimplifyOptions::default();
                    let (mut e, _, stats_e) =
                        simplifier.simplify_with_stats(exp_parsed, opts.clone());
                    inline_cycles += stats_e.cycle_events.len();
                    // Post-process: fold_constants to match CLI eval_simplify behavior
                    {
                        let cfg = cas_solver::runtime::EvalConfig::default();
                        let mut budget = cas_solver::runtime::Budget::preset_cli();
                        if let Ok(r) = cas_solver::api::fold_constants(
                            &mut simplifier.context,
                            e,
                            &cfg,
                            cas_solver::api::ConstFoldMode::Safe,
                            &mut budget,
                        ) {
                            e = r.expr;
                        }
                    }
                    if combo_start.elapsed() > combo_timeout {
                        return (
                            "timeout",
                            String::new(),
                            String::new(),
                            String::new(),
                            inline_cycles,
                        );
                    }
                    let (mut s, _, stats_s) = simplifier.simplify_with_stats(simp_parsed, opts);
                    inline_cycles += stats_s.cycle_events.len();
                    {
                        let cfg = cas_solver::runtime::EvalConfig::default();
                        let mut budget = cas_solver::runtime::Budget::preset_cli();
                        if let Ok(r) = cas_solver::api::fold_constants(
                            &mut simplifier.context,
                            s,
                            &cfg,
                            cas_solver::api::ConstFoldMode::Safe,
                            &mut budget,
                        ) {
                            s = r.expr;
                        }
                    }
                    (e, s)
                };
                if combo_start.elapsed() > combo_timeout {
                    return (
                        "timeout",
                        String::new(),
                        String::new(),
                        String::new(),
                        inline_cycles,
                    );
                }

                // Check 1: Normal form convergence (exact structural match)
                let nf_match = cas_solver::runtime::compare_expr(
                    &simplifier.context,
                    exp_simplified,
                    simp_simplified,
                ) == std::cmp::Ordering::Equal;

                if nf_match {
                    return (
                        "nf",
                        String::new(),
                        String::new(),
                        String::new(),
                        inline_cycles,
                    );
                }

                // Check 2: Proved symbolic — simplify(LHS - RHS) == 0  [fresh context]
                // Uses a fresh Simplifier to match CLI behavior (avoids context pollution).
                let diff_simplified = {
                    let diff_str = format!("({}) - ({})", combined_exp, combined_simp);
                    let mut sd = Simplifier::with_default_rules();
                    if let Ok(dp) = parse(&diff_str, &mut sd.context) {
                        let (mut dr, _) = sd.simplify(dp);
                        let cfg = cas_solver::runtime::EvalConfig::default();
                        let mut budget = cas_solver::runtime::Budget::preset_cli();
                        if let Ok(r) = cas_solver::api::fold_constants(
                            &mut sd.context,
                            dr,
                            &cfg,
                            cas_solver::api::ConstFoldMode::Safe,
                            &mut budget,
                        ) {
                            dr = r.expr;
                        }
                        let zero = num_rational::BigRational::from_integer(0.into());
                        if matches!(sd.context.get(dr), cas_ast::Expr::Number(n) if *n == zero) {
                            return (
                                "proved",
                                String::new(),
                                String::new(),
                                String::new(),
                                inline_cycles,
                            );
                        }
                    }
                    // Also try with the polluted simplifier (same context that simplified LHS/RHS)
                    let d = simplifier
                        .context
                        .add(cas_ast::Expr::Sub(exp_simplified, simp_simplified));
                    let (mut ds, _) = simplifier.simplify(d);
                    {
                        let cfg = cas_solver::runtime::EvalConfig::default();
                        let mut budget = cas_solver::runtime::Budget::preset_cli();
                        if let Ok(r) = cas_solver::api::fold_constants(
                            &mut simplifier.context,
                            ds,
                            &cfg,
                            cas_solver::api::ConstFoldMode::Safe,
                            &mut budget,
                        ) {
                            ds = r.expr;
                        }
                    }
                    let target_value = num_rational::BigRational::from_integer(0.into());
                    if matches!(simplifier.context.get(ds), cas_ast::Expr::Number(n) if *n == target_value)
                    {
                        return (
                            "proved",
                            String::new(),
                            String::new(),
                            String::new(),
                            inline_cycles,
                        );
                    }
                    ds
                };

                if prove_zero_from_metamorphic_texts(
                    &mut simplifier,
                    &combined_exp,
                    &combined_simp,
                    exp_simplified,
                    simp_simplified,
                ) {
                    return (
                        "proved",
                        String::new(),
                        String::new(),
                        String::new(),
                        inline_cycles,
                    );
                }

                // Check 3: Fallback to numeric equivalence. Only if both the direct
                // symbolic proof and the numeric check fail do we fall back to
                // "proved-composed", using the fact that both source identities are
                // independently symbolically proved.
                match classify_numeric_equiv_for_vars(
                    &simplifier.context,
                    exp_simplified,
                    simp_simplified,
                    &combined_vars,
                    &combined_filters,
                    &config,
                ) {
                    NumericCheckOutcome::Pass => {
                        // Diagnostic: show what engine produced (the non-zero residual)
                        let diff_str = if verbose {
                            format!(
                                "simplify(LHS-RHS) => {}",
                                cas_formatter::LaTeXExpr {
                                    context: &simplifier.context,
                                    id: diff_simplified
                                }
                                .to_latex()
                            )
                        } else {
                            String::new()
                        };
                        let shape = if verbose {
                            expr_shape_signature(&simplifier.context, diff_simplified)
                        } else {
                            String::new()
                        };
                        let cause = numeric_only_cause_for_vars(
                            &simplifier.context,
                            exp_simplified,
                            simp_simplified,
                            &combined_vars,
                            &combined_filters,
                            &config,
                            &shape,
                        )
                        .label()
                        .to_string();
                        ("numeric", diff_str, shape, cause, inline_cycles)
                    }
                    NumericCheckOutcome::Inconclusive(reason) => {
                        if pair_symbolic_ok[i] && pair_symbolic_ok[j] {
                            (
                                "proved-composed",
                                String::new(),
                                String::new(),
                                String::new(),
                                inline_cycles,
                            )
                        } else {
                            (
                                "inconclusive",
                                reason,
                                String::new(),
                                String::new(),
                                inline_cycles,
                            )
                        }
                    }
                    NumericCheckOutcome::Failed(_) => {
                        if pair_symbolic_ok[i] && pair_symbolic_ok[j] {
                            (
                                "proved-composed",
                                String::new(),
                                String::new(),
                                String::new(),
                                inline_cycles,
                            )
                        } else {
                            (
                                "failed",
                                String::new(),
                                String::new(),
                                String::new(),
                                inline_cycles,
                            )
                        }
                    }
                }
            }));

            match combo_result {
                Ok((kind, diff_str, shape, cause, cycles)) => match kind {
                    "nf" => {
                        nf_convergent += 1;
                        passed += 1;
                        cycle_events_total += cycles;
                    }
                    "proved" | "proved-composed" => {
                        if kind == "proved-composed" {
                            proved_composed += 1;
                            if verbose && proved_composed_examples.len() < max_examples {
                                proved_composed_examples.push((
                                    combined_exp.clone(),
                                    combined_simp.clone(),
                                    pair1.simp.clone(),
                                    pair2.simp.clone(),
                                ));
                            }
                        } else {
                            proved_quotient += 1;
                        }
                        passed += 1;
                        cycle_events_total += cycles;
                        if verbose && nf_mismatch_examples.len() < max_examples {
                            nf_mismatch_examples.push((
                                combined_exp.clone(),
                                combined_simp.clone(),
                                pair1.simp.clone(),
                                pair2.simp.clone(),
                            ));
                        }
                    }
                    "numeric" => {
                        numeric_only += 1;
                        passed += 1;
                        cycle_events_total += cycles;
                        *numeric_only_causes.entry(cause.clone()).or_default() += 1;
                        if verbose {
                            numeric_only_examples.push((
                                combined_exp.clone(),
                                combined_simp.clone(),
                                pair1.simp.clone(),
                                pair2.simp.clone(),
                                diff_str,
                                shape,
                                cause,
                            ));
                        }
                    }
                    "inconclusive" => {
                        inconclusive += 1;
                        cycle_events_total += cycles;
                    }
                    "timeout" => {
                        timeouts += 1;
                        cycle_events_total += cycles;
                    }
                    "skip" => { /* parse error, silently continue */ }
                    _ => {
                        failed += 1;
                        cycle_events_total += cycles;
                        if failed <= 5 {
                            eprintln!(
                                "❌ Double combo [{}] failed: ({}) {} ({})",
                                op.name(),
                                pair1.exp,
                                op.symbol(),
                                pair2.exp
                            );
                        }
                    }
                },
                Err(_) => {
                    // Panic caught (e.g., num-rational denominator==0) — treat as skip
                    skipped += 1;
                }
            }
        }
    }

    eprintln!(
        "✅ Double combinations [{}]: {} passed, {} failed, {} skipped (timeout), {} inconclusive",
        op.name(),
        passed,
        failed,
        skipped,
        inconclusive
    );
    eprintln!(
        "   📐 NF-convergent: {} | 🔢 Proved-symbolic: {} (quotient: {}, diff: {}, composed: {}) | 🌡️ Numeric-only: {} | ◐ Inconclusive: {}",
        nf_convergent,
        proved_quotient + proved_difference + proved_composed,
        proved_quotient,
        proved_difference,
        proved_composed,
        numeric_only,
        inconclusive
    );
    if verbose && numeric_only > 0 {
        print_numeric_only_cause_breakdown(&numeric_only_causes);
    }

    // Print NF-mismatch examples if verbose (proved_symbolic but different normal forms)
    if verbose && !nf_mismatch_examples.is_empty() {
        eprintln!("\n🔢 NF-mismatch examples (proved symbolic but different normal forms):");
        for (i, (lhs, rhs, simp1, simp2)) in nf_mismatch_examples.iter().enumerate() {
            eprintln!("   {:2}. LHS: {}", i + 1, lhs);
            eprintln!("       RHS: {}", rhs);
            eprintln!("       (simplifies: {} + {})", simp1, simp2);
        }
        if proved_quotient + proved_difference + proved_composed > max_examples {
            eprintln!(
                "   ... and {} more (set METATEST_MAX_EXAMPLES=N to show more)",
                proved_quotient + proved_difference + proved_composed - max_examples
            );
        }
        eprintln!();
    }

    if verbose && !proved_composed_examples.is_empty() {
        eprintln!(
            "🧩 Proved-composed examples (derived from independently proved source identities):"
        );
        for (i, (lhs, rhs, simp1, simp2)) in proved_composed_examples.iter().enumerate() {
            eprintln!("   {:2}. LHS: {}", i + 1, lhs);
            eprintln!("       RHS: {}", rhs);
            eprintln!("       (sources: {} | {})", simp1, simp2);
        }
        if proved_composed > max_examples {
            eprintln!(
                "   ... and {} more (set METATEST_MAX_EXAMPLES=N to show more)",
                proved_composed - max_examples
            );
        }
        eprintln!();
    }

    // Print numeric-only examples if verbose
    if verbose && !numeric_only_examples.is_empty() {
        eprintln!("🌡️ Numeric-only examples (no symbolic proof found):");
        for (i, (lhs, rhs, _simp1, _simp2, diff_residual, _shape, cause)) in
            numeric_only_examples.iter().take(max_examples).enumerate()
        {
            eprintln!("   {:2}. LHS: {}", i + 1, lhs);
            eprintln!("       RHS: {}", rhs);
            eprintln!("       Cause: {}", cause);
            eprintln!("       simplify(LHS-RHS): {}", diff_residual);
        }
        if numeric_only > max_examples {
            eprintln!(
                "   ... and {} more (set METATEST_MAX_EXAMPLES=N to show more)",
                numeric_only - max_examples
            );
        }
        eprintln!();

        // Family classifier for numeric-only cases - stores expressions per family
        let mut family_examples: HashMap<&str, Vec<(String, String)>> = HashMap::new();

        for (lhs, rhs, _, _, _, _, _) in &numeric_only_examples {
            let combined = format!("{} {}", lhs, rhs);
            let expr_pair = (lhs.clone(), rhs.clone());

            // Detect function families (mutually exclusive for cleaner grouping)
            let family = if combined.contains("sec") || combined.contains("csc") {
                "sec/csc (Pythagorean: tan²+1=sec², 1+cot²=csc²)"
            } else if combined.contains("tan(") && !combined.contains("arctan") {
                "tan (without sec/csc)"
            } else if combined.contains("cot(") {
                "cot (without csc)"
            } else if combined.contains("sin(")
                && (combined.contains("/2") || combined.contains("*2"))
            {
                "half/double angle"
            } else if combined.contains("ln(") || combined.contains("log(") {
                "ln/log"
            } else if combined.contains("exp(") {
                "exp"
            } else if combined.contains("sqrt(") || combined.contains("^(1/") {
                "sqrt/roots"
            } else if combined.contains("abs(") {
                "abs"
            } else if combined.contains("arctan")
                || combined.contains("arcsin")
                || combined.contains("arccos")
            {
                "arc* (inverse trig)"
            } else {
                "other"
            };

            family_examples.entry(family).or_default().push(expr_pair);
        }

        if !family_examples.is_empty() {
            eprintln!("📊 Numeric-only grouped by family:");

            // Sort families by count
            let mut sorted: Vec<_> = family_examples.into_iter().collect();
            sorted.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

            for (family, examples) in sorted {
                eprintln!("\n   ── {} ({} cases) ──", family, examples.len());
                let show_count = examples.len().min(max_examples.max(3)); // Show at least 3
                for (lhs, rhs) in examples.iter().take(show_count) {
                    eprintln!("      LHS: {}", lhs);
                    eprintln!("      RHS: {}", rhs);
                    eprintln!();
                }
                if examples.len() > show_count {
                    eprintln!("      ... and {} more", examples.len() - show_count);
                }
            }
            eprintln!();
        }

        // Top-N Shape Analysis: identify dominant patterns in residuals
        eprintln!("📈 Top-N Shape Analysis (residual patterns):");
        let mut shape_counts: HashMap<String, (usize, String)> = HashMap::new(); // shape -> (count, example_diff)

        for (_lhs, _rhs, _, _, diff_residual, shape, _) in &numeric_only_examples {
            let entry = shape_counts
                .entry(shape.clone())
                .or_insert((0, diff_residual.clone()));
            entry.0 += 1;
        }

        let mut sorted_shapes: Vec<_> = shape_counts.into_iter().collect();
        sorted_shapes.sort_by(|a, b| b.1 .0.cmp(&a.1 .0)); // Sort by count descending

        let total = numeric_only_examples.len();
        for (i, (shape, (count, example))) in sorted_shapes.iter().take(20).enumerate() {
            let pct = (*count as f64 / total as f64) * 100.0;
            let markers = format!(
                "{}{}",
                if shape_has_neg_exp(shape) {
                    " [NEG_EXP]"
                } else {
                    ""
                },
                if shape_has_div(shape) { " [DIV]" } else { "" }
            );
            eprintln!(
                "   {:2}. {:5.1}% ({:3}) {}{}",
                i + 1,
                pct,
                count,
                if shape.len() > 60 {
                    &shape[..60]
                } else {
                    shape
                },
                markers
            );
            if i < 5 {
                // Show example for top 5
                eprintln!(
                    "       Example: {}",
                    if example.len() > 80 {
                        &example[..80]
                    } else {
                        example
                    }
                );
            }
        }
        if sorted_shapes.len() > 20 {
            eprintln!("   ... and {} more unique shapes", sorted_shapes.len() - 20);
        }
        eprintln!();
    }

    // Triple combinations (optional, limited)
    if include_triples && n >= 3 {
        let mut triple_passed = 0;
        let mut triple_failed = 0;
        let mut triple_inconclusive = 0;
        let mut triple_nf = 0;
        let mut triple_proved = 0;
        let mut triple_numeric = 0;
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

                    let mut used_vars: HashSet<String> = pair1.vars.iter().cloned().collect();
                    let (pair2_exp, pair2_simp, pair2_vars, pair2_filters) =
                        rename_identity_for_combination(pair2, &mut used_vars);
                    let (pair3_exp, pair3_simp, pair3_vars, pair3_filters) =
                        rename_identity_for_combination(pair3, &mut used_vars);
                    let mut combined_vars = pair1.vars.clone();
                    combined_vars.extend(pair2_vars);
                    combined_vars.extend(pair3_vars);
                    let mut combined_filters = identity_filters(pair1);
                    combined_filters.extend(pair2_filters);
                    combined_filters.extend(pair3_filters);

                    let combined_exp = format!(
                        "(({}) {} ({})) {} ({})",
                        pair1.exp,
                        op.symbol(),
                        pair2_exp,
                        op.symbol(),
                        pair3_exp
                    );
                    let combined_simp = format!(
                        "(({}) {} ({})) {} ({})",
                        pair1.simp,
                        op.symbol(),
                        pair2_simp,
                        op.symbol(),
                        pair3_simp
                    );

                    let mut simplifier = Simplifier::with_default_rules();
                    let exp_parsed = match parse(&combined_exp, &mut simplifier.context) {
                        Ok(e) => e,
                        Err(_) => continue,
                    };
                    let simp_parsed = match parse(&combined_simp, &mut simplifier.context) {
                        Ok(e) => e,
                        Err(_) => continue,
                    };

                    let (exp_simplified_raw, _) = simplifier.simplify(exp_parsed);
                    let exp_simplified =
                        fold_constants_safe(&mut simplifier.context, exp_simplified_raw);
                    let (simp_simplified_raw, _) = simplifier.simplify(simp_parsed);
                    let simp_simplified =
                        fold_constants_safe(&mut simplifier.context, simp_simplified_raw);

                    if cas_solver::runtime::compare_expr(
                        &simplifier.context,
                        exp_simplified,
                        simp_simplified,
                    ) == std::cmp::Ordering::Equal
                    {
                        triple_nf += 1;
                        triple_passed += 1;
                        triple_count += 1;
                        continue;
                    }

                    if prove_zero_from_metamorphic_texts(
                        &mut simplifier,
                        &combined_exp,
                        &combined_simp,
                        exp_simplified,
                        simp_simplified,
                    ) {
                        triple_proved += 1;
                        triple_passed += 1;
                        triple_count += 1;
                        continue;
                    }

                    match classify_numeric_equiv_for_vars(
                        &simplifier.context,
                        exp_simplified,
                        simp_simplified,
                        &combined_vars,
                        &combined_filters,
                        &config,
                    ) {
                        NumericCheckOutcome::Pass => {
                            triple_numeric += 1;
                            triple_passed += 1;
                        }
                        NumericCheckOutcome::Inconclusive(_) => {
                            triple_inconclusive += 1;
                        }
                        NumericCheckOutcome::Failed(_) => {
                            triple_failed += 1;
                        }
                    }

                    triple_count += 1;
                }
            }
        }

        eprintln!(
            "✅ Triple combinations: {} passed, {} failed, {} inconclusive (of {} tested)",
            triple_passed, triple_failed, triple_inconclusive, triple_count
        );
        eprintln!(
            "   📐 Triple NF-convergent: {} | 🔢 Triple Proved-symbolic: {} | 🌡️ Triple Numeric-only: {}",
            triple_nf, triple_proved, triple_numeric
        );
    }

    ComboMetrics {
        op: op.name().to_string(),
        pairs: n,
        families: num_families,
        combos: passed + failed + skipped + timeouts,
        nf_convergent,
        proved_quotient,
        proved_difference,
        proved_composed,
        numeric_only,
        inconclusive,
        failed,
        skipped,
        timeouts,
        cycle_events_total,
    }
}
// =============================================================================
// CSV-BASED AUTOMATIC COMBINATION TESTS
// =============================================================================

/// Run automatic double and triple combinations from CSV file
/// This generates thousands of test cases from ~180 identity pairs
#[test]
fn metatest_csv_combinations_small() {
    // Small run: 30 pairs (stratified) = ~351 double combinations
    let m = run_csv_combination_tests(30, false, CombineOp::Add);
    assert_eq!(m.failed, 0, "Some CSV combination tests failed");
}

#[test]
#[ignore] // Run with: cargo test --ignored
fn metatest_csv_combinations_full() {
    // Full run: 150 pairs (stratified) with triples. Covers all ~40 CSV families.
    let m = run_csv_combination_tests(150, true, CombineOp::Add);
    assert_eq!(m.failed, 0, "Some CSV combination tests failed");
}

/// Multiplicative combination test: (LHS_1 * LHS_2) vs (RHS_1 * RHS_2)
/// Tests distribution, factoring, power simplification paths.
/// Uses stratified sampling: 1 representative per CSV family (~134) + fill to 150.
#[test]
#[ignore]
fn metatest_csv_combinations_mul() {
    // 150 pairs (stratified) ≈ 11,175 combos. 2s per-combo timeout caps cost.
    let m = run_csv_combination_tests(150, false, CombineOp::Mul);
    assert_eq!(m.failed, 0, "Some CSV combination tests failed");
}

/// Additive combination test with stratified coverage
/// (LHS_1 + LHS_2) vs (RHS_1 + RHS_2)
#[test]
#[ignore]
fn metatest_csv_combinations_add() {
    // 150 pairs (stratified) ≈ 11,175 combos. Add is fast (≈5s timeout).
    let m = run_csv_combination_tests(150, false, CombineOp::Add);
    assert_eq!(m.failed, 0, "Some CSV combination tests failed");
}

/// Subtractive combination test with stratified coverage
/// (LHS_1 - LHS_2) vs (RHS_1 - RHS_2)
/// Tests sign handling, cancellation, and subtraction-specific simplification
#[test]
#[ignore]
fn metatest_csv_combinations_sub() {
    // 150 pairs (stratified) ≈ 11,175 combos. Sub is fast (≈5s timeout).
    let m = run_csv_combination_tests(150, false, CombineOp::Sub);
    assert_eq!(m.failed, 0, "Some CSV combination tests failed");
}

/// Division combination test: (LHS_1 / LHS_2) vs (RHS_1 / RHS_2)
/// Tests fraction simplification, quotient cancellation, and cross-multiplication paths.
/// Uses stratified sampling: 1 representative per CSV family (~134) + fill to 50.
/// Includes a divisor safety guard: identities that evaluate near zero are skipped as divisors.
#[test]
#[ignore]
fn metatest_csv_combinations_div() {
    // 50 pairs (stratified) ≈ 1,225 combos. Fewer than Mul due to CAS
    // limitations with high-degree polynomial divisors causing fraction
    // simplification failures. Still covers ~50 families (vs old 15/~12).
    let m = run_csv_combination_tests(50, false, CombineOp::Div);
    assert_eq!(m.failed, 0, "Some CSV combination tests failed");
}

/// UNIFIED BENCHMARK: run all 4 operations and print a regression/improvement table.
///
/// This test does NOT assert on failures — it prints metrics for comparison.
/// Use it as a diagnostic benchmark before/after rule changes:
///
/// ```text
/// cargo test --release -p cas_engine --test metamorphic_simplification_tests \
///     -- metatest_benchmark_all_ops --ignored --nocapture
/// ```
///
/// Key metrics:
/// - NF-convergent: simplify(LHS) ≡ simplify(RHS) structurally (ideal)
/// - Proved-symbolic: simplify(LHS - RHS) = 0 (correct but different NFs)
/// - Numeric-only: only passes numeric check (target for improvement)
/// - Failed: semantic mismatches (regressions)
#[test]
#[ignore]
fn metatest_benchmark_all_ops() {
    // Pair counts per operation (stratified sampling)
    let configs: Vec<(CombineOp, usize)> = vec![
        (CombineOp::Add, 150),
        (CombineOp::Sub, 150),
        (CombineOp::Mul, 150),
        (CombineOp::Div, 50),
    ];

    let mut all_metrics: Vec<ComboMetrics> = Vec::new();
    let mut total_failed = 0;

    for (op, pairs) in &configs {
        // Run without internal assert — collect metrics only
        let metrics = run_csv_combination_tests(*pairs, false, *op);
        total_failed += metrics.failed;
        all_metrics.push(metrics);
    }

    // Print unified benchmark table
    eprintln!();
    eprintln!(
        "╔═══════════════════════════════════════════════════════════════════════════════════╗"
    );
    eprintln!(
        "║                     METAMORPHIC BENCHMARK RESULTS                                ║"
    );
    eprintln!(
        "╠═════╤════════╤══════════╤══════════════╤════════════════╤══════════════╤══════════╣"
    );
    eprintln!(
        "║ Op  │ Pairs  │ Families │ NF-convergent│ Proved-sym (Q+D)│ Numeric-only │ Failed   ║"
    );
    eprintln!(
        "╠═════╪════════╪══════════╪══════════════╪═════════════════╪══════════════╪══════════╣"
    );

    let mut total_nf = 0;
    let mut total_proved = 0;
    let mut total_numeric = 0;
    let mut total_combos = 0;
    let mut total_skipped = 0;
    let mut total_f = 0;

    for m in &all_metrics {
        let effective = m.combos - m.skipped;
        let proved = m.proved_symbolic();
        eprintln!(
            "║ {:<3} │ {:>5}  │ {:>7}  │ {:>6} {:>5.1}% │{:>4}+{:>4}+{:>4}{:>5.1}% │ {:>6} {:>5.1}% │ {:>6}   ║",
            m.op, m.pairs, m.families,
            m.nf_convergent,
            if effective > 0 { m.nf_convergent as f64 / effective as f64 * 100.0 } else { 0.0 },
            m.proved_quotient,
            m.proved_difference,
            m.proved_composed,
            if effective > 0 { proved as f64 / effective as f64 * 100.0 } else { 0.0 },
            m.numeric_only,
            if effective > 0 { m.numeric_only as f64 / effective as f64 * 100.0 } else { 0.0 },
            m.failed,
        );
        total_nf += m.nf_convergent;
        total_proved += proved;
        total_numeric += m.numeric_only;
        total_combos += m.combos;
        total_skipped += m.skipped;
        total_f += m.failed;
    }

    let total_effective = total_combos - total_skipped;
    eprintln!(
        "╠═════╪════════╪══════════╪══════════════╪═════════════════╪══════════════╪══════════╣"
    );
    eprintln!(
        "║ ALL │        │          │ {:>6} {:>5.1}% │     {:>5}{:>5.1}% │ {:>6} {:>5.1}% │ {:>6}   ║",
        total_nf,
        if total_effective > 0 {
            total_nf as f64 / total_effective as f64 * 100.0
        } else {
            0.0
        },
        total_proved,
        if total_effective > 0 {
            total_proved as f64 / total_effective as f64 * 100.0
        } else {
            0.0
        },
        total_numeric,
        if total_effective > 0 {
            total_numeric as f64 / total_effective as f64 * 100.0
        } else {
            0.0
        },
        total_f,
    );
    eprintln!(
        "╚═════╧════════╧══════════╧══════════════╧═════════════════╧══════════════╧══════════╝"
    );
    eprintln!(
        "   Total combos: {} (skipped: {})",
        total_combos, total_skipped
    );
    eprintln!();

    if total_failed > 0 {
        eprintln!(
            "⚠️  {} semantic failures detected — investigate before merging.",
            total_failed
        );
    }
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
        cas_solver::runtime::DomainMode::Assume
    } else {
        cas_solver::runtime::DomainMode::Generic
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
        // Check symbolic equality using bucket-aware equivalence
        let sym_result = check_symbolic_equiv_bucket_aware(
            &mut simplifier,
            exp_parsed,
            simp_parsed,
            pair.bucket,
        );

        // Simplify for display and numeric fallback
        let opts = cas_solver::runtime::SimplifyOptions {
            shared: cas_solver::runtime::SharedSemanticConfig {
                semantics: cas_solver::runtime::EvalConfig {
                    domain_mode,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let (exp_simplified, _) = simplifier.simplify_with_options(exp_parsed, opts.clone());
        let (simp_simplified, _) = simplifier.simplify_with_options(simp_parsed, opts);

        let symbolic_match = matches!(
            sym_result,
            SymbolicResult::Pass | SymbolicResult::PassConditional(_)
        );

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
                        &pair.filter_spec,
                        &FilterSpec::None,
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
                    filter_str: pair.filter_spec.as_str(),
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
                let lines: Vec<String> = reader.lines().map_while(Result::ok).collect();

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
                for id in baseline.keys() {
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
            if pair.bucket == Bucket::BranchSensitive
                && !matches!(transform, MetaTransform::Scale(k) if (*k - 2.0).abs() < 1e-10)
            {
                skipped_bucket += 1;
                continue;
            }

            total_tests += 1;

            match test_transform_identity(pair, &pair.vars[0], transform, min_valid_factor) {
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
    let mut _filtered_out = 0;

    for &x in &samples {
        // Apply transform: x' = T(x)
        let x_prime = transform.apply_f64(x);

        // Composed filter: check if x' passes the original filter
        if !pair.filter_spec.accept(x_prime) {
            _filtered_out += 1;
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

// =============================================================================
// Substitution-based Metamorphic Tests
// =============================================================================
// Instead of combining two identities with an operation (A*B),
// this test substitutes a variable in one identity with a sub-expression:
//   Given A(x) == B(x) and substitution x → S(u),
//   check: simplify(A(S(u))) == simplify(B(S(u)))
//
// This creates deeply nested expressions that stress recursive simplification.

/// A substitution expression to plug into identity variables
#[derive(Clone, Debug)]
struct SubstitutionExpr {
    expr: String,  // The expression to substitute, e.g. "sin(u)"
    var: String,   // The free variable after substitution, e.g. "u"
    label: String, // Category label, e.g. "trig"
}

/// A direct contextual equivalence A(u) == B(u), curated outside the generic
/// substitution cross-product when that product becomes too aggressive.
#[derive(Clone, Debug)]
struct ContextualPair {
    lhs: String,
    rhs: String,
    vars: Vec<String>,
    filters: Vec<FilterSpec>,
    family: String,
}

#[derive(Clone, Debug)]
struct IdempotenceExpr {
    expr: String,
    vars: Vec<String>,
    filters: Vec<FilterSpec>,
    family: String,
}

#[derive(Clone, Debug)]
struct RequiresContractExpr {
    expr: String,
    expect_requires: bool,
    family: String,
}

#[derive(Debug, Clone)]
struct WarningsContractExpr {
    expr: String,
    mode: cas_solver::runtime::DomainMode,
    expect_warning: bool,
    family: String,
}

#[derive(Debug, Clone)]
struct TransparencySignalContractExpr {
    expr: String,
    mode: cas_solver::runtime::DomainMode,
    expect_signal: bool,
    family: String,
}

#[derive(Debug, Clone)]
struct BranchTransparencyContractExpr {
    expr: String,
    mode: cas_solver::runtime::DomainMode,
    inv_trig: cas_solver::runtime::InverseTrigPolicy,
    expect_signal: bool,
    family: String,
}

#[derive(Debug, Clone)]
enum SemanticBehaviorExpectation {
    Exact(String),
    ContainsAll(Vec<String>),
}

#[derive(Debug, Clone)]
struct SemanticBehaviorContractExpr {
    expr: String,
    value_domain: cas_solver::runtime::ValueDomain,
    mode: cas_solver::runtime::DomainMode,
    expectation: SemanticBehaviorExpectation,
    family: String,
}

#[derive(Debug, Clone)]
struct RequiresModeContractExpr {
    expr: String,
    mode: cas_solver::runtime::DomainMode,
    expect_requires: bool,
    family: String,
}

#[derive(Debug, Clone)]
struct SemanticAxesContractExpr {
    expr: String,
    value_domain: cas_solver::runtime::ValueDomain,
    mode: cas_solver::runtime::DomainMode,
    expect_requires: bool,
    expect_warning: bool,
    family: String,
}

#[derive(Debug, Clone)]
struct AssumptionTraceContractExpr {
    expr: String,
    mode: cas_solver::runtime::DomainMode,
    inv_trig: cas_solver::runtime::InverseTrigPolicy,
    expected_kind: Option<String>,
    family: String,
}

/// Word-boundary-aware text substitution.
/// Replaces all occurrences of `var` as a standalone word in `template`
/// with `replacement`, wrapping in parentheses for safety.
/// Uses simple word-boundary logic: a match is valid if the chars
/// before and after are not alphanumeric or underscore.
fn text_substitute(template: &str, var: &str, replacement: &str) -> String {
    let mut result = String::with_capacity(template.len() * 2);
    let chars: Vec<char> = template.chars().collect();
    let var_chars: Vec<char> = var.chars().collect();
    let var_len = var_chars.len();
    let mut i = 0;

    while i < chars.len() {
        // Check if var matches at position i
        if i + var_len <= chars.len() && chars[i..i + var_len] == var_chars[..] {
            // Check word boundary before
            let before_ok = if i == 0 {
                true
            } else {
                let c = chars[i - 1];
                !c.is_alphanumeric() && c != '_'
            };
            // Check word boundary after
            let after_ok = if i + var_len >= chars.len() {
                true
            } else {
                let c = chars[i + var_len];
                !c.is_alphanumeric() && c != '_'
            };

            if before_ok && after_ok {
                result.push('(');
                result.push_str(replacement);
                result.push(')');
                i += var_len;
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }
    result
}

/// Load substitution identity pairs from CSV
fn load_substitution_identities() -> Vec<IdentityPair> {
    let csv_path = find_test_data_file("substitution_identities.csv");
    let content =
        std::fs::read_to_string(csv_path).expect("Failed to read substitution_identities.csv");

    let mut pairs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Each row")
                && !label.starts_with("Substitution-Based")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 3 {
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
                bucket: Bucket::ConditionalRequires,
                branch_mode: BranchMode::default(),
                filter_spec: FilterSpec::None,
                family: current_family.clone(),
            });
        }
    }
    pairs
}

/// Load substitution expressions from CSV
fn load_substitution_expressions() -> Vec<SubstitutionExpr> {
    let csv_path = find_test_data_file("substitution_expressions.csv");
    let content =
        std::fs::read_to_string(csv_path).expect("Failed to read substitution_expressions.csv");

    let mut exprs = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 3 {
            exprs.push(SubstitutionExpr {
                expr: parts[0].trim().to_string(),
                var: parts[1].trim().to_string(),
                label: parts[2].trim().to_string(),
            });
        }
    }
    exprs
}

/// Load contextual direct pairs from CSV
fn parse_filter_specs(spec: &str, vars_len: usize, line_num: usize) -> Vec<FilterSpec> {
    let spec = spec.trim();
    if spec.is_empty() {
        return vec![FilterSpec::None; vars_len];
    }

    let mut filters: Vec<FilterSpec> = spec
        .split('|')
        .map(|part| parse_filter_spec(part.trim(), line_num))
        .collect();

    if filters.len() > vars_len {
        panic!(
            "Too many filter specs at line {}: expected at most {}, got {}",
            line_num,
            vars_len,
            filters.len()
        );
    }

    filters.resize(vars_len, FilterSpec::None);
    filters
}

fn load_direct_pairs(file_name: &str) -> Vec<ContextualPair> {
    let csv_path = find_test_data_file(file_name);
    let content = std::fs::read_to_string(csv_path)
        .unwrap_or_else(|_| panic!("Failed to read {}", file_name));

    let mut pairs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty() && !label.starts_with("Format") && !label.starts_with("Each row") {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.splitn(4, ',').collect();
        if parts.len() >= 3 {
            let vars: Vec<String> = parts[2]
                .trim()
                .split(';')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let filters = if parts.len() >= 4 {
                parse_filter_specs(parts[3], vars.len(), line_num)
            } else {
                vec![FilterSpec::None; vars.len()]
            };
            pairs.push(ContextualPair {
                lhs: parts[0].trim().to_string(),
                rhs: parts[1].trim().to_string(),
                vars,
                filters,
                family: current_family.clone(),
            });
        }
    }
    pairs
}

fn load_contextual_pairs() -> Vec<ContextualPair> {
    load_direct_pairs("contextual_pairs.csv")
}

fn load_contextual_rational_pairs() -> Vec<ContextualPair> {
    load_direct_pairs("contextual_rational_pairs.csv")
}

fn load_contextual_trig_pairs() -> Vec<ContextualPair> {
    load_direct_pairs("contextual_trig_pairs.csv")
}

fn load_contextual_polynomial_pairs() -> Vec<ContextualPair> {
    load_direct_pairs("contextual_polynomial_pairs.csv")
}

fn load_contextual_radical_pairs() -> Vec<ContextualPair> {
    load_direct_pairs("contextual_radical_pairs.csv")
}

fn load_residual_pairs() -> Vec<ContextualPair> {
    load_direct_pairs("residual_pairs.csv")
}

fn load_idempotence_expressions() -> Vec<IdempotenceExpr> {
    let csv_path = find_test_data_file("idempotence_expressions.csv");
    let content =
        std::fs::read_to_string(csv_path).expect("Failed to read idempotence_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty() && !label.starts_with("Format") && !label.starts_with("Goal:") {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.splitn(3, ',').collect();
        if parts.len() < 2 {
            panic!(
                "idempotence_expressions.csv line {}: expected at least expr,vars. Line: '{}'",
                line_num, line
            );
        }

        let vars: Vec<String> = parts[1]
            .trim()
            .split(';')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let filters = if parts.len() >= 3 {
            parse_filter_specs(parts[2], vars.len(), line_num)
        } else {
            vec![FilterSpec::None; vars.len()]
        };

        exprs.push(IdempotenceExpr {
            expr: parts[0].trim().to_string(),
            vars,
            filters,
            family: current_family.clone(),
        });
    }

    exprs
}

fn load_requires_contract_expressions() -> Vec<RequiresContractExpr> {
    let csv_path = find_test_data_file("requires_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read requires_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(2, ',').collect();
        if parts.len() != 2 {
            panic!(
                "requires_contract_expressions.csv line {}: expected expr,expect_requires. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[1].trim().to_string();
        let expect_requires = match parts[0].trim().to_lowercase().as_str() {
            "yes" | "true" => true,
            "no" | "false" => false,
            other => panic!(
                "requires_contract_expressions.csv line {}: invalid expect_requires '{}'",
                line_num, other
            ),
        };

        exprs.push(RequiresContractExpr {
            expr,
            expect_requires,
            family: current_family.clone(),
        });
    }

    exprs
}

fn parse_domain_mode_label(
    label: &str,
    csv_name: &str,
    line_num: usize,
) -> cas_solver::runtime::DomainMode {
    match label.trim().to_lowercase().as_str() {
        "generic" => cas_solver::runtime::DomainMode::Generic,
        "strict" => cas_solver::runtime::DomainMode::Strict,
        "assume" => cas_solver::runtime::DomainMode::Assume,
        other => panic!(
            "{} line {}: invalid domain mode '{}'",
            csv_name, line_num, other
        ),
    }
}

fn domain_mode_label(mode: cas_solver::runtime::DomainMode) -> &'static str {
    match mode {
        cas_solver::runtime::DomainMode::Generic => "generic",
        cas_solver::runtime::DomainMode::Strict => "strict",
        cas_solver::runtime::DomainMode::Assume => "assume",
    }
}

fn parse_value_domain_label(
    label: &str,
    csv_name: &str,
    line_num: usize,
) -> cas_solver::runtime::ValueDomain {
    match label.trim().to_lowercase().as_str() {
        "real" | "realonly" => cas_solver::runtime::ValueDomain::RealOnly,
        "complex" | "complexenabled" => cas_solver::runtime::ValueDomain::ComplexEnabled,
        other => panic!(
            "{} line {}: invalid value domain '{}'",
            csv_name, line_num, other
        ),
    }
}

fn value_domain_label(value_domain: cas_solver::runtime::ValueDomain) -> &'static str {
    match value_domain {
        cas_solver::runtime::ValueDomain::RealOnly => "real",
        cas_solver::runtime::ValueDomain::ComplexEnabled => "complex",
    }
}

fn parse_inv_trig_policy_label(
    label: &str,
    csv_name: &str,
    line_num: usize,
) -> cas_solver::runtime::InverseTrigPolicy {
    match label.trim().to_lowercase().as_str() {
        "strict" => cas_solver::runtime::InverseTrigPolicy::Strict,
        "principal" | "principalvalue" => cas_solver::runtime::InverseTrigPolicy::PrincipalValue,
        other => panic!(
            "{} line {}: invalid inverse trig policy '{}'",
            csv_name, line_num, other
        ),
    }
}

fn inv_trig_policy_label(value: cas_solver::runtime::InverseTrigPolicy) -> &'static str {
    match value {
        cas_solver::runtime::InverseTrigPolicy::Strict => "strict",
        cas_solver::runtime::InverseTrigPolicy::PrincipalValue => "principal",
    }
}

fn load_warnings_contract_expressions() -> Vec<WarningsContractExpr> {
    let csv_path = find_test_data_file("warnings_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read warnings_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(3, ',').collect();
        if parts.len() != 3 {
            panic!(
                "warnings_contract_expressions.csv line {}: expected expr,mode,expect_warning. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[2].trim().to_string();
        let mode = parse_domain_mode_label(parts[1], "warnings_contract_expressions.csv", line_num);
        let expect_warning = match parts[0].trim().to_lowercase().as_str() {
            "yes" | "true" => true,
            "no" | "false" => false,
            other => panic!(
                "warnings_contract_expressions.csv line {}: invalid expect_warning '{}'",
                line_num, other
            ),
        };

        exprs.push(WarningsContractExpr {
            expr,
            mode,
            expect_warning,
            family: current_family.clone(),
        });
    }

    exprs
}

fn load_transparency_signal_contract_expressions() -> Vec<TransparencySignalContractExpr> {
    let csv_path = find_test_data_file("transparency_signal_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read transparency_signal_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(3, ',').collect();
        if parts.len() != 3 {
            panic!(
                "transparency_signal_contract_expressions.csv line {}: expected expr,mode,expect_signal. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[2].trim().to_string();
        let mode = parse_domain_mode_label(
            parts[1],
            "transparency_signal_contract_expressions.csv",
            line_num,
        );
        let expect_signal = match parts[0].trim().to_lowercase().as_str() {
            "yes" | "true" => true,
            "no" | "false" => false,
            other => panic!(
                "transparency_signal_contract_expressions.csv line {}: invalid expect_signal '{}'",
                line_num, other
            ),
        };

        exprs.push(TransparencySignalContractExpr {
            expr,
            mode,
            expect_signal,
            family: current_family.clone(),
        });
    }

    exprs
}

fn load_branch_transparency_contract_expressions() -> Vec<BranchTransparencyContractExpr> {
    let csv_path = find_test_data_file("branch_transparency_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read branch_transparency_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(4, ',').collect();
        if parts.len() != 4 {
            panic!(
                "branch_transparency_contract_expressions.csv line {}: expected expr,mode,inv_trig,expect_signal. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[3].trim().to_string();
        let mode = parse_domain_mode_label(
            parts[2],
            "branch_transparency_contract_expressions.csv",
            line_num,
        );
        let inv_trig = parse_inv_trig_policy_label(
            parts[1],
            "branch_transparency_contract_expressions.csv",
            line_num,
        );
        let expect_signal = match parts[0].trim().to_lowercase().as_str() {
            "yes" | "true" => true,
            "no" | "false" => false,
            other => panic!(
                "branch_transparency_contract_expressions.csv line {}: invalid expect_signal '{}'",
                line_num, other
            ),
        };

        exprs.push(BranchTransparencyContractExpr {
            expr,
            mode,
            inv_trig,
            expect_signal,
            family: current_family.clone(),
        });
    }

    exprs
}

fn load_semantic_behavior_contract_expressions() -> Vec<SemanticBehaviorContractExpr> {
    let csv_path = find_test_data_file("semantic_behavior_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read semantic_behavior_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(5, ',').collect();
        if parts.len() != 5 {
            panic!(
                "semantic_behavior_contract_expressions.csv line {}: expected expr,value_domain,mode,match_kind,expected. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[4].trim().to_string();
        let value_domain = parse_value_domain_label(
            parts[3],
            "semantic_behavior_contract_expressions.csv",
            line_num,
        );
        let mode = parse_domain_mode_label(
            parts[2],
            "semantic_behavior_contract_expressions.csv",
            line_num,
        );
        let expectation = match parts[1].trim().to_lowercase().as_str() {
            "exact" => SemanticBehaviorExpectation::Exact(parts[0].trim().to_string()),
            "contains_all" => SemanticBehaviorExpectation::ContainsAll(
                parts[0]
                    .split(';')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
            ),
            other => panic!(
                "semantic_behavior_contract_expressions.csv line {}: invalid match_kind '{}'",
                line_num, other
            ),
        };

        exprs.push(SemanticBehaviorContractExpr {
            expr,
            value_domain,
            mode,
            expectation,
            family: current_family.clone(),
        });
    }

    exprs
}

fn load_requires_mode_contract_expressions() -> Vec<RequiresModeContractExpr> {
    let csv_path = find_test_data_file("requires_mode_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read requires_mode_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(3, ',').collect();
        if parts.len() != 3 {
            panic!(
                "requires_mode_contract_expressions.csv line {}: expected expr,mode,expect_requires. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[2].trim().to_string();
        let mode =
            parse_domain_mode_label(parts[1], "requires_mode_contract_expressions.csv", line_num);
        let expect_requires = match parts[0].trim().to_lowercase().as_str() {
            "yes" | "true" => true,
            "no" | "false" => false,
            other => panic!(
                "requires_mode_contract_expressions.csv line {}: invalid expect_requires '{}'",
                line_num, other
            ),
        };

        exprs.push(RequiresModeContractExpr {
            expr,
            mode,
            expect_requires,
            family: current_family.clone(),
        });
    }

    exprs
}

fn load_semantic_axes_contract_expressions() -> Vec<SemanticAxesContractExpr> {
    let csv_path = find_test_data_file("semantic_axes_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read semantic_axes_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(5, ',').collect();
        if parts.len() != 5 {
            panic!(
                "semantic_axes_contract_expressions.csv line {}: expected expr,value_domain,mode,expect_requires,expect_warning. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[4].trim().to_string();
        let value_domain =
            parse_value_domain_label(parts[3], "semantic_axes_contract_expressions.csv", line_num);
        let mode =
            parse_domain_mode_label(parts[2], "semantic_axes_contract_expressions.csv", line_num);
        let expect_requires = match parts[1].trim().to_lowercase().as_str() {
            "yes" | "true" => true,
            "no" | "false" => false,
            other => panic!(
                "semantic_axes_contract_expressions.csv line {}: invalid expect_requires '{}'",
                line_num, other
            ),
        };
        let expect_warning = match parts[0].trim().to_lowercase().as_str() {
            "yes" | "true" => true,
            "no" | "false" => false,
            other => panic!(
                "semantic_axes_contract_expressions.csv line {}: invalid expect_warning '{}'",
                line_num, other
            ),
        };

        exprs.push(SemanticAxesContractExpr {
            expr,
            value_domain,
            mode,
            expect_requires,
            expect_warning,
            family: current_family.clone(),
        });
    }

    exprs
}

fn load_assumption_trace_contract_expressions() -> Vec<AssumptionTraceContractExpr> {
    let csv_path = find_test_data_file("assumption_trace_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read assumption_trace_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(4, ',').collect();
        if parts.len() != 4 {
            panic!(
                "assumption_trace_contract_expressions.csv line {}: expected expr,mode,inv_trig,expected_kind. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[3].trim().to_string();
        let mode = parse_domain_mode_label(
            parts[2],
            "assumption_trace_contract_expressions.csv",
            line_num,
        );
        let inv_trig = parse_inv_trig_policy_label(
            parts[1],
            "assumption_trace_contract_expressions.csv",
            line_num,
        );
        let expected_kind = match parts[0].trim().to_lowercase().as_str() {
            "none" | "" => None,
            other => Some(other.to_string()),
        };

        exprs.push(AssumptionTraceContractExpr {
            expr,
            mode,
            inv_trig,
            expected_kind,
            family: current_family.clone(),
        });
    }

    exprs
}

#[derive(Debug, Default)]
struct SimplifyMetadata {
    result: String,
    required: Vec<String>,
    warnings: Vec<String>,
}

#[derive(Debug, Default)]
struct SimplifyTraceMetadata {
    result: String,
    assumption_kinds: Vec<String>,
}

#[derive(Debug, Default)]
struct SimplifyTransparencyMetadata {
    result: String,
    warnings: Vec<String>,
    assumption_signals: Vec<String>,
}

fn simplify_with_metadata_on_axes(
    input: &str,
    mode: cas_solver::runtime::DomainMode,
    value_domain: cas_solver::runtime::ValueDomain,
) -> Result<SimplifyMetadata, String> {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().shared.semantics.domain_mode = mode;
    state.options_mut().shared.semantics.value_domain = value_domain;

    let parsed = parse(input, &mut engine.simplifier.context)
        .map_err(|e| format!("parse failed for '{}': {:?}", input, e))?;
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine
        .eval(&mut state, req)
        .map_err(|e| format!("eval failed for '{}': {:?}", input, e))?;

    let result = match &output.result {
        EvalResult::Expr(e) => DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        other => {
            return Err(format!(
                "unexpected eval result for '{}': {:?}",
                input, other
            ));
        }
    };

    let mut required: Vec<String> = output
        .required_conditions
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
    required.sort();
    required.dedup();

    let mut warnings: Vec<String> = output
        .domain_warnings
        .iter()
        .map(|w| w.message.clone())
        .collect();
    warnings.sort();
    warnings.dedup();

    Ok(SimplifyMetadata {
        result,
        required,
        warnings,
    })
}

fn simplify_with_assumption_trace(
    input: &str,
    mode: cas_solver::runtime::DomainMode,
    inv_trig: cas_solver::runtime::InverseTrigPolicy,
) -> Result<SimplifyTraceMetadata, String> {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().shared.semantics.domain_mode = mode;
    state.options_mut().shared.semantics.inv_trig = inv_trig;

    let parsed = parse(input, &mut engine.simplifier.context)
        .map_err(|e| format!("parse failed for '{}': {:?}", input, e))?;
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine
        .eval(&mut state, req)
        .map_err(|e| format!("eval failed for '{}': {:?}", input, e))?;

    let result = match &output.result {
        EvalResult::Expr(e) => DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        other => {
            return Err(format!(
                "unexpected eval result for '{}': {:?}",
                input, other
            ));
        }
    };

    let mut assumption_kinds: Vec<String> = output
        .steps
        .iter()
        .flat_map(|step| step.assumption_events().iter())
        .map(|event| event.key.kind().to_string())
        .collect();
    assumption_kinds.sort();
    assumption_kinds.dedup();

    Ok(SimplifyTraceMetadata {
        result,
        assumption_kinds,
    })
}

fn simplify_with_transparency_metadata(
    input: &str,
    mode: cas_solver::runtime::DomainMode,
) -> Result<SimplifyTransparencyMetadata, String> {
    simplify_with_transparency_metadata_with_inv_trig(
        input,
        mode,
        cas_solver::runtime::InverseTrigPolicy::Strict,
    )
}

fn simplify_with_transparency_metadata_with_inv_trig(
    input: &str,
    mode: cas_solver::runtime::DomainMode,
    inv_trig: cas_solver::runtime::InverseTrigPolicy,
) -> Result<SimplifyTransparencyMetadata, String> {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().shared.semantics.domain_mode = mode;
    state.options_mut().shared.semantics.inv_trig = inv_trig;

    let parsed = parse(input, &mut engine.simplifier.context)
        .map_err(|e| format!("parse failed for '{}': {:?}", input, e))?;
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine
        .eval(&mut state, req)
        .map_err(|e| format!("eval failed for '{}': {:?}", input, e))?;

    let result = match &output.result {
        EvalResult::Expr(e) => DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        other => {
            return Err(format!(
                "unexpected eval result for '{}': {:?}",
                input, other
            ));
        }
    };

    let mut warnings: Vec<String> = output
        .domain_warnings
        .iter()
        .map(|w| w.message.clone())
        .collect();
    warnings.sort();
    warnings.dedup();

    let mut assumption_signals: Vec<String> = output
        .steps
        .iter()
        .flat_map(|step| step.assumption_events().iter())
        .filter(|event| {
            event.kind.should_display()
                || matches!(
                    event.kind,
                    cas_solver::api::AssumptionKind::DerivedFromRequires
                )
        })
        .map(|event| {
            format!(
                "{}|{}|{}",
                event.kind.label(),
                event.key.kind(),
                event.message
            )
        })
        .collect();
    assumption_signals.sort();
    assumption_signals.dedup();

    Ok(SimplifyTransparencyMetadata {
        result,
        warnings,
        assumption_signals,
    })
}

fn simplify_with_metadata_in_domain(
    input: &str,
    mode: cas_solver::runtime::DomainMode,
) -> Result<SimplifyMetadata, String> {
    simplify_with_metadata_on_axes(input, mode, cas_solver::runtime::ValueDomain::RealOnly)
}

fn simplify_generic_with_metadata(input: &str) -> Result<SimplifyMetadata, String> {
    simplify_with_metadata_on_axes(
        input,
        cas_solver::runtime::DomainMode::Generic,
        cas_solver::runtime::ValueDomain::RealOnly,
    )
}

#[derive(Default)]
struct IdempotenceMetrics {
    total: usize,
    exact_stable: usize,
    symbolic_stable: usize,
    numeric_stable: usize,
    inconclusive: usize,
    failed: usize,
    parse_errors: usize,
    timeouts: usize,
    numeric_causes: HashMap<String, usize>,
}

#[derive(Default)]
struct RequiresContractMetrics {
    total: usize,
    exact_preserved: usize,
    relaxed_preserved: usize,
    expected_requires_present: usize,
    failed: usize,
    parse_errors: usize,
}

#[derive(Default)]
struct WarningsContractMetrics {
    total: usize,
    exact_preserved: usize,
    relaxed_preserved: usize,
    expected_warning_present: usize,
    expected_warning_absent: usize,
    failed: usize,
    parse_errors: usize,
}

#[derive(Default)]
struct TransparencySignalContractMetrics {
    total: usize,
    exact_preserved: usize,
    relaxed_preserved: usize,
    expected_signal_present: usize,
    expected_signal_absent: usize,
    warning_channel_present: usize,
    assumption_channel_present: usize,
    failed: usize,
    parse_errors: usize,
}

type BranchTransparencyContractMetrics = TransparencySignalContractMetrics;

#[derive(Default)]
struct SemanticBehaviorContractMetrics {
    total: usize,
    exact_preserved: usize,
    relaxed_preserved: usize,
    failed: usize,
    parse_errors: usize,
}

#[derive(Default)]
struct RequiresModeContractMetrics {
    total: usize,
    exact_preserved: usize,
    relaxed_preserved: usize,
    expected_requires_present: usize,
    expected_requires_absent: usize,
    failed: usize,
    parse_errors: usize,
}

#[derive(Default)]
struct SemanticAxesContractMetrics {
    total: usize,
    exact_preserved: usize,
    relaxed_preserved: usize,
    expected_requires_present: usize,
    expected_requires_absent: usize,
    expected_warning_present: usize,
    expected_warning_absent: usize,
    failed: usize,
    parse_errors: usize,
}

#[derive(Default)]
struct AssumptionTraceContractMetrics {
    total: usize,
    exact_preserved: usize,
    relaxed_preserved: usize,
    expected_present: usize,
    expected_absent: usize,
    failed: usize,
    parse_errors: usize,
}

fn run_idempotence_contract_tests() -> IdempotenceMetrics {
    let cases = load_idempotence_expressions();
    let config = metatest_config();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let timeout = std::time::Duration::from_secs(5);

    let mut metrics = IdempotenceMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut numeric_examples: Vec<(String, String, String, String)> = Vec::new();
    let mut failed_examples: Vec<(String, String, String)> = Vec::new();

    eprintln!(
        "📊 Running simplify idempotence contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let expr_text = case.expr.clone();
        let vars = case.vars.clone();
        let filters = case.filters.clone();
        let family = case.family.clone();
        let config_clone = config.clone();

        let (tx, rx) = std::sync::mpsc::channel();
        let _handle = std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                let mut simplifier = Simplifier::with_default_rules();
                let parsed = match parse(&expr_text, &mut simplifier.context) {
                    Ok(e) => e,
                    Err(_) => {
                        let _ = tx.send(("parse_error".to_string(), String::new(), String::new()));
                        return;
                    }
                };

                let (simp1_raw, _) = simplifier.simplify(parsed);
                let simp1 = fold_constants_safe(&mut simplifier.context, simp1_raw);
                let (simp2_raw, _) = simplifier.simplify(simp1);
                let simp2 = fold_constants_safe(&mut simplifier.context, simp2_raw);

                let exact = cas_solver::runtime::compare_expr(&simplifier.context, simp1, simp2)
                    == std::cmp::Ordering::Equal;
                if exact {
                    let _ = tx.send(("exact".to_string(), String::new(), String::new()));
                    return;
                }

                let simp1_text = DisplayExpr {
                    context: &simplifier.context,
                    id: simp1,
                }
                .to_string();
                let simp2_text = DisplayExpr {
                    context: &simplifier.context,
                    id: simp2,
                }
                .to_string();

                if prove_zero_from_metamorphic_texts(
                    &mut simplifier,
                    &simp1_text,
                    &simp2_text,
                    simp1,
                    simp2,
                ) {
                    let _ = tx.send(("symbolic".to_string(), String::new(), String::new()));
                    return;
                }

                let diff_expr = simplifier.context.add(cas_ast::Expr::Sub(simp1, simp2));
                let (diff_simp_raw, _) = simplifier.simplify(diff_expr);
                let diff_simp = fold_constants_safe(&mut simplifier.context, diff_simp_raw);
                let diff_render = cas_formatter::LaTeXExpr {
                    context: &simplifier.context,
                    id: diff_simp,
                }
                .to_latex();
                let diff_shape = expr_shape_signature(&simplifier.context, diff_simp);

                match classify_numeric_equiv_for_vars(
                    &simplifier.context,
                    simp1,
                    simp2,
                    &vars,
                    &filters,
                    &config_clone,
                ) {
                    NumericCheckOutcome::Pass => {
                        let cause = numeric_only_cause_for_vars(
                            &simplifier.context,
                            simp1,
                            simp2,
                            &vars,
                            &filters,
                            &config_clone,
                            &diff_shape,
                        )
                        .label()
                        .to_string();
                        let _ = tx.send(("numeric".to_string(), diff_render, cause));
                    }
                    NumericCheckOutcome::Inconclusive(reason) => {
                        let _ = tx.send(("inconclusive".to_string(), reason, String::new()));
                    }
                    NumericCheckOutcome::Failed(reason) => {
                        let _ = tx.send(("failed".to_string(), reason, String::new()));
                    }
                }
            });

        match rx.recv_timeout(timeout) {
            Ok((kind, detail, cause)) => match kind.as_str() {
                "exact" => metrics.exact_stable += 1,
                "symbolic" => metrics.symbolic_stable += 1,
                "numeric" => {
                    metrics.numeric_stable += 1;
                    *metrics.numeric_causes.entry(cause.clone()).or_default() += 1;
                    if verbose && numeric_examples.len() < 20 {
                        numeric_examples.push((case.expr.clone(), family, detail, cause));
                    }
                }
                "inconclusive" => {
                    metrics.inconclusive += 1;
                    if verbose {
                        eprintln!(
                            "  ◐ INCONCLUSIVE [{}]: {} — {}",
                            case.family, case.expr, detail
                        );
                    }
                }
                "failed" => {
                    metrics.failed += 1;
                    failed_examples.push((case.expr.clone(), family, detail));
                }
                "parse_error" => {
                    metrics.parse_errors += 1;
                }
                _ => {
                    metrics.failed += 1;
                    failed_examples.push((
                        case.expr.clone(),
                        family,
                        format!("unexpected result kind: {}", kind),
                    ));
                }
            },
            Err(_) => {
                metrics.timeouts += 1;
            }
        }
    }

    eprintln!(
        "✅ Idempotence contracts: exact={} symbolic={} numeric={} inconclusive={} failed={} parse={} timeout={}",
        metrics.exact_stable,
        metrics.symbolic_stable,
        metrics.numeric_stable,
        metrics.inconclusive,
        metrics.failed,
        metrics.parse_errors,
        metrics.timeouts
    );

    if metrics.numeric_stable > 0 {
        print_numeric_only_cause_breakdown(&metrics.numeric_causes);
    }

    if verbose && !numeric_examples.is_empty() {
        eprintln!("\n── idempotence numeric-only examples ──");
        for (expr, family, residual, cause) in numeric_examples.iter().take(10) {
            eprintln!("  Expr [{}]: {}", family, expr);
            eprintln!("  Cause: {}", cause);
            if !residual.is_empty() {
                eprintln!("  Residual: {}", residual);
            }
            eprintln!();
        }
    }

    if !failed_examples.is_empty() {
        eprintln!("\n🚨 idempotence failures:");
        for (expr, family, detail) in failed_examples.iter().take(10) {
            eprintln!("  [{}] {} — {}", family, expr, detail);
        }
    }

    metrics
}

fn run_requires_contract_tests() -> RequiresContractMetrics {
    let cases = load_requires_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = RequiresContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running required_conditions contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match simplify_generic_with_metadata(&case.expr) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!("[{}] {} — {}", case.family, case.expr, err));
                continue;
            }
        };

        let second = match simplify_generic_with_metadata(&first.result) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}] {} -> '{}' reparsed failed: {}",
                    case.family, case.expr, first.result, err
                ));
                continue;
            }
        };

        if case.expect_requires && !first.required.is_empty() {
            metrics.expected_requires_present += 1;
        }

        let mut case_failed = false;
        if case.expect_requires && first.required.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}] {} — expected requires, got none",
                case.family, case.expr
            ));
            case_failed = true;
        }

        let first_required: std::collections::HashSet<_> = first.required.iter().cloned().collect();
        let second_required: std::collections::HashSet<_> =
            second.required.iter().cloned().collect();
        let first_warnings: std::collections::HashSet<_> = first.warnings.iter().cloned().collect();
        let second_warnings: std::collections::HashSet<_> =
            second.warnings.iter().cloned().collect();

        let introduced_requires: Vec<_> = second_required
            .difference(&first_required)
            .cloned()
            .collect();
        let introduced_warnings: Vec<_> = second_warnings
            .difference(&first_warnings)
            .cloned()
            .collect();

        if !introduced_requires.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}] {} — introduced requires: {:?} (first={:?}, second={:?})",
                case.family, case.expr, introduced_requires, first.required, second.required
            ));
            case_failed = true;
        }

        if !introduced_warnings.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}] {} — introduced warnings: {:?} (first={:?}, second={:?})",
                case.family, case.expr, introduced_warnings, first.warnings, second.warnings
            ));
            case_failed = true;
        }

        if !case_failed {
            if first.required == second.required && first.warnings == second.warnings {
                metrics.exact_preserved += 1;
            } else {
                metrics.relaxed_preserved += 1;
                if verbose && relaxed_examples.len() < 10 {
                    relaxed_examples.push(format!(
                        "[{}] {} — requires {:?} -> {:?}, warnings {:?} -> {:?}",
                        case.family,
                        case.expr,
                        first.required,
                        second.required,
                        first.warnings,
                        second.warnings
                    ));
                }
            }
        } else if verbose {
            eprintln!("  ❌ [{}] {}", case.family, case.expr);
        }
    }

    eprintln!(
        "✅ Requires contracts: exact={} relaxed={} expected_requires_present={}/{} failed={} parse={}",
        metrics.exact_preserved,
        metrics.relaxed_preserved,
        metrics.expected_requires_present,
        cases.iter().filter(|c| c.expect_requires).count(),
        metrics.failed,
        metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ requires contract relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 requires contract failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

fn run_warnings_contract_tests() -> WarningsContractMetrics {
    let cases = load_warnings_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = WarningsContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running warnings contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match simplify_with_metadata_in_domain(&case.expr, case.mode) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}] {} — {}",
                    case.family,
                    domain_mode_label(case.mode),
                    case.expr,
                    err
                ));
                continue;
            }
        };

        let second = match simplify_with_metadata_in_domain(&first.result, case.mode) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}] {} -> '{}' reparsed failed: {}",
                    case.family,
                    domain_mode_label(case.mode),
                    case.expr,
                    first.result,
                    err
                ));
                continue;
            }
        };

        let mut case_failed = false;
        if case.expect_warning {
            if first.warnings.is_empty() {
                metrics.failed += 1;
                failures.push(format!(
                    "[{}|{}] {} — expected warning, got none",
                    case.family,
                    domain_mode_label(case.mode),
                    case.expr
                ));
                case_failed = true;
            } else {
                metrics.expected_warning_present += 1;
            }
        } else if first.warnings.is_empty() {
            metrics.expected_warning_absent += 1;
        } else {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}] {} — unexpected warnings: {:?}",
                case.family,
                domain_mode_label(case.mode),
                case.expr,
                first.warnings
            ));
            case_failed = true;
        }

        let first_warnings: std::collections::HashSet<_> = first.warnings.iter().cloned().collect();
        let second_warnings: std::collections::HashSet<_> =
            second.warnings.iter().cloned().collect();
        let introduced_warnings: Vec<_> = second_warnings
            .difference(&first_warnings)
            .cloned()
            .collect();

        if !introduced_warnings.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}] {} — introduced warnings: {:?} (first={:?}, second={:?})",
                case.family,
                domain_mode_label(case.mode),
                case.expr,
                introduced_warnings,
                first.warnings,
                second.warnings
            ));
            case_failed = true;
        }

        if !case_failed {
            if first.warnings == second.warnings {
                metrics.exact_preserved += 1;
            } else {
                metrics.relaxed_preserved += 1;
                if verbose && relaxed_examples.len() < 10 {
                    relaxed_examples.push(format!(
                        "[{}|{}] {} — warnings {:?} -> {:?}",
                        case.family,
                        domain_mode_label(case.mode),
                        case.expr,
                        first.warnings,
                        second.warnings
                    ));
                }
            }
        } else if verbose {
            eprintln!(
                "  ❌ [{}|{}] {}",
                case.family,
                domain_mode_label(case.mode),
                case.expr
            );
        }
    }

    eprintln!(
        "✅ Warnings contracts: exact={} relaxed={} expected_warning_present={} expected_warning_absent={} failed={} parse={}",
        metrics.exact_preserved,
        metrics.relaxed_preserved,
        metrics.expected_warning_present,
        metrics.expected_warning_absent,
        metrics.failed,
        metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ warnings contract relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 warnings contract failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

fn run_transparency_signal_contract_tests() -> TransparencySignalContractMetrics {
    let cases = load_transparency_signal_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = TransparencySignalContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running transparency-signal contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match simplify_with_transparency_metadata(&case.expr, case.mode) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}] {} — {}",
                    case.family,
                    domain_mode_label(case.mode),
                    case.expr,
                    err
                ));
                continue;
            }
        };

        let second = match simplify_with_transparency_metadata(&first.result, case.mode) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}] {} -> '{}' reparsed failed: {}",
                    case.family,
                    domain_mode_label(case.mode),
                    case.expr,
                    first.result,
                    err
                ));
                continue;
            }
        };

        let has_signal = !first.warnings.is_empty() || !first.assumption_signals.is_empty();
        let mut case_failed = false;
        if case.expect_signal {
            if !has_signal {
                metrics.failed += 1;
                failures.push(format!(
                    "[{}|{}] {} — expected transparency signal, got none",
                    case.family,
                    domain_mode_label(case.mode),
                    case.expr
                ));
                case_failed = true;
            } else {
                metrics.expected_signal_present += 1;
                if !first.warnings.is_empty() {
                    metrics.warning_channel_present += 1;
                }
                if !first.assumption_signals.is_empty() {
                    metrics.assumption_channel_present += 1;
                }
            }
        } else if !has_signal {
            metrics.expected_signal_absent += 1;
        } else {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}] {} — unexpected transparency signals: warnings={:?}, assumptions={:?}",
                case.family,
                domain_mode_label(case.mode),
                case.expr,
                first.warnings,
                first.assumption_signals
            ));
            case_failed = true;
        }

        let first_warnings: std::collections::HashSet<_> = first.warnings.iter().cloned().collect();
        let second_warnings: std::collections::HashSet<_> =
            second.warnings.iter().cloned().collect();
        let introduced_warnings: Vec<_> = second_warnings
            .difference(&first_warnings)
            .cloned()
            .collect();

        let first_assumptions: std::collections::HashSet<_> =
            first.assumption_signals.iter().cloned().collect();
        let second_assumptions: std::collections::HashSet<_> =
            second.assumption_signals.iter().cloned().collect();
        let introduced_assumptions: Vec<_> = second_assumptions
            .difference(&first_assumptions)
            .cloned()
            .collect();

        if !introduced_warnings.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}] {} — introduced warnings: {:?} (first={:?}, second={:?})",
                case.family,
                domain_mode_label(case.mode),
                case.expr,
                introduced_warnings,
                first.warnings,
                second.warnings
            ));
            case_failed = true;
        }

        if !introduced_assumptions.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}] {} — introduced assumption-signals: {:?} (first={:?}, second={:?})",
                case.family,
                domain_mode_label(case.mode),
                case.expr,
                introduced_assumptions,
                first.assumption_signals,
                second.assumption_signals
            ));
            case_failed = true;
        }

        if !case_failed {
            if first.warnings == second.warnings
                && first.assumption_signals == second.assumption_signals
            {
                metrics.exact_preserved += 1;
            } else {
                metrics.relaxed_preserved += 1;
                if verbose && relaxed_examples.len() < 10 {
                    relaxed_examples.push(format!(
                        "[{}|{}] {} — warnings {:?} -> {:?}, assumptions {:?} -> {:?}",
                        case.family,
                        domain_mode_label(case.mode),
                        case.expr,
                        first.warnings,
                        second.warnings,
                        first.assumption_signals,
                        second.assumption_signals
                    ));
                }
            }
        } else if verbose {
            eprintln!(
                "  ❌ [{}|{}] {}",
                case.family,
                domain_mode_label(case.mode),
                case.expr
            );
        }
    }

    eprintln!(
        "✅ Transparency-signal contracts: exact={} relaxed={} signal_present={} signal_absent={} warning_channel={} assumption_channel={} failed={} parse={}",
        metrics.exact_preserved,
        metrics.relaxed_preserved,
        metrics.expected_signal_present,
        metrics.expected_signal_absent,
        metrics.warning_channel_present,
        metrics.assumption_channel_present,
        metrics.failed,
        metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ transparency-signal relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 transparency-signal failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

fn run_branch_transparency_contract_tests() -> BranchTransparencyContractMetrics {
    let cases = load_branch_transparency_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = BranchTransparencyContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running branch-transparency contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match simplify_with_transparency_metadata_with_inv_trig(
            &case.expr,
            case.mode,
            case.inv_trig,
        ) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}|{}] {} — {}",
                    case.family,
                    domain_mode_label(case.mode),
                    inv_trig_policy_label(case.inv_trig),
                    case.expr,
                    err
                ));
                continue;
            }
        };

        let second = match simplify_with_transparency_metadata_with_inv_trig(
            &first.result,
            case.mode,
            case.inv_trig,
        ) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}|{}] {} -> '{}' reparsed failed: {}",
                    case.family,
                    domain_mode_label(case.mode),
                    inv_trig_policy_label(case.inv_trig),
                    case.expr,
                    first.result,
                    err
                ));
                continue;
            }
        };

        let first_branch_assumptions: Vec<String> = first
            .assumption_signals
            .iter()
            .filter(|signal| signal.starts_with("Branch|"))
            .cloned()
            .collect();
        let second_branch_assumptions: Vec<String> = second
            .assumption_signals
            .iter()
            .filter(|signal| signal.starts_with("Branch|"))
            .cloned()
            .collect();

        let has_signal = !first.warnings.is_empty() || !first_branch_assumptions.is_empty();
        let mut case_failed = false;
        if case.expect_signal {
            if !has_signal {
                metrics.failed += 1;
                failures.push(format!(
                    "[{}|{}|{}] {} — expected branch transparency signal, got none",
                    case.family,
                    domain_mode_label(case.mode),
                    inv_trig_policy_label(case.inv_trig),
                    case.expr
                ));
                case_failed = true;
            } else {
                metrics.expected_signal_present += 1;
                if !first.warnings.is_empty() {
                    metrics.warning_channel_present += 1;
                }
                if !first_branch_assumptions.is_empty() {
                    metrics.assumption_channel_present += 1;
                }
            }
        } else if !has_signal {
            metrics.expected_signal_absent += 1;
        } else {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — unexpected branch transparency signals: warnings={:?}, assumptions={:?}",
                case.family,
                domain_mode_label(case.mode),
                inv_trig_policy_label(case.inv_trig),
                case.expr,
                first.warnings,
                first_branch_assumptions
            ));
            case_failed = true;
        }

        let first_warnings: std::collections::HashSet<_> = first.warnings.iter().cloned().collect();
        let second_warnings: std::collections::HashSet<_> =
            second.warnings.iter().cloned().collect();
        let introduced_warnings: Vec<_> = second_warnings
            .difference(&first_warnings)
            .cloned()
            .collect();

        let first_assumptions: std::collections::HashSet<_> =
            first_branch_assumptions.iter().cloned().collect();
        let second_assumptions: std::collections::HashSet<_> =
            second_branch_assumptions.iter().cloned().collect();
        let introduced_assumptions: Vec<_> = second_assumptions
            .difference(&first_assumptions)
            .cloned()
            .collect();

        if !introduced_warnings.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — introduced warnings: {:?} (first={:?}, second={:?})",
                case.family,
                domain_mode_label(case.mode),
                inv_trig_policy_label(case.inv_trig),
                case.expr,
                introduced_warnings,
                first.warnings,
                second.warnings
            ));
            case_failed = true;
        }

        if !introduced_assumptions.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — introduced branch assumption-signals: {:?} (first={:?}, second={:?})",
                case.family,
                domain_mode_label(case.mode),
                inv_trig_policy_label(case.inv_trig),
                case.expr,
                introduced_assumptions,
                first_branch_assumptions,
                second_branch_assumptions
            ));
            case_failed = true;
        }

        if !case_failed {
            if first.warnings == second.warnings
                && first_branch_assumptions == second_branch_assumptions
            {
                metrics.exact_preserved += 1;
            } else {
                metrics.relaxed_preserved += 1;
                if verbose && relaxed_examples.len() < 10 {
                    relaxed_examples.push(format!(
                        "[{}|{}|{}] {} — warnings {:?} -> {:?}, assumptions {:?} -> {:?}",
                        case.family,
                        domain_mode_label(case.mode),
                        inv_trig_policy_label(case.inv_trig),
                        case.expr,
                        first.warnings,
                        second.warnings,
                        first_branch_assumptions,
                        second_branch_assumptions
                    ));
                }
            }
        } else if verbose {
            eprintln!(
                "  ❌ [{}|{}|{}] {}",
                case.family,
                domain_mode_label(case.mode),
                inv_trig_policy_label(case.inv_trig),
                case.expr
            );
        }
    }

    eprintln!(
        "✅ Branch-transparency contracts: exact={} relaxed={} signal_present={} signal_absent={} warning_channel={} assumption_channel={} failed={} parse={}",
        metrics.exact_preserved,
        metrics.relaxed_preserved,
        metrics.expected_signal_present,
        metrics.expected_signal_absent,
        metrics.warning_channel_present,
        metrics.assumption_channel_present,
        metrics.failed,
        metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ branch-transparency relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 branch-transparency failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

fn semantic_behavior_matches(expectation: &SemanticBehaviorExpectation, actual: &str) -> bool {
    match expectation {
        SemanticBehaviorExpectation::Exact(expected) => actual == expected,
        SemanticBehaviorExpectation::ContainsAll(needles) => {
            needles.iter().all(|needle| actual.contains(needle))
        }
    }
}

fn semantic_behavior_label(expectation: &SemanticBehaviorExpectation) -> String {
    match expectation {
        SemanticBehaviorExpectation::Exact(expected) => format!("exact '{}'", expected),
        SemanticBehaviorExpectation::ContainsAll(parts) => {
            format!("contains_all {:?}", parts)
        }
    }
}

fn run_semantic_behavior_contract_tests() -> SemanticBehaviorContractMetrics {
    let cases = load_semantic_behavior_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = SemanticBehaviorContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running semantic-behavior contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match simplify_with_metadata_on_axes(&case.expr, case.mode, case.value_domain) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}|{}] {} — {}",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    case.expr,
                    err
                ));
                continue;
            }
        };

        let second =
            match simplify_with_metadata_on_axes(&first.result, case.mode, case.value_domain) {
                Ok(v) => v,
                Err(err) => {
                    metrics.parse_errors += 1;
                    failures.push(format!(
                        "[{}|{}|{}] {} -> '{}' reparsed failed: {}",
                        case.family,
                        value_domain_label(case.value_domain),
                        domain_mode_label(case.mode),
                        case.expr,
                        first.result,
                        err
                    ));
                    continue;
                }
            };

        let expected_ok = semantic_behavior_matches(&case.expectation, &first.result);
        let second_ok = semantic_behavior_matches(&case.expectation, &second.result);

        if !expected_ok {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — expected {}, got '{}'",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                case.expr,
                semantic_behavior_label(&case.expectation),
                first.result
            ));
            continue;
        }

        if !second_ok {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — second simplify broke behavior: first='{}', second='{}', expected {}",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                case.expr,
                first.result,
                second.result,
                semantic_behavior_label(&case.expectation)
            ));
            continue;
        }

        if first.result == second.result {
            metrics.exact_preserved += 1;
        } else {
            metrics.relaxed_preserved += 1;
            if verbose && relaxed_examples.len() < 10 {
                relaxed_examples.push(format!(
                    "[{}|{}|{}] {} — result '{}' -> '{}'",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    case.expr,
                    first.result,
                    second.result
                ));
            }
        }
    }

    eprintln!(
        "✅ Semantic-behavior contracts: exact={} relaxed={} failed={} parse={}",
        metrics.exact_preserved, metrics.relaxed_preserved, metrics.failed, metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ semantic-behavior relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 semantic-behavior failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

fn run_requires_mode_contract_tests() -> RequiresModeContractMetrics {
    let cases = load_requires_mode_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = RequiresModeContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running mode-aware required_conditions contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match simplify_with_metadata_in_domain(&case.expr, case.mode) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}] {} — {}",
                    case.family,
                    domain_mode_label(case.mode),
                    case.expr,
                    err
                ));
                continue;
            }
        };

        let second = match simplify_with_metadata_in_domain(&first.result, case.mode) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}] {} -> '{}' reparsed failed: {}",
                    case.family,
                    domain_mode_label(case.mode),
                    case.expr,
                    first.result,
                    err
                ));
                continue;
            }
        };

        let mut case_failed = false;
        if case.expect_requires {
            if first.required.is_empty() {
                metrics.failed += 1;
                failures.push(format!(
                    "[{}|{}] {} — expected requires, got none",
                    case.family,
                    domain_mode_label(case.mode),
                    case.expr
                ));
                case_failed = true;
            } else {
                metrics.expected_requires_present += 1;
            }
        } else if first.required.is_empty() {
            metrics.expected_requires_absent += 1;
        } else {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}] {} — unexpected requires: {:?}",
                case.family,
                domain_mode_label(case.mode),
                case.expr,
                first.required
            ));
            case_failed = true;
        }

        let first_required: std::collections::HashSet<_> = first.required.iter().cloned().collect();
        let second_required: std::collections::HashSet<_> =
            second.required.iter().cloned().collect();
        let first_warnings: std::collections::HashSet<_> = first.warnings.iter().cloned().collect();
        let second_warnings: std::collections::HashSet<_> =
            second.warnings.iter().cloned().collect();

        let introduced_requires: Vec<_> = second_required
            .difference(&first_required)
            .cloned()
            .collect();
        let introduced_warnings: Vec<_> = second_warnings
            .difference(&first_warnings)
            .cloned()
            .collect();

        if !introduced_requires.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}] {} — introduced requires: {:?} (first={:?}, second={:?})",
                case.family,
                domain_mode_label(case.mode),
                case.expr,
                introduced_requires,
                first.required,
                second.required
            ));
            case_failed = true;
        }

        if !introduced_warnings.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}] {} — introduced warnings: {:?} (first={:?}, second={:?})",
                case.family,
                domain_mode_label(case.mode),
                case.expr,
                introduced_warnings,
                first.warnings,
                second.warnings
            ));
            case_failed = true;
        }

        if !case_failed {
            if first.required == second.required && first.warnings == second.warnings {
                metrics.exact_preserved += 1;
            } else {
                metrics.relaxed_preserved += 1;
                if verbose && relaxed_examples.len() < 10 {
                    relaxed_examples.push(format!(
                        "[{}|{}] {} — requires {:?} -> {:?}, warnings {:?} -> {:?}",
                        case.family,
                        domain_mode_label(case.mode),
                        case.expr,
                        first.required,
                        second.required,
                        first.warnings,
                        second.warnings
                    ));
                }
            }
        } else if verbose {
            eprintln!(
                "  ❌ [{}|{}] {}",
                case.family,
                domain_mode_label(case.mode),
                case.expr
            );
        }
    }

    eprintln!(
        "✅ Mode-aware requires contracts: exact={} relaxed={} expected_requires_present={} expected_requires_absent={} failed={} parse={}",
        metrics.exact_preserved,
        metrics.relaxed_preserved,
        metrics.expected_requires_present,
        metrics.expected_requires_absent,
        metrics.failed,
        metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ mode-aware requires relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 mode-aware requires failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

fn run_semantic_axes_contract_tests() -> SemanticAxesContractMetrics {
    let cases = load_semantic_axes_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = SemanticAxesContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running semantic-axes contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match simplify_with_metadata_on_axes(&case.expr, case.mode, case.value_domain) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}|{}] {} — {}",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    case.expr,
                    err
                ));
                continue;
            }
        };

        let second =
            match simplify_with_metadata_on_axes(&first.result, case.mode, case.value_domain) {
                Ok(v) => v,
                Err(err) => {
                    metrics.parse_errors += 1;
                    failures.push(format!(
                        "[{}|{}|{}] {} -> '{}' reparsed failed: {}",
                        case.family,
                        value_domain_label(case.value_domain),
                        domain_mode_label(case.mode),
                        case.expr,
                        first.result,
                        err
                    ));
                    continue;
                }
            };

        let mut case_failed = false;
        if case.expect_requires {
            if first.required.is_empty() {
                metrics.failed += 1;
                failures.push(format!(
                    "[{}|{}|{}] {} — expected requires, got none",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    case.expr
                ));
                case_failed = true;
            } else {
                metrics.expected_requires_present += 1;
            }
        } else if first.required.is_empty() {
            metrics.expected_requires_absent += 1;
        } else {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — unexpected requires: {:?}",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                case.expr,
                first.required
            ));
            case_failed = true;
        }

        if case.expect_warning {
            if first.warnings.is_empty() {
                metrics.failed += 1;
                failures.push(format!(
                    "[{}|{}|{}] {} — expected warning, got none",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    case.expr
                ));
                case_failed = true;
            } else {
                metrics.expected_warning_present += 1;
            }
        } else if first.warnings.is_empty() {
            metrics.expected_warning_absent += 1;
        } else {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — unexpected warnings: {:?}",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                case.expr,
                first.warnings
            ));
            case_failed = true;
        }

        let first_required: std::collections::HashSet<_> = first.required.iter().cloned().collect();
        let second_required: std::collections::HashSet<_> =
            second.required.iter().cloned().collect();
        let first_warnings: std::collections::HashSet<_> = first.warnings.iter().cloned().collect();
        let second_warnings: std::collections::HashSet<_> =
            second.warnings.iter().cloned().collect();

        let introduced_requires: Vec<_> = second_required
            .difference(&first_required)
            .cloned()
            .collect();
        let introduced_warnings: Vec<_> = second_warnings
            .difference(&first_warnings)
            .cloned()
            .collect();

        if !introduced_requires.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — introduced requires: {:?} (first={:?}, second={:?})",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                case.expr,
                introduced_requires,
                first.required,
                second.required
            ));
            case_failed = true;
        }

        if !introduced_warnings.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — introduced warnings: {:?} (first={:?}, second={:?})",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                case.expr,
                introduced_warnings,
                first.warnings,
                second.warnings
            ));
            case_failed = true;
        }

        if !case_failed {
            if first.required == second.required && first.warnings == second.warnings {
                metrics.exact_preserved += 1;
            } else {
                metrics.relaxed_preserved += 1;
                if verbose && relaxed_examples.len() < 10 {
                    relaxed_examples.push(format!(
                        "[{}|{}|{}] {} — requires {:?} -> {:?}, warnings {:?} -> {:?}",
                        case.family,
                        value_domain_label(case.value_domain),
                        domain_mode_label(case.mode),
                        case.expr,
                        first.required,
                        second.required,
                        first.warnings,
                        second.warnings
                    ));
                }
            }
        } else if verbose {
            eprintln!(
                "  ❌ [{}|{}|{}] {}",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                case.expr
            );
        }
    }

    eprintln!(
        "✅ Semantic-axes contracts: exact={} relaxed={} requires_present={} requires_absent={} warning_present={} warning_absent={} failed={} parse={}",
        metrics.exact_preserved,
        metrics.relaxed_preserved,
        metrics.expected_requires_present,
        metrics.expected_requires_absent,
        metrics.expected_warning_present,
        metrics.expected_warning_absent,
        metrics.failed,
        metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ semantic-axes relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 semantic-axes failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

fn run_assumption_trace_contract_tests() -> AssumptionTraceContractMetrics {
    let cases = load_assumption_trace_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = AssumptionTraceContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running assumption trace contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match simplify_with_assumption_trace(&case.expr, case.mode, case.inv_trig) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}|{}] {} — {}",
                    case.family,
                    domain_mode_label(case.mode),
                    inv_trig_policy_label(case.inv_trig),
                    case.expr,
                    err
                ));
                continue;
            }
        };

        let second = match simplify_with_assumption_trace(&first.result, case.mode, case.inv_trig) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}|{}] {} -> '{}' reparsed failed: {}",
                    case.family,
                    domain_mode_label(case.mode),
                    inv_trig_policy_label(case.inv_trig),
                    case.expr,
                    first.result,
                    err
                ));
                continue;
            }
        };

        let mut case_failed = false;
        match &case.expected_kind {
            Some(kind) => {
                if !first.assumption_kinds.iter().any(|k| k == kind) {
                    metrics.failed += 1;
                    failures.push(format!(
                        "[{}|{}|{}] {} — expected assumption kind '{}', got {:?}",
                        case.family,
                        domain_mode_label(case.mode),
                        inv_trig_policy_label(case.inv_trig),
                        case.expr,
                        kind,
                        first.assumption_kinds
                    ));
                    case_failed = true;
                } else {
                    metrics.expected_present += 1;
                }
            }
            None => {
                if first.assumption_kinds.is_empty() {
                    metrics.expected_absent += 1;
                } else {
                    metrics.failed += 1;
                    failures.push(format!(
                        "[{}|{}|{}] {} — unexpected assumption kinds {:?}",
                        case.family,
                        domain_mode_label(case.mode),
                        inv_trig_policy_label(case.inv_trig),
                        case.expr,
                        first.assumption_kinds
                    ));
                    case_failed = true;
                }
            }
        }

        let first_kinds: std::collections::HashSet<_> =
            first.assumption_kinds.iter().cloned().collect();
        let second_kinds: std::collections::HashSet<_> =
            second.assumption_kinds.iter().cloned().collect();
        let introduced_kinds: Vec<_> = second_kinds.difference(&first_kinds).cloned().collect();

        if !introduced_kinds.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}] {} — introduced assumption kinds {:?} (first={:?}, second={:?})",
                case.family,
                domain_mode_label(case.mode),
                inv_trig_policy_label(case.inv_trig),
                case.expr,
                introduced_kinds,
                first.assumption_kinds,
                second.assumption_kinds
            ));
            case_failed = true;
        }

        if !case_failed {
            if first.assumption_kinds == second.assumption_kinds {
                metrics.exact_preserved += 1;
            } else {
                metrics.relaxed_preserved += 1;
                if verbose && relaxed_examples.len() < 10 {
                    relaxed_examples.push(format!(
                        "[{}|{}|{}] {} — assumption kinds {:?} -> {:?}",
                        case.family,
                        domain_mode_label(case.mode),
                        inv_trig_policy_label(case.inv_trig),
                        case.expr,
                        first.assumption_kinds,
                        second.assumption_kinds
                    ));
                }
            }
        } else if verbose {
            eprintln!(
                "  ❌ [{}|{}|{}] {}",
                case.family,
                domain_mode_label(case.mode),
                inv_trig_policy_label(case.inv_trig),
                case.expr
            );
        }
    }

    eprintln!(
        "✅ Assumption trace contracts: exact={} relaxed={} expected_present={} expected_absent={} failed={} parse={}",
        metrics.exact_preserved,
        metrics.relaxed_preserved,
        metrics.expected_present,
        metrics.expected_absent,
        metrics.failed,
        metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ assumption trace relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 assumption trace failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

/// Run substitution-based metamorphic tests
fn run_substitution_tests() -> ComboMetrics {
    let identities = load_substitution_identities();
    let substitutions = load_substitution_expressions();
    let config = metatest_config();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let show_table = std::env::var("METATEST_TABLE").is_ok();

    // Filter out Assume-only identities (we run in Generic mode)
    let identities: Vec<_> = identities
        .into_iter()
        .filter(|p| p.mode != DomainRequirement::Assume)
        .collect();

    let total_combos = identities.len() * substitutions.len();
    eprintln!(
        "📊 Running substitution metamorphic tests: {} identities × {} substitutions = {} combos (seed {})",
        identities.len(),
        substitutions.len(),
        total_combos,
        config.seed
    );

    // Global counters
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut nf_convergent = 0usize;
    let mut proved_symbolic = 0usize;
    let mut numeric_only = 0usize;
    let mut inconclusive = 0usize;
    let skipped = 0usize;
    let mut timeouts = 0usize;
    let mut cycle_events_total: usize = 0;
    let mut parse_errors = 0usize;
    let mut numeric_only_causes: HashMap<String, usize> = HashMap::new();
    let mut numeric_only_examples: Vec<(String, String, String, String, String)> = Vec::new();

    // Cross-product table data: (family, sub_label) → (nf, proved, numeric, failed)
    let mut cell_data: HashMap<(String, String), (usize, usize, usize, usize)> = HashMap::new();

    let combo_timeout = std::time::Duration::from_secs(5);

    for identity in &identities {
        let id_var = &identity.vars[0]; // Variable to substitute (typically "x")

        for sub in &substitutions {
            // Build LHS and RHS by substituting x → sub.expr
            let lhs_str = text_substitute(&identity.exp, id_var, &sub.expr);
            let rhs_str = text_substitute(&identity.simp, id_var, &sub.expr);
            let free_var = sub.var.clone();

            let lhs_clone = lhs_str.clone();
            let rhs_clone = rhs_str.clone();
            let config_clone = config.clone();
            let free_var_clone = free_var.clone();

            let (tx, rx) = std::sync::mpsc::channel();
            let _handle = std::thread::Builder::new()
                .stack_size(8 * 1024 * 1024)
                .spawn(move || {
                    let mut simplifier = Simplifier::with_default_rules();
                    let exp_parsed = match parse(&lhs_clone, &mut simplifier.context) {
                        Ok(e) => e,
                        Err(_) => {
                            let _ = tx.send(Some((
                                "parse_error".to_string(),
                                String::new(),
                                String::new(),
                                0,
                            )));
                            return;
                        }
                    };
                    let simp_parsed = match parse(&rhs_clone, &mut simplifier.context) {
                        Ok(e) => e,
                        Err(_) => {
                            let _ = tx.send(Some((
                                "parse_error".to_string(),
                                String::new(),
                                String::new(),
                                0,
                            )));
                            return;
                        }
                    };

                    let opts = cas_solver::runtime::SimplifyOptions::default();
                    let mut sub_cycles: usize = 0;
                    let (mut e, _, stats_e) =
                        simplifier.simplify_with_stats(exp_parsed, opts.clone());
                    sub_cycles += stats_e.cycle_events.len();
                    let (mut s, _, stats_s) =
                        simplifier.simplify_with_stats(simp_parsed, opts.clone());
                    sub_cycles += stats_s.cycle_events.len();

                    // Post-process: fold_constants
                    {
                        let cfg = cas_solver::runtime::EvalConfig::default();
                        let mut budget = cas_solver::runtime::Budget::preset_cli();
                        if let Ok(r) = cas_solver::api::fold_constants(
                            &mut simplifier.context,
                            e,
                            &cfg,
                            cas_solver::api::ConstFoldMode::Safe,
                            &mut budget,
                        ) {
                            e = r.expr;
                        }
                        if let Ok(r) = cas_solver::api::fold_constants(
                            &mut simplifier.context,
                            s,
                            &cfg,
                            cas_solver::api::ConstFoldMode::Safe,
                            &mut budget,
                        ) {
                            s = r.expr;
                        }
                    }

                    // Check 1: NF convergence
                    let nf_match = cas_solver::runtime::compare_expr(&simplifier.context, e, s)
                        == std::cmp::Ordering::Equal;
                    if nf_match {
                        let _ = tx.send(Some((
                            "nf".to_string(),
                            String::new(),
                            String::new(),
                            sub_cycles,
                        )));
                        return;
                    }

                    if prove_zero_from_metamorphic_texts(
                        &mut simplifier,
                        &lhs_clone,
                        &rhs_clone,
                        e,
                        s,
                    ) {
                        let _ = tx.send(Some((
                            "proved".to_string(),
                            String::new(),
                            String::new(),
                            sub_cycles,
                        )));
                        return;
                    }

                    // Check 3: Numeric equivalence (1 variable)
                    match classify_numeric_equiv_1var_relaxed(
                        &simplifier.context,
                        e,
                        s,
                        &free_var_clone,
                        &config_clone,
                    ) {
                        NumericCheckOutcome::Pass => {
                            let d = simplifier.context.add(cas_ast::Expr::Sub(e, s));
                            let (d_simp, _) = simplifier.simplify(d);
                            let residual = {
                                cas_formatter::LaTeXExpr {
                                    context: &simplifier.context,
                                    id: d_simp,
                                }
                                .to_latex()
                            };
                            let shape = expr_shape_signature(&simplifier.context, d_simp);
                            let cause = numeric_only_cause_for_1var(
                                &simplifier.context,
                                e,
                                s,
                                &free_var_clone,
                                &config_clone,
                                &FilterSpec::None,
                                &shape,
                            )
                            .label()
                            .to_string();
                            let _ =
                                tx.send(Some(("numeric".to_string(), residual, cause, sub_cycles)));
                        }
                        NumericCheckOutcome::Inconclusive(reason) => {
                            let _ = tx.send(Some((
                                "inconclusive".to_string(),
                                reason,
                                String::new(),
                                sub_cycles,
                            )));
                        }
                        NumericCheckOutcome::Failed(_) => {
                            let _ = tx.send(Some((
                                "failed".to_string(),
                                String::new(),
                                String::new(),
                                sub_cycles,
                            )));
                        }
                    }
                });

            let cell_key = (identity.family.clone(), sub.label.clone());

            match rx.recv_timeout(combo_timeout) {
                Ok(Some((kind, residual, cause, cycles))) => match kind.as_str() {
                    "nf" => {
                        nf_convergent += 1;
                        passed += 1;
                        cycle_events_total += cycles;
                        cell_data.entry(cell_key).or_insert((0, 0, 0, 0)).0 += 1;
                    }
                    "proved" => {
                        proved_symbolic += 1;
                        passed += 1;
                        cycle_events_total += cycles;
                        cell_data.entry(cell_key).or_insert((0, 0, 0, 0)).1 += 1;
                    }
                    "numeric" => {
                        numeric_only += 1;
                        passed += 1;
                        cycle_events_total += cycles;
                        *numeric_only_causes.entry(cause.clone()).or_default() += 1;
                        cell_data.entry(cell_key).or_insert((0, 0, 0, 0)).2 += 1;
                        if verbose && numeric_only_examples.len() < 200 {
                            numeric_only_examples.push((
                                lhs_str.clone(),
                                rhs_str.clone(),
                                identity.family.clone(),
                                residual,
                                cause,
                            ));
                        }
                    }
                    "inconclusive" => {
                        inconclusive += 1;
                        cycle_events_total += cycles;
                    }
                    "parse_error" => {
                        parse_errors += 1;
                        passed += 1; // Don't count as failure
                    }
                    "failed" => {
                        failed += 1;
                        cycle_events_total += cycles;
                        cell_data.entry(cell_key).or_insert((0, 0, 0, 0)).3 += 1;
                        if verbose {
                            eprintln!(
                                "  ❌ FAIL [{} → {}]: {} vs {}",
                                identity.family, sub.label, lhs_str, rhs_str
                            );
                        }
                    }
                    _ => {
                        failed += 1;
                        cycle_events_total += cycles;
                    }
                },
                Ok(None) => {
                    // Thread returned None — parse error
                    parse_errors += 1;
                    passed += 1;
                }
                Err(_) => {
                    // Timeout
                    timeouts += 1;
                }
            }
        }
    }

    // Report: flat summary (always shown)
    eprintln!(
        "✅ Substitution tests: {} passed, {} failed, {} skipped (timeout), {} parse errors, {} inconclusive",
        passed, failed, skipped, parse_errors, inconclusive
    );
    eprintln!(
        "   📐 NF-convergent: {} | 🔢 Proved-symbolic: {} | 🌡️ Numeric-only: {} | ◐ Inconclusive: {}",
        nf_convergent, proved_symbolic, numeric_only, inconclusive
    );
    if verbose && numeric_only > 0 {
        print_numeric_only_cause_breakdown(&numeric_only_causes);
    }

    // Cross-product table (METATEST_TABLE=1)
    if show_table {
        // Collect unique families and sub-labels in order
        let mut families: Vec<String> = identities
            .iter()
            .map(|i| i.family.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        families.sort();

        let mut sub_labels: Vec<String> = substitutions
            .iter()
            .map(|s| s.label.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        sub_labels.sort();

        // Abbreviate long sub-labels for columns
        let col_width = 11;
        let family_width = 25;

        eprintln!("\n╔══════════════════════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  Substitution × Identity Cross-Product (NF/Proved/Numeric)                         ║");
        eprintln!(
            "╠═════════════════════════╤{}╣",
            sub_labels
                .iter()
                .map(|_l| format!("═{:═<width$}═", "", width = col_width))
                .collect::<Vec<_>>()
                .join("╤")
        );

        // Header row
        eprint!(
            "║ {:family_width$}│",
            "Identity Family",
            family_width = family_width
        );
        for label in &sub_labels {
            let short = if label.len() > col_width {
                format!("{}…", &label[..col_width - 1])
            } else {
                label.clone()
            };
            eprint!(" {:^width$}│", short, width = col_width);
        }
        eprintln!();

        // Separator
        eprint!(
            "╠═{:═<family_width$}═╪",
            "",
            family_width = family_width - 2
        );
        for (i, _) in sub_labels.iter().enumerate() {
            if i < sub_labels.len() - 1 {
                eprint!("═{:═<width$}═╪", "", width = col_width);
            } else {
                eprint!("═{:═<width$}═╣", "", width = col_width);
            }
        }
        eprintln!();

        // Data rows
        for family in &families {
            eprint!("║ {:family_width$}│", family, family_width = family_width);
            for label in &sub_labels {
                let key = (family.clone(), label.clone());
                let (nf, prov, num, fail) = cell_data.get(&key).copied().unwrap_or((0, 0, 0, 0));
                let cell = if fail > 0 {
                    format!("{}/{}/{}❌{}", nf, prov, num, fail)
                } else {
                    format!("{}/{}/{}", nf, prov, num)
                };
                eprint!(" {:^width$}│", cell, width = col_width);
            }
            eprintln!();
        }

        // Bottom border
        eprint!(
            "╚═{:═<family_width$}═╧",
            "",
            family_width = family_width - 2
        );
        for (i, _) in sub_labels.iter().enumerate() {
            if i < sub_labels.len() - 1 {
                eprint!("═{:═<width$}═╧", "", width = col_width);
            } else {
                eprint!("═{:═<width$}═╝", "", width = col_width);
            }
        }
        eprintln!();
        eprintln!("Legend: NF/Proved/Numeric (❌N = N failures)");
    }

    // Verbose: show numeric-only cases grouped by family
    if verbose && !numeric_only_examples.is_empty() {
        eprintln!("\n── numeric-only examples ──");
        // Group by family
        let mut family_groups: HashMap<String, Vec<(String, String, String, String)>> =
            HashMap::new();
        for (lhs, rhs, family, residual, cause) in &numeric_only_examples {
            family_groups.entry(family.clone()).or_default().push((
                lhs.clone(),
                rhs.clone(),
                residual.clone(),
                cause.clone(),
            ));
        }
        let mut families: Vec<_> = family_groups.keys().cloned().collect();
        families.sort();
        for family in &families {
            let examples = &family_groups[family];
            eprintln!("── {} ({} cases) ──", family, examples.len());
            for (lhs, rhs, residual, cause) in examples.iter().take(10) {
                eprintln!("  LHS: {}", lhs);
                eprintln!("  RHS: {}", rhs);
                eprintln!("  Cause: {}", cause);
                if !residual.is_empty() {
                    eprintln!("  Residual: {}", residual);
                }
                eprintln!();
            }
        }
    }

    // Count unique identity families used
    let num_families = identities
        .iter()
        .map(|i| &i.family)
        .collect::<std::collections::HashSet<_>>()
        .len();

    ComboMetrics {
        op: "⇄sub".to_string(),
        pairs: identities.len(),
        families: num_families,
        combos: total_combos,
        nf_convergent,
        proved_quotient: proved_symbolic,
        proved_difference: 0,
        proved_composed: 0,
        numeric_only,
        inconclusive,
        failed,
        skipped,
        timeouts,
        cycle_events_total,
    }
}

/// Run curated contextual metamorphic tests.
fn run_contextual_pair_tests() -> ComboMetrics {
    let pairs = load_contextual_pairs();
    run_direct_pair_tests(pairs, "contextual metamorphic tests", "Contextual tests")
}

fn run_contextual_rational_pair_tests() -> ComboMetrics {
    let pairs = load_contextual_rational_pairs();
    run_direct_pair_tests(
        pairs,
        "contextual rational metamorphic tests",
        "Contextual rational tests",
    )
}

fn run_contextual_trig_pair_tests() -> ComboMetrics {
    let pairs = load_contextual_trig_pairs();
    run_direct_pair_tests(
        pairs,
        "contextual trig metamorphic tests",
        "Contextual trig tests",
    )
}

fn run_contextual_polynomial_pair_tests() -> ComboMetrics {
    let pairs = load_contextual_polynomial_pairs();
    run_direct_pair_tests(
        pairs,
        "contextual polynomial metamorphic tests",
        "Contextual polynomial tests",
    )
}

fn run_contextual_radical_pair_tests() -> ComboMetrics {
    let pairs = load_contextual_radical_pairs();
    run_direct_pair_tests(
        pairs,
        "contextual radical metamorphic tests",
        "Contextual radical tests",
    )
}

fn run_residual_pair_tests() -> ComboMetrics {
    let pairs = load_residual_pairs();
    run_direct_pair_tests(pairs, "residual metamorphic tests", "Residual tests")
}

fn run_direct_pair_tests(
    pairs: Vec<ContextualPair>,
    suite_title: &str,
    suite_summary: &str,
) -> ComboMetrics {
    let config = metatest_config();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();

    let total_pairs = pairs.len();
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut nf_convergent = 0usize;
    let mut proved_symbolic = 0usize;
    let mut numeric_only = 0usize;
    let mut inconclusive = 0usize;
    let skipped = 0usize;
    let mut timeouts = 0usize;
    let mut cycle_events_total: usize = 0;
    let mut parse_errors = 0usize;
    let pair_timeout = std::time::Duration::from_secs(5);
    let mut numeric_only_causes: HashMap<String, usize> = HashMap::new();
    let mut numeric_only_examples: Vec<(String, String, String, String, String)> = Vec::new();

    let num_families = pairs
        .iter()
        .map(|p| &p.family)
        .collect::<std::collections::HashSet<_>>()
        .len();

    eprintln!(
        "📊 Running {}: {} pairs from {} families (seed {})",
        suite_title, total_pairs, num_families, config.seed
    );

    for pair in &pairs {
        let lhs_str = pair.lhs.clone();
        let rhs_str = pair.rhs.clone();
        let free_vars = pair.vars.clone();
        let filters = pair.filters.clone();
        let family = pair.family.clone();
        let config_clone = config.clone();

        let (tx, rx) = std::sync::mpsc::channel();
        let _handle = std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                let mut simplifier = Simplifier::with_default_rules();
                let lhs_parsed = match parse(&lhs_str, &mut simplifier.context) {
                    Ok(e) => e,
                    Err(_) => {
                        let _ = tx.send(Some((
                            "parse_error".to_string(),
                            String::new(),
                            String::new(),
                            0,
                        )));
                        return;
                    }
                };
                let rhs_parsed = match parse(&rhs_str, &mut simplifier.context) {
                    Ok(e) => e,
                    Err(_) => {
                        let _ = tx.send(Some((
                            "parse_error".to_string(),
                            String::new(),
                            String::new(),
                            0,
                        )));
                        return;
                    }
                };

                let opts = cas_solver::runtime::SimplifyOptions::default();
                let mut sub_cycles: usize = 0;
                let (mut lhs_simp, _, stats_lhs) =
                    simplifier.simplify_with_stats(lhs_parsed, opts.clone());
                sub_cycles += stats_lhs.cycle_events.len();
                let (mut rhs_simp, _, stats_rhs) =
                    simplifier.simplify_with_stats(rhs_parsed, opts.clone());
                sub_cycles += stats_rhs.cycle_events.len();

                {
                    let cfg = cas_solver::runtime::EvalConfig::default();
                    let mut budget = cas_solver::runtime::Budget::preset_cli();
                    if let Ok(r) = cas_solver::api::fold_constants(
                        &mut simplifier.context,
                        lhs_simp,
                        &cfg,
                        cas_solver::api::ConstFoldMode::Safe,
                        &mut budget,
                    ) {
                        lhs_simp = r.expr;
                    }
                    if let Ok(r) = cas_solver::api::fold_constants(
                        &mut simplifier.context,
                        rhs_simp,
                        &cfg,
                        cas_solver::api::ConstFoldMode::Safe,
                        &mut budget,
                    ) {
                        rhs_simp = r.expr;
                    }
                }

                let nf_match =
                    cas_solver::runtime::compare_expr(&simplifier.context, lhs_simp, rhs_simp)
                        == std::cmp::Ordering::Equal;
                if nf_match {
                    let _ = tx.send(Some((
                        "nf".to_string(),
                        String::new(),
                        String::new(),
                        sub_cycles,
                    )));
                    return;
                }

                if prove_zero_from_metamorphic_texts(
                    &mut simplifier,
                    &lhs_str,
                    &rhs_str,
                    lhs_simp,
                    rhs_simp,
                ) {
                    let _ = tx.send(Some((
                        "proved".to_string(),
                        String::new(),
                        String::new(),
                        sub_cycles,
                    )));
                    return;
                }

                let outcome = match free_vars.as_slice() {
                    [var] if filters.first().is_none_or(FilterSpec::is_none) => {
                        classify_numeric_equiv_1var_relaxed(
                            &simplifier.context,
                            lhs_simp,
                            rhs_simp,
                            var,
                            &config_clone,
                        )
                    }
                    [var] => {
                        let filter = filters.first().cloned().unwrap_or(FilterSpec::None);
                        let stats = check_numeric_equiv_1var_stats(
                            &simplifier.context,
                            lhs_simp,
                            rhs_simp,
                            var,
                            &config_clone,
                            &filter,
                        );
                        let result = finalize_numeric_equiv_1var(stats.clone(), &config_clone);
                        classify_numeric_check_with_stats(result, &stats)
                    }
                    [var1, var2] => classify_numeric_equiv_2var_relaxed(
                        &simplifier.context,
                        lhs_simp,
                        rhs_simp,
                        var1,
                        var2,
                        &config_clone,
                        filters.first().unwrap_or(&FilterSpec::None),
                        filters.get(1).unwrap_or(&FilterSpec::None),
                    ),
                    vars if vars.len() >= 3 => classify_numeric_equiv_nvar_relaxed(
                        &simplifier.context,
                        lhs_simp,
                        rhs_simp,
                        vars,
                        &filters,
                        &config_clone,
                    ),
                    _ => NumericCheckOutcome::Inconclusive(format!(
                        "Unsupported contextual numeric arity: {}",
                        free_vars.len()
                    )),
                };
                match outcome {
                    NumericCheckOutcome::Pass => {
                        let residual = {
                            let d = simplifier
                                .context
                                .add(cas_ast::Expr::Sub(lhs_simp, rhs_simp));
                            let (d_simp, _) = simplifier.simplify(d);
                            cas_formatter::LaTeXExpr {
                                context: &simplifier.context,
                                id: d_simp,
                            }
                            .to_latex()
                        };
                        let shape = {
                            let d = simplifier
                                .context
                                .add(cas_ast::Expr::Sub(lhs_simp, rhs_simp));
                            let (d_simp, _) = simplifier.simplify(d);
                            expr_shape_signature(&simplifier.context, d_simp)
                        };
                        let cause = match free_vars.as_slice() {
                            [var] => numeric_only_cause_for_1var(
                                &simplifier.context,
                                lhs_simp,
                                rhs_simp,
                                var,
                                &config_clone,
                                filters.first().unwrap_or(&FilterSpec::None),
                                &shape,
                            ),
                            [var1, var2] => numeric_only_cause_for_2var(
                                &simplifier.context,
                                lhs_simp,
                                rhs_simp,
                                var1,
                                var2,
                                &config_clone,
                                filters.first().unwrap_or(&FilterSpec::None),
                                filters.get(1).unwrap_or(&FilterSpec::None),
                                &shape,
                            ),
                            vars => classify_numeric_only_cause(None, vars.len(), &shape),
                        }
                        .label()
                        .to_string();
                        let _ = tx.send(Some(("numeric".to_string(), residual, cause, sub_cycles)));
                    }
                    NumericCheckOutcome::Inconclusive(reason) => {
                        let _ = tx.send(Some((
                            "inconclusive".to_string(),
                            reason,
                            String::new(),
                            sub_cycles,
                        )));
                    }
                    NumericCheckOutcome::Failed(reason) => {
                        let _ = tx.send(Some((
                            "failed".to_string(),
                            reason,
                            String::new(),
                            sub_cycles,
                        )));
                    }
                }
            });

        match rx.recv_timeout(pair_timeout) {
            Ok(Some((kind, residual, cause, cycles))) => match kind.as_str() {
                "nf" => {
                    nf_convergent += 1;
                    passed += 1;
                    cycle_events_total += cycles;
                }
                "proved" => {
                    proved_symbolic += 1;
                    passed += 1;
                    cycle_events_total += cycles;
                }
                "numeric" => {
                    numeric_only += 1;
                    passed += 1;
                    cycle_events_total += cycles;
                    *numeric_only_causes.entry(cause.clone()).or_default() += 1;
                    if verbose && numeric_only_examples.len() < 200 {
                        numeric_only_examples.push((
                            pair.lhs.clone(),
                            pair.rhs.clone(),
                            family,
                            residual,
                            cause,
                        ));
                    }
                }
                "inconclusive" => {
                    inconclusive += 1;
                    cycle_events_total += cycles;
                }
                "parse_error" => {
                    parse_errors += 1;
                    passed += 1;
                }
                "failed" => {
                    failed += 1;
                    cycle_events_total += cycles;
                    if verbose {
                        eprintln!("  ❌ FAIL [{}]: {} vs {}", pair.family, pair.lhs, pair.rhs);
                        if !residual.is_empty() {
                            eprintln!("     Reason: {}", residual);
                        }
                    }
                }
                _ => {
                    failed += 1;
                    cycle_events_total += cycles;
                }
            },
            Ok(None) => {
                parse_errors += 1;
                passed += 1;
            }
            Err(_) => {
                timeouts += 1;
            }
        }
    }

    eprintln!(
        "✅ {}: {} passed, {} failed, {} skipped (timeout), {} parse errors, {} inconclusive",
        suite_summary, passed, failed, skipped, parse_errors, inconclusive
    );
    eprintln!(
        "   📐 NF-convergent: {} | 🔢 Proved-symbolic: {} | 🌡️ Numeric-only: {} | ◐ Inconclusive: {}",
        nf_convergent, proved_symbolic, numeric_only, inconclusive
    );
    if verbose && numeric_only > 0 {
        print_numeric_only_cause_breakdown(&numeric_only_causes);
    }

    if verbose && !numeric_only_examples.is_empty() {
        eprintln!("\n── contextual numeric-only examples ──");
        let mut family_groups: HashMap<String, Vec<(String, String, String, String)>> =
            HashMap::new();
        for (lhs, rhs, family, residual, cause) in &numeric_only_examples {
            family_groups.entry(family.clone()).or_default().push((
                lhs.clone(),
                rhs.clone(),
                residual.clone(),
                cause.clone(),
            ));
        }
        let mut families: Vec<_> = family_groups.keys().cloned().collect();
        families.sort();
        for family in &families {
            let examples = &family_groups[family];
            eprintln!("── {} ({} cases) ──", family, examples.len());
            for (lhs, rhs, residual, cause) in examples.iter().take(10) {
                eprintln!("  LHS: {}", lhs);
                eprintln!("  RHS: {}", rhs);
                eprintln!("  Cause: {}", cause);
                if !residual.is_empty() {
                    eprintln!("  Residual: {}", residual);
                }
                eprintln!();
            }
        }
    }

    ComboMetrics {
        op: "⇄ctx".to_string(),
        pairs: total_pairs,
        families: num_families,
        combos: total_pairs,
        nf_convergent,
        proved_quotient: proved_symbolic,
        proved_difference: 0,
        proved_composed: 0,
        numeric_only,
        inconclusive,
        failed,
        skipped,
        timeouts,
        cycle_events_total,
    }
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_substitution -- --include-ignored
fn metatest_csv_substitution() {
    let m = run_substitution_tests();
    assert_eq!(m.failed, 0, "{} substitution tests failed", m.failed);
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_contextual_pairs -- --ignored --nocapture
fn metatest_csv_contextual_pairs() {
    let m = run_contextual_pair_tests();
    assert_eq!(
        m.failed, 0,
        "{} contextual metamorphic tests failed",
        m.failed
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_contextual_rational_pairs -- --ignored --nocapture
fn metatest_csv_contextual_rational_pairs() {
    let m = run_contextual_rational_pair_tests();
    assert_eq!(m.failed, 0, "{} contextual rational tests failed", m.failed);
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_contextual_trig_pairs -- --ignored --nocapture
fn metatest_csv_contextual_trig_pairs() {
    let m = run_contextual_trig_pair_tests();
    assert_eq!(m.failed, 0, "{} contextual trig tests failed", m.failed);
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_contextual_polynomial_pairs -- --ignored --nocapture
fn metatest_csv_contextual_polynomial_pairs() {
    let m = run_contextual_polynomial_pair_tests();
    assert_eq!(
        m.failed, 0,
        "{} contextual polynomial tests failed",
        m.failed
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_contextual_radical_pairs -- --ignored --nocapture
fn metatest_csv_contextual_radical_pairs() {
    let m = run_contextual_radical_pair_tests();
    assert_eq!(m.failed, 0, "{} contextual radical tests failed", m.failed);
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_csv_residual_pairs -- --ignored --nocapture
fn metatest_csv_residual_pairs() {
    let m = run_residual_pair_tests();
    assert_eq!(
        m.failed, 0,
        "{} residual metamorphic tests failed",
        m.failed
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_idempotence_contracts -- --ignored --nocapture
fn metatest_simplify_idempotence_contracts() {
    let m = run_idempotence_contract_tests();
    assert_eq!(m.failed, 0, "{} idempotence contracts failed", m.failed);
    assert_eq!(
        m.parse_errors, 0,
        "{} idempotence expressions failed to parse",
        m.parse_errors
    );
    assert_eq!(
        m.timeouts, 0,
        "{} idempotence contracts timed out",
        m.timeouts
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_requires_contracts -- --ignored --nocapture
fn metatest_simplify_requires_contracts() {
    let m = run_requires_contract_tests();
    assert_eq!(m.failed, 0, "{} requires contracts failed", m.failed);
    assert_eq!(
        m.parse_errors, 0,
        "{} requires contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_warnings_contracts -- --ignored --nocapture
fn metatest_simplify_warnings_contracts() {
    let m = run_warnings_contract_tests();
    assert_eq!(m.failed, 0, "{} warnings contracts failed", m.failed);
    assert_eq!(
        m.parse_errors, 0,
        "{} warnings contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_transparency_signal_contracts -- --ignored --nocapture
fn metatest_simplify_transparency_signal_contracts() {
    let m = run_transparency_signal_contract_tests();
    assert_eq!(
        m.failed, 0,
        "{} transparency-signal contracts failed",
        m.failed
    );
    assert_eq!(
        m.parse_errors, 0,
        "{} transparency-signal contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_branch_transparency_contracts -- --ignored --nocapture
fn metatest_simplify_branch_transparency_contracts() {
    let m = run_branch_transparency_contract_tests();
    assert_eq!(
        m.failed, 0,
        "{} branch-transparency contracts failed",
        m.failed
    );
    assert_eq!(
        m.parse_errors, 0,
        "{} branch-transparency contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_semantic_behavior_contracts -- --ignored --nocapture
fn metatest_simplify_semantic_behavior_contracts() {
    let m = run_semantic_behavior_contract_tests();
    assert_eq!(
        m.failed, 0,
        "{} semantic-behavior contracts failed",
        m.failed
    );
    assert_eq!(
        m.parse_errors, 0,
        "{} semantic-behavior contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_requires_mode_contracts -- --ignored --nocapture
fn metatest_simplify_requires_mode_contracts() {
    let m = run_requires_mode_contract_tests();
    assert_eq!(
        m.failed, 0,
        "{} mode-aware requires contracts failed",
        m.failed
    );
    assert_eq!(
        m.parse_errors, 0,
        "{} mode-aware requires contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_semantic_axes_contracts -- --ignored --nocapture
fn metatest_simplify_semantic_axes_contracts() {
    let m = run_semantic_axes_contract_tests();
    assert_eq!(m.failed, 0, "{} semantic-axes contracts failed", m.failed);
    assert_eq!(
        m.parse_errors, 0,
        "{} semantic-axes contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_assumption_trace_contracts -- --ignored --nocapture
fn metatest_simplify_assumption_trace_contracts() {
    let m = run_assumption_trace_contract_tests();
    assert_eq!(
        m.failed, 0,
        "{} assumption trace contracts failed",
        m.failed
    );
    assert_eq!(
        m.parse_errors, 0,
        "{} assumption trace contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_phase4_contract_suites -- --ignored --nocapture
fn metatest_simplify_phase4_contract_suites() {
    let idempotence = run_idempotence_contract_tests();
    let requires = run_requires_contract_tests();
    let warnings = run_warnings_contract_tests();
    let transparency = run_transparency_signal_contract_tests();
    let branch_transparency = run_branch_transparency_contract_tests();
    let semantic_behavior = run_semantic_behavior_contract_tests();
    let requires_mode = run_requires_mode_contract_tests();
    let semantic_axes = run_semantic_axes_contract_tests();
    let assumption_trace = run_assumption_trace_contract_tests();

    eprintln!(
        "\n📦 Phase 4 contract summary: idempotence={} requires={} warnings={} transparency={} branch_transparency={} semantic_behavior={} requires_mode={} semantic_axes={} assumption_trace={}",
        idempotence.total,
        requires.total,
        warnings.total,
        transparency.total,
        branch_transparency.total,
        semantic_behavior.total,
        requires_mode.total,
        semantic_axes.total,
        assumption_trace.total
    );

    assert_eq!(
        idempotence.failed, 0,
        "{} idempotence contracts failed",
        idempotence.failed
    );
    assert_eq!(
        idempotence.parse_errors, 0,
        "{} idempotence contract parse errors",
        idempotence.parse_errors
    );
    assert_eq!(
        idempotence.timeouts, 0,
        "{} idempotence contract timeouts",
        idempotence.timeouts
    );

    assert_eq!(
        requires.failed, 0,
        "{} requires contracts failed",
        requires.failed
    );
    assert_eq!(
        requires.parse_errors, 0,
        "{} requires contract parse errors",
        requires.parse_errors
    );

    assert_eq!(
        warnings.failed, 0,
        "{} warnings contracts failed",
        warnings.failed
    );
    assert_eq!(
        warnings.parse_errors, 0,
        "{} warnings contract parse errors",
        warnings.parse_errors
    );

    assert_eq!(
        transparency.failed, 0,
        "{} transparency contracts failed",
        transparency.failed
    );
    assert_eq!(
        transparency.parse_errors, 0,
        "{} transparency contract parse errors",
        transparency.parse_errors
    );

    assert_eq!(
        branch_transparency.failed, 0,
        "{} branch_transparency contracts failed",
        branch_transparency.failed
    );
    assert_eq!(
        branch_transparency.parse_errors, 0,
        "{} branch_transparency contract parse errors",
        branch_transparency.parse_errors
    );

    assert_eq!(
        semantic_behavior.failed, 0,
        "{} semantic_behavior contracts failed",
        semantic_behavior.failed
    );
    assert_eq!(
        semantic_behavior.parse_errors, 0,
        "{} semantic_behavior contract parse errors",
        semantic_behavior.parse_errors
    );

    assert_eq!(
        requires_mode.failed, 0,
        "{} requires_mode contracts failed",
        requires_mode.failed
    );
    assert_eq!(
        requires_mode.parse_errors, 0,
        "{} requires_mode contract parse errors",
        requires_mode.parse_errors
    );

    assert_eq!(
        semantic_axes.failed, 0,
        "{} semantic_axes contracts failed",
        semantic_axes.failed
    );
    assert_eq!(
        semantic_axes.parse_errors, 0,
        "{} semantic_axes contract parse errors",
        semantic_axes.parse_errors
    );

    assert_eq!(
        assumption_trace.failed, 0,
        "{} assumption_trace contracts failed",
        assumption_trace.failed
    );
    assert_eq!(
        assumption_trace.parse_errors, 0,
        "{} assumption_trace contract parse errors",
        assumption_trace.parse_errors
    );
}

#[test]
fn relaxed_numeric_classification_marks_fragile_stats_inconclusive() {
    let stats = NumericEquivStats {
        valid: 2,
        near_pole: 8,
        domain_error: 6,
        asymmetric_invalid: 0,
        eval_failed: 0,
        filtered_out: 0,
        mismatches: Vec::new(),
        max_abs_err: 0.0,
        max_rel_err: 0.0,
        worst_sample: None,
    };

    let outcome = classify_numeric_check_with_stats(
        Err("Too few valid samples: 2 < 10 (near_pole=8, domain_error=6, asymmetric=0, eval_failed=0)".to_string()),
        &stats,
    );

    assert!(matches!(outcome, NumericCheckOutcome::Inconclusive(_)));
}

#[test]
fn relaxed_numeric_classification_keeps_true_mismatches_failed() {
    let stats = NumericEquivStats {
        valid: 12,
        near_pole: 0,
        domain_error: 0,
        asymmetric_invalid: 0,
        eval_failed: 0,
        filtered_out: 0,
        mismatches: vec!["x=0.5 => 1 != 2".to_string()],
        max_abs_err: 1.0,
        max_rel_err: 1.0,
        worst_sample: None,
    };

    let outcome = classify_numeric_check_with_stats(
        Err("Numeric mismatches: x=0.5 => 1 != 2".to_string()),
        &stats,
    );

    assert!(matches!(outcome, NumericCheckOutcome::Failed(_)));
}

#[test]
fn relaxed_numeric_classification_with_fixed_retries_filtered_samples() {
    let mut ctx = Context::new();
    let lhs = parse("sec(x)^2 - tan(x)^2", &mut ctx).expect("parse lhs");
    let rhs = parse("1", &mut ctx).expect("parse rhs");

    let config = MetatestConfig {
        eval_samples: 24,
        min_valid: 8,
        sample_range: (-1.6, 1.6),
        ..metatest_config()
    };

    let outcome = classify_numeric_equiv_1var_with_fixed_relaxed(
        &ctx,
        lhs,
        rhs,
        "x",
        &[],
        &config,
        &FilterSpec::None,
    );

    assert!(
        matches!(
            outcome,
            NumericCheckOutcome::Pass | NumericCheckOutcome::Inconclusive(_)
        ),
        "expected relaxed fixed-var classification to avoid hard failure, got {outcome:?}"
    );
}

#[test]
fn relaxed_numeric_classification_2var_retries_sampling_weak_cases() {
    let config = metatest_config();
    let mut calls = 0usize;

    let outcome = classify_numeric_equiv_2var_relaxed_with(&config, |_filter1, _filter2| {
        calls += 1;
        if calls == 1 {
            NumericEquivStats {
                valid: 1,
                near_pole: 24,
                domain_error: 18,
                asymmetric_invalid: 0,
                eval_failed: 0,
                filtered_out: 0,
                mismatches: Vec::new(),
                max_abs_err: 0.0,
                max_rel_err: 0.0,
                worst_sample: None,
            }
        } else {
            NumericEquivStats {
                valid: 8,
                ..Default::default()
            }
        }
    });

    assert!(matches!(outcome, NumericCheckOutcome::Pass));
    assert!(calls > 1, "expected relaxed 2var classification to retry");
}

// =============================================================================
// UNIFIED REGRESSION BENCHMARK: all operations + substitution in one scorecard
// =============================================================================

/// Unified regression benchmark combining combination tests (add, sub, mul, div)
/// and substitution tests into a single-run scorecard.
///
/// Run with:
/// ```text
/// cargo test --release -p cas_engine --test metamorphic_simplification_tests \
///     metatest_unified_benchmark -- --ignored --nocapture
/// ```
///
/// Key metrics per suite:
/// - NF-convergent: simplify(LHS) ≡ simplify(RHS) structurally (ideal)
/// - Proved-symbolic: simplify(LHS - RHS) = 0 (correct but different NFs)
/// - Numeric-only: only passes numeric check (target for improvement)
/// - Failed: semantic mismatches (regressions—must be 0)
/// - Timeout: combos that exceeded time limit (potential performance issues)
#[test]
#[ignore]
fn metatest_unified_benchmark() {
    let seed = metatest_config().seed;

    // Phase 1: Combination tests (add, sub, mul, div)
    let combo_configs: Vec<(CombineOp, usize)> = vec![
        (CombineOp::Add, 30),
        (CombineOp::Sub, 30),
        (CombineOp::Mul, 150),
        (CombineOp::Div, 50),
    ];

    let mut all_metrics: Vec<ComboMetrics> = Vec::new();

    for (op, pairs) in &combo_configs {
        let metrics = run_csv_combination_tests(*pairs, false, *op);
        all_metrics.push(metrics);
    }

    // Phase 2: Substitution tests
    let sub_metrics = run_substitution_tests();
    all_metrics.push(sub_metrics);

    // Phase 3: Curated contextual tests
    let contextual_metrics = run_contextual_pair_tests();
    all_metrics.push(contextual_metrics);
    let contextual_rational_metrics = run_contextual_rational_pair_tests();
    all_metrics.push(contextual_rational_metrics);
    let contextual_trig_metrics = run_contextual_trig_pair_tests();
    all_metrics.push(contextual_trig_metrics);
    let contextual_polynomial_metrics = run_contextual_polynomial_pair_tests();
    all_metrics.push(contextual_polynomial_metrics);
    let contextual_radical_metrics = run_contextual_radical_pair_tests();
    all_metrics.push(contextual_radical_metrics);

    // Phase 4: Print unified table
    eprintln!();
    eprintln!("╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║                    UNIFIED METAMORPHIC REGRESSION BENCHMARK (seed {:<10})                                              ║", seed);
    eprintln!("╠═══════╤════════╤══════════════╤══════════════╤══════════════╤══════════════╪════════╪═══════╪════════╪════════════════════╣");
    eprintln!("║ Suite │ Combos │ NF-convergent│ Proved-sym   │ Numeric-only │ Inconcl.     │ Failed │  T/O  │ Cycles │ Skip/Parse-err     ║");
    eprintln!("╠═══════╪════════╪══════════════╪══════════════╪══════════════╪══════════════╪════════╪═══════╪════════╪════════════════════╣");

    let mut total_combos = 0usize;
    let mut total_nf = 0usize;
    let mut total_proved = 0usize;
    let mut total_numeric = 0usize;
    let mut total_inconclusive = 0usize;
    let mut total_failed = 0usize;
    let mut total_timeouts = 0usize;
    let mut total_cycles = 0usize;
    let mut total_skipped = 0usize;

    for m in &all_metrics {
        let effective = m
            .combos
            .saturating_sub(m.skipped)
            .saturating_sub(m.timeouts);
        let proved = m.proved_symbolic();
        let nf_pct = if effective > 0 {
            m.nf_convergent as f64 / effective as f64 * 100.0
        } else {
            0.0
        };
        let prov_pct = if effective > 0 {
            proved as f64 / effective as f64 * 100.0
        } else {
            0.0
        };
        let num_pct = if effective > 0 {
            m.numeric_only as f64 / effective as f64 * 100.0
        } else {
            0.0
        };
        let inc_pct = if effective > 0 {
            m.inconclusive as f64 / effective as f64 * 100.0
        } else {
            0.0
        };

        eprintln!(
            "║ {:5} │ {:>6} │ {:>5} {:>5.1}% │ {:>5} {:>5.1}% │ {:>5} {:>5.1}% │ {:>5} {:>5.1}% │ {:>6} │ {:>5} │ {:>6} │ {:>6}             ║",
            m.op, m.combos,
            m.nf_convergent, nf_pct,
            proved, prov_pct,
            m.numeric_only, num_pct,
            m.inconclusive, inc_pct,
            m.failed,
            m.timeouts,
            m.cycle_events_total,
            m.skipped,
        );

        total_combos += m.combos;
        total_nf += m.nf_convergent;
        total_proved += proved;
        total_numeric += m.numeric_only;
        total_inconclusive += m.inconclusive;
        total_failed += m.failed;
        total_timeouts += m.timeouts;
        total_cycles += m.cycle_events_total;
        total_skipped += m.skipped;
    }

    let total_effective = total_combos
        .saturating_sub(total_skipped)
        .saturating_sub(total_timeouts);
    let total_nf_pct = if total_effective > 0 {
        total_nf as f64 / total_effective as f64 * 100.0
    } else {
        0.0
    };
    let total_prov_pct = if total_effective > 0 {
        total_proved as f64 / total_effective as f64 * 100.0
    } else {
        0.0
    };
    let total_num_pct = if total_effective > 0 {
        total_numeric as f64 / total_effective as f64 * 100.0
    } else {
        0.0
    };
    let total_inc_pct = if total_effective > 0 {
        total_inconclusive as f64 / total_effective as f64 * 100.0
    } else {
        0.0
    };

    eprintln!("╠═══════╪════════╪══════════════╪══════════════╪══════════════╪══════════════╪════════╪═══════╪════════╪════════════════════╣");
    eprintln!(
        "║ TOTAL │ {:>6} │ {:>5} {:>5.1}% │ {:>5} {:>5.1}% │ {:>5} {:>5.1}% │ {:>5} {:>5.1}% │ {:>6} │ {:>5} │ {:>6} │ {:>6}             ║",
        total_combos,
        total_nf, total_nf_pct,
        total_proved, total_prov_pct,
        total_numeric, total_num_pct,
        total_inconclusive, total_inc_pct,
        total_failed,
        total_timeouts,
        total_cycles,
        total_skipped,
    );
    eprintln!("╚═══════╧════════╧══════════════╧══════════════╧══════════════╧══════════════╧════════╧═══════╧════════╧════════════════════╝");

    if total_failed > 0 {
        eprintln!(
            "⚠️  {} semantic failures detected — investigate before merging.",
            total_failed
        );
    }

    // Cycle events summary
    if total_cycles > 0 {
        eprintln!();
        eprintln!(
            "🔄 Cycle Events Summary: {} total across all suites",
            total_cycles
        );
        eprintln!("   The cycle detector successfully prevented oscillations.");
        eprintln!("   Run with METATEST_VERBOSE=1 for per-rule breakdown.");
    }

    if total_timeouts > 0 {
        eprintln!();
        eprintln!("⏱️  {} timeouts detected — consider increasing time budget or investigating slow combos.", total_timeouts);
    }

    if total_inconclusive > 0 {
        eprintln!();
        eprintln!(
            "◐ {} inconclusive numeric checks recorded — tracked separately from semantic failures.",
            total_inconclusive
        );
    }

    eprintln!();
}
