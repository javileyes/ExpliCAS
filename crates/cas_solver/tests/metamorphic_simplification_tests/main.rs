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

#[path = "../test_utils/mod.rs"]
mod test_utils;

use cas_ast::views::as_rational_const;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_session::SessionState;
use cas_solver::api::{
    eval_f64, eval_f64_checked, EquivalenceResult, EvalCheckedError, EvalCheckedOptions,
};
use cas_solver::runtime::{Engine, EvalAction, EvalRequest, EvalResult, Simplifier, StepsMode};
use cas_solver::wire::eval_str_to_wire;
use num_traits::Signed;
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

// Configuration
// =============================================================================

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

// =============================================================================
// Reporting Helpers
// =============================================================================

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

// =============================================================================
// Shuffle Canonicalization (Phase A)
// =============================================================================

// =============================================================================
// Numeric Equivalence Check
// =============================================================================

const NUMERIC_DENOM_GUARD_ATOL: f64 = 1e-8;
const NUMERIC_INTERIOR_VALUES: [f64; 10] = [
    -0.9, -0.75, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 0.75, 0.9,
];
const NUMERIC_GENERAL_VALUES: [f64; 12] = [
    -4.0, -2.5, -1.5, -0.75, -0.25, 0.25, 0.75, 1.5, 2.5, 4.0, 0.1, 5.0,
];
const NUMERIC_POSITIVE_VALUES: [f64; 13] = [
    0.1, 0.2, 0.35, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0,
];
const NUMERIC_RATIONAL_VALUES: [f64; 12] = [
    -5.0, -3.5, -2.5, -1.5, -0.5, -0.2, 0.2, 0.5, 1.5, 2.5, 3.5, 5.0,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumericSampleProfile {
    Interior,
    General,
    Positive,
    Rational,
}

#[derive(Debug, Default, Clone, Copy)]
struct NumericSamplingFeatures {
    bounded_inverse_trig: bool,
    positivity_sensitive: bool,
    rational_sensitive: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumericAnalyticGuardKind {
    Positive,
    NonNegative,
    NotOne,
    UnitInterval,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NumericAnalyticGuard {
    expr: ExprId,
    kind: NumericAnalyticGuardKind,
}

#[derive(Debug, Clone)]
enum NumericCheckOutcome {
    Pass,
    Inconclusive(String),
    Failed(String),
}

struct CuratedPairCorpus {
    raw: HashSet<(String, String)>,
    alpha: HashSet<(String, String)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NfFirstAddSubChildOutcome {
    Nf,
    Proved,
    Inconclusive,
    Timeout,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum NfFirstMulDivChildOutcome {
    Nf {
        cycles: usize,
    },
    ProvedQuotient {
        cycles: usize,
    },
    ProvedDifference {
        cycles: usize,
    },
    Numeric {
        diff_str: String,
        shape: String,
        cause: String,
        cycles: usize,
    },
    DomainFrontier {
        reason: String,
        shape: String,
        cause: String,
        cycles: usize,
    },
    Inconclusive {
        reason: String,
        cycles: usize,
    },
    Failed {
        cycles: usize,
    },
    Skip,
    Timeout,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SubstitutionComboOutcome {
    kind: String,
    residual: String,
    cause: String,
    cycles: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetamorphicProofFlavor {
    Curated,
    RawPressure,
}

impl MetamorphicProofFlavor {
    fn child_label(self) -> &'static str {
        match self {
            Self::Curated => "curated",
            Self::RawPressure => "raw-pressure",
        }
    }

    fn from_child_label(label: &str) -> Option<Self> {
        match label {
            "curated" => Some(Self::Curated),
            "raw-pressure" => Some(Self::RawPressure),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetatestShortcutMode {
    SmokeClosure,
    StrictPressure,
    NfFirstPressure,
}

impl MetatestShortcutMode {
    fn allows_curated_shortcuts(self) -> bool {
        matches!(self, Self::SmokeClosure)
    }

    fn allows_pre_nf_proof_shortcuts(self) -> bool {
        matches!(self, Self::StrictPressure)
    }

    fn allows_composed_promotion(self) -> bool {
        matches!(self, Self::SmokeClosure | Self::StrictPressure)
    }

    fn requires_deep_combo_worker(self) -> bool {
        matches!(self, Self::StrictPressure | Self::NfFirstPressure)
    }

    fn benchmark_label(self) -> &'static str {
        match self {
            Self::SmokeClosure => "SMOKE",
            Self::StrictPressure => "STRICT",
            Self::NfFirstPressure => "NF-FIRST",
        }
    }

    fn child_label(self) -> &'static str {
        match self {
            Self::SmokeClosure => "smoke",
            Self::StrictPressure => "strict",
            Self::NfFirstPressure => "nf-first",
        }
    }

    fn from_child_label(label: &str) -> Option<Self> {
        match label {
            "smoke" => Some(Self::SmokeClosure),
            "strict" => Some(Self::StrictPressure),
            "nf-first" => Some(Self::NfFirstPressure),
            _ => None,
        }
    }
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

struct NfFirstMulDivSkipExample<'a> {
    op_name: &'a str,
    pair1: &'a IdentityPair,
    pair2: &'a IdentityPair,
    pair2_exp: &'a str,
    pair2_simp: &'a str,
    combined_exp: &'a str,
    combined_simp: &'a str,
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

// =============================================================================
// Standard Sample Filters
// =============================================================================

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

/// Check if a snapshot represents a regression compared to baseline
#[derive(Debug)]
struct RegressionResult {
    id: String,
    exp: String,
    reasons: Vec<String>,
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
    known_symbolic_residuals: usize,
    numeric_only_causes: HashMap<String, usize>,
    inconclusive_causes: HashMap<String, usize>,
    domain_frontier_examples: Vec<(String, String, String)>,
}

impl ComboMetrics {
    fn proved_symbolic(&self) -> usize {
        self.proved_quotient + self.proved_difference + self.proved_composed
    }

    fn known_domain_frontier_count(&self) -> usize {
        self.inconclusive_causes
            .iter()
            .filter(|(label, _)| label.starts_with("domain-frontier:"))
            .map(|(_, count)| *count)
            .sum()
    }

    fn numeric_only_cause_count(&self, label: &str) -> usize {
        self.numeric_only_causes.get(label).copied().unwrap_or(0)
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

const DEFAULT_METATEST_PROGRESS_EVERY: usize = 1000;
// Release metamorphic runners hit genuinely deep stacks on some dense
// trig+rational combinations; keep the worker stack comfortably above the
// default to avoid aborting the whole test process.
const METATEST_WORKER_STACK_SIZE_BYTES: usize = 512 * 1024 * 1024;
const METATEST_DEEP_WORKER_STACK_SIZE_BYTES: usize = 512 * 1024 * 1024;
const METATEST_CHILD_RAW_PROOF_LHS_ENV: &str = "METATEST_CHILD_RAW_PROOF_LHS";
const METATEST_CHILD_RAW_PROOF_RHS_ENV: &str = "METATEST_CHILD_RAW_PROOF_RHS";
const METATEST_CHILD_RAW_PROOF_TIMEOUT_MS: u64 = 5_000;
const METATEST_CHILD_NF_LHS_ENV: &str = "METATEST_CHILD_NF_LHS";
const METATEST_CHILD_NF_RHS_ENV: &str = "METATEST_CHILD_NF_RHS";
const METATEST_CHILD_NF_TIMEOUT_MS: u64 = 5_000;
const METATEST_CHILD_NF_ADD_SUB_EXP_ENV: &str = "METATEST_CHILD_NF_ADD_SUB_EXP";
const METATEST_CHILD_NF_ADD_SUB_SIMP_ENV: &str = "METATEST_CHILD_NF_ADD_SUB_SIMP";
const METATEST_CHILD_NF_ADD_SUB_OUTCOME_ENV: &str = "METATEST_CHILD_NF_ADD_SUB_OUTCOME";
const METATEST_CHILD_NF_ADD_SUB_TIMEOUT_MS: u64 = 5_000;
const METATEST_CHILD_NF_MUL_DIV_EXP_ENV: &str = "METATEST_CHILD_NF_MUL_DIV_EXP";
const METATEST_CHILD_NF_MUL_DIV_SIMP_ENV: &str = "METATEST_CHILD_NF_MUL_DIV_SIMP";
const METATEST_CHILD_NF_MUL_DIV_VARS_ENV: &str = "METATEST_CHILD_NF_MUL_DIV_VARS";
const METATEST_CHILD_NF_MUL_DIV_FILTERS_ENV: &str = "METATEST_CHILD_NF_MUL_DIV_FILTERS";
const METATEST_CHILD_NF_MUL_DIV_OUTCOME_ENV: &str = "METATEST_CHILD_NF_MUL_DIV_OUTCOME";
const METATEST_CHILD_SUBSTITUTION_LHS_ENV: &str = "METATEST_CHILD_SUBSTITUTION_LHS";
const METATEST_CHILD_SUBSTITUTION_RHS_ENV: &str = "METATEST_CHILD_SUBSTITUTION_RHS";
const METATEST_CHILD_SUBSTITUTION_VAR_ENV: &str = "METATEST_CHILD_SUBSTITUTION_VAR";
const METATEST_CHILD_SUBSTITUTION_FILTERS_ENV: &str = "METATEST_CHILD_SUBSTITUTION_FILTERS";
const METATEST_CHILD_SUBSTITUTION_PROOF_ENV: &str = "METATEST_CHILD_SUBSTITUTION_PROOF";
const METATEST_CHILD_SUBSTITUTION_MODE_ENV: &str = "METATEST_CHILD_SUBSTITUTION_MODE";
const METATEST_CHILD_SUBSTITUTION_OUTCOME_ENV: &str = "METATEST_CHILD_SUBSTITUTION_OUTCOME";

struct ComboProgressSnapshot {
    processed_combos: usize,
    total_combos: usize,
    nf_convergent: usize,
    proved_symbolic: usize,
    numeric_only: usize,
    inconclusive: usize,
    skipped: usize,
    timeouts: usize,
    failed: usize,
}

// =============================================================================
// CSV-BASED AUTOMATIC COMBINATION TESTS
// =============================================================================

// =============================================================================
// Shuffle Canonicalization Test (Phase A)
// =============================================================================

enum ShuffleResult {
    Ok,
    StructuralDiff(String),
    SemanticFail(String),
    ParseSkip, // Expression couldn't be parsed (syntax not supported)
}

// =============================================================================
// MetaTransform Test (Phase B)
// =============================================================================

enum TransformResult {
    Pass,
    Skip(String),
    Fail(String),
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
    expr: String,             // The expression to substitute, e.g. "sin(u)"
    var: String,              // The free variable after substitution, e.g. "u"
    label: String,            // Category label, e.g. "trig"
    filters: Vec<FilterSpec>, // Optional numeric-domain filters for the free vars
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
struct ComplexModeBehaviorContractExpr {
    expr: String,
    value_domain: cas_solver::runtime::ValueDomain,
    complex_mode: cas_solver::runtime::ComplexMode,
    expectation: SemanticBehaviorExpectation,
    family: String,
}

#[derive(Debug, Clone)]
struct ConstFoldBehaviorContractExpr {
    expr: String,
    value_domain: cas_solver::runtime::ValueDomain,
    const_fold_mode: cas_solver::api::ConstFoldMode,
    expectation: SemanticBehaviorExpectation,
    family: String,
}

#[derive(Debug, Clone)]
struct EvalPathBehaviorContractExpr {
    expr: String,
    value_domain: cas_solver::runtime::ValueDomain,
    mode: cas_solver::runtime::DomainMode,
    complex_mode: cas_solver::runtime::ComplexMode,
    const_fold_mode: cas_solver::api::ConstFoldMode,
    expectation: SemanticBehaviorExpectation,
    family: String,
}

#[derive(Debug, Clone)]
struct EvalPathAxesContractExpr {
    expr: String,
    value_domain: cas_solver::runtime::ValueDomain,
    mode: cas_solver::runtime::DomainMode,
    complex_mode: cas_solver::runtime::ComplexMode,
    const_fold_mode: cas_solver::api::ConstFoldMode,
    expect_requires: bool,
    expect_warning: bool,
    family: String,
}

#[derive(Debug, Clone)]
struct EvalPathInvTrigAxesContractExpr {
    expr: String,
    value_domain: cas_solver::runtime::ValueDomain,
    mode: cas_solver::runtime::DomainMode,
    inv_trig: cas_solver::runtime::InverseTrigPolicy,
    expect_requires: bool,
    expect_warning: bool,
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

type ComplexModeBehaviorContractMetrics = SemanticBehaviorContractMetrics;
type ConstFoldBehaviorContractMetrics = SemanticBehaviorContractMetrics;
type EvalPathBehaviorContractMetrics = SemanticBehaviorContractMetrics;
type EvalPathAxesContractMetrics = SemanticAxesContractMetrics;
type EvalPathInvTrigAxesContractMetrics = SemanticAxesContractMetrics;

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

// =============================================================================
// UNIFIED REGRESSION BENCHMARK: all operations + substitution in one scorecard
// =============================================================================

mod csv_pairs;
mod domain_conditions;
mod general;
mod numeric_check;
mod prove_zero;
mod runners;
mod substitution_runs;
mod support;
mod text_render;

use csv_pairs::*;
use domain_conditions::*;
use general::*;
use numeric_check::*;
use prove_zero::*;
use runners::*;
use substitution_runs::*;
use support::*;
use text_render::*;
