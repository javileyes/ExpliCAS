//! JSON output types for the non-interactive CLI API.
//!
//! These structures are used by eval-json, script-json, and mm-gcd-modp-json
//! subcommands to provide structured output suitable for scripting and notebooks.

use serde::Serialize;

/// Result of evaluating a single expression via eval-json
#[derive(Serialize, Debug)]
pub struct EvalJsonOutput {
    pub ok: bool,
    pub input: String,

    /// Pretty-printed result (truncated if too large)
    pub result: String,
    pub result_truncated: bool,
    pub result_chars: usize,

    /// Steps mode used and count
    pub steps_mode: String,
    pub steps_count: usize,

    /// Domain warnings from simplification
    pub warnings: Vec<WarningJson>,

    /// Expression statistics
    pub stats: ExprStatsJson,

    /// Hash for identity checking without printing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,

    /// Timing breakdown in microseconds
    pub timings_us: TimingsJson,

    /// Options that were used
    pub options: OptionsJson,
}

/// An error result
#[derive(Serialize, Debug)]
pub struct ErrorJsonOutput {
    pub ok: bool,
    pub error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
}

impl ErrorJsonOutput {
    pub fn new(error: impl Into<String>) -> Self {
        Self {
            ok: false,
            error: error.into(),
            input: None,
        }
    }

    pub fn with_input(error: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            ok: false,
            error: error.into(),
            input: Some(input.into()),
        }
    }
}

/// A domain assumption warning with its source rule
#[derive(Serialize, Debug, Clone)]
pub struct WarningJson {
    pub rule: String,
    pub assumption: String,
}

/// Expression statistics (node count, depth)
#[derive(Serialize, Debug, Default)]
pub struct ExprStatsJson {
    pub node_count: usize,
    pub depth: usize,
}

/// Timing breakdown in microseconds
#[derive(Serialize, Debug, Default)]
pub struct TimingsJson {
    pub parse_us: u64,
    pub simplify_us: u64,
    pub total_us: u64,
}

/// Options used for evaluation
#[derive(Serialize, Debug, Default)]
pub struct OptionsJson {
    pub context_mode: String,
    pub branch_mode: String,
    pub expand_policy: String,
    pub complex_mode: String,
    pub steps_mode: String,
}

// ============================================================================
// script-json types
// ============================================================================

/// Result of processing a script via script-json
#[derive(Serialize, Debug)]
pub struct ScriptJsonOutput {
    pub ok: bool,
    pub lines: Vec<ScriptLineResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_result: Option<EvalJsonOutput>,
    pub total_time_us: u64,
}

/// Result of processing a single line in a script
#[derive(Serialize, Debug)]
pub struct ScriptLineResult {
    pub line_no: usize,
    pub input: String,
    /// "command" | "let" | "expr" | "empty" | "error"
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<EvalJsonOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ============================================================================
// mm-gcd-modp-json types
// ============================================================================

/// Result of running mm_gcd benchmark via mm-gcd-modp-json
#[derive(Serialize, Debug)]
pub struct MmGcdModpJsonOutput {
    pub ok: bool,
    pub modulus: u64,

    /// Term counts for the polynomials
    pub a_terms: usize,
    pub b_terms: usize,
    pub g_terms: usize,
    pub ag_terms: usize,
    pub bg_terms: usize,

    /// GCD result info
    pub gcd_terms: usize,
    pub gcd_total_degree: u32,
    pub gcd_matches_g: bool,

    /// Timings in milliseconds
    pub timings_ms: MmGcdTimingsMs,
}

/// Timing breakdown for mm_gcd in milliseconds
#[derive(Serialize, Debug, Default)]
pub struct MmGcdTimingsMs {
    pub build_ms: f64,
    pub mul_ms: f64,
    pub gcd_ms: f64,
    pub full_ms: f64,
}
