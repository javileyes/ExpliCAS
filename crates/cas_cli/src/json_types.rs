//! JSON output types for the non-interactive CLI API.
//!
//! These structures are used by eval-json, script-json, and mm-gcd-modp-json
//! subcommands to provide structured output suitable for scripting and notebooks.

use serde::Serialize;

/// Result of evaluating a single expression via eval-json
#[derive(Serialize, Debug)]
pub struct EvalJsonOutput {
    /// Schema version for API stability (increment on breaking changes)
    pub schema_version: u8,

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

    /// Budget information
    pub budget: BudgetJson,

    /// Domain mode information
    pub domain: DomainJson,

    /// Expression statistics
    pub stats: ExprStatsJson,

    /// Hash for identity checking without printing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,

    /// Timing breakdown in microseconds
    pub timings_us: TimingsJson,

    /// Options that were used
    pub options: OptionsJson,

    /// Complete semantics configuration
    pub semantics: SemanticsJson,
}

/// Budget configuration and status
#[derive(Serialize, Debug, Default)]
pub struct BudgetJson {
    pub preset: String,
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exceeded: Option<BudgetExceededJson>,
}

/// Details when budget was exceeded
#[derive(Serialize, Debug)]
pub struct BudgetExceededJson {
    pub op: String,
    pub metric: String,
    pub used: u64,
    pub limit: u64,
}

/// Domain mode information
#[derive(Serialize, Debug, Default)]
pub struct DomainJson {
    /// Current domain mode: "strict", "generic", or "assume"
    pub mode: String,
}

/// Complete semantics configuration in JSON output
#[derive(Serialize, Debug, Default)]
pub struct SemanticsJson {
    /// Domain assumption mode
    pub domain_mode: String,
    /// Value domain (real/complex)
    pub value_domain: String,
    /// Branch policy for multi-valued functions
    pub branch: String,
    /// Inverse trig composition policy
    pub inv_trig: String,
}

/// An error result with stable kind/code for API consumers.
///
/// The `kind` and `code` fields are stable and should not change between versions.
/// See POLICY.md "Error API Stability Contract".
#[derive(Serialize, Debug)]
pub struct ErrorJsonOutput {
    /// Schema version
    pub schema_version: u8,

    pub ok: bool,

    /// Stable error kind for routing (ParseError, DomainError, etc.)
    pub kind: String,

    /// Stable error code for UI mapping (E_PARSE, E_DIV_ZERO, etc.)
    pub code: String,

    /// Human-readable error message
    pub error: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
}

impl ErrorJsonOutput {
    pub fn new(error: impl Into<String>) -> Self {
        Self {
            schema_version: 1,
            ok: false,
            kind: "InternalError".into(),
            code: "E_INTERNAL".into(),
            error: error.into(),
            input: None,
        }
    }

    pub fn with_input(error: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            schema_version: 1,
            ok: false,
            kind: "InternalError".into(),
            code: "E_INTERNAL".into(),
            error: error.into(),
            input: Some(input.into()),
        }
    }

    /// Create from a CasError with stable kind/code.
    pub fn from_cas_error(e: &cas_engine::CasError, input: Option<String>) -> Self {
        Self {
            schema_version: 1,
            ok: false,
            kind: e.kind().to_string(),
            code: e.code().to_string(),
            error: e.to_string(),
            input,
        }
    }

    /// Create a parse error.
    pub fn parse_error(message: impl Into<String>, input: Option<String>) -> Self {
        Self {
            schema_version: 1,
            ok: false,
            kind: "ParseError".into(),
            code: "E_PARSE".into(),
            error: message.into(),
            input,
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
    pub domain_mode: String,
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
