//! JSON output types for the non-interactive CLI API.
//!
//! These structures are used by eval-json, script-json, and mm-gcd-modp-json
//! subcommands to provide structured output suitable for scripting and notebooks.
//!
//! Lightweight types shared with the engine are re-exported from `cas_engine::json`.

use serde::Serialize;

// Re-export canonical types from engine (single source of truth)
pub use cas_engine::json::{
    DomainJson, ErrorJsonOutput, ExprStatsJson, OptionsJson, RequiredConditionJson, SemanticsJson,
    SolveStepJson, SolveSubStepJson, StepJson, SubStepJson, TimingsJson, WarningJson,
};

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

    /// LaTeX formatted result for rendering
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_latex: Option<String>,

    /// Steps mode used and count
    pub steps_mode: String,
    pub steps_count: usize,

    /// Detailed steps when steps_mode is "on"
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<StepJson>,

    /// Equation solving steps when context_mode is "solve" and steps_mode is "on"
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub solve_steps: Vec<SolveStepJson>,

    /// Domain warnings from simplification
    pub warnings: Vec<WarningJson>,

    /// Required conditions (implicit domain constraints from input expression)
    /// These are NOT assumptions - they were already implied by the input.
    pub required_conditions: Vec<RequiredConditionJson>,

    /// Human-readable required conditions for simple frontends
    pub required_display: Vec<String>,

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

    /// Unified wire output (stable messaging format)
    /// Contains all messages in a structured, versioned format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wire: Option<crate::repl::wire::WireReply>,
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

// SemanticsJson is now re-exported from cas_engine::json

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

// =============================================================================
// OutputEnvelope V1 — re-exported from engine (canonical source of truth)
// =============================================================================

pub use cas_engine::json::{
    AssumptionDto, BlockedHintDto, BoundDto, CaseDto, ConditionDto, EngineInfo, ExprDto,
    OutputEnvelope, RequestInfo, RequestOptions, ResultDto, SolutionSetDto, StepDto, ThenDto,
    TransparencyDto, WhenDto,
};
