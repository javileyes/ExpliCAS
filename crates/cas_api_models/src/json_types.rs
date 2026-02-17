//! JSON output types for the non-interactive CLI API.
//!
//! These structures are used by eval-json, script-json, and mm-gcd-modp-json
//! subcommands to provide structured output suitable for scripting and notebooks.
//!
//! Lightweight types shared with the engine are re-exported from `cas_engine::json`.

use serde::Serialize;

// Re-export canonical types from engine (single source of truth)
pub use cas_engine::json::{
    DomainJson, ErrorJsonOutput, EvalJsonOutput, ExprStatsJson, OptionsJson, RequiredConditionJson,
    SemanticsJson, SolveStepJson, SolveSubStepJson, StepJson, SubStepJson, TimingsJson,
    WarningJson,
};

// Budget: use engine's BudgetJsonInfo (identical fields: preset, mode, exceeded)
pub use cas_engine::json::BudgetJsonInfo;
/// Backward-compatible alias — `BudgetJson` maps to the engine's `BudgetJsonInfo`.
pub type BudgetJson = cas_engine::json::BudgetJsonInfo;

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
