//! JSON DTOs shared by CLI/FFI layers.
//!
//! This crate intentionally keeps transport models independent from engine internals.

use serde::{Deserialize, Serialize};

/// Stable schema version for JSON outputs.
pub const SCHEMA_VERSION: u8 = 1;

// =============================================================================
// Shared Eval JSON types
// =============================================================================

/// Expression statistics (node count, depth).
#[derive(Serialize, Debug, Default)]
pub struct ExprStatsJson {
    pub node_count: usize,
    pub depth: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub term_count: Option<usize>,
}

/// Timing breakdown in microseconds.
#[derive(Serialize, Debug, Default)]
pub struct TimingsJson {
    pub parse_us: u64,
    pub simplify_us: u64,
    pub total_us: u64,
}

/// Domain mode information.
#[derive(Serialize, Debug, Default)]
pub struct DomainJson {
    pub mode: String,
}

/// Options used for evaluation.
#[derive(Serialize, Debug, Default)]
pub struct OptionsJson {
    pub context_mode: String,
    pub branch_mode: String,
    pub expand_policy: String,
    pub complex_mode: String,
    pub steps_mode: String,
    pub domain_mode: String,
    pub const_fold: String,
}

/// Complete semantics configuration in JSON output.
#[derive(Serialize, Debug, Default)]
pub struct SemanticsJson {
    pub domain_mode: String,
    pub value_domain: String,
    pub branch: String,
    pub inv_trig: String,
    pub assume_scope: String,
}

/// A required condition (implicit domain constraint) from the input expression.
#[derive(Serialize, Debug, Clone)]
pub struct RequiredConditionJson {
    pub kind: String,
    pub expr_display: String,
    pub expr_canonical: String,
}

/// A simplification step for JSON output.
#[derive(Serialize, Debug, Clone)]
pub struct StepJson {
    pub index: usize,
    pub rule: String,
    pub rule_latex: String,
    pub before: String,
    pub after: String,
    pub before_latex: String,
    pub after_latex: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub substeps: Vec<SubStepJson>,
}

/// A sub-step within a step for detailed explanations.
#[derive(Serialize, Debug, Clone)]
pub struct SubStepJson {
    pub title: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub lines: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before_latex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_latex: Option<String>,
}

/// A solver step for equation-solving JSON output.
#[derive(Serialize, Debug, Clone)]
pub struct SolveStepJson {
    pub index: usize,
    pub description: String,
    pub equation: String,
    pub lhs_latex: String,
    pub relop: String,
    pub rhs_latex: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub substeps: Vec<SolveSubStepJson>,
}

/// A sub-step within a solve step for detailed derivation.
#[derive(Serialize, Debug, Clone)]
pub struct SolveSubStepJson {
    pub index: String,
    pub description: String,
    pub equation: String,
    pub lhs_latex: String,
    pub relop: String,
    pub rhs_latex: String,
}

/// A domain assumption warning with its source rule.
#[derive(Serialize, Debug, Clone)]
pub struct WarningJson {
    pub rule: String,
    pub assumption: String,
}

/// Stable budget information for JSON responses.
#[derive(Serialize, Debug, Default, Clone)]
pub struct BudgetJsonInfo {
    pub preset: String,
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exceeded: Option<BudgetExceededJson>,
}

impl BudgetJsonInfo {
    pub fn new(preset: impl Into<String>, strict: bool) -> Self {
        Self {
            preset: preset.into(),
            mode: if strict {
                "strict".to_string()
            } else {
                "best-effort".to_string()
            },
            exceeded: None,
        }
    }

    pub fn cli(strict: bool) -> Self {
        Self::new("cli", strict)
    }

    pub fn small(strict: bool) -> Self {
        Self::new("small", strict)
    }
}

/// Backward-compatible alias.
pub type BudgetJson = BudgetJsonInfo;

/// Budget exceeded details.
#[derive(Serialize, Debug, Clone)]
pub struct BudgetExceededJson {
    pub op: String,
    pub metric: String,
    pub used: u64,
    pub limit: u64,
}

/// An error result with stable kind/code for API consumers.
#[derive(Serialize, Debug)]
pub struct ErrorJsonOutput {
    pub schema_version: u8,
    pub ok: bool,
    pub kind: String,
    pub code: String,
    pub error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
}

impl ErrorJsonOutput {
    pub fn new(error: impl Into<String>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            ok: false,
            kind: "InternalError".to_string(),
            code: "E_INTERNAL".to_string(),
            error: error.into(),
            input: None,
        }
    }

    pub fn with_input(error: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            ok: false,
            kind: "InternalError".to_string(),
            code: "E_INTERNAL".to_string(),
            error: error.into(),
            input: Some(input.into()),
        }
    }

    pub fn parse_error(message: impl Into<String>, input: Option<String>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            ok: false,
            kind: "ParseError".to_string(),
            code: "E_PARSE".to_string(),
            error: message.into(),
            input,
        }
    }
}

/// Result of evaluating a single expression via eval-json.
#[derive(Serialize, Debug)]
pub struct EvalJsonOutput {
    pub schema_version: u8,
    pub ok: bool,
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_latex: Option<String>,
    pub result: String,
    pub result_truncated: bool,
    pub result_chars: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_latex: Option<String>,
    pub steps_mode: String,
    pub steps_count: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<StepJson>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub solve_steps: Vec<SolveStepJson>,
    pub warnings: Vec<WarningJson>,
    pub required_conditions: Vec<RequiredConditionJson>,
    pub required_display: Vec<String>,
    pub budget: BudgetJsonInfo,
    pub domain: DomainJson,
    pub stats: ExprStatsJson,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
    pub timings_us: TimingsJson,
    pub options: OptionsJson,
    pub semantics: SemanticsJson,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wire: Option<serde_json::Value>,
}

// =============================================================================
// Engine-style response DTOs (used by FFI fallback paths)
// =============================================================================

/// Source span for JSON serialization.
#[derive(Serialize, Debug, Clone, Copy)]
pub struct SpanJson {
    pub start: usize,
    pub end: usize,
}

/// A sub-step representing a rewrite in a subexpression.
#[derive(Serialize, Debug, Clone)]
pub struct EngineJsonSubstep {
    pub rule: String,
    pub before: String,
    pub after: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// A simplification step in a response.
#[derive(Serialize, Debug, Clone)]
pub struct EngineJsonStep {
    pub phase: String,
    pub rule: String,
    pub before: String,
    pub after: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub substeps: Vec<EngineJsonSubstep>,
}

/// Assumption summary record for response payloads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssumptionRecord {
    pub kind: String,
    pub expr: String,
    pub message: String,
    pub count: u32,
}

/// Structured error in a response.
#[derive(Serialize, Debug, Clone)]
pub struct EngineJsonError {
    pub kind: &'static str,
    pub code: &'static str,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<SpanJson>,
    #[serde(default)]
    pub details: serde_json::Value,
}

impl EngineJsonError {
    pub fn simple(kind: &'static str, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            code,
            message: message.into(),
            span: None,
            details: serde_json::Value::Null,
        }
    }
}

/// Unified JSON response for engine-like operations.
#[derive(Serialize, Debug, Clone)]
pub struct EngineJsonResponse {
    pub schema_version: u8,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<EngineJsonError>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<EngineJsonStep>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<WarningJson>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assumptions: Vec<AssumptionRecord>,
    pub budget: BudgetJsonInfo,
}

impl EngineJsonResponse {
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|e| {
            format!(
                r#"{{"schema_version":1,"ok":false,"error":{{"kind":"InternalError","code":"E_INTERNAL","message":"JSON serialization failed: {}"}}}}"#,
                e
            )
        })
    }

    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| {
            format!(
                r#"{{"schema_version":1,"ok":false,"error":{{"kind":"InternalError","code":"E_INTERNAL","message":"JSON serialization failed: {}"}}}}"#,
                e
            )
        })
    }
}

// =============================================================================
// script-json types
// =============================================================================

/// Result of processing a script via script-json.
#[derive(Serialize, Debug)]
pub struct ScriptJsonOutput {
    pub ok: bool,
    pub lines: Vec<ScriptLineResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_result: Option<EvalJsonOutput>,
    pub total_time_us: u64,
}

/// Result of processing a single line in a script.
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

// =============================================================================
// mm-gcd-modp-json types
// =============================================================================

/// Result of running mm_gcd benchmark via mm-gcd-modp-json.
#[derive(Serialize, Debug)]
pub struct MmGcdModpJsonOutput {
    pub ok: bool,
    pub modulus: u64,
    pub a_terms: usize,
    pub b_terms: usize,
    pub g_terms: usize,
    pub ag_terms: usize,
    pub bg_terms: usize,
    pub gcd_terms: usize,
    pub gcd_total_degree: u32,
    pub gcd_matches_g: bool,
    pub timings_ms: MmGcdTimingsMs,
}

/// Timing breakdown for mm_gcd in milliseconds.
#[derive(Serialize, Debug, Default)]
pub struct MmGcdTimingsMs {
    pub build_ms: f64,
    pub mul_ms: f64,
    pub gcd_ms: f64,
    pub full_ms: f64,
}

// =============================================================================
// OutputEnvelope V1
// =============================================================================

/// Root envelope for all API responses (eval & solve).
#[derive(Serialize, Debug)]
pub struct OutputEnvelope {
    pub schema_version: u8,
    pub engine: EngineInfo,
    pub request: RequestInfo,
    pub result: ResultDto,
    pub transparency: TransparencyDto,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<StepDto>,
}

/// Engine metadata.
#[derive(Serialize, Debug)]
pub struct EngineInfo {
    pub name: String,
    pub version: String,
}

impl Default for EngineInfo {
    fn default() -> Self {
        Self {
            name: "ExpliCAS".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Request information.
#[derive(Serialize, Debug)]
pub struct RequestInfo {
    pub kind: String,
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solve_var: Option<String>,
    pub options: RequestOptions,
}

/// Request options.
#[derive(Serialize, Debug, Default)]
pub struct RequestOptions {
    pub domain_mode: String,
    pub value_domain: String,
    pub hints: bool,
    pub explain: bool,
}

/// Expression with dual rendering.
#[derive(Serialize, Debug, Clone)]
pub struct ExprDto {
    pub display: String,
    pub canonical: String,
}

/// Result (polymorphic by kind).
#[derive(Serialize, Debug)]
#[serde(tag = "kind")]
pub enum ResultDto {
    #[serde(rename = "eval_result")]
    Eval { value: ExprDto },
    #[serde(rename = "solve_result")]
    Solve {
        solutions: SolutionSetDto,
        #[serde(skip_serializing_if = "Option::is_none")]
        residual: Option<ExprDto>,
    },
    #[serde(rename = "error")]
    Error { message: String },
}

/// Solution set (polymorphic by kind).
#[derive(Serialize, Debug)]
#[serde(tag = "kind")]
pub enum SolutionSetDto {
    #[serde(rename = "finite_set")]
    FiniteSet { elements: Vec<ExprDto> },
    #[serde(rename = "all_reals")]
    AllReals,
    #[serde(rename = "empty_set")]
    EmptySet,
    #[serde(rename = "interval")]
    Interval { lower: BoundDto, upper: BoundDto },
    #[serde(rename = "conditional")]
    Conditional { cases: Vec<CaseDto> },
}

/// Interval bound.
#[derive(Serialize, Debug)]
#[serde(tag = "kind")]
pub enum BoundDto {
    #[serde(rename = "closed")]
    Closed { value: ExprDto },
    #[serde(rename = "open")]
    Open { value: ExprDto },
    #[serde(rename = "neg_infinity")]
    NegInfinity,
    #[serde(rename = "infinity")]
    Infinity,
}

/// Conditional case.
#[derive(Serialize, Debug)]
pub struct CaseDto {
    pub when: WhenDto,
    pub then: ThenDto,
}

/// Predicate set for conditional.
#[derive(Serialize, Debug)]
pub struct WhenDto {
    pub predicates: Vec<ConditionDto>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub is_otherwise: bool,
}

/// Result branch for conditional.
#[derive(Serialize, Debug)]
pub struct ThenDto {
    pub solutions: SolutionSetDto,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residual: Option<ExprDto>,
}

/// Transparency section (global summary).
#[derive(Serialize, Debug, Default)]
pub struct TransparencyDto {
    pub required_conditions: Vec<ConditionDto>,
    pub assumptions_used: Vec<AssumptionDto>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub blocked_hints: Vec<BlockedHintDto>,
}

/// Domain condition (Requires).
#[derive(Serialize, Debug, Clone)]
pub struct ConditionDto {
    pub kind: String,
    pub display: String,
    pub expr_display: String,
    pub expr_canonical: String,
}

/// Assumption used (Assumed).
#[derive(Serialize, Debug, Clone)]
pub struct AssumptionDto {
    pub kind: String,
    pub display: String,
    pub expr_canonical: String,
    pub rule: String,
}

/// Blocked hint.
#[derive(Serialize, Debug, Clone)]
pub struct BlockedHintDto {
    pub rule: String,
    pub requires: Vec<String>,
    pub tip: String,
}

/// Step DTO for trace.
#[derive(Serialize, Debug)]
pub struct StepDto {
    pub index: usize,
    pub rule: String,
    pub before: ExprDto,
    pub after: ExprDto,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub assumptions_used: Vec<AssumptionDto>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub required_conditions: Vec<ConditionDto>,
}
