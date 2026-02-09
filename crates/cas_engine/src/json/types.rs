use serde::Serialize;

use super::response::*;

// =============================================================================
// Shared CLI/FFI Types (canonical definitions)
// =============================================================================

/// Expression statistics (node count, depth).
#[derive(Serialize, Debug, Default)]
pub struct ExprStatsJson {
    pub node_count: usize,
    pub depth: usize,
    /// Number of polynomial terms (for expanded polynomials)
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
    /// Current domain mode: "strict", "generic", or "assume"
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
    /// Domain assumption mode
    pub domain_mode: String,
    /// Value domain (real/complex)
    pub value_domain: String,
    /// Branch policy for multi-valued functions
    pub branch: String,
    /// Inverse trig composition policy
    pub inv_trig: String,
    /// Assume scope (only active when domain_mode=assume)
    pub assume_scope: String,
}

/// A required condition (implicit domain constraint) from the input expression.
/// These are NOT assumptions - they were already implied by the input structure.
#[derive(Serialize, Debug, Clone)]
pub struct RequiredConditionJson {
    /// Condition kind: "NonNegative", "Positive", or "NonZero"
    pub kind: String,
    /// Human-readable expression display (may vary with display transforms/scopes)
    pub expr_display: String,
    /// Canonical expression string (stable, without transforms/scopes)
    pub expr_canonical: String,
}

// =============================================================================
// OutputEnvelope V1 — Stable Android/FFI API
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
    pub kind: String, // "eval" | "solve"
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

// =============================================================================
// Eval-JSON Types (shared between CLI and FFI)
// =============================================================================

/// A simplification step for JSON output.
#[derive(Serialize, Debug, Clone)]
pub struct StepJson {
    /// Step index (1-based)
    pub index: usize,
    /// Rule name that was applied
    pub rule: String,
    /// Rule displayed as LaTeX: "red(antecedent) → green(consequent)"
    pub rule_latex: String,
    /// Expression before transformation (plain text)
    pub before: String,
    /// Expression after transformation (plain text)
    pub after: String,
    /// LaTeX for before expression (for MathJax rendering)
    pub before_latex: String,
    /// LaTeX for after expression (for MathJax rendering)
    pub after_latex: String,
    /// Optional sub-steps for detailed explanations
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub substeps: Vec<SubStepJson>,
}

/// A sub-step within a step for detailed explanations.
#[derive(Serialize, Debug, Clone)]
pub struct SubStepJson {
    /// Title of the sub-step
    pub title: String,
    /// Explanation lines (for engine substeps)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub lines: Vec<String>,
    /// LaTeX for before expression (for didactic substeps)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before_latex: Option<String>,
    /// LaTeX for after expression (for didactic substeps)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_latex: Option<String>,
}

/// A solver step for equation-solving JSON output.
#[derive(Serialize, Debug, Clone)]
pub struct SolveStepJson {
    /// Step index (1-based)
    pub index: usize,
    /// Description of the step (e.g., "Subtract 3 from both sides")
    pub description: String,
    /// Equation after this step as plain text
    pub equation: String,
    /// LHS of equation after step (LaTeX)
    pub lhs_latex: String,
    /// Relation operator (=, <, >, etc.)
    pub relop: String,
    /// RHS of equation after step (LaTeX)
    pub rhs_latex: String,
    /// Optional sub-steps for detailed derivation
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub substeps: Vec<SolveSubStepJson>,
}

/// A sub-step within a solve step for detailed derivation.
#[derive(Serialize, Debug, Clone)]
pub struct SolveSubStepJson {
    /// Step index (e.g., "1.1", "1.2")
    pub index: String,
    /// Description of the sub-step
    pub description: String,
    /// Equation after this sub-step as plain text
    pub equation: String,
    /// LHS of equation after sub-step (LaTeX)
    pub lhs_latex: String,
    /// Relation operator
    pub relop: String,
    /// RHS of equation after sub-step (LaTeX)
    pub rhs_latex: String,
}

/// A domain assumption warning with its source rule.
#[derive(Serialize, Debug, Clone)]
pub struct WarningJson {
    pub rule: String,
    pub assumption: String,
}

/// An error result with stable kind/code for API consumers.
///
/// The `kind` and `code` fields are stable and should not change between versions.
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
    pub fn from_cas_error(e: &crate::error::CasError, input: Option<String>) -> Self {
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

/// Result of evaluating a single expression via eval-json.
///
/// This is the rich output format for the `eval-json` CLI subcommand.
/// The `wire` field uses `serde_json::Value` to avoid coupling to any
/// specific wire-protocol crate.
#[derive(Serialize, Debug)]
pub struct EvalJsonOutput {
    /// Schema version for API stability (increment on breaking changes)
    pub schema_version: u8,

    pub ok: bool,
    pub input: String,

    /// LaTeX formatted input expression for rendering
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_latex: Option<String>,

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
    pub budget: BudgetJsonInfo,

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

    /// Unified wire output (stable messaging format).
    /// Serialized as opaque JSON to avoid coupling to wire-protocol types.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wire: Option<serde_json::Value>,
}
