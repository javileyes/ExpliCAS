//! Canonical JSON API types for engine responses.
//!
//! This module provides stable, serializable types for CLI and FFI consumers.
//! All callsites should use these types to ensure consistent JSON schema.
//!
//! # Schema Version
//!
//! Current schema version: **1**
//!
//! # Stability Contract
//!
//! - `schema_version`, `ok`, `kind`, `code` are **stable** - do not change
//! - `message` is human-readable and may change between versions
//! - `details` is extensible (new keys may be added)
//!
//! # Example Response (Success)
//!
//! ```json
//! {
//!   "schema_version": 1,
//!   "ok": true,
//!   "result": "x^2 + 2*x + 1",
//!   "budget": { "preset": "cli", "mode": "strict" }
//! }
//! ```
//!
//! # Example Response (Error)
//!
//! ```json
//! {
//!   "schema_version": 1,
//!   "ok": false,
//!   "error": {
//!     "kind": "DomainError",
//!     "code": "E_DIV_ZERO",
//!     "message": "division by zero"
//!   },
//!   "budget": { "preset": "cli", "mode": "strict" }
//! }
//! ```

use cas_ast::Span;
use serde::{Deserialize, Serialize};

use crate::budget::BudgetExceeded;
use crate::error::CasError;

/// Current JSON schema version.
pub const SCHEMA_VERSION: u8 = 1;

// =============================================================================
// Main Response Type
// =============================================================================

/// Unified JSON response for all engine operations.
///
/// Use `EngineJsonResponse::ok()` or `EngineJsonResponse::err()` to construct.
#[derive(Serialize, Debug)]
pub struct EngineJsonResponse {
    /// Schema version for API stability (currently 1)
    pub schema_version: u8,

    /// True if operation succeeded
    pub ok: bool,

    /// Result expression (success only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,

    /// Error details (failure only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<EngineJsonError>,

    /// Simplification steps (if requested)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<EngineJsonStep>,

    /// Warnings (domain assumptions, budget soft-exceeded, etc.)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<EngineJsonWarning>,

    /// Collected assumptions (deduplicated, with counts)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assumptions: Vec<crate::assumptions::AssumptionRecord>,

    /// Budget information (always present)
    pub budget: BudgetJsonInfo,
}

impl EngineJsonResponse {
    /// Create a success response.
    ///
    /// # Example
    ///
    /// ```
    /// use cas_engine::{EngineJsonResponse, BudgetJsonInfo};
    ///
    /// let budget = BudgetJsonInfo::cli(true);
    /// let resp = EngineJsonResponse::ok("2*x".into(), budget);
    ///
    /// assert!(resp.ok);
    /// assert_eq!(resp.result, Some("2*x".into()));
    /// ```
    pub fn ok(result: String, budget: BudgetJsonInfo) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            ok: true,
            result: Some(result),
            error: None,
            steps: vec![],
            warnings: vec![],
            assumptions: vec![],
            budget,
        }
    }

    /// Create a success response with steps.
    ///
    /// # Example
    ///
    /// ```
    /// use cas_engine::{EngineJsonResponse, EngineJsonStep, BudgetJsonInfo};
    ///
    /// let step = EngineJsonStep {
    ///     phase: "Simplify".into(),
    ///     rule: "Combine".into(),
    ///     before: "x+x".into(),
    ///     after: "2*x".into(),
    ///     substeps: vec![],
    /// };
    /// let budget = BudgetJsonInfo::cli(false);
    /// let resp = EngineJsonResponse::ok_with_steps("2*x".into(), vec![step], budget);
    ///
    /// assert!(resp.ok);
    /// assert_eq!(resp.steps.len(), 1);
    /// ```
    pub fn ok_with_steps(
        result: String,
        steps: Vec<EngineJsonStep>,
        budget: BudgetJsonInfo,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            ok: true,
            result: Some(result),
            error: None,
            steps,
            warnings: vec![],
            assumptions: vec![],
            budget,
        }
    }

    /// Create an error response.
    pub fn err(error: &CasError, budget: BudgetJsonInfo) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            ok: false,
            result: None,
            error: Some(EngineJsonError::from_cas_error(error)),
            steps: vec![],
            warnings: vec![],
            assumptions: vec![],
            budget,
        }
    }

    /// Add a warning to this response.
    pub fn with_warning(mut self, warning: EngineJsonWarning) -> Self {
        self.warnings.push(warning);
        self
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|e| {
            format!(r#"{{"schema_version":1,"ok":false,"error":{{"kind":"InternalError","code":"E_INTERNAL","message":"JSON serialization failed: {}"}}}}"#, e)
        })
    }

    /// Serialize to pretty JSON string.
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| {
            format!(r#"{{"schema_version":1,"ok":false,"error":{{"kind":"InternalError","code":"E_INTERNAL","message":"JSON serialization failed: {}"}}}}"#, e)
        })
    }
}

// =============================================================================
// Error Type
// =============================================================================

/// Structured error in JSON response.
///
/// Fields `kind` and `code` are **stable** and must not change between versions.
#[derive(Serialize, Debug)]
pub struct EngineJsonError {
    /// Stable error kind for routing (ParseError, DomainError, etc.)
    pub kind: &'static str,

    /// Stable error code for UI mapping (E_PARSE, E_DIV_ZERO, etc.)
    pub code: &'static str,

    /// Human-readable error message (may change between versions)
    pub message: String,

    /// Source location (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<SpanJson>,

    /// Additional structured details (extensible)
    #[serde(default)]
    pub details: serde_json::Value,
}

impl EngineJsonError {
    /// Create from a CasError.
    pub fn from_cas_error(e: &CasError) -> Self {
        let details = match e {
            CasError::BudgetExceeded(b) => serde_json::json!({
                "op": format!("{:?}", b.op),
                "metric": format!("{:?}", b.metric),
                "used": b.used,
                "limit": b.limit,
            }),
            CasError::InvalidMatrix { reason } => serde_json::json!({
                "reason": reason
            }),
            CasError::ConversionFailed { from, to } => serde_json::json!({
                "from": from,
                "to": to
            }),
            _ => serde_json::Value::Null,
        };

        Self {
            kind: e.kind(),
            code: e.code(),
            message: e.to_string(),
            span: None, // TODO: extract from ParseError when available
            details,
        }
    }

    /// Create a simple error without details.
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

// =============================================================================
// Supporting Types
// =============================================================================

/// Source span for JSON serialization.
#[derive(Serialize, Debug, Clone, Copy)]
pub struct SpanJson {
    pub start: usize,
    pub end: usize,
}

impl From<Span> for SpanJson {
    fn from(s: Span) -> Self {
        SpanJson {
            start: s.start,
            end: s.end,
        }
    }
}

/// Budget information in JSON response.
#[derive(Serialize, Debug, Default)]
pub struct BudgetJsonInfo {
    /// Budget preset name
    pub preset: String,

    /// Budget mode: "strict" or "best-effort"
    pub mode: String,

    /// Budget exceeded info (only in best-effort mode)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exceeded: Option<BudgetExceededJson>,
}

impl BudgetJsonInfo {
    /// Create a new budget info with preset name.
    pub fn new(preset: impl Into<String>, strict: bool) -> Self {
        Self {
            preset: preset.into(),
            mode: if strict {
                "strict".into()
            } else {
                "best-effort".into()
            },
            exceeded: None,
        }
    }

    /// Create CLI budget info (convenience).
    pub fn cli(strict: bool) -> Self {
        Self::new("cli", strict)
    }

    /// Create small budget info (convenience).
    pub fn small(strict: bool) -> Self {
        Self::new("small", strict)
    }

    /// Mark budget as exceeded (for best-effort mode).
    pub fn with_exceeded(mut self, b: &BudgetExceeded) -> Self {
        self.exceeded = Some(BudgetExceededJson {
            op: format!("{:?}", b.op),
            metric: format!("{:?}", b.metric),
            used: b.used,
            limit: b.limit,
        });
        self
    }
}

/// Budget exceeded details.
#[derive(Serialize, Debug)]
pub struct BudgetExceededJson {
    pub op: String,
    pub metric: String,
    pub used: u64,
    pub limit: u64,
}

/// A simplification step in JSON response.
#[derive(Serialize, Debug)]
pub struct EngineJsonStep {
    /// Phase name (Core, Transform, etc.)
    pub phase: String,

    /// Rule that was applied
    pub rule: String,

    /// Expression before (must NOT contain __hold)
    pub before: String,

    /// Expression after (must NOT contain __hold)
    pub after: String,

    /// Sub-steps that occurred in subexpressions before this step
    /// (e.g., EvenPowSubSwapRule rewriting (y-x)^2 → (x-y)^2 before cancellation)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub substeps: Vec<EngineJsonSubstep>,
}

/// A sub-step representing a rewrite in a subexpression.
#[derive(Serialize, Debug, Clone)]
pub struct EngineJsonSubstep {
    /// Rule that was applied
    pub rule: String,

    /// Local expression before (the subexpression, not full expression)
    pub before: String,

    /// Local expression after
    pub after: String,

    /// Optional note explaining the transformation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// A warning in JSON response.
#[derive(Serialize, Debug)]
pub struct EngineJsonWarning {
    /// Warning kind
    pub kind: String,

    /// Warning message
    pub message: String,
}

impl EngineJsonWarning {
    /// Create a budget exceeded warning.
    pub fn budget_exceeded(b: &BudgetExceeded) -> Self {
        Self {
            kind: "BudgetExceeded".into(),
            message: format!(
                "Budget exceeded: {:?}/{:?} used={} limit={}",
                b.op, b.metric, b.used, b.limit
            ),
        }
    }

    /// Create a domain assumption warning.
    pub fn domain_assumption(rule: &str, assumption: &str) -> Self {
        Self {
            kind: "DomainAssumption".into(),
            message: format!("{}: {}", rule, assumption),
        }
    }
}

// =============================================================================
// Run Options (input from CLI/FFI)
// =============================================================================

/// Options for JSON evaluation (parsed from optsJson).
#[derive(Deserialize, Debug, Default)]
pub struct JsonRunOptions {
    /// Budget configuration
    #[serde(default)]
    pub budget: BudgetOpts,

    /// Include simplification steps in output
    #[serde(default)]
    pub steps: bool,

    /// Pretty-print JSON output
    #[serde(default)]
    pub pretty: bool,
}

/// Budget options in JSON input.
#[derive(Deserialize, Debug)]
pub struct BudgetOpts {
    /// Preset name: "small", "cli", "unlimited"
    #[serde(default = "default_preset")]
    pub preset: String,

    /// Mode: "strict" or "best-effort"
    #[serde(default = "default_mode")]
    pub mode: String,
}

impl Default for BudgetOpts {
    fn default() -> Self {
        Self {
            preset: default_preset(),
            mode: default_mode(),
        }
    }
}

fn default_preset() -> String {
    "cli".into()
}

fn default_mode() -> String {
    "best-effort".into()
}

// =============================================================================
// Unified Evaluation Entry Point (for CLI and FFI)
// =============================================================================

/// Evaluate an expression and return JSON response.
///
/// This is the **canonical entry point** for all JSON-returning evaluation.
/// Both CLI and FFI should use this to ensure consistent behavior.
///
/// # Arguments
/// * `expr` - Expression string to evaluate
/// * `opts_json` - Options JSON string (see `JsonRunOptions`)
///
/// # Returns
/// JSON string with `EngineJsonResponse` (schema v1).
/// Always returns valid JSON, even on errors.
///
/// # Example
/// ```
/// use cas_engine::json::eval_str_to_json;
///
/// let json = eval_str_to_json("x + x", r#"{"budget":{"preset":"cli"}}"#);
/// assert!(json.contains("\"ok\":true"));
/// ```
pub fn eval_str_to_json(expr: &str, opts_json: &str) -> String {
    // Parse options (with defaults)
    let opts: JsonRunOptions = match serde_json::from_str(opts_json) {
        Ok(o) => o,
        Err(e) => {
            // Invalid optsJson -> return error response
            let budget = BudgetJsonInfo::new("unknown", true);
            let error = EngineJsonError {
                kind: "InvalidInput",
                code: "E_INVALID_INPUT",
                message: format!("Invalid options JSON: {}", e),
                span: None,
                details: serde_json::json!({ "error": e.to_string() }),
            };
            let resp = EngineJsonResponse {
                schema_version: SCHEMA_VERSION,
                ok: false,
                result: None,
                error: Some(error),
                steps: vec![],
                warnings: vec![],
                assumptions: vec![],
                budget,
            };
            return if opts_json.contains("\"pretty\":true") {
                resp.to_json_pretty()
            } else {
                resp.to_json()
            };
        }
    };

    let strict = opts.budget.mode == "strict";
    let budget_info = BudgetJsonInfo::new(&opts.budget.preset, strict);

    // Create engine and session state
    let mut engine = crate::eval::Engine::new();
    let mut state = crate::session_state::SessionState::new();

    // Parse expression
    let parsed = match cas_parser::parse(expr, &mut engine.simplifier.context) {
        Ok(id) => id,
        Err(e) => {
            let error = EngineJsonError {
                kind: "ParseError",
                code: "E_PARSE",
                message: e.to_string(),
                span: e.span().map(SpanJson::from),
                details: serde_json::Value::Null,
            };
            let resp = EngineJsonResponse {
                schema_version: SCHEMA_VERSION,
                ok: false,
                result: None,
                error: Some(error),
                steps: vec![],
                warnings: vec![],
                assumptions: vec![],
                budget: budget_info,
            };
            return if opts.pretty {
                resp.to_json_pretty()
            } else {
                resp.to_json()
            };
        }
    };

    // Build eval request
    let req = crate::eval::EvalRequest {
        raw_input: expr.to_string(),
        parsed,
        kind: crate::session::EntryKind::Expr(parsed),
        action: crate::eval::EvalAction::Simplify,
        auto_store: false,
    };

    // Evaluate
    let output = match engine.eval(&mut state, req) {
        Ok(o) => o,
        Err(e) => {
            // anyhow::Error - create generic error
            let error = EngineJsonError::simple("InternalError", "E_INTERNAL", e.to_string());
            let resp = EngineJsonResponse {
                schema_version: SCHEMA_VERSION,
                ok: false,
                result: None,
                error: Some(error),
                steps: vec![],
                warnings: vec![],
                assumptions: vec![],
                budget: budget_info,
            };
            return if opts.pretty {
                resp.to_json_pretty()
            } else {
                resp.to_json()
            };
        }
    };

    // Format result
    let result_str = match &output.result {
        crate::eval::EvalResult::Expr(e) => {
            let clean = crate::engine::strip_all_holds(&mut engine.simplifier.context, *e);
            format!(
                "{}",
                cas_ast::display::DisplayExpr {
                    context: &engine.simplifier.context,
                    id: clean
                }
            )
        }
        crate::eval::EvalResult::Set(v) if !v.is_empty() => {
            let clean = crate::engine::strip_all_holds(&mut engine.simplifier.context, v[0]);
            format!(
                "{}",
                cas_ast::display::DisplayExpr {
                    context: &engine.simplifier.context,
                    id: clean
                }
            )
        }
        crate::eval::EvalResult::Bool(b) => b.to_string(),
        _ => "(no result)".to_string(),
    };

    // Build steps (if requested)
    let steps = if opts.steps {
        output
            .steps
            .iter()
            .map(|s| {
                let before_str = s.global_before.map(|id| {
                    let clean = crate::engine::strip_all_holds(&mut engine.simplifier.context, id);
                    format!(
                        "{}",
                        cas_ast::display::DisplayExpr {
                            context: &engine.simplifier.context,
                            id: clean
                        }
                    )
                });
                let after_str = s.global_after.map(|id| {
                    let clean = crate::engine::strip_all_holds(&mut engine.simplifier.context, id);
                    format!(
                        "{}",
                        cas_ast::display::DisplayExpr {
                            context: &engine.simplifier.context,
                            id: clean
                        }
                    )
                });
                EngineJsonStep {
                    phase: "Simplify".to_string(),
                    rule: s.rule_name.clone(),
                    before: before_str.unwrap_or_default(),
                    after: after_str.unwrap_or_default(),
                    substeps: vec![],
                }
            })
            .collect()
    } else {
        vec![]
    };

    let resp = EngineJsonResponse::ok_with_steps(result_str, steps, budget_info);
    if opts.pretty {
        resp.to_json_pretty()
    } else {
        resp.to_json()
    }
}

// =============================================================================
// Substitute JSON Entry Point
// =============================================================================

/// Options for substitute JSON operation.
#[derive(Deserialize, Debug)]
pub struct SubstituteJsonOptions {
    /// Substitution mode: "exact" or "power" (default: "power")
    #[serde(default = "default_substitute_mode")]
    pub mode: String,

    /// Include substitution steps in output
    #[serde(default)]
    pub steps: bool,

    /// Pretty-print JSON output
    #[serde(default)]
    pub pretty: bool,
}

impl Default for SubstituteJsonOptions {
    fn default() -> Self {
        Self {
            mode: "power".into(),
            steps: false,
            pretty: false,
        }
    }
}

fn default_substitute_mode() -> String {
    "power".into()
}

/// Substitute JSON response with request echo and options.
#[derive(Serialize, Debug)]
pub struct SubstituteJsonResponse {
    /// Schema version for API stability
    pub schema_version: u8,

    /// True if operation succeeded
    pub ok: bool,

    /// Result expression (success only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,

    /// Error details (failure only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<EngineJsonError>,

    /// Request echo for reproducibility
    pub request: SubstituteRequestEcho,

    /// Options used
    pub options: SubstituteOptionsJson,

    /// Substitution steps (if requested)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<EngineJsonSubstep>,
}

impl SubstituteJsonResponse {
    /// Serialize to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|e| {
            format!(r#"{{"schema_version":1,"ok":false,"error":{{"kind":"InternalError","code":"E_INTERNAL","message":"JSON serialization failed: {}"}}}}"#, e)
        })
    }

    /// Serialize to pretty JSON string.
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| {
            format!(r#"{{"schema_version":1,"ok":false,"error":{{"kind":"InternalError","code":"E_INTERNAL","message":"JSON serialization failed: {}"}}}}"#, e)
        })
    }
}

/// Request echo for substitute.
#[derive(Serialize, Debug)]
pub struct SubstituteRequestEcho {
    pub expr: String,
    pub target: String,
    #[serde(rename = "with")]
    pub with_expr: String,
}

/// Options echo for substitute.
#[derive(Serialize, Debug)]
pub struct SubstituteOptionsJson {
    pub substitute: SubstituteOptionsInner,
}

#[derive(Serialize, Debug)]
pub struct SubstituteOptionsInner {
    pub mode: String,
    pub steps: bool,
}

/// Substitute an expression and return JSON response.
///
/// This is the **canonical entry point** for all JSON-returning substitution.
/// Both CLI and FFI should use this to ensure consistent behavior.
///
/// # Arguments
/// * `expr_str` - Expression string to substitute in
/// * `target_str` - Target expression to replace
/// * `with_str` - Replacement expression
/// * `opts_json` - Options JSON string (optional, see `SubstituteJsonOptions`)
///
/// # Returns
/// JSON string with `SubstituteJsonResponse` (schema v1).
/// Always returns valid JSON, even on errors.
pub fn substitute_str_to_json(
    expr_str: &str,
    target_str: &str,
    with_str: &str,
    opts_json: Option<&str>,
) -> String {
    // Parse options (with defaults)
    let opts: SubstituteJsonOptions = match opts_json {
        Some(json) => serde_json::from_str(json).unwrap_or_default(),
        None => SubstituteJsonOptions::default(),
    };

    let request = SubstituteRequestEcho {
        expr: expr_str.to_string(),
        target: target_str.to_string(),
        with_expr: with_str.to_string(),
    };

    let options = SubstituteOptionsJson {
        substitute: SubstituteOptionsInner {
            mode: opts.mode.clone(),
            steps: opts.steps,
        },
    };

    // Create context
    let mut ctx = cas_ast::Context::new();

    // Parse expressions
    let expr = match cas_parser::parse(expr_str, &mut ctx) {
        Ok(id) => id,
        Err(e) => {
            let resp = SubstituteJsonResponse {
                schema_version: SCHEMA_VERSION,
                ok: false,
                result: None,
                error: Some(EngineJsonError {
                    kind: "ParseError",
                    code: "E_PARSE",
                    message: format!("Failed to parse expression: {}", e),
                    span: e.span().map(SpanJson::from),
                    details: serde_json::Value::Null,
                }),
                request,
                options,
                steps: vec![],
            };
            return if opts.pretty {
                resp.to_json_pretty()
            } else {
                resp.to_json()
            };
        }
    };

    let target = match cas_parser::parse(target_str, &mut ctx) {
        Ok(id) => id,
        Err(e) => {
            let resp = SubstituteJsonResponse {
                schema_version: SCHEMA_VERSION,
                ok: false,
                result: None,
                error: Some(EngineJsonError {
                    kind: "ParseError",
                    code: "E_PARSE",
                    message: format!("Failed to parse target: {}", e),
                    span: e.span().map(SpanJson::from),
                    details: serde_json::Value::Null,
                }),
                request,
                options,
                steps: vec![],
            };
            return if opts.pretty {
                resp.to_json_pretty()
            } else {
                resp.to_json()
            };
        }
    };

    let replacement = match cas_parser::parse(with_str, &mut ctx) {
        Ok(id) => id,
        Err(e) => {
            let resp = SubstituteJsonResponse {
                schema_version: SCHEMA_VERSION,
                ok: false,
                result: None,
                error: Some(EngineJsonError {
                    kind: "ParseError",
                    code: "E_PARSE",
                    message: format!("Failed to parse replacement: {}", e),
                    span: e.span().map(SpanJson::from),
                    details: serde_json::Value::Null,
                }),
                request,
                options,
                steps: vec![],
            };
            return if opts.pretty {
                resp.to_json_pretty()
            } else {
                resp.to_json()
            };
        }
    };

    // Build substitute options
    let sub_opts = match opts.mode.as_str() {
        "exact" => crate::substitute::SubstituteOptions::exact(),
        _ => crate::substitute::SubstituteOptions::power_aware_no_remainder(),
    };

    // Perform substitution
    let sub_result =
        crate::substitute::substitute_with_steps(&mut ctx, expr, target, replacement, sub_opts);

    // Strip __hold from result
    let clean_result = crate::engine::strip_all_holds(&mut ctx, sub_result.expr);
    let result_str = format!(
        "{}",
        cas_ast::display::DisplayExpr {
            context: &ctx,
            id: clean_result
        }
    );

    // Convert steps to JSON format
    let json_steps: Vec<EngineJsonSubstep> = if opts.steps {
        sub_result
            .steps
            .into_iter()
            .map(|s| EngineJsonSubstep {
                rule: s.rule,
                before: s.before,
                after: s.after,
                note: s.note,
            })
            .collect()
    } else {
        vec![]
    };

    let resp = SubstituteJsonResponse {
        schema_version: SCHEMA_VERSION,
        ok: true,
        result: Some(result_str),
        error: None,
        request,
        options,
        steps: json_steps,
    };

    if opts.pretty {
        resp.to_json_pretty()
    } else {
        resp.to_json()
    }
}

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
    pub fn from_cas_error(e: &CasError, input: Option<String>) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::{Metric, Operation};

    #[test]
    fn test_response_ok_serialization() {
        let budget = BudgetJsonInfo::cli(true);
        let resp = EngineJsonResponse::ok("x + 1".into(), budget);

        let json = resp.to_json();
        assert!(json.contains(r#""ok":true"#));
        assert!(json.contains(r#""schema_version":1"#));
        assert!(json.contains(r#""result":"x + 1""#));
    }

    #[test]
    fn test_response_err_serialization() {
        let budget = BudgetJsonInfo::cli(true);
        let err = CasError::DivisionByZero;
        let resp = EngineJsonResponse::err(&err, budget);

        let json = resp.to_json();
        assert!(json.contains(r#""ok":false"#));
        assert!(json.contains(r#""kind":"DomainError""#));
        assert!(json.contains(r#""code":"E_DIV_ZERO""#));
    }

    #[test]
    fn test_budget_exceeded_details() {
        let budget_err = CasError::BudgetExceeded(BudgetExceeded {
            op: Operation::Expand,
            metric: Metric::TermsMaterialized,
            used: 150,
            limit: 100,
        });

        let json_err = EngineJsonError::from_cas_error(&budget_err);
        assert_eq!(json_err.kind, "BudgetExceeded");
        assert_eq!(json_err.code, "E_BUDGET");
        assert!(json_err.details["used"].as_u64() == Some(150));
        assert!(json_err.details["limit"].as_u64() == Some(100));
    }

    #[test]
    fn test_warning_creation() {
        let warn = EngineJsonWarning::domain_assumption("PowerRule", "x ≠ 0");
        assert_eq!(warn.kind, "DomainAssumption");
        assert!(warn.message.contains("x ≠ 0"));
    }
}
