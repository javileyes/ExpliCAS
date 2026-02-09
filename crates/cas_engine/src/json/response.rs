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
