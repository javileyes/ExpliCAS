use serde::Serialize;

use crate::budget::BudgetExceeded;
use crate::error::CasError;

pub use cas_api_models::SCHEMA_VERSION;

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

pub type SpanJson = cas_api_models::SpanJson;

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

pub type BudgetExceededJson = cas_api_models::BudgetExceededJson;
pub type EngineJsonStep = cas_api_models::EngineJsonStep;
pub type EngineJsonSubstep = cas_api_models::EngineJsonSubstep;
pub type EngineJsonWarning = cas_api_models::EngineJsonWarning;

/// Constructors for engine warning DTOs.
pub trait EngineJsonWarningExt {
    /// Create a budget exceeded warning.
    fn budget_exceeded(b: &BudgetExceeded) -> Self;

    /// Create a domain assumption warning.
    fn domain_assumption(rule: &str, assumption: &str) -> Self;
}

impl EngineJsonWarningExt for EngineJsonWarning {
    fn budget_exceeded(b: &BudgetExceeded) -> Self {
        Self {
            kind: "BudgetExceeded".into(),
            message: format!(
                "Budget exceeded: {:?}/{:?} used={} limit={}",
                b.op, b.metric, b.used, b.limit
            ),
        }
    }

    fn domain_assumption(rule: &str, assumption: &str) -> Self {
        Self {
            kind: "DomainAssumption".into(),
            message: format!("{}: {}", rule, assumption),
        }
    }
}

pub use cas_api_models::{BudgetOpts, JsonRunOptions};
