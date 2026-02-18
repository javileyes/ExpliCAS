use crate::budget::BudgetExceeded;
use crate::error::CasError;

pub use cas_api_models::SCHEMA_VERSION;

pub use cas_api_models::{BudgetOpts, JsonRunOptions};

/// Compatibility constructor for engine warnings.
pub trait EngineJsonWarningExt {
    fn budget_exceeded(b: &BudgetExceeded) -> Self;
    fn domain_assumption(rule: &str, assumption: &str) -> Self;
}

impl EngineJsonWarningExt for cas_api_models::EngineJsonWarning {
    fn budget_exceeded(b: &BudgetExceeded) -> Self {
        Self {
            kind: "BudgetExceeded".to_string(),
            message: format!(
                "Budget exceeded: {:?}/{:?} used={} limit={}",
                b.op, b.metric, b.used, b.limit
            ),
        }
    }

    fn domain_assumption(rule: &str, assumption: &str) -> Self {
        Self {
            kind: "DomainAssumption".to_string(),
            message: format!("{}: {}", rule, assumption),
        }
    }
}

/// Compatibility constructor for `BudgetJsonInfo::with_exceeded`.
pub trait BudgetJsonInfoExt {
    fn with_exceeded(self, b: &BudgetExceeded) -> Self;
}

impl BudgetJsonInfoExt for cas_api_models::BudgetJsonInfo {
    fn with_exceeded(mut self, b: &BudgetExceeded) -> Self {
        self.exceeded = Some(cas_api_models::BudgetExceededJson {
            op: format!("{:?}", b.op),
            metric: format!("{:?}", b.metric),
            used: b.used,
            limit: b.limit,
        });
        self
    }
}

/// Compatibility constructor for `EngineJsonError::from_cas_error`.
pub trait EngineJsonErrorExt {
    fn from_cas_error(e: &CasError) -> Self;
}

impl EngineJsonErrorExt for cas_api_models::EngineJsonError {
    fn from_cas_error(e: &CasError) -> Self {
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
            span: None,
            details,
        }
    }
}

/// Compatibility constructor for `EngineJsonResponse::err`.
pub trait EngineJsonResponseExt {
    fn err(error: &CasError, budget: cas_api_models::BudgetJsonInfo) -> Self;
}

impl EngineJsonResponseExt for cas_api_models::EngineJsonResponse {
    fn err(error: &CasError, budget: cas_api_models::BudgetJsonInfo) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            ok: false,
            result: None,
            error: Some(cas_api_models::EngineJsonError::from_cas_error(error)),
            steps: vec![],
            warnings: vec![],
            assumptions: vec![],
            budget,
        }
    }
}
