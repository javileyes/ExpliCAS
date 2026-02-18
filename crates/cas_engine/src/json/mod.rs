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

mod eval;
mod response;
mod substitute;

// Re-export everything for backward compatibility
pub use eval::*;
pub use response::*;
pub use substitute::*;

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
        let err = crate::error::CasError::DivisionByZero;
        let resp = EngineJsonResponse::err(&err, budget);

        let json = resp.to_json();
        assert!(json.contains(r#""ok":false"#));
        assert!(json.contains(r#""kind":"DomainError""#));
        assert!(json.contains(r#""code":"E_DIV_ZERO""#));
    }

    #[test]
    fn test_budget_exceeded_details() {
        let budget_err = crate::error::CasError::BudgetExceeded(crate::budget::BudgetExceeded {
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
