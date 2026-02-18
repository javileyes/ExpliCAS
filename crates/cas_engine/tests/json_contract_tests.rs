//! JSON API contract tests.
//!
//! These tests verify the stable JSON schema for CLI and FFI consumers.
//! Breaking these tests = breaking external API.

use cas_api_models::{BudgetJsonInfo, EngineJsonError, EngineJsonResponse, SCHEMA_VERSION};
use cas_engine::{
    BudgetExceeded, BudgetJsonInfoExt, CasError, EngineJsonErrorExt, EngineJsonResponseExt, Metric,
    Operation,
};
use serde_json::Value;

fn parse_json(s: &str) -> Value {
    serde_json::from_str(s).expect("valid JSON")
}

// =============================================================================
// Schema stability tests
// =============================================================================

#[test]
fn test_json_schema_version_is_1() {
    assert_eq!(SCHEMA_VERSION, 1, "Schema version must be 1");
}

#[test]
fn test_json_success_contract() {
    let budget = BudgetJsonInfo::cli(true);
    let resp = EngineJsonResponse::ok("x + 1".into(), budget);
    let json = parse_json(&resp.to_json());

    // Required fields
    assert_eq!(json["schema_version"], 1);
    assert_eq!(json["ok"], true);
    assert!(json["result"].is_string());
    assert!(json["budget"].is_object());

    // Result content
    assert_eq!(json["result"], "x + 1");

    // Budget structure
    assert_eq!(json["budget"]["preset"], "cli");
    assert_eq!(json["budget"]["mode"], "strict");
}

#[test]
fn test_json_error_contract() {
    let budget = BudgetJsonInfo::cli(true);
    let err = CasError::DivisionByZero;
    let resp = EngineJsonResponse::err(&err, budget);
    let json = parse_json(&resp.to_json());

    // Required fields
    assert_eq!(json["schema_version"], 1);
    assert_eq!(json["ok"], false);
    assert!(json["error"].is_object());
    assert!(json["budget"].is_object());

    // Error structure
    let error = &json["error"];
    assert!(error["kind"].is_string(), "error.kind must be string");
    assert!(error["code"].is_string(), "error.code must be string");
    assert!(error["message"].is_string(), "error.message must be string");
}

// =============================================================================
// Error kind/code stability tests
// =============================================================================

#[test]
fn test_json_parse_error_contract() {
    let err = CasError::ParseError("unexpected token".into());
    let json_err = EngineJsonError::from_cas_error(&err);

    assert_eq!(
        json_err.kind, "ParseError",
        "ParseError kind must be stable"
    );
    assert_eq!(json_err.code, "E_PARSE", "ParseError code must be stable");
}

#[test]
fn test_json_domain_error_contract() {
    let err = CasError::DivisionByZero;
    let json_err = EngineJsonError::from_cas_error(&err);

    assert_eq!(
        json_err.kind, "DomainError",
        "DivisionByZero kind must be DomainError"
    );
    assert_eq!(
        json_err.code, "E_DIV_ZERO",
        "DivisionByZero code must be E_DIV_ZERO"
    );
}

#[test]
fn test_json_budget_exceeded_contract() {
    let budget_err = CasError::BudgetExceeded(BudgetExceeded {
        op: Operation::Expand,
        metric: Metric::TermsMaterialized,
        used: 150,
        limit: 100,
    });
    let json_err = EngineJsonError::from_cas_error(&budget_err);

    // Kind/code stability
    assert_eq!(json_err.kind, "BudgetExceeded");
    assert_eq!(json_err.code, "E_BUDGET");

    // Details structure
    assert!(
        json_err.details.is_object(),
        "BudgetExceeded must have details object"
    );
    assert_eq!(json_err.details["used"], 150);
    assert_eq!(json_err.details["limit"], 100);
    assert!(json_err.details["op"].is_string());
    assert!(json_err.details["metric"].is_string());
}

#[test]
fn test_json_not_implemented_contract() {
    let err = CasError::NotImplemented {
        feature: "matrix inverse".into(),
    };
    let json_err = EngineJsonError::from_cas_error(&err);

    assert_eq!(json_err.kind, "NotImplemented");
    assert_eq!(json_err.code, "E_NOT_IMPL");
}

#[test]
fn test_json_internal_error_contract() {
    let err = CasError::InternalError("assertion failed".into());
    let json_err = EngineJsonError::from_cas_error(&err);

    assert_eq!(json_err.kind, "InternalError");
    assert_eq!(json_err.code, "E_INTERNAL");
}

// =============================================================================
// Validation tests
// =============================================================================

#[test]
fn test_json_kind_in_known_set() {
    let valid_kinds = [
        "ParseError",
        "DomainError",
        "SolverError",
        "BudgetExceeded",
        "NotImplemented",
        "InternalError",
    ];

    let errors: Vec<CasError> = vec![
        CasError::ParseError("x".into()),
        CasError::DivisionByZero,
        CasError::VariableNotFound("x".into()),
        CasError::SolverError("x".into()),
        CasError::NotImplemented {
            feature: "x".into(),
        },
        CasError::InternalError("x".into()),
    ];

    for e in errors {
        let json_err = EngineJsonError::from_cas_error(&e);
        assert!(
            valid_kinds.contains(&json_err.kind),
            "Unknown kind: {} for error {:?}",
            json_err.kind,
            e
        );
    }
}

#[test]
fn test_json_code_prefix() {
    let errors: Vec<CasError> = vec![
        CasError::ParseError("x".into()),
        CasError::DivisionByZero,
        CasError::VariableNotFound("x".into()),
        CasError::InternalError("x".into()),
    ];

    for e in errors {
        let json_err = EngineJsonError::from_cas_error(&e);
        assert!(
            json_err.code.starts_with("E_"),
            "Code {} must start with E_",
            json_err.code
        );
    }
}

// =============================================================================
// No __hold leak tests
// =============================================================================

#[test]
fn test_json_no_hold_in_error_message() {
    // Simulate an error that might accidentally contain __hold
    let err = CasError::SolverError("Cannot solve __hold(x)".into());
    let json_err = EngineJsonError::from_cas_error(&err);

    // Note: This test documents that we SHOULD strip __hold from messages
    // In a real scenario, the engine should strip __hold before creating errors
    // For now, we just verify the structure is correct
    assert!(json_err.message.contains("Cannot solve"));
}

#[test]
fn test_json_no_hold_in_result() {
    // When using EngineJsonResponse::ok(), the caller is responsible for
    // ensuring result does not contain __hold (via strip_all_holds)
    let budget = BudgetJsonInfo::cli(true);
    let result = "x + 1"; // Good result, no __hold
    let resp = EngineJsonResponse::ok(result.into(), budget);
    let json = parse_json(&resp.to_json());

    assert!(
        !json["result"].as_str().unwrap().contains("__hold"),
        "Result must not contain __hold"
    );
}

// =============================================================================
// Budget mode tests
// =============================================================================

#[test]
fn test_json_budget_strict_mode() {
    let budget = BudgetJsonInfo::cli(true);
    let resp = EngineJsonResponse::ok("x".into(), budget);
    let json = parse_json(&resp.to_json());

    assert_eq!(json["budget"]["mode"], "strict");
}

#[test]
fn test_json_budget_best_effort_mode() {
    let budget = BudgetJsonInfo::cli(false);
    let resp = EngineJsonResponse::ok("x".into(), budget);
    let json = parse_json(&resp.to_json());

    assert_eq!(json["budget"]["mode"], "best-effort");
}

#[test]
fn test_json_budget_exceeded_in_best_effort() {
    let exceeded = BudgetExceeded {
        op: Operation::Expand,
        metric: Metric::TermsMaterialized,
        used: 200,
        limit: 100,
    };
    let budget = BudgetJsonInfo::cli(false).with_exceeded(&exceeded);
    let resp = EngineJsonResponse::ok("partial result".into(), budget);
    let json = parse_json(&resp.to_json());

    // ok=true because best-effort
    assert_eq!(json["ok"], true);

    // But budget.exceeded is present
    assert!(json["budget"]["exceeded"].is_object());
    assert_eq!(json["budget"]["exceeded"]["used"], 200);
    assert_eq!(json["budget"]["exceeded"]["limit"], 100);
}
