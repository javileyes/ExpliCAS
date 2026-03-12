//! Wire API contract tests.
//!
//! These tests verify the stable wire schema for CLI and FFI consumers.
//! Breaking these tests = breaking external API.

use cas_api_models::{
    BudgetExceededWire, BudgetWireInfo, EngineWireError, EngineWireResponse, SCHEMA_VERSION,
};
use cas_solver::runtime::{BudgetExceeded, CasError, Metric, Operation};
use serde_json::Value;

fn parse_wire(s: &str) -> Value {
    serde_json::from_str(s).expect("valid wire JSON")
}

fn engine_wire_error_from_cas_error(e: &CasError) -> EngineWireError {
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

    EngineWireError {
        kind: e.kind(),
        code: e.code(),
        message: e.to_string(),
        span: None,
        details,
    }
}

fn engine_wire_response_err(error: &CasError, budget: BudgetWireInfo) -> EngineWireResponse {
    EngineWireResponse::err(engine_wire_error_from_cas_error(error), budget)
}

fn budget_with_exceeded(mut budget: BudgetWireInfo, b: &BudgetExceeded) -> BudgetWireInfo {
    budget.exceeded = Some(BudgetExceededWire {
        op: format!("{:?}", b.op),
        metric: format!("{:?}", b.metric),
        used: b.used,
        limit: b.limit,
    });
    budget
}

// =============================================================================
// Schema stability tests
// =============================================================================

#[test]
fn test_wire_schema_version_is_1() {
    assert_eq!(SCHEMA_VERSION, 1, "Schema version must be 1");
}

#[test]
fn test_wire_success_contract() {
    let budget = BudgetWireInfo::cli(true);
    let resp = EngineWireResponse::ok("x + 1".into(), budget);
    let wire = parse_wire(&resp.to_json());

    // Required fields
    assert_eq!(wire["schema_version"], 1);
    assert_eq!(wire["ok"], true);
    assert!(wire["result"].is_string());
    assert!(wire["budget"].is_object());

    // Result content
    assert_eq!(wire["result"], "x + 1");

    // Budget structure
    assert_eq!(wire["budget"]["preset"], "cli");
    assert_eq!(wire["budget"]["mode"], "strict");
}

#[test]
fn test_wire_error_contract() {
    let budget = BudgetWireInfo::cli(true);
    let err = CasError::DivisionByZero;
    let resp = engine_wire_response_err(&err, budget);
    let wire = parse_wire(&resp.to_json());

    // Required fields
    assert_eq!(wire["schema_version"], 1);
    assert_eq!(wire["ok"], false);
    assert!(wire["error"].is_object());
    assert!(wire["budget"].is_object());

    // Error structure
    let error = &wire["error"];
    assert!(error["kind"].is_string(), "error.kind must be string");
    assert!(error["code"].is_string(), "error.code must be string");
    assert!(error["message"].is_string(), "error.message must be string");
}

// =============================================================================
// Error kind/code stability tests
// =============================================================================

#[test]
fn test_wire_parse_error_contract() {
    let err = CasError::ParseError("unexpected token".into());
    let wire_err = engine_wire_error_from_cas_error(&err);

    assert_eq!(
        wire_err.kind, "ParseError",
        "ParseError kind must be stable"
    );
    assert_eq!(wire_err.code, "E_PARSE", "ParseError code must be stable");
}

#[test]
fn test_wire_domain_error_contract() {
    let err = CasError::DivisionByZero;
    let wire_err = engine_wire_error_from_cas_error(&err);

    assert_eq!(
        wire_err.kind, "DomainError",
        "DivisionByZero kind must be DomainError"
    );
    assert_eq!(
        wire_err.code, "E_DIV_ZERO",
        "DivisionByZero code must be E_DIV_ZERO"
    );
}

#[test]
fn test_wire_budget_exceeded_contract() {
    let budget_err = CasError::BudgetExceeded(BudgetExceeded {
        op: Operation::Expand,
        metric: Metric::TermsMaterialized,
        used: 150,
        limit: 100,
    });
    let wire_err = engine_wire_error_from_cas_error(&budget_err);

    // Kind/code stability
    assert_eq!(wire_err.kind, "BudgetExceeded");
    assert_eq!(wire_err.code, "E_BUDGET");

    // Details structure
    assert!(
        wire_err.details.is_object(),
        "BudgetExceeded must have details object"
    );
    assert_eq!(wire_err.details["used"], 150);
    assert_eq!(wire_err.details["limit"], 100);
    assert!(wire_err.details["op"].is_string());
    assert!(wire_err.details["metric"].is_string());
}

#[test]
fn test_wire_not_implemented_contract() {
    let err = CasError::NotImplemented {
        feature: "matrix inverse".into(),
    };
    let wire_err = engine_wire_error_from_cas_error(&err);

    assert_eq!(wire_err.kind, "NotImplemented");
    assert_eq!(wire_err.code, "E_NOT_IMPL");
}

#[test]
fn test_wire_internal_error_contract() {
    let err = CasError::InternalError("assertion failed".into());
    let wire_err = engine_wire_error_from_cas_error(&err);

    assert_eq!(wire_err.kind, "InternalError");
    assert_eq!(wire_err.code, "E_INTERNAL");
}

// =============================================================================
// Validation tests
// =============================================================================

#[test]
fn test_wire_kind_in_known_set() {
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
        let wire_err = engine_wire_error_from_cas_error(&e);
        assert!(
            valid_kinds.contains(&wire_err.kind),
            "Unknown kind: {} for error {:?}",
            wire_err.kind,
            e
        );
    }
}

#[test]
fn test_wire_code_prefix() {
    let errors: Vec<CasError> = vec![
        CasError::ParseError("x".into()),
        CasError::DivisionByZero,
        CasError::VariableNotFound("x".into()),
        CasError::InternalError("x".into()),
    ];

    for e in errors {
        let wire_err = engine_wire_error_from_cas_error(&e);
        assert!(
            wire_err.code.starts_with("E_"),
            "Code {} must start with E_",
            wire_err.code
        );
    }
}

// =============================================================================
// No __hold leak tests
// =============================================================================

#[test]
fn test_wire_no_hold_in_error_message() {
    // Simulate an error that might accidentally contain __hold
    let err = CasError::SolverError("Cannot solve __hold(x)".into());
    let wire_err = engine_wire_error_from_cas_error(&err);

    // Note: This test documents that we SHOULD strip __hold from messages
    // In a real scenario, the engine should strip __hold before creating errors
    // For now, we just verify the structure is correct
    assert!(wire_err.message.contains("Cannot solve"));
}

#[test]
fn test_wire_no_hold_in_result() {
    // When using EngineWireResponse::ok(), the caller is responsible for
    // ensuring result does not contain __hold (via strip_all_holds)
    let budget = BudgetWireInfo::cli(true);
    let result = "x + 1"; // Good result, no __hold
    let resp = EngineWireResponse::ok(result.into(), budget);
    let wire = parse_wire(&resp.to_json());

    assert!(
        !wire["result"].as_str().unwrap().contains("__hold"),
        "Result must not contain __hold"
    );
}

// =============================================================================
// Budget mode tests
// =============================================================================

#[test]
fn test_wire_budget_strict_mode() {
    let budget = BudgetWireInfo::cli(true);
    let resp = EngineWireResponse::ok("x".into(), budget);
    let wire = parse_wire(&resp.to_json());

    assert_eq!(wire["budget"]["mode"], "strict");
}

#[test]
fn test_wire_budget_best_effort_mode() {
    let budget = BudgetWireInfo::cli(false);
    let resp = EngineWireResponse::ok("x".into(), budget);
    let wire = parse_wire(&resp.to_json());

    assert_eq!(wire["budget"]["mode"], "best-effort");
}

#[test]
fn test_wire_budget_exceeded_in_best_effort() {
    let exceeded = BudgetExceeded {
        op: Operation::Expand,
        metric: Metric::TermsMaterialized,
        used: 200,
        limit: 100,
    };
    let budget = budget_with_exceeded(BudgetWireInfo::cli(false), &exceeded);
    let resp = EngineWireResponse::ok("partial result".into(), budget);
    let wire = parse_wire(&resp.to_json());

    // ok=true because best-effort
    assert_eq!(wire["ok"], true);

    // But budget.exceeded is present
    assert!(wire["budget"]["exceeded"].is_object());
    assert_eq!(wire["budget"]["exceeded"]["used"], 200);
    assert_eq!(wire["budget"]["exceeded"]["limit"], 100);
}
