use crate::budget::{BudgetExceeded, Metric, Operation};
use crate::error::CasError;

#[test]
fn test_budget_exceeded_conversion() {
    let budget_err = BudgetExceeded {
        op: Operation::Expand,
        metric: Metric::TermsMaterialized,
        used: 150,
        limit: 100,
    };

    let cas_err: CasError = budget_err.clone().into();

    // Verify it's wrapped correctly
    match cas_err {
        CasError::BudgetExceeded(inner) => {
            assert_eq!(inner.op, Operation::Expand);
            assert_eq!(inner.metric, Metric::TermsMaterialized);
            assert_eq!(inner.used, 150);
            assert_eq!(inner.limit, 100);
        }
        _ => panic!("Expected BudgetExceeded variant"),
    }
}

#[test]
fn test_budget_exceeded_display() {
    let budget_err = BudgetExceeded {
        op: Operation::GcdZippel,
        metric: Metric::PolyOps,
        used: 500,
        limit: 300,
    };

    let cas_err: CasError = budget_err.into();
    let msg = format!("{}", cas_err);

    assert!(msg.contains("GcdZippel"), "Missing op in: {}", msg);
    assert!(msg.contains("PolyOps"), "Missing metric in: {}", msg);
    assert!(msg.contains("500"), "Missing used in: {}", msg);
    assert!(msg.contains("300"), "Missing limit in: {}", msg);
}

#[test]
fn test_budget_exceeded_helper() {
    let err = CasError::budget_exceeded(Operation::SimplifyCore, Metric::RewriteSteps);

    match err {
        CasError::BudgetExceeded(inner) => {
            assert_eq!(inner.op, Operation::SimplifyCore);
            assert_eq!(inner.metric, Metric::RewriteSteps);
        }
        _ => panic!("Expected BudgetExceeded variant"),
    }
}

#[test]
fn test_poly_error_to_cas_error() {
    use cas_math::multipoly::PolyError;

    let poly_err = PolyError::BudgetExceeded;
    let cas_err: CasError = poly_err.into();

    match cas_err {
        CasError::BudgetExceeded(inner) => {
            assert_eq!(inner.op, Operation::PolyOps);
        }
        _ => panic!("Expected BudgetExceeded, got: {:?}", cas_err),
    }
}

// =========================================================================
// Stability tests for kind/code API (Error API Stability Contract)
// =========================================================================

#[test]
fn test_error_kind_stable() {
    // These kind values are STABLE and must not change
    assert_eq!(CasError::ParseError("x".into()).kind(), "ParseError");
    assert_eq!(CasError::DivisionByZero.kind(), "DomainError");
    assert_eq!(CasError::VariableNotFound("x".into()).kind(), "DomainError");
    assert_eq!(CasError::SolverError("x".into()).kind(), "SolverError");
    assert_eq!(
        CasError::NotImplemented {
            feature: "x".into()
        }
        .kind(),
        "NotImplemented"
    );
    assert_eq!(CasError::InternalError("x".into()).kind(), "InternalError");

    let budget_err = CasError::BudgetExceeded(BudgetExceeded {
        op: Operation::Expand,
        metric: Metric::TermsMaterialized,
        used: 0,
        limit: 0,
    });
    assert_eq!(budget_err.kind(), "BudgetExceeded");
}

#[test]
fn test_error_code_stable() {
    // These code values are STABLE and must not change
    assert_eq!(CasError::ParseError("x".into()).code(), "E_PARSE");
    assert_eq!(CasError::DivisionByZero.code(), "E_DIV_ZERO");
    assert_eq!(
        CasError::VariableNotFound("x".into()).code(),
        "E_VAR_NOT_FOUND"
    );
    assert_eq!(CasError::SolverError("x".into()).code(), "E_SOLVER");
    assert_eq!(
        CasError::NotImplemented {
            feature: "x".into()
        }
        .code(),
        "E_NOT_IMPL"
    );
    assert_eq!(CasError::InternalError("x".into()).code(), "E_INTERNAL");

    let budget_err = CasError::BudgetExceeded(BudgetExceeded {
        op: Operation::Expand,
        metric: Metric::TermsMaterialized,
        used: 0,
        limit: 0,
    });
    assert_eq!(budget_err.code(), "E_BUDGET");
}

#[test]
fn test_error_code_prefix() {
    // All codes must start with E_
    let errors: Vec<CasError> = vec![
        CasError::ParseError("x".into()),
        CasError::DivisionByZero,
        CasError::VariableNotFound("x".into()),
        CasError::InternalError("x".into()),
    ];

    for e in errors {
        assert!(
            e.code().starts_with("E_"),
            "Code {} must start with E_",
            e.code()
        );
    }
}

#[test]
fn test_error_kind_known_set() {
    // kind() must return one of the known values
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
        CasError::InternalError("x".into()),
    ];

    for e in errors {
        assert!(
            valid_kinds.contains(&e.kind()),
            "Unknown kind: {}",
            e.kind()
        );
    }
}

#[test]
fn test_budget_details_accessor() {
    let budget_err = CasError::BudgetExceeded(BudgetExceeded {
        op: Operation::Expand,
        metric: Metric::TermsMaterialized,
        used: 100,
        limit: 50,
    });

    let details = budget_err
        .budget_details()
        .expect("BudgetExceeded should expose details");
    assert_eq!(details.used, 100);
    assert_eq!(details.limit, 50);

    // Non-budget errors should return None
    assert!(CasError::DivisionByZero.budget_details().is_none());
}

#[test]
fn test_parse_error_kind_and_code() {
    let cas_err = CasError::ParseError("unexpected token".to_string());
    assert_eq!(cas_err.kind(), "ParseError");
    assert_eq!(cas_err.code(), "E_PARSE");
}
