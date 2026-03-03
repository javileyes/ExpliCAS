use crate::budget::{Budget, BudgetExceeded, Metric, Operation};

#[test]
fn test_charge_under_limit() {
    let mut budget = Budget::new();
    budget.set_limit(Operation::Expand, Metric::TermsMaterialized, 100);

    assert!(budget
        .charge(Operation::Expand, Metric::TermsMaterialized, 50)
        .is_ok());
    assert_eq!(
        budget.used(Operation::Expand, Metric::TermsMaterialized),
        50
    );
}

#[test]
fn test_charge_exceeds_limit_strict() {
    let mut budget = Budget::new();
    budget.set_limit(Operation::Expand, Metric::TermsMaterialized, 100);

    let result = budget.charge(Operation::Expand, Metric::TermsMaterialized, 150);
    assert!(result.is_err());

    let err = result.expect_err("strict budget should return error");
    assert_eq!(err.op, Operation::Expand);
    assert_eq!(err.metric, Metric::TermsMaterialized);
    assert_eq!(err.used, 150);
    assert_eq!(err.limit, 100);
}

#[test]
fn test_charge_exceeds_limit_best_effort() {
    let mut budget = Budget::with_strict(false);
    budget.set_limit(Operation::Expand, Metric::TermsMaterialized, 100);

    // Should not error in best-effort mode
    assert!(budget
        .charge(Operation::Expand, Metric::TermsMaterialized, 150)
        .is_ok());
    // But usage is capped at limit
    assert_eq!(
        budget.used(Operation::Expand, Metric::TermsMaterialized),
        100
    );
}

#[test]
fn test_unlimited_when_zero() {
    let mut budget = Budget::new();
    // No limit set (0 = unlimited)

    assert!(budget
        .charge(Operation::Expand, Metric::TermsMaterialized, 1_000_000)
        .is_ok());
    assert_eq!(
        budget.used(Operation::Expand, Metric::TermsMaterialized),
        1_000_000
    );
}

#[test]
fn test_scope_tracks_operation() {
    let mut budget = Budget::new();
    budget.set_limit(Operation::Expand, Metric::TermsMaterialized, 10);

    assert_eq!(budget.current_op(), Operation::SimplifyCore);

    {
        let mut scope = budget.scope(Operation::Expand);
        assert!(scope.charge(Metric::TermsMaterialized, 7).is_ok());
    }

    // Should restore after scope drops
    assert_eq!(budget.current_op(), Operation::SimplifyCore);
    assert_eq!(budget.used(Operation::Expand, Metric::TermsMaterialized), 7);
}

#[test]
fn test_scope_charge_current() {
    let mut budget = Budget::new();
    budget.set_limit(Operation::Expand, Metric::TermsMaterialized, 100);

    {
        let mut scope = budget.scope(Operation::Expand);
        assert!(scope.charge(Metric::TermsMaterialized, 25).is_ok());
    }

    assert_eq!(
        budget.used(Operation::Expand, Metric::TermsMaterialized),
        25
    );
}

#[test]
fn test_would_exceed() {
    let mut budget = Budget::new();
    budget.set_limit(Operation::PolyOps, Metric::PolyOps, 100);
    budget
        .charge(Operation::PolyOps, Metric::PolyOps, 80)
        .expect("initial charge should pass");

    assert!(!budget.would_exceed(Operation::PolyOps, Metric::PolyOps, 10));
    assert!(budget.would_exceed(Operation::PolyOps, Metric::PolyOps, 30));
}

#[test]
fn test_reset() {
    let mut budget = Budget::new();
    budget
        .charge(Operation::Expand, Metric::TermsMaterialized, 100)
        .expect("charge should pass");
    assert_eq!(
        budget.used(Operation::Expand, Metric::TermsMaterialized),
        100
    );

    budget.reset();
    assert_eq!(budget.used(Operation::Expand, Metric::TermsMaterialized), 0);
}

#[test]
fn test_accumulative_charge() {
    let mut budget = Budget::new();
    budget.set_limit(Operation::SimplifyCore, Metric::RewriteSteps, 100);

    budget
        .charge(Operation::SimplifyCore, Metric::RewriteSteps, 30)
        .expect("charge 1 should pass");
    budget
        .charge(Operation::SimplifyCore, Metric::RewriteSteps, 30)
        .expect("charge 2 should pass");
    budget
        .charge(Operation::SimplifyCore, Metric::RewriteSteps, 30)
        .expect("charge 3 should pass");

    assert_eq!(
        budget.used(Operation::SimplifyCore, Metric::RewriteSteps),
        90
    );

    // This should exceed
    let result = budget.charge(Operation::SimplifyCore, Metric::RewriteSteps, 20);
    assert!(result.is_err());
}

#[test]
fn test_default_limits() {
    let budget = Budget::with_defaults();

    assert_eq!(
        budget.limit(Operation::SimplifyCore, Metric::RewriteSteps),
        500
    );
    assert_eq!(
        budget.limit(Operation::Expand, Metric::TermsMaterialized),
        300
    );
    assert_eq!(budget.limit(Operation::GcdZippel, Metric::PolyOps), 500);
}

#[test]
fn test_error_display() {
    let err = BudgetExceeded {
        op: Operation::Expand,
        metric: Metric::TermsMaterialized,
        used: 150,
        limit: 100,
    };

    let msg = format!("{}", err);
    assert!(msg.contains("Expand"));
    assert!(msg.contains("TermsMaterialized"));
    assert!(msg.contains("150"));
    assert!(msg.contains("100"));
}
