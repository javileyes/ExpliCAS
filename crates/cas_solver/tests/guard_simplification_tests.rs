//! V2.0 Phase 3: Tests for guard simplification

use cas_ast::{Case, ConditionPredicate, ConditionSet, Context, SolutionSet, SolveResult};

// =============================================================================
// Phase 3A: Simplify redundant predicates
// =============================================================================

#[test]
fn simplify_positive_implies_nonzero() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    // {Positive(x), NonZero(x)} should simplify to {Positive(x)}
    let set = ConditionSet::from_predicates(vec![
        ConditionPredicate::Positive(x),
        ConditionPredicate::NonZero(x),
    ]);

    let simplified = set.simplify();

    assert_eq!(
        simplified.predicates().len(),
        1,
        "Should remove redundant NonZero"
    );
    assert!(
        matches!(simplified.predicates()[0], ConditionPredicate::Positive(_)),
        "Should keep Positive"
    );
}

#[test]
fn simplify_eqone_implies_nonzero() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    // {EqOne(x), NonZero(x)} should simplify to {EqOne(x)}
    let set = ConditionSet::from_predicates(vec![
        ConditionPredicate::EqOne(x),
        ConditionPredicate::NonZero(x),
    ]);

    let simplified = set.simplify();

    assert_eq!(
        simplified.predicates().len(),
        1,
        "Should remove redundant NonZero"
    );
    assert!(
        matches!(simplified.predicates()[0], ConditionPredicate::EqOne(_)),
        "Should keep EqOne"
    );
}

// =============================================================================
// Phase 3B: Contradiction detection
// =============================================================================

#[test]
fn is_contradiction_eqzero_nonzero() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let set = ConditionSet::from_predicates(vec![
        ConditionPredicate::EqZero(x),
        ConditionPredicate::NonZero(x),
    ]);

    assert!(set.is_contradiction(), "EqZero AND NonZero is impossible");
}

#[test]
fn is_contradiction_eqzero_positive() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let set = ConditionSet::from_predicates(vec![
        ConditionPredicate::EqZero(x),
        ConditionPredicate::Positive(x),
    ]);

    assert!(set.is_contradiction(), "EqZero AND Positive is impossible");
}

#[test]
fn is_contradiction_eqzero_eqone() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let set = ConditionSet::from_predicates(vec![
        ConditionPredicate::EqZero(x),
        ConditionPredicate::EqOne(x),
    ]);

    assert!(set.is_contradiction(), "EqZero AND EqOne is impossible");
}

#[test]
fn not_contradiction_valid_set() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let set = ConditionSet::from_predicates(vec![
        ConditionPredicate::Positive(x),
        ConditionPredicate::NonZero(x),
    ]);

    assert!(
        !set.is_contradiction(),
        "Positive AND NonZero is valid (redundant but not contradictory)"
    );
}

// =============================================================================
// Phase 3C: Simplify_cases filters contradictions
// =============================================================================

#[test]
fn simplify_cases_removes_contradictions() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);

    // Create a case with contradictory guard
    let contradiction_guard = ConditionSet::from_predicates(vec![
        ConditionPredicate::EqZero(x),
        ConditionPredicate::Positive(x),
    ]);

    let cases = vec![
        Case::with_result(
            contradiction_guard,
            SolveResult::solved(SolutionSet::AllReals),
        ),
        Case::with_result(
            ConditionSet::empty(), // otherwise
            SolveResult::solved(SolutionSet::Discrete(vec![one])),
        ),
    ];

    let conditional = SolutionSet::Conditional(cases);
    let simplified = conditional.simplify();

    // The contradictory case should be removed
    // Only "otherwise" remains, so it unwraps to just the solution
    match simplified {
        SolutionSet::Discrete(sols) => {
            assert_eq!(sols.len(), 1, "Should unwrap to single solution");
        }
        SolutionSet::Conditional(cases) => {
            assert_eq!(
                cases.len(),
                1,
                "Should only have otherwise case after filtering"
            );
        }
        other => panic!("Unexpected result: {:?}", other),
    }
}
