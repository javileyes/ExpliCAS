//! V2.0 Phase 2D: Tests for SolutionSet::flatten()

use cas_ast::{Case, ConditionPredicate, ConditionSet, Context, SolutionSet, SolveResult};

/// Helper to create a simple discrete solution
fn discrete_one(ctx: &mut Context) -> SolutionSet {
    let one = ctx.num(1);
    SolutionSet::Discrete(vec![one])
}

/// Helper to create a simple discrete solution with value 2
fn discrete_two(ctx: &mut Context) -> SolutionSet {
    let two = ctx.num(2);
    SolutionSet::Discrete(vec![two])
}

// =============================================================================
// Test 1: Flatten single nested Conditional
// =============================================================================
#[test]
fn flatten_single_nested_conditional() {
    let mut ctx = Context::new();
    let a = ctx.var("a");
    let b = ctx.var("b");

    // Inner Conditional: if b=0 -> {2} else -> AllReals
    let inner_cases = vec![
        Case::with_result(
            ConditionSet::single(ConditionPredicate::EqZero(b)),
            SolveResult::solved(discrete_two(&mut ctx)),
        ),
        Case::with_result(
            ConditionSet::empty(), // otherwise
            SolveResult::solved(SolutionSet::AllReals),
        ),
    ];
    let inner_conditional = SolutionSet::Conditional(inner_cases);

    // Outer Conditional: if a=1 -> (inner) else -> {1}
    let outer_cases = vec![
        Case::with_result(
            ConditionSet::single(ConditionPredicate::EqOne(a)),
            SolveResult::solved(inner_conditional),
        ),
        Case::with_result(
            ConditionSet::empty(), // otherwise
            SolveResult::solved(discrete_one(&mut ctx)),
        ),
    ];
    let outer = SolutionSet::Conditional(outer_cases);

    // Before flatten: 2 outer cases
    if let SolutionSet::Conditional(cases) = &outer {
        assert_eq!(cases.len(), 2, "Before flatten: 2 outer cases");
    }

    // Flatten
    let flattened = outer.flatten();

    // After flatten: 4 cases (a=1∧b=0, a=1∧otherwise, otherwise from outer)
    // But actually: inner has 2 cases, outer has 2 cases
    // Result: (a=1 expanded to 2 inner cases) + (otherwise = {1})
    // = 3 total cases (a=1∧b=0, a=1∧otherwise, otherwise)
    if let SolutionSet::Conditional(cases) = &flattened {
        assert_eq!(cases.len(), 3, "After flatten: 3 cases expected");

        // First case should have combined guards (a=1 AND b=0)
        let first = &cases[0];
        assert!(
            !first.when.is_otherwise(),
            "First case should not be otherwise"
        );
        assert!(
            !first.when.predicates().is_empty(),
            "First case should have predicates"
        );
    } else {
        panic!("Expected Conditional after flatten");
    }
}

// =============================================================================
// Test 2: Flatten is idempotent
// =============================================================================
#[test]
fn flatten_is_idempotent() {
    let mut ctx = Context::new();
    let a = ctx.var("a");
    let b = ctx.var("b");

    // Create nested structure
    let inner_cases = vec![Case::with_result(
        ConditionSet::single(ConditionPredicate::EqZero(b)),
        SolveResult::solved(SolutionSet::AllReals),
    )];
    let inner = SolutionSet::Conditional(inner_cases);

    let outer_cases = vec![Case::with_result(
        ConditionSet::single(ConditionPredicate::EqOne(a)),
        SolveResult::solved(inner),
    )];
    let outer = SolutionSet::Conditional(outer_cases);

    // Flatten once
    let flat1 = outer.flatten();
    // Flatten twice
    let flat2 = flat1.clone().flatten();

    // Should be identical
    assert_eq!(flat1, flat2, "flatten() should be idempotent");
}

// =============================================================================
// Test 3: Flatten preserves otherwise last
// =============================================================================
#[test]
fn flatten_preserves_otherwise_last() {
    let mut ctx = Context::new();
    let a = ctx.var("a");

    // Create: otherwise first, then condition
    let cases = vec![
        Case::with_result(
            ConditionSet::empty(), // otherwise
            SolveResult::solved(discrete_one(&mut ctx)),
        ),
        Case::with_result(
            ConditionSet::single(ConditionPredicate::EqOne(a)),
            SolveResult::solved(SolutionSet::AllReals),
        ),
    ];
    let solution = SolutionSet::Conditional(cases);

    let flattened = solution.flatten();

    // After flatten, otherwise should be last
    if let SolutionSet::Conditional(cases) = flattened {
        assert_eq!(cases.len(), 2);
        // Last case should be otherwise
        assert!(
            cases.last().unwrap().when.is_otherwise(),
            "otherwise should be last after flatten"
        );
        // First case should NOT be otherwise
        assert!(
            !cases.first().unwrap().when.is_otherwise(),
            "First case should not be otherwise"
        );
    } else {
        panic!("Expected Conditional");
    }
}
