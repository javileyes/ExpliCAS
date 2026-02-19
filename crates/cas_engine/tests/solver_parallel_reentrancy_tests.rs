//! Parallel reentrancy tests for the solver.
//!
//! Verifies that two concurrent solves in separate threads produce
//! correct, independent diagnostics — no cross-thread condition leakage.

use std::thread;

use cas_ast::{Equation, Expr, RelOp};
use cas_engine::semantics::{AssumeScope, ValueDomain};
use cas_engine::solver::{solve_with_display_steps, SolveBudget, SolverOptions};
use cas_engine::DomainMode;
use cas_engine::Engine;
use cas_engine::ImplicitCondition;

fn make_opts() -> SolverOptions {
    SolverOptions {
        value_domain: ValueDomain::RealOnly,
        domain_mode: DomainMode::Assume,
        assume_scope: AssumeScope::Real,
        budget: SolveBudget::default(),
        ..Default::default()
    }
}

/// Two threads solve different equations concurrently.
/// Thread A: `ln(x) = 0` (may derive positive(x))
/// Thread B: `x = 1`     (trivial, no conditions)
///
/// Asserts that Thread B's diagnostics are clean — no leakage from Thread A.
#[test]
fn parallel_solves_have_independent_diagnostics() {
    let handle_a = thread::spawn(|| {
        let mut engine = Engine::new();
        let x = engine.simplifier.context.var("x");
        let ln_x = engine.simplifier.context.call("ln", vec![x]);
        let zero = engine.simplifier.context.num(0);

        let eq = Equation {
            lhs: ln_x,
            rhs: zero,
            op: RelOp::Eq,
        };

        let opts = make_opts();
        let result = solve_with_display_steps(&eq, "x", &mut engine.simplifier, opts);
        assert!(result.is_ok(), "Thread A: ln(x)=0 should solve");

        let required = result
            .as_ref()
            .map(|(_, _, d)| d.required.clone())
            .unwrap_or_default();

        // Return required conditions for cross-thread assertion
        required
    });

    let handle_b = thread::spawn(|| {
        let mut engine = Engine::new();
        let x = engine.simplifier.context.var("x");
        let one = engine.simplifier.context.num(1);

        let eq = Equation {
            lhs: x,
            rhs: one,
            op: RelOp::Eq,
        };

        let opts = make_opts();
        let result = solve_with_display_steps(&eq, "x", &mut engine.simplifier, opts);
        assert!(result.is_ok(), "Thread B: x=1 should solve");

        let required = result
            .as_ref()
            .map(|(_, _, d)| d.required.clone())
            .unwrap_or_default();

        required
    });

    let _required_a = handle_a.join().expect("Thread A panicked");
    let required_b = handle_b.join().expect("Thread B panicked");

    // Thread B (x=1) must have empty required conditions
    assert!(
        required_b.is_empty(),
        "Thread B (x=1) should have no required conditions, got: {:?}",
        required_b
    );
}

/// Multiple parallel solves of the same equation produce identical results.
/// Guards against any non-determinism from shared global state.
#[test]
fn parallel_identical_solves_produce_consistent_results() {
    let handles: Vec<_> = (0..4)
        .map(|_| {
            thread::spawn(|| {
                let mut engine = Engine::new();
                let ctx = &mut engine.simplifier.context;

                let two = ctx.num(2);
                let x = ctx.var("x");
                let y = ctx.var("y");
                let pow = ctx.add(Expr::Pow(two, x));

                let eq = Equation {
                    lhs: pow,
                    rhs: y,
                    op: RelOp::Eq,
                };

                let opts = make_opts();
                let result = solve_with_display_steps(&eq, "x", &mut engine.simplifier, opts);
                assert!(result.is_ok(), "2^x = y should solve");

                let required = result
                    .as_ref()
                    .map(|(_, _, d)| d.required.clone())
                    .unwrap_or_default();

                // Should always have positive(y)
                let has_positive_y = required
                    .iter()
                    .any(|cond| matches!(cond, ImplicitCondition::Positive(_)));

                assert!(
                    has_positive_y,
                    "Each thread should derive positive(y), got: {:?}",
                    required
                );

                required.len()
            })
        })
        .collect();

    let counts: Vec<usize> = handles
        .into_iter()
        .map(|h| h.join().expect("Thread panicked"))
        .collect();

    // All threads should produce same number of required conditions
    assert!(
        counts.windows(2).all(|w| w[0] == w[1]),
        "All threads should produce same required count, got: {:?}",
        counts
    );
}
