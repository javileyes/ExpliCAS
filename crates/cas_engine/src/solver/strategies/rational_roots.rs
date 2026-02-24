//! RationalRootsStrategy — solves polynomial equations of degree ≥ 3
//! with all-numeric (rational) coefficients using the Rational Root Theorem
//! plus synthetic division (deflation).
//!
//! Pipeline:
//! 1. Extract univariate polynomial coefficients from `simplify(lhs - rhs)`
//! 2. Normalize to integer coefficients (scale by LCM of denominators)
//! 3. Enumerate candidate rational roots ±p/q
//! 4. Verify each candidate via exact Horner evaluation
//! 5. Deflate by confirmed roots (synthetic division)
//! 6. Delegate residual polynomial (degree ≤ 2) to existing strategies

use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{medium_step, SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, ExprId, SolutionSet};
use cas_solver_core::rational_roots::{
    solve_rational_roots_strategy_with_runtime_and_item, RationalRootsExecutionItem,
    RationalRootsStrategyRuntime,
};

/// Maximum number of candidate rational roots to try before bailing.
/// Prevents combinatorial blowup on polynomials with large leading/constant coefficients.
const MAX_CANDIDATES: usize = 200;

/// Maximum polynomial degree we attempt.
const MAX_DEGREE: usize = 10;

pub struct RationalRootsStrategy;

struct EngineRationalRootsRuntime<'a> {
    simplifier: &'a mut Simplifier,
}

impl RationalRootsStrategyRuntime for EngineRationalRootsRuntime<'_> {
    fn context(&mut self) -> &mut cas_ast::Context {
        &mut self.simplifier.context
    }

    fn simplify_expr(&mut self, expr: ExprId) -> ExprId {
        let (simplified, _) = self.simplifier.simplify(expr);
        simplified
    }

    fn expand_expr(&mut self, expr: ExprId) -> ExprId {
        crate::expand::expand(&mut self.simplifier.context, expr)
    }
}

impl SolverStrategy for RationalRootsStrategy {
    fn name(&self) -> &str {
        "Rational Roots"
    }

    fn apply(
        &self,
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        _opts: &SolverOptions,
        _ctx: &SolveCtx,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        let include_item = simplifier.collect_steps();
        let mut runtime = EngineRationalRootsRuntime { simplifier };
        let solved = solve_rational_roots_strategy_with_runtime_and_item(
            &mut runtime,
            eq.lhs,
            eq.rhs,
            eq.op.clone(),
            var,
            3,
            MAX_DEGREE,
            MAX_CANDIDATES,
            include_item,
            |item: RationalRootsExecutionItem| {
                medium_step(item.description().to_string(), item.equation)
            },
        )?;
        Some(Ok((solved.solution_set, solved.steps)))
    }

    fn should_verify(&self) -> bool {
        true // Verify roots against original equation
    }
}
