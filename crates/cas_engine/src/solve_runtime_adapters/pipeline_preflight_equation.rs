use crate::Simplifier;
use cas_ast::Equation;
use cas_solver_core::solve_analysis::PreparedEquationResidual;

pub(crate) fn prepare_equation_for_strategy(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
) -> PreparedEquationResidual {
    cas_solver_core::solve_runtime_pipeline_preflight_equation_bound_runtime::prepare_equation_for_strategy_with_adapter_state_and_default_structural_recompose(
        simplifier, equation, var,
    )
}
