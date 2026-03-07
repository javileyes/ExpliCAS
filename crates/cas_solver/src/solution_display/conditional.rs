use cas_ast::{Case, SolutionSet};

/// Returns true when a conditional case is an "otherwise" containing only a
/// residual expression.
pub fn is_pure_residual_otherwise(case: &Case) -> bool {
    case.when.is_empty() && matches!(&case.then.solutions, SolutionSet::Residual(_))
}
