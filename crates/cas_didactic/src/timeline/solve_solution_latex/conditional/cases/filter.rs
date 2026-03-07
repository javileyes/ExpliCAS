use cas_ast::{Case, SolutionSet};

pub(super) fn skip_conditional_case(case: &Case) -> bool {
    case.when.is_otherwise() && matches!(&case.then.solutions, SolutionSet::Residual(_))
}
