use cas_ast::{Context, Equation};
use cas_formatter::DisplayExpr;

pub(super) fn build_solve_timeline_title(context: &Context, original_eq: &Equation) -> String {
    format!(
        "{} {} {}",
        DisplayExpr {
            context,
            id: original_eq.lhs
        },
        original_eq.op,
        DisplayExpr {
            context,
            id: original_eq.rhs
        }
    )
}
