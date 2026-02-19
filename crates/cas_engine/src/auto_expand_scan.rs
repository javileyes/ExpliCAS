use crate::pattern_marks::PatternMarks;
use crate::phase::ExpandBudget;
use cas_ast::{Context, ExprId};

pub use cas_math::auto_expand_scan::looks_polynomial_like;

fn to_math_budget(budget: &ExpandBudget) -> cas_math::auto_expand_scan::ExpandBudget {
    cas_math::auto_expand_scan::ExpandBudget {
        max_pow_exp: budget.max_pow_exp,
        max_base_terms: budget.max_base_terms,
        max_generated_terms: budget.max_generated_terms,
        max_vars: budget.max_vars,
    }
}

pub fn mark_auto_expand_candidates(
    ctx: &Context,
    root: ExprId,
    budget: &ExpandBudget,
    marks: &mut PatternMarks,
) {
    let math_budget = to_math_budget(budget);
    cas_math::auto_expand_scan::mark_auto_expand_candidates(ctx, root, &math_budget, marks);
}
