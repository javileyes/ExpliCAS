//! Shared structural support for polynomial expansion rules.

use crate::expr_relations::count_additive_terms;
use crate::multinomial_expand::{
    multinomial_term_count, try_expand_multinomial_direct, MultinomialExpandBudget,
};
use crate::multipoly::PolyBudget;
use crate::poly_convert::try_multipoly_from_expr_with_var_limit;
use crate::{
    auto_expand_budget_support::{
        count_add_terms_for_pow_base, count_distinct_variables_in_expr,
        estimate_multinomial_terms_for_pow,
    },
    auto_expand_scan::looks_polynomial_like,
    expand_ops::build_binomial_power_expansion,
    expr_destructure::{as_add, as_pow},
};
use cas_ast::{Context, Expr, ExprId};
use num_traits::{Signed, ToPrimitive};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmallMultinomialPolicy {
    pub max_exp: u32,
    pub max_terms: usize,
    pub max_output_terms: usize,
    pub max_base_nodes: usize,
    pub max_output_nodes: usize,
    pub max_vars: usize,
}

impl Default for SmallMultinomialPolicy {
    fn default() -> Self {
        Self {
            max_exp: 4,
            max_terms: 6,
            max_output_terms: 35,
            max_base_nodes: 25,
            max_output_nodes: 350,
            max_vars: 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmallMultinomialExpansionPlan {
    pub expanded: ExprId,
    pub term_count: usize,
    pub exponent: u32,
}

/// Try to expand small multinomials in default simplification mode.
///
/// Applies bounded guards before and after expansion to avoid blow-ups.
pub fn try_expand_small_multinomial_expr(
    ctx: &mut Context,
    expr: ExprId,
    policy: SmallMultinomialPolicy,
) -> Option<SmallMultinomialExpansionPlan> {
    let (base, exp) = match ctx.get(expr) {
        Expr::Pow(b, e) => (*b, *e),
        _ => return None,
    };

    let n = match ctx.get(exp) {
        Expr::Number(num) => {
            if !num.is_integer() || num.is_negative() {
                return None;
            }
            num.to_integer().to_u32()?
        }
        _ => return None,
    };
    if !(2..=policy.max_exp).contains(&n) {
        return None;
    }

    let k = count_additive_terms(ctx, base);
    if !(3..=policy.max_terms).contains(&k) {
        return None;
    }

    let pred_terms = multinomial_term_count(n, k, policy.max_output_terms + 1)?;
    if pred_terms > policy.max_output_terms {
        return None;
    }

    let base_nodes = cas_ast::count_all_nodes(ctx, base);
    if base_nodes > policy.max_base_nodes {
        return None;
    }

    let budget = MultinomialExpandBudget {
        max_exp: policy.max_exp,
        max_base_terms: policy.max_terms,
        max_vars: policy.max_vars,
        max_output_terms: policy.max_output_terms,
    };
    let expanded = try_expand_multinomial_direct(ctx, base, exp, &budget)?;

    let output_nodes = cas_ast::count_all_nodes(ctx, expanded);
    if output_nodes > policy.max_output_nodes {
        return None;
    }

    Some(SmallMultinomialExpansionPlan {
        expanded,
        term_count: k,
        exponent: n,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoSubCancelPolicy {
    pub max_terms: usize,
    pub max_total_degree: u32,
    pub max_pow_exp: u32,
    pub max_vars: usize,
}

impl Default for AutoSubCancelPolicy {
    fn default() -> Self {
        Self {
            max_terms: 100,
            max_total_degree: 8,
            max_pow_exp: 4,
            max_vars: 4,
        }
    }
}

/// Returns true when Add/Sub expression is polynomially provable as zero
/// under bounded multi-polynomial conversion.
pub fn is_auto_sub_cancel_zero(ctx: &Context, expr: ExprId, policy: AutoSubCancelPolicy) -> bool {
    if !matches!(ctx.get(expr), Expr::Sub(_, _) | Expr::Add(_, _)) {
        return false;
    }

    let budget = PolyBudget {
        max_terms: policy.max_terms,
        max_total_degree: policy.max_total_degree,
        max_pow_exp: policy.max_pow_exp,
    };

    try_multipoly_from_expr_with_var_limit(ctx, expr, &budget, policy.max_vars)
        .is_some_and(|poly| poly.is_zero())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoExpandPowSumPolicy {
    pub max_pow_exp: u32,
    pub max_base_terms: u32,
    pub max_generated_terms: u32,
    pub max_vars: u32,
}

impl Default for AutoExpandPowSumPolicy {
    fn default() -> Self {
        Self {
            max_pow_exp: 6,
            max_base_terms: 4,
            max_generated_terms: 300,
            max_vars: 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoExpandPowSumPlan {
    pub expanded: ExprId,
    pub num_terms: u32,
    pub exponent: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinomialExpansionPlan {
    pub expanded: ExprId,
    pub exponent: u32,
}

/// Expand `(a ± b)^n` under bounded exponent limits.
///
/// Returns `None` when input is not a true binomial power or exponent is out
/// of range.
pub fn try_expand_binomial_pow_expr(
    ctx: &mut Context,
    expr: ExprId,
    min_exp: u32,
    max_exp: u32,
) -> Option<BinomialExpansionPlan> {
    let (base, exp) = as_pow(ctx, expr)?;

    let n_val = {
        let exp_expr = ctx.get(exp);
        match exp_expr {
            Expr::Number(n) if n.is_integer() && !n.is_negative() => n.to_integer().to_u32()?,
            _ => return None,
        }
    };

    if !(min_exp..=max_exp).contains(&n_val) {
        return None;
    }

    if count_additive_terms(ctx, base) != 2 {
        return None;
    }

    let (a, b) = match ctx.get(base) {
        Expr::Add(a, b) => (*a, *b),
        Expr::Sub(a, b) => {
            let b = *b;
            let a = *a;
            let neg_b = ctx.add(Expr::Neg(b));
            (a, neg_b)
        }
        _ => return None,
    };

    let expanded = build_binomial_power_expansion(ctx, a, b, n_val);
    Some(BinomialExpansionPlan {
        expanded,
        exponent: n_val,
    })
}

/// Plan and materialize a bounded auto-expansion for `Pow(Add(..), n)`.
///
/// Returns `None` when shape/budget checks fail or expansion is a no-op.
pub fn try_auto_expand_pow_sum_expr(
    ctx: &mut Context,
    expr: ExprId,
    policy: AutoExpandPowSumPolicy,
) -> Option<AutoExpandPowSumPlan> {
    let (base, exp) = as_pow(ctx, expr)?;

    let n_val = {
        let exp_expr = ctx.get(exp);
        match exp_expr {
            Expr::Number(n) if n.is_integer() && !n.is_negative() => n.to_integer().to_u32()?,
            _ => return None,
        }
    };

    if n_val < 2 || n_val > policy.max_pow_exp {
        return None;
    }

    let (a, b) = as_add(ctx, base)?;

    let num_terms = count_add_terms_for_pow_base(ctx, base);
    if num_terms > policy.max_base_terms {
        return None;
    }

    let estimated_result_terms = estimate_multinomial_terms_for_pow(num_terms, n_val)?;
    if estimated_result_terms > policy.max_generated_terms {
        return None;
    }

    if count_distinct_variables_in_expr(ctx, base) > policy.max_vars {
        return None;
    }

    let expanded = if num_terms == 2 {
        build_binomial_power_expansion(ctx, a, b, n_val)
    } else {
        crate::expand_ops::expand(ctx, expr)
    };

    if expanded == expr {
        return None;
    }

    Some(AutoExpandPowSumPlan {
        expanded,
        num_terms,
        exponent: n_val,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmallPowExpandPolicy {
    pub max_exp: u32,
    pub max_base_terms: usize,
    pub max_vars: usize,
    pub max_output_terms: usize,
}

impl Default for SmallPowExpandPolicy {
    fn default() -> Self {
        Self {
            max_exp: 6,
            max_base_terms: 3,
            max_vars: 2,
            max_output_terms: 20,
        }
    }
}

/// Expand `Pow(sum, n)` under strict multinomial budget caps.
pub fn try_expand_small_pow_sum_expr(
    ctx: &mut Context,
    expr: ExprId,
    policy: SmallPowExpandPolicy,
) -> Option<ExprId> {
    let (base, exp) = as_pow(ctx, expr)?;
    let budget = MultinomialExpandBudget {
        max_exp: policy.max_exp,
        max_base_terms: policy.max_base_terms,
        max_vars: policy.max_vars,
        max_output_terms: policy.max_output_terms,
    };
    try_expand_multinomial_direct(ctx, base, exp, &budget)
}

/// Returns true when expression tree contains `Pow(Add(..), n)` where
/// `n` is in `[min_exp, max_exp]` and base looks polynomial-like.
pub fn contains_small_polynomial_pow_sum_candidate(
    ctx: &Context,
    expr: ExprId,
    min_exp: u32,
    max_exp: u32,
    max_depth: usize,
) -> bool {
    fn walk(
        ctx: &Context,
        expr: ExprId,
        min_exp: u32,
        max_exp: u32,
        depth: usize,
        max_depth: usize,
    ) -> bool {
        if depth > max_depth {
            return false;
        }
        match ctx.get(expr) {
            Expr::Pow(base, exp) => {
                if matches!(ctx.get(*base), Expr::Add(_, _)) {
                    if let Expr::Number(n) = ctx.get(*exp) {
                        if n.is_integer() && !n.is_negative() {
                            if let Some(e) = n.to_integer().to_u32() {
                                if (min_exp..=max_exp).contains(&e)
                                    && looks_polynomial_like(ctx, *base)
                                {
                                    return true;
                                }
                            }
                        }
                    }
                }
                walk(ctx, *base, min_exp, max_exp, depth + 1, max_depth)
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                walk(ctx, *l, min_exp, max_exp, depth + 1, max_depth)
                    || walk(ctx, *r, min_exp, max_exp, depth + 1, max_depth)
            }
            Expr::Neg(inner) | Expr::Hold(inner) => {
                walk(ctx, *inner, min_exp, max_exp, depth + 1, max_depth)
            }
            Expr::Function(_, args) => args
                .iter()
                .any(|a| walk(ctx, *a, min_exp, max_exp, depth + 1, max_depth)),
            Expr::Matrix { data, .. } => data
                .iter()
                .any(|e| walk(ctx, *e, min_exp, max_exp, depth + 1, max_depth)),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => false,
        }
    }

    walk(ctx, expr, min_exp, max_exp, 0, max_depth)
}

#[derive(Debug, Clone)]
pub struct HeuristicPolyNormalizePolicy {
    pub pow_min_exp: u32,
    pub pow_max_exp: u32,
    pub max_scan_depth: usize,
    pub max_nodes: usize,
    pub max_terms: usize,
    pub max_vars: usize,
    pub poly_budget: PolyBudget,
    pub conversion_var_limit: usize,
}

impl Default for HeuristicPolyNormalizePolicy {
    fn default() -> Self {
        Self {
            pow_min_exp: 2,
            pow_max_exp: 6,
            max_scan_depth: 20,
            max_nodes: 80,
            max_terms: 30,
            max_vars: 3,
            poly_budget: PolyBudget {
                max_terms: 40,
                max_total_degree: 6,
                max_pow_exp: 6,
            },
            conversion_var_limit: 4,
        }
    }
}

/// Normalize Add/Sub polynomial expressions heuristically by converting to
/// bounded MultiPoly and rebuilding a flattened expression.
pub fn try_heuristic_poly_normalize_add_expr(
    ctx: &mut Context,
    expr: ExprId,
    policy: HeuristicPolyNormalizePolicy,
) -> Option<ExprId> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    if !contains_small_polynomial_pow_sum_candidate(
        ctx,
        expr,
        policy.pow_min_exp,
        policy.pow_max_exp,
        policy.max_scan_depth,
    ) {
        return None;
    }

    if cas_ast::count_nodes(ctx, expr) > policy.max_nodes {
        return None;
    }

    let poly = try_multipoly_from_expr_with_var_limit(
        ctx,
        expr,
        &policy.poly_budget,
        policy.conversion_var_limit,
    )?;

    if poly.terms.len() > policy.max_terms || poly.vars.len() > policy.max_vars || poly.is_zero() {
        return None;
    }

    let new_expr = crate::multipoly::multipoly_to_expr(&poly, ctx);
    if new_expr == expr {
        return None;
    }
    Some(new_expr)
}

#[cfg(test)]
mod tests {
    use super::{
        contains_small_polynomial_pow_sum_candidate, is_auto_sub_cancel_zero,
        try_auto_expand_pow_sum_expr, try_expand_binomial_pow_expr,
        try_expand_small_multinomial_expr, try_expand_small_pow_sum_expr,
        try_heuristic_poly_normalize_add_expr, AutoExpandPowSumPlan, AutoExpandPowSumPolicy,
        AutoSubCancelPolicy, BinomialExpansionPlan, HeuristicPolyNormalizePolicy,
        SmallMultinomialPolicy, SmallPowExpandPolicy,
    };
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn small_multinomial_expands_trinomial_square() {
        let mut ctx = Context::new();
        let expr = parse("(x+y+1)^2", &mut ctx).expect("parse");
        let plan =
            try_expand_small_multinomial_expr(&mut ctx, expr, SmallMultinomialPolicy::default());
        let plan = plan.expect("plan");
        assert_ne!(plan.expanded, expr);
        assert_eq!(plan.term_count, 3);
        assert_eq!(plan.exponent, 2);
    }

    #[test]
    fn small_multinomial_rejects_non_pow() {
        let mut ctx = Context::new();
        let expr = parse("x+y+1", &mut ctx).expect("parse");
        let plan =
            try_expand_small_multinomial_expr(&mut ctx, expr, SmallMultinomialPolicy::default());
        assert!(plan.is_none());
    }

    #[test]
    fn auto_sub_cancel_detects_zero_polynomial() {
        let mut ctx = Context::new();
        let expr = parse("(x+1)^2 - (x^2 + 2*x + 1)", &mut ctx).expect("parse");
        assert!(is_auto_sub_cancel_zero(
            &ctx,
            expr,
            AutoSubCancelPolicy::default()
        ));
    }

    #[test]
    fn auto_sub_cancel_rejects_non_zero_polynomial() {
        let mut ctx = Context::new();
        let expr = parse("(x+1)^2 - x^2", &mut ctx).expect("parse");
        assert!(!is_auto_sub_cancel_zero(
            &ctx,
            expr,
            AutoSubCancelPolicy::default()
        ));
    }

    fn plan(
        mut ctx: Context,
        input: &str,
        policy: AutoExpandPowSumPolicy,
    ) -> Option<AutoExpandPowSumPlan> {
        let expr = parse(input, &mut ctx).expect("parse");
        try_auto_expand_pow_sum_expr(&mut ctx, expr, policy)
    }

    #[test]
    fn auto_expand_pow_sum_expands_binomial_within_budget() {
        let plan =
            plan(Context::new(), "(x+1)^3", AutoExpandPowSumPolicy::default()).expect("plan");
        assert_eq!(plan.num_terms, 2);
        assert_eq!(plan.exponent, 3);
    }

    #[test]
    fn auto_expand_pow_sum_rejects_exponent_over_budget() {
        let plan = plan(
            Context::new(),
            "(x+1)^7",
            AutoExpandPowSumPolicy {
                max_pow_exp: 6,
                ..AutoExpandPowSumPolicy::default()
            },
        );
        assert!(plan.is_none());
    }

    fn binomial_plan(mut ctx: Context, input: &str) -> Option<BinomialExpansionPlan> {
        let expr = parse(input, &mut ctx).expect("parse");
        try_expand_binomial_pow_expr(&mut ctx, expr, 2, 20)
    }

    #[test]
    fn try_expand_binomial_pow_expr_accepts_add_and_sub() {
        let add_plan = binomial_plan(Context::new(), "(x+1)^4").expect("add plan");
        assert_eq!(add_plan.exponent, 4);

        let sub_plan = binomial_plan(Context::new(), "(x-1)^3").expect("sub plan");
        assert_eq!(sub_plan.exponent, 3);
    }

    #[test]
    fn try_expand_binomial_pow_expr_rejects_non_binomial_shape() {
        let plan = binomial_plan(Context::new(), "(x+y+1)^2");
        assert!(plan.is_none());
    }

    #[test]
    fn contains_small_polynomial_pow_sum_candidate_detects_binomial_power() {
        let mut ctx = Context::new();
        let expr = parse("(x+1)^2 + y", &mut ctx).expect("parse");
        assert!(contains_small_polynomial_pow_sum_candidate(
            &ctx, expr, 2, 6, 20
        ));
    }

    #[test]
    fn try_expand_small_pow_sum_expr_expands_with_default_policy() {
        let mut ctx = Context::new();
        let expr = parse("(x+1)^2", &mut ctx).expect("parse");
        let expanded =
            try_expand_small_pow_sum_expr(&mut ctx, expr, SmallPowExpandPolicy::default());
        assert!(expanded.is_some());
        assert_ne!(expanded.expect("expanded"), expr);
    }

    #[test]
    fn heuristic_poly_normalize_rewrites_sum_with_binomial_power() {
        let mut ctx = Context::new();
        let expr = parse("(x+1)^2 + x", &mut ctx).expect("parse");
        let rewritten = try_heuristic_poly_normalize_add_expr(
            &mut ctx,
            expr,
            HeuristicPolyNormalizePolicy::default(),
        );
        assert!(rewritten.is_some());
        assert_ne!(rewritten.expect("rewritten"), expr);
    }
}
