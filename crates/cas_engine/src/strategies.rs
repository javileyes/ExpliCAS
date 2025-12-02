use cas_ast::{ExprId, Context, DisplayExpr};
use crate::expand::expand;
use crate::factor::factor;
use crate::collect::collect;
use crate::rule::Rule;
use crate::rules::polynomial::CombineLikeTermsRule;
use crate::rules::arithmetic::CombineConstantsRule;
use crate::rules::canonicalization::{CanonicalizeNegationRule, CanonicalizeMulRule};
use crate::rules::exponents::{ProductPowerRule, IdentityPowerRule, EvaluatePowerRule};
use crate::rules::algebra::SimplifyFractionRule;

/// Strategy to simplify polynomials by trying expansion and factorization.
/// Returns the simplest form found.
pub fn simplify_polynomial(ctx: &mut Context, expr: ExprId) -> ExprId {
    // 1. Expand
    let expanded = expand(ctx, expr);
    
    // 2. Clean up expansion (Collect + Combine Like Terms + Power Rules)
    let mut current = expanded;
    let mut i = 0;
    while i < 10 {
        let prev = current;
        
        // Collect first
        current = collect(ctx, current);
        
        // Apply rules recursively
        current = apply_rules_to_tree(ctx, current);
        
        if current == prev {
            break;
        }
        i += 1;
    }
    let simplified_expanded = current;

    // 3. Factor the result
    let factored = factor(ctx, simplified_expanded);
    
    // 4. Compare and choose best
    // Heuristic: Prefer 0, then prefer factored form if it has structure (Mul/Pow), then shortest.
    
    let s_orig = format!("{}", DisplayExpr { context: ctx, id: expr });
    let s_exp = format!("{}", DisplayExpr { context: ctx, id: simplified_expanded });
    let s_fact = format!("{}", DisplayExpr { context: ctx, id: factored });
    
    // 1. Prefer 0
    if s_exp == "0" { return simplified_expanded; }
    if s_fact == "0" { return factored; }

    // 2. Prefer Factored if it's a Product or Power and Expanded is Sum
    // This prevents undoing factorization like (x-3)(x+3) -> x^2-9
    let fact_data = ctx.get(factored);
    let exp_data = ctx.get(simplified_expanded);
    
    let is_fact_structured = matches!(fact_data, cas_ast::Expr::Mul(_, _) | cas_ast::Expr::Pow(_, _));
    let is_exp_sum = matches!(exp_data, cas_ast::Expr::Add(_, _) | cas_ast::Expr::Sub(_, _));
    
    if is_fact_structured && is_exp_sum {
        // If factored is structured and expanded is a sum, prefer factored
        // UNLESS expanded is significantly simpler (e.g. much shorter)
        // (x-3)(x+3) len 11 vs x^2-9 len 5. 
        // We want (x-3)(x+3).
        return factored;
    }

    // 3. Length heuristic as fallback
    let len_orig = s_orig.len();
    let len_exp = s_exp.len();
    let len_fact = s_fact.len();

    // Prefer factored if significantly shorter or same length
    if len_fact <= len_exp && len_fact <= len_orig {
        return factored;
    }
    
    // Prefer expanded if shorter than original
    if len_exp < len_orig {
        return simplified_expanded;
    }
    
    // Default to original if nothing is better
    expr
}

fn apply_rules_to_tree(ctx: &mut Context, expr: ExprId) -> ExprId {
    use cas_ast::Expr;
    
    // 1. Recurse on children
    let expr_data = ctx.get(expr).clone();
    let mut new_expr = match expr_data {
        Expr::Add(l, r) => {
            let nl = apply_rules_to_tree(ctx, l);
            let nr = apply_rules_to_tree(ctx, r);
            if nl != l || nr != r { ctx.add(Expr::Add(nl, nr)) } else { expr }
        },
        Expr::Sub(l, r) => {
            let nl = apply_rules_to_tree(ctx, l);
            let nr = apply_rules_to_tree(ctx, r);
            if nl != l || nr != r { ctx.add(Expr::Sub(nl, nr)) } else { expr }
        },
        Expr::Mul(l, r) => {
            let nl = apply_rules_to_tree(ctx, l);
            let nr = apply_rules_to_tree(ctx, r);
            if nl != l || nr != r { ctx.add(Expr::Mul(nl, nr)) } else { expr }
        },
        Expr::Div(l, r) => {
            let nl = apply_rules_to_tree(ctx, l);
            let nr = apply_rules_to_tree(ctx, r);
            if nl != l || nr != r { ctx.add(Expr::Div(nl, nr)) } else { expr }
        },
        Expr::Pow(b, e) => {
            let nb = apply_rules_to_tree(ctx, b);
            let ne = apply_rules_to_tree(ctx, e);
            if nb != b || ne != e { ctx.add(Expr::Pow(nb, ne)) } else { expr }
        },
        Expr::Neg(e) => {
            let ne = apply_rules_to_tree(ctx, e);
            if ne != e { ctx.add(Expr::Neg(ne)) } else { expr }
        },
        Expr::Function(name, args) => {
            let new_args: Vec<ExprId> = args.iter().map(|a| apply_rules_to_tree(ctx, *a)).collect();
            if new_args != args { ctx.add(Expr::Function(name, new_args)) } else { expr }
        },
        _ => expr
    };

    // 2. Apply rules to self
    let mut changed = true;
    while changed {
        changed = false;
        
        // Canonicalize Negation first (important for combining)
        if let Some(rw) = CanonicalizeNegationRule.apply(ctx, new_expr) {
            new_expr = rw.new_expr;
            changed = true;
        }

        // Canonicalize Multiplication (sorts factors for ProductPowerRule)
        if let Some(rw) = CanonicalizeMulRule.apply(ctx, new_expr) {
            new_expr = rw.new_expr;
            changed = true;
        }

        // Power Rules
        if let Some(rw) = ProductPowerRule.apply(ctx, new_expr) {
            new_expr = rw.new_expr;
            changed = true;
        }
        if let Some(rw) = IdentityPowerRule.apply(ctx, new_expr) {
            new_expr = rw.new_expr;
            changed = true;
        }
        if let Some(rw) = EvaluatePowerRule.apply(ctx, new_expr) {
            new_expr = rw.new_expr;
            changed = true;
        }

        if let Some(rw) = CombineLikeTermsRule.apply(ctx, new_expr) {
            new_expr = rw.new_expr;
            changed = true;
        }
        
        if let Some(rw) = CombineConstantsRule.apply(ctx, new_expr) {
            new_expr = rw.new_expr;
            changed = true;
        }

        if let Some(rw) = SimplifyFractionRule.apply(ctx, new_expr) {
            new_expr = rw.new_expr;
            changed = true;
        }
    }
    new_expr
}
