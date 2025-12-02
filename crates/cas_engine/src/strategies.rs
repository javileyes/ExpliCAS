use cas_ast::{ExprId, Context, DisplayExpr};
use crate::expand::expand;
use crate::factor::factor;
use crate::collect::collect;
use crate::rule::Rule;
use crate::step::{Step, PathStep};
use crate::rules::polynomial::CombineLikeTermsRule;
use crate::rules::arithmetic::CombineConstantsRule;
use crate::rules::canonicalization::{CanonicalizeNegationRule, CanonicalizeMulRule};
use crate::rules::exponents::{ProductPowerRule, IdentityPowerRule, EvaluatePowerRule};
use crate::rules::algebra::SimplifyFractionRule;

/// Strategy to simplify polynomials by trying expansion and factorization.
/// Returns the simplest form found.
pub fn simplify_polynomial(ctx: &mut Context, expr: ExprId) -> (ExprId, Vec<Step>) {
    let mut steps = Vec::new();

    // 1. Expand
    let expanded = expand(ctx, expr);
    if expanded != expr {
        steps.push(Step::new(
            "Expand Polynomial",
            "Expand",
            expr,
            expanded,
            Vec::new(),
        ));
    }
    
    // 2. Clean up expansion (Collect + Combine Like Terms + Power Rules)
    let mut current = expanded;
    let mut i = 0;
    while i < 10 {
        let prev = current;
        
        // Collect first
        let collected = collect(ctx, current);
        if collected != current {
             // We could add a step here, but collect is often implicit.
             // Let's add it if it changes things significantly?
             // For consistency with orchestrator, let's skip explicit collect steps here 
             // unless we want very detailed traces.
             // But since we are inside "Polynomial Strategy", maybe we want details?
             // Let's skip for now to avoid noise, as apply_rules_to_tree will show the main simplifications.
             current = collected;
        }
        
        // Apply rules recursively
        current = apply_rules_to_tree(ctx, current, &mut steps, Vec::new());
        
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
    
    let mut chosen = expr; // Default to original

    // 1. Prefer 0
    if s_exp == "0" { 
        chosen = simplified_expanded;
    } else if s_fact == "0" { 
        chosen = factored;
    } else {
        // 2. Prefer Factored if it's a Product or Power and Expanded is Sum
        let fact_data = ctx.get(factored);
        let exp_data = ctx.get(simplified_expanded);
        
        let is_fact_structured = matches!(fact_data, cas_ast::Expr::Mul(_, _) | cas_ast::Expr::Pow(_, _));
        let is_exp_sum = matches!(exp_data, cas_ast::Expr::Add(_, _) | cas_ast::Expr::Sub(_, _));
        
        if is_fact_structured && is_exp_sum {
            chosen = factored;
        } else {
            // 3. Length heuristic as fallback
            let len_orig = s_orig.len();
            let len_exp = s_exp.len();
            let len_fact = s_fact.len();

            // Prefer factored if significantly shorter or same length
            if len_fact <= len_exp && len_fact <= len_orig {
                chosen = factored;
            } else if len_exp < len_orig {
                chosen = simplified_expanded;
            }
        }
    }

    if chosen == factored {
        if factored != simplified_expanded {
            steps.push(Step::new(
                "Factor Polynomial",
                "Factor",
                simplified_expanded,
                factored,
                Vec::new(),
            ));
        }
        return (factored, steps);
    } else if chosen == simplified_expanded {
        return (simplified_expanded, steps);
    } else {
        // Reverted to original
        return (expr, Vec::new());
    }
}

fn apply_rules_to_tree(ctx: &mut Context, expr: ExprId, steps: &mut Vec<Step>, path: Vec<PathStep>) -> ExprId {
    use cas_ast::Expr;
    
    // 1. Recurse on children
    let expr_data = ctx.get(expr).clone();
    let mut new_expr = match expr_data {
        Expr::Add(l, r) => {
            let mut p_l = path.clone(); p_l.push(PathStep::Left);
            let nl = apply_rules_to_tree(ctx, l, steps, p_l);
            
            let mut p_r = path.clone(); p_r.push(PathStep::Right);
            let nr = apply_rules_to_tree(ctx, r, steps, p_r);
            
            if nl != l || nr != r { ctx.add(Expr::Add(nl, nr)) } else { expr }
        },
        Expr::Sub(l, r) => {
            let mut p_l = path.clone(); p_l.push(PathStep::Left);
            let nl = apply_rules_to_tree(ctx, l, steps, p_l);
            
            let mut p_r = path.clone(); p_r.push(PathStep::Right);
            let nr = apply_rules_to_tree(ctx, r, steps, p_r);
            
            if nl != l || nr != r { ctx.add(Expr::Sub(nl, nr)) } else { expr }
        },
        Expr::Mul(l, r) => {
            let mut p_l = path.clone(); p_l.push(PathStep::Left);
            let nl = apply_rules_to_tree(ctx, l, steps, p_l);
            
            let mut p_r = path.clone(); p_r.push(PathStep::Right);
            let nr = apply_rules_to_tree(ctx, r, steps, p_r);
            
            if nl != l || nr != r { ctx.add(Expr::Mul(nl, nr)) } else { expr }
        },
        Expr::Div(l, r) => {
            let mut p_l = path.clone(); p_l.push(PathStep::Left);
            let nl = apply_rules_to_tree(ctx, l, steps, p_l);
            
            let mut p_r = path.clone(); p_r.push(PathStep::Right);
            let nr = apply_rules_to_tree(ctx, r, steps, p_r);
            
            if nl != l || nr != r { ctx.add(Expr::Div(nl, nr)) } else { expr }
        },
        Expr::Pow(b, e) => {
            let mut p_b = path.clone(); p_b.push(PathStep::Base);
            let nb = apply_rules_to_tree(ctx, b, steps, p_b);
            
            let mut p_e = path.clone(); p_e.push(PathStep::Exponent);
            let ne = apply_rules_to_tree(ctx, e, steps, p_e);
            
            if nb != b || ne != e { ctx.add(Expr::Pow(nb, ne)) } else { expr }
        },
        Expr::Neg(e) => {
            let mut p_e = path.clone(); p_e.push(PathStep::Inner);
            let ne = apply_rules_to_tree(ctx, e, steps, p_e);
            
            if ne != e { ctx.add(Expr::Neg(ne)) } else { expr }
        },
        Expr::Function(name, args) => {
            let mut new_args = Vec::new();
            let mut changed = false;
            for (i, arg) in args.iter().enumerate() {
                let mut p_arg = path.clone(); p_arg.push(PathStep::Arg(i));
                let new_arg = apply_rules_to_tree(ctx, *arg, steps, p_arg);
                if new_arg != *arg { changed = true; }
                new_args.push(new_arg);
            }
            if changed { ctx.add(Expr::Function(name, new_args)) } else { expr }
        },
        _ => expr
    };

    // 2. Apply rules to self
    let mut changed = true;
    while changed {
        changed = false;
        
        let mut apply_rule = |rule: &dyn Rule, current_expr: ExprId| -> Option<ExprId> {
            if let Some(rw) = rule.apply(ctx, current_expr) {
                steps.push(Step::new(
                    &rw.description,
                    rule.name(),
                    current_expr,
                    rw.new_expr,
                    path.clone(),
                ));
                Some(rw.new_expr)
            } else {
                None
            }
        };

        // Canonicalize Negation first (important for combining)
        if let Some(res) = apply_rule(&CanonicalizeNegationRule, new_expr) {
            new_expr = res;
            changed = true;
        }

        // Canonicalize Multiplication (sorts factors for ProductPowerRule)
        if let Some(res) = apply_rule(&CanonicalizeMulRule, new_expr) {
            new_expr = res;
            changed = true;
        }

        // Power Rules
        if let Some(res) = apply_rule(&ProductPowerRule, new_expr) {
            new_expr = res;
            changed = true;
        }
        if let Some(res) = apply_rule(&IdentityPowerRule, new_expr) {
            new_expr = res;
            changed = true;
        }
        if let Some(res) = apply_rule(&EvaluatePowerRule, new_expr) {
            new_expr = res;
            changed = true;
        }

        if let Some(res) = apply_rule(&CombineLikeTermsRule, new_expr) {
            new_expr = res;
            changed = true;
        }
        
        if let Some(res) = apply_rule(&CombineConstantsRule, new_expr) {
            new_expr = res;
            changed = true;
        }

        if let Some(res) = apply_rule(&SimplifyFractionRule, new_expr) {
            new_expr = res;
            changed = true;
        }
    }
    new_expr
}
