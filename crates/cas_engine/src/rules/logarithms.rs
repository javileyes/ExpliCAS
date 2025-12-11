use crate::rule::Rewrite;
use crate::define_rule;
use cas_ast::{Expr, ExprId, Context};
use num_traits::{Zero, One};
use crate::ordering::compare_expr;
use std::cmp::Ordering;

define_rule!(
    EvaluateLogRule,
    "Evaluate Logarithms",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Function(name, args) = expr_data {
            // Handle ln(x) as log(e, x)
            let (base, arg) = if name == "ln" && args.len() == 1 {
                let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                (e, args[0])
            } else if name == "log" && args.len() == 2 {
                (args[0], args[1])
            } else {
                return None;
            };

            let arg_data = ctx.get(arg).clone();

                // 1. log(b, 1) = 0, log(b, 0) = -infinity, log(b, neg) = undefined
                if let Expr::Number(n) = &arg_data {
                    if n.is_one() {
                        let zero = ctx.num(0);
                        return Some(Rewrite {
                            new_expr: zero,
                            description: "log(b, 1) = 0".to_string(),
                before_local: None,
                after_local: None,
            });
                    }
                    if n.is_zero() {
                        let inf = ctx.add(Expr::Constant(cas_ast::Constant::Infinity));
                        let neg_inf = ctx.add(Expr::Neg(inf));
                        return Some(Rewrite {
                            new_expr: neg_inf,
                            description: "log(b, 0) = -infinity".to_string(),
                before_local: None,
                after_local: None,
            });
                    }
                    if *n < num_rational::BigRational::zero() {
                         let undef = ctx.add(Expr::Constant(cas_ast::Constant::Undefined));
                         return Some(Rewrite {
                            new_expr: undef,
                            description: "log(b, neg) = undefined".to_string(),
                before_local: None,
                after_local: None,
            });
                    }
                    
                    // Check if n is a power of base (if base is a number)
                    let base_data = ctx.get(base).clone();
                    if let Expr::Number(b) = base_data {
                        // Simple check for integer powers for now
                        if b.is_integer() && n.is_integer() {
                            let b_int = b.to_integer();
                            let n_int = n.to_integer();
                            if b_int > num_bigint::BigInt::from(1) {
                                let mut temp = b_int.clone();
                                let mut power = 1;
                                while temp < n_int {
                                    temp *= &b_int;
                                    power += 1;
                                }
                                if temp == n_int {
                                    let new_expr = ctx.num(power);
                                    return Some(Rewrite {
                                        new_expr,
                                        description: format!("log({}, {}) = {}", b, n, power),
                before_local: None,
                after_local: None,
            });
                                }
                            }
                        }
                    }
                }

                // 2. log(b, b) = 1
                if base == arg || ctx.get(base) == ctx.get(arg) {
                    let one = ctx.num(1);
                    return Some(Rewrite {
                        new_expr: one,
                        description: "log(b, b) = 1".to_string(),
                before_local: None,
                after_local: None,
            });
                }

                // 3. log(b, b^x) = x
                if let Expr::Pow(p_base, p_exp) = &arg_data {
                    if *p_base == base || ctx.get(*p_base) == ctx.get(base) {
                        return Some(Rewrite {
                            new_expr: *p_exp,
                            description: "log(b, b^x) = x".to_string(),
                before_local: None,
                after_local: None,
            });
                    }
                }

                // 4. Expansion: log(b, x^y) = y * log(b, x)
                // Note: This overlaps with rule 3 if x == b. Rule 3 is more specific/simpler, so it should match first.
                // This rule is good for canonicalization.
                if let Expr::Pow(p_base, p_exp) = arg_data {
                    let log_inner = ctx.add(Expr::Function("log".to_string(), vec![base, p_base]));
                    let new_expr = ctx.add(Expr::Mul(p_exp, log_inner));
                    return Some(Rewrite {
                        new_expr,
                        description: "log(b, x^y) = y * log(b, x)".to_string(),
                before_local: None,
                after_local: None,
            });
                }

                // 5. Product: log(b, x*y) = log(b, x) + log(b, y)
                if let Expr::Mul(lhs, rhs) = arg_data {
                    let log_lhs = ctx.add(Expr::Function("log".to_string(), vec![base, lhs]));
                    let log_rhs = ctx.add(Expr::Function("log".to_string(), vec![base, rhs]));
                    let new_expr = ctx.add(Expr::Add(log_lhs, log_rhs));
                    return Some(Rewrite {
                        new_expr,
                        description: "log(b, x*y) = log(b, x) + log(b, y)".to_string(),
                before_local: None,
                after_local: None,
            });
                }

                // 6. Quotient: log(b, x/y) = log(b, x) - log(b, y)
                if let Expr::Div(num, den) = arg_data {
                    let log_num = ctx.add(Expr::Function("log".to_string(), vec![base, num]));
                    let log_den = ctx.add(Expr::Function("log".to_string(), vec![base, den]));
                    let new_expr = ctx.add(Expr::Sub(log_num, log_den));
                    return Some(Rewrite {
                        new_expr,
                        description: "log(b, x/y) = log(b, x) - log(b, y)".to_string(),
                before_local: None,
                after_local: None,
            });
                }
                // 7. log(b, |x|) -> log(b, x)
                // We assume x is in the domain of log (x > 0) if this simplification is requested,
                // similar to how we handle log(b, x^y) -> y*log(b, x).
                if let Expr::Function(fname, fargs) = &arg_data {
                    if fname == "abs" && fargs.len() == 1 {
                        let abs_arg = fargs[0];
                        let new_expr = ctx.add(Expr::Function("log".to_string(), vec![base, abs_arg]));
                        return Some(Rewrite {
                            new_expr,
                            description: "log(b, |x|) -> log(b, x)".to_string(),
                before_local: None,
                after_local: None,
            });
                    }
                }
            }
        None
    }
);

define_rule!(
    ExponentialLogRule,
    "Exponential-Log Inverse",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Pow(base, exp) = expr_data {
            let exp_data = ctx.get(exp).clone();
            
            // Helper to get log base and arg
            let get_log_parts = |ctx: &mut Context, e_id: ExprId| -> Option<(ExprId, ExprId)> {
                let e_data = ctx.get(e_id).clone();
                if let Expr::Function(name, args) = e_data {
                    if name == "log" && args.len() == 2 {
                        return Some((args[0], args[1]));
                    } else if name == "ln" && args.len() == 1 {
                        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                        return Some((e, args[0]));
                    }
                }
                None
            };

            // Case 1: b^log(b, x) -> x
            if let Some((log_base, log_arg)) = get_log_parts(ctx, exp) {
                 if compare_expr(ctx, log_base, base) == Ordering::Equal {
                    return Some(Rewrite {
                        new_expr: log_arg,
                        description: "b^log(b, x) = x".to_string(),
                before_local: None,
                after_local: None,
            });
                }
            }
            
            // Case 2: b^(c * log(b, x)) -> x^c
            if let Expr::Mul(lhs, rhs) = &exp_data {
                // Check if lhs or rhs is log(b, x)
                let mut check_log = |target: ExprId, coeff: ExprId| -> Option<Rewrite> {
                    if let Some((log_base, log_arg)) = get_log_parts(ctx, target) {
                        if compare_expr(ctx, log_base, base) == Ordering::Equal {
                            // Found log(b, x). Result is x^coeff
                            let new_expr = ctx.add(Expr::Pow(log_arg, coeff));
                            return Some(Rewrite {
                                new_expr,
                                description: "b^(c*log(b, x)) = x^c".to_string(),
                before_local: None,
                after_local: None,
            });
                        }
                    }
                    None
                };
                
                if let Some(rw) = check_log(*lhs, *rhs) { return Some(rw); }
                if let Some(rw) = check_log(*rhs, *lhs) { return Some(rw); }
            }
        }
        None
    }
);


define_rule!(
    SplitLogExponentsRule,
    "Split Log Exponents",
    |ctx, expr| {
        // e^(a + b) -> e^a * e^b IF a or b is a log
        let expr_data = ctx.get(expr).clone();
        if let Expr::Pow(base, exp) = expr_data {
            let base_is_e = matches!(ctx.get(base), Expr::Constant(cas_ast::Constant::E));
            if base_is_e {
                let exp_data = ctx.get(exp).clone();
                if let Expr::Add(lhs, rhs) = exp_data {
                    let lhs_is_log = is_log(ctx, lhs);
                    let rhs_is_log = is_log(ctx, rhs);
                    
                    if lhs_is_log || rhs_is_log {
                        let term1 = simplify_exp_log(ctx, base, lhs);
                        let term2 = simplify_exp_log(ctx, base, rhs);
                        let new_expr = ctx.add(Expr::Mul(term1, term2));
                        return Some(Rewrite {
                            new_expr,
                            description: "e^(a+b) -> e^a * e^b (log cancellation)".to_string(),
                before_local: None,
                after_local: None,
            });
                    }
                }
            }
        }
        None
    }
);

fn simplify_exp_log(context: &mut Context, base: ExprId, exp: ExprId) -> ExprId {
    // Check if exp is log(base, x)
    if let Expr::Function(name, args) = context.get(exp) {
        if name == "log" && args.len() == 2 {
            let log_base = args[0];
            let log_arg = args[1];
            if log_base == base {
                return log_arg;
            }
        }
    }
    // Also check n*log(base, x) -> x^n?
    // Maybe later. For now just direct cancellation.
    context.add(Expr::Pow(base, exp))
}

fn is_log(context: &Context, expr: ExprId) -> bool {
    if let Expr::Function(name, _) = context.get(expr) {
        return name == "log" || name == "ln";
    }
    // Also check for n*log(x)
    if let Expr::Mul(l, r) = context.get(expr) {
        return is_log(context, *l) || is_log(context, *r);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_parser::parse;
    use cas_ast::{DisplayExpr, Context};

    #[test]
    fn test_log_one() {
        let mut ctx = Context::new();
        let rule = EvaluateLogRule;
        // log(x, 1) -> 0
        let expr = parse("log(x, 1)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr, &crate::parent_context::ParentContext::root()).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "0");
    }

    #[test]
    fn test_log_base_base() {
        let mut ctx = Context::new();
        let rule = EvaluateLogRule;
        // log(x, x) -> 1
        let expr = parse("log(x, x)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr, &crate::parent_context::ParentContext::root()).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "1");
    }

    #[test]
    fn test_log_inverse() {
        let mut ctx = Context::new();
        let rule = EvaluateLogRule;
        // log(x, x^2) -> 2
        let expr = parse("log(x, x^2)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr, &crate::parent_context::ParentContext::root()).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "2");
    }

    #[test]
    fn test_log_expansion() {
        let mut ctx = Context::new();
        let rule = EvaluateLogRule;
        // log(b, x^y) -> y * log(b, x)
        let expr = parse("log(2, x^3)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr, &crate::parent_context::ParentContext::root()).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "3 * log(2, x)");
    }

    #[test]
    fn test_log_product() {
        let mut ctx = Context::new();
        let rule = EvaluateLogRule;
        // log(b, x*y) -> log(b, x) + log(b, y)
        let expr = parse("log(2, x * y)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr, &crate::parent_context::ParentContext::root()).unwrap();
        let res = format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr });
        assert!(res.contains("log(2, x)"));
        assert!(res.contains("log(2, y)"));
        assert!(res.contains("+"));
    }

    #[test]
    fn test_log_quotient() {
        let mut ctx = Context::new();
        let rule = EvaluateLogRule;
        // log(b, x/y) -> log(b, x) - log(b, y)
        let expr = parse("log(2, x / y)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr, &crate::parent_context::ParentContext::root()).unwrap();
        let res = format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr });
        assert!(res.contains("log(2, x)"));
        assert!(res.contains("log(2, y)"));
        assert!(res.contains("-"));
    }

    #[test]
    fn test_ln_e() {
        let mut ctx = Context::new();
        let rule = EvaluateLogRule;
        // ln(e) -> 1
        // Note: ln(e) parses to log(e, e)
        let expr = parse("ln(e)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr, &crate::parent_context::ParentContext::root()).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "1");
    }
}



define_rule!(
    LogInversePowerRule,
    "Log Inverse Power",
    |ctx, expr| {
        // println!("LogInversePowerRule checking {:?}", expr);
        let expr_data = ctx.get(expr).clone();
        if let Expr::Pow(base, exp) = expr_data {
            // Check for x^(c / log(b, x))
            // exp could be Div(c, log(b, x)) or Mul(c, Pow(log(b, x), -1))
            
            // Returns Some(Some(base)) for log(b, x), Some(None) for ln(x) -> base e
            let check_log_denom = |ctx: &Context, denom: cas_ast::ExprId| -> Option<Option<cas_ast::ExprId>> {
                // println!("check_log_denom checking {:?}", denom);
                if let Expr::Function(name, args) = ctx.get(denom) {
                    // Debug: checking log denominator
                    if name == "log" && args.len() == 2 {
                        let log_base = args[0];
                        let log_arg = args[1];
                        // Check if log_arg == base
                        if compare_expr(ctx, log_arg, base) == Ordering::Equal {
                            // Debug: found matching log
                            return Some(Some(log_base));
                        }
                    } else if name == "ln" && args.len() == 1 {
                        let log_arg = args[0];
                        if compare_expr(ctx, log_arg, base) == Ordering::Equal {
                            // Debug: found matching ln
                            return Some(None); // Base e
                        }
                    }
                }
                None
            };

            let mut target_b_opt: Option<Option<cas_ast::ExprId>> = None;
            let mut coeff: Option<cas_ast::ExprId> = None;

            let exp_data = ctx.get(exp).clone();
            match exp_data {
                Expr::Div(num, den) => {
                    if let Some(b_opt) = check_log_denom(ctx, den) {
                        target_b_opt = Some(b_opt);
                        coeff = Some(num);
                    }
                },
                Expr::Mul(l, r) => {
                    // Check l * r^-1
                    if let Expr::Pow(b, e) = ctx.get(r) {
                         if let Expr::Number(n) = ctx.get(*e) {
                            if n.is_integer() && *n == num_rational::BigRational::from_integer((-1).into()) {
                                if let Some(b_opt) = check_log_denom(ctx, *b) {
                                    target_b_opt = Some(b_opt);
                                    coeff = Some(l);
                                }
                            }
                        }
                    }
                    // Check r * l^-1
                    if target_b_opt.is_none() {
                        if let Expr::Pow(b, e) = ctx.get(l) {
                             if let Expr::Number(n) = ctx.get(*e) {
                                if n.is_integer() && *n == num_rational::BigRational::from_integer((-1).into()) {
                                    if let Some(b_opt) = check_log_denom(ctx, *b) {
                                        target_b_opt = Some(b_opt);
                                        coeff = Some(r);
                                    }
                                }
                            }
                        }
                    }
                },
                Expr::Pow(b, e) => {
                    // Check if it's log(b, x)^-1
                    if let Expr::Number(n) = ctx.get(e) {
                        if n.is_integer() && *n == num_rational::BigRational::from_integer((-1).into()) {
                            if let Some(b_opt) = check_log_denom(ctx, b) {
                                target_b_opt = Some(b_opt);
                                coeff = Some(ctx.num(1));
                            }
                        }
                    }
                },
                _ => {}
            }

            if let (Some(b_opt), Some(c)) = (target_b_opt, coeff) {
                // Result is b^c
                let b = b_opt.unwrap_or_else(|| ctx.add(Expr::Constant(cas_ast::Constant::E)));
                // Debug: applying log inverse power rule
                let new_expr = ctx.add(Expr::Pow(b, c));
                return Some(Rewrite {
                    new_expr,
                    description: "x^(c/log(b, x)) = b^c".to_string(),
                before_local: None,
                after_local: None,
            });
            }
        }
        None
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(EvaluateLogRule));
    simplifier.add_rule(Box::new(ExponentialLogRule));
    simplifier.add_rule(Box::new(SplitLogExponentsRule));
    simplifier.add_rule(Box::new(LogInversePowerRule));
}
