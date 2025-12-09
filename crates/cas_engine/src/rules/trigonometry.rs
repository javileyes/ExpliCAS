use crate::define_rule;
use crate::helpers::{extract_double_angle_arg, is_pi, is_pi_over_n};
use crate::rule::Rewrite;
use cas_ast::{Expr, ExprId};
use num_traits::{One, Zero};

use std::cmp::Ordering;

define_rule!(
    EvaluateTrigRule,
    "Evaluate Trigonometric Functions",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Function(name, args) = expr_data {
            if args.len() == 1 {
                let arg = args[0];

                // Case 1: Known Values (0)
                if let Expr::Number(n) = ctx.get(arg) {
                    if n.is_zero() {
                        match name.as_str() {
                            "sin" | "tan" | "arcsin" | "arctan" => {
                                let zero = ctx.num(0);
                                return Some(Rewrite {
                                    new_expr: zero,
                                    description: format!("{}(0) = 0", name),
                                });
                            }
                            "cos" => {
                                let one = ctx.num(1);
                                return Some(Rewrite {
                                    new_expr: one,
                                    description: "cos(0) = 1".to_string(),
                                });
                            }
                            "arccos" => {
                                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                let two = ctx.num(2);
                                let new_expr = ctx.add(Expr::Div(pi, two));
                                return Some(Rewrite {
                                    new_expr,
                                    description: "arccos(0) = pi/2".to_string(),
                                });
                            }
                            _ => {}
                        }
                    } else if n.is_one() {
                        match name.as_str() {
                            "arcsin" => {
                                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                let two = ctx.num(2);
                                let new_expr = ctx.add(Expr::Div(pi, two));
                                return Some(Rewrite {
                                    new_expr,
                                    description: "arcsin(1) = pi/2".to_string(),
                                });
                            }
                            "arccos" => {
                                let zero = ctx.num(0);
                                return Some(Rewrite {
                                    new_expr: zero,
                                    description: "arccos(1) = 0".to_string(),
                                });
                            }
                            "arctan" => {
                                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                let four = ctx.num(4);
                                let new_expr = ctx.add(Expr::Div(pi, four));
                                return Some(Rewrite {
                                    new_expr,
                                    description: "arctan(1) = pi/4".to_string(),
                                });
                            }
                            _ => {}
                        }
                    } else if *n == num_rational::BigRational::new(1.into(), 2.into()) {
                        // 1/2
                        match name.as_str() {
                            "arcsin" => {
                                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                let six = ctx.num(6);
                                let new_expr = ctx.add(Expr::Div(pi, six));
                                return Some(Rewrite {
                                    new_expr,
                                    description: "arcsin(1/2) = pi/6".to_string(),
                                });
                            }
                            "arccos" => {
                                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                let three = ctx.num(3);
                                let new_expr = ctx.add(Expr::Div(pi, three));
                                return Some(Rewrite {
                                    new_expr,
                                    description: "arccos(1/2) = pi/3".to_string(),
                                });
                            }
                            _ => {}
                        }
                    }
                }

                // Case 2: Known Values (pi) - using shared helper
                if is_pi(ctx, arg) {
                    match name.as_str() {
                        "sin" | "tan" => {
                            let zero = ctx.num(0);
                            return Some(Rewrite {
                                new_expr: zero,
                                description: format!("{}(pi) = 0", name),
                            });
                        }
                        "cos" => {
                            let neg_one = ctx.num(-1);
                            return Some(Rewrite {
                                new_expr: neg_one,
                                description: "cos(pi) = -1".to_string(),
                            });
                        }
                        _ => {}
                    }
                }

                // Case 3: Known Values (pi/2) - using shared helper for both Div and Mul formats
                if is_pi_over_n(ctx, arg, 2) {
                    match name.as_str() {
                        "sin" => {
                            let one = ctx.num(1);
                            return Some(Rewrite {
                                new_expr: one,
                                description: "sin(pi/2) = 1".to_string(),
                            });
                        }
                        "cos" => {
                            let zero = ctx.num(0);
                            return Some(Rewrite {
                                new_expr: zero,
                                description: "cos(pi/2) = 0".to_string(),
                            });
                        }
                        "tan" => {
                            let undefined = ctx.add(Expr::Constant(cas_ast::Constant::Undefined));
                            return Some(Rewrite {
                                new_expr: undefined,
                                description: "tan(pi/2) = undefined".to_string(),
                            });
                        }
                        _ => {}
                    }
                }

                // Case 4: Known Values (pi/3) - sin(π/3) = √3/2, cos(π/3) = 1/2, tan(π/3) = √3
                if is_pi_over_n(ctx, arg, 3) {
                    match name.as_str() {
                        "sin" => {
                            // sin(π/3) = √3/2
                            let three = ctx.num(3);
                            let one = ctx.num(1);
                            let two = ctx.num(2);
                            let half_exp = ctx.add(Expr::Div(one, two));
                            let sqrt3 = ctx.add(Expr::Pow(three, half_exp));
                            let two2 = ctx.num(2);
                            let new_expr = ctx.add(Expr::Div(sqrt3, two2));
                            return Some(Rewrite {
                                new_expr,
                                description: "sin(π/3) = √3/2".to_string(),
                            });
                        }
                        "cos" => {
                            // cos(π/3) = 1/2
                            let one = ctx.num(1);
                            let two = ctx.num(2);
                            let new_expr = ctx.add(Expr::Div(one, two));
                            return Some(Rewrite {
                                new_expr,
                                description: "cos(π/3) = 1/2".to_string(),
                            });
                        }
                        "tan" => {
                            // tan(π/3) = √3
                            let three = ctx.num(3);
                            let one = ctx.num(1);
                            let two = ctx.num(2);
                            let half_exp = ctx.add(Expr::Div(one, two));
                            let new_expr = ctx.add(Expr::Pow(three, half_exp));
                            return Some(Rewrite {
                                new_expr,
                                description: "tan(π/3) = √3".to_string(),
                            });
                        }
                        _ => {}
                    }
                }

                // Case 5: Known Values (pi/4) - sin(π/4) = cos(π/4) = √2/2, tan(π/4) = 1
                if is_pi_over_n(ctx, arg, 4) {
                    match name.as_str() {
                        "sin" | "cos" => {
                            // sin(π/4) = cos(π/4) = √2/2
                            let two = ctx.num(2);
                            let one = ctx.num(1);
                            let two2 = ctx.num(2);
                            let half_exp = ctx.add(Expr::Div(one, two2));
                            let sqrt2 = ctx.add(Expr::Pow(two, half_exp));
                            let two3 = ctx.num(2);
                            let new_expr = ctx.add(Expr::Div(sqrt2, two3));
                            return Some(Rewrite {
                                new_expr,
                                description: format!("{}(π/4) = √2/2", name),
                            });
                        }
                        "tan" => {
                            // tan(π/4) = 1
                            let one = ctx.num(1);
                            return Some(Rewrite {
                                new_expr: one,
                                description: "tan(π/4) = 1".to_string(),
                            });
                        }
                        _ => {}
                    }
                }

                // Case 6: Known Values (pi/6) - sin(π/6) = 1/2, cos(π/6) = √3/2, tan(π/6) = 1/√3
                if is_pi_over_n(ctx, arg, 6) {
                    match name.as_str() {
                        "sin" => {
                            // sin(π/6) = 1/2
                            let one = ctx.num(1);
                            let two = ctx.num(2);
                            let new_expr = ctx.add(Expr::Div(one, two));
                            return Some(Rewrite {
                                new_expr,
                                description: "sin(π/6) = 1/2".to_string(),
                            });
                        }
                        "cos" => {
                            // cos(π/6) = √3/2
                            let three = ctx.num(3);
                            let one = ctx.num(1);
                            let two = ctx.num(2);
                            let half_exp = ctx.add(Expr::Div(one, two));
                            let sqrt3 = ctx.add(Expr::Pow(three, half_exp));
                            let two2 = ctx.num(2);
                            let new_expr = ctx.add(Expr::Div(sqrt3, two2));
                            return Some(Rewrite {
                                new_expr,
                                description: "cos(π/6) = √3/2".to_string(),
                            });
                        }
                        "tan" => {
                            // tan(π/6) = 1/√3
                            let three = ctx.num(3);
                            let one = ctx.num(1);
                            let two = ctx.num(2);
                            let half_exp = ctx.add(Expr::Div(one, two));
                            let sqrt3 = ctx.add(Expr::Pow(three, half_exp));
                            let one2 = ctx.num(1);
                            let new_expr = ctx.add(Expr::Div(one2, sqrt3));
                            return Some(Rewrite {
                                new_expr,
                                description: "tan(π/6) = 1/√3".to_string(),
                            });
                        }
                        _ => {}
                    }
                }

                // Case 7: Identities for negative arguments
                // Check for Expr::Neg(inner) OR Expr::Mul(-1, inner)
                let inner_opt = match ctx.get(arg) {
                    Expr::Neg(inner) => Some(*inner),
                    Expr::Mul(l, r) => {
                        if let Expr::Number(n) = ctx.get(*l) {
                            if *n == num_rational::BigRational::from_integer((-1).into()) {
                                Some(*r)
                            } else {
                                None
                            }
                        } else if let Expr::Number(n) = ctx.get(*r) {
                            if *n == num_rational::BigRational::from_integer((-1).into()) {
                                Some(*l)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                if let Some(inner) = inner_opt {
                    match name.as_str() {
                        "sin" => {
                            let sin_inner = ctx.add(Expr::Function("sin".to_string(), vec![inner]));
                            let new_expr = ctx.add(Expr::Neg(sin_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "sin(-x) = -sin(x)".to_string(),
                            });
                        }
                        "cos" => {
                            let new_expr = ctx.add(Expr::Function("cos".to_string(), vec![inner]));
                            return Some(Rewrite {
                                new_expr,
                                description: "cos(-x) = cos(x)".to_string(),
                            });
                        }
                        "tan" => {
                            let tan_inner = ctx.add(Expr::Function("tan".to_string(), vec![inner]));
                            let new_expr = ctx.add(Expr::Neg(tan_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "tan(-x) = -tan(x)".to_string(),
                            });
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }
);

define_rule!(
    PythagoreanIdentityRule,
    "Pythagorean Identity",
    |ctx, expr| {
        // Look for sin(x)^2 + cos(x)^2 = 1
        // Or a*sin(x)^2 + a*cos(x)^2 = a

        let expr_data = ctx.get(expr).clone();
        if let Expr::Add(_, _) = expr_data {
            // Flatten add
            let mut terms = Vec::new();
            crate::helpers::flatten_add(ctx, expr, &mut terms);

            // Helper to extract (coeff, func_name, arg, is_negated) from a term
            // Returns (coeff_expr_id, func_name, arg_expr_id, is_negated)
            // is_negated indicates if the entire term is wrapped in Neg
            let extract_trig_part = |ctx: &mut cas_ast::Context,
                                     term: ExprId|
             -> Option<(ExprId, String, ExprId, bool)> {
                let term_data = ctx.get(term).clone();

                // Check if term is negated: Neg(...)
                let (inner_term, is_negated) = match term_data {
                    Expr::Neg(inner) => (inner, true),
                    _ => (term, false),
                };

                let inner_data = ctx.get(inner_term).clone();

                // Check if term itself is sin^n or cos^n with n >= 2
                if let Expr::Pow(base, exp) = inner_data.clone() {
                    if let Expr::Number(n) = ctx.get(exp) {
                        if n.clone() >= num_rational::BigRational::from_integer(2.into())
                            && n.is_integer()
                        {
                            let trig_info = if let Expr::Function(name, args) = ctx.get(base) {
                                if (name == "sin" || name == "cos") && args.len() == 1 {
                                    Some((name.clone(), args[0]))
                                } else {
                                    None
                                }
                            } else {
                                None
                            };

                            if let Some((name, arg)) = trig_info {
                                // If n > 2, coeff is sin^(n-2)
                                let two = num_rational::BigRational::from_integer(2.into());
                                if n.clone() == two {
                                    return Some((ctx.num(1), name, arg, is_negated));
                                } else {
                                    let rem_exp = n.clone() - two;
                                    if rem_exp.is_one() {
                                        return Some((base, name, arg, is_negated));
                                    } else {
                                        let rem_exp_expr = ctx.add(Expr::Number(rem_exp));
                                        let rem_pow = ctx.add(Expr::Pow(base, rem_exp_expr));
                                        return Some((rem_pow, name, arg, is_negated));
                                    }
                                }
                            }
                        }
                    }
                }

                // Check if inner term is Mul containing sin^n or cos^n
                if let Expr::Mul(_, _) = inner_data {
                    let mut factors = Vec::new();
                    crate::helpers::flatten_mul(ctx, inner_term, &mut factors);

                    // Find the trig square factor (or higher power)
                    let mut trig_idx = None;
                    let mut trig_info = None;
                    let mut trig_rem = None; // Remaining power if n > 2

                    for (i, &factor) in factors.iter().enumerate() {
                        if let Expr::Pow(base, exp) = ctx.get(factor).clone() {
                            if let Expr::Number(n) = ctx.get(exp) {
                                if n.clone() >= num_rational::BigRational::from_integer(2.into())
                                    && n.is_integer()
                                {
                                    if let Expr::Function(name, args) = ctx.get(base) {
                                        if (name == "sin" || name == "cos") && args.len() == 1 {
                                            trig_idx = Some(i);
                                            trig_info = Some((name.clone(), args[0]));

                                            let two =
                                                num_rational::BigRational::from_integer(2.into());
                                            if n.clone() > two {
                                                let rem_exp = n.clone() - two;
                                                if rem_exp.is_one() {
                                                    trig_rem = Some(base);
                                                } else {
                                                    let rem_exp_expr =
                                                        ctx.add(Expr::Number(rem_exp));
                                                    trig_rem = Some(
                                                        ctx.add(Expr::Pow(base, rem_exp_expr)),
                                                    );
                                                }
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if let (Some(idx), Some((name, arg))) = (trig_idx, trig_info) {
                        // Construct coefficient from remaining factors AND remaining power
                        let mut coeff_factors = Vec::new();
                        for (i, &f) in factors.iter().enumerate() {
                            if i != idx {
                                coeff_factors.push(f);
                            }
                        }
                        if let Some(rem) = trig_rem {
                            coeff_factors.push(rem);
                        }

                        let coeff = if coeff_factors.is_empty() {
                            ctx.num(1)
                        } else {
                            let mut c = coeff_factors[0];
                            for &f in coeff_factors.iter().skip(1) {
                                c = ctx.add(Expr::Mul(c, f));
                            }
                            c
                        };
                        return Some((coeff, name, arg, is_negated));
                    }
                }

                None
            };

            // Analyze terms
            struct TrigTerm {
                index: usize,
                coeff: ExprId,
                func_name: String,
                arg: ExprId,
                is_negated: bool, // NEW: Track if term is negated
            }

            let mut trig_terms = Vec::new();
            for (i, &term) in terms.iter().enumerate() {
                if let Some((coeff, name, arg, is_negated)) = extract_trig_part(ctx, term) {
                    trig_terms.push(TrigTerm {
                        index: i,
                        coeff,
                        func_name: name,
                        arg,
                        is_negated,
                    });
                }
            }

            // Find pairs
            for i in 0..trig_terms.len() {
                for j in (i + 1)..trig_terms.len() {
                    let t1 = &trig_terms[i];
                    let t2 = &trig_terms[j];

                    if t1.func_name != t2.func_name {
                        // Check args equality
                        if t1.arg == t2.arg
                            || crate::ordering::compare_expr(ctx, t1.arg, t2.arg)
                                == std::cmp::Ordering::Equal
                        {
                            // Check coefficient equality
                            if t1.coeff == t2.coeff
                                || crate::ordering::compare_expr(ctx, t1.coeff, t2.coeff)
                                    == std::cmp::Ordering::Equal
                            {
                                // NEW: Check negation equality (both positive OR both negative)
                                if t1.is_negated == t2.is_negated {
                                    // Found match!
                                    // Positive pair: coeff * sin^2 + coeff * cos^2 = coeff
                                    // Negated pair: -coeff * sin^2 - coeff * cos^2 = -coeff

                                    // Construct new expression
                                    let mut new_terms = Vec::new();
                                    for k in 0..terms.len() {
                                        if k != t1.index && k != t2.index {
                                            new_terms.push(terms[k]);
                                        }
                                    }

                                    // Add coefficient (negated if pair was negated)
                                    let result_coeff = if t1.is_negated {
                                        ctx.add(Expr::Neg(t1.coeff))
                                    } else {
                                        t1.coeff
                                    };
                                    new_terms.push(result_coeff);

                                    if new_terms.is_empty() {
                                        return Some(Rewrite {
                                            new_expr: ctx.num(0),
                                            description: "Pythagorean Identity (empty)".to_string(),
                                        });
                                    }

                                    let mut new_expr = new_terms[0];
                                    for k in 1..new_terms.len() {
                                        new_expr = ctx.add(Expr::Add(new_expr, new_terms[k]));
                                    }

                                    let description = if t1.is_negated {
                                        "Pythagorean Identity (negated)".to_string()
                                    } else {
                                        "Pythagorean Identity".to_string()
                                    };

                                    return Some(Rewrite {
                                        new_expr,
                                        description,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }
);

define_rule!(AngleIdentityRule, "Angle Sum/Diff Identity", |ctx, expr| {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if args.len() == 1 {
            let inner = args[0];
            match name.as_str() {
                "sin" => {
                    let inner_data = ctx.get(inner).clone();
                    if let Expr::Add(lhs, rhs) = inner_data {
                        // sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
                        let sin_a = ctx.add(Expr::Function("sin".to_string(), vec![lhs]));
                        let cos_b = ctx.add(Expr::Function("cos".to_string(), vec![rhs]));
                        let term1 = ctx.add(Expr::Mul(sin_a, cos_b));

                        let cos_a = ctx.add(Expr::Function("cos".to_string(), vec![lhs]));
                        let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![rhs]));
                        let term2 = ctx.add(Expr::Mul(cos_a, sin_b));

                        let new_expr = ctx.add(Expr::Add(term1, term2));
                        return Some(Rewrite {
                            new_expr,
                            description: "sin(a + b) -> sin(a)cos(b) + cos(a)sin(b)".to_string(),
                        });
                    } else if let Expr::Sub(lhs, rhs) = inner_data {
                        // sin(a - b) = sin(a)cos(b) - cos(a)sin(b)
                        let sin_a = ctx.add(Expr::Function("sin".to_string(), vec![lhs]));
                        let cos_b = ctx.add(Expr::Function("cos".to_string(), vec![rhs]));
                        let term1 = ctx.add(Expr::Mul(sin_a, cos_b));

                        let cos_a = ctx.add(Expr::Function("cos".to_string(), vec![lhs]));
                        let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![rhs]));
                        let term2 = ctx.add(Expr::Mul(cos_a, sin_b));

                        let new_expr = ctx.add(Expr::Sub(term1, term2));
                        return Some(Rewrite {
                            new_expr,
                            description: "sin(a - b) -> sin(a)cos(b) - cos(a)sin(b)".to_string(),
                        });
                    } else if let Expr::Div(num, den) = inner_data {
                        // sin((a + b) / c) -> sin(a/c + b/c) -> ...
                        let num_data = ctx.get(num).clone();
                        if let Expr::Add(lhs, rhs) = num_data {
                            let a = ctx.add(Expr::Div(lhs, den));
                            let b = ctx.add(Expr::Div(rhs, den));

                            let sin_a = ctx.add(Expr::Function("sin".to_string(), vec![a]));
                            let cos_b = ctx.add(Expr::Function("cos".to_string(), vec![b]));
                            let term1 = ctx.add(Expr::Mul(sin_a, cos_b));

                            let cos_a = ctx.add(Expr::Function("cos".to_string(), vec![a]));
                            let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![b]));
                            let term2 = ctx.add(Expr::Mul(cos_a, sin_b));

                            let new_expr = ctx.add(Expr::Add(term1, term2));
                            return Some(Rewrite {
                                new_expr,
                                description:
                                    "sin((a + b)/c) -> sin(a/c)cos(b/c) + cos(a/c)sin(b/c)"
                                        .to_string(),
                            });
                        }
                    }
                }
                "cos" => {
                    let inner_data = ctx.get(inner).clone();
                    if let Expr::Add(lhs, rhs) = inner_data {
                        // cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
                        let cos_a = ctx.add(Expr::Function("cos".to_string(), vec![lhs]));
                        let cos_b = ctx.add(Expr::Function("cos".to_string(), vec![rhs]));
                        let term1 = ctx.add(Expr::Mul(cos_a, cos_b));

                        let sin_a = ctx.add(Expr::Function("sin".to_string(), vec![lhs]));
                        let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![rhs]));
                        let term2 = ctx.add(Expr::Mul(sin_a, sin_b));

                        let new_expr = ctx.add(Expr::Sub(term1, term2));
                        return Some(Rewrite {
                            new_expr,
                            description: "cos(a + b) -> cos(a)cos(b) - sin(a)sin(b)".to_string(),
                        });
                    } else if let Expr::Sub(lhs, rhs) = inner_data {
                        // cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
                        let cos_a = ctx.add(Expr::Function("cos".to_string(), vec![lhs]));
                        let cos_b = ctx.add(Expr::Function("cos".to_string(), vec![rhs]));
                        let term1 = ctx.add(Expr::Mul(cos_a, cos_b));

                        let sin_a = ctx.add(Expr::Function("sin".to_string(), vec![lhs]));
                        let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![rhs]));
                        let term2 = ctx.add(Expr::Mul(sin_a, sin_b));

                        let new_expr = ctx.add(Expr::Add(term1, term2));
                        return Some(Rewrite {
                            new_expr,
                            description: "cos(a - b) -> cos(a)cos(b) + sin(a)sin(b)".to_string(),
                        });
                    } else if let Expr::Div(num, den) = inner_data {
                        // cos((a + b) / c) -> cos(a/c + b/c) -> ...
                        let num_data = ctx.get(num).clone();
                        if let Expr::Add(lhs, rhs) = num_data {
                            let a = ctx.add(Expr::Div(lhs, den));
                            let b = ctx.add(Expr::Div(rhs, den));

                            let cos_a = ctx.add(Expr::Function("cos".to_string(), vec![a]));
                            let cos_b = ctx.add(Expr::Function("cos".to_string(), vec![b]));
                            let term1 = ctx.add(Expr::Mul(cos_a, cos_b));

                            let sin_a = ctx.add(Expr::Function("sin".to_string(), vec![a]));
                            let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![b]));
                            let term2 = ctx.add(Expr::Mul(sin_a, sin_b));

                            let new_expr = ctx.add(Expr::Sub(term1, term2));
                            return Some(Rewrite {
                                new_expr,
                                description:
                                    "cos((a + b)/c) -> cos(a/c)cos(b/c) - sin(a/c)sin(b/c)"
                                        .to_string(),
                            });
                        }
                    }
                }
                _ => {}
            }
        }
    }
    None
});

/// Convert tan(x) to sin(x)/cos(x) UNLESS it's part of a Pythagorean pattern
pub struct TanToSinCosRule;

impl crate::rule::Rule for TanToSinCosRule {
    fn name(&self) -> &str {
        "Tan to Sin/Cos"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use cas_ast::Expr;

        // GUARD: Check pattern_marks - don't convert if protected
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_pythagorean_protected(expr) {
                return None; // Skip conversion - part of Pythagorean identity
            }
        }

        // Original conversion logic
        let expr_data = ctx.get(expr).clone();
        if let Expr::Function(name, args) = expr_data {
            if name == "tan" && args.len() == 1 {
                // tan(x) -> sin(x) / cos(x)
                let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![args[0]]));
                let cos_x = ctx.add(Expr::Function("cos".to_string(), vec![args[0]]));
                let new_expr = ctx.add(Expr::Div(sin_x, cos_x));
                return Some(crate::rule::Rewrite {
                    new_expr,
                    description: "tan(x) -> sin(x)/cos(x)".to_string(),
                });
            }
        }
        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Function"])
    }
}

// Secant-Tangent Pythagorean Identity: sec²(x) - tan²(x) = 1
// Also recognizes factored form: (sec(x) + tan(x)) * (sec(x) - tan(x)) = 1
define_rule!(
    SecTanPythagoreanRule,
    "Secant-Tangent Pythagorean Identity",
    |ctx, expr| {
        use crate::pattern_detection::{is_sec_squared, is_tan_squared};

        let expr_data = ctx.get(expr).clone();

        // Pattern 1: sec²(x) - tan²(x) = 1
        // NOTE: Subtraction is normalized to Add(a, Neg(b))
        if let Expr::Add(left, right) = expr_data {
            // Try both orderings: Add(sec², Neg(tan²)) or Add(Neg(tan²), sec²)
            for (pos, neg) in [(left, right), (right, left)] {
                if let Expr::Neg(neg_inner) = ctx.get(neg) {
                    // Check if pos=sec²  and neg_inner=tan²
                    if let (Some(sec_arg), Some(tan_arg)) =
                        (is_sec_squared(ctx, pos), is_tan_squared(ctx, *neg_inner))
                    {
                        if crate::ordering::compare_expr(ctx, sec_arg, tan_arg)
                            == std::cmp::Ordering::Equal
                        {
                            return Some(Rewrite {
                                new_expr: ctx.num(1),
                                description: "sec²(x) - tan²(x) = 1".to_string(),
                            });
                        }
                    }
                }
            }
        }

        None
    }
);

// Cosecant-Cotangent Pythagorean Identity: csc²(x) - cot²(x) = 1
// NOTE: Subtraction is normalized to Add(a, Neg(b))
define_rule!(
    CscCotPythagoreanRule,
    "Cosecant-Cotangent Pythagorean Identity",
    |ctx, expr| {
        use crate::pattern_detection::{is_cot_squared, is_csc_squared};

        let expr_data = ctx.get(expr).clone();

        // Pattern: csc²(x) - cot²(x) = 1
        if let Expr::Add(left, right) = expr_data {
            for (pos, neg) in [(left, right), (right, left)] {
                if let Expr::Neg(neg_inner) = ctx.get(neg) {
                    // Check if pos=csc² and neg_inner=cot²
                    if let (Some(csc_arg), Some(cot_arg)) =
                        (is_csc_squared(ctx, pos), is_cot_squared(ctx, *neg_inner))
                    {
                        if crate::ordering::compare_expr(ctx, csc_arg, cot_arg)
                            == std::cmp::Ordering::Equal
                        {
                            return Some(Rewrite {
                                new_expr: ctx.num(1),
                                description: "csc²(x) - cot²(x) = 1".to_string(),
                            });
                        }
                    }
                }
            }
        }

        None
    }
);

define_rule!(DoubleAngleRule, "Double Angle Identity", |ctx, expr| {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if args.len() == 1 {
            // Check if arg is 2*x or x*2
            // We need to match "2 * x"
            if let Some(inner_var) = extract_double_angle_arg(ctx, args[0]) {
                match name.as_str() {
                    "sin" => {
                        // sin(2x) -> 2sin(x)cos(x)
                        let two = ctx.num(2);
                        let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![inner_var]));
                        let cos_x = ctx.add(Expr::Function("cos".to_string(), vec![inner_var]));
                        let sin_cos = ctx.add(Expr::Mul(sin_x, cos_x));
                        let new_expr = ctx.add(Expr::Mul(two, sin_cos));
                        return Some(Rewrite {
                            new_expr,
                            description: "sin(2x) -> 2sin(x)cos(x)".to_string(),
                        });
                    }
                    "cos" => {
                        // cos(2x) -> cos^2(x) - sin^2(x)
                        let two = ctx.num(2);
                        let cos_x = ctx.add(Expr::Function("cos".to_string(), vec![inner_var]));
                        let cos2 = ctx.add(Expr::Pow(cos_x, two));

                        let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![inner_var]));
                        let sin2 = ctx.add(Expr::Pow(sin_x, two));

                        let new_expr = ctx.add(Expr::Sub(cos2, sin2));
                        return Some(Rewrite {
                            new_expr,
                            description: "cos(2x) -> cos^2(x) - sin^2(x)".to_string(),
                        });
                    }
                    _ => {}
                }
            }
        }
    }
    None
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::{Context, DisplayExpr};
    use cas_parser::parse;

    #[test]
    fn test_evaluate_trig_zero() {
        let mut ctx = Context::new();
        let rule = EvaluateTrigRule;

        // sin(0) -> 0
        let expr = parse("sin(0)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );

        // cos(0) -> 1
        let expr = parse("cos(0)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "1"
        );

        // tan(0) -> 0
        let expr = parse("tan(0)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
    }

    #[test]
    fn test_evaluate_trig_identities() {
        let mut ctx = Context::new();
        let rule = EvaluateTrigRule;

        // sin(-x) -> -sin(x)
        let expr = parse("sin(-x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "-sin(x)"
        );

        // cos(-x) -> cos(x)
        let expr = parse("cos(-x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "cos(x)"
        );

        // tan(-x) -> -tan(x)
        let expr = parse("tan(-x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "-tan(x)"
        );
    }

    #[test]
    fn test_trig_identities() {
        let mut ctx = Context::new();
        let rule = AngleIdentityRule;

        // sin(x + y)
        let expr = parse("sin(x + y)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("sin(x)"));

        // cos(x + y) -> cos(x)cos(y) - sin(x)sin(y)
        let expr = parse("cos(x + y)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        let res = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        assert!(res.contains("cos(x)"));
        assert!(res.contains("-"));

        // sin(x - y)
        let expr = parse("sin(x - y)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("-"));
    }

    #[test]
    fn test_tan_to_sin_cos() {
        let mut ctx = Context::new();
        let rule = TanToSinCosRule;
        let expr = parse("tan(x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "sin(x) / cos(x)"
        );
    }

    #[test]
    fn test_double_angle() {
        let mut ctx = Context::new();
        let rule = DoubleAngleRule;

        // sin(2x)
        let expr = parse("sin(2 * x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        // Check that result contains the key components, regardless of order
        assert!(
            result_str.contains("sin(x)"),
            "Result should contain sin(x), got: {}",
            result_str
        );
        assert!(
            result_str.contains("cos(x)"),
            "Result should contain cos(x), got: {}",
            result_str
        );
        assert!(
            result_str.contains("2") || result_str.contains("* 2") || result_str.contains("2 *"),
            "Result should contain 2, got: {}",
            result_str
        );

        // cos(2x)
        let expr = parse("cos(2 * x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("cos(x)^2 - sin(x)^2"));
    }

    #[test]
    fn test_evaluate_inverse_trig() {
        let mut ctx = Context::new();
        let rule = EvaluateTrigRule;

        // arcsin(0) -> 0
        let expr = parse("arcsin(0)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );

        // arccos(1) -> 0
        let expr = parse("arccos(1)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );

        // arcsin(1) -> pi/2
        // Note: pi/2 might be formatted as "pi / 2" or similar depending on Display impl
        let expr = parse("arcsin(1)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("pi"));
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("2"));

        // arccos(0) -> pi/2
        let expr = parse("arccos(0)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("pi"));
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("2"));
    }
}

define_rule!(
    RecursiveTrigExpansionRule,
    "Recursive Trig Expansion",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Function(name, args) = expr_data {
            if args.len() == 1 && (name == "sin" || name == "cos") {
                // Check for n * x where n is integer > 2
                let inner = args[0];
                let inner_data = ctx.get(inner).clone();

                let (n_val, x_val) = if let Expr::Mul(l, r) = inner_data {
                    if let Expr::Number(n) = ctx.get(l) {
                        if n.is_integer() {
                            (n.to_integer(), r)
                        } else {
                            return None;
                        }
                    } else if let Expr::Number(n) = ctx.get(r) {
                        if n.is_integer() {
                            (n.to_integer(), l)
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else {
                    return None;
                };

                if n_val > num_bigint::BigInt::from(2) {
                    // Rewrite sin(nx) -> sin((n-1)x + x)

                    let n_minus_1 = n_val.clone() - 1;
                    let n_minus_1_expr = ctx.add(Expr::Number(
                        num_rational::BigRational::from_integer(n_minus_1),
                    ));
                    let term_nm1 = ctx.add(Expr::Mul(n_minus_1_expr, x_val));

                    // sin(nx) = sin((n-1)x)cos(x) + cos((n-1)x)sin(x)
                    // cos(nx) = cos((n-1)x)cos(x) - sin((n-1)x)sin(x)

                    let sin_nm1 = ctx.add(Expr::Function("sin".to_string(), vec![term_nm1]));
                    let cos_nm1 = ctx.add(Expr::Function("cos".to_string(), vec![term_nm1]));
                    let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![x_val]));
                    let cos_x = ctx.add(Expr::Function("cos".to_string(), vec![x_val]));

                    if name == "sin" {
                        let t1 = ctx.add(Expr::Mul(sin_nm1, cos_x));
                        let t2 = ctx.add(Expr::Mul(cos_nm1, sin_x));
                        let new_expr = ctx.add(Expr::Add(t1, t2));
                        return Some(Rewrite {
                            new_expr,
                            description: format!("sin({}x) expansion", n_val),
                        });
                    } else {
                        // cos
                        let t1 = ctx.add(Expr::Mul(cos_nm1, cos_x));
                        let t2 = ctx.add(Expr::Mul(sin_nm1, sin_x));
                        let new_expr = ctx.add(Expr::Sub(t1, t2));
                        return Some(Rewrite {
                            new_expr,
                            description: format!("cos({}x) expansion", n_val),
                        });
                    }
                }
            }
        }
        None
    }
);

define_rule!(
    CanonicalizeTrigSquareRule,
    "Canonicalize Trig Square",
    |ctx, expr| {
        // cos^n(x) -> (1 - sin^2(x))^(n/2) for even n
        let expr_data = ctx.get(expr).clone();
        if let Expr::Pow(base, exp) = expr_data {
            let n_opt = if let Expr::Number(n) = ctx.get(exp) {
                Some(n.clone())
            } else {
                None
            };

            if let Some(n) = n_opt {
                if n.is_integer()
                    && n.to_integer() % 2 == 0.into()
                    && n > num_rational::BigRational::zero()
                {
                    // Limit power to avoid explosion? Let's say <= 4 for now.
                    if n <= num_rational::BigRational::from_integer(4.into()) {
                        if let Expr::Function(name, args) = ctx.get(base) {
                            if name == "cos" && args.len() == 1 {
                                let arg = args[0];
                                // (1 - sin^2(x))^(n/2)
                                let one = ctx.num(1);
                                let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![arg]));
                                let two = ctx.num(2);
                                let sin_sq = ctx.add(Expr::Pow(sin_x, two));
                                let base_term = ctx.add(Expr::Sub(one, sin_sq));

                                let half_n =
                                    n.clone() / num_rational::BigRational::from_integer(2.into());

                                if half_n.is_one() {
                                    return Some(Rewrite {
                                        new_expr: base_term,
                                        description: "cos^2(x) -> 1 - sin^2(x)".to_string(),
                                    });
                                } else {
                                    let half_n_expr = ctx.add(Expr::Number(half_n));
                                    let new_expr = ctx.add(Expr::Pow(base_term, half_n_expr));
                                    return Some(Rewrite {
                                        new_expr,
                                        description: "cos^2k(x) -> (1 - sin^2(x))^k".to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(EvaluateTrigRule));
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    simplifier.add_rule(Box::new(SecTanPythagoreanRule));
    simplifier.add_rule(Box::new(CscCotPythagoreanRule));
    simplifier.add_rule(Box::new(AngleIdentityRule));
    simplifier.add_rule(Box::new(TanToSinCosRule));
    simplifier.add_rule(Box::new(DoubleAngleRule));
    simplifier.add_rule(Box::new(RecursiveTrigExpansionRule));

    // DISABLED: Conflicts with Pythagorean identity rules causing infinite loops
    // This rule converts cos²(x) → 1-sin²(x) which interacts badly with:
    // - Pythagorean identities (sec²-tan²=1)
    // - Reciprocal trig canonicalization
    // Creating transformation cycles like: sec² → 1/cos² → 1/(1-sin²) → ...
    // See: debug_sec_tan.rs test and GitHub issue #X
    // simplifier.add_rule(Box::new(CanonicalizeTrigSquareRule));

    simplifier.add_rule(Box::new(AngleConsistencyRule));
}

define_rule!(
    AngleConsistencyRule,
    "Angle Consistency (Half-Angle)",
    |ctx, expr| {
        // Only run on Add/Sub/Mul/Div to capture context
        match ctx.get(expr) {
            Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Mul(_, _) | Expr::Div(_, _) => {}
            _ => return None,
        }

        // 1. Collect all trig arguments
        let mut trig_args = Vec::new();
        collect_trig_args_recursive(ctx, expr, &mut trig_args);

        if trig_args.is_empty() {
            return None;
        }

        // 2. Check for half-angle relationship
        // We look for pair (A, B) such that A = 2*B.
        // Then we expand trig(A) into trig(B).

        let mut target_expansion: Option<(ExprId, ExprId)> = None; // (A, B) where A=2B

        for i in 0..trig_args.len() {
            for j in 0..trig_args.len() {
                if i == j {
                    continue;
                }
                let a = trig_args[i];
                let b = trig_args[j];

                if is_double(ctx, a, b) {
                    target_expansion = Some((a, b));
                    break;
                }
            }
            if target_expansion.is_some() {
                break;
            }
        }

        if let Some((large_angle, small_angle)) = target_expansion {
            // Expand all occurrences of trig(large_angle) in expr
            // We need a recursive replacement helper
            let new_expr = expand_trig_angle(ctx, expr, large_angle, small_angle);
            if new_expr != expr {
                return Some(Rewrite {
                    new_expr,
                    description: "Half-Angle Expansion".to_string(),
                });
            }
        }

        None
    }
);

fn collect_trig_args_recursive(ctx: &cas_ast::Context, expr: ExprId, args: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Function(name, fargs) => {
            if (name == "sin" || name == "cos" || name == "tan") && fargs.len() == 1 {
                args.push(fargs[0]);
            }
            for arg in fargs {
                collect_trig_args_recursive(ctx, *arg, args);
            }
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            collect_trig_args_recursive(ctx, *l, args);
            collect_trig_args_recursive(ctx, *r, args);
        }
        Expr::Neg(e) => collect_trig_args_recursive(ctx, *e, args),
        _ => {}
    }
}

fn is_double(ctx: &cas_ast::Context, large: ExprId, small: ExprId) -> bool {
    // Check if large == 2 * small

    // Case 1: large = 2 * small
    if let Expr::Mul(l, r) = ctx.get(large) {
        if let Expr::Number(n) = ctx.get(*l) {
            if n == &num_rational::BigRational::from_integer(2.into())
                && crate::ordering::compare_expr(ctx, *r, small) == Ordering::Equal
            {
                return true;
            }
        }
        if let Expr::Number(n) = ctx.get(*r) {
            if n == &num_rational::BigRational::from_integer(2.into())
                && crate::ordering::compare_expr(ctx, *l, small) == Ordering::Equal
            {
                return true;
            }
        }
    }

    // Case 2: small = large / 2
    if let Expr::Div(n, d) = ctx.get(small) {
        if let Expr::Number(val) = ctx.get(*d) {
            if val == &num_rational::BigRational::from_integer(2.into())
                && crate::ordering::compare_expr(ctx, *n, large) == Ordering::Equal
            {
                return true;
            }
        }
    }

    // Case 3: small = large * 0.5
    if let Expr::Mul(l, r) = ctx.get(small) {
        if let Expr::Number(n) = ctx.get(*l) {
            if n == &num_rational::BigRational::new(1.into(), 2.into())
                && crate::ordering::compare_expr(ctx, *r, large) == Ordering::Equal
            {
                return true;
            }
        }
        if let Expr::Number(n) = ctx.get(*r) {
            if n == &num_rational::BigRational::new(1.into(), 2.into())
                && crate::ordering::compare_expr(ctx, *l, large) == Ordering::Equal
            {
                return true;
            }
        }
    }

    false
}

fn expand_trig_angle(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    large_angle: ExprId,
    small_angle: ExprId,
) -> ExprId {
    let expr_data = ctx.get(expr).clone();

    // Check if this node is trig(large_angle)
    if let Expr::Function(name, args) = &expr_data {
        if args.len() == 1
            && crate::ordering::compare_expr(ctx, args[0], large_angle) == Ordering::Equal
        {
            match name.as_str() {
                "sin" => {
                    // sin(A) -> 2sin(A/2)cos(A/2)
                    let two = ctx.num(2);
                    let sin_half = ctx.add(Expr::Function("sin".to_string(), vec![small_angle]));
                    let cos_half = ctx.add(Expr::Function("cos".to_string(), vec![small_angle]));
                    let term = ctx.add(Expr::Mul(sin_half, cos_half));
                    return ctx.add(Expr::Mul(two, term));
                }
                "cos" => {
                    // cos(A) -> 2cos^2(A/2) - 1
                    let two = ctx.num(2);
                    let one = ctx.num(1);
                    let cos_half = ctx.add(Expr::Function("cos".to_string(), vec![small_angle]));
                    let cos_sq = ctx.add(Expr::Pow(cos_half, two));
                    let term = ctx.add(Expr::Mul(two, cos_sq));
                    return ctx.add(Expr::Sub(term, one));
                }
                "tan" => {
                    // tan(A) -> 2tan(A/2) / (1 - tan^2(A/2))
                    let two = ctx.num(2);
                    let one = ctx.num(1);
                    let tan_half = ctx.add(Expr::Function("tan".to_string(), vec![small_angle]));
                    let num = ctx.add(Expr::Mul(two, tan_half));

                    let tan_sq = ctx.add(Expr::Pow(tan_half, two));
                    let den = ctx.add(Expr::Sub(one, tan_sq));

                    return ctx.add(Expr::Div(num, den));
                }
                _ => {}
            }
        }
    }

    // Recurse
    match expr_data {
        Expr::Add(l, r) => {
            let nl = expand_trig_angle(ctx, l, large_angle, small_angle);
            let nr = expand_trig_angle(ctx, r, large_angle, small_angle);
            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                expr
            }
        }
        Expr::Sub(l, r) => {
            let nl = expand_trig_angle(ctx, l, large_angle, small_angle);
            let nr = expand_trig_angle(ctx, r, large_angle, small_angle);
            if nl != l || nr != r {
                ctx.add(Expr::Sub(nl, nr))
            } else {
                expr
            }
        }
        Expr::Mul(l, r) => {
            let nl = expand_trig_angle(ctx, l, large_angle, small_angle);
            let nr = expand_trig_angle(ctx, r, large_angle, small_angle);
            if nl != l || nr != r {
                ctx.add(Expr::Mul(nl, nr))
            } else {
                expr
            }
        }
        Expr::Div(l, r) => {
            let nl = expand_trig_angle(ctx, l, large_angle, small_angle);
            let nr = expand_trig_angle(ctx, r, large_angle, small_angle);
            if nl != l || nr != r {
                ctx.add(Expr::Div(nl, nr))
            } else {
                expr
            }
        }
        Expr::Pow(b, e) => {
            let nb = expand_trig_angle(ctx, b, large_angle, small_angle);
            let ne = expand_trig_angle(ctx, e, large_angle, small_angle);
            if nb != b || ne != e {
                ctx.add(Expr::Pow(nb, ne))
            } else {
                expr
            }
        }
        Expr::Neg(e) => {
            let ne = expand_trig_angle(ctx, e, large_angle, small_angle);
            if ne != e {
                ctx.add(Expr::Neg(ne))
            } else {
                expr
            }
        }
        Expr::Function(name, args) => {
            let mut new_args = Vec::new();
            let mut changed = false;
            for arg in args {
                let na = expand_trig_angle(ctx, arg, large_angle, small_angle);
                if na != arg {
                    changed = true;
                }
                new_args.push(na);
            }
            if changed {
                ctx.add(Expr::Function(name, new_args))
            } else {
                expr
            }
        }
        _ => expr,
    }
}
