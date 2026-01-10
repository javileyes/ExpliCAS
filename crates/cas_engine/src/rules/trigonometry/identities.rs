use crate::define_rule;
use crate::helpers::{extract_double_angle_arg, extract_triple_angle_arg, is_pi, is_pi_over_n};
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Expr, ExprId};
use num_traits::{One, Zero};

use std::cmp::Ordering;

// NOTE: EvaluateTrigRule is deprecated - use EvaluateTrigTableRule from evaluation.rs instead
// This rule is kept for reference but should not be registered in the simplifier.
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
                                    before_local: None,
                                    after_local: None,
                                    assumption_events: Default::default(),
                                    required_conditions: vec![],
                                    poly_proof: None,
                                });
                            }
                            "cos" => {
                                let one = ctx.num(1);
                                return Some(Rewrite::new(one).desc("cos(0) = 1"));
                            }
                            "arccos" => {
                                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                let two = ctx.num(2);
                                let new_expr = ctx.add(Expr::Div(pi, two));
                                return Some(Rewrite {
                                    new_expr,
                                    description: "arccos(0) = pi/2".to_string(),
                                    before_local: None,
                                    after_local: None,
                                    assumption_events: Default::default(),
                                    required_conditions: vec![],
                                    poly_proof: None,
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
                                    before_local: None,
                                    after_local: None,
                                    assumption_events: Default::default(),
                                    required_conditions: vec![],
                                    poly_proof: None,
                                });
                            }
                            "arccos" => {
                                let zero = ctx.num(0);
                                return Some(Rewrite::new(zero).desc("arccos(1) = 0"));
                            }
                            "arctan" => {
                                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                let four = ctx.num(4);
                                let new_expr = ctx.add(Expr::Div(pi, four));
                                return Some(Rewrite {
                                    new_expr,
                                    description: "arctan(1) = pi/4".to_string(),
                                    before_local: None,
                                    after_local: None,
                                    assumption_events: Default::default(),
                                    required_conditions: vec![],
                                    poly_proof: None,
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
                                    before_local: None,
                                    after_local: None,
                                    assumption_events: Default::default(),
                                    required_conditions: vec![],
                                    poly_proof: None,
                                });
                            }
                            "arccos" => {
                                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                let three = ctx.num(3);
                                let new_expr = ctx.add(Expr::Div(pi, three));
                                return Some(Rewrite {
                                    new_expr,
                                    description: "arccos(1/2) = pi/3".to_string(),
                                    before_local: None,
                                    after_local: None,
                                    assumption_events: Default::default(),
                                    required_conditions: vec![],
                                    poly_proof: None,
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
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                                poly_proof: None,
                            });
                        }
                        "cos" => {
                            let neg_one = ctx.num(-1);
                            return Some(Rewrite::new(neg_one).desc("cos(pi) = -1"));
                        }
                        _ => {}
                    }
                }

                // Case 3: Known Values (pi/2) - using shared helper for both Div and Mul formats
                if is_pi_over_n(ctx, arg, 2) {
                    match name.as_str() {
                        "sin" => {
                            let one = ctx.num(1);
                            return Some(Rewrite::new(one).desc("sin(pi/2) = 1"));
                        }
                        "cos" => {
                            let zero = ctx.num(0);
                            return Some(Rewrite::new(zero).desc("cos(pi/2) = 0"));
                        }
                        "tan" => {
                            let undefined = ctx.add(Expr::Constant(cas_ast::Constant::Undefined));
                            return Some(Rewrite::new(undefined).desc("tan(pi/2) = undefined"));
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
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                                poly_proof: None,
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
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                                poly_proof: None,
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
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                                poly_proof: None,
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
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                                poly_proof: None,
                            });
                        }
                        "tan" => {
                            // tan(π/4) = 1
                            let one = ctx.num(1);
                            return Some(Rewrite::new(one).desc("tan(π/4) = 1"));
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
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                                poly_proof: None,
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
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                                poly_proof: None,
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
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                                poly_proof: None,
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
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                                poly_proof: None,
                            });
                        }
                        "cos" => {
                            let new_expr = ctx.add(Expr::Function("cos".to_string(), vec![inner]));
                            return Some(Rewrite {
                                new_expr,
                                description: "cos(-x) = cos(x)".to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                                poly_proof: None,
                            });
                        }
                        "tan" => {
                            let tan_inner = ctx.add(Expr::Function("tan".to_string(), vec![inner]));
                            let new_expr = ctx.add(Expr::Neg(tan_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "tan(-x) = -tan(x)".to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                                poly_proof: None,
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
    None,
    crate::phase::PhaseMask::TRANSFORM, // Match phase with TrigHiddenCubicIdentityRule
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
                                c = smart_mul(ctx, c, f);
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
                                    for (k, &term) in terms.iter().enumerate() {
                                        if k != t1.index && k != t2.index {
                                            new_terms.push(term);
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
                                            before_local: None,
                                            after_local: None,
                                            assumption_events: Default::default(),
                                            required_conditions: vec![],
                                            poly_proof: None,
                                        });
                                    }

                                    let mut new_expr = new_terms[0];
                                    for &term in new_terms.iter().skip(1) {
                                        new_expr = ctx.add(Expr::Add(new_expr, term));
                                    }

                                    let description = if t1.is_negated {
                                        "Pythagorean Identity (negated)".to_string()
                                    } else {
                                        "Pythagorean Identity".to_string()
                                    };

                                    return Some(Rewrite {
                                        new_expr,
                                        description,
                                        before_local: None,
                                        after_local: None,
                                        assumption_events: Default::default(),
                                        required_conditions: vec![],
                                        poly_proof: None,
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

// MANUAL IMPLEMENTATION: AngleIdentityRule with parent context guard
// Don't expand sin(a+b)/cos(a+b) when they are being raised to power ≥ 2
// This preserves the Pythagorean identity pattern sin²+cos²=1
pub struct AngleIdentityRule;

impl crate::rule::Rule for AngleIdentityRule {
    fn name(&self) -> &str {
        "Angle Sum/Diff Identity"
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        crate::phase::PhaseMask::TRANSFORM
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        use cas_ast::Expr;

        // GUARD: Don't expand sin(a+b)/cos(a+b) if this function is part of sin²+cos²=1 pattern
        // The pattern marks are set by pre-scan before simplification
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_trig_square_protected(expr) {
                return None; // Skip expansion to preserve Pythagorean identity
            }
        }

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
                            let term1 = smart_mul(ctx, sin_a, cos_b);

                            let cos_a = ctx.add(Expr::Function("cos".to_string(), vec![lhs]));
                            let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![rhs]));
                            let term2 = smart_mul(ctx, cos_a, sin_b);

                            let new_expr = ctx.add(Expr::Add(term1, term2));
                            return Some(Rewrite {
                                new_expr,
                                description: "sin(a + b) -> sin(a)cos(b) + cos(a)sin(b)"
                                    .to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                                poly_proof: None,
                            });
                        } else if let Expr::Sub(lhs, rhs) = inner_data {
                            // sin(a - b) = sin(a)cos(b) - cos(a)sin(b)
                            let sin_a = ctx.add(Expr::Function("sin".to_string(), vec![lhs]));
                            let cos_b = ctx.add(Expr::Function("cos".to_string(), vec![rhs]));
                            let term1 = smart_mul(ctx, sin_a, cos_b);

                            let cos_a = ctx.add(Expr::Function("cos".to_string(), vec![lhs]));
                            let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![rhs]));
                            let term2 = smart_mul(ctx, cos_a, sin_b);

                            let new_expr = ctx.add(Expr::Sub(term1, term2));
                            return Some(Rewrite {
                                new_expr,
                                description: "sin(a - b) -> sin(a)cos(b) - cos(a)sin(b)"
                                    .to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                                poly_proof: None,
                            });
                        } else if let Expr::Div(num, den) = inner_data {
                            // sin((a + b) / c) -> sin(a/c + b/c) -> ...
                            let num_data = ctx.get(num).clone();
                            if let Expr::Add(lhs, rhs) = num_data {
                                let a = ctx.add(Expr::Div(lhs, den));
                                let b = ctx.add(Expr::Div(rhs, den));

                                let sin_a = ctx.add(Expr::Function("sin".to_string(), vec![a]));
                                let cos_b = ctx.add(Expr::Function("cos".to_string(), vec![b]));
                                let term1 = smart_mul(ctx, sin_a, cos_b);

                                let cos_a = ctx.add(Expr::Function("cos".to_string(), vec![a]));
                                let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![b]));
                                let term2 = smart_mul(ctx, cos_a, sin_b);

                                let new_expr = ctx.add(Expr::Add(term1, term2));
                                return Some(Rewrite {
                                    new_expr,
                                    description:
                                        "sin((a + b)/c) -> sin(a/c)cos(b/c) + cos(a/c)sin(b/c)"
                                            .to_string(),
                                    before_local: None,
                                    after_local: None,
                                    assumption_events: Default::default(),
                                    required_conditions: vec![],
                                    poly_proof: None,
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
                            let term1 = smart_mul(ctx, cos_a, cos_b);

                            let sin_a = ctx.add(Expr::Function("sin".to_string(), vec![lhs]));
                            let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![rhs]));
                            let term2 = smart_mul(ctx, sin_a, sin_b);

                            let new_expr = ctx.add(Expr::Sub(term1, term2));
                            return Some(Rewrite {
                                new_expr,
                                description: "cos(a + b) -> cos(a)cos(b) - sin(a)sin(b)"
                                    .to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                                poly_proof: None,
                            });
                        } else if let Expr::Sub(lhs, rhs) = inner_data {
                            // cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
                            let cos_a = ctx.add(Expr::Function("cos".to_string(), vec![lhs]));
                            let cos_b = ctx.add(Expr::Function("cos".to_string(), vec![rhs]));
                            let term1 = smart_mul(ctx, cos_a, cos_b);

                            let sin_a = ctx.add(Expr::Function("sin".to_string(), vec![lhs]));
                            let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![rhs]));
                            let term2 = smart_mul(ctx, sin_a, sin_b);

                            let new_expr = ctx.add(Expr::Add(term1, term2));
                            return Some(Rewrite {
                                new_expr,
                                description: "cos(a - b) -> cos(a)cos(b) + sin(a)sin(b)"
                                    .to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                                poly_proof: None,
                            });
                        } else if let Expr::Div(num, den) = inner_data {
                            // cos((a + b) / c) -> cos(a/c + b/c) -> ...
                            let num_data = ctx.get(num).clone();
                            if let Expr::Add(lhs, rhs) = num_data {
                                let a = ctx.add(Expr::Div(lhs, den));
                                let b = ctx.add(Expr::Div(rhs, den));

                                let cos_a = ctx.add(Expr::Function("cos".to_string(), vec![a]));
                                let cos_b = ctx.add(Expr::Function("cos".to_string(), vec![b]));
                                let term1 = smart_mul(ctx, cos_a, cos_b);

                                let sin_a = ctx.add(Expr::Function("sin".to_string(), vec![a]));
                                let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![b]));
                                let term2 = smart_mul(ctx, sin_a, sin_b);

                                let new_expr = ctx.add(Expr::Sub(term1, term2));
                                return Some(Rewrite {
                                    new_expr,
                                    description:
                                        "cos((a + b)/c) -> cos(a/c)cos(b/c) - sin(a/c)sin(b/c)"
                                            .to_string(),
                                    before_local: None,
                                    after_local: None,
                                    assumption_events: Default::default(),
                                    required_conditions: vec![],
                                    poly_proof: None,
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        None
    }
}

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
            // Inverse trig pattern protection is UNCONDITIONAL.
            // We always preserve arctan(tan(x)), arcsin(sin(x)), etc.
            // The policy only controls whether it SIMPLIFIES to x, not whether we expand it.
            if marks.is_inverse_trig_protected(expr) {
                return None; // Preserve pattern: arctan(tan(x)) stays as-is
            }
        }

        // GUARD: Also check immediate parent for inverse trig composition.
        // This is a fallback in case pattern_marks wasn't pre-scanned.
        if let Some(parent_id) = parent_ctx.immediate_parent() {
            if let Expr::Function(name, _) = ctx.get(parent_id) {
                if matches!(
                    name.as_str(),
                    "arctan" | "arcsin" | "arccos" | "atan" | "asin" | "acos"
                ) {
                    return None; // Preserve arctan(tan(x)) pattern
                }
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
                    before_local: None,
                    after_local: None,
                    assumption_events: Default::default(),
                    required_conditions: vec![],
                    poly_proof: None,
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
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                                poly_proof: None,
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
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                                poly_proof: None,
                            });
                        }
                    }
                }
            }
        }

        None
    }
);

// =============================================================================
// HIDDEN CUBIC TRIG IDENTITY
// sin^6(x) + cos^6(x) + 3*sin^2(x)*cos^2(x) = (sin^2(x) + cos^2(x))^3
// =============================================================================

/// Extract sin(arg)^6 or cos(arg)^6 from a term.
/// Returns Some((arg, "sin"|"cos")) if matched.
fn extract_trig_pow6(ctx: &cas_ast::Context, term: ExprId) -> Option<(ExprId, &'static str)> {
    if let Expr::Pow(base, exp) = ctx.get(term) {
        // Check exponent is 6
        if let Expr::Number(n) = ctx.get(*exp) {
            if n.is_integer() && *n.numer() == 6.into() {
                // Check base is sin(arg) or cos(arg)
                if let Expr::Function(name, args) = ctx.get(*base) {
                    if args.len() == 1 {
                        match name.as_str() {
                            "sin" => return Some((args[0], "sin")),
                            "cos" => return Some((args[0], "cos")),
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    None
}

/// Extract sin(arg)^2 or cos(arg)^2 from terms.
/// Returns Some((arg, "sin"|"cos")) if matched.
fn extract_trig_pow2(ctx: &cas_ast::Context, term: ExprId) -> Option<(ExprId, &'static str)> {
    if let Expr::Pow(base, exp) = ctx.get(term) {
        // Check exponent is 2
        if let Expr::Number(n) = ctx.get(*exp) {
            if n.is_integer() && *n.numer() == 2.into() {
                // Check base is sin(arg) or cos(arg)
                if let Expr::Function(name, args) = ctx.get(*base) {
                    if args.len() == 1 {
                        match name.as_str() {
                            "sin" => return Some((args[0], "sin")),
                            "cos" => return Some((args[0], "cos")),
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    None
}

/// Extract coeff * sin(arg)^2 * cos(arg)^2 from a term.
/// Returns Some((coeff_expr, arg)) where coeff_expr is the coefficient expression.
/// The caller should verify coeff_expr simplifies to 3.
fn extract_sin2_cos2_product(ctx: &mut cas_ast::Context, term: ExprId) -> Option<(ExprId, ExprId)> {
    // Flatten the multiplication
    let factors = crate::helpers::flatten_mul_chain(ctx, term);

    if factors.len() < 2 {
        return None;
    }

    let mut sin2_arg: Option<ExprId> = None;
    let mut cos2_arg: Option<ExprId> = None;
    let mut other_factors: Vec<ExprId> = Vec::new();

    for factor in &factors {
        if let Some((arg, name)) = extract_trig_pow2(ctx, *factor) {
            match name {
                "sin" if sin2_arg.is_none() => sin2_arg = Some(arg),
                "cos" if cos2_arg.is_none() => cos2_arg = Some(arg),
                _ => other_factors.push(*factor), // Duplicate or already matched
            }
        } else {
            other_factors.push(*factor);
        }
    }

    // Must have exactly one sin^2 and one cos^2 with same argument
    let sin_arg = sin2_arg?;
    let cos_arg = cos2_arg?;

    // Verify same argument
    if crate::ordering::compare_expr(ctx, sin_arg, cos_arg) != Ordering::Equal {
        return None;
    }

    // Build the coefficient expression from remaining factors
    let coeff = if other_factors.is_empty() {
        ctx.num(1)
    } else if other_factors.len() == 1 {
        other_factors[0]
    } else {
        // Build product of remaining factors
        let mut result = other_factors[0];
        for &f in &other_factors[1..] {
            result = ctx.add(Expr::Mul(result, f));
        }
        result
    };

    Some((coeff, sin_arg))
}

/// Check if a coefficient expression equals 3.
/// Uses simplification: coeff - 3 == 0
fn coeff_is_three(ctx: &mut cas_ast::Context, coeff: ExprId) -> bool {
    // Fast path: direct number check
    if let Expr::Number(n) = ctx.get(coeff) {
        return n.is_integer() && *n.numer() == 3.into();
    }

    // Use as_rational_const for expressions like 6/2
    if let Some(val) = crate::helpers::as_rational_const(ctx, coeff) {
        return val == num_rational::BigRational::from_integer(3.into());
    }

    false
}

define_rule!(
    TrigHiddenCubicIdentityRule,
    "Hidden Cubic Trig Identity",
    None,
    crate::phase::PhaseMask::TRANSFORM,
    |ctx, expr| {
        // Only match Add nodes (sums)
        if !matches!(ctx.get(expr), Expr::Add(_, _)) {
            return None;
        }

        // Flatten the sum to get all terms
        let mut terms = Vec::new();
        crate::helpers::flatten_add(ctx, expr, &mut terms);

        // We need exactly 3 terms for the pattern
        if terms.len() != 3 {
            return None;
        }

        // Try to find: sin^6(arg), cos^6(arg), coeff*sin^2(arg)*cos^2(arg)
        let mut sin6_arg: Option<ExprId> = None;
        let mut cos6_arg: Option<ExprId> = None;
        let mut sin2cos2_info: Option<(ExprId, ExprId)> = None; // (coeff, arg)
        let mut sin6_idx: Option<usize> = None;
        let mut cos6_idx: Option<usize> = None;
        let mut sin2cos2_idx: Option<usize> = None;

        for (i, &term) in terms.iter().enumerate() {
            // Try to match sin^6 or cos^6
            if let Some((arg, name)) = extract_trig_pow6(ctx, term) {
                match name {
                    "sin" if sin6_arg.is_none() => {
                        sin6_arg = Some(arg);
                        sin6_idx = Some(i);
                    }
                    "cos" if cos6_arg.is_none() => {
                        cos6_arg = Some(arg);
                        cos6_idx = Some(i);
                    }
                    _ => {} // Already matched or duplicate
                }
            }
        }

        // Find the sin^2*cos^2 term (the remaining one)
        for (i, &term) in terms.iter().enumerate() {
            if Some(i) == sin6_idx || Some(i) == cos6_idx {
                continue;
            }

            if let Some((coeff, arg)) = extract_sin2_cos2_product(ctx, term) {
                sin2cos2_info = Some((coeff, arg));
                sin2cos2_idx = Some(i);
                break;
            }
        }

        // Verify we found all three pieces
        let sin6_a = sin6_arg?;
        let cos6_a = cos6_arg?;
        let (coeff, sin2cos2_a) = sin2cos2_info?;

        // Ensure we used all three terms (no extras)
        if sin6_idx.is_none() || cos6_idx.is_none() || sin2cos2_idx.is_none() {
            return None;
        }

        // Verify all arguments are the same
        if crate::ordering::compare_expr(ctx, sin6_a, cos6_a) != Ordering::Equal {
            return None;
        }
        if crate::ordering::compare_expr(ctx, sin6_a, sin2cos2_a) != Ordering::Equal {
            return None;
        }

        // Verify coefficient is 3
        if !coeff_is_three(ctx, coeff) {
            return None;
        }

        // All conditions met! Rewrite to (sin^2(arg) + cos^2(arg))^3
        let arg = sin6_a;
        let two = ctx.num(2);
        let three = ctx.num(3);

        let sin_arg = ctx.add(Expr::Function("sin".to_string(), vec![arg]));
        let cos_arg = ctx.add(Expr::Function("cos".to_string(), vec![arg]));
        let sin2 = ctx.add(Expr::Pow(sin_arg, two));
        let two_again = ctx.num(2);
        let cos2 = ctx.add(Expr::Pow(cos_arg, two_again));
        let sum = ctx.add(Expr::Add(sin2, cos2));
        let result = ctx.add(Expr::Pow(sum, three));

        Some(Rewrite {
            new_expr: result,
            description: "sin⁶(x) + cos⁶(x) + 3sin²(x)cos²(x) = (sin²(x) + cos²(x))³".to_string(),
            before_local: None,
            after_local: None,
            assumption_events: Default::default(),
            required_conditions: vec![],
            poly_proof: None,
        })
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
                        let sin_cos = smart_mul(ctx, sin_x, cos_x);
                        let new_expr = smart_mul(ctx, two, sin_cos);
                        return Some(Rewrite {
                            new_expr,
                            description: "sin(2x) -> 2sin(x)cos(x)".to_string(),
                            before_local: None,
                            after_local: None,
                            assumption_events: Default::default(),
                            required_conditions: vec![],
                            poly_proof: None,
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
                            before_local: None,
                            after_local: None,
                            assumption_events: Default::default(),
                            required_conditions: vec![],
                            poly_proof: None,
                        });
                    }
                    _ => {}
                }
            }
        }
    }
    None
});

// Triple Angle Shortcut Rule: sin(3x) → 3sin(x) - 4sin³(x), cos(3x) → 4cos³(x) - 3cos(x)
// This is a performance optimization to avoid recursive expansion via double-angle rules.
// Reduces ~23 rewrites to ~3-5 for triple angle expressions.
define_rule!(TripleAngleRule, "Triple Angle Identity", |ctx, expr| {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if args.len() == 1 {
            // Check if arg is 3*x or x*3
            if let Some(inner_var) = extract_triple_angle_arg(ctx, args[0]) {
                match name.as_str() {
                    "sin" => {
                        // sin(3x) → 3sin(x) - 4sin³(x)
                        let three = ctx.num(3);
                        let four = ctx.num(4);
                        let exp_three = ctx.num(3); // Separate for Pow exponent
                        let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![inner_var]));

                        // 3*sin(x)
                        let term1 = smart_mul(ctx, three, sin_x);

                        // sin³(x) = sin(x)^3
                        let sin_cubed = ctx.add(Expr::Pow(sin_x, exp_three));
                        // 4*sin³(x)
                        let term2 = smart_mul(ctx, four, sin_cubed);

                        // 3sin(x) - 4sin³(x)
                        let new_expr = ctx.add(Expr::Sub(term1, term2));
                        return Some(Rewrite {
                            new_expr,
                            description: "sin(3x) → 3sin(x) - 4sin³(x)".to_string(),
                            before_local: None,
                            after_local: None,
                            assumption_events: Default::default(),
                            required_conditions: vec![],
                            poly_proof: None,
                        });
                    }
                    "cos" => {
                        // cos(3x) → 4cos³(x) - 3cos(x)
                        let three = ctx.num(3);
                        let four = ctx.num(4);
                        let exp_three = ctx.num(3); // Separate for Pow exponent
                        let cos_x = ctx.add(Expr::Function("cos".to_string(), vec![inner_var]));

                        // cos³(x) = cos(x)^3
                        let cos_cubed = ctx.add(Expr::Pow(cos_x, exp_three));
                        // 4*cos³(x)
                        let term1 = smart_mul(ctx, four, cos_cubed);

                        // 3*cos(x)
                        let term2 = smart_mul(ctx, three, cos_x);

                        // 4cos³(x) - 3cos(x)
                        let new_expr = ctx.add(Expr::Sub(term1, term2));
                        return Some(Rewrite {
                            new_expr,
                            description: "cos(3x) → 4cos³(x) - 3cos(x)".to_string(),
                            before_local: None,
                            after_local: None,
                            assumption_events: Default::default(),
                            required_conditions: vec![],
                            poly_proof: None,
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
                    let term_nm1 = smart_mul(ctx, n_minus_1_expr, x_val);

                    // sin(nx) = sin((n-1)x)cos(x) + cos((n-1)x)sin(x)
                    // cos(nx) = cos((n-1)x)cos(x) - sin((n-1)x)sin(x)

                    let sin_nm1 = ctx.add(Expr::Function("sin".to_string(), vec![term_nm1]));
                    let cos_nm1 = ctx.add(Expr::Function("cos".to_string(), vec![term_nm1]));
                    let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![x_val]));
                    let cos_x = ctx.add(Expr::Function("cos".to_string(), vec![x_val]));

                    if name == "sin" {
                        let t1 = smart_mul(ctx, sin_nm1, cos_x);
                        let t2 = smart_mul(ctx, cos_nm1, sin_x);
                        let new_expr = ctx.add(Expr::Add(t1, t2));
                        return Some(Rewrite {
                            new_expr,
                            description: format!("sin({}x) expansion", n_val),
                            before_local: None,
                            after_local: None,
                            assumption_events: Default::default(),
                            required_conditions: vec![],
                            poly_proof: None,
                        });
                    } else {
                        // cos
                        let t1 = smart_mul(ctx, cos_nm1, cos_x);
                        let t2 = smart_mul(ctx, sin_nm1, sin_x);
                        let new_expr = ctx.add(Expr::Sub(t1, t2));
                        return Some(Rewrite {
                            new_expr,
                            description: format!("cos({}x) expansion", n_val),
                            before_local: None,
                            after_local: None,
                            assumption_events: Default::default(),
                            required_conditions: vec![],
                            poly_proof: None,
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
    importance: crate::step::ImportanceLevel::Low,
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

                                let half_n = n / num_rational::BigRational::from_integer(2.into());

                                if half_n.is_one() {
                                    return Some(Rewrite::new(base_term).desc("cos^2(x) -> 1 - sin^2(x)"));
                                } else {
                                    let half_n_expr = ctx.add(Expr::Number(half_n));
                                    let new_expr = ctx.add(Expr::Pow(base_term, half_n_expr));
                                    return Some(Rewrite {
                                        new_expr,
                                        description: "cos^2k(x) -> (1 - sin^2(x))^k".to_string(),
                                        before_local: None,
                                        after_local: None,
                                        assumption_events: Default::default(),
            required_conditions: vec![],
            poly_proof: None,
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
    // Use the new data-driven EvaluateTrigTableRule instead of deprecated EvaluateTrigRule
    simplifier.add_rule(Box::new(super::evaluation::EvaluateTrigTableRule));
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    simplifier.add_rule(Box::new(SecTanPythagoreanRule));
    simplifier.add_rule(Box::new(CscCotPythagoreanRule));

    // Hidden Cubic Identity: sin^6 + cos^6 + 3sin^2cos^2 = (sin^2+cos^2)^3
    // Should run in TRANSFORM phase before power expansions
    simplifier.add_rule(Box::new(TrigHiddenCubicIdentityRule));

    simplifier.add_rule(Box::new(AngleIdentityRule));
    simplifier.add_rule(Box::new(TanToSinCosRule));
    simplifier.add_rule(Box::new(DoubleAngleRule));
    simplifier.add_rule(Box::new(TripleAngleRule)); // Shortcut: sin(3x), cos(3x)
    simplifier.add_rule(Box::new(RecursiveTrigExpansionRule));

    // DISABLED: ProductToSumRule conflicts with AngleIdentityRule creating infinite loops
    // ProductToSumRule: 2*sin(a)*cos(b) → sin(a+b) + sin(a-b)
    // AngleIdentityRule: sin(a+b) → sin(a)*cos(b) + cos(a)*sin(b)
    // When combined, they create cycles. Use manually for specific cases like Dirichlet kernel.
    // simplifier.add_rule(Box::new(ProductToSumRule));

    // DISABLED: Conflicts with Pythagorean identity rules causing infinite loops
    // This rule converts cos²(x) → 1-sin²(x) which interacts badly with:
    // - Pythagorean identities (sec²-tan²=1)
    // - Reciprocal trig canonicalization
    // Creating transformation cycles like: sec² → 1/cos² → 1/(1-sin²) → ...
    // See: debug_sec_tan.rs test and GitHub issue #X
    // simplifier.add_rule(Box::new(CanonicalizeTrigSquareRule));

    // Pythagorean Identity simplification: k - k*sin² → k*cos², k - k*cos² → k*sin²
    // This rule was extracted from CancelCommonFactorsRule for pedagogical clarity
    simplifier.add_rule(Box::new(super::pythagorean::TrigPythagoreanSimplifyRule));

    simplifier.add_rule(Box::new(AngleConsistencyRule));

    // Phase shift: sin(x + π/2) → cos(x), cos(x + π/2) → -sin(x), etc.
    simplifier.add_rule(Box::new(TrigPhaseShiftRule));

    // Fourth power difference: sin⁴(x) - cos⁴(x) → sin²(x) - cos²(x)
    simplifier.add_rule(Box::new(super::pythagorean::TrigEvenPowerDifferenceRule));

    // Cotangent half-angle difference: cot(u/2) - cot(u) = 1/sin(u)
    simplifier.add_rule(Box::new(CotHalfAngleDifferenceRule));
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
                    before_local: None,
                    after_local: None,
                    assumption_events: Default::default(),
                    required_conditions: vec![],
                    poly_proof: None,
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
                    let term = smart_mul(ctx, sin_half, cos_half);
                    return smart_mul(ctx, two, term);
                }
                "cos" => {
                    // cos(A) -> 2cos^2(A/2) - 1
                    let two = ctx.num(2);
                    let one = ctx.num(1);
                    let cos_half = ctx.add(Expr::Function("cos".to_string(), vec![small_angle]));
                    let cos_sq = ctx.add(Expr::Pow(cos_half, two));
                    let term = smart_mul(ctx, two, cos_sq);
                    return ctx.add(Expr::Sub(term, one));
                }
                "tan" => {
                    // tan(A) -> 2tan(A/2) / (1 - tan^2(A/2))
                    let two = ctx.num(2);
                    let one = ctx.num(1);
                    let tan_half = ctx.add(Expr::Function("tan".to_string(), vec![small_angle]));
                    let num = smart_mul(ctx, two, tan_half);

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
                smart_mul(ctx, nl, nr)
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

// =============================================================================
// PRODUCT-TO-SUM IDENTITIES
// =============================================================================
// 2*sin(a)*cos(b) → sin(a+b) + sin(a-b)
// 2*cos(a)*sin(b) → sin(a+b) - sin(a-b)
// 2*cos(a)*cos(b) → cos(a+b) + cos(a-b)
// 2*sin(a)*sin(b) → cos(a-b) - cos(a+b)

define_rule!(ProductToSumRule, "Product to Sum", |ctx, expr| {
    // Look for patterns like: 2 * sin(a) * cos(b)
    // or: sin(a) * cos(b) * 2
    let expr_data = ctx.get(expr).clone();

    if let Expr::Mul(_, _) = expr_data {
        let mut factors = Vec::new();
        crate::helpers::flatten_mul(ctx, expr, &mut factors);

        // Find the coefficient 2 and two trig functions
        let mut has_two = false;
        let mut two_idx = None;
        let mut trig_funcs: Vec<(usize, String, ExprId)> = Vec::new();

        for (i, &factor) in factors.iter().enumerate() {
            match ctx.get(factor) {
                Expr::Number(n) => {
                    if *n == num_rational::BigRational::from_integer(2.into()) {
                        has_two = true;
                        two_idx = Some(i);
                    }
                }
                Expr::Function(name, args) => {
                    if args.len() == 1 && (name == "sin" || name == "cos") {
                        trig_funcs.push((i, name.clone(), args[0]));
                    }
                }
                _ => {}
            }
        }

        // Need exactly: coefficient 2 and exactly 2 trig functions
        if has_two && trig_funcs.len() == 2 {
            let (idx1, name1, arg1) = &trig_funcs[0];
            let (idx2, name2, arg2) = &trig_funcs[1];

            // Build remaining factors (everything except 2 and the two trigs)
            let mut remaining: Vec<ExprId> = Vec::new();
            for (i, &factor) in factors.iter().enumerate() {
                if Some(i) != two_idx && i != *idx1 && i != *idx2 {
                    remaining.push(factor);
                }
            }

            // Determine which identity to apply
            let (new_expr, description) = match (name1.as_str(), name2.as_str()) {
                ("sin", "cos") => {
                    // 2*sin(a)*cos(b) → sin(a+b) + sin(a-b)
                    let sum_arg = ctx.add(Expr::Add(*arg1, *arg2));
                    let diff_arg = ctx.add(Expr::Sub(*arg1, *arg2));
                    let sin_sum = ctx.add(Expr::Function("sin".to_string(), vec![sum_arg]));
                    let sin_diff = ctx.add(Expr::Function("sin".to_string(), vec![diff_arg]));
                    let result = ctx.add(Expr::Add(sin_sum, sin_diff));
                    (result, "2·sin(a)·cos(b) → sin(a+b) + sin(a-b)")
                }
                ("cos", "sin") => {
                    // 2*cos(a)*sin(b) → sin(a+b) - sin(a-b)
                    let sum_arg = ctx.add(Expr::Add(*arg1, *arg2));
                    let diff_arg = ctx.add(Expr::Sub(*arg1, *arg2));
                    let sin_sum = ctx.add(Expr::Function("sin".to_string(), vec![sum_arg]));
                    let sin_diff = ctx.add(Expr::Function("sin".to_string(), vec![diff_arg]));
                    let result = ctx.add(Expr::Sub(sin_sum, sin_diff));
                    (result, "2·cos(a)·sin(b) → sin(a+b) - sin(a-b)")
                }
                ("cos", "cos") => {
                    // 2*cos(a)*cos(b) → cos(a+b) + cos(a-b)
                    let sum_arg = ctx.add(Expr::Add(*arg1, *arg2));
                    let diff_arg = ctx.add(Expr::Sub(*arg1, *arg2));
                    let cos_sum = ctx.add(Expr::Function("cos".to_string(), vec![sum_arg]));
                    let cos_diff = ctx.add(Expr::Function("cos".to_string(), vec![diff_arg]));
                    let result = ctx.add(Expr::Add(cos_sum, cos_diff));
                    (result, "2·cos(a)·cos(b) → cos(a+b) + cos(a-b)")
                }
                ("sin", "sin") => {
                    // 2*sin(a)*sin(b) → cos(a-b) - cos(a+b)
                    let sum_arg = ctx.add(Expr::Add(*arg1, *arg2));
                    let diff_arg = ctx.add(Expr::Sub(*arg1, *arg2));
                    let cos_sum = ctx.add(Expr::Function("cos".to_string(), vec![sum_arg]));
                    let cos_diff = ctx.add(Expr::Function("cos".to_string(), vec![diff_arg]));
                    let result = ctx.add(Expr::Sub(cos_diff, cos_sum));
                    (result, "2·sin(a)·sin(b) → cos(a-b) - cos(a+b)")
                }
                _ => return None,
            };

            // If there are remaining factors, multiply them back
            let final_expr = if remaining.is_empty() {
                new_expr
            } else {
                let mut result = new_expr;
                for factor in remaining {
                    result = smart_mul(ctx, result, factor);
                }
                result
            };

            return Some(Rewrite {
                new_expr: final_expr,
                description: description.to_string(),
                before_local: None,
                after_local: None,
                assumption_events: Default::default(),
                required_conditions: vec![],
                poly_proof: None,
            });
        }
    }
    None
});
// ============================================================================
// Trig Phase Shift Rule
// ============================================================================
// sin(x + π/2) → cos(x)
// sin(x - π/2) → -cos(x)
// sin(x + π) → -sin(x)
// cos(x + π/2) → -sin(x)
// cos(x - π/2) → sin(x)
// cos(x + π) → -cos(x)
//
// Also handles canonical form: sin((2*x + π)/2) where arg = (2*x + π)/2

define_rule!(TrigPhaseShiftRule, "Trig Phase Shift", |ctx, expr| {
    let expr_data = ctx.get(expr).clone();

    if let Expr::Function(name, args) = expr_data {
        if args.len() != 1 {
            return None;
        }

        let is_sin = name == "sin";
        let is_cos = name == "cos";
        if !is_sin && !is_cos {
            return None;
        }

        let arg = args[0];

        // Try to extract (base_term, pi_multiple) where arg = base_term + pi_multiple * π/2
        let (base_term, pi_multiple) = extract_phase_shift(ctx, arg)?;

        if pi_multiple == 0 {
            return None;
        }

        // Normalize k to 0..3 range (mod 4)
        let k = ((pi_multiple % 4) + 4) % 4;

        // Apply phase shift
        // sin(x + k*π/2): k=0→sin(x), k=1→cos(x), k=2→-sin(x), k=3→-cos(x)
        // cos(x + k*π/2): k=0→cos(x), k=1→-sin(x), k=2→-cos(x), k=3→sin(x)
        let (new_func, negate) = if is_sin {
            match k {
                0 => ("sin", false),
                1 => ("cos", false),
                2 => ("sin", true),
                3 => ("cos", true),
                _ => return None,
            }
        } else {
            match k {
                0 => ("cos", false),
                1 => ("sin", true),
                2 => ("cos", true),
                3 => ("sin", false),
                _ => return None,
            }
        };

        let new_trig = ctx.add(Expr::Function(new_func.to_string(), vec![base_term]));
        let new_expr = if negate {
            ctx.add(Expr::Neg(new_trig))
        } else {
            new_trig
        };

        let shift_desc = match pi_multiple {
            1 => "π/2",
            -1 => "-π/2",
            2 => "π",
            -2 => "-π",
            3 => "3π/2",
            -3 => "-3π/2",
            _ => "kπ/2",
        };

        return Some(Rewrite {
            new_expr,
            description: format!("{}(x + {}) phase shift", name, shift_desc),
            before_local: None,
            after_local: None,
            assumption_events: Default::default(),
            required_conditions: vec![],
            poly_proof: None,
        });
    }

    None
});

/// Extract (base_term, k) from arg such that arg = base_term + k*π/2
/// Handles multiple canonical forms:
/// - Add(x, π/2) → (x, 1)
/// - Sub(x, π/2) → (x, -1)  
/// - Div(Add(n*x, k*π), m) → (x, k*2/m) if m divides k*2
fn extract_phase_shift(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<(ExprId, i32)> {
    // Form 1: Div((coeff*x + k*pi), denom) - the canonical form!
    // Example: (2*x + pi)/2 means x + pi/2, so k=1
    if let Expr::Div(num, den) = ctx.get(expr) {
        let num = *num;
        let den = *den;

        // Get denominator value
        let denom_val: i32 = if let Expr::Number(n) = ctx.get(den) {
            if n.is_integer() {
                n.to_integer().try_into().ok()?
            } else {
                return None;
            }
        } else {
            return None;
        };

        // Check if numerator is Add/Sub
        if let Expr::Add(l, r) = ctx.get(num).clone() {
            // Check both terms for π
            // Form: (base + k*pi)/denom where we want k/denom = m/2 for some integer m

            // Check right term for π (most common: 2*x + pi)
            if is_pi(ctx, r) {
                // pi/denom * 2 = shift in units of pi/2
                // For denom=2, shift = 1 (one pi/2)
                let k = 2 / denom_val; // Only works if denom divides 2
                if 2 % denom_val != 0 {
                    // Not a clean pi/2 multiple
                } else {
                    let base = ctx.add(Expr::Div(l, den));
                    return Some((base, k));
                }
            }

            // Check left term for π (less common: pi + 2*x)
            if is_pi(ctx, l) {
                let k = 2 / denom_val;
                if 2 % denom_val != 0 {
                    // Not a clean pi/2 multiple
                } else {
                    let base = ctx.add(Expr::Div(r, den));
                    return Some((base, k));
                }
            }

            // Also check for k*pi form using extract_pi_coefficient
            if let Some(pi_coeff) = extract_pi_coefficient(ctx, r) {
                let k_times_2 = 2 * pi_coeff;
                if k_times_2 % denom_val == 0 {
                    let k = k_times_2 / denom_val;
                    let base = ctx.add(Expr::Div(l, den));
                    return Some((base, k));
                }
            }

            if let Some(pi_coeff) = extract_pi_coefficient(ctx, l) {
                let k_times_2 = 2 * pi_coeff;
                if k_times_2 % denom_val == 0 {
                    let k = k_times_2 / denom_val;
                    let base = ctx.add(Expr::Div(r, den));
                    return Some((base, k));
                }
            }
        }
    }

    // Form 1b: Mul(1/n, Add(coeff*x, k*pi)) - the canonical form for (a + b)/n!
    // Example: (2*x + pi)/2 becomes Mul(1/2, Add(2*x, pi)); shift = 1
    if let Expr::Mul(coeff_id, inner) = ctx.get(expr).clone() {
        // Check if coeff is 1/n (a rational with numerator 1)
        if let Expr::Number(coeff) = ctx.get(coeff_id) {
            if coeff.numer() == &num_bigint::BigInt::from(1) && !coeff.denom().is_one() {
                let denom_val: i32 = coeff.denom().try_into().ok().unwrap_or(0);
                if denom_val > 0 {
                    // Check if inner is Add containing pi
                    if let Expr::Add(l, r) = ctx.get(inner).clone() {
                        // Check right term for pi
                        if is_pi(ctx, r) {
                            let k = 2 / denom_val;
                            if 2 % denom_val == 0 {
                                // base = l / denom = l * (1/denom) = coeff * l
                                let base = ctx.add(Expr::Mul(coeff_id, l));
                                return Some((base, k));
                            }
                        }

                        // Check left term for pi
                        if is_pi(ctx, l) {
                            let k = 2 / denom_val;
                            if 2 % denom_val == 0 {
                                let base = ctx.add(Expr::Mul(coeff_id, r));
                                return Some((base, k));
                            }
                        }

                        // Check for k*pi form
                        if let Some(pi_coeff) = extract_pi_coefficient(ctx, r) {
                            let k_times_2 = 2 * pi_coeff;
                            if k_times_2 % denom_val == 0 {
                                let k = k_times_2 / denom_val;
                                let base = ctx.add(Expr::Mul(coeff_id, l));
                                return Some((base, k));
                            }
                        }
                    }
                }
            }
        }
    }

    // Form 2: Add(x, k*π/2)
    if let Expr::Add(l, r) = ctx.get(expr) {
        if let Some(k) = extract_pi_half_multiple(ctx, *r) {
            return Some((*l, k));
        }
        if let Some(k) = extract_pi_half_multiple(ctx, *l) {
            return Some((*r, k));
        }
    }

    // Form 3: Sub(x, k*π/2)
    if let Expr::Sub(l, r) = ctx.get(expr) {
        if let Some(k) = extract_pi_half_multiple(ctx, *r) {
            return Some((*l, -k));
        }
    }

    None
}

/// Extract the coefficient of π from an expression.
/// - π → 1
/// - k*π → k  
/// - π*k → k
fn extract_pi_coefficient(ctx: &cas_ast::Context, expr: ExprId) -> Option<i32> {
    // Check for π alone
    if is_pi(ctx, expr) {
        return Some(1);
    }

    // Check for Mul(k, π) or Mul(π, k)
    if let Expr::Mul(l, r) = ctx.get(expr) {
        if is_pi(ctx, *r) {
            if let Expr::Number(n) = ctx.get(*l) {
                if n.is_integer() {
                    return n.to_integer().try_into().ok();
                }
            }
        }
        if is_pi(ctx, *l) {
            if let Expr::Number(n) = ctx.get(*r) {
                if n.is_integer() {
                    return n.to_integer().try_into().ok();
                }
            }
        }
    }

    None
}

/// Extract k from expressions like k*π/2, π/2, π, 3π/2, etc.
/// Returns Some(k) if the expression equals k*π/2 for integer k.
fn extract_pi_half_multiple(ctx: &cas_ast::Context, expr: ExprId) -> Option<i32> {
    // Check for π/2 (k=1)
    if is_pi_over_n(ctx, expr, 2) {
        return Some(1);
    }

    // Check for π (k=2)
    if is_pi(ctx, expr) {
        return Some(2);
    }

    // Check for Mul(k, π/2) or Mul(π/2, k)
    if let Expr::Mul(l, r) = ctx.get(expr) {
        // Check Mul(Number, π/2)
        if let Expr::Number(n) = ctx.get(*l) {
            if is_pi_over_n(ctx, *r, 2) && n.is_integer() {
                if let Ok(k) = n.to_integer().try_into() {
                    return Some(k);
                }
            }
            // Check Mul(Number, π) means k = 2*number
            if is_pi(ctx, *r) && n.is_integer() {
                if let Ok(k_half) = n.to_integer().try_into() {
                    let k: i32 = k_half;
                    return Some(k * 2);
                }
            }
        }
        // Check Mul(π/2, Number)
        if let Expr::Number(n) = ctx.get(*r) {
            if is_pi_over_n(ctx, *l, 2) && n.is_integer() {
                if let Ok(k) = n.to_integer().try_into() {
                    return Some(k);
                }
            }
            if is_pi(ctx, *l) && n.is_integer() {
                if let Ok(k_half) = n.to_integer().try_into() {
                    let k: i32 = k_half;
                    return Some(k * 2);
                }
            }
        }
    }

    // Check for Div(k*π, 2)
    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Expr::Number(d) = ctx.get(*den) {
            if d.is_integer() && *d == num_rational::BigRational::from_integer(2.into()) {
                // Check if numerator is k*π or just π
                if is_pi(ctx, *num) {
                    return Some(1);
                }
                if let Expr::Mul(l, r) = ctx.get(*num) {
                    if let Expr::Number(n) = ctx.get(*l) {
                        if is_pi(ctx, *r) && n.is_integer() {
                            if let Ok(k) = n.to_integer().try_into() {
                                return Some(k);
                            }
                        }
                    }
                    if let Expr::Number(n) = ctx.get(*r) {
                        if is_pi(ctx, *l) && n.is_integer() {
                            if let Ok(k) = n.to_integer().try_into() {
                                return Some(k);
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

// ============================================================================
// Cotangent Half-Angle Difference Rule
// ============================================================================
// cot(u/2) - cot(u) = 1/sin(u) = csc(u)
//
// This is a common precalculus identity that avoids term explosion from
// brute-force expansion via cot→cos/sin + double angle formulas.
//
// Pattern matching:
// - cot(u/2) - cot(u) → 1/sin(u)
// - k*cot(u/2) - k*cot(u) → k/sin(u)
// - Works on n-ary sums via flatten_add

/// Helper: Check if arg represents u/2 and return u
/// Supports: Mul(1/2, u), Div(u, 2)
fn is_half_angle(ctx: &cas_ast::Context, arg: ExprId) -> Option<ExprId> {
    match ctx.get(arg) {
        Expr::Mul(coef, inner) => {
            if let Expr::Number(n) = ctx.get(*coef) {
                if *n == num_rational::BigRational::new(1.into(), 2.into()) {
                    return Some(*inner);
                }
            }
            // Check reversed order: inner * 1/2
            if let Expr::Number(n) = ctx.get(*inner) {
                if *n == num_rational::BigRational::new(1.into(), 2.into()) {
                    return Some(*coef);
                }
            }
        }
        Expr::Div(numer, denom) => {
            if let Expr::Number(d) = ctx.get(*denom) {
                if *d == num_rational::BigRational::from_integer(2.into()) {
                    return Some(*numer);
                }
            }
        }
        _ => {}
    }
    None
}

/// Helper: Extract coefficient and cot argument from a term
/// Returns (coefficient_opt, cot_arg, is_positive)
/// coefficient_opt=None means coefficient is implicitly 1
/// is_positive=false means the term is negated (represents -coeff*cot(arg))
fn extract_cot_term(
    ctx: &cas_ast::Context,
    term: ExprId,
) -> Option<(Option<ExprId>, ExprId, bool)> {
    let term_data = ctx.get(term);

    // Check for Neg(...)
    let (inner_term, is_positive) = match term_data {
        Expr::Neg(inner) => (*inner, false),
        _ => (term, true),
    };

    let inner_data = ctx.get(inner_term);

    // Check for cot(arg) directly
    if let Expr::Function(name, args) = inner_data {
        if name == "cot" && args.len() == 1 {
            // Coefficient is implicitly 1
            return Some((None, args[0], is_positive));
        }
    }

    // Check for Mul(coef, cot(arg))
    if let Expr::Mul(l, r) = inner_data {
        // Check if right is cot
        if let Expr::Function(name, args) = ctx.get(*r) {
            if name == "cot" && args.len() == 1 {
                return Some((Some(*l), args[0], is_positive));
            }
        }
        // Check if left is cot
        if let Expr::Function(name, args) = ctx.get(*l) {
            if name == "cot" && args.len() == 1 {
                return Some((Some(*r), args[0], is_positive));
            }
        }
    }

    None
}

define_rule!(
    CotHalfAngleDifferenceRule,
    "Cotangent Half-Angle Difference",
    |ctx, expr| {
        // Only match Add or Sub at top level
        let expr_data = ctx.get(expr).clone();

        // Normalize Sub to Add(a, Neg(b)) conceptually by handling both
        let terms: Vec<ExprId> = match expr_data {
            Expr::Add(_, _) => {
                let mut ts = Vec::new();
                crate::helpers::flatten_add(ctx, expr, &mut ts);
                ts
            }
            Expr::Sub(l, r) => {
                // Treat as [l, -r]
                vec![l, r] // We'll handle the sign in matching
            }
            _ => return None,
        };

        if terms.len() < 2 {
            return None;
        }

        // For Sub, we have special handling
        let is_explicit_sub = matches!(ctx.get(expr), Expr::Sub(_, _));

        // Collect cot terms: (index, coeff, arg, is_positive_in_original)
        struct CotTerm {
            index: usize,
            coeff: Option<ExprId>, // None means coefficient is 1
            arg: ExprId,
            is_positive: bool,
        }

        let mut cot_terms = Vec::new();

        if is_explicit_sub {
            // For Sub(a, b): a is positive, b is effectively negative
            if let Some((c1, arg1, _)) = extract_cot_term(ctx, terms[0]) {
                cot_terms.push(CotTerm {
                    index: 0,
                    coeff: c1,
                    arg: arg1,
                    is_positive: true,
                });
            }
            if let Some((c2, arg2, sign2)) = extract_cot_term(ctx, terms[1]) {
                // In Sub(a, b), b appears with flipped sign
                cot_terms.push(CotTerm {
                    index: 1,
                    coeff: c2,
                    arg: arg2,
                    is_positive: !sign2, // Flip because it's subtracted
                });
            }
        } else {
            // For Add chain
            for (i, &term) in terms.iter().enumerate() {
                if let Some((c, arg, is_pos)) = extract_cot_term(ctx, term) {
                    cot_terms.push(CotTerm {
                        index: i,
                        coeff: c,
                        arg,
                        is_positive: is_pos,
                    });
                }
            }
        }

        // Look for pairs: cot(u/2) and cot(u) with opposite signs
        for i in 0..cot_terms.len() {
            for j in 0..cot_terms.len() {
                if i == j {
                    continue;
                }

                let t_half = &cot_terms[i];
                let t_full = &cot_terms[j];

                // Check if t_half.arg is half of t_full.arg
                if let Some(full_angle) = is_half_angle(ctx, t_half.arg) {
                    // Verify full_angle == t_full.arg
                    if crate::ordering::compare_expr(ctx, full_angle, t_full.arg) != Ordering::Equal
                    {
                        continue;
                    }

                    // Check that coefficients match (or both are 1)
                    let coeffs_match = match (&t_half.coeff, &t_full.coeff) {
                        (None, None) => true,
                        (Some(c1), Some(c2)) => {
                            crate::ordering::compare_expr(ctx, *c1, *c2) == Ordering::Equal
                        }
                        _ => false,
                    };

                    if !coeffs_match {
                        continue;
                    }

                    // Check signs: cot(u/2) positive AND cot(u) negative = cot(u/2) - cot(u)
                    // OR cot(u/2) negative AND cot(u) positive = -cot(u/2) + cot(u) = -(cot(u/2) - cot(u))
                    if t_half.is_positive && !t_full.is_positive {
                        // cot(u/2) - cot(u) → 1/sin(u)
                        let one = ctx.num(1);
                        let sin_u = ctx.add(Expr::Function("sin".to_string(), vec![t_full.arg]));
                        let result = ctx.add(Expr::Div(one, sin_u));

                        // Apply coefficient if present
                        let final_result = if let Some(c) = t_half.coeff {
                            smart_mul(ctx, c, result)
                        } else {
                            result
                        };

                        // Reconstruct expression without the matched terms
                        if is_explicit_sub && terms.len() == 2 {
                            // Simple case: Sub(cot(u/2), cot(u)) → 1/sin(u)
                            return Some(
                                Rewrite::new(final_result).desc("cot(u/2) - cot(u) = 1/sin(u)"),
                            );
                        }

                        // N-ary case: rebuild sum without matched terms
                        let mut new_terms: Vec<ExprId> = Vec::new();
                        for (k, &term) in terms.iter().enumerate() {
                            if k != t_half.index && k != t_full.index {
                                new_terms.push(term);
                            }
                        }
                        new_terms.push(final_result);

                        let mut new_expr = new_terms[0];
                        for &term in new_terms.iter().skip(1) {
                            new_expr = ctx.add(Expr::Add(new_expr, term));
                        }

                        return Some(Rewrite {
                            new_expr,
                            description: "cot(u/2) - cot(u) = 1/sin(u)".to_string(),
                            before_local: None,
                            after_local: None,
                            assumption_events: Default::default(),
                            required_conditions: vec![],
                            poly_proof: None,
                        });
                    } else if !t_half.is_positive && t_full.is_positive {
                        // -cot(u/2) + cot(u) → -1/sin(u)
                        let one = ctx.num(1);
                        let sin_u = ctx.add(Expr::Function("sin".to_string(), vec![t_full.arg]));
                        let result = ctx.add(Expr::Div(one, sin_u));
                        let neg_result = ctx.add(Expr::Neg(result));

                        // Apply coefficient if present
                        let final_result = if let Some(c) = t_half.coeff {
                            smart_mul(ctx, c, neg_result)
                        } else {
                            neg_result
                        };

                        if is_explicit_sub && terms.len() == 2 {
                            return Some(
                                Rewrite::new(final_result).desc("-cot(u/2) + cot(u) = -1/sin(u)"),
                            );
                        }

                        // N-ary case
                        let mut new_terms: Vec<ExprId> = Vec::new();
                        for (k, &term) in terms.iter().enumerate() {
                            if k != t_half.index && k != t_full.index {
                                new_terms.push(term);
                            }
                        }
                        new_terms.push(final_result);

                        let mut new_expr = new_terms[0];
                        for &term in new_terms.iter().skip(1) {
                            new_expr = ctx.add(Expr::Add(new_expr, term));
                        }

                        return Some(Rewrite {
                            new_expr,
                            description: "-cot(u/2) + cot(u) = -1/sin(u)".to_string(),
                            before_local: None,
                            after_local: None,
                            assumption_events: Default::default(),
                            required_conditions: vec![],
                            poly_proof: None,
                        });
                    }
                }
            }
        }

        None
    }
);

#[cfg(test)]
mod cot_half_angle_tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::{Context, DisplayExpr};
    use cas_parser::parse;

    #[test]
    fn test_cot_half_angle_basic() {
        let mut ctx = Context::new();
        let rule = CotHalfAngleDifferenceRule;

        // cot(x/2) - cot(x) → 1/sin(x)
        let expr = parse("cot(x/2) - cot(x)", &mut ctx).unwrap();
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_some(), "Should match cot(x/2) - cot(x)");

        let result = rewrite.unwrap();
        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: result.new_expr
            }
        );
        assert!(
            result_str.contains("sin"),
            "Result should contain sin, got: {}",
            result_str
        );
    }

    #[test]
    fn test_cot_half_angle_no_match_different_args() {
        let mut ctx = Context::new();
        let rule = CotHalfAngleDifferenceRule;

        // cot(x/2) - cot(y) → no change (different args)
        let expr = parse("cot(x/2) - cot(y)", &mut ctx).unwrap();
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match cot(x/2) - cot(y)");
    }

    #[test]
    fn test_cot_half_angle_no_match_third() {
        let mut ctx = Context::new();
        let rule = CotHalfAngleDifferenceRule;

        // cot(x/3) - cot(x) → no change (not half-angle)
        let expr = parse("cot(x/3) - cot(x)", &mut ctx).unwrap();
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match cot(x/3) - cot(x)");
    }

    // =========================================================================
    // TrigHiddenCubicIdentityRule tests
    // =========================================================================

    #[test]
    fn test_hidden_cubic_basic() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)^6 + cos(x)^6 + 3*sin(x)^2*cos(x)^2", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "1");
    }

    #[test]
    fn test_hidden_cubic_permutation_cos_first() {
        let mut ctx = Context::new();
        // Different order: cos^6 first
        let expr = parse("cos(x)^6 + 3*cos(x)^2*sin(x)^2 + sin(x)^6", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "1");
    }

    #[test]
    fn test_hidden_cubic_coeff_product_first() {
        let mut ctx = Context::new();
        // Coefficient product first
        let expr = parse("3*sin(x)^2*cos(x)^2 + sin(x)^6 + cos(x)^6", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "1");
    }

    #[test]
    fn test_hidden_cubic_equivalent_coeff() {
        let mut ctx = Context::new();
        // Coefficient 6/2 = 3
        let expr = parse("sin(x)^6 + cos(x)^6 + (6/2)*sin(x)^2*cos(x)^2", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "1");
    }

    #[test]
    fn test_hidden_cubic_no_match_wrong_coeff() {
        let mut ctx = Context::new();
        // Wrong coefficient: 2 instead of 3
        let expr = parse("sin(x)^6 + cos(x)^6 + 2*sin(x)^2*cos(x)^2", &mut ctx).unwrap();

        let rule = TrigHiddenCubicIdentityRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match with coeff=2");
    }

    #[test]
    fn test_hidden_cubic_no_match_different_args() {
        let mut ctx = Context::new();
        // Different arguments: x vs y
        let expr = parse("sin(x)^6 + cos(y)^6 + 3*sin(x)^2*cos(y)^2", &mut ctx).unwrap();

        let rule = TrigHiddenCubicIdentityRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match with different args");
    }

    #[test]
    fn test_hidden_cubic_no_match_extra_terms() {
        let mut ctx = Context::new();
        // Extra term: should not match partially
        let expr = parse("sin(x)^6 + cos(x)^6 + 3*sin(x)^2*cos(x)^2 + 1", &mut ctx).unwrap();

        let rule = TrigHiddenCubicIdentityRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        // flatten_add will produce 4 terms, so rule should not match (requires exactly 3)
        assert!(rewrite.is_none(), "Should not match with extra terms");
    }
}
