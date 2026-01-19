use crate::define_rule;
use crate::helpers::{extract_double_angle_arg, extract_triple_angle_arg, is_pi, is_pi_over_n};
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Expr, ExprId};
use num_traits::{One, Zero};
use std::cmp::Ordering;

// =============================================================================
// SinCosIntegerPiRule: Pre-order evaluation of sin(n·π) and cos(n·π)
// =============================================================================
// sin(n·π) = 0 for any integer n
// cos(n·π) = (-1)^n for any integer n
//
// This rule runs BEFORE any expansion rules (TripleAngle, DoubleAngle, etc.)
// to avoid unnecessary polynomial expansion of expressions like sin(3π).
//
// Priority: 100 (higher than most rules to ensure pre-order evaluation)

pub struct SinCosIntegerPiRule;

impl crate::rule::Rule for SinCosIntegerPiRule {
    fn name(&self) -> &str {
        "Evaluate Trig at Integer Multiple of π"
    }

    fn priority(&self) -> i32 {
        100 // Run before expansion rules
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::helpers::extract_rational_pi_multiple;

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

            // Try to extract k from k·π
            if let Some(k) = extract_rational_pi_multiple(ctx, arg) {
                if k.is_integer() {
                    let n = k.to_integer();

                    if is_sin {
                        // sin(n·π) = 0 for any integer n
                        let zero = ctx.num(0);
                        return Some(Rewrite::new(zero).desc(format!("sin({}·π) = 0", n)));
                    } else {
                        // cos(n·π) = (-1)^n
                        // n even → 1, n odd → -1
                        let is_even = &n % 2 == num_bigint::BigInt::from(0);
                        let result = if is_even { ctx.num(1) } else { ctx.num(-1) };
                        let result_str = if is_even { "1" } else { "-1" };
                        return Some(
                            Rewrite::new(result).desc(format!("cos({}·π) = {}", n, result_str)),
                        );
                    }
                }
            }
        }

        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Function"])
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

// =============================================================================
// TrigOddEvenParityRule: sin(-u) = -sin(u), cos(-u) = cos(u), tan(-u) = -tan(u)
// =============================================================================
// sin, tan, csc, cot are ODD functions: f(-x) = -f(x)
// cos, sec are EVEN functions: f(-x) = f(x)
//
// This enables simplification of expressions like sin(pi/9)/sin(-pi/9) → -1

define_rule!(
    TrigOddEvenParityRule,
    "Trig Parity (Odd/Even)",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();

        if let Expr::Function(name, args) = expr_data {
            if args.len() != 1 {
                return None;
            }

            let arg = args[0];
            let arg_data = ctx.get(arg).clone();

            // Try to extract "negated form" of argument
            let negated_info: Option<(ExprId, Option<num_rational::BigRational>)> = match &arg_data
            {
                // Direct negation: Neg(u)
                Expr::Neg(inner) => Some((*inner, None)),

                // Multiplication by negative number
                Expr::Mul(a, b) => {
                    let a_data = ctx.get(*a).clone();
                    let b_data = ctx.get(*b).clone();

                    if let Expr::Number(n) = a_data {
                        if n < num_rational::BigRational::from_integer(0.into()) {
                            Some((*b, Some(-n)))
                        } else {
                            None
                        }
                    } else if let Expr::Number(n) = b_data {
                        if n < num_rational::BigRational::from_integer(0.into()) {
                            Some((*a, Some(-n)))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            };

            if let Some((base, opt_coeff)) = negated_info {
                // Build positive argument
                let positive_arg = if let Some(coeff) = opt_coeff {
                    if coeff == num_rational::BigRational::from_integer(1.into()) {
                        base
                    } else {
                        let c = ctx.add(Expr::Number(coeff));
                        ctx.add(Expr::Mul(c, base))
                    }
                } else {
                    base
                };

                match name.as_str() {
                    // ODD functions: f(-u) = -f(u)
                    "sin" | "tan" | "csc" | "cot" | "sinh" | "tanh" => {
                        let f_u = ctx.add(Expr::Function(name.clone(), vec![positive_arg]));
                        let neg_f_u = ctx.add(Expr::Neg(f_u));
                        return Some(
                            Rewrite::new(neg_f_u)
                                .desc(format!("{}(-u) = -{}(u) [odd function]", name, name)),
                        );
                    }
                    // EVEN functions: f(-u) = f(u)
                    "cos" | "sec" | "cosh" => {
                        let f_u = ctx.add(Expr::Function(name.clone(), vec![positive_arg]));
                        return Some(
                            Rewrite::new(f_u)
                                .desc(format!("{}(-u) = {}(u) [even function]", name, name)),
                        );
                    }
                    _ => {}
                }
            }
        }

        None
    }
);

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
                                return Some(Rewrite::new(zero).desc(format!("{}(0) = 0", name)));
                            }
                            "cos" => {
                                let one = ctx.num(1);
                                return Some(Rewrite::new(one).desc("cos(0) = 1"));
                            }
                            "arccos" => {
                                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                let two = ctx.num(2);
                                let new_expr = ctx.add(Expr::Div(pi, two));
                                return Some(Rewrite::new(new_expr).desc("arccos(0) = pi/2"));
                            }
                            _ => {}
                        }
                    } else if n.is_one() {
                        match name.as_str() {
                            "arcsin" => {
                                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                let two = ctx.num(2);
                                let new_expr = ctx.add(Expr::Div(pi, two));
                                return Some(Rewrite::new(new_expr).desc("arcsin(1) = pi/2"));
                            }
                            "arccos" => {
                                let zero = ctx.num(0);
                                return Some(Rewrite::new(zero).desc("arccos(1) = 0"));
                            }
                            "arctan" => {
                                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                let four = ctx.num(4);
                                let new_expr = ctx.add(Expr::Div(pi, four));
                                return Some(Rewrite::new(new_expr).desc("arctan(1) = pi/4"));
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
                                return Some(Rewrite::new(new_expr).desc("arcsin(1/2) = pi/6"));
                            }
                            "arccos" => {
                                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                let three = ctx.num(3);
                                let new_expr = ctx.add(Expr::Div(pi, three));
                                return Some(Rewrite::new(new_expr).desc("arccos(1/2) = pi/3"));
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
                            return Some(Rewrite::new(zero).desc(format!("{}(pi) = 0", name)));
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
                            return Some(Rewrite::new(new_expr).desc("sin(π/3) = √3/2"));
                        }
                        "cos" => {
                            // cos(π/3) = 1/2
                            let one = ctx.num(1);
                            let two = ctx.num(2);
                            let new_expr = ctx.add(Expr::Div(one, two));
                            return Some(Rewrite::new(new_expr).desc("cos(π/3) = 1/2"));
                        }
                        "tan" => {
                            // tan(π/3) = √3
                            let three = ctx.num(3);
                            let one = ctx.num(1);
                            let two = ctx.num(2);
                            let half_exp = ctx.add(Expr::Div(one, two));
                            let new_expr = ctx.add(Expr::Pow(three, half_exp));
                            return Some(Rewrite::new(new_expr).desc("tan(π/3) = √3"));
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
                            return Some(
                                Rewrite::new(new_expr).desc(format!("{}(π/4) = √2/2", name)),
                            );
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
                            return Some(Rewrite::new(new_expr).desc("sin(π/6) = 1/2"));
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
                            return Some(Rewrite::new(new_expr).desc("cos(π/6) = √3/2"));
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
                            return Some(Rewrite::new(new_expr).desc("tan(π/6) = 1/√3"));
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
                            return Some(Rewrite::new(new_expr).desc("sin(-x) = -sin(x)"));
                        }
                        "cos" => {
                            let new_expr = ctx.add(Expr::Function("cos".to_string(), vec![inner]));
                            return Some(Rewrite::new(new_expr).desc("cos(-x) = cos(x)"));
                        }
                        "tan" => {
                            let tan_inner = ctx.add(Expr::Function("tan".to_string(), vec![inner]));
                            let new_expr = ctx.add(Expr::Neg(tan_inner));
                            return Some(Rewrite::new(new_expr).desc("tan(-x) = -tan(x)"));
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
                                        return Some(
                                            Rewrite::new(ctx.num(0))
                                                .desc("Pythagorean Identity (empty)"),
                                        );
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

                                    return Some(Rewrite::new(new_expr).desc(description));
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

        // GUARD: Don't expand sin/cos if the argument has a large coefficient.
        // sin(n*x) with |n| > 2 should NOT be expanded because it leads to
        // exponential explosion: sin(16x) → sin(13x+3x) → ... huge tree.
        // This guard blocks the expansion at the source.
        if !parent_ctx.is_expand_mode() {
            if let Expr::Function(name, args) = ctx.get(expr) {
                if (name == "sin" || name == "cos" || name == "tan")
                    && args.len() == 1
                    && has_large_coefficient(ctx, args[0])
                {
                    return None;
                }
            }
        }

        // GUARD: Don't expand sin(a+b)/cos(a+b) if this function is part of sin²+cos²=1 pattern
        // The pattern marks are set by pre-scan before simplification
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_trig_square_protected(expr) {
                return None; // Skip expansion to preserve Pythagorean identity
            }
        }

        // GUARD: Centralized anti-worsen for large trig coefficients.
        // If we're inside sin(n*x) with |n| > 2, block all trig expansions.
        // This prevents exponential explosion from recursive angle decomposition.
        if parent_ctx.is_trig_large_coeff_protected() && !parent_ctx.is_expand_mode() {
            return None;
        }

        // GUARD: Anti-worsen for multiple angles.
        // Don't expand sin(a+b) or cos(a+b) if:
        // - Either a or b is already a multiple angle (n*x where |n| > 1)
        // - This would cause exponential expansion: sin(12x + 4x) → huge tree
        // Note: We allow sin(x + y) with distinct variables, only block multiples of same var
        if let Expr::Function(name, args) = ctx.get(expr) {
            if (name == "sin" || name == "cos") && args.len() == 1 {
                let inner = args[0];
                if let Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) = ctx.get(inner) {
                    // Check if either side is a multiple angle that would cause explosion
                    if is_multiple_angle(ctx, *lhs) || is_multiple_angle(ctx, *rhs) {
                        return None; // Block expansion - would cause exponential growth
                    }
                }
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
                            return Some(
                                Rewrite::new(new_expr)
                                    .desc("sin(a + b) -> sin(a)cos(b) + cos(a)sin(b)"),
                            );
                        } else if let Expr::Sub(lhs, rhs) = inner_data {
                            // sin(a - b) = sin(a)cos(b) - cos(a)sin(b)
                            let sin_a = ctx.add(Expr::Function("sin".to_string(), vec![lhs]));
                            let cos_b = ctx.add(Expr::Function("cos".to_string(), vec![rhs]));
                            let term1 = smart_mul(ctx, sin_a, cos_b);

                            let cos_a = ctx.add(Expr::Function("cos".to_string(), vec![lhs]));
                            let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![rhs]));
                            let term2 = smart_mul(ctx, cos_a, sin_b);

                            let new_expr = ctx.add(Expr::Sub(term1, term2));
                            return Some(
                                Rewrite::new(new_expr)
                                    .desc("sin(a - b) -> sin(a)cos(b) - cos(a)sin(b)"),
                            );
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
                                return Some(Rewrite::new(new_expr).desc(
                                    "sin((a + b)/c) -> sin(a/c)cos(b/c) + cos(a/c)sin(b/c)",
                                ));
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
                            return Some(
                                Rewrite::new(new_expr)
                                    .desc("cos(a + b) -> cos(a)cos(b) - sin(a)sin(b)"),
                            );
                        } else if let Expr::Sub(lhs, rhs) = inner_data {
                            // cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
                            let cos_a = ctx.add(Expr::Function("cos".to_string(), vec![lhs]));
                            let cos_b = ctx.add(Expr::Function("cos".to_string(), vec![rhs]));
                            let term1 = smart_mul(ctx, cos_a, cos_b);

                            let sin_a = ctx.add(Expr::Function("sin".to_string(), vec![lhs]));
                            let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![rhs]));
                            let term2 = smart_mul(ctx, sin_a, sin_b);

                            let new_expr = ctx.add(Expr::Add(term1, term2));
                            return Some(
                                Rewrite::new(new_expr)
                                    .desc("cos(a - b) -> cos(a)cos(b) + sin(a)sin(b)"),
                            );
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
                                return Some(Rewrite::new(new_expr).desc(
                                    "cos((a + b)/c) -> cos(a/c)cos(b/c) - sin(a/c)sin(b/c)",
                                ));
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

// =============================================================================
// TRIPLE TANGENT PRODUCT IDENTITY
// tan(u) · tan(π/3 - u) · tan(π/3 + u) = tan(3u)
// =============================================================================

/// Matches tan(u)·tan(π/3+u)·tan(π/3-u) and simplifies to tan(3u).
/// Must run BEFORE TanToSinCosRule to prevent expansion.
pub struct TanTripleProductRule;

impl crate::rule::Rule for TanTripleProductRule {
    fn name(&self) -> &str {
        "Triple Tangent Product (π/3)"
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Mul"])
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::TRANSFORM
    }

    fn soundness(&self) -> crate::rule::SoundnessLabel {
        // This rule introduces requires (cos ≠ 0) for the tangent definitions
        crate::rule::SoundnessLabel::EquivalenceUnderIntroducedRequires
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        use crate::helpers::{as_fn1, flatten_mul_chain};

        // Flatten multiplication to get factors
        let factors = flatten_mul_chain(ctx, expr);

        // We need at least 3 factors
        if factors.len() < 3 {
            return None;
        }

        // Extract all tan(arg) functions
        let mut tan_args: Vec<(ExprId, ExprId)> = Vec::new(); // (factor_id, arg)
        for &factor in &factors {
            if let Some(arg) = as_fn1(ctx, factor, "tan") {
                tan_args.push((factor, arg));
            }
        }

        // We need exactly 3 tan factors
        if tan_args.len() != 3 {
            return None;
        }

        // Try each argument as the potential "u"
        for i in 0..3 {
            let u = tan_args[i].1;
            let (j, k) = match i {
                0 => (1, 2),
                1 => (0, 2),
                2 => (0, 1),
                _ => unreachable!(),
            };

            let arg_j = tan_args[j].1;
            let arg_k = tan_args[k].1;

            // Check both orderings: (u+π/3, π/3-u) or (π/3-u, u+π/3)
            let match1 = is_u_plus_pi_over_3(ctx, arg_j, u) && is_pi_over_3_minus_u(ctx, arg_k, u);
            let match2 = is_pi_over_3_minus_u(ctx, arg_j, u) && is_u_plus_pi_over_3(ctx, arg_k, u);

            if match1 || match2 {
                // Build tan(3u)
                let three = ctx.num(3);
                let three_u = smart_mul(ctx, three, u);
                let tan_3u = ctx.add(Expr::Function("tan".to_string(), vec![three_u]));

                // If there are other factors beyond the 3 tans, multiply them
                let other_factors: Vec<ExprId> = factors
                    .iter()
                    .copied()
                    .filter(|&f| f != tan_args[0].0 && f != tan_args[1].0 && f != tan_args[2].0)
                    .collect();

                let result = if other_factors.is_empty() {
                    // Wrap in __hold to prevent expansion
                    ctx.add(Expr::Function("__hold".to_string(), vec![tan_3u]))
                } else {
                    // Multiply tan(3u) with other factors
                    let held_tan = ctx.add(Expr::Function("__hold".to_string(), vec![tan_3u]));
                    let mut product = held_tan;
                    for &f in &other_factors {
                        product = smart_mul(ctx, product, f);
                    }
                    product
                };

                // Build domain conditions: cos(u), cos(u+π/3), cos(π/3−u) ≠ 0
                // These are required for the tangent functions to be defined
                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                let three = ctx.num(3);
                let pi_over_3 = ctx.add(Expr::Div(pi, three));
                let u_plus_pi3 = ctx.add(Expr::Add(u, pi_over_3));
                let pi3_minus_u = ctx.add(Expr::Sub(pi_over_3, u));
                let cos_u = ctx.add(Expr::Function("cos".to_string(), vec![u]));
                let cos_u_plus = ctx.add(Expr::Function("cos".to_string(), vec![u_plus_pi3]));
                let cos_pi3_minus = ctx.add(Expr::Function("cos".to_string(), vec![pi3_minus_u]));

                // Format u for display in substeps
                let u_str = cas_ast::DisplayExpr {
                    context: ctx,
                    id: u,
                }
                .to_string();

                return Some(
                    Rewrite::new(result)
                        .desc("tan(u)·tan(π/3+u)·tan(π/3−u) = tan(3u)")
                        .substep(
                            "Normalizar argumentos",
                            vec![format!(
                                "π/3 − u se representa como −u + π/3 para comparar como u + const"
                            )],
                        )
                        .substep(
                            "Reconocer patrón",
                            vec![
                                format!("Sea u = {}", u_str),
                                format!("Factores: tan(u), tan(u + π/3), tan(π/3 − u)"),
                            ],
                        )
                        .substep(
                            "Aplicar identidad",
                            vec![format!("tan(u)·tan(u + π/3)·tan(π/3 − u) = tan(3u)")],
                        )
                        .requires(crate::implicit_domain::ImplicitCondition::NonZero(cos_u))
                        .requires(crate::implicit_domain::ImplicitCondition::NonZero(
                            cos_u_plus,
                        ))
                        .requires(crate::implicit_domain::ImplicitCondition::NonZero(
                            cos_pi3_minus,
                        )),
                );
            }
        }

        None
    }
}

/// Check if expr equals u + π/3 (or π/3 + u)
fn is_u_plus_pi_over_3(ctx: &cas_ast::Context, expr: ExprId, u: ExprId) -> bool {
    if let Expr::Add(l, r) = ctx.get(expr).clone() {
        // Case: u + π/3
        if crate::ordering::compare_expr(ctx, l, u) == std::cmp::Ordering::Equal {
            return is_pi_over_3(ctx, r);
        }
        // Case: π/3 + u
        if crate::ordering::compare_expr(ctx, r, u) == std::cmp::Ordering::Equal {
            return is_pi_over_3(ctx, l);
        }
    }
    false
}

/// Check if expr equals π/3 - u (or -u + π/3 in canonicalized form)
fn is_pi_over_3_minus_u(ctx: &cas_ast::Context, expr: ExprId, u: ExprId) -> bool {
    // Pattern 1: Sub(π/3, u)
    if let Expr::Sub(l, r) = ctx.get(expr).clone() {
        if is_pi_over_3(ctx, l)
            && crate::ordering::compare_expr(ctx, r, u) == std::cmp::Ordering::Equal
        {
            return true;
        }
    }
    // Pattern 2: Add(π/3, Neg(u)) or Add(Neg(u), π/3) - canonicalized subtraction
    if let Expr::Add(l, r) = ctx.get(expr).clone() {
        // Add(π/3, Neg(u))
        if is_pi_over_3(ctx, l) {
            if let Expr::Neg(inner) = ctx.get(r).clone() {
                if crate::ordering::compare_expr(ctx, inner, u) == std::cmp::Ordering::Equal {
                    return true;
                }
            }
        }
        // Add(Neg(u), π/3)
        if is_pi_over_3(ctx, r) {
            if let Expr::Neg(inner) = ctx.get(l).clone() {
                if crate::ordering::compare_expr(ctx, inner, u) == std::cmp::Ordering::Equal {
                    return true;
                }
            }
        }
    }
    false
}

/// Check if an expression is π/3 (i.e., Div(π, 3) or canonicalized Mul(1/3, π))
fn is_pi_over_3(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    // Pattern 1: Div(π, 3)
    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        if matches!(ctx.get(num), Expr::Constant(cas_ast::Constant::Pi)) {
            if let Expr::Number(n) = ctx.get(den) {
                if n.is_integer() && *n.numer() == 3.into() {
                    return true;
                }
            }
        }
    }

    // Pattern 2: Mul(Number(1/3), π) - canonicalized form from CanonicalizeDivRule
    if let Expr::Mul(l, r) = ctx.get(expr).clone() {
        // Check Mul(1/3, π)
        if let Expr::Number(n) = ctx.get(l) {
            if *n == num_rational::BigRational::new(1.into(), 3.into())
                && matches!(ctx.get(r), Expr::Constant(cas_ast::Constant::Pi))
            {
                return true;
            }
        }
        // Check Mul(π, 1/3)
        if let Expr::Number(n) = ctx.get(r) {
            if *n == num_rational::BigRational::new(1.into(), 3.into())
                && matches!(ctx.get(l), Expr::Constant(cas_ast::Constant::Pi))
            {
                return true;
            }
        }
    }

    false
}

/// Runtime check: is this tan() part of a tan(u)·tan(π/3+u)·tan(π/3-u) triple product?
/// This is called during rule application to prevent TanToSinCosRule from expanding
/// tan() nodes that will be handled by TanTripleProductRule.
fn is_part_of_tan_triple_product(
    ctx: &cas_ast::Context,
    tan_expr: ExprId,
    parent_ctx: &crate::parent_context::ParentContext,
) -> bool {
    // Verify this is actually a tan() function
    if !matches!(ctx.get(tan_expr), Expr::Function(name, args) if name == "tan" && args.len() == 1)
    {
        return false;
    }

    // Find the highest Mul ancestor in the chain
    // Ancestors are stored from furthest to closest: [great-grandparent, grandparent, parent]
    // We want to find the outermost Mul that contains this tan()
    let ancestors = parent_ctx.all_ancestors();

    // Find the first (earliest in list = highest in tree) Mul ancestor
    let mut mul_root: Option<ExprId> = None;
    for &ancestor in ancestors {
        if matches!(ctx.get(ancestor), Expr::Mul(_, _)) {
            mul_root = Some(ancestor);
            break; // Take the highest Mul (first in ancestor list)
        }
    }

    let Some(mul_root) = mul_root else {
        return false;
    };

    // Flatten the Mul to get all factors
    let mut factors = Vec::new();
    let mut stack = vec![mul_root];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Mul(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            _ => factors.push(id),
        }
    }

    // Collect tan() arguments
    let mut tan_args: Vec<ExprId> = Vec::new();
    for &factor in &factors {
        if let Expr::Function(name, args) = ctx.get(factor) {
            if name == "tan" && args.len() == 1 {
                tan_args.push(args[0]);
            }
        }
    }

    // Need exactly 3 tan() factors for triple product
    if tan_args.len() != 3 {
        return false;
    }

    // Check if they form the triple product pattern {u, u+π/3, π/3-u}
    for i in 0..3 {
        let u = tan_args[i];
        let others: Vec<_> = tan_args
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, &arg)| arg)
            .collect();

        let arg_j = others[0];
        let arg_k = others[1];

        // Check both orderings
        let match1 = is_u_plus_pi_over_3(ctx, arg_j, u) && is_pi_over_3_minus_u(ctx, arg_k, u);
        let match2 = is_pi_over_3_minus_u(ctx, arg_j, u) && is_u_plus_pi_over_3(ctx, arg_k, u);

        if match1 || match2 {
            return true;
        }
    }

    false
}

/// Check if an expression is a "multiple angle" pattern: n*x where n is integer > 1.
/// This is used to gate tan(n*x) → sin/cos expansion, which leads to complexity explosion
/// via triple-angle formulas.
fn is_multiple_angle(ctx: &cas_ast::Context, arg: ExprId) -> bool {
    use cas_ast::Expr;

    // Pattern: Mul(Number(n), x) or Mul(x, Number(n)) where n is integer > 1
    if let Expr::Mul(l, r) = ctx.get(arg) {
        // Check left side for integer > 1
        if let Expr::Number(n) = ctx.get(*l) {
            if n.is_integer() {
                let val = n.numer().clone();
                if val > 1.into() || val < (-1).into() {
                    return true;
                }
            }
        }
        // Check right side for integer > 1
        if let Expr::Number(n) = ctx.get(*r) {
            if n.is_integer() {
                let val = n.numer().clone();
                if val > 1.into() || val < (-1).into() {
                    return true;
                }
            }
        }
    }

    false
}

/// Check if an expression has a "large coefficient" pattern: n*x where |n| > 2.
/// This guards against exponential explosion in trig expansions.
/// sin(16*x) would trigger this, blocking sin(a+b) decomposition.
fn has_large_coefficient(ctx: &cas_ast::Context, arg: ExprId) -> bool {
    use cas_ast::Expr;

    // Pattern: Mul(Number(n), x) or Mul(x, Number(n)) where |n| > 2
    if let Expr::Mul(l, r) = ctx.get(arg) {
        let check_large = |id: ExprId| -> bool {
            if let Expr::Number(n) = ctx.get(id) {
                if n.is_integer() {
                    let val = n.numer().clone();
                    val > num_bigint::BigInt::from(2) || val < num_bigint::BigInt::from(-2)
                } else {
                    false
                }
            } else {
                false
            }
        };
        // Check both sides
        if check_large(*l) || check_large(*r) {
            return true;
        }
    }

    // Also check for Add/Sub patterns that contain multiples
    // This catches sin(13x + 3x) patterns
    if let Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) = ctx.get(arg) {
        if is_multiple_angle(ctx, *lhs) || is_multiple_angle(ctx, *rhs) {
            return true;
        }
    }

    false
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
            // Tan triple product protection: tan(u)·tan(π/3+u)·tan(π/3-u) = tan(3u)
            // Don't expand tan() if it's part of this pattern - let TanTripleProductRule handle it.
            if marks.is_tan_triple_product_protected(expr) {
                return None;
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

        // GUARD: Runtime check for triple product pattern.
        // If this tan() is inside a Mul that forms tan(u)·tan(π/3+u)·tan(π/3-u), don't expand.
        // This works even after ExprIds change from canonicalization because we check the
        // current structure, not pre-scanned marks.
        if is_part_of_tan_triple_product(ctx, expr, parent_ctx) {
            return None; // Let TanTripleProductRule handle it
        }

        // GUARD: Anti-worsen for multiple angles.
        // Don't expand tan(n*x) for integer n > 1, as it leads to explosive
        // triple-angle formulas: tan(3x) → (3sin(x) - 4sin³(x))/(4cos³(x) - 3cos(x))
        // This is almost never useful for simplification.
        let expr_data = ctx.get(expr).clone();
        if let Expr::Function(name, args) = &expr_data {
            if name == "tan" && args.len() == 1 {
                // GUARD: Don't expand tan(n*x) - causes complexity explosion
                if is_multiple_angle(ctx, args[0]) {
                    return None;
                }
                // GUARD: Don't expand tan at special angles that have known values
                // Let EvaluateTrigTableRule handle these instead
                if super::values::detect_special_angle(ctx, args[0]).is_some() {
                    return None;
                }
            }
        }

        // Original conversion logic
        if let Expr::Function(name, args) = expr_data {
            if name == "tan" && args.len() == 1 {
                // tan(x) -> sin(x) / cos(x)
                let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![args[0]]));
                let cos_x = ctx.add(Expr::Function("cos".to_string(), vec![args[0]]));
                let new_expr = ctx.add(Expr::Div(sin_x, cos_x));
                return Some(crate::rule::Rewrite::new(new_expr).desc("tan(x) -> sin(x)/cos(x)"));
            }
        }
        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Function"])
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        // Exclude PostCleanup to avoid cycle with TrigQuotientRule
        // TanToSinCos expands for algebra, TrigQuotient reconverts to canonical form
        // NOTE: CORE is included because some tests (e.g., test_tangent_sum) need tan→sin/cos expansion
        // TanTripleProductRule is registered BEFORE this rule and will handle triple product patterns
        crate::phase::PhaseMask::CORE
            | crate::phase::PhaseMask::TRANSFORM
            | crate::phase::PhaseMask::RATIONALIZE
    }
}

/// Convert trig quotients to their canonical function forms:
/// - sin(x)/cos(x) → tan(x)
/// - cos(x)/sin(x) → cot(x)
/// - 1/sin(x) → csc(x)
/// - 1/cos(x) → sec(x)
/// - 1/tan(x) → cot(x)
pub struct TrigQuotientRule;

impl crate::rule::Rule for TrigQuotientRule {
    fn name(&self) -> &str {
        "Trig Quotient"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use cas_ast::Expr;

        let expr_data = ctx.get(expr).clone();

        if let Expr::Div(num, den) = expr_data {
            // Extract function names and arguments from numerator and denominator
            let num_data = ctx.get(num).clone();
            let den_data = ctx.get(den).clone();

            // Pattern: sin(x)/cos(x) → tan(x)
            if let (Expr::Function(num_name, num_args), Expr::Function(den_name, den_args)) =
                (&num_data, &den_data)
            {
                if num_name == "sin"
                    && den_name == "cos"
                    && num_args.len() == 1
                    && den_args.len() == 1
                    && crate::ordering::compare_expr(ctx, num_args[0], den_args[0])
                        == std::cmp::Ordering::Equal
                {
                    let tan_x = ctx.add(Expr::Function("tan".to_string(), vec![num_args[0]]));
                    return Some(crate::rule::Rewrite::new(tan_x).desc("sin(x)/cos(x) → tan(x)"));
                }

                // Pattern: cos(x)/sin(x) → cot(x)
                if num_name == "cos"
                    && den_name == "sin"
                    && num_args.len() == 1
                    && den_args.len() == 1
                    && crate::ordering::compare_expr(ctx, num_args[0], den_args[0])
                        == std::cmp::Ordering::Equal
                {
                    let cot_x = ctx.add(Expr::Function("cot".to_string(), vec![num_args[0]]));
                    return Some(crate::rule::Rewrite::new(cot_x).desc("cos(x)/sin(x) → cot(x)"));
                }
            }

            // Pattern: 1/sin(x) → csc(x)
            if crate::helpers::is_one(ctx, num) {
                if let Expr::Function(den_name, den_args) = &den_data {
                    if den_name == "sin" && den_args.len() == 1 {
                        let csc_x = ctx.add(Expr::Function("csc".to_string(), vec![den_args[0]]));
                        return Some(crate::rule::Rewrite::new(csc_x).desc("1/sin(x) → csc(x)"));
                    }
                    if den_name == "cos" && den_args.len() == 1 {
                        let sec_x = ctx.add(Expr::Function("sec".to_string(), vec![den_args[0]]));
                        return Some(crate::rule::Rewrite::new(sec_x).desc("1/cos(x) → sec(x)"));
                    }
                    if den_name == "tan" && den_args.len() == 1 {
                        let cot_x = ctx.add(Expr::Function("cot".to_string(), vec![den_args[0]]));
                        return Some(crate::rule::Rewrite::new(cot_x).desc("1/tan(x) → cot(x)"));
                    }
                }
            }
        }

        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Div"])
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        // Only run in PostCleanup to avoid cycle with TanToSinCosRule
        crate::phase::PhaseMask::POST
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
                            return Some(Rewrite::new(ctx.num(1)).desc("sec²(x) - tan²(x) = 1"));
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
                            return Some(Rewrite::new(ctx.num(1)).desc("csc²(x) - cot²(x) = 1"));
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

        Some(
            Rewrite::new(result).desc("sin⁶(x) + cos⁶(x) + 3sin²(x)cos²(x) = (sin²(x) + cos²(x))³"),
        )
    }
);

// =============================================================================
// SUM-TO-PRODUCT QUOTIENT RULE
// (sin(A)+sin(B))/(cos(A)+cos(B)) → sin((A+B)/2)/cos((A+B)/2)
// =============================================================================

/// Extract the argument from a trig function: sin(arg) → Some(arg), else None
fn extract_trig_arg(ctx: &cas_ast::Context, id: ExprId, fn_name: &str) -> Option<ExprId> {
    if let Expr::Function(name, args) = ctx.get(id) {
        if name == fn_name && args.len() == 1 {
            return Some(args[0]);
        }
    }
    None
}

/// Extract two trig function args from a 2-term sum: sin(A) + sin(B) → Some((A, B))
/// Uses flatten_add for robustness against nested Add structures
fn extract_trig_two_term_sum(
    ctx: &cas_ast::Context,
    expr: ExprId,
    fn_name: &str,
) -> Option<(ExprId, ExprId)> {
    use crate::helpers::flatten_add;

    let mut terms = Vec::new();
    flatten_add(ctx, expr, &mut terms);

    // Must have exactly 2 terms (both same trig function)
    if terms.len() != 2 {
        return None;
    }

    // Check both are the target function (sin or cos)
    let arg1 = extract_trig_arg(ctx, terms[0], fn_name)?;
    let arg2 = extract_trig_arg(ctx, terms[1], fn_name)?;

    Some((arg1, arg2))
}

/// Extract two trig function args from a 2-term difference: sin(A) - sin(B) → Some((A, B))
/// Handles both Sub(sin(A), sin(B)) and Add(sin(A), Neg(sin(B)))
fn extract_trig_two_term_diff(
    ctx: &cas_ast::Context,
    expr: ExprId,
    fn_name: &str,
) -> Option<(ExprId, ExprId)> {
    // Pattern 1: Sub(sin(A), sin(B))
    if let Expr::Sub(l, r) = ctx.get(expr) {
        let arg1 = extract_trig_arg(ctx, *l, fn_name)?;
        let arg2 = extract_trig_arg(ctx, *r, fn_name)?;
        return Some((arg1, arg2));
    }

    // Pattern 2: Add(sin(A), Neg(sin(B))) - normalized form
    if let Expr::Add(l, r) = ctx.get(expr) {
        // Check if one is Neg(sin(B))
        if let Expr::Neg(inner) = ctx.get(*r) {
            let arg1 = extract_trig_arg(ctx, *l, fn_name)?;
            let arg2 = extract_trig_arg(ctx, *inner, fn_name)?;
            return Some((arg1, arg2));
        }
        if let Expr::Neg(inner) = ctx.get(*l) {
            let arg1 = extract_trig_arg(ctx, *r, fn_name)?;
            let arg2 = extract_trig_arg(ctx, *inner, fn_name)?;
            // Note: sin(A) - sin(B) vs -sin(B) + sin(A) = sin(A) - sin(B)
            return Some((arg1, arg2));
        }
    }

    None
}

/// Check if two pairs of args match as multisets: {A,B} == {C,D}
fn args_match_as_multiset(
    ctx: &cas_ast::Context,
    a1: ExprId,
    a2: ExprId,
    b1: ExprId,
    b2: ExprId,
) -> bool {
    use crate::ordering::compare_expr;
    use std::cmp::Ordering;

    // Direct match: (A,B) == (C,D)
    let direct = compare_expr(ctx, a1, b1) == Ordering::Equal
        && compare_expr(ctx, a2, b2) == Ordering::Equal;

    // Crossed match: (A,B) == (D,C)
    let crossed = compare_expr(ctx, a1, b2) == Ordering::Equal
        && compare_expr(ctx, a2, b1) == Ordering::Equal;

    direct || crossed
}

/// Build half_diff = (A-B)/2 preserving the order of A and B.
/// Use this for sin((A-B)/2) where the sign matters.
/// Pre-simplifies the difference to produce cleaner output.
fn build_half_diff(ctx: &mut cas_ast::Context, a: ExprId, b: ExprId) -> ExprId {
    let diff = ctx.add(Expr::Sub(a, b));
    // Pre-simplify the difference (e.g., 5x - 3x → 2x)
    let diff_simplified = crate::collect::collect(ctx, diff);
    let two = ctx.num(2);
    let result = ctx.add(Expr::Div(diff_simplified, two));
    // Try to simplify the division (e.g., 2x/2 → x)
    simplify_numeric_div(ctx, result)
}

/// Build canonical half_diff = (A-B)/2 with consistent ordering.
/// Since cos((A-B)/2) == cos((B-A)/2), we use canonical order to ensure
/// numerator and denominator produce identical expressions for cancellation.
/// Use this for cos((A-B)/2) where the sign doesn't matter.
fn build_canonical_half_diff(ctx: &mut cas_ast::Context, a: ExprId, b: ExprId) -> ExprId {
    use crate::ordering::compare_expr;
    use std::cmp::Ordering;

    // Use canonical order: if A > B, swap to (B-A)/2
    // This ensures consistent expression for cos(half_diff) in num and den
    let (first, second) = if compare_expr(ctx, a, b) == Ordering::Greater {
        (b, a)
    } else {
        (a, b)
    };

    let diff = ctx.add(Expr::Sub(first, second));
    // Pre-simplify the difference (e.g., x - 3x → -2x)
    let diff_simplified = crate::collect::collect(ctx, diff);
    let two = ctx.num(2);
    let result = ctx.add(Expr::Div(diff_simplified, two));
    // Try to simplify the division (e.g., -2x/2 → -x)
    simplify_numeric_div(ctx, result)
}

/// Normalize an expression for even functions like cos.
/// For even functions: f(-x) = f(x), so we can strip the negation.
/// Returns the unwrapped inner expression if input is Neg(inner), else returns input.
fn normalize_for_even_fn(ctx: &cas_ast::Context, expr: ExprId) -> ExprId {
    use num_bigint::BigInt;
    use num_rational::BigRational;

    // If expr is Neg(inner), return inner
    if let Expr::Neg(inner) = ctx.get(expr) {
        return *inner;
    }
    // Also handle Mul(-1, x) or Mul(x, -1)
    if let Expr::Mul(l, r) = ctx.get(expr) {
        let minus_one = BigRational::from_integer(BigInt::from(-1));
        if let Expr::Number(n) = ctx.get(*l) {
            if n == &minus_one {
                return *r;
            }
        }
        if let Expr::Number(n) = ctx.get(*r) {
            if n == &minus_one {
                return *l;
            }
        }
    }
    expr
}

/// Build avg = (A+B)/2, pre-simplifying sum for cleaner output
/// This eliminates the need for a separate "Combine Like Terms" step
fn build_avg(ctx: &mut cas_ast::Context, a: ExprId, b: ExprId) -> ExprId {
    let sum = ctx.add(Expr::Add(a, b));
    // Pre-simplify the sum (e.g., x + 3x → 4x)
    let sum_simplified = crate::collect::collect(ctx, sum);
    let two = ctx.num(2);
    let result = ctx.add(Expr::Div(sum_simplified, two));
    // Try to simplify the division (e.g., 4x/2 → 2x)
    simplify_numeric_div(ctx, result)
}

/// Try to simplify a division when numerator has a coefficient divisible by denominator
/// e.g., 4x/2 → 2x, -2x/2 → -x
fn simplify_numeric_div(ctx: &mut cas_ast::Context, expr: ExprId) -> ExprId {
    use crate::helpers::as_i64;

    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        // Check if denominator is a small integer
        if let Some(den_val) = as_i64(ctx, den) {
            if den_val == 0 {
                return expr; // Avoid division by zero
            }

            // Check if numerator is a product k*x where k is divisible by den
            if let Expr::Mul(l, r) = ctx.get(num).clone() {
                if let Some(coeff) = as_i64(ctx, l) {
                    if coeff % den_val == 0 {
                        let new_coeff = coeff / den_val;
                        if new_coeff == 1 {
                            return r; // 2x/2 → x
                        } else if new_coeff == -1 {
                            return ctx.add(Expr::Neg(r)); // -2x/2 → -x
                        } else {
                            let new_coeff_expr = ctx.num(new_coeff);
                            return ctx.add(Expr::Mul(new_coeff_expr, r)); // 4x/2 → 2x
                        }
                    }
                }
                if let Some(coeff) = as_i64(ctx, r) {
                    if coeff % den_val == 0 {
                        let new_coeff = coeff / den_val;
                        if new_coeff == 1 {
                            return l; // x*2/2 → x
                        } else if new_coeff == -1 {
                            return ctx.add(Expr::Neg(l)); // x*(-2)/2 → -x
                        } else {
                            let new_coeff_expr = ctx.num(new_coeff);
                            return ctx.add(Expr::Mul(l, new_coeff_expr)); // x*4/2 → x*2
                        }
                    }
                }
            }

            // Check if numerator is a plain number divisible by den
            if let Some(num_val) = as_i64(ctx, num) {
                if num_val % den_val == 0 {
                    return ctx.num(num_val / den_val); // 4/2 → 2
                }
            }
        }
    }
    expr
}

// SinCosSumQuotientRule: Handles two patterns:
// 1. (sin(A)+sin(B))/(cos(A)+cos(B)) → tan((A+B)/2)  [uses sin sum identity]
// 2. (sin(A)-sin(B))/(cos(A)+cos(B)) → tan((A-B)/2)  [uses sin diff identity]
//
// Sum-to-product identities:
//   sin(A) + sin(B) = 2·sin((A+B)/2)·cos((A-B)/2)
//   sin(A) - sin(B) = 2·cos((A+B)/2)·sin((A-B)/2)
//   cos(A) + cos(B) = 2·cos((A+B)/2)·cos((A-B)/2)
//
// For sum case: common factor is 2·cos((A-B)/2) → result is tan((A+B)/2)
// For diff case: common factor is 2·cos((A+B)/2) → result is tan((A-B)/2)
//
// This rule runs BEFORE TripleAngleRule to avoid polynomial explosion.
define_rule!(
    SinCosSumQuotientRule,
    "Sum-to-Product Quotient",
    |ctx, expr| {
        // Only match Div nodes
        let Expr::Div(num_id, den_id) = ctx.get(expr).clone() else {
            return None;
        };

        // Extract cos(C) + cos(D) from denominator (required for both cases)
        let (cos_c, cos_d) = extract_trig_two_term_sum(ctx, den_id, "cos")?;

        // Try both patterns for numerator
        enum NumeratorPattern {
            Sum { sin_a: ExprId, sin_b: ExprId },
            Diff { sin_a: ExprId, sin_b: ExprId },
        }

        let pattern = if let Some((sin_a, sin_b)) = extract_trig_two_term_sum(ctx, num_id, "sin") {
            // Pattern 1: sin(A) + sin(B)
            NumeratorPattern::Sum { sin_a, sin_b }
        } else if let Some((sin_a, sin_b)) = extract_trig_two_term_diff(ctx, num_id, "sin") {
            // Pattern 2: sin(A) - sin(B)
            NumeratorPattern::Diff { sin_a, sin_b }
        } else {
            return None;
        };

        // Extract the sin arguments
        let (sin_a, sin_b, is_diff) = match pattern {
            NumeratorPattern::Sum { sin_a, sin_b } => (sin_a, sin_b, false),
            NumeratorPattern::Diff { sin_a, sin_b } => (sin_a, sin_b, true),
        };

        // Verify {A,B} == {C,D} as multisets
        if !args_match_as_multiset(ctx, sin_a, sin_b, cos_c, cos_d) {
            return None;
        }

        // Build avg = (A+B)/2 (commutative, order doesn't matter)
        let avg = build_avg(ctx, sin_a, sin_b);

        // Normalize avg for even functions (cos)
        let avg_normalized = normalize_for_even_fn(ctx, avg);

        use crate::rule::ChainedRewrite;

        if is_diff {
            // DIFFERENCE CASE: sin(A) - sin(B) = 2·cos(avg)·sin(half_diff)
            // half_diff = (A-B)/2 - ORDER MATTERS for sin! Use build_half_diff.
            let half_diff = build_half_diff(ctx, sin_a, sin_b);
            // For cos(half_diff), we can normalize since cos is even
            let half_diff_for_cos = normalize_for_even_fn(ctx, half_diff);
            // DIFFERENCE CASE: sin(A) - sin(B) = 2·cos(avg)·sin(half_diff)
            // cos(A) + cos(B) = 2·cos(avg)·cos(half_diff)
            // Cancel 2·cos(avg) → sin(half_diff)/cos(half_diff) = tan(half_diff)
            // Build sin/cos quotient form for intermediate display

            let sin_half_diff = ctx.add(Expr::Function("sin".to_string(), vec![half_diff]));
            let cos_half_diff = ctx.add(Expr::Function("cos".to_string(), vec![half_diff_for_cos]));
            let quotient_result = ctx.add(Expr::Div(sin_half_diff, cos_half_diff));

            // Intermediate states
            let two = ctx.num(2);
            let cos_avg = ctx.add(Expr::Function("cos".to_string(), vec![avg_normalized]));
            let sin_half = ctx.add(Expr::Function("sin".to_string(), vec![half_diff]));

            // Intermediate numerator: 2·cos(avg)·sin(half_diff)
            let num_product = smart_mul(ctx, cos_avg, sin_half);
            let intermediate_num = smart_mul(ctx, two, num_product);

            // Intermediate denominator: 2·cos(avg)·cos(half_diff)
            let two_2 = ctx.num(2);
            let cos_avg_2 = ctx.add(Expr::Function("cos".to_string(), vec![avg_normalized]));
            let cos_half = ctx.add(Expr::Function("cos".to_string(), vec![half_diff_for_cos]));
            let den_product = smart_mul(ctx, cos_avg_2, cos_half);
            let intermediate_den = smart_mul(ctx, two_2, den_product);

            let state_after_step1 = ctx.add(Expr::Div(intermediate_num, den_id));
            let state_after_step2 = ctx.add(Expr::Div(intermediate_num, intermediate_den));

            let rewrite = Rewrite::new(state_after_step1)
                .desc("sin(A)−sin(B) = 2·cos((A+B)/2)·sin((A-B)/2)")
                .local(num_id, intermediate_num)
                .chain(
                    ChainedRewrite::new(state_after_step2)
                        .desc("cos(A)+cos(B) = 2·cos((A+B)/2)·cos((A-B)/2)")
                        .local(den_id, intermediate_den),
                )
                .chain(
                    ChainedRewrite::new(quotient_result)
                        .desc("Cancel common factors 2 and cos(avg)")
                        .local(state_after_step2, quotient_result),
                );

            Some(rewrite)
        } else {
            // SUM CASE: sin(A) + sin(B) = 2·sin(avg)·cos(half_diff)
            // cos(A) + cos(B) = 2·cos(avg)·cos(half_diff)
            // Cancel 2·cos(half_diff) → sin(avg)/cos(avg) = tan(avg)
            // For sum case, we use canonical half_diff since only cos uses it (even function)
            let half_diff = build_canonical_half_diff(ctx, sin_a, sin_b);
            let half_diff_normalized = normalize_for_even_fn(ctx, half_diff);

            // Final result: sin(avg)/cos(avg)
            let sin_avg = ctx.add(Expr::Function("sin".to_string(), vec![avg]));
            let cos_avg = ctx.add(Expr::Function("cos".to_string(), vec![avg]));
            let final_result = ctx.add(Expr::Div(sin_avg, cos_avg));

            // Intermediate states
            let two = ctx.num(2);
            let cos_half_diff = ctx.add(Expr::Function(
                "cos".to_string(),
                vec![half_diff_normalized],
            ));

            // Intermediate numerator: 2·sin(avg)·cos(half_diff)
            let sin_avg_for_num = ctx.add(Expr::Function("sin".to_string(), vec![avg]));
            let num_product = smart_mul(ctx, sin_avg_for_num, cos_half_diff);
            let intermediate_num = smart_mul(ctx, two, num_product);

            // Intermediate denominator: 2·cos(avg)·cos(half_diff)
            let two_2 = ctx.num(2);
            let cos_avg_for_den = ctx.add(Expr::Function("cos".to_string(), vec![avg]));
            let cos_half_diff_2 = ctx.add(Expr::Function(
                "cos".to_string(),
                vec![half_diff_normalized],
            ));
            let den_product = smart_mul(ctx, cos_avg_for_den, cos_half_diff_2);
            let intermediate_den = smart_mul(ctx, two_2, den_product);

            let state_after_step1 = ctx.add(Expr::Div(intermediate_num, den_id));
            let state_after_step2 = ctx.add(Expr::Div(intermediate_num, intermediate_den));

            let rewrite = Rewrite::new(state_after_step1)
                .desc("sin(A)+sin(B) = 2·sin((A+B)/2)·cos((A-B)/2)")
                .local(num_id, intermediate_num)
                .chain(
                    ChainedRewrite::new(state_after_step2)
                        .desc("cos(A)+cos(B) = 2·cos((A+B)/2)·cos((A-B)/2)")
                        .local(den_id, intermediate_den),
                )
                .chain(
                    ChainedRewrite::new(final_result)
                        .desc("Cancel common factors 2 and cos(half_diff)")
                        .local(state_after_step2, final_result),
                );

            Some(rewrite)
        }
    }
);

// =============================================================================
// STANDALONE SUM-TO-PRODUCT RULE
// sin(A)+sin(B) → 2·sin((A+B)/2)·cos((A-B)/2)
// sin(A)-sin(B) → 2·cos((A+B)/2)·sin((A-B)/2)
// cos(A)+cos(B) → 2·cos((A+B)/2)·cos((A-B)/2)
// cos(A)-cos(B) → -2·sin((A+B)/2)·sin((A-B)/2)
// =============================================================================
// This rule applies sum-to-product identities to standalone sums/differences
// of trig functions (not inside quotients handled by SinCosSumQuotientRule).
//
// GATING: Only apply when both arguments are rational multiples of π, ensuring
// the transformed expression can be evaluated via trig table lookup (π/4, π/6, etc.)
// This prevents unnecessary expansion of symbolic expressions like sin(a)+sin(b).
//
// MATCHERS: Uses semantic TrigSumMatch (unordered) and TrigDiffMatch (ordered)
// to ensure correct sign handling for difference identities.
define_rule!(
    TrigSumToProductRule,
    "Sum-to-Product Identity",
    |ctx, expr| {
        use crate::helpers::{extract_rational_pi_multiple, match_trig_diff, match_trig_sum};

        // Try all four patterns
        enum Pattern {
            SinSum { arg1: ExprId, arg2: ExprId },
            SinDiff { a: ExprId, b: ExprId }, // ordered!
            CosSum { arg1: ExprId, arg2: ExprId },
            CosDiff { a: ExprId, b: ExprId }, // ordered!
        }

        let pattern = if let Some(m) = match_trig_sum(ctx, expr, "sin") {
            Pattern::SinSum {
                arg1: m.arg1,
                arg2: m.arg2,
            }
        } else if let Some(m) = match_trig_diff(ctx, expr, "sin") {
            Pattern::SinDiff { a: m.a, b: m.b }
        } else if let Some(m) = match_trig_sum(ctx, expr, "cos") {
            Pattern::CosSum {
                arg1: m.arg1,
                arg2: m.arg2,
            }
        } else if let Some(m) = match_trig_diff(ctx, expr, "cos") {
            Pattern::CosDiff { a: m.a, b: m.b }
        } else {
            return None;
        };

        // Extract (A, B) and the function name
        let (arg_a, arg_b, is_diff, fn_name) = match pattern {
            Pattern::SinSum { arg1, arg2 } => (arg1, arg2, false, "sin"),
            Pattern::SinDiff { a, b } => (a, b, true, "sin"),
            Pattern::CosSum { arg1, arg2 } => (arg1, arg2, false, "cos"),
            Pattern::CosDiff { a, b } => (a, b, true, "cos"),
        };

        // GATING: Only apply when BOTH arguments are rational multiples of π
        // This ensures the result can be simplified via trig table lookup
        let pi_a = extract_rational_pi_multiple(ctx, arg_a);
        let pi_b = extract_rational_pi_multiple(ctx, arg_b);
        if pi_a.is_none() || pi_b.is_none() {
            return None; // Don't expand symbolic sums
        }

        // Build avg = (A+B)/2 and half_diff = (A-B)/2
        let avg = build_avg(ctx, arg_a, arg_b);
        let half_diff = build_half_diff(ctx, arg_a, arg_b);
        let two = ctx.num(2);

        let (result, desc) = match (fn_name, is_diff) {
            // sin(A) + sin(B) → 2·sin(avg)·cos(half_diff)
            ("sin", false) => {
                let sin_avg = ctx.add(Expr::Function("sin".to_string(), vec![avg]));
                let cos_half = ctx.add(Expr::Function("cos".to_string(), vec![half_diff]));
                let product = smart_mul(ctx, sin_avg, cos_half);
                let result = smart_mul(ctx, two, product);
                (result, "sin(A)+sin(B) = 2·sin((A+B)/2)·cos((A-B)/2)")
            }
            // sin(A) - sin(B) → 2·cos(avg)·sin(half_diff)
            // Note: half_diff preserves order (A-B)/2 for correct sign
            ("sin", true) => {
                let cos_avg = ctx.add(Expr::Function("cos".to_string(), vec![avg]));
                let sin_half = ctx.add(Expr::Function("sin".to_string(), vec![half_diff]));
                let product = smart_mul(ctx, cos_avg, sin_half);
                let result = smart_mul(ctx, two, product);
                (result, "sin(A)-sin(B) = 2·cos((A+B)/2)·sin((A-B)/2)")
            }
            // cos(A) + cos(B) → 2·cos(avg)·cos(half_diff)
            ("cos", false) => {
                // For cos, half_diff sign doesn't matter (even function)
                let half_diff_normalized = normalize_for_even_fn(ctx, half_diff);
                let cos_avg = ctx.add(Expr::Function("cos".to_string(), vec![avg]));
                let cos_half = ctx.add(Expr::Function(
                    "cos".to_string(),
                    vec![half_diff_normalized],
                ));
                let product = smart_mul(ctx, cos_avg, cos_half);
                let result = smart_mul(ctx, two, product);
                (result, "cos(A)+cos(B) = 2·cos((A+B)/2)·cos((A-B)/2)")
            }
            // cos(A) - cos(B) → -2·sin(avg)·sin(half_diff)
            ("cos", true) => {
                let sin_avg = ctx.add(Expr::Function("sin".to_string(), vec![avg]));
                let sin_half = ctx.add(Expr::Function("sin".to_string(), vec![half_diff]));
                let product = smart_mul(ctx, sin_avg, sin_half);
                let two_product = smart_mul(ctx, two, product);
                let result = ctx.add(Expr::Neg(two_product));
                (result, "cos(A)-cos(B) = -2·sin((A+B)/2)·sin((A-B)/2)")
            }
            _ => return None,
        };

        Some(Rewrite::new(result).desc(desc))
    }
);

// =============================================================================
// HALF-ANGLE TANGENT RULE
// (1 - cos(2x)) / sin(2x) → tan(x)
// sin(2x) / (1 + cos(2x)) → tan(x)
// =============================================================================
// These are half-angle tangent identities derived from:
//   1 - cos(2x) = 2·sin²(x)
//   1 + cos(2x) = 2·cos²(x)
//   sin(2x) = 2·sin(x)·cos(x)
//
// DOMAIN WARNING: This transformation can extend the domain:
// - Pattern 1: Original requires sin(2x) ≠ 0, but tan(x) only requires cos(x) ≠ 0
// - Pattern 2: Original requires 1+cos(2x) ≠ 0, but tan(x) only requires cos(x) ≠ 0
//
// To preserve soundness, we introduce requires for cos(x) ≠ 0 (for tan(x) to be defined)
// and inherit the original denominator ≠ 0 condition.
//
// Uses SoundnessLabel::EquivalenceUnderIntroducedRequires
struct HalfAngleTangentRule;

impl crate::rule::Rule for HalfAngleTangentRule {
    fn name(&self) -> &str {
        "Half-Angle Tangent Identity"
    }

    fn priority(&self) -> i32 {
        50 // Normal priority
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::helpers::extract_double_angle_arg;
        use crate::implicit_domain::ImplicitCondition;

        // Only match Div nodes
        let Expr::Div(num_id, den_id) = ctx.get(expr).clone() else {
            return None;
        };

        // Pattern 1: (1 - cos(2x)) / sin(2x) → tan(x)
        // Pattern 2: sin(2x) / (1 + cos(2x)) → tan(x)

        enum Pattern {
            OneMinusCosOverSin { x: ExprId, sin_2x: ExprId },
            SinOverOnePlusCos { x: ExprId, one_plus_cos_2x: ExprId },
        }

        let pattern = 'pattern: {
            // Try Pattern 1: (1 - cos(2x)) / sin(2x)
            // Numerator can be: Sub(1, cos(2x)) OR Add(1, Neg(cos(2x))) (canonicalized)

            // Helper to extract cos(2x) from either cos(2x) or Neg(cos(2x))
            let try_extract_cos_2x =
                |ctx: &cas_ast::Context, id: ExprId| -> Option<(ExprId, bool)> {
                    if let Expr::Function(name, args) = ctx.get(id) {
                        if name == "cos" && args.len() == 1 {
                            return extract_double_angle_arg(ctx, args[0]).map(|x| (x, false));
                        }
                    }
                    // Check for Neg(cos(2x))
                    if let Expr::Neg(inner) = ctx.get(id) {
                        if let Expr::Function(name, args) = ctx.get(*inner) {
                            if name == "cos" && args.len() == 1 {
                                return extract_double_angle_arg(ctx, args[0]).map(|x| (x, true));
                                // negated
                            }
                        }
                    }
                    None
                };

            // Check Sub(1, cos(2x))
            if let Expr::Sub(one_id, cos_id) = ctx.get(num_id) {
                if let Expr::Number(n) = ctx.get(*one_id) {
                    if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                        if let Some((x, false)) = try_extract_cos_2x(ctx, *cos_id) {
                            // Check if denominator is sin(2x) with same argument
                            if let Expr::Function(den_name, den_args) = ctx.get(den_id) {
                                if den_name == "sin" && den_args.len() == 1 {
                                    if let Some(x2) = extract_double_angle_arg(ctx, den_args[0]) {
                                        if crate::ordering::compare_expr(ctx, x, x2)
                                            == std::cmp::Ordering::Equal
                                        {
                                            break 'pattern Some(Pattern::OneMinusCosOverSin {
                                                x,
                                                sin_2x: den_id,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Check Add(1, Neg(cos(2x))) or Add(Neg(cos(2x)), 1) - canonicalized form
            if let Expr::Add(left, right) = ctx.get(num_id) {
                // Try left=1, right=Neg(cos)
                let try_order = |one: ExprId, neg_cos: ExprId| -> Option<ExprId> {
                    if let Expr::Number(n) = ctx.get(one) {
                        if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into())
                        {
                            if let Some((x, true)) = try_extract_cos_2x(ctx, neg_cos) {
                                return Some(x);
                            }
                        }
                    }
                    None
                };

                // Try both orders
                let x_opt = try_order(*left, *right).or_else(|| try_order(*right, *left));

                if let Some(x) = x_opt {
                    // Check if denominator is sin(2x) with same argument
                    if let Expr::Function(den_name, den_args) = ctx.get(den_id) {
                        if den_name == "sin" && den_args.len() == 1 {
                            if let Some(x2) = extract_double_angle_arg(ctx, den_args[0]) {
                                if crate::ordering::compare_expr(ctx, x, x2)
                                    == std::cmp::Ordering::Equal
                                {
                                    break 'pattern Some(Pattern::OneMinusCosOverSin {
                                        x,
                                        sin_2x: den_id,
                                    });
                                }
                            }
                        }
                    }
                }
            }

            // Try Pattern 2: sin(2x) / (1 + cos(2x))
            // Numerator: sin(2x)
            if let Expr::Function(name, args) = ctx.get(num_id) {
                if name == "sin" && args.len() == 1 {
                    if let Some(x) = extract_double_angle_arg(ctx, args[0]) {
                        // Denominator: 1 + cos(2x) or Add(1, cos(2x))
                        if let Expr::Add(left, right) = ctx.get(den_id) {
                            // Check both orders: 1 + cos(2x) or cos(2x) + 1
                            let (one_id, cos_id) = if matches!(ctx.get(*left), Expr::Number(n) if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()))
                            {
                                (*left, *right)
                            } else if matches!(ctx.get(*right), Expr::Number(n) if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()))
                            {
                                (*right, *left)
                            } else {
                                break 'pattern None;
                            };

                            // Verify one_id is 1
                            if let Expr::Number(n) = ctx.get(one_id) {
                                if n.is_integer()
                                    && *n == num_rational::BigRational::from_integer(1.into())
                                {
                                    // Check if cos_id is cos(2x) with same x
                                    if let Expr::Function(cos_name, cos_args) = ctx.get(cos_id) {
                                        if cos_name == "cos" && cos_args.len() == 1 {
                                            if let Some(x2) =
                                                extract_double_angle_arg(ctx, cos_args[0])
                                            {
                                                if crate::ordering::compare_expr(ctx, x, x2)
                                                    == std::cmp::Ordering::Equal
                                                {
                                                    break 'pattern Some(
                                                        Pattern::SinOverOnePlusCos {
                                                            x,
                                                            one_plus_cos_2x: den_id,
                                                        },
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            None
        }?;

        // Build tan(x)
        let (x, denom_expr, desc) = match pattern {
            Pattern::OneMinusCosOverSin { x, sin_2x } => {
                (x, sin_2x, "(1 - cos(2x))/sin(2x) = tan(x)")
            }
            Pattern::SinOverOnePlusCos { x, one_plus_cos_2x } => {
                (x, one_plus_cos_2x, "sin(2x)/(1 + cos(2x)) = tan(x)")
            }
        };

        let tan_x = ctx.add(Expr::Function("tan".to_string(), vec![x]));

        // Build cos(x) for the NonZero require
        let cos_x = ctx.add(Expr::Function("cos".to_string(), vec![x]));

        // Create rewrite with requires:
        // 1. Original denominator ≠ 0 (inherited from the division)
        // 2. cos(x) ≠ 0 (for tan(x) to be defined)
        let rewrite = Rewrite::new(tan_x)
            .desc(desc)
            .requires(ImplicitCondition::NonZero(denom_expr))
            .requires(ImplicitCondition::NonZero(cos_x));

        Some(rewrite)
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Div"])
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn soundness(&self) -> crate::rule::SoundnessLabel {
        crate::rule::SoundnessLabel::EquivalenceUnderIntroducedRequires
    }
}

define_rule!(
    DoubleAngleRule,
    "Double Angle Identity",
    |ctx, expr, parent_ctx| {
        // GUARD: Don't expand double angle inside a Div context
        // This prevents sin(2x)/cos(2x) from being "polinomized" to a worse form.
        // Expansion should only happen when it helps simplification, not in canonical quotients.
        if parent_ctx
            .has_ancestor_matching(ctx, |c, id| matches!(c.get(id), cas_ast::Expr::Div(_, _)))
        {
            return None;
        }

        if let Expr::Function(name, args) = ctx.get(expr) {
            if args.len() == 1 {
                // Check if arg is 2*x or x*2
                // We need to match "2 * x"
                if let Some(inner_var) = extract_double_angle_arg(ctx, args[0]) {
                    // GUARD: Anti-worsen for multiple angles.
                    // Don't expand sin(2*(8x)) = sin(16x) because the inner argument
                    // is already a multiple (8x). This would cause exponential recursion:
                    // sin(16x) → 2sin(8x)cos(8x) → 2·2sin(4x)cos(4x)·... = explosion
                    if is_multiple_angle(ctx, inner_var) {
                        return None;
                    }

                    match name.as_str() {
                        "sin" => {
                            // sin(2x) -> 2sin(x)cos(x)
                            let two = ctx.num(2);
                            let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![inner_var]));
                            let cos_x = ctx.add(Expr::Function("cos".to_string(), vec![inner_var]));
                            let sin_cos = smart_mul(ctx, sin_x, cos_x);
                            let new_expr = smart_mul(ctx, two, sin_cos);
                            return Some(Rewrite::new(new_expr).desc("sin(2x) -> 2sin(x)cos(x)"));
                        }
                        "cos" => {
                            // cos(2x) -> cos^2(x) - sin^2(x)
                            let two = ctx.num(2);
                            let cos_x = ctx.add(Expr::Function("cos".to_string(), vec![inner_var]));
                            let cos2 = ctx.add(Expr::Pow(cos_x, two));

                            let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![inner_var]));
                            let sin2 = ctx.add(Expr::Pow(sin_x, two));

                            let new_expr = ctx.add(Expr::Sub(cos2, sin2));
                            return Some(
                                Rewrite::new(new_expr).desc("cos(2x) -> cos^2(x) - sin^2(x)"),
                            );
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }
);

// Double Angle Contraction Rule: 2·sin(t)·cos(t) → sin(2t), cos²(t) - sin²(t) → cos(2t)
// This is the INVERSE of DoubleAngleRule - contracts expanded forms back to double angle.
// Essential for recognizing Weierstrass substitution identities.
struct DoubleAngleContractionRule;

impl crate::rule::Rule for DoubleAngleContractionRule {
    fn name(&self) -> &str {
        "Double Angle Contraction"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Pattern 1: 2·sin(t)·cos(t) → sin(2t)
        // Matches Mul(2, Mul(sin(t), cos(t))) or Mul(Mul(2, sin(t)), cos(t)) etc.
        if let Expr::Mul(l, r) = ctx.get(expr).clone() {
            if let Some((sin_arg, cos_arg)) = self.extract_two_sin_cos(ctx, l, r) {
                // Check if sin and cos have the same argument
                if crate::ordering::compare_expr(ctx, sin_arg, cos_arg) == std::cmp::Ordering::Equal
                {
                    // Build sin(2*t)
                    let two = ctx.num(2);
                    let double_arg = ctx.add(Expr::Mul(two, sin_arg));
                    let sin_2t = ctx.add(Expr::Function("sin".to_string(), vec![double_arg]));
                    return Some(Rewrite::new(sin_2t).desc("2·sin(t)·cos(t) = sin(2t)"));
                }
            }
        }

        // Pattern 2: cos²(t) - sin²(t) → cos(2t)
        if let Expr::Sub(l, r) = ctx.get(expr).clone() {
            if let Some((cos_arg, sin_arg)) = self.extract_cos2_minus_sin2(ctx, l, r) {
                // Check if cos² and sin² have the same argument
                if crate::ordering::compare_expr(ctx, cos_arg, sin_arg) == std::cmp::Ordering::Equal
                {
                    // Build cos(2*t)
                    let two = ctx.num(2);
                    let double_arg = ctx.add(Expr::Mul(two, cos_arg));
                    let cos_2t = ctx.add(Expr::Function("cos".to_string(), vec![double_arg]));
                    return Some(Rewrite::new(cos_2t).desc("cos²(t) - sin²(t) = cos(2t)"));
                }
            }
        }

        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Mul", "Sub"])
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

impl DoubleAngleContractionRule {
    /// Extract (sin_arg, cos_arg) from 2·sin(t)·cos(t) pattern
    fn extract_two_sin_cos(
        &self,
        ctx: &cas_ast::Context,
        l: ExprId,
        r: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        // Check all possible arrangements of 2, sin(t), cos(t)
        let two_rat = num_rational::BigRational::from_integer(2.into());

        // Case: Mul(2, Mul(sin, cos))
        if let Expr::Number(n) = ctx.get(l) {
            if *n == two_rat {
                if let Expr::Mul(a, b) = ctx.get(r) {
                    return self.extract_sin_cos_pair(ctx, *a, *b);
                }
            }
        }

        // Case: Mul(Mul(...), 2)
        if let Expr::Number(n) = ctx.get(r) {
            if *n == two_rat {
                if let Expr::Mul(a, b) = ctx.get(l) {
                    return self.extract_sin_cos_pair(ctx, *a, *b);
                }
            }
        }

        // Case: Mul(Mul(2, sin), cos) or Mul(Mul(2, cos), sin)
        if let Expr::Mul(inner_l, inner_r) = ctx.get(l) {
            if let Expr::Number(n) = ctx.get(*inner_l) {
                if *n == two_rat {
                    // inner_r is either sin or cos
                    return self.extract_trig_and_match(ctx, *inner_r, r);
                }
            }
            if let Expr::Number(n) = ctx.get(*inner_r) {
                if *n == two_rat {
                    return self.extract_trig_and_match(ctx, *inner_l, r);
                }
            }
        }

        // Case: Mul(sin, Mul(2, cos)) or similar
        if let Expr::Mul(inner_l, inner_r) = ctx.get(r) {
            if let Expr::Number(n) = ctx.get(*inner_l) {
                if *n == two_rat {
                    return self.extract_trig_and_match(ctx, *inner_r, l);
                }
            }
            if let Expr::Number(n) = ctx.get(*inner_r) {
                if *n == two_rat {
                    return self.extract_trig_and_match(ctx, *inner_l, l);
                }
            }
        }

        None
    }

    fn extract_sin_cos_pair(
        &self,
        ctx: &cas_ast::Context,
        a: ExprId,
        b: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        // Check if a is sin and b is cos, or vice versa
        if let Expr::Function(name_a, args_a) = ctx.get(a) {
            if let Expr::Function(name_b, args_b) = ctx.get(b) {
                if args_a.len() == 1 && args_b.len() == 1 {
                    if name_a == "sin" && name_b == "cos" {
                        return Some((args_a[0], args_b[0]));
                    }
                    if name_a == "cos" && name_b == "sin" {
                        return Some((args_b[0], args_a[0]));
                    }
                }
            }
        }
        None
    }

    fn extract_trig_and_match(
        &self,
        ctx: &cas_ast::Context,
        trig1: ExprId,
        trig2: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        if let Expr::Function(name1, args1) = ctx.get(trig1) {
            if let Expr::Function(name2, args2) = ctx.get(trig2) {
                if args1.len() == 1 && args2.len() == 1 {
                    if name1 == "sin" && name2 == "cos" {
                        return Some((args1[0], args2[0]));
                    }
                    if name1 == "cos" && name2 == "sin" {
                        return Some((args2[0], args1[0]));
                    }
                }
            }
        }
        None
    }

    /// Extract (cos_arg, sin_arg) from cos²(t) - sin²(t) pattern
    fn extract_cos2_minus_sin2(
        &self,
        ctx: &cas_ast::Context,
        l: ExprId,
        r: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        // l should be cos²(t), r should be sin²(t)
        let two_rat = num_rational::BigRational::from_integer(2.into());

        if let Expr::Pow(base_l, exp_l) = ctx.get(l) {
            if let Expr::Number(n) = ctx.get(*exp_l) {
                if *n == two_rat {
                    if let Expr::Function(name_l, args_l) = ctx.get(*base_l) {
                        if name_l == "cos" && args_l.len() == 1 {
                            // Check r is sin²
                            if let Expr::Pow(base_r, exp_r) = ctx.get(r) {
                                if let Expr::Number(m) = ctx.get(*exp_r) {
                                    if *m == two_rat {
                                        if let Expr::Function(name_r, args_r) = ctx.get(*base_r) {
                                            if name_r == "sin" && args_r.len() == 1 {
                                                return Some((args_l[0], args_r[0]));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

// Triple Angle Shortcut Rule: sin(3x) → 3sin(x) - 4sin³(x), cos(3x) → 4cos³(x) - 3cos(x)
// This is a performance optimization to avoid recursive expansion via double-angle rules.
// Reduces ~23 rewrites to ~3-5 for triple angle expressions.
define_rule!(
    TripleAngleRule,
    "Triple Angle Identity",
    |ctx, expr, parent_ctx| {
        // GUARD 1: Skip if marked for protection
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_sum_quotient_protected(expr) {
                return None;
            }
        }

        // GUARD 2: Skip if inside sum-quotient pattern (defer to SinCosSumQuotientRule)
        if is_inside_trig_quotient_pattern(ctx, expr, parent_ctx) {
            return None;
        }

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
                            return Some(
                                Rewrite::new(new_expr).desc("sin(3x) → 3sin(x) - 4sin³(x)"),
                            );
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
                            return Some(
                                Rewrite::new(new_expr).desc("cos(3x) → 4cos³(x) - 3cos(x)"),
                            );
                        }
                        "tan" => {
                            // tan(3x) → (3tan(x) - tan³(x)) / (1 - 3tan²(x))
                            let one = ctx.num(1);
                            let three = ctx.num(3);
                            let exp_two = ctx.num(2);
                            let exp_three = ctx.num(3);
                            let tan_x = ctx.add(Expr::Function("tan".to_string(), vec![inner_var]));

                            // Numerator: 3tan(x) - tan³(x)
                            let three_tan = smart_mul(ctx, three, tan_x);
                            let tan_cubed = ctx.add(Expr::Pow(tan_x, exp_three));
                            let numer = ctx.add(Expr::Sub(three_tan, tan_cubed));

                            // Denominator: 1 - 3tan²(x)
                            let tan_squared = ctx.add(Expr::Pow(tan_x, exp_two));
                            let three_tan_squared = smart_mul(ctx, three, tan_squared);
                            let denom = ctx.add(Expr::Sub(one, three_tan_squared));

                            let new_expr = ctx.add(Expr::Div(numer, denom));
                            return Some(
                                Rewrite::new(new_expr)
                                    .desc("tan(3x) → (3tan(x) - tan³(x))/(1 - 3tan²(x))"),
                            );
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }
);

// Quintuple Angle Rule: sin(5x) → 16sin⁵(x) - 20sin³(x) + 5sin(x)
// This is a direct expansion to avoid recursive explosion via double/triple angle.
define_rule!(
    QuintupleAngleRule,
    "Quintuple Angle Identity",
    |ctx, expr, parent_ctx| {
        // GUARD 1: Skip if marked for protection
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_sum_quotient_protected(expr) {
                return None;
            }
        }

        // GUARD 2: Skip if inside sum-quotient pattern
        if is_inside_trig_quotient_pattern(ctx, expr, parent_ctx) {
            return None;
        }

        if let Expr::Function(name, args) = ctx.get(expr) {
            if args.len() == 1 {
                // Check if arg is 5*x or x*5
                if let Some(inner_var) = crate::helpers::extract_quintuple_angle_arg(ctx, args[0]) {
                    match name.as_str() {
                        "sin" => {
                            // sin(5x) → 16sin⁵(x) - 20sin³(x) + 5sin(x)
                            let five = ctx.num(5);
                            let sixteen = ctx.num(16);
                            let twenty = ctx.num(20);
                            let exp_three = ctx.num(3);
                            let exp_five = ctx.num(5);
                            let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![inner_var]));

                            // 16sin⁵(x)
                            let sin_5 = ctx.add(Expr::Pow(sin_x, exp_five));
                            let term1 = smart_mul(ctx, sixteen, sin_5);

                            // 20sin³(x)
                            let sin_3 = ctx.add(Expr::Pow(sin_x, exp_three));
                            let term2 = smart_mul(ctx, twenty, sin_3);

                            // 5sin(x)
                            let term3 = smart_mul(ctx, five, sin_x);

                            // 16sin⁵(x) - 20sin³(x) + 5sin(x)
                            let sub1 = ctx.add(Expr::Sub(term1, term2));
                            let new_expr = ctx.add(Expr::Add(sub1, term3));
                            return Some(
                                Rewrite::new(new_expr)
                                    .desc("sin(5x) → 16sin⁵(x) - 20sin³(x) + 5sin(x)"),
                            );
                        }
                        "cos" => {
                            // cos(5x) → 16cos⁵(x) - 20cos³(x) + 5cos(x)
                            let five = ctx.num(5);
                            let sixteen = ctx.num(16);
                            let twenty = ctx.num(20);
                            let exp_three = ctx.num(3);
                            let exp_five = ctx.num(5);
                            let cos_x = ctx.add(Expr::Function("cos".to_string(), vec![inner_var]));

                            // 16cos⁵(x)
                            let cos_5 = ctx.add(Expr::Pow(cos_x, exp_five));
                            let term1 = smart_mul(ctx, sixteen, cos_5);

                            // 20cos³(x)
                            let cos_3 = ctx.add(Expr::Pow(cos_x, exp_three));
                            let term2 = smart_mul(ctx, twenty, cos_3);

                            // 5cos(x)
                            let term3 = smart_mul(ctx, five, cos_x);

                            // 16cos⁵(x) - 20cos³(x) + 5cos(x)
                            let sub1 = ctx.add(Expr::Sub(term1, term2));
                            let new_expr = ctx.add(Expr::Add(sub1, term3));
                            return Some(
                                Rewrite::new(new_expr)
                                    .desc("cos(5x) → 16cos⁵(x) - 20cos³(x) + 5cos(x)"),
                            );
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }
);

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

/// Check if a trig function is inside a potential sum-quotient pattern
/// (sin(A)±sin(B)) / (cos(A)±cos(B))
/// Returns true if expansion should be deferred to SinCosSumQuotientRule
fn is_inside_trig_quotient_pattern(
    ctx: &cas_ast::Context,
    _expr: ExprId,
    parent_ctx: &crate::parent_context::ParentContext,
) -> bool {
    // Check if any ancestor is a Div with the sum-quotient pattern
    parent_ctx.has_ancestor_matching(ctx, |c, id| {
        if let Expr::Div(num, den) = c.get(id) {
            // Check if numerator is Add or Sub of sin functions
            let num_is_sin_sum_or_diff = is_binary_trig_op(c, *num, "sin");
            // Check if denominator is Add of cos functions
            let den_is_cos_sum = is_trig_sum(c, *den, "cos");
            num_is_sin_sum_or_diff && den_is_cos_sum
        } else {
            false
        }
    })
}

/// Check if expr is Add(trig(A), trig(B)) or Sub(trig(A), trig(B)) or Add(trig(A), Neg(trig(B)))
fn is_binary_trig_op(ctx: &cas_ast::Context, expr: ExprId, fn_name: &str) -> bool {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            // Check for Add(sin(A), sin(B))
            if extract_trig_arg(ctx, *l, fn_name).is_some()
                && extract_trig_arg(ctx, *r, fn_name).is_some()
            {
                return true;
            }
            // Check for Add(sin(A), Neg(sin(B)))
            if let Expr::Neg(inner) = ctx.get(*r) {
                if extract_trig_arg(ctx, *l, fn_name).is_some()
                    && extract_trig_arg(ctx, *inner, fn_name).is_some()
                {
                    return true;
                }
            }
            if let Expr::Neg(inner) = ctx.get(*l) {
                if extract_trig_arg(ctx, *r, fn_name).is_some()
                    && extract_trig_arg(ctx, *inner, fn_name).is_some()
                {
                    return true;
                }
            }
            false
        }
        Expr::Sub(l, r) => {
            extract_trig_arg(ctx, *l, fn_name).is_some()
                && extract_trig_arg(ctx, *r, fn_name).is_some()
        }
        _ => false,
    }
}

/// Check if expr is Add(trig(A), trig(B))
fn is_trig_sum(ctx: &cas_ast::Context, expr: ExprId, fn_name: &str) -> bool {
    if let Expr::Add(l, r) = ctx.get(expr) {
        return extract_trig_arg(ctx, *l, fn_name).is_some()
            && extract_trig_arg(ctx, *r, fn_name).is_some();
    }
    false
}

define_rule!(
    RecursiveTrigExpansionRule,
    "Recursive Trig Expansion",
    |ctx, expr, parent_ctx| {
        // GUARD 1: Skip if this trig function is marked for protection
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_sum_quotient_protected(expr) {
                return None;
            }
        }

        // GUARD 2: Skip if we're inside a potential sum-quotient pattern
        // This heuristic checks: if the trig function is inside a Div, and both
        // numerator and denominator are Add/Sub of trig functions, defer to
        // SinCosSumQuotientRule instead of expanding.
        if is_inside_trig_quotient_pattern(ctx, expr, parent_ctx) {
            return None;
        }

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

                if n_val > num_bigint::BigInt::from(2) && n_val <= num_bigint::BigInt::from(6) {
                    // GUARD: Only expand sin(n*x) for small n (3-6).
                    // For n > 6, the expansion grows exponentially without benefit.
                    // This prevents catastrophic expansion like sin(671*x) → 670 recursive steps.

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
                        return Some(
                            Rewrite::new(new_expr).desc(format!("sin({}x) expansion", n_val)),
                        );
                    } else {
                        // cos
                        let t1 = smart_mul(ctx, cos_nm1, cos_x);
                        let t2 = smart_mul(ctx, sin_nm1, sin_x);
                        let new_expr = ctx.add(Expr::Sub(t1, t2));
                        return Some(
                            Rewrite::new(new_expr).desc(format!("cos({}x) expansion", n_val)),
                        );
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
                                    return Some(Rewrite::new(new_expr).desc("cos^2k(x) -> (1 - sin^2(x))^k"));
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

// =============================================================================
// DyadicCosProductToSinRule: 2^n · ∏_{k=0}^{n-1} cos(2^k·θ) → sin(2^n·θ)/sin(θ)
// =============================================================================
//
// This identity simplifies products like:
// - 2·cos(θ) → sin(2θ)/sin(θ)
// - 4·cos(θ)·cos(2θ) → sin(4θ)/sin(θ)
// - 8·cos(θ)·cos(2θ)·cos(4θ) → sin(8θ)/sin(θ)
//
// Combined with sin supplementary angle (sin(π-x)=sin(x)) and cancellation,
// this solves problems like: 8·cos(π/9)·cos(2π/9)·cos(4π/9) = 1.
//
// Domain: Requires sin(θ) ≠ 0. In Generic mode, this is only allowed when
// θ is a rational multiple of π that is NOT an integer (proven case).

/// DyadicCosProductToSinRule: Recognizes products 2^n · ∏cos(2^k·θ)
pub struct DyadicCosProductToSinRule;

impl crate::rule::Rule for DyadicCosProductToSinRule {
    fn name(&self) -> &str {
        "Dyadic Cos Product"
    }

    fn priority(&self) -> i32 {
        // Run BEFORE DoubleAngleRule and RecursiveTrigExpansionRule
        95
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::helpers::{as_number, flatten_mul, is_provably_sin_nonzero};
        use crate::rule::Rewrite;
        use num_bigint::BigInt;
        use num_rational::BigRational;

        // Only match Mul expressions
        if !matches!(ctx.get(expr), Expr::Mul(_, _)) {
            return None;
        }

        // Flatten the multiplication
        let mut factors = Vec::new();
        flatten_mul(ctx, expr, &mut factors);

        // Separate numeric coefficient from cos factors
        let mut numeric_coeff = BigRational::one();
        let mut cos_args: Vec<ExprId> = Vec::new();
        let mut other_factors: Vec<ExprId> = Vec::new();

        for &factor in &factors {
            if let Some(n) = as_number(ctx, factor) {
                numeric_coeff *= n.clone();
            } else if let Expr::Function(name, args) = ctx.get(factor) {
                if name == "cos" && args.len() == 1 {
                    cos_args.push(args[0]);
                } else {
                    other_factors.push(factor);
                }
            } else {
                other_factors.push(factor);
            }
        }

        // Must have no other factors and at least 1 cos
        if !other_factors.is_empty() || cos_args.is_empty() {
            return None;
        }

        let n = cos_args.len() as u32;

        // Numeric coefficient must be exactly 2^n
        let expected_coeff = BigRational::from_integer(BigInt::from(1u64 << n));
        if numeric_coeff != expected_coeff {
            return None;
        }

        // Find θ by trying each cos_arg as base and verifying dyadic sequence
        let mut theta: Option<ExprId> = None;

        for candidate in &cos_args {
            if verify_dyadic_sequence(ctx, *candidate, &cos_args) {
                theta = Some(*candidate);
                break;
            }
        }

        let theta = theta?;

        // Domain check: sin(θ) ≠ 0
        let domain_mode = parent_ctx.domain_mode();

        if !is_provably_sin_nonzero(ctx, theta) {
            match domain_mode {
                crate::domain::DomainMode::Generic | crate::domain::DomainMode::Strict => {
                    // Block with hint
                    let sin_theta = ctx.add(Expr::Function("sin".to_string(), vec![theta]));
                    crate::domain::register_blocked_hint(crate::domain::BlockedHint {
                        rule: "Dyadic Cos Product".to_string(),
                        expr_id: sin_theta,
                        key: crate::assumptions::AssumptionKey::nonzero_key(ctx, sin_theta),
                        suggestion: "use `domain assume` to allow this transformation",
                    });
                    return None;
                }
                crate::domain::DomainMode::Assume => {
                    // Allow but will record assumption in result
                }
            }
        }

        // Build result: sin(2^n · θ) / sin(θ)
        let two_pow_n = ctx.num((1u64 << n) as i64);
        let scaled_theta = smart_mul(ctx, two_pow_n, theta);
        let sin_scaled = ctx.add(Expr::Function("sin".to_string(), vec![scaled_theta]));
        let sin_theta = ctx.add(Expr::Function("sin".to_string(), vec![theta]));
        let result = ctx.add(Expr::Div(sin_scaled, sin_theta));

        // Build description
        let desc = format!("2^{n}·∏cos(2^k·θ) = sin(2^{n}·θ)/sin(θ)", n = n);

        let mut rewrite = Rewrite::new(result).desc(&desc).local(expr, result);

        // Add assumption if in Assume mode and sin(θ)≠0 not proven
        if domain_mode == crate::domain::DomainMode::Assume && !is_provably_sin_nonzero(ctx, theta)
        {
            // Create NonZero assumption with HeuristicAssumption kind
            let mut event = crate::assumptions::AssumptionEvent::nonzero(ctx, sin_theta);
            event.kind = crate::assumptions::AssumptionKind::HeuristicAssumption;
            rewrite = rewrite.assume(event);
        }

        Some(rewrite)
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Mul"])
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn soundness(&self) -> crate::rule::SoundnessLabel {
        // The transformation requires sin(θ) ≠ 0, which is either proven or assumed
        crate::rule::SoundnessLabel::EquivalenceUnderIntroducedRequires
    }
}

/// Verify that cos_args form a dyadic sequence: θ, 2θ, 4θ, ..., 2^(n-1)θ
///
/// Instead of structural comparison (which fails on normalized forms),
/// we extract the rational coefficient of each arg relative to π and check
/// if they form the sequence k, 2k, 4k, ..., 2^(n-1)k for some base k.
fn verify_dyadic_sequence(ctx: &mut cas_ast::Context, theta: ExprId, cos_args: &[ExprId]) -> bool {
    use crate::helpers::extract_rational_pi_multiple;
    use num_rational::BigRational;

    let n = cos_args.len() as u32;
    if n == 0 {
        return false;
    }

    // Extract the base coefficient from theta
    let base_coeff = match extract_rational_pi_multiple(ctx, theta) {
        Some(k) => k,
        None => return false, // theta must be a rational multiple of π
    };

    // Collect all coefficients from cos_args
    let mut coeffs: Vec<BigRational> = Vec::with_capacity(n as usize);
    for &arg in cos_args {
        match extract_rational_pi_multiple(ctx, arg) {
            Some(k) => coeffs.push(k),
            None => return false, // All args must be rational multiples of π
        }
    }

    // Build expected coefficients: base, 2*base, 4*base, ..., 2^(n-1)*base
    let mut expected: Vec<BigRational> = Vec::with_capacity(n as usize);
    for k in 0..n {
        let multiplier = BigRational::from_integer((1u64 << k).into());
        expected.push(&base_coeff * &multiplier);
    }

    // Check if coeffs matches expected as multiset
    let mut used = vec![false; expected.len()];
    for coeff in &coeffs {
        let mut found = false;
        for (i, exp) in expected.iter().enumerate() {
            if !used[i] && coeff == exp {
                used[i] = true;
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }

    used.iter().all(|&u| u)
}

pub fn register(simplifier: &mut crate::Simplifier) {
    // PRE-ORDER: Evaluate sin(n·π) = 0 and cos(n·π) = (-1)^n BEFORE any expansion
    // This prevents unnecessary triple/double angle expansions on integer multiples of π
    simplifier.add_rule(Box::new(SinCosIntegerPiRule));

    // PRE-ORDER: Trig parity (odd/even functions)
    // sin(-u) = -sin(u), cos(-u) = cos(u), tan(-u) = -tan(u)
    simplifier.add_rule(Box::new(TrigOddEvenParityRule));

    // Use the new data-driven EvaluateTrigTableRule instead of deprecated EvaluateTrigRule
    simplifier.add_rule(Box::new(super::evaluation::EvaluateTrigTableRule));
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    simplifier.add_rule(Box::new(SecTanPythagoreanRule));
    simplifier.add_rule(Box::new(CscCotPythagoreanRule));

    // Hidden Cubic Identity: sin^6 + cos^6 + 3sin^2cos^2 = (sin^2+cos^2)^3
    // Should run in TRANSFORM phase before power expansions
    simplifier.add_rule(Box::new(TrigHiddenCubicIdentityRule));

    simplifier.add_rule(Box::new(AngleIdentityRule));
    // Triple tangent product: tan(u)·tan(π/3+u)·tan(π/3-u) → tan(3u)
    // Must run BEFORE TanToSinCosRule to prevent expansion
    simplifier.add_rule(Box::new(TanTripleProductRule));
    // Weierstrass Identity Zero Rules: MUST run BEFORE TanToSinCosRule
    // Pattern-driven cancellation: sin(x) - 2t/(1+t²) → 0, cos(x) - (1-t²)/(1+t²) → 0
    simplifier.add_rule(Box::new(WeierstrassSinIdentityZeroRule));
    simplifier.add_rule(Box::new(WeierstrassCosIdentityZeroRule));
    simplifier.add_rule(Box::new(TanToSinCosRule));
    // Dyadic Cos Product: 2^n·∏cos(2^k·θ) → sin(2^n·θ)/sin(θ)
    // Must run BEFORE DoubleAngleRule to recognize the pattern
    simplifier.add_rule(Box::new(DyadicCosProductToSinRule));
    simplifier.add_rule(Box::new(DoubleAngleRule));
    simplifier.add_rule(Box::new(DoubleAngleContractionRule)); // 2sin·cos→sin(2t), cos²-sin²→cos(2t)
                                                               // Sum-to-Product Quotient: runs BEFORE TripleAngleRule to avoid polynomial explosion
    simplifier.add_rule(Box::new(SinCosSumQuotientRule));
    // Standalone Sum-to-Product: sin(A)+sin(B), cos(A)+cos(B) etc. when args are k*π
    simplifier.add_rule(Box::new(TrigSumToProductRule));
    simplifier.add_rule(Box::new(TripleAngleRule)); // Shortcut: sin(3x), cos(3x), tan(3x)
    simplifier.add_rule(Box::new(QuintupleAngleRule)); // Shortcut: sin(5x), cos(5x)
    simplifier.add_rule(Box::new(RecursiveTrigExpansionRule));
    // Trig Quotient: sin(x)/cos(x) → tan(x) - runs after sum-to-product
    simplifier.add_rule(Box::new(TrigQuotientRule));
    // Half-Angle Tangent: (1-cos(2x))/sin(2x) → tan(x), sin(2x)/(1+cos(2x)) → tan(x)
    simplifier.add_rule(Box::new(HalfAngleTangentRule));
    // Weierstrass Contraction: 2*tan(x/2)/(1+tan²) → sin(x), (1-tan²)/(1+tan²) → cos(x)
    simplifier.add_rule(Box::new(WeierstrassContractionRule));

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
    // N-ary Pythagorean: sin²(t) + cos²(t) → 1 in chains of any length
    simplifier.add_rule(Box::new(super::pythagorean::TrigPythagoreanChainRule));
    // Contraction: 1 + tan²(x) → sec²(x), 1 + cot²(x) → csc²(x)
    simplifier.add_rule(Box::new(super::pythagorean::RecognizeSecSquaredRule));
    simplifier.add_rule(Box::new(super::pythagorean::RecognizeCscSquaredRule));
    // Expansion: sec(x) → 1/cos(x), csc(x) → 1/sin(x) for canonical unification
    simplifier.add_rule(Box::new(super::pythagorean::SecToRecipCosRule));
    simplifier.add_rule(Box::new(super::pythagorean::CscToRecipSinRule));
    // Expansion: cot(x) → cos(x)/sin(x) for canonical unification
    simplifier.add_rule(Box::new(super::pythagorean::CotToCosSinRule));

    simplifier.add_rule(Box::new(AngleConsistencyRule));

    // Phase shift: sin(x + π/2) → cos(x), cos(x + π/2) → -sin(x), etc.
    simplifier.add_rule(Box::new(TrigPhaseShiftRule));

    // Supplementary angle: sin(8π/9) = sin(π - π/9) = sin(π/9)
    simplifier.add_rule(Box::new(SinSupplementaryAngleRule));

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
                return Some(Rewrite::new(new_expr).desc("Half-Angle Expansion"));
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

            return Some(Rewrite::new(final_expr).desc(description));
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

        return Some(
            Rewrite::new(new_expr).desc(format!("{}(x + {}) phase shift", name, shift_desc)),
        );
    }

    None
});

// =============================================================================
// Sin Supplementary Angle Rule
// =============================================================================
// sin(π - x) → sin(x)
// sin(k·π - x) → (-1)^(k+1) · sin(x) for integer k
// cos(π - x) → -cos(x)
//
// This enables simplification of expressions like sin(8π/9) = sin(π - π/9) = sin(π/9)

define_rule!(
    SinSupplementaryAngleRule,
    "Supplementary Angle",
    |ctx, expr| {
        use crate::helpers::extract_rational_pi_multiple;
        use num_rational::BigRational;

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

            // Try to check if arg is a rational multiple of π
            // where the coefficient is of the form (n - small) for some positive integer n
            // e.g., 8/9 = 1 - 1/9, so sin(8π/9) = sin(π - π/9) = sin(π/9)

            if let Some(k) = extract_rational_pi_multiple(ctx, arg) {
                // k = p/q in lowest terms
                let p = k.numer();
                let q = k.denom();

                // Check if p/q is close enough to an integer that the supplementary form is simpler.
                // For sin(k·π) where k = (n*q - m)/q, we can write it as sin((n*q - m)/q · π) = sin(n·π - m/q·π)
                // This simplifies when m < p (i.e., the remainder is smaller than the original numerator)
                //
                // Example: sin(8/9·π) = sin(1·π - 1/9·π) = sin(π/9) because 1 < 8

                // Only for positive k (p > 0)
                if p > &num_bigint::BigInt::from(0) {
                    let one = num_bigint::BigInt::from(1);
                    // n = ceil(p/q) = floor((p + q - 1) / q)
                    let n_candidate = (p + q - &one) / q;
                    let remainder = &n_candidate * q - p; // m = n*q - p

                    // Apply simplification if:
                    // 1. remainder > 0 (i.e., k is not an integer)
                    // 2. remainder < p (i.e., the new form is simpler)
                    // 3. n >= 1 (always true since p > 0)
                    if remainder > num_bigint::BigInt::from(0) && &remainder < p {
                        // The supplementary angle is m/q * π
                        let new_coeff = BigRational::new(remainder.clone(), q.clone());

                        // Build the new angle: (m/q) * π
                        let new_angle = if new_coeff == BigRational::from_integer(1.into()) {
                            ctx.add(Expr::Constant(cas_ast::Constant::Pi))
                        } else {
                            let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                            let coeff_expr = ctx.add(Expr::Number(new_coeff));
                            ctx.add(Expr::Mul(coeff_expr, pi))
                        };

                        // Determine sign based on parity of n
                        // sin(n·π - x) = (-1)^(n+1) · sin(x)
                        // cos(n·π - x) = (-1)^n · cos(x)
                        let n_parity_odd = &n_candidate % 2 == one;

                        let (result, desc) = if is_sin {
                            // sin(n·π - x) = (-1)^(n+1) · sin(x)
                            // n odd → (-1)^(n+1) = 1, so sin(x)
                            // n even → (-1)^(n+1) = -1, so -sin(x)
                            let new_trig =
                                ctx.add(Expr::Function("sin".to_string(), vec![new_angle]));
                            if n_parity_odd {
                                (new_trig, format!("sin({}π - x) = sin(x)", n_candidate))
                            } else {
                                (
                                    ctx.add(Expr::Neg(new_trig)),
                                    format!("sin({}π - x) = -sin(x)", n_candidate),
                                )
                            }
                        } else {
                            // cos(n·π - x) = (-1)^n · cos(x)
                            // n odd → -cos(x), n even → cos(x)
                            let new_trig =
                                ctx.add(Expr::Function("cos".to_string(), vec![new_angle]));
                            if n_parity_odd {
                                (
                                    ctx.add(Expr::Neg(new_trig)),
                                    format!("cos({}π - x) = -cos(x)", n_candidate),
                                )
                            } else {
                                (new_trig, format!("cos({}π - x) = cos(x)", n_candidate))
                            }
                        };

                        return Some(Rewrite::new(result).desc(&desc));
                    }
                }
            }
        }

        None
    }
);

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

// =============================================================================
// WEIERSTRASS HALF-ANGLE TANGENT CONTRACTION RULES
// =============================================================================
// Recognize patterns with t = tan(x/2) and contract to sin(x), cos(x):
// - 2*t / (1 + t²) → sin(x)
// - (1 - t²) / (1 + t²) → cos(x)
// This is the CONTRACTION direction (safe, doesn't worsen expressions)

/// Helper: Check if expr is tan(arg/2) and return Some(arg), i.e. the full angle
fn extract_tan_half_angle(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if name == "tan" && args.len() == 1 {
            // Check if the argument is x/2 or (1/2)*x
            let arg = args[0];
            // Pattern: Div(x, 2) or Mul(1/2, x) or Mul(x, 1/2)
            match ctx.get(arg) {
                Expr::Div(num, den) => {
                    // x/2 pattern
                    if let Expr::Number(n) = ctx.get(*den) {
                        if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into())
                        {
                            return Some(*num); // return x (the full angle)
                        }
                    }
                }
                Expr::Mul(l, r) => {
                    // (1/2)*x or x*(1/2) pattern
                    let half = num_rational::BigRational::new(1.into(), 2.into());
                    if let Expr::Number(n) = ctx.get(*l) {
                        if *n == half {
                            return Some(*r);
                        }
                    }
                    if let Expr::Number(n) = ctx.get(*r) {
                        if *n == half {
                            return Some(*l);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// Helper: Check if expr is 1 + tan(x/2)² and return (x, tan_half_id)
fn match_one_plus_tan_squared(ctx: &cas_ast::Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    if let Expr::Add(l, r) = ctx.get(expr) {
        // Check both orders: 1 + tan²(...) or tan²(...) + 1
        let (one_id, pow_id) = if matches!(ctx.get(*l), Expr::Number(n) if n.is_one()) {
            (*l, *r)
        } else if matches!(ctx.get(*r), Expr::Number(n) if n.is_one()) {
            (*r, *l)
        } else {
            return None;
        };
        let _ = one_id;

        // Check if pow_id is tan(x/2)^2
        if let Expr::Pow(base, exp) = ctx.get(pow_id) {
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()) {
                    if let Some(full_angle) = extract_tan_half_angle(ctx, *base) {
                        return Some((full_angle, *base));
                    }
                }
            }
        }
    }
    None
}

/// Helper: Check if expr is 1 - tan(x/2)² and return (x, tan_half_id)
fn match_one_minus_tan_squared(ctx: &cas_ast::Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    // Check Sub(1, tan²)
    if let Expr::Sub(l, r) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*l) {
            if n.is_one() {
                // Check if r is tan(x/2)^2
                if let Expr::Pow(base, exp) = ctx.get(*r) {
                    if let Expr::Number(e) = ctx.get(*exp) {
                        if e.is_integer() && *e == num_rational::BigRational::from_integer(2.into())
                        {
                            if let Some(full_angle) = extract_tan_half_angle(ctx, *base) {
                                return Some((full_angle, *base));
                            }
                        }
                    }
                }
            }
        }
    }

    // Also check Add(1, Neg(tan²)) which is canonicalized form
    if let Expr::Add(l, r) = ctx.get(expr) {
        let (one_id, neg_id) = if matches!(ctx.get(*l), Expr::Number(n) if n.is_one()) {
            (*l, *r)
        } else if matches!(ctx.get(*r), Expr::Number(n) if n.is_one()) {
            (*r, *l)
        } else {
            return None;
        };
        let _ = one_id;

        if let Expr::Neg(inner) = ctx.get(neg_id) {
            if let Expr::Pow(base, exp) = ctx.get(*inner) {
                if let Expr::Number(e) = ctx.get(*exp) {
                    if e.is_integer() && *e == num_rational::BigRational::from_integer(2.into()) {
                        if let Some(full_angle) = extract_tan_half_angle(ctx, *base) {
                            return Some((full_angle, *base));
                        }
                    }
                }
            }
        }
    }

    None
}

// Weierstrass Contraction Rule: 2*tan(x/2)/(1+tan²(x/2)) → sin(x)
// and (1-tan²(x/2))/(1+tan²(x/2)) → cos(x)
struct WeierstrassContractionRule;

impl crate::rule::Rule for WeierstrassContractionRule {
    fn name(&self) -> &str {
        "Weierstrass Half-Angle Contraction"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        // Only match Div nodes
        let Expr::Div(num_id, den_id) = ctx.get(expr).clone() else {
            return None;
        };

        // Pattern 1: 2*tan(x/2) / (1 + tan²(x/2)) → sin(x)
        // Check denominator: 1 + tan²(x/2)
        if let Some((full_angle, tan_half)) = match_one_plus_tan_squared(ctx, den_id) {
            // Check numerator: 2*tan(x/2)
            if let Expr::Mul(l, r) = ctx.get(num_id) {
                let (two_id, tan_id) = if matches!(ctx.get(*l), Expr::Number(n) if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()))
                {
                    (*l, *r)
                } else if matches!(ctx.get(*r), Expr::Number(n) if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()))
                {
                    (*r, *l)
                } else {
                    return self.try_cos_pattern(ctx, num_id, den_id, full_angle, tan_half);
                };
                let _ = two_id;

                // Check if tan_id is tan(x/2) with same argument
                if let Some(tan_arg) = extract_tan_half_angle(ctx, tan_id) {
                    if crate::ordering::compare_expr(ctx, tan_arg, full_angle)
                        == std::cmp::Ordering::Equal
                    {
                        let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![full_angle]));
                        return Some(
                            Rewrite::new(sin_x).desc("2·tan(x/2)/(1 + tan²(x/2)) = sin(x)"),
                        );
                    }
                }
            }

            // Pattern 2: (1 - tan²(x/2)) / (1 + tan²(x/2)) → cos(x)
            return self.try_cos_pattern(ctx, num_id, den_id, full_angle, tan_half);
        }

        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Div"])
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

impl WeierstrassContractionRule {
    fn try_cos_pattern(
        &self,
        ctx: &mut cas_ast::Context,
        num_id: ExprId,
        den_id: ExprId,
        _expected_angle: ExprId,
        _expected_tan_half: ExprId,
    ) -> Option<Rewrite> {
        // Pattern 2: (1 - tan²(x/2)) / (1 + tan²(x/2)) → cos(x)
        if let Some((num_angle, _num_tan_half)) = match_one_minus_tan_squared(ctx, num_id) {
            if let Some((den_angle, _den_tan_half)) = match_one_plus_tan_squared(ctx, den_id) {
                // Check angles are the same
                if crate::ordering::compare_expr(ctx, num_angle, den_angle)
                    == std::cmp::Ordering::Equal
                {
                    let cos_x = ctx.add(Expr::Function("cos".to_string(), vec![num_angle]));
                    return Some(
                        Rewrite::new(cos_x).desc("(1 - tan²(x/2))/(1 + tan²(x/2)) = cos(x)"),
                    );
                }
            }
        }

        None
    }
}

// =============================================================================
// WEIERSTRASS IDENTITY ZERO RULES (Pattern-Driven Cancellation)
// =============================================================================
// These rules detect the complete Weierstrass identity patterns and cancel to 0
// directly, avoiding explosive expansion through tan→sin/cos conversion.
//
// sin(x) - 2*tan(x/2)/(1 + tan(x/2)²) → 0
// cos(x) - (1 - tan(x/2)²)/(1 + tan(x/2)²) → 0

/// Helper: Check if expr matches 2*tan(x/2) and return the full angle x
fn match_two_tan_half(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let two_rat = num_rational::BigRational::from_integer(2.into());

    if let Expr::Mul(l, r) = ctx.get(expr) {
        // Check Mul(2, tan(x/2)) or Mul(tan(x/2), 2)
        if let Expr::Number(n) = ctx.get(*l) {
            if *n == two_rat {
                return extract_tan_half_angle(ctx, *r);
            }
        }
        if let Expr::Number(n) = ctx.get(*r) {
            if *n == two_rat {
                return extract_tan_half_angle(ctx, *l);
            }
        }
    }
    None
}

/// Helper: Check if expr matches 1 + tan(x/2)² and return (full_angle, tan_half_id)
fn match_one_plus_tan_half_squared(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Add(l, r) = ctx.get(expr) {
        let two_rat = num_rational::BigRational::from_integer(2.into());

        // Pattern: 1 + tan²(x/2) or tan²(x/2) + 1
        let (one_candidate, pow_candidate) = if matches!(ctx.get(*l), Expr::Number(n) if n.is_one())
        {
            (*l, *r)
        } else if matches!(ctx.get(*r), Expr::Number(n) if n.is_one()) {
            (*r, *l)
        } else {
            return None;
        };
        let _ = one_candidate;

        // Check pow_candidate is tan(x/2)^2
        if let Expr::Pow(base, exp) = ctx.get(pow_candidate) {
            if let Expr::Number(n) = ctx.get(*exp) {
                if *n == two_rat {
                    return extract_tan_half_angle(ctx, *base);
                }
            }
        }
    }
    None
}

/// Helper: Check if expr matches 1 - tan(x/2)² and return full_angle
fn match_one_minus_tan_half_squared(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let two_rat = num_rational::BigRational::from_integer(2.into());

    // Pattern: 1 - tan²(x/2) as Sub(1, tan²)
    if let Expr::Sub(l, r) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*l) {
            if n.is_one() {
                if let Expr::Pow(base, exp) = ctx.get(*r) {
                    if let Expr::Number(e) = ctx.get(*exp) {
                        if *e == two_rat {
                            return extract_tan_half_angle(ctx, *base);
                        }
                    }
                }
            }
        }
    }

    // Also try Add(1, Neg(tan²)) or Add(Neg(tan²), 1)
    if let Expr::Add(l, r) = ctx.get(expr) {
        // 1 + (-tan²) or (-tan²) + 1
        let (one_candidate, neg_candidate) = if matches!(ctx.get(*l), Expr::Number(n) if n.is_one())
        {
            (*l, *r)
        } else if matches!(ctx.get(*r), Expr::Number(n) if n.is_one()) {
            (*r, *l)
        } else {
            return None;
        };
        let _ = one_candidate;

        if let Expr::Neg(inner) = ctx.get(neg_candidate) {
            if let Expr::Pow(base, exp) = ctx.get(*inner) {
                if let Expr::Number(e) = ctx.get(*exp) {
                    if *e == two_rat {
                        return extract_tan_half_angle(ctx, *base);
                    }
                }
            }
        }
    }

    None
}

// WeierstrassSinIdentityZeroRule: sin(x) - 2*tan(x/2)/(1+tan²(x/2)) → 0
// Pattern-driven cancellation, no expansion.
struct WeierstrassSinIdentityZeroRule;

impl crate::rule::Rule for WeierstrassSinIdentityZeroRule {
    fn name(&self) -> &str {
        "Weierstrass Sin Identity Zero"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only match Sub nodes: sin(x) - RHS or RHS - sin(x)
        let (left, right, negated) = match ctx.get(expr).clone() {
            Expr::Sub(l, r) => (l, r, false),
            Expr::Add(l, r) => {
                // Check if one side is negated
                if let Expr::Neg(inner) = ctx.get(r) {
                    (l, *inner, false)
                } else if let Expr::Neg(inner) = ctx.get(l) {
                    (r, *inner, false)
                } else {
                    return None;
                }
            }
            _ => return None,
        };
        let _ = negated;

        // Try both orderings: sin(x) - RHS and RHS - sin(x)
        if let Some(result) = self.try_match(ctx, left, right) {
            return Some(result);
        }
        if let Some(result) = self.try_match(ctx, right, left) {
            return Some(result);
        }

        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Sub", "Add"])
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn priority(&self) -> i32 {
        200 // Very high - must run BEFORE Pythagorean 1+tan²→sec²
    }
}

impl WeierstrassSinIdentityZeroRule {
    /// Try to match sin(x) = left, RHS = right
    fn try_match(
        &self,
        ctx: &mut cas_ast::Context,
        sin_side: ExprId,
        rhs: ExprId,
    ) -> Option<Rewrite> {
        // Check if sin_side is sin(x)
        if let Expr::Function(name, args) = ctx.get(sin_side) {
            if name != "sin" || args.len() != 1 {
                return None;
            }
            let full_angle = args[0];

            // Check if rhs is 2*tan(x/2) / (1 + tan²(x/2))
            if let Expr::Div(num, den) = ctx.get(rhs) {
                // Numerator: 2*tan(x/2)
                if let Some(num_angle) = match_two_tan_half(ctx, *num) {
                    // Denominator: 1 + tan²(x/2)
                    if let Some(den_angle) = match_one_plus_tan_half_squared(ctx, *den) {
                        // Check all angles match
                        if crate::ordering::compare_expr(ctx, full_angle, num_angle)
                            == std::cmp::Ordering::Equal
                            && crate::ordering::compare_expr(ctx, full_angle, den_angle)
                                == std::cmp::Ordering::Equal
                        {
                            let zero = ctx.num(0);
                            return Some(
                                Rewrite::new(zero)
                                    .desc("sin(x) = 2·tan(x/2)/(1 + tan²(x/2)) [Weierstrass]"),
                            );
                        }
                    }
                }
            }
        }
        None
    }
}

// WeierstrassCosIdentityZeroRule: cos(x) - (1-tan²(x/2))/(1+tan²(x/2)) → 0
struct WeierstrassCosIdentityZeroRule;

impl crate::rule::Rule for WeierstrassCosIdentityZeroRule {
    fn name(&self) -> &str {
        "Weierstrass Cos Identity Zero"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only match Sub nodes: cos(x) - RHS or RHS - cos(x)
        let (left, right) = match ctx.get(expr).clone() {
            Expr::Sub(l, r) => (l, r),
            Expr::Add(l, r) => {
                // Check if one side is negated
                if let Expr::Neg(inner) = ctx.get(r) {
                    (l, *inner)
                } else if let Expr::Neg(inner) = ctx.get(l) {
                    (r, *inner)
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        // Try both orderings
        if let Some(result) = self.try_match(ctx, left, right) {
            return Some(result);
        }
        if let Some(result) = self.try_match(ctx, right, left) {
            return Some(result);
        }

        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Sub", "Add"])
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn priority(&self) -> i32 {
        200 // Very high - must run BEFORE Pythagorean 1+tan²→sec²
    }
}

impl WeierstrassCosIdentityZeroRule {
    /// Try to match cos(x) = cos_side, RHS = rhs
    fn try_match(
        &self,
        ctx: &mut cas_ast::Context,
        cos_side: ExprId,
        rhs: ExprId,
    ) -> Option<Rewrite> {
        // Check if cos_side is cos(x)
        if let Expr::Function(name, args) = ctx.get(cos_side) {
            if name != "cos" || args.len() != 1 {
                return None;
            }
            let full_angle = args[0];

            // Check if rhs is (1 - tan²(x/2)) / (1 + tan²(x/2))
            if let Expr::Div(num, den) = ctx.get(rhs) {
                // Numerator: 1 - tan²(x/2)
                if let Some(num_angle) = match_one_minus_tan_half_squared(ctx, *num) {
                    // Denominator: 1 + tan²(x/2)
                    if let Some(den_angle) = match_one_plus_tan_half_squared(ctx, *den) {
                        // Check all angles match
                        if crate::ordering::compare_expr(ctx, full_angle, num_angle)
                            == std::cmp::Ordering::Equal
                            && crate::ordering::compare_expr(ctx, full_angle, den_angle)
                                == std::cmp::Ordering::Equal
                        {
                            let zero = ctx.num(0);
                            return Some(
                                Rewrite::new(zero)
                                    .desc("cos(x) = (1 - tan²(x/2))/(1 + tan²(x/2)) [Weierstrass]"),
                            );
                        }
                    }
                }
            }
        }
        None
    }
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

                        return Some(Rewrite::new(new_expr).desc("cot(u/2) - cot(u) = 1/sin(u)"));
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

                        return Some(Rewrite::new(new_expr).desc("-cot(u/2) + cot(u) = -1/sin(u)"));
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
