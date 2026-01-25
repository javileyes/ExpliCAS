//! Core trigonometric identity rules for evaluation at special values.

use crate::define_rule;
use crate::helpers::{is_pi, is_pi_over_n};
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Expr, ExprId};
use num_traits::{One, Zero};

// Import helpers from sibling modules (via re-exports in parent)
use super::{has_large_coefficient, is_multiple_angle};

// Import table-driven evaluation
use super::trig_table::{eval_inv_trig_special, eval_trig_special, InvTrigFn, TrigFn};

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

        if let Expr::Function(fn_id, args) = expr_data {
            let name = ctx.sym_name(fn_id);
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

        if let Expr::Function(fn_id, args) = expr_data {
            let name = ctx.sym_name(fn_id).to_string();
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
                        let f_u = ctx.add(Expr::Function(fn_id, vec![positive_arg]));
                        let neg_f_u = ctx.add(Expr::Neg(f_u));
                        return Some(
                            Rewrite::new(neg_f_u)
                                .desc(format!("{}(-u) = -{}(u) [odd function]", name, name)),
                        );
                    }
                    // EVEN functions: f(-u) = f(u)
                    "cos" | "sec" | "cosh" => {
                        let f_u = ctx.add(Expr::Function(fn_id, vec![positive_arg]));
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
        if let Expr::Function(fn_id, args) = expr_data {
            let name = ctx.sym_name(fn_id).to_string();
            if args.len() == 1 {
                let arg = args[0];

                // ============================================================
                // TABLE-DRIVEN FAST PATH: Try table lookup first
                // ============================================================
                // Direct trig functions
                let trig_fn = match name.as_str() {
                    "sin" => Some(TrigFn::Sin),
                    "cos" => Some(TrigFn::Cos),
                    "tan" => Some(TrigFn::Tan),
                    _ => None,
                };
                if let Some(f) = trig_fn {
                    if let Some(result) = eval_trig_special(ctx, f, arg) {
                        let desc = format!("{}(...) evaluated via table", name);
                        return Some(Rewrite::new(result).desc(desc));
                    }
                }

                // Inverse trig functions
                let inv_trig_fn = match name.as_str() {
                    "arcsin" | "asin" => Some(InvTrigFn::Asin),
                    "arccos" | "acos" => Some(InvTrigFn::Acos),
                    "arctan" | "atan" => Some(InvTrigFn::Atan),
                    _ => None,
                };
                if let Some(f) = inv_trig_fn {
                    if let Some(result) = eval_inv_trig_special(ctx, f, arg) {
                        let desc = format!("{}(...) evaluated via table", name);
                        return Some(Rewrite::new(result).desc(desc));
                    }
                }
                // ============================================================

                // Case 1: Known Values (0) - fallback for non-table cases
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
                            let sin_inner = ctx.call("sin", vec![inner]);
                            let new_expr = ctx.add(Expr::Neg(sin_inner));
                            return Some(Rewrite::new(new_expr).desc("sin(-x) = -sin(x)"));
                        }
                        "cos" => {
                            let new_expr = ctx.call("cos", vec![inner]);
                            return Some(Rewrite::new(new_expr).desc("cos(-x) = cos(x)"));
                        }
                        "tan" => {
                            let tan_inner = ctx.call("tan", vec![inner]);
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
                            let trig_info = if let Expr::Function(fn_id, args) = ctx.get(base) {
                                let name = ctx.sym_name(*fn_id);
                                if (name == "sin" || name == "cos") && args.len() == 1 {
                                    Some((name.to_string(), args[0]))
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
                                    if let Expr::Function(fn_id, args) = ctx.get(base) {
                                        let name = ctx.sym_name(*fn_id);
                                        if (name == "sin" || name == "cos") && args.len() == 1 {
                                            trig_idx = Some(i);
                                            trig_info = Some((name.to_string(), args[0]));

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
            if let Expr::Function(fn_id, args) = ctx.get(expr) {
                let name = ctx.sym_name(*fn_id);
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
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            let name = ctx.sym_name(*fn_id);
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

        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            if args.len() == 1 {
                let inner = args[0];
                match ctx.sym_name(*fn_id) {
                    "sin" => {
                        let inner_data = ctx.get(inner).clone();
                        if let Expr::Add(lhs, rhs) = inner_data {
                            // sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
                            let sin_a = ctx.call("sin", vec![lhs]);
                            let cos_b = ctx.call("cos", vec![rhs]);
                            let term1 = smart_mul(ctx, sin_a, cos_b);

                            let cos_a = ctx.call("cos", vec![lhs]);
                            let sin_b = ctx.call("sin", vec![rhs]);
                            let term2 = smart_mul(ctx, cos_a, sin_b);

                            let new_expr = ctx.add(Expr::Add(term1, term2));
                            return Some(
                                Rewrite::new(new_expr)
                                    .desc("sin(a + b) -> sin(a)cos(b) + cos(a)sin(b)"),
                            );
                        } else if let Expr::Sub(lhs, rhs) = inner_data {
                            // sin(a - b) = sin(a)cos(b) - cos(a)sin(b)
                            let sin_a = ctx.call("sin", vec![lhs]);
                            let cos_b = ctx.call("cos", vec![rhs]);
                            let term1 = smart_mul(ctx, sin_a, cos_b);

                            let cos_a = ctx.call("cos", vec![lhs]);
                            let sin_b = ctx.call("sin", vec![rhs]);
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

                                let sin_a = ctx.call("sin", vec![a]);
                                let cos_b = ctx.call("cos", vec![b]);
                                let term1 = smart_mul(ctx, sin_a, cos_b);

                                let cos_a = ctx.call("cos", vec![a]);
                                let sin_b = ctx.call("sin", vec![b]);
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
                            let cos_a = ctx.call("cos", vec![lhs]);
                            let cos_b = ctx.call("cos", vec![rhs]);
                            let term1 = smart_mul(ctx, cos_a, cos_b);

                            let sin_a = ctx.call("sin", vec![lhs]);
                            let sin_b = ctx.call("sin", vec![rhs]);
                            let term2 = smart_mul(ctx, sin_a, sin_b);

                            let new_expr = ctx.add(Expr::Sub(term1, term2));
                            return Some(
                                Rewrite::new(new_expr)
                                    .desc("cos(a + b) -> cos(a)cos(b) - sin(a)sin(b)"),
                            );
                        } else if let Expr::Sub(lhs, rhs) = inner_data {
                            // cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
                            let cos_a = ctx.call("cos", vec![lhs]);
                            let cos_b = ctx.call("cos", vec![rhs]);
                            let term1 = smart_mul(ctx, cos_a, cos_b);

                            let sin_a = ctx.call("sin", vec![lhs]);
                            let sin_b = ctx.call("sin", vec![rhs]);
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

                                let cos_a = ctx.call("cos", vec![a]);
                                let cos_b = ctx.call("cos", vec![b]);
                                let term1 = smart_mul(ctx, cos_a, cos_b);

                                let sin_a = ctx.call("sin", vec![a]);
                                let sin_b = ctx.call("sin", vec![b]);
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
