use crate::define_rule;
use crate::helpers::{is_one, is_zero};
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use std::cmp::Ordering;

// ==================== Helper Functions ====================

// is_zero and is_one are now imported from crate::helpers

/// Check if expression equals 2
fn is_two(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        *n == num_rational::Ratio::from_integer(2.into())
    } else {
        false
    }
}

// ==================== Hyperbolic Function Rules ====================

// Rule 1: Evaluate hyperbolic functions at special values
define_rule!(
    EvaluateHyperbolicRule,
    "Evaluate Hyperbolic Functions",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if args.len() == 1 {
                let arg = args[0];
                let name = name.clone(); // Clone to avoid borrow issues

                match name.as_str() {
                    // sinh(0) = 0, tanh(0) = 0
                    "sinh" | "tanh" => {
                        if is_zero(ctx, arg) {
                            return Some(Rewrite::new(ctx.num(0)).desc(format!("{}(0) = 0", name)));
                        }
                    }
                    // cosh(0) = 1
                    "cosh" => {
                        if is_zero(ctx, arg) {
                            return Some(Rewrite::new(ctx.num(1)).desc("cosh(0) = 1"));
                        }
                    }
                    // asinh(0) = 0, atanh(0) = 0
                    "asinh" | "atanh" => {
                        if is_zero(ctx, arg) {
                            return Some(Rewrite::new(ctx.num(0)).desc(format!("{}(0) = 0", name)));
                        }
                    }
                    // acosh(1) = 0
                    "acosh" => {
                        if is_one(ctx, arg) {
                            return Some(Rewrite::new(ctx.num(0)).desc("acosh(1) = 0"));
                        }
                    }
                    _ => {}
                }
            }
        }
        None
    }
);

// Rule 2: Composition identities - sinh(asinh(x)) = x, etc.
// HIGH PRIORITY: Must run BEFORE TanhToSinhCoshRule - ensured by registration order
define_rule!(
    HyperbolicCompositionRule,
    "Hyperbolic Composition",
    Some(vec!["Function"]),
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Analytic
    ),
    |ctx, expr, _parent_ctx| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_args.len() == 1 {
                let inner_expr = outer_args[0];
                if let Expr::Function(inner_name, inner_args) = ctx.get(inner_expr) {
                    if inner_args.len() == 1 {
                        let x = inner_args[0];

                        // sinh(asinh(x)) = x
                        if outer_name == "sinh" && inner_name == "asinh" {
                            return Some(Rewrite::new(x).desc("sinh(asinh(x)) = x"));
                        }

                        // cosh(acosh(x)) = x
                        if outer_name == "cosh" && inner_name == "acosh" {
                            return Some(Rewrite::new(x).desc("cosh(acosh(x)) = x"));
                        }

                        // tanh(atanh(x)) = x
                        if outer_name == "tanh" && inner_name == "atanh" {
                            return Some(Rewrite::new(x).desc("tanh(atanh(x)) = x"));
                        }

                        // asinh(sinh(x)) = x
                        if outer_name == "asinh" && inner_name == "sinh" {
                            return Some(Rewrite::new(x).desc("asinh(sinh(x)) = x"));
                        }

                        // acosh(cosh(x)) = x
                        if outer_name == "acosh" && inner_name == "cosh" {
                            return Some(Rewrite::new(x).desc("acosh(cosh(x)) = x"));
                        }

                        // atanh(tanh(x)) = x
                        if outer_name == "atanh" && inner_name == "tanh" {
                            return Some(Rewrite::new(x).desc("atanh(tanh(x)) = x"));
                        }
                    }
                }
            }
        }
        None
    }
);

// Rule 3: Negative argument identities
define_rule!(
    HyperbolicNegativeRule,
    "Hyperbolic Negative Argument",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if args.len() == 1 {
                let arg = args[0];
                if let Expr::Neg(inner) = ctx.get(arg) {
                    match name.as_str() {
                        // sinh(-x) = -sinh(x) (odd function)
                        "sinh" => {
                            let sinh_inner =
                                ctx.add(Expr::Function("sinh".to_string(), vec![*inner]));
                            let new_expr = ctx.add(Expr::Neg(sinh_inner));
                            return Some(Rewrite::new(new_expr).desc("sinh(-x) = -sinh(x)"));
                        }
                        // cosh(-x) = cosh(x) (even function)
                        "cosh" => {
                            let new_expr =
                                ctx.add(Expr::Function("cosh".to_string(), vec![*inner]));
                            return Some(Rewrite::new(new_expr).desc("cosh(-x) = cosh(x)"));
                        }
                        // tanh(-x) = -tanh(x) (odd function)
                        "tanh" => {
                            let tanh_inner =
                                ctx.add(Expr::Function("tanh".to_string(), vec![*inner]));
                            let new_expr = ctx.add(Expr::Neg(tanh_inner));
                            return Some(Rewrite::new(new_expr).desc("tanh(-x) = -tanh(x)"));
                        }
                        // asinh(-x) = -asinh(x) (odd function)
                        "asinh" => {
                            let asinh_inner =
                                ctx.add(Expr::Function("asinh".to_string(), vec![*inner]));
                            let new_expr = ctx.add(Expr::Neg(asinh_inner));
                            return Some(Rewrite::new(new_expr).desc("asinh(-x) = -asinh(x)"));
                        }
                        // atanh(-x) = -atanh(x) (odd function)
                        "atanh" => {
                            let atanh_inner =
                                ctx.add(Expr::Function("atanh".to_string(), vec![*inner]));
                            let new_expr = ctx.add(Expr::Neg(atanh_inner));
                            return Some(Rewrite::new(new_expr).desc("atanh(-x) = -atanh(x)"));
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }
);

// Rule 4: Hyperbolic Pythagorean identity: cosh²(x) - sinh²(x) = 1
define_rule!(
    HyperbolicPythagoreanRule,
    "Hyperbolic Pythagorean Identity",
    Some(vec!["Sub"]),
    |ctx, expr| {
        if let Expr::Sub(l, r) = ctx.get(expr) {
            let l_data = ctx.get(*l).clone();
            let r_data = ctx.get(*r).clone();

            // Check pattern: cosh(x)^2 - sinh(x)^2
            if let (Expr::Pow(l_base, l_exp), Expr::Pow(r_base, r_exp)) = (&l_data, &r_data) {
                // Both should be squared
                if is_two(ctx, *l_exp) && is_two(ctx, *r_exp) {
                    if let (Expr::Function(l_fn, l_args), Expr::Function(r_fn, r_args)) =
                        (ctx.get(*l_base), ctx.get(*r_base))
                    {
                        // Case 1: cosh(x)^2 - sinh(x)^2 = 1
                        if l_fn == "cosh"
                            && r_fn == "sinh"
                            && l_args.len() == 1
                            && r_args.len() == 1
                        {
                            // Check if arguments are the same (semantic comparison)
                            if crate::ordering::compare_expr(ctx, l_args[0], r_args[0])
                                == Ordering::Equal
                            {
                                return Some(
                                    Rewrite::new(ctx.num(1)).desc("cosh²(x) - sinh²(x) = 1"),
                                );
                            }
                        }

                        // Case 2: sinh(x)^2 - cosh(x)^2 = -1
                        if l_fn == "sinh"
                            && r_fn == "cosh"
                            && l_args.len() == 1
                            && r_args.len() == 1
                        {
                            // Check if arguments are the same (semantic comparison)
                            if crate::ordering::compare_expr(ctx, l_args[0], r_args[0])
                                == Ordering::Equal
                            {
                                return Some(
                                    Rewrite::new(ctx.num(-1)).desc("sinh²(x) - cosh²(x) = -1"),
                                );
                            }
                        }
                    }
                }
            }
        }
        None
    }
);

// Rule 5: Hyperbolic double angle identity: cosh²(x) + sinh²(x) = cosh(2x)
// This direction collapses two squared terms into a single term, reducing complexity.
// The inverse (expansion) is not implemented to avoid loops.
define_rule!(
    HyperbolicDoubleAngleRule,
    "Hyperbolic Double Angle",
    Some(vec!["Add"]),
    |ctx, expr| {
        if let Expr::Add(l, r) = ctx.get(expr) {
            let l_data = ctx.get(*l).clone();
            let r_data = ctx.get(*r).clone();

            // Check pattern: cosh(x)^2 + sinh(x)^2 or sinh(x)^2 + cosh(x)^2
            if let (Expr::Pow(l_base, l_exp), Expr::Pow(r_base, r_exp)) = (&l_data, &r_data) {
                // Both should be squared
                if is_two(ctx, *l_exp) && is_two(ctx, *r_exp) {
                    if let (Expr::Function(l_fn, l_args), Expr::Function(r_fn, r_args)) =
                        (ctx.get(*l_base), ctx.get(*r_base))
                    {
                        // Check cosh + sinh or sinh + cosh with same argument
                        let is_cosh_sinh = l_fn == "cosh" && r_fn == "sinh";
                        let is_sinh_cosh = l_fn == "sinh" && r_fn == "cosh";

                        if (is_cosh_sinh || is_sinh_cosh) && l_args.len() == 1 && r_args.len() == 1
                        {
                            // Check if arguments are the same
                            if crate::ordering::compare_expr(ctx, l_args[0], r_args[0])
                                == Ordering::Equal
                            {
                                // Build cosh(2*x)
                                let x = l_args[0];
                                let two = ctx.num(2);
                                let two_x = ctx.add(Expr::Mul(two, x));
                                let cosh_2x =
                                    ctx.add(Expr::Function("cosh".to_string(), vec![two_x]));

                                return Some(
                                    Rewrite::new(cosh_2x).desc("cosh²(x) + sinh²(x) = cosh(2x)"),
                                );
                            }
                        }
                    }
                }
            }
        }
        None
    }
);

// Rule: tanh(x) → sinh(x) / cosh(x)
// This is the hyperbolic analogue of tan(x) → sin(x) / cos(x)
// GUARD: Skip if argument is inverse hyperbolic (let composition rule handle tanh(atanh(x)) → x)
define_rule!(
    TanhToSinhCoshRule,
    "tanh(x) = sinh(x)/cosh(x)",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if name == "tanh" && args.len() == 1 {
                let x = args[0];

                // GUARD: Don't expand if argument is inverse hyperbolic function
                // This preserves tanh(atanh(z)) → z via HyperbolicCompositionRule
                if let Expr::Function(inner_name, _) = ctx.get(x) {
                    if inner_name == "atanh" || inner_name == "asinh" || inner_name == "acosh" {
                        return None;
                    }
                }

                // GUARD: Don't expand if argument is Neg - let HyperbolicNegativeRule handle it
                // This preserves tanh(-z) → -tanh(z)
                if matches!(ctx.get(x), Expr::Neg(_)) {
                    return None;
                }

                let sinh_x = ctx.add(Expr::Function("sinh".to_string(), vec![x]));
                let cosh_x = ctx.add(Expr::Function("cosh".to_string(), vec![x]));
                let result = ctx.add(Expr::Div(sinh_x, cosh_x));
                return Some(Rewrite::new(result).desc("tanh(x) = sinh(x)/cosh(x)"));
            }
        }
        None
    }
);

// Rule: sinh(2x) → 2·sinh(x)·cosh(x)
// Expansion of double angle for sinh
define_rule!(
    SinhDoubleAngleExpansionRule,
    "sinh(2x) = 2·sinh(x)·cosh(x)",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if name == "sinh" && args.len() == 1 {
                // Check if arg is 2*x or x*2
                if let Some(inner_var) = crate::helpers::extract_double_angle_arg(ctx, args[0]) {
                    // sinh(2x) → 2·sinh(x)·cosh(x)
                    let two = ctx.num(2);
                    let sinh_x = ctx.add(Expr::Function("sinh".to_string(), vec![inner_var]));
                    let cosh_x = ctx.add(Expr::Function("cosh".to_string(), vec![inner_var]));
                    let sinh_cosh = crate::build::mul2_raw(ctx, sinh_x, cosh_x);
                    let result = crate::build::mul2_raw(ctx, two, sinh_cosh);
                    return Some(Rewrite::new(result).desc("sinh(2x) = 2·sinh(x)·cosh(x)"));
                }
            }
        }
        None
    }
);

// Rule: sinh(x) / cosh(x) → tanh(x)
// Contraction rule (inverse of TanhToSinhCoshRule) - safe direction that doesn't break composition tests
define_rule!(
    SinhCoshToTanhRule,
    "sinh(x)/cosh(x) = tanh(x)",
    Some(vec!["Div"]),
    |ctx, expr| {
        if let Expr::Div(num, den) = ctx.get(expr) {
            // Check if numerator is sinh(x) and denominator is cosh(x)
            if let Expr::Function(num_name, num_args) = ctx.get(*num) {
                if num_name == "sinh" && num_args.len() == 1 {
                    if let Expr::Function(den_name, den_args) = ctx.get(*den) {
                        if den_name == "cosh" && den_args.len() == 1 {
                            // Check if arguments are the same
                            if crate::ordering::compare_expr(ctx, num_args[0], den_args[0])
                                == std::cmp::Ordering::Equal
                            {
                                let x = num_args[0];
                                let tanh_x = ctx.add(Expr::Function("tanh".to_string(), vec![x]));
                                return Some(
                                    Rewrite::new(tanh_x).desc("sinh(x)/cosh(x) = tanh(x)"),
                                );
                            }
                        }
                    }
                }
            }
        }
        None
    }
);

// ==================== Recognize Hyperbolic From Exponential ====================

/// Helper: Check if expression is e^arg and return Some(arg), handling both Pow(e, arg) and exp(arg)
fn as_exp(ctx: &Context, id: ExprId) -> Option<ExprId> {
    match ctx.get(id) {
        // Case: e^arg (Pow with base = Constant(E))
        Expr::Pow(base, exp) => {
            if matches!(ctx.get(*base), Expr::Constant(cas_ast::Constant::E)) {
                Some(*exp)
            } else {
                None
            }
        }
        // Case: exp(arg) function
        Expr::Function(name, args) if name == "exp" && args.len() == 1 => Some(args[0]),
        _ => None,
    }
}

/// Helper: Check if arg is the negation of target (i.e., arg = -target)
fn is_negation_of(ctx: &Context, arg: ExprId, target: ExprId) -> bool {
    match ctx.get(arg) {
        // Case: Neg(inner) where inner == target
        Expr::Neg(inner) => {
            *inner == target
                || crate::ordering::compare_expr(ctx, *inner, target) == Ordering::Equal
        }
        // Case: Mul(-1, inner) where inner == target
        Expr::Mul(l, r) => {
            let minus_one = num_rational::BigRational::from_integer((-1).into());
            if let Expr::Number(n) = ctx.get(*l) {
                if *n == minus_one {
                    return *r == target
                        || crate::ordering::compare_expr(ctx, *r, target) == Ordering::Equal;
                }
            }
            if let Expr::Number(n) = ctx.get(*r) {
                if *n == minus_one {
                    return *l == target
                        || crate::ordering::compare_expr(ctx, *l, target) == Ordering::Equal;
                }
            }
            // Case: Mul(n, x) vs Mul(-n, x) - e.g., 2*x vs -2*x
            // Check if target is also a Mul with negated coefficient
            if let Expr::Mul(tl, tr) = ctx.get(target) {
                // Try: arg = Mul(n, x), target = Mul(-n, x)
                if let (Expr::Number(n_arg), Expr::Number(n_target)) = (ctx.get(*l), ctx.get(*tl)) {
                    if n_arg == &(-n_target.clone())
                        && (*r == *tr
                            || crate::ordering::compare_expr(ctx, *r, *tr) == Ordering::Equal)
                    {
                        return true;
                    }
                }
                if let (Expr::Number(n_arg), Expr::Number(n_target)) = (ctx.get(*l), ctx.get(*tr)) {
                    if n_arg == &(-n_target.clone())
                        && (*r == *tl
                            || crate::ordering::compare_expr(ctx, *r, *tl) == Ordering::Equal)
                    {
                        return true;
                    }
                }
                if let (Expr::Number(n_arg), Expr::Number(n_target)) = (ctx.get(*r), ctx.get(*tl)) {
                    if n_arg == &(-n_target.clone())
                        && (*l == *tr
                            || crate::ordering::compare_expr(ctx, *l, *tr) == Ordering::Equal)
                    {
                        return true;
                    }
                }
                if let (Expr::Number(n_arg), Expr::Number(n_target)) = (ctx.get(*r), ctx.get(*tr)) {
                    if n_arg == &(-n_target.clone())
                        && (*l == *tl
                            || crate::ordering::compare_expr(ctx, *l, *tl) == Ordering::Equal)
                    {
                        return true;
                    }
                }
            }
            false
        }
        _ => false,
    }
}

/// Helper: Check if expression equals 1/2
fn is_half(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        *n == num_rational::BigRational::new(1.into(), 2.into())
    } else {
        false
    }
}

/// Try to extract the pair (exp(a), exp(-a)) from an Add or Sub expression.
/// Returns Some((arg, is_positive)) where:
/// - arg is the argument 'a' (such that we have exp(a) and exp(-a))
/// - is_positive is true for Add (cosh pattern), false for Sub (sinh pattern)
/// - For Sub: also returns whether the positive exp came first (affects sinh sign)
fn extract_exp_pair(ctx: &Context, id: ExprId) -> Option<(ExprId, bool, bool)> {
    // Try Add: exp(a) + exp(-a) -> cosh(a)
    if let Expr::Add(l, r) = ctx.get(id) {
        // Try l = exp(a), r = exp(-a)
        if let (Some(l_arg), Some(r_arg)) = (as_exp(ctx, *l), as_exp(ctx, *r)) {
            if is_negation_of(ctx, r_arg, l_arg) {
                return Some((l_arg, true, true)); // cosh pattern
            }
            if is_negation_of(ctx, l_arg, r_arg) {
                return Some((r_arg, true, true)); // cosh pattern (order swapped)
            }
        }
    }

    // Try Sub: exp(a) - exp(-a) -> sinh(a), or exp(-a) - exp(a) -> -sinh(a)
    if let Expr::Sub(l, r) = ctx.get(id) {
        if let (Some(l_arg), Some(r_arg)) = (as_exp(ctx, *l), as_exp(ctx, *r)) {
            // exp(a) - exp(-a) = sinh(a) * 2, positive_first = true
            if is_negation_of(ctx, r_arg, l_arg) {
                return Some((l_arg, false, true)); // sinh pattern, positive
            }
            // exp(-a) - exp(a) = -sinh(a) * 2, positive_first = false
            if is_negation_of(ctx, l_arg, r_arg) {
                return Some((r_arg, false, false)); // sinh pattern, negative
            }
        }
    }

    // Try Add with Neg: Add(l, Neg(r)) which is like Sub(l, r)
    if let Expr::Add(l, r) = ctx.get(id) {
        if let Expr::Neg(neg_inner) = ctx.get(*r) {
            if let (Some(l_arg), Some(r_arg)) = (as_exp(ctx, *l), as_exp(ctx, *neg_inner)) {
                if is_negation_of(ctx, r_arg, l_arg) {
                    return Some((l_arg, false, true)); // sinh pattern
                }
                if is_negation_of(ctx, l_arg, r_arg) {
                    return Some((r_arg, false, false)); // -sinh pattern
                }
            }
        }
        if let Expr::Neg(neg_inner) = ctx.get(*l) {
            if let (Some(l_arg), Some(r_arg)) = (as_exp(ctx, *neg_inner), as_exp(ctx, *r)) {
                if is_negation_of(ctx, l_arg, r_arg) {
                    return Some((r_arg, false, true)); // sinh pattern (r - l)
                }
                if is_negation_of(ctx, r_arg, l_arg) {
                    return Some((l_arg, false, false)); // -sinh pattern
                }
            }
        }
    }

    None
}

// Rule 5: Recognize hyperbolic functions from exponential definitions
// (e^x + e^(-x))/2 → cosh(x)
// (e^x - e^(-x))/2 → sinh(x)
// (e^(-x) - e^x)/2 → -sinh(x)
define_rule!(
    RecognizeHyperbolicFromExpRule,
    "Recognize Hyperbolic from Exponential",
    importance: crate::step::ImportanceLevel::Medium,
    |ctx, expr| {
        // Pattern 1: Div(sum_or_diff, 2)
        if let Expr::Div(num, den) = ctx.get(expr) {
            if is_two(ctx, *den) {
                if let Some((arg, is_cosh, positive_first)) = extract_exp_pair(ctx, *num) {
                    if is_cosh {
                        // (e^x + e^(-x))/2 = cosh(x)
                        let cosh = ctx.add(Expr::Function("cosh".to_string(), vec![arg]));
                        return Some(Rewrite::new(cosh).desc("(e^x + e^(-x))/2 = cosh(x)"));
                    } else if positive_first {
                        // (e^x - e^(-x))/2 = sinh(x)
                        let sinh = ctx.add(Expr::Function("sinh".to_string(), vec![arg]));
                        return Some(Rewrite::new(sinh).desc("(e^x - e^(-x))/2 = sinh(x)"));
                    } else {
                        // (e^(-x) - e^x)/2 = -sinh(x)
                        let sinh = ctx.add(Expr::Function("sinh".to_string(), vec![arg]));
                        let neg_sinh = ctx.add(Expr::Neg(sinh));
                        return Some(Rewrite::new(neg_sinh).desc("(e^(-x) - e^x)/2 = -sinh(x)"));
                    }
                }
            }
        }

        // Pattern 2: Mul(1/2, sum_or_diff) or Mul(sum_or_diff, 1/2)
        if let Expr::Mul(l, r) = ctx.get(expr) {
            let (half_id, sum_id) = if is_half(ctx, *l) {
                (*l, *r)
            } else if is_half(ctx, *r) {
                (*r, *l)
            } else {
                return None;
            };
            let _ = half_id; // suppress unused warning

            if let Some((arg, is_cosh, positive_first)) = extract_exp_pair(ctx, sum_id) {
                if is_cosh {
                    let cosh = ctx.add(Expr::Function("cosh".to_string(), vec![arg]));
                    return Some(Rewrite::new(cosh).desc("(e^x + e^(-x))/2 = cosh(x)"));
                } else if positive_first {
                    let sinh = ctx.add(Expr::Function("sinh".to_string(), vec![arg]));
                    return Some(Rewrite::new(sinh).desc("(e^x - e^(-x))/2 = sinh(x)"));
                } else {
                    let sinh = ctx.add(Expr::Function("sinh".to_string(), vec![arg]));
                    let neg_sinh = ctx.add(Expr::Neg(sinh));
                    return Some(Rewrite::new(neg_sinh).desc("(e^(-x) - e^x)/2 = -sinh(x)"));
                }
            }
        }

        // Pattern 3: (e^x - e^(-x)) / (e^x + e^(-x)) → tanh(x)
        // This is sinh(x)/cosh(x) without the 1/2 factors (they cancel)
        if let Expr::Div(num, den) = ctx.get(expr) {
            // Check if numerator is e^x - e^(-x) pattern (sinh-like)
            if let Some((num_arg, false, num_positive_first)) = extract_exp_pair(ctx, *num) {
                // Check if denominator is e^x + e^(-x) pattern (cosh-like)
                if let Some((den_arg, true, _)) = extract_exp_pair(ctx, *den) {
                    // Arguments must be the same
                    if crate::ordering::compare_expr(ctx, num_arg, den_arg)
                        == std::cmp::Ordering::Equal
                    {
                        let tanh_x = ctx.add(Expr::Function("tanh".to_string(), vec![num_arg]));
                        if num_positive_first {
                            return Some(
                                Rewrite::new(tanh_x)
                                    .desc("(e^x - e^(-x))/(e^x + e^(-x)) = tanh(x)"),
                            );
                        } else {
                            // (e^(-x) - e^x) / (e^x + e^(-x)) = -tanh(x)
                            let neg_tanh = ctx.add(Expr::Neg(tanh_x));
                            return Some(
                                Rewrite::new(neg_tanh)
                                    .desc("(e^(-x) - e^x)/(e^x + e^(-x)) = -tanh(x)"),
                            );
                        }
                    }
                }
            }
        }

        None
    }
);

/// Register all hyperbolic function rules
pub fn register(simplifier: &mut crate::engine::Simplifier) {
    simplifier.add_rule(Box::new(EvaluateHyperbolicRule));
    simplifier.add_rule(Box::new(HyperbolicCompositionRule));
    simplifier.add_rule(Box::new(HyperbolicNegativeRule));
    simplifier.add_rule(Box::new(HyperbolicPythagoreanRule));
    simplifier.add_rule(Box::new(HyperbolicDoubleAngleRule));
    // DISABLED: TanhToSinhCoshRule breaks tanh(atanh(x))→x and tanh(-x)→-tanh(x) paths
    // simplifier.add_rule(Box::new(TanhToSinhCoshRule)); // tanh(x) → sinh(x)/cosh(x)
    simplifier.add_rule(Box::new(SinhCoshToTanhRule)); // sinh(x)/cosh(x) → tanh(x) (contraction)
    simplifier.add_rule(Box::new(SinhDoubleAngleExpansionRule)); // sinh(2x) → 2sinh(x)cosh(x)
    simplifier.add_rule(Box::new(RecognizeHyperbolicFromExpRule));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::DisplayExpr;

    #[test]
    fn test_recognize_cosh_from_exp() {
        // (e^x + e^(-x))/2 -> cosh(x)
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let neg_x = ctx.add(Expr::Neg(x));
        let exp_x = ctx.add(Expr::Pow(e, x));
        let e2 = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let exp_neg_x = ctx.add(Expr::Pow(e2, neg_x));
        let sum = ctx.add(Expr::Add(exp_x, exp_neg_x));
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Div(sum, two));

        let rule = RecognizeHyperbolicFromExpRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_some(), "Should recognize cosh(x)");
        let result = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.unwrap().new_expr
            }
        );
        assert_eq!(result, "cosh(x)");
    }

    #[test]
    fn test_recognize_sinh_from_exp() {
        // (e^x - e^(-x))/2 -> sinh(x)
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let neg_x = ctx.add(Expr::Neg(x));
        let exp_x = ctx.add(Expr::Pow(e, x));
        let e2 = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let exp_neg_x = ctx.add(Expr::Pow(e2, neg_x));
        let diff = ctx.add(Expr::Sub(exp_x, exp_neg_x));
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Div(diff, two));

        let rule = RecognizeHyperbolicFromExpRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_some(), "Should recognize sinh(x)");
        let result = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.unwrap().new_expr
            }
        );
        assert_eq!(result, "sinh(x)");
    }

    #[test]
    fn test_recognize_neg_sinh_from_exp() {
        // (e^(-x) - e^x)/2 -> -sinh(x)
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let neg_x = ctx.add(Expr::Neg(x));
        let exp_x = ctx.add(Expr::Pow(e, x));
        let e2 = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let exp_neg_x = ctx.add(Expr::Pow(e2, neg_x));
        // Note: order is reversed
        let diff = ctx.add(Expr::Sub(exp_neg_x, exp_x));
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Div(diff, two));

        let rule = RecognizeHyperbolicFromExpRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_some(), "Should recognize -sinh(x)");
        let result = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.unwrap().new_expr
            }
        );
        assert!(
            result.contains("sinh") && result.contains("-"),
            "Should be -sinh(x), got: {}",
            result
        );
    }

    #[test]
    fn test_no_match_different_args() {
        // (e^x + e^(-y))/2 should NOT match (different args)
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let neg_y = ctx.add(Expr::Neg(y));
        let exp_x = ctx.add(Expr::Pow(e, x));
        let e2 = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let exp_neg_y = ctx.add(Expr::Pow(e2, neg_y));
        let sum = ctx.add(Expr::Add(exp_x, exp_neg_y));
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Div(sum, two));

        let rule = RecognizeHyperbolicFromExpRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_none(), "Should NOT match different args");
    }

    #[test]
    fn test_no_match_wrong_divisor() {
        // (e^x + e^(-x))/3 should NOT match (not divided by 2)
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let neg_x = ctx.add(Expr::Neg(x));
        let exp_x = ctx.add(Expr::Pow(e, x));
        let e2 = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let exp_neg_x = ctx.add(Expr::Pow(e2, neg_x));
        let sum = ctx.add(Expr::Add(exp_x, exp_neg_x));
        let three = ctx.num(3);
        let expr = ctx.add(Expr::Div(sum, three));

        let rule = RecognizeHyperbolicFromExpRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_none(), "Should NOT match divisor != 2");
    }

    #[test]
    fn test_hyperbolic_double_angle_rule() {
        // cosh(x)^2 + sinh(x)^2 -> cosh(2*x)
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let cosh_x = ctx.add(Expr::Function("cosh".to_string(), vec![x]));
        let sinh_x = ctx.add(Expr::Function("sinh".to_string(), vec![x]));
        let two = ctx.num(2);
        let two2 = ctx.num(2);
        let cosh_sq = ctx.add(Expr::Pow(cosh_x, two));
        let sinh_sq = ctx.add(Expr::Pow(sinh_x, two2));
        let expr = ctx.add(Expr::Add(cosh_sq, sinh_sq));

        let rule = HyperbolicDoubleAngleRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_some(), "Should apply cosh²+sinh² -> cosh(2x)");
        let result = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.unwrap().new_expr
            }
        );
        assert!(
            result.contains("cosh") && result.contains("2"),
            "Should be cosh(2*x), got: {}",
            result
        );
    }
}
