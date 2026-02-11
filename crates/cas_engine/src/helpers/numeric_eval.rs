// ========== Numeric Evaluation ==========

use cas_ast::{Context, Expr, ExprId};

/// Default depth limit for numeric evaluation.
/// Prevents stack overflow on deeply nested expressions.
pub const DEFAULT_NUMERIC_EVAL_DEPTH: usize = 50;

/// Extract a rational constant from an expression, handling multiple representations.
/// Uses default depth limit (50) to prevent stack overflow.
///
/// Supports (all must be purely numeric - returns None if any variable/function present):
/// - `Number(n)` - direct rational
/// - `Div(a, b)` - fraction (recursive)
/// - `Neg(a)` - negation (recursive)
/// - `Mul(a, b)` - product (recursive)
/// - `Add(a, b)` - sum (recursive)
/// - `Sub(a, b)` - difference (recursive)
///
/// This is the canonical helper for numeric evaluation. Used by:
/// - `SemanticEqualityChecker::try_evaluate_numeric`
/// - `EvaluatePowerRule` for exponent matching
pub(crate) fn as_rational_const(ctx: &Context, expr: ExprId) -> Option<num_rational::BigRational> {
    as_rational_const_depth(ctx, expr, DEFAULT_NUMERIC_EVAL_DEPTH)
}

/// Extract a rational constant with explicit depth limit.
/// Returns None if depth is exhausted (prevents stack overflow on deep expressions).
pub(crate) fn as_rational_const_depth(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
) -> Option<num_rational::BigRational> {
    use num_traits::Zero;

    if depth == 0 {
        return None; // Depth budget exhausted
    }

    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),

        Expr::Div(num, den) => {
            let n = as_rational_const_depth(ctx, *num, depth - 1)?;
            let d = as_rational_const_depth(ctx, *den, depth - 1)?;
            if !d.is_zero() {
                Some(n / d)
            } else {
                None
            }
        }

        Expr::Neg(inner) => {
            let val = as_rational_const_depth(ctx, *inner, depth - 1)?;
            Some(-val)
        }

        Expr::Mul(l, r) => {
            let lv = as_rational_const_depth(ctx, *l, depth - 1)?;
            let rv = as_rational_const_depth(ctx, *r, depth - 1)?;
            Some(lv * rv)
        }

        Expr::Add(l, r) => {
            let lv = as_rational_const_depth(ctx, *l, depth - 1)?;
            let rv = as_rational_const_depth(ctx, *r, depth - 1)?;
            Some(lv + rv)
        }

        Expr::Sub(l, r) => {
            let lv = as_rational_const_depth(ctx, *l, depth - 1)?;
            let rv = as_rational_const_depth(ctx, *r, depth - 1)?;
            Some(lv - rv)
        }

        // Variables, Constants, Functions, Pow, Matrix -> not purely numeric
        _ => None,
    }
}

/// Check if an expression contains an integral (for auto-context detection).
///
/// Searches the expression tree for `integrate(...)` function calls.
/// Uses iterative traversal to avoid stack overflow on deep expressions.
pub(crate) fn contains_integral(ctx: &Context, root: ExprId) -> bool {
    let mut stack = vec![root];

    while let Some(e) = stack.pop() {
        match ctx.get(e) {
            Expr::Function(name, args) => {
                let fn_name = ctx.sym_name(*name);
                if fn_name == "integrate" || fn_name == "int" {
                    return true;
                }
                for arg in args {
                    stack.push(*arg);
                }
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => {
                stack.push(*inner);
            }
            Expr::Matrix { data, .. } => {
                for elem in data {
                    stack.push(*elem);
                }
            }
            Expr::Div(num, den) => {
                stack.push(*num);
                stack.push(*den);
            }
            // Leaf nodes: nothing to push
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
            // Hold is transparent - traverse inner
            Expr::Hold(inner) => stack.push(*inner),
        }
    }

    false
}

/// Check if an expression contains the imaginary unit `i` anywhere.
/// Check if an expression contains the imaginary unit `i` or imaginary-producing expressions.
/// Detects: Constant::I, sqrt(-1), (-1)^(1/2), and similar patterns.
/// Uses iterative traversal to avoid stack overflow on deep expressions.
pub(crate) fn contains_i(ctx: &Context, root: ExprId) -> bool {
    let mut stack = vec![root];

    while let Some(e) = stack.pop() {
        match ctx.get(e) {
            Expr::Constant(c) if *c == cas_ast::Constant::I => {
                return true;
            }
            // Check for sqrt(-1) pattern
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 =>
            {
                if is_negative_one(ctx, args[0]) {
                    return true;
                }
                // Still need to traverse the arg for nested i
                stack.push(args[0]);
            }
            // Check for (-1)^(1/2) pattern
            Expr::Pow(base, exp) => {
                if is_negative_one(ctx, *base) && is_one_half(ctx, *exp) {
                    return true;
                }
                stack.push(*base);
                stack.push(*exp);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => {
                stack.push(*inner);
            }
            Expr::Function(_, args) => {
                for arg in args {
                    stack.push(*arg);
                }
            }
            Expr::Matrix { data, .. } => {
                for elem in data {
                    stack.push(*elem);
                }
            }
            Expr::Div(num, den) => {
                stack.push(*num);
                stack.push(*den);
            }
            // Leaf nodes: nothing to push
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
            // Hold is transparent - traverse inner
            Expr::Hold(inner) => stack.push(*inner),
        }
    }

    false
}

/// Check if an expression represents -1
fn is_negative_one(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n == num_rational::BigRational::from_integer((-1).into()),
        Expr::Neg(inner) => {
            matches!(
                ctx.get(*inner),
                Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into())
            )
        }
        _ => false,
    }
}

/// Check if an expression represents 1/2
fn is_one_half(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n == num_rational::BigRational::new(1.into(), 2.into()),
        Expr::Div(num, den) => {
            matches!((ctx.get(*num), ctx.get(*den)),
                (Expr::Number(n), Expr::Number(d))
                if *n == num_rational::BigRational::from_integer(1.into())
                && *d == num_rational::BigRational::from_integer(2.into())
            )
        }
        _ => false,
    }
}

/// Probabilistic polynomial zero check via numeric evaluation.
///
/// Tests whether an expression is identically zero by substituting several
/// random rational values for each variable and evaluating. If ALL evaluations
/// yield 0, the expression is extremely likely to be identically zero.
///
/// This handles cases where `expand()` produces raw unsimplified AST (e.g.,
/// `u·u + u·1 - u² - u`) that structural comparison can't match.
///
/// Uses 3 independent probe points. For a non-zero polynomial of degree d in
/// n variables, the probability of a false positive is at most d/|S| per
/// probe, where |S| is the evaluation domain size. With rational probes
/// chosen from {2/3, 3/5, 5/7, 7/11, 11/13, ...}, false positives are
/// astronomically unlikely.
///
/// Returns `true` if the expression evaluates to 0 at all probe points.
/// Returns `false` if any evaluation is non-zero or evaluation fails.
pub(crate) fn numeric_poly_zero_check(ctx: &Context, expr: ExprId) -> bool {
    use num_rational::BigRational;
    use num_traits::Zero;

    // Collect variables in the expression
    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.is_empty() {
        // No variables: direct numeric evaluation
        if let Some(v) = as_rational_const(ctx, expr) {
            return v.is_zero();
        }
        // Fallback: try f64 evaluation for constant expressions with surds
        if let Some(v) = eval_f64(ctx, expr) {
            return v.abs() < 1e-10;
        }
        return false;
    }

    // Limit to reasonable number of variables (avoid combinatorial explosion)
    if vars.len() > 5 {
        return false;
    }

    // Probe points: distinct rationals unlikely to be roots of spurious polynomials
    let probes: Vec<Vec<BigRational>> = vec![
        // Probe 1: small primes/coprime rationals
        vec![
            BigRational::new(2.into(), 3.into()),
            BigRational::new(5.into(), 7.into()),
            BigRational::new(3.into(), 11.into()),
            BigRational::new(7.into(), 13.into()),
            BigRational::new(11.into(), 17.into()),
        ],
        // Probe 2: different values
        vec![
            BigRational::new(3.into(), 5.into()),
            BigRational::new(7.into(), 11.into()),
            BigRational::new(11.into(), 13.into()),
            BigRational::new(13.into(), 17.into()),
            BigRational::new(17.into(), 19.into()),
        ],
        // Probe 3: yet another set
        vec![
            BigRational::new(5.into(), 3.into()),
            BigRational::new(11.into(), 7.into()),
            BigRational::new(13.into(), 11.into()),
            BigRational::new(17.into(), 13.into()),
            BigRational::new(19.into(), 17.into()),
        ],
    ];

    let var_list: Vec<String> = vars.into_iter().collect();

    // Try exact BigRational evaluation first
    let mut exact_failed = false;
    for probe_set in &probes {
        let result = eval_with_substitution(ctx, expr, &var_list, probe_set);
        match result {
            Some(val) => {
                if !val.is_zero() {
                    return false; // Non-zero at this point => not identically zero
                }
            }
            None => {
                exact_failed = true;
                break;
            }
        }
    }

    if !exact_failed {
        return true; // Zero at all probe points (exact)
    }

    // Fallback: f64 evaluation for expressions with surds/fractional exponents
    let f64_probes: Vec<Vec<f64>> = vec![
        vec![2.0 / 3.0, 5.0 / 7.0, 3.0 / 11.0, 7.0 / 13.0, 11.0 / 17.0],
        vec![3.0 / 5.0, 7.0 / 11.0, 11.0 / 13.0, 13.0 / 17.0, 17.0 / 19.0],
        vec![5.0 / 3.0, 11.0 / 7.0, 13.0 / 11.0, 17.0 / 13.0, 19.0 / 17.0],
    ];

    for probe_set in &f64_probes {
        let result = eval_f64_with_substitution(ctx, expr, &var_list, probe_set);
        match result {
            Some(val) => {
                if val.abs() > 1e-10 {
                    return false; // Non-zero at this point
                }
            }
            None => return false, // Evaluation failed
        }
    }

    true // Near-zero at all probe points
}

/// Evaluate an expression by substituting rational values for variables.
/// Returns None if evaluation fails (e.g., division by zero, unsupported operation).
fn eval_with_substitution(
    ctx: &Context,
    expr: ExprId,
    var_names: &[String],
    values: &[num_rational::BigRational],
) -> Option<num_rational::BigRational> {
    use num_rational::BigRational;
    use num_traits::Zero;

    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),

        Expr::Variable(v) => {
            // Resolve variable name and look up in substitution map
            let name = ctx.sym_name(*v);
            var_names
                .iter()
                .position(|var_name| var_name == name)
                .and_then(|idx| values.get(idx).cloned())
        }

        Expr::Constant(c) => {
            // Use approximate rational value for constants
            match c {
                cas_ast::Constant::Pi => Some(BigRational::new(355.into(), 113.into())),
                cas_ast::Constant::E => Some(BigRational::new(193.into(), 71.into())),
                _ => None, // I, Infinity, Undefined, Phi: can't evaluate rationally
            }
        }

        Expr::Add(l, r) => {
            let lv = eval_with_substitution(ctx, *l, var_names, values)?;
            let rv = eval_with_substitution(ctx, *r, var_names, values)?;
            Some(lv + rv)
        }

        Expr::Sub(l, r) => {
            let lv = eval_with_substitution(ctx, *l, var_names, values)?;
            let rv = eval_with_substitution(ctx, *r, var_names, values)?;
            Some(lv - rv)
        }

        Expr::Mul(l, r) => {
            let lv = eval_with_substitution(ctx, *l, var_names, values)?;
            let rv = eval_with_substitution(ctx, *r, var_names, values)?;
            Some(lv * rv)
        }

        Expr::Div(n, d) => {
            let nv = eval_with_substitution(ctx, *n, var_names, values)?;
            let dv = eval_with_substitution(ctx, *d, var_names, values)?;
            if dv.is_zero() {
                None // Division by zero
            } else {
                Some(nv / dv)
            }
        }

        Expr::Neg(inner) => {
            let v = eval_with_substitution(ctx, *inner, var_names, values)?;
            Some(-v)
        }

        Expr::Pow(base, exp) => {
            let bv = eval_with_substitution(ctx, *base, var_names, values)?;
            let ev = eval_with_substitution(ctx, *exp, var_names, values)?;
            // Only handle integer exponents for exact computation
            if ev.is_integer() {
                let n: i64 = ev.to_integer().try_into().ok()?;
                if (0..=20).contains(&n) {
                    let mut result = BigRational::from_integer(1.into());
                    for _ in 0..n {
                        result *= &bv;
                    }
                    Some(result)
                } else if (-20..0).contains(&n) && !bv.is_zero() {
                    let mut result = BigRational::from_integer(1.into());
                    for _ in 0..(-n) {
                        result *= &bv;
                    }
                    Some(BigRational::from_integer(1.into()) / result)
                } else {
                    None
                }
            } else {
                None // Non-integer exponent: can't compute exactly
            }
        }

        // Functions, Hold, Matrix, SessionRef: bail out
        _ => None,
    }
}

/// Evaluate a constant expression (no variables) to f64.
/// Handles fractional exponents via `f64::powf()`, enabling evaluation of surds like `3^(1/2)`.
fn eval_f64(ctx: &Context, expr: ExprId) -> Option<f64> {
    eval_f64_with_substitution(ctx, expr, &[], &[])
}

/// Evaluate an expression by substituting f64 values for variables.
/// Supports fractional exponents via `f64::powf()`.
/// Returns None if evaluation fails (division by zero, NaN, unsupported operations).
fn eval_f64_with_substitution(
    ctx: &Context,
    expr: ExprId,
    var_names: &[String],
    values: &[f64],
) -> Option<f64> {
    match ctx.get(expr) {
        Expr::Number(n) => {
            use num_traits::ToPrimitive;
            let f = n.numer().to_f64()? / n.denom().to_f64()?;
            if f.is_finite() {
                Some(f)
            } else {
                None
            }
        }

        Expr::Variable(v) => {
            let name = ctx.sym_name(*v);
            var_names
                .iter()
                .position(|var_name| var_name == name)
                .and_then(|idx| values.get(idx).copied())
        }

        Expr::Constant(c) => match c {
            cas_ast::Constant::Pi => Some(std::f64::consts::PI),
            cas_ast::Constant::E => Some(std::f64::consts::E),
            cas_ast::Constant::Phi => Some(1.618033988749895),
            _ => None,
        },

        Expr::Add(l, r) => {
            let lv = eval_f64_with_substitution(ctx, *l, var_names, values)?;
            let rv = eval_f64_with_substitution(ctx, *r, var_names, values)?;
            let result = lv + rv;
            if result.is_finite() {
                Some(result)
            } else {
                None
            }
        }

        Expr::Sub(l, r) => {
            let lv = eval_f64_with_substitution(ctx, *l, var_names, values)?;
            let rv = eval_f64_with_substitution(ctx, *r, var_names, values)?;
            let result = lv - rv;
            if result.is_finite() {
                Some(result)
            } else {
                None
            }
        }

        Expr::Mul(l, r) => {
            let lv = eval_f64_with_substitution(ctx, *l, var_names, values)?;
            let rv = eval_f64_with_substitution(ctx, *r, var_names, values)?;
            let result = lv * rv;
            if result.is_finite() {
                Some(result)
            } else {
                None
            }
        }

        Expr::Div(n, d) => {
            let nv = eval_f64_with_substitution(ctx, *n, var_names, values)?;
            let dv = eval_f64_with_substitution(ctx, *d, var_names, values)?;
            if dv.abs() < 1e-15 {
                return None;
            } // Avoid division by near-zero
            let result = nv / dv;
            if result.is_finite() {
                Some(result)
            } else {
                None
            }
        }

        Expr::Neg(inner) => {
            let v = eval_f64_with_substitution(ctx, *inner, var_names, values)?;
            Some(-v)
        }

        Expr::Pow(base, exp) => {
            let bv = eval_f64_with_substitution(ctx, *base, var_names, values)?;
            let ev = eval_f64_with_substitution(ctx, *exp, var_names, values)?;
            let result = bv.powf(ev);
            if result.is_finite() {
                Some(result)
            } else {
                None
            }
        }

        Expr::Function(fn_id, args) => {
            let name = ctx.sym_name(*fn_id);
            match name {
                "sqrt" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    if av >= 0.0 {
                        let result = av.sqrt();
                        if result.is_finite() {
                            Some(result)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                "abs" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    Some(av.abs())
                }
                "sin" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    Some(av.sin())
                }
                "cos" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    Some(av.cos())
                }
                "tan" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    let result = av.tan();
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }
                "ln" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    if av > 0.0 {
                        let result = av.ln();
                        if result.is_finite() {
                            Some(result)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                "log" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    if av > 0.0 {
                        let result = av.log10();
                        if result.is_finite() {
                            Some(result)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                "exp" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    let result = av.exp();
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }
                "sinh" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    let result = av.sinh();
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }
                "cosh" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    let result = av.cosh();
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }
                "tanh" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    Some(av.tanh())
                }
                "cot" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    let s = av.sin();
                    if s.abs() < 1e-15 {
                        return None;
                    }
                    let result = av.cos() / s;
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }
                "sec" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    let c = av.cos();
                    if c.abs() < 1e-15 {
                        return None;
                    }
                    let result = 1.0 / c;
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }
                "csc" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    let s = av.sin();
                    if s.abs() < 1e-15 {
                        return None;
                    }
                    let result = 1.0 / s;
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }
                _ => None, // Unsupported function
            }
        }

        Expr::Hold(inner) => eval_f64_with_substitution(ctx, *inner, var_names, values),

        // Matrix, SessionRef: bail out
        _ => None,
    }
}
