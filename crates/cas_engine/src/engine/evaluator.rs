//! Numeric evaluation of expressions using f64 values.
//!
//! Used for numeric property testing to verify rewrite correctness
//! and equivalence checking via numeric substitution.

use cas_ast::{Context, Expr, ExprId};
use num_traits::ToPrimitive;
use std::collections::HashMap;

// =============================================================================
// Checked evaluator for robust numeric testing
// =============================================================================

/// Error types for checked numeric evaluation.
/// Provides detailed information about why evaluation failed.
#[derive(Debug, Clone, PartialEq)]
pub enum EvalCheckedError {
    /// Denominator is too close to zero (likely a pole)
    NearPole {
        /// Operation that caused the pole (e.g., "Div", "Tan", "Sec")
        op: &'static str,
        /// Denominator value
        denom: f64,
        /// Threshold that was exceeded
        threshold: f64,
    },
    /// Division by exactly zero
    DivisionByZero { op: &'static str },
    /// Domain error (e.g., log of non-positive, sqrt of negative in RealOnly)
    Domain { function: String, arg: f64 },
    /// Result is not finite (NaN or Inf)
    NonFinite,
    /// Depth limit exceeded
    DepthExceeded,
    /// Variable not found in var_map
    UnboundVariable { name: String },
    /// Unsupported expression type
    Unsupported,
}

/// Options for checked evaluation.
#[derive(Debug, Clone)]
pub struct EvalCheckedOptions {
    /// Absolute epsilon for near-zero denominator detection in Div
    pub zero_abs_eps: f64,
    /// Relative epsilon for near-zero denominator detection in Div
    pub zero_rel_eps: f64,
    /// Epsilon for trigonometric pole detection (tan, sec, csc, cot)
    /// Should be larger than zero_abs_eps due to floating point errors near π/2
    pub trig_pole_eps: f64,
    /// Maximum recursion depth
    pub max_depth: usize,
}

impl Default for EvalCheckedOptions {
    fn default() -> Self {
        Self {
            zero_abs_eps: 1e-12,
            zero_rel_eps: 1e-12,
            trig_pole_eps: 1e-9, // Larger for trig due to FP errors near π/2
            max_depth: 200,
        }
    }
}

/// Evaluate an expression numerically with f64 values.
/// Used for numeric property testing to verify rewrite correctness.
/// Has a depth limit of 200 to prevent stack overflow on deeply nested expressions.
pub fn eval_f64(ctx: &Context, expr: ExprId, var_map: &HashMap<String, f64>) -> Option<f64> {
    eval_f64_depth(ctx, expr, var_map, 200)
}

/// Evaluate an expression numerically with detailed error reporting.
/// Used for robust numeric testing where we need to distinguish between:
/// - Near-pole singularities (denominator close to zero)
/// - Domain errors (log of negative, etc.)
/// - Other evaluation failures
pub fn eval_f64_checked(
    ctx: &Context,
    expr: ExprId,
    var_map: &HashMap<String, f64>,
    opts: &EvalCheckedOptions,
) -> Result<f64, EvalCheckedError> {
    eval_f64_checked_depth(ctx, expr, var_map, opts, opts.max_depth)
}

/// Internal checked evaluator with depth tracking.
fn eval_f64_checked_depth(
    ctx: &Context,
    expr: ExprId,
    var_map: &HashMap<String, f64>,
    opts: &EvalCheckedOptions,
    depth: usize,
) -> Result<f64, EvalCheckedError> {
    if depth == 0 {
        return Err(EvalCheckedError::DepthExceeded);
    }

    let result = match ctx.get(expr) {
        Expr::Number(n) => n.to_f64().ok_or(EvalCheckedError::NonFinite)?,

        Expr::Variable(sym_id) => {
            let name = ctx.sym_name(*sym_id);
            *var_map
                .get(name)
                .ok_or_else(|| EvalCheckedError::UnboundVariable {
                    name: name.to_string(),
                })?
        }

        Expr::Add(l, r) => {
            eval_f64_checked_depth(ctx, *l, var_map, opts, depth - 1)?
                + eval_f64_checked_depth(ctx, *r, var_map, opts, depth - 1)?
        }

        Expr::Sub(l, r) => {
            eval_f64_checked_depth(ctx, *l, var_map, opts, depth - 1)?
                - eval_f64_checked_depth(ctx, *r, var_map, opts, depth - 1)?
        }

        Expr::Mul(l, r) => {
            eval_f64_checked_depth(ctx, *l, var_map, opts, depth - 1)?
                * eval_f64_checked_depth(ctx, *r, var_map, opts, depth - 1)?
        }

        Expr::Div(l, r) => {
            // Evaluate denominator first to check for near-pole
            let b = eval_f64_checked_depth(ctx, *r, var_map, opts, depth - 1)?;

            if !b.is_finite() {
                return Err(EvalCheckedError::NonFinite);
            }

            if b == 0.0 {
                return Err(EvalCheckedError::DivisionByZero { op: "Div" });
            }

            // Evaluate numerator for relative threshold calculation
            let a = eval_f64_checked_depth(ctx, *l, var_map, opts, depth - 1)?;

            if !a.is_finite() {
                return Err(EvalCheckedError::NonFinite);
            }

            // Check for near-pole: |b| <= eps_abs + eps_rel * max(1, |a|)
            let scale = f64::max(1.0, a.abs());
            let threshold = opts.zero_abs_eps + opts.zero_rel_eps * scale;

            if b.abs() <= threshold {
                return Err(EvalCheckedError::NearPole {
                    op: "Div",
                    denom: b,
                    threshold,
                });
            }

            a / b
        }

        Expr::Pow(b, e) => {
            let base = eval_f64_checked_depth(ctx, *b, var_map, opts, depth - 1)?;
            let exp = eval_f64_checked_depth(ctx, *e, var_map, opts, depth - 1)?;

            // Domain check: base < 0 and non-integer exponent -> complex result
            if base < 0.0 && exp.fract() != 0.0 {
                return Err(EvalCheckedError::Domain {
                    function: "pow".to_string(),
                    arg: base,
                });
            }

            base.powf(exp)
        }

        Expr::Neg(e) => -eval_f64_checked_depth(ctx, *e, var_map, opts, depth - 1)?,

        Expr::Function(fn_id, args) => {
            eval_function_checked(ctx, ctx.sym_name(*fn_id), args, var_map, opts, depth)?
        }

        Expr::Constant(c) => match c {
            cas_ast::Constant::Pi => std::f64::consts::PI,
            cas_ast::Constant::E => std::f64::consts::E,
            cas_ast::Constant::Phi => 1.618033988749895, // (1+√5)/2
            cas_ast::Constant::Infinity => return Err(EvalCheckedError::NonFinite),
            cas_ast::Constant::Undefined => return Err(EvalCheckedError::NonFinite),
            cas_ast::Constant::I => {
                return Err(EvalCheckedError::Domain {
                    function: "constant".to_string(),
                    arg: 0.0,
                })
            }
        },

        Expr::Matrix { .. } | Expr::SessionRef(_) => {
            return Err(EvalCheckedError::Unsupported);
        }
        // Hold is transparent for evaluation - unwrap and evaluate inner
        Expr::Hold(inner) => eval_f64_checked_depth(ctx, *inner, var_map, opts, depth - 1)?,
    };

    // Final check for non-finite result
    if !result.is_finite() {
        return Err(EvalCheckedError::NonFinite);
    }

    Ok(result)
}

/// Evaluate functions with domain checking.
#[inline(never)]
fn eval_function_checked(
    ctx: &Context,
    name: &str,
    args: &[ExprId],
    var_map: &HashMap<String, f64>,
    opts: &EvalCheckedOptions,
    depth: usize,
) -> Result<f64, EvalCheckedError> {
    // Evaluate all arguments
    let arg_vals: Result<Vec<f64>, _> = args
        .iter()
        .map(|a| eval_f64_checked_depth(ctx, *a, var_map, opts, depth - 1))
        .collect();
    let arg_vals = arg_vals?;

    match name {
        // Basic trig - check for tan/sec/csc poles via cos/sin near zero
        "sin" => Ok(arg_vals.first().copied().unwrap_or(0.0).sin()),
        "cos" => Ok(arg_vals.first().copied().unwrap_or(0.0).cos()),
        "tan" => {
            let x = arg_vals.first().copied().unwrap_or(0.0);
            let cos_x = x.cos();
            let threshold = opts.trig_pole_eps;
            if cos_x.abs() <= threshold {
                return Err(EvalCheckedError::NearPole {
                    op: "Tan",
                    denom: cos_x,
                    threshold,
                });
            }
            Ok(x.tan())
        }

        // Reciprocal trig with pole detection
        "sec" => {
            let x = arg_vals.first().copied().unwrap_or(0.0);
            let cos_x = x.cos();
            let threshold = opts.trig_pole_eps;
            if cos_x.abs() <= threshold {
                return Err(EvalCheckedError::NearPole {
                    op: "Sec",
                    denom: cos_x,
                    threshold,
                });
            }
            Ok(1.0 / cos_x)
        }
        "csc" => {
            let x = arg_vals.first().copied().unwrap_or(0.0);
            let sin_x = x.sin();
            let threshold = opts.trig_pole_eps;
            if sin_x.abs() <= threshold {
                return Err(EvalCheckedError::NearPole {
                    op: "Csc",
                    denom: sin_x,
                    threshold,
                });
            }
            Ok(1.0 / sin_x)
        }
        "cot" => {
            let x = arg_vals.first().copied().unwrap_or(0.0);
            let sin_x = x.sin();
            let threshold = opts.trig_pole_eps;
            if sin_x.abs() <= threshold {
                return Err(EvalCheckedError::NearPole {
                    op: "Cot",
                    denom: sin_x,
                    threshold,
                });
            }
            Ok(1.0 / x.tan())
        }

        // Inverse trig
        "asin" | "arcsin" => Ok(arg_vals.first().copied().unwrap_or(0.0).asin()),
        "acos" | "arccos" => Ok(arg_vals.first().copied().unwrap_or(0.0).acos()),
        "atan" | "arctan" => Ok(arg_vals.first().copied().unwrap_or(0.0).atan()),

        // Hyperbolic
        "sinh" => Ok(arg_vals.first().copied().unwrap_or(0.0).sinh()),
        "cosh" => Ok(arg_vals.first().copied().unwrap_or(0.0).cosh()),
        "tanh" => Ok(arg_vals.first().copied().unwrap_or(0.0).tanh()),
        "asinh" | "arcsinh" => Ok(arg_vals.first().copied().unwrap_or(0.0).asinh()),
        "acosh" | "arccosh" => Ok(arg_vals.first().copied().unwrap_or(0.0).acosh()),
        "atanh" | "arctanh" => Ok(arg_vals.first().copied().unwrap_or(0.0).atanh()),

        // Logarithm with domain checking
        "ln" => {
            let arg = arg_vals.first().copied().unwrap_or(0.0);
            if arg <= 0.0 {
                return Err(EvalCheckedError::Domain {
                    function: "ln".to_string(),
                    arg,
                });
            }
            Ok(arg.ln())
        }
        "log" => {
            if arg_vals.len() == 2 {
                let base = arg_vals[0];
                let arg = arg_vals[1];
                if base <= 0.0 || base == 1.0 {
                    return Err(EvalCheckedError::Domain {
                        function: "log_base".to_string(),
                        arg: base,
                    });
                }
                if arg <= 0.0 {
                    return Err(EvalCheckedError::Domain {
                        function: "log_arg".to_string(),
                        arg,
                    });
                }
                Ok(arg.ln() / base.ln())
            } else if arg_vals.len() == 1 {
                let arg = arg_vals[0];
                if arg <= 0.0 {
                    return Err(EvalCheckedError::Domain {
                        function: "log10".to_string(),
                        arg,
                    });
                }
                Ok(arg.log10())
            } else {
                Err(EvalCheckedError::Unsupported)
            }
        }

        // Exponential
        "exp" => Ok(arg_vals.first().copied().unwrap_or(0.0).exp()),

        // Square root with domain checking
        "sqrt" => {
            let arg = arg_vals.first().copied().unwrap_or(0.0);
            if arg < 0.0 {
                return Err(EvalCheckedError::Domain {
                    function: "sqrt".to_string(),
                    arg,
                });
            }
            Ok(arg.sqrt())
        }

        // Other functions
        "abs" => Ok(arg_vals.first().copied().unwrap_or(0.0).abs()),
        "floor" => Ok(arg_vals.first().copied().unwrap_or(0.0).floor()),
        "ceil" => Ok(arg_vals.first().copied().unwrap_or(0.0).ceil()),
        "round" => Ok(arg_vals.first().copied().unwrap_or(0.0).round()),
        "sign" | "sgn" => Ok(arg_vals.first().copied().unwrap_or(0.0).signum()),

        // Hold barrier is transparent for numeric evaluation
        _ if cas_ast::hold::is_hold_name(name) => {
            if let Some(&arg_id) = args.first() {
                eval_f64_checked_depth(ctx, arg_id, var_map, opts, depth - 1)
            } else {
                Err(EvalCheckedError::Unsupported)
            }
        }

        _ => Err(EvalCheckedError::Unsupported),
    }
}

/// Internal eval_f64 with explicit depth limit.
fn eval_f64_depth(
    ctx: &Context,
    expr: ExprId,
    var_map: &HashMap<String, f64>,
    depth: usize,
) -> Option<f64> {
    if depth == 0 {
        return None; // Depth budget exhausted
    }

    match ctx.get(expr) {
        Expr::Number(n) => n.to_f64(),
        Expr::Variable(sym_id) => var_map.get(ctx.sym_name(*sym_id)).cloned(),
        Expr::Add(l, r) => Some(
            eval_f64_depth(ctx, *l, var_map, depth - 1)?
                + eval_f64_depth(ctx, *r, var_map, depth - 1)?,
        ),
        Expr::Sub(l, r) => Some(
            eval_f64_depth(ctx, *l, var_map, depth - 1)?
                - eval_f64_depth(ctx, *r, var_map, depth - 1)?,
        ),
        Expr::Mul(l, r) => Some(
            eval_f64_depth(ctx, *l, var_map, depth - 1)?
                * eval_f64_depth(ctx, *r, var_map, depth - 1)?,
        ),
        Expr::Div(l, r) => Some(
            eval_f64_depth(ctx, *l, var_map, depth - 1)?
                / eval_f64_depth(ctx, *r, var_map, depth - 1)?,
        ),
        Expr::Pow(b, e) => Some(
            eval_f64_depth(ctx, *b, var_map, depth - 1)?.powf(eval_f64_depth(
                ctx,
                *e,
                var_map,
                depth - 1,
            )?),
        ),
        Expr::Neg(e) => Some(-eval_f64_depth(ctx, *e, var_map, depth - 1)?),
        Expr::Function(fn_id, args) => {
            let arg_vals: Option<Vec<f64>> = args
                .iter()
                .map(|a| eval_f64_depth(ctx, *a, var_map, depth - 1))
                .collect();
            let arg_vals = arg_vals?;
            match ctx.sym_name(*fn_id) {
                // Basic trig
                "sin" => Some(arg_vals.first()?.sin()),
                "cos" => Some(arg_vals.first()?.cos()),
                "tan" => Some(arg_vals.first()?.tan()),

                // Reciprocal trig
                "sec" => Some(1.0 / arg_vals.first()?.cos()),
                "csc" => Some(1.0 / arg_vals.first()?.sin()),
                "cot" => Some(1.0 / arg_vals.first()?.tan()),

                // Inverse trig
                "asin" | "arcsin" => Some(arg_vals.first()?.asin()),
                "acos" | "arccos" => Some(arg_vals.first()?.acos()),
                "atan" | "arctan" => Some(arg_vals.first()?.atan()),

                // Hyperbolic
                "sinh" => Some(arg_vals.first()?.sinh()),
                "cosh" => Some(arg_vals.first()?.cosh()),
                "tanh" => Some(arg_vals.first()?.tanh()),

                // Inverse hyperbolic
                "asinh" | "arcsinh" => Some(arg_vals.first()?.asinh()),
                "acosh" | "arccosh" => Some(arg_vals.first()?.acosh()),
                "atanh" | "arctanh" => Some(arg_vals.first()?.atanh()),

                // Exponential and logarithm
                "exp" => Some(arg_vals.first()?.exp()),
                "ln" => Some(arg_vals.first()?.ln()),
                // log(base, arg) -> ln(arg) / ln(base)
                "log" => {
                    if arg_vals.len() == 2 {
                        let base = arg_vals[0];
                        let arg = arg_vals[1];
                        Some(arg.ln() / base.ln())
                    } else if arg_vals.len() == 1 {
                        // log(x) = log base 10
                        Some(arg_vals[0].log10())
                    } else {
                        None
                    }
                }

                // Other
                "sqrt" => Some(arg_vals.first()?.sqrt()),
                "abs" => Some(arg_vals.first()?.abs()),
                "floor" => Some(arg_vals.first()?.floor()),
                "ceil" => Some(arg_vals.first()?.ceil()),
                "round" => Some(arg_vals.first()?.round()),
                "sign" | "sgn" => Some(arg_vals.first()?.signum()),

                // Hold barrier is transparent for numeric evaluation
                // This fallback uses is_builtin because fn_id is in scope
                _ if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Hold) => {
                    if args.len() == 1 {
                        eval_f64_depth(ctx, args[0], var_map, depth - 1)
                    } else {
                        None
                    }
                }

                _ => None,
            }
        }
        Expr::Constant(c) => match c {
            cas_ast::Constant::Pi => Some(std::f64::consts::PI),
            cas_ast::Constant::E => Some(std::f64::consts::E),
            cas_ast::Constant::Phi => Some(1.618033988749895), // (1+√5)/2
            cas_ast::Constant::Infinity => Some(f64::INFINITY),
            cas_ast::Constant::Undefined => Some(f64::NAN),
            cas_ast::Constant::I => None, // Imaginary unit cannot be evaluated to f64
        },
        Expr::Matrix { .. } => None, // Matrix evaluation not supported in f64
        Expr::SessionRef(_) => None, // SessionRef should be resolved before eval
        // Hold is transparent for evaluation - unwrap and evaluate inner
        Expr::Hold(inner) => eval_f64_depth(ctx, *inner, var_map, depth - 1),
    }
}
