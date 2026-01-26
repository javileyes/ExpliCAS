// ========== Solver Domain Helpers ==========

use super::predicates::prove_positive;
use cas_ast::{Context, Expr, ExprId};
use num_traits::ToPrimitive;

/// Decision result for `can_take_ln_real`.
///
/// Used by the solver to determine if ln(arg) is valid in RealOnly mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LnDecision {
    /// Argument is provably positive - ln() is safe with no assumption.
    Safe,
    /// Argument positivity is unknown but allowed under assumption (Assume mode only).
    /// Contains the assumption message to be emitted.
    AssumePositive,
}

/// Check if ln(arg) is valid in RealOnly mode.
///
/// This is used by the solver to gate log operations on exponential equations.
///
/// # Arguments
/// * `ctx` - Expression context
/// * `arg` - The argument to ln()
/// * `mode` - The current DomainMode
/// * `value_domain` - RealOnly or ComplexEnabled
///
/// # Returns
/// * `Ok(LnDecision::Safe)` if arg is provably positive (no assumption needed)
/// * `Ok(LnDecision::AssumePositive)` if allowed with assumption (Assume mode only)
/// * `Err(reason)` if ln is invalid (arg ≤ 0 proven, or unknown in Strict/Generic)
///
/// # Examples
/// ```ignore
/// can_take_ln_real(ctx, ctx.num(2), DomainMode::Strict, RealOnly)   // Ok(Safe)
/// can_take_ln_real(ctx, ctx.num(-5), DomainMode::Strict, RealOnly)  // Err("argument ≤ 0")
/// can_take_ln_real(ctx, ctx.var("x"), DomainMode::Strict, RealOnly) // Err("cannot prove > 0")
/// can_take_ln_real(ctx, ctx.var("x"), DomainMode::Assume, RealOnly) // Ok(AssumePositive)
/// ```
pub fn can_take_ln_real(
    ctx: &Context,
    arg: ExprId,
    mode: crate::domain::DomainMode,
    value_domain: crate::semantics::ValueDomain,
) -> Result<LnDecision, &'static str> {
    use crate::domain::{DomainMode, Proof};

    let proof = prove_positive(ctx, arg, value_domain);

    match proof {
        Proof::Proven | Proof::ProvenImplicit => Ok(LnDecision::Safe),
        Proof::Disproven => Err("argument is ≤ 0"),
        Proof::Unknown => match mode {
            DomainMode::Strict | DomainMode::Generic => Err("cannot prove argument > 0 for ln()"),
            DomainMode::Assume => Ok(LnDecision::AssumePositive),
        },
    }
}

/// Try to extract an integer value from an expression.
///
/// Returns `None` if:
/// - Expression is not a Number
/// - Number is not an integer (has non-1 denominator)
/// - Integer value doesn't fit in `i64`
///
/// For BigInt extraction without i64 limitations, use `get_integer_exact`.
pub fn get_integer(ctx: &Context, expr: ExprId) -> Option<i64> {
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_integer() {
            n.to_integer().to_i64()
        } else {
            None
        }
    } else {
        None
    }
}

/// Extract an integer value from an expression as BigInt.
///
/// Unlike `get_integer`, this:
/// - Returns the full BigInt without i64 truncation
/// - Also handles `Neg(e)` by recursively extracting and negating
///
/// Use this for number theory operations where large integers are expected.
pub fn get_integer_exact(ctx: &Context, expr: ExprId) -> Option<num_bigint::BigInt> {
    match ctx.get(expr) {
        Expr::Number(n) => {
            if n.is_integer() {
                Some(n.to_integer())
            } else {
                None
            }
        }
        Expr::Neg(e) => get_integer_exact(ctx, *e).map(|n| -n),
        _ => None,
    }
}

/// Get the variant name of an expression (for debugging/display)
pub fn get_variant_name(expr: &Expr) -> &'static str {
    match expr {
        Expr::Number(_) => "Number",
        Expr::Variable(_) => "Variable",
        Expr::Constant(_) => "Constant",
        Expr::Add(_, _) => "Add",
        Expr::Sub(_, _) => "Sub",
        Expr::Mul(_, _) => "Mul",
        Expr::Div(_, _) => "Div",
        Expr::Pow(_, _) => "Pow",
        Expr::Neg(_) => "Neg",
        Expr::Function(_, _) => "Function",
        Expr::Matrix { .. } => "Matrix",
        Expr::SessionRef(_) => "SessionRef",
        Expr::Hold(_) => "Hold",
    }
}
