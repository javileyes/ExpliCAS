//! Implicit Domain Inference.
//!
//! This module infers domain constraints that are implicitly required by
//! expression structure. For example, `sqrt(x)` in RealOnly mode implies `x ≥ 0`.
//!
//! # Key Concepts
//!
//! - **Implicit Domain**: Constraints derived from expression structure, not assumptions.
//! - **Witness**: The subexpression that enforces a constraint (e.g., `sqrt(x)` for `x ≥ 0`).
//! - **Witness Survival**: A constraint is only valid if its witness survives in the output.
//!
//! # Usage
//!
//! ```ignore
//! let implicit = infer_implicit_domain(ctx, root, ValueDomain::RealOnly);
//! // Later, when checking if x ≥ 0 is valid:
//! if implicit.contains_nonnegative(x) && witness_survives(ctx, x, output, WitnessKind::Sqrt) {
//!     // Can use ProvenImplicit
//! }
//! ```

use crate::semantics::ValueDomain;
use cas_ast::{Context, Expr, ExprId};
use num_integer::Integer;
use num_rational::BigRational;
use std::collections::HashSet;

// =============================================================================
// Implicit Condition Types
// =============================================================================

/// An implicit condition inferred from expression structure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ImplicitCondition {
    /// x ≥ 0 (from sqrt(x) or x^(1/2))
    NonNegative(ExprId),
    /// x > 0 (from ln(x) or log(x))  
    Positive(ExprId),
    /// x ≠ 0 (from 1/x or Div(_, x))
    NonZero(ExprId),
}

/// Set of implicit conditions inferred from an expression.
#[derive(Debug, Clone, Default)]
pub struct ImplicitDomain {
    conditions: HashSet<ImplicitCondition>,
}

impl ImplicitDomain {
    /// Create an empty implicit domain.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Check if domain is empty.
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }

    /// Check if an expression has an implicit NonNegative constraint.
    pub fn contains_nonnegative(&self, expr: ExprId) -> bool {
        self.conditions
            .contains(&ImplicitCondition::NonNegative(expr))
    }

    /// Check if an expression has an implicit Positive constraint.
    pub fn contains_positive(&self, expr: ExprId) -> bool {
        self.conditions.contains(&ImplicitCondition::Positive(expr))
    }

    /// Check if an expression has an implicit NonZero constraint.
    pub fn contains_nonzero(&self, expr: ExprId) -> bool {
        self.conditions.contains(&ImplicitCondition::NonZero(expr))
    }

    /// Add a NonNegative condition.
    fn add_nonnegative(&mut self, expr: ExprId) {
        self.conditions.insert(ImplicitCondition::NonNegative(expr));
    }

    /// Add a Positive condition.
    fn add_positive(&mut self, expr: ExprId) {
        self.conditions.insert(ImplicitCondition::Positive(expr));
    }

    /// Add a NonZero condition.
    fn add_nonzero(&mut self, expr: ExprId) {
        self.conditions.insert(ImplicitCondition::NonZero(expr));
    }
}

// =============================================================================
// Inference
// =============================================================================

/// Infer implicit domain constraints from expression structure.
///
/// Only operates in RealOnly mode. Returns empty in ComplexEnabled.
///
/// Traverses the AST and collects:
/// - `sqrt(t)` or `t^(1/2)` → NonNegative(t)
/// - `ln(t)` or `log(t)` → Positive(t)
/// - `1/t` or `Div(_, t)` → NonZero(t)
pub fn infer_implicit_domain(ctx: &Context, root: ExprId, vd: ValueDomain) -> ImplicitDomain {
    // Only apply in RealOnly mode
    if vd != ValueDomain::RealOnly {
        return ImplicitDomain::empty();
    }

    let mut domain = ImplicitDomain::default();
    infer_recursive(ctx, root, &mut domain);
    domain
}

fn infer_recursive(ctx: &Context, expr: ExprId, domain: &mut ImplicitDomain) {
    match ctx.get(expr) {
        // sqrt(t) → NonNegative(t)
        Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => {
            domain.add_nonnegative(args[0]);
            infer_recursive(ctx, args[0], domain);
        }

        // ln(t) or log(t) → Positive(t)
        Expr::Function(name, args) if (name == "ln" || name == "log") && args.len() == 1 => {
            domain.add_positive(args[0]);
            infer_recursive(ctx, args[0], domain);
        }

        // t^(1/2) or t^(p/q) where q is even → NonNegative(t)
        Expr::Pow(base, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                // Check if denominator is 2 (sqrt) or any even number (even root)
                if is_even_root_exponent(n) {
                    domain.add_nonnegative(*base);
                }
            }
            infer_recursive(ctx, *base, domain);
            infer_recursive(ctx, *exp, domain);
        }

        // Div(_, t) → NonZero(t)
        Expr::Div(num, den) => {
            domain.add_nonzero(*den);
            infer_recursive(ctx, *num, domain);
            infer_recursive(ctx, *den, domain);
        }

        // Recursively process children
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            infer_recursive(ctx, *l, domain);
            infer_recursive(ctx, *r, domain);
        }
        Expr::Neg(inner) => {
            infer_recursive(ctx, *inner, domain);
        }
        Expr::Function(_, args) => {
            for arg in args {
                infer_recursive(ctx, *arg, domain);
            }
        }

        // Leaf nodes: nothing to infer
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}

        Expr::Matrix { data, .. } => {
            for elem in data {
                infer_recursive(ctx, *elem, domain);
            }
        }
    }
}

/// Check if an exponent represents an even root (e.g., 1/2, 1/4, 3/4).
fn is_even_root_exponent(n: &BigRational) -> bool {
    let denom = n.denom();
    // Check if denominator is even (2, 4, 6, ...)
    denom.is_even()
}

// =============================================================================
// Witness Survival
// =============================================================================

/// Kind of witness to look for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WitnessKind {
    /// sqrt(t) or t^(1/2) for NonNegative(t)
    Sqrt,
    /// ln(t) or log(t) for Positive(t)
    Log,
    /// 1/t or Div(_, t) for NonZero(t)
    Division,
}

/// Check if a witness for a condition survives in the output expression.
///
/// This is the critical safety check: a condition from implicit domain
/// is only valid if the witness that enforces it still exists in the output.
///
/// # Arguments
/// * `ctx` - AST context
/// * `target` - The target expression (e.g., `x` in `x ≥ 0`)
/// * `output` - The expression to search for witnesses
/// * `kind` - What kind of witness to look for
///
/// # Returns
/// `true` if a witness survives in output, `false` otherwise.
pub fn witness_survives(ctx: &Context, target: ExprId, output: ExprId, kind: WitnessKind) -> bool {
    search_witness(ctx, target, output, kind)
}

fn search_witness(ctx: &Context, target: ExprId, expr: ExprId, kind: WitnessKind) -> bool {
    match ctx.get(expr) {
        // Check if this node is a witness
        Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => {
            if kind == WitnessKind::Sqrt && exprs_equal(ctx, args[0], target) {
                return true;
            }
            search_witness(ctx, target, args[0], kind)
        }

        Expr::Function(name, args) if (name == "ln" || name == "log") && args.len() == 1 => {
            if kind == WitnessKind::Log && exprs_equal(ctx, args[0], target) {
                return true;
            }
            search_witness(ctx, target, args[0], kind)
        }

        Expr::Pow(base, exp) => {
            // Check for t^(1/2) form as witness for sqrt
            if kind == WitnessKind::Sqrt {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if is_even_root_exponent(n) && exprs_equal(ctx, *base, target) {
                        return true;
                    }
                }
            }
            search_witness(ctx, target, *base, kind) || search_witness(ctx, target, *exp, kind)
        }

        Expr::Div(num, den) => {
            if kind == WitnessKind::Division && exprs_equal(ctx, *den, target) {
                return true;
            }
            search_witness(ctx, target, *num, kind) || search_witness(ctx, target, *den, kind)
        }

        // Recursively search children
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            search_witness(ctx, target, *l, kind) || search_witness(ctx, target, *r, kind)
        }
        Expr::Neg(inner) => search_witness(ctx, target, *inner, kind),
        Expr::Function(_, args) => args.iter().any(|a| search_witness(ctx, target, *a, kind)),

        // Leaf nodes
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => false,

        Expr::Matrix { data, .. } => data.iter().any(|e| search_witness(ctx, target, *e, kind)),
    }
}

/// Check if two expressions are equal (by ExprId or structural comparison).
fn exprs_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    if a == b {
        return true;
    }
    // Use ordering comparison for structural equality
    crate::ordering::compare_expr(ctx, a, b) == std::cmp::Ordering::Equal
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;

    #[test]
    fn test_infer_sqrt_implies_nonnegative() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sqrt_x = ctx.add(Expr::Function("sqrt".to_string(), vec![x]));

        let domain = infer_implicit_domain(&ctx, sqrt_x, ValueDomain::RealOnly);

        assert!(domain.contains_nonnegative(x));
        assert!(!domain.contains_positive(x));
    }

    #[test]
    fn test_infer_ln_implies_positive() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let ln_x = ctx.add(Expr::Function("ln".to_string(), vec![x]));

        let domain = infer_implicit_domain(&ctx, ln_x, ValueDomain::RealOnly);

        assert!(domain.contains_positive(x));
    }

    #[test]
    fn test_infer_div_implies_nonzero() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let one_over_x = ctx.add(Expr::Div(one, x));

        let domain = infer_implicit_domain(&ctx, one_over_x, ValueDomain::RealOnly);

        assert!(domain.contains_nonzero(x));
    }

    #[test]
    fn test_witness_survives_sqrt() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sqrt_x = ctx.add(Expr::Function("sqrt".to_string(), vec![x]));
        let y = ctx.var("y");
        let output = ctx.add(Expr::Add(sqrt_x, y)); // sqrt(x) + y

        assert!(witness_survives(&ctx, x, output, WitnessKind::Sqrt));
    }

    #[test]
    fn test_witness_not_survives() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        // Output is just x, no sqrt(x) witness

        assert!(!witness_survives(&ctx, x, x, WitnessKind::Sqrt));
    }

    #[test]
    fn test_complex_enabled_returns_empty() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sqrt_x = ctx.add(Expr::Function("sqrt".to_string(), vec![x]));

        let domain = infer_implicit_domain(&ctx, sqrt_x, ValueDomain::ComplexEnabled);

        assert!(domain.is_empty());
    }
}
