//! Best-So-Far Simplifier Guard
//!
//! Prevents simplification from returning expressions worse than the input,
//! while preferring rationalized forms within a controlled budget.
//!
//! # Design
//!
//! After each phase (Core, Transform, Rationalize, PostCleanup), we compare
//! the current expression against the best seen so far. The "best" is chosen
//! by a lexicographic score that prioritizes:
//!
//! 1. Fewer sqrt/roots in denominators (rationalization quality)
//! 2. Fewer nested divisions (cleaner fractions)
//! 3. Fewer additions in denominators (simpler denoms)
//! 4. Fewer nodes (smaller expression)
//!
//! A hard budget guard ensures we never exceed `input_nodes + max_extra_nodes`.

use crate::helpers::node_count;
use crate::Step;
use cas_ast::{Context, Expr, ExprId};
use num_traits::One;

/// Budget configuration for best-so-far tracking.
#[derive(Clone, Copy, Debug)]
pub struct BestSoFarBudget {
    /// Maximum additional nodes allowed beyond input size.
    /// Default: 8 (allows modest expansion for rationalization)
    pub max_extra_nodes: usize,
}

impl Default for BestSoFarBudget {
    fn default() -> Self {
        Self { max_extra_nodes: 8 }
    }
}

/// Score for comparing expression quality.
/// Lower values are better. Ordering is lexicographic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Score {
    /// Count of sqrt/roots in denominators (fewer = better)
    pub sqrt_in_den: u16,
    /// Count of nested Div expressions (fewer = better)
    pub nested_div: u16,
    /// Whether any denominator contains Add (false = better)
    pub add_in_den: bool,
    /// Total node count (fewer = better)
    pub nodes: usize,
}

impl Score {
    /// Tuple key for lexicographic comparison (lower is better)
    #[inline]
    fn key(&self) -> (u16, u16, bool, usize) {
        (
            self.sqrt_in_den,
            self.nested_div,
            self.add_in_den,
            self.nodes,
        )
    }
}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key().cmp(&other.key())
    }
}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Compute the score for an expression.
pub fn score_expr(ctx: &Context, expr: ExprId) -> Score {
    Score {
        sqrt_in_den: count_sqrt_in_den(ctx, expr),
        nested_div: count_nested_div(ctx, expr),
        add_in_den: has_add_in_den(ctx, expr),
        nodes: node_count(ctx, expr),
    }
}

/// Count sqrt/root expressions in denominators.
fn count_sqrt_in_den(ctx: &Context, expr: ExprId) -> u16 {
    let mut count = 0u16;
    let mut stack = vec![(expr, false)]; // (expr, in_denominator)

    while let Some((id, in_den)) = stack.pop() {
        match ctx.get(id) {
            Expr::Div(num, den) => {
                stack.push((*num, false));
                stack.push((*den, true)); // denominator context
            }
            Expr::Pow(base, exp) => {
                // Check if this is a root: Pow(_, 1/n) or Pow(_, Number < 1)
                let is_root = match ctx.get(*exp) {
                    Expr::Div(n, d) => {
                        matches!(ctx.get(*n), Expr::Number(num) if num.is_one())
                            && matches!(ctx.get(*d), Expr::Number(_))
                    }
                    Expr::Number(r) => {
                        *r > num_rational::BigRational::from_integer(0.into())
                            && *r < num_rational::BigRational::from_integer(1.into())
                    }
                    _ => false,
                };

                if in_den && is_root {
                    count = count.saturating_add(1);
                }
                stack.push((*base, in_den));
                stack.push((*exp, in_den));
            }
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt) && in_den =>
            {
                count = count.saturating_add(1);
                for arg in args {
                    stack.push((*arg, in_den));
                }
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                stack.push((*l, in_den));
                stack.push((*r, in_den));
            }
            Expr::Neg(inner) => {
                stack.push((*inner, in_den));
            }
            Expr::Function(_, args) => {
                for arg in args {
                    stack.push((*arg, false)); // function args reset denominator context
                }
            }
            _ => {}
        }
    }
    count
}

/// Count nested Div expressions (Div inside Div).
fn count_nested_div(ctx: &Context, expr: ExprId) -> u16 {
    let mut count = 0u16;
    let mut stack = vec![(expr, false)]; // (expr, inside_div)

    while let Some((id, inside_div)) = stack.pop() {
        match ctx.get(id) {
            Expr::Div(num, den) => {
                if inside_div {
                    count = count.saturating_add(1);
                }
                stack.push((*num, true));
                stack.push((*den, true));
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Pow(l, r) => {
                stack.push((*l, inside_div));
                stack.push((*r, inside_div));
            }
            Expr::Neg(inner) => {
                stack.push((*inner, inside_div));
            }
            Expr::Function(_, args) => {
                for arg in args {
                    stack.push((*arg, false));
                }
            }
            _ => {}
        }
    }
    count
}

/// Check if any denominator contains Add/Sub.
fn has_add_in_den(ctx: &Context, expr: ExprId) -> bool {
    let mut stack = vec![(expr, false)]; // (expr, in_denominator)

    while let Some((id, in_den)) = stack.pop() {
        match ctx.get(id) {
            Expr::Div(num, den) => {
                stack.push((*num, false));
                stack.push((*den, true));
            }
            Expr::Add(l, r) | Expr::Sub(l, r) => {
                if in_den {
                    return true;
                }
                stack.push((*l, in_den));
                stack.push((*r, in_den));
            }
            Expr::Mul(l, r) | Expr::Pow(l, r) => {
                stack.push((*l, in_den));
                stack.push((*r, in_den));
            }
            Expr::Neg(inner) => {
                stack.push((*inner, in_den));
            }
            Expr::Function(_, args) => {
                for arg in args {
                    stack.push((*arg, false));
                }
            }
            _ => {}
        }
    }
    false
}

/// Tracker for best expression seen during simplification.
pub struct BestSoFar {
    baseline_nodes: usize,
    budget: BestSoFarBudget,
    best_expr: ExprId,
    best_score: Score,
    best_steps: Vec<Step>,
}

impl BestSoFar {
    /// Create a new tracker initialized with the input expression.
    pub fn new(input: ExprId, steps: &[Step], ctx: &Context, budget: BestSoFarBudget) -> Self {
        let score = score_expr(ctx, input);
        Self {
            baseline_nodes: score.nodes,
            budget,
            best_expr: input,
            best_score: score,
            best_steps: steps.to_vec(),
        }
    }

    /// Check if candidate is within budget.
    #[inline]
    fn admissible(&self, cand_nodes: usize) -> bool {
        cand_nodes <= self.baseline_nodes + self.budget.max_extra_nodes
    }

    /// Consider a candidate expression. Updates best if candidate is better and admissible.
    pub fn consider(&mut self, cand_expr: ExprId, all_steps: &[Step], ctx: &Context) {
        let cand_score = score_expr(ctx, cand_expr);

        // Hard guard: reject if exceeds budget
        if !self.admissible(cand_score.nodes) {
            return;
        }

        // Soft preference: accept if strictly better
        if cand_score < self.best_score {
            self.best_expr = cand_expr;
            self.best_score = cand_score;
            self.best_steps = all_steps.to_vec();
        }
    }

    /// Extract the best expression and steps.
    pub fn into_parts(self) -> (ExprId, Vec<Step>) {
        (self.best_expr, self.best_steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_ordering() {
        // Score with sqrt in den is worse than without
        let s1 = Score {
            sqrt_in_den: 1,
            nested_div: 0,
            add_in_den: false,
            nodes: 10,
        };
        let s2 = Score {
            sqrt_in_den: 0,
            nested_div: 0,
            add_in_den: false,
            nodes: 15,
        };
        assert!(
            s2 < s1,
            "Fewer sqrt_in_den should be better even with more nodes"
        );

        // Score with nested div is worse
        let s3 = Score {
            sqrt_in_den: 0,
            nested_div: 2,
            add_in_den: false,
            nodes: 10,
        };
        let s4 = Score {
            sqrt_in_den: 0,
            nested_div: 0,
            add_in_den: false,
            nodes: 12,
        };
        assert!(
            s4 < s3,
            "Fewer nested_div should be better even with more nodes"
        );
    }

    #[test]
    fn test_budget_enforcement() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let budget = BestSoFarBudget { max_extra_nodes: 5 };
        let tracker = BestSoFar::new(x, &[], &ctx, budget);

        // Expression with 1 node should be admissible (1 <= 1 + 5)
        assert!(tracker.admissible(1));
        assert!(tracker.admissible(6)); // 6 <= 1 + 5
        assert!(!tracker.admissible(7)); // 7 > 1 + 5
    }
}
