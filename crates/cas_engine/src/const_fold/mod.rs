//! Constant folding module - allowlist-only evaluation of constant subtrees.
//!
//! # Safety Contract
//!
//! This module NEVER:
//! - Calls simplify, rationalize, expand, or poly operations
//! - Makes domain assumptions (x≠0, x>0)
//! - Touches variables or non-constant subtrees
//!
//! It ONLY folds fully-constant subtrees using a strict allowlist.
//!
//! # Entry Point
//!
//! The only public entry point is [`fold_constants`], which uses an iterative
//! postorder walker to avoid stack overflow on deep trees.

mod helpers;

use crate::budget::{Budget, Metric, Operation};
use crate::semantics::EvalConfig;
use crate::CasError;
use cas_ast::{Context, Expr, ExprId};

/// Constant folding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConstFoldMode {
    /// No constant folding (default, preserves all expressions).
    #[default]
    Off,
    /// Safe constant folding - only allowlist operations on provably constant subtrees.
    Safe,
}

/// Constant folding result with statistics.
#[derive(Debug, Clone)]
pub struct ConstFoldResult {
    /// The resulting expression (may be same as input if no folding occurred).
    pub expr: ExprId,
    /// Number of nodes created during folding.
    pub nodes_created: u64,
    /// Number of fold operations performed.
    pub folds_performed: u64,
}

/// Fold constants in an expression using allowlist-only operations.
///
/// This function walks the expression tree in postorder (iteratively, to avoid
/// stack overflow) and folds constant subtrees according to the allowlist.
///
/// # Arguments
///
/// * `ctx` - The AST context
/// * `expr` - The expression to fold
/// * `cfg` - Semantic configuration (controls ValueDomain, etc.)
/// * `mode` - Folding mode (Off or Safe)
/// * `budget` - Budget for resource limits
///
/// # Returns
///
/// `Ok(ConstFoldResult)` with the folded expression and stats, or `Err` on budget exceeded.
///
/// # Safety
///
/// This function is designed to never change semantics in unexpected ways:
/// - Off mode: returns input unchanged
/// - Safe mode: only folds what can be proven safe
pub fn fold_constants(
    ctx: &mut Context,
    expr: ExprId,
    cfg: &EvalConfig,
    mode: ConstFoldMode,
    budget: &mut Budget,
) -> Result<ConstFoldResult, CasError> {
    // Off mode: no-op
    if mode == ConstFoldMode::Off {
        return Ok(ConstFoldResult {
            expr,
            nodes_created: 0,
            folds_performed: 0,
        });
    }

    // Budget check at start
    budget.charge(Operation::ConstFold, Metric::Iterations, 1)?;

    // Iterative postorder traversal
    let mut result = IterativeFolder::new(ctx, cfg);
    let out_expr = result.fold(expr, budget)?;

    Ok(ConstFoldResult {
        expr: out_expr,
        nodes_created: result.nodes_created,
        folds_performed: result.folds_performed,
    })
}

/// Stack frame for iterative postorder traversal.
#[derive(Clone, Copy)]
struct Frame {
    id: ExprId,
    /// 0 = push children, 1 = fold node
    state: u8,
}

/// Iterative folder to avoid stack overflow.
struct IterativeFolder<'a> {
    ctx: &'a mut Context,
    cfg: &'a EvalConfig,
    /// Memoization: original ExprId -> folded ExprId
    memo: std::collections::HashMap<ExprId, ExprId>,
    nodes_created: u64,
    folds_performed: u64,
}

impl<'a> IterativeFolder<'a> {
    fn new(ctx: &'a mut Context, cfg: &'a EvalConfig) -> Self {
        Self {
            ctx,
            cfg,
            memo: std::collections::HashMap::new(),
            nodes_created: 0,
            folds_performed: 0,
        }
    }

    fn fold(&mut self, root: ExprId, budget: &mut Budget) -> Result<ExprId, CasError> {
        let mut stack = vec![Frame { id: root, state: 0 }];

        while let Some(frame) = stack.pop() {
            if frame.state == 0 {
                // Skip if already memoized
                if self.memo.contains_key(&frame.id) {
                    continue;
                }

                // Push self for fold phase
                stack.push(Frame {
                    id: frame.id,
                    state: 1,
                });

                // Push children (postorder: children first)
                let children = helpers::get_children(self.ctx, frame.id);
                for child in children.into_iter().rev() {
                    stack.push(Frame {
                        id: child,
                        state: 0,
                    });
                }
            } else {
                // Fold phase: try to fold this node
                let folded = self.try_fold_node(frame.id);
                self.memo.insert(frame.id, folded);

                if folded != frame.id {
                    self.folds_performed += 1;
                }

                // Budget check per fold
                budget.charge(Operation::ConstFold, Metric::Iterations, 1)?;
            }
        }

        Ok(self.memo.get(&root).copied().unwrap_or(root))
    }

    /// Try to fold a single node if it's constant and matches allowlist.
    fn try_fold_node(&mut self, id: ExprId) -> ExprId {
        let expr = self.ctx.get(id).clone();

        match &expr {
            // Literals are already folded
            Expr::Number(_) | Expr::Constant(_) => id,

            // Negation of constant
            Expr::Neg(inner) => {
                let inner_folded = self.get_folded(*inner);
                if helpers::is_constant_literal(self.ctx, inner_folded) {
                    if let Some(result) = helpers::fold_neg(self.ctx, inner_folded) {
                        self.nodes_created += 1;
                        return result;
                    }
                }
                // Rebuild if child changed
                if inner_folded != *inner {
                    self.nodes_created += 1;
                    self.ctx.add(Expr::Neg(inner_folded))
                } else {
                    id
                }
            }

            // Function: check for sqrt(...literal)
            Expr::Function(name, args) => {
                let args_folded: Vec<_> = args.iter().map(|a| self.get_folded(*a)).collect();

                // sqrt(literal) folding
                if self.ctx.sym_name(*name) == "sqrt"
                    && args_folded.len() == 1
                    && helpers::is_constant_literal(self.ctx, args_folded[0])
                {
                    if let Some(result) =
                        helpers::fold_sqrt(self.ctx, args_folded[0], self.cfg.value_domain)
                    {
                        self.nodes_created += 1;
                        return result;
                    }
                }

                // Rebuild if args changed
                if args_folded != *args {
                    self.nodes_created += 1;
                    self.ctx.add(Expr::Function(*name, args_folded))
                } else {
                    id
                }
            }

            // Mul: check for i*i pattern (binary)
            Expr::Mul(a, b) => {
                let a_folded = self.get_folded(*a);
                let b_folded = self.get_folded(*b);

                // Check for i*i → -1
                if let Some(result) =
                    helpers::fold_mul_imaginary(self.ctx, a_folded, b_folded, self.cfg.value_domain)
                {
                    self.nodes_created += 1;
                    return result;
                }

                // Rebuild if any child changed
                if a_folded != *a || b_folded != *b {
                    self.nodes_created += 1;
                    self.ctx.add(Expr::Mul(a_folded, b_folded))
                } else {
                    id
                }
            }

            // Add: propagate folded children
            Expr::Add(a, b) => {
                let a_folded = self.get_folded(*a);
                let b_folded = self.get_folded(*b);
                if a_folded != *a || b_folded != *b {
                    self.nodes_created += 1;
                    self.ctx.add(Expr::Add(a_folded, b_folded))
                } else {
                    id
                }
            }

            // Pow: try fold literal^literal, else propagate
            Expr::Pow(base, exp) => {
                let base_f = self.get_folded(*base);
                let exp_f = self.get_folded(*exp);

                // Try fold if both are constant literals (or Neg of literal)
                if let Some(result) = helpers::fold_pow(
                    self.ctx,
                    base_f,
                    exp_f,
                    self.cfg.value_domain,
                    self.cfg.branch,
                ) {
                    self.nodes_created += 1;
                    return result;
                }

                // Rebuild if children changed
                if base_f != *base || exp_f != *exp {
                    self.nodes_created += 1;
                    self.ctx.add(Expr::Pow(base_f, exp_f))
                } else {
                    id
                }
            }

            // Div: propagate folded children
            Expr::Div(num, den) => {
                let num_f = self.get_folded(*num);
                let den_f = self.get_folded(*den);
                if num_f != *num || den_f != *den {
                    self.nodes_created += 1;
                    self.ctx.add(Expr::Div(num_f, den_f))
                } else {
                    id
                }
            }

            // Sub: propagate folded children
            Expr::Sub(a, b) => {
                let a_f = self.get_folded(*a);
                let b_f = self.get_folded(*b);
                if a_f != *a || b_f != *b {
                    self.nodes_created += 1;
                    self.ctx.add(Expr::Sub(a_f, b_f))
                } else {
                    id
                }
            }

            // Everything else: no rebuild needed
            _ => id,
        }
    }

    fn get_folded(&self, id: ExprId) -> ExprId {
        self.memo.get(&id).copied().unwrap_or(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;

    #[test]
    fn test_off_mode_is_noop() {
        let mut ctx = Context::new();
        let expr = ctx.num(42);
        let cfg = EvalConfig::default();
        let mut budget = Budget::preset_unlimited();

        let result = fold_constants(&mut ctx, expr, &cfg, ConstFoldMode::Off, &mut budget).unwrap();
        assert_eq!(result.expr, expr);
        assert_eq!(result.folds_performed, 0);
    }

    #[test]
    fn test_literal_unchanged() {
        let mut ctx = Context::new();
        let expr = ctx.num(42);
        let cfg = EvalConfig::default();
        let mut budget = Budget::preset_unlimited();

        let result =
            fold_constants(&mut ctx, expr, &cfg, ConstFoldMode::Safe, &mut budget).unwrap();
        assert_eq!(result.expr, expr);
    }
}
