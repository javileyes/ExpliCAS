//! Local constant-fold implementation for solver facade.
//!
//! Mirrors engine allowlist folding behavior so solver can expose the same
//! API without routing through `cas_engine::fold_constants`.

use crate::{Budget, CasError, ConstFoldMode, ConstFoldResult, EvalConfig, Metric, Operation};
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{Signed, Zero};

/// Fold constants in an expression using allowlist-only operations.
pub fn fold_constants_local(
    ctx: &mut Context,
    expr: ExprId,
    cfg: &EvalConfig,
    mode: ConstFoldMode,
    budget: &mut Budget,
) -> Result<ConstFoldResult, CasError> {
    if mode == ConstFoldMode::Off {
        return Ok(ConstFoldResult {
            expr,
            nodes_created: 0,
            folds_performed: 0,
        });
    }

    budget.charge(Operation::ConstFold, Metric::Iterations, 1)?;

    let mut folder = IterativeFolder::new(ctx, cfg);
    let out_expr = folder.fold(expr, budget)?;

    Ok(ConstFoldResult {
        expr: out_expr,
        nodes_created: folder.nodes_created,
        folds_performed: folder.folds_performed,
    })
}

#[derive(Clone, Copy)]
struct Frame {
    id: ExprId,
    // 0 = push children, 1 = fold node
    state: u8,
}

struct IterativeFolder<'a> {
    ctx: &'a mut Context,
    cfg: &'a EvalConfig,
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
                if self.memo.contains_key(&frame.id) {
                    continue;
                }

                stack.push(Frame {
                    id: frame.id,
                    state: 1,
                });

                let children = get_children(self.ctx, frame.id);
                for child in children.into_iter().rev() {
                    stack.push(Frame {
                        id: child,
                        state: 0,
                    });
                }
            } else {
                let folded = self.try_fold_node(frame.id);
                self.memo.insert(frame.id, folded);
                if folded != frame.id {
                    self.folds_performed += 1;
                }
                budget.charge(Operation::ConstFold, Metric::Iterations, 1)?;
            }
        }

        Ok(self.memo.get(&root).copied().unwrap_or(root))
    }

    fn try_fold_node(&mut self, id: ExprId) -> ExprId {
        let expr = self.ctx.get(id).clone();

        match &expr {
            Expr::Number(_) | Expr::Constant(_) => id,

            Expr::Neg(inner) => {
                let inner_folded = self.get_folded(*inner);
                if is_constant_literal(self.ctx, inner_folded) {
                    if let Some(result) = fold_neg(self.ctx, inner_folded) {
                        self.nodes_created += 1;
                        return result;
                    }
                }
                if inner_folded != *inner {
                    self.nodes_created += 1;
                    self.ctx.add(Expr::Neg(inner_folded))
                } else {
                    id
                }
            }

            Expr::Function(name, args) => {
                let args_folded: Vec<_> = args.iter().map(|a| self.get_folded(*a)).collect();

                if self.ctx.is_builtin(*name, cas_ast::BuiltinFn::Sqrt)
                    && args_folded.len() == 1
                    && is_constant_literal(self.ctx, args_folded[0])
                {
                    if let Some(result) = fold_sqrt(self.ctx, args_folded[0], self.cfg.value_domain)
                    {
                        self.nodes_created += 1;
                        return result;
                    }
                }

                if args_folded != *args {
                    self.nodes_created += 1;
                    self.ctx.add(Expr::Function(*name, args_folded))
                } else {
                    id
                }
            }

            Expr::Mul(a, b) => {
                let a_folded = self.get_folded(*a);
                let b_folded = self.get_folded(*b);

                if let Some(result) =
                    fold_mul_imaginary(self.ctx, a_folded, b_folded, self.cfg.value_domain)
                {
                    self.nodes_created += 1;
                    return result;
                }

                if a_folded != *a || b_folded != *b {
                    self.nodes_created += 1;
                    self.ctx.add(Expr::Mul(a_folded, b_folded))
                } else {
                    id
                }
            }

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

            Expr::Pow(base, exp) => {
                let base_folded = self.get_folded(*base);
                let exp_folded = self.get_folded(*exp);

                if let Some(result) = fold_pow(
                    self.ctx,
                    base_folded,
                    exp_folded,
                    self.cfg.value_domain,
                    self.cfg.branch,
                ) {
                    self.nodes_created += 1;
                    return result;
                }

                if base_folded != *base || exp_folded != *exp {
                    self.nodes_created += 1;
                    self.ctx.add(Expr::Pow(base_folded, exp_folded))
                } else {
                    id
                }
            }

            Expr::Div(num, den) => {
                let num_folded = self.get_folded(*num);
                let den_folded = self.get_folded(*den);
                if num_folded != *num || den_folded != *den {
                    self.nodes_created += 1;
                    self.ctx.add(Expr::Div(num_folded, den_folded))
                } else {
                    id
                }
            }

            Expr::Sub(a, b) => {
                let a_folded = self.get_folded(*a);
                let b_folded = self.get_folded(*b);
                if a_folded != *a || b_folded != *b {
                    self.nodes_created += 1;
                    self.ctx.add(Expr::Sub(a_folded, b_folded))
                } else {
                    id
                }
            }

            Expr::Hold(inner) => {
                let inner_folded = self.get_folded(*inner);
                if inner_folded != *inner {
                    self.nodes_created += 1;
                    self.ctx.add(Expr::Hold(inner_folded))
                } else {
                    id
                }
            }

            Expr::Matrix { rows, cols, data } => {
                let data_folded: Vec<_> = data.iter().map(|d| self.get_folded(*d)).collect();
                if data_folded != *data {
                    self.nodes_created += 1;
                    self.ctx.add(Expr::Matrix {
                        rows: *rows,
                        cols: *cols,
                        data: data_folded,
                    })
                } else {
                    id
                }
            }

            Expr::Variable(_) | Expr::SessionRef(_) => id,
        }
    }

    fn get_folded(&self, id: ExprId) -> ExprId {
        self.memo.get(&id).copied().unwrap_or(id)
    }
}

fn get_children(ctx: &Context, id: ExprId) -> Vec<ExprId> {
    match ctx.get(id) {
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            vec![*a, *b]
        }
        Expr::Neg(inner) | Expr::Hold(inner) => vec![*inner],
        Expr::Function(_, args) => args.clone(),
        Expr::Matrix { data, .. } => data.clone(),
        _ => vec![],
    }
}

fn is_constant_literal(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(_) => true,
        Expr::Constant(c) => matches!(
            c,
            cas_ast::Constant::Pi
                | cas_ast::Constant::E
                | cas_ast::Constant::I
                | cas_ast::Constant::Infinity
                | cas_ast::Constant::Undefined
        ),
        _ => false,
    }
}

fn fold_neg(ctx: &mut Context, inner: ExprId) -> Option<ExprId> {
    match ctx.get(inner) {
        Expr::Number(n) => Some(ctx.add(Expr::Number(-n.clone()))),
        _ => None,
    }
}

fn fold_sqrt(ctx: &mut Context, base: ExprId, value_domain: crate::ValueDomain) -> Option<ExprId> {
    let n = match ctx.get(base) {
        Expr::Number(n) => n.clone(),
        _ => return None,
    };

    if n.is_negative() {
        match value_domain {
            crate::ValueDomain::RealOnly => {
                Some(ctx.add(Expr::Constant(cas_ast::Constant::Undefined)))
            }
            crate::ValueDomain::ComplexEnabled => {
                let pos_n = -n;
                let pos_n_expr = ctx.add(Expr::Number(pos_n));
                let sqrt_pos = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![pos_n_expr]);
                let i = ctx.add(Expr::Constant(cas_ast::Constant::I));
                Some(ctx.add(Expr::Mul(i, sqrt_pos)))
            }
        }
    } else if n.is_zero() {
        Some(ctx.num(0))
    } else {
        try_exact_sqrt(ctx, &n)
    }
}

fn try_exact_sqrt(ctx: &mut Context, n: &BigRational) -> Option<ExprId> {
    if !n.is_integer() {
        return None;
    }
    let num = n.numer();
    let sqrt_num = num.sqrt();
    if &(&sqrt_num * &sqrt_num) == num {
        Some(ctx.add(Expr::Number(BigRational::from_integer(sqrt_num))))
    } else {
        None
    }
}

fn is_imaginary_unit(ctx: &Context, id: ExprId) -> bool {
    matches!(ctx.get(id), Expr::Constant(cas_ast::Constant::I))
}

fn is_neg_of_i(ctx: &Context, id: ExprId) -> bool {
    if let Expr::Neg(inner) = ctx.get(id) {
        is_imaginary_unit(ctx, *inner)
    } else {
        false
    }
}

fn fold_mul_imaginary(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    value_domain: crate::ValueDomain,
) -> Option<ExprId> {
    if value_domain != crate::ValueDomain::ComplexEnabled {
        return None;
    }

    if is_imaginary_unit(ctx, a) && is_imaginary_unit(ctx, b) {
        return Some(ctx.num(-1));
    }

    let a_is_neg_i = is_neg_of_i(ctx, a);
    let b_is_neg_i = is_neg_of_i(ctx, b);

    if (a_is_neg_i && is_imaginary_unit(ctx, b)) || (is_imaginary_unit(ctx, a) && b_is_neg_i) {
        return Some(ctx.num(1));
    }

    if a_is_neg_i && b_is_neg_i {
        return Some(ctx.num(-1));
    }

    None
}

fn literal_rat(ctx: &Context, id: ExprId) -> Option<BigRational> {
    match ctx.get(id) {
        Expr::Number(n) => Some(n.clone()),
        Expr::Neg(inner) => {
            if let Expr::Number(n) = ctx.get(*inner) {
                Some(-n.clone())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn fold_pow(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    value_domain: crate::ValueDomain,
    _branch: crate::BranchPolicy,
) -> Option<ExprId> {
    if let Some(result) = cas_math::const_eval::try_eval_pow_literal(ctx, base, exp) {
        return Some(result);
    }

    let base_rat = literal_rat(ctx, base)?;
    let exp_rat = literal_rat(ctx, exp)?;

    let exp_rat = if exp_rat.denom().is_negative() {
        BigRational::new(-exp_rat.numer().clone(), -exp_rat.denom().clone())
    } else {
        exp_rat
    };

    if exp_rat == BigRational::new(1.into(), 2.into())
        && base_rat == BigRational::from_integer((-1).into())
    {
        return match value_domain {
            crate::ValueDomain::RealOnly => {
                Some(ctx.add(Expr::Constant(cas_ast::Constant::Undefined)))
            }
            crate::ValueDomain::ComplexEnabled => {
                Some(ctx.add(Expr::Constant(cas_ast::Constant::I)))
            }
        };
    }

    None
}
