use cas_ast::{Expr, ExprId};

use super::IterativeFolder;
use crate::const_fold_local::folds::{
    fold_mul_imaginary, fold_neg, fold_pow, fold_sqrt, is_constant_literal,
};

impl<'a> IterativeFolder<'a> {
    pub(super) fn try_fold_node(&mut self, id: ExprId) -> ExprId {
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

            Expr::Add(a, b) => self.rebuild_binary(id, Expr::Add, *a, *b),
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
            Expr::Div(num, den) => self.rebuild_binary(id, Expr::Div, *num, *den),
            Expr::Sub(a, b) => self.rebuild_binary(id, Expr::Sub, *a, *b),

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

    fn rebuild_binary(
        &mut self,
        id: ExprId,
        build: fn(ExprId, ExprId) -> Expr,
        a: ExprId,
        b: ExprId,
    ) -> ExprId {
        let a_folded = self.get_folded(a);
        let b_folded = self.get_folded(b);
        if a_folded != a || b_folded != b {
            self.nodes_created += 1;
            self.ctx.add(build(a_folded, b_folded))
        } else {
            id
        }
    }

    fn get_folded(&self, id: ExprId) -> ExprId {
        self.memo.get(&id).copied().unwrap_or(id)
    }
}
