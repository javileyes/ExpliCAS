mod binary;
mod function;
mod matrix;
mod unary;

use cas_ast::{Expr, ExprId};

use super::IterativeFolder;

impl<'a> IterativeFolder<'a> {
    pub(super) fn try_fold_node(&mut self, id: ExprId) -> ExprId {
        let expr = self.ctx.get(id).clone();

        match &expr {
            Expr::Number(_) | Expr::Constant(_) => id,

            Expr::Neg(inner) => unary::try_fold_neg(self, id, *inner),
            Expr::Hold(inner) => unary::try_fold_hold(self, id, *inner),
            Expr::Function(name, args) => function::try_fold_function(self, id, *name, args),
            Expr::Matrix { rows, cols, data } => {
                matrix::try_fold_matrix(self, id, *rows, *cols, data)
            }

            Expr::Mul(a, b) => binary::try_fold_mul(self, id, *a, *b),

            Expr::Add(a, b) => self.rebuild_binary(id, Expr::Add, *a, *b),
            Expr::Pow(base, exp) => binary::try_fold_pow(self, id, *base, *exp),
            Expr::Div(num, den) => self.rebuild_binary(id, Expr::Div, *num, *den),
            Expr::Sub(a, b) => self.rebuild_binary(id, Expr::Sub, *a, *b),

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
