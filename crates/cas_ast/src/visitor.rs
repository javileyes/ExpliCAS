use crate::expression::{Constant, Context, Expr, ExprId};

pub trait Visitor {
    fn visit_expr(&mut self, context: &Context, id: ExprId) {
        let expr = context.get(id);
        match expr {
            Expr::Number(n) => self.visit_number(n),
            Expr::Constant(c) => self.visit_constant(c),
            Expr::Variable(name) => self.visit_variable(name),
            Expr::Add(l, r) => self.visit_add(context, *l, *r),
            Expr::Sub(l, r) => self.visit_sub(context, *l, *r),
            Expr::Mul(l, r) => self.visit_mul(context, *l, *r),
            Expr::Div(l, r) => self.visit_div(context, *l, *r),
            Expr::Pow(b, e) => self.visit_pow(context, *b, *e),
            Expr::Neg(e) => self.visit_neg(context, *e),
            Expr::Function(name, args) => self.visit_function(context, name, args),
            Expr::Matrix { data, .. } => self.visit_matrix(context, data),
        }
    }

    fn visit_number(&mut self, _n: &num_rational::BigRational) {}
    fn visit_constant(&mut self, _c: &Constant) {}
    fn visit_variable(&mut self, _name: &str) {}

    fn visit_add(&mut self, context: &Context, left: ExprId, right: ExprId) {
        self.visit_expr(context, left);
        self.visit_expr(context, right);
    }

    fn visit_sub(&mut self, context: &Context, left: ExprId, right: ExprId) {
        self.visit_expr(context, left);
        self.visit_expr(context, right);
    }

    fn visit_mul(&mut self, context: &Context, left: ExprId, right: ExprId) {
        self.visit_expr(context, left);
        self.visit_expr(context, right);
    }

    fn visit_div(&mut self, context: &Context, left: ExprId, right: ExprId) {
        self.visit_expr(context, left);
        self.visit_expr(context, right);
    }

    fn visit_pow(&mut self, context: &Context, base: ExprId, exp: ExprId) {
        self.visit_expr(context, base);
        self.visit_expr(context, exp);
    }

    fn visit_neg(&mut self, context: &Context, expr: ExprId) {
        self.visit_expr(context, expr);
    }

    fn visit_function(&mut self, context: &Context, _name: &str, args: &[ExprId]) {
        for arg in args {
            self.visit_expr(context, *arg);
        }
    }

    fn visit_matrix(&mut self, context: &Context, data: &[ExprId]) {
        for elem in data {
            self.visit_expr(context, *elem);
        }
    }
}

pub trait Transformer {
    fn transform_expr(&mut self, context: &mut Context, id: ExprId) -> ExprId {
        let expr = context.get(id).clone(); // Clone to avoid borrow issues
        match expr {
            Expr::Number(_) => self.transform_number(context, id),
            Expr::Constant(_) => self.transform_constant(context, id),
            Expr::Variable(_) => self.transform_variable(context, id),
            Expr::Add(l, r) => self.transform_add(context, id, l, r),
            Expr::Sub(l, r) => self.transform_sub(context, id, l, r),
            Expr::Mul(l, r) => self.transform_mul(context, id, l, r),
            Expr::Div(l, r) => self.transform_div(context, id, l, r),
            Expr::Pow(b, e) => self.transform_pow(context, id, b, e),
            Expr::Neg(e) => self.transform_neg(context, id, e),
            Expr::Function(name, args) => self.transform_function(context, id, &name, &args),
            Expr::Matrix { rows, cols, data } => {
                self.transform_matrix(context, id, rows, cols, &data)
            }
        }
    }

    fn transform_number(&mut self, _context: &mut Context, id: ExprId) -> ExprId {
        id
    }
    fn transform_constant(&mut self, _context: &mut Context, id: ExprId) -> ExprId {
        id
    }
    fn transform_variable(&mut self, _context: &mut Context, id: ExprId) -> ExprId {
        id
    }

    fn transform_add(
        &mut self,
        context: &mut Context,
        original: ExprId,
        l: ExprId,
        r: ExprId,
    ) -> ExprId {
        let new_l = self.transform_expr(context, l);
        let new_r = self.transform_expr(context, r);
        if new_l != l || new_r != r {
            context.add(Expr::Add(new_l, new_r))
        } else {
            original
        }
    }

    fn transform_sub(
        &mut self,
        context: &mut Context,
        original: ExprId,
        l: ExprId,
        r: ExprId,
    ) -> ExprId {
        let new_l = self.transform_expr(context, l);
        let new_r = self.transform_expr(context, r);
        if new_l != l || new_r != r {
            context.add(Expr::Sub(new_l, new_r))
        } else {
            original
        }
    }

    fn transform_mul(
        &mut self,
        context: &mut Context,
        original: ExprId,
        l: ExprId,
        r: ExprId,
    ) -> ExprId {
        let new_l = self.transform_expr(context, l);
        let new_r = self.transform_expr(context, r);
        if new_l != l || new_r != r {
            context.add(Expr::Mul(new_l, new_r))
        } else {
            original
        }
    }

    fn transform_div(
        &mut self,
        context: &mut Context,
        original: ExprId,
        l: ExprId,
        r: ExprId,
    ) -> ExprId {
        let new_l = self.transform_expr(context, l);
        let new_r = self.transform_expr(context, r);
        if new_l != l || new_r != r {
            context.add(Expr::Div(new_l, new_r))
        } else {
            original
        }
    }

    fn transform_pow(
        &mut self,
        context: &mut Context,
        original: ExprId,
        b: ExprId,
        e: ExprId,
    ) -> ExprId {
        let new_b = self.transform_expr(context, b);
        let new_e = self.transform_expr(context, e);
        if new_b != b || new_e != e {
            context.add(Expr::Pow(new_b, new_e))
        } else {
            original
        }
    }

    fn transform_neg(&mut self, context: &mut Context, original: ExprId, e: ExprId) -> ExprId {
        let new_e = self.transform_expr(context, e);
        if new_e != e {
            context.add(Expr::Neg(new_e))
        } else {
            original
        }
    }

    fn transform_function(
        &mut self,
        context: &mut Context,
        original: ExprId,
        name: &str,
        args: &[ExprId],
    ) -> ExprId {
        let mut new_args = Vec::new();
        let mut changed = false;
        for arg in args {
            let new_arg = self.transform_expr(context, *arg);
            if new_arg != *arg {
                changed = true;
            }
            new_args.push(new_arg);
        }
        if changed {
            context.add(Expr::Function(name.to_string(), new_args))
        } else {
            original
        }
    }

    fn transform_matrix(
        &mut self,
        context: &mut Context,
        original: ExprId,
        rows: usize,
        cols: usize,
        data: &[ExprId],
    ) -> ExprId {
        let mut new_data = Vec::new();
        let mut changed = false;
        for elem in data {
            let new_elem = self.transform_expr(context, *elem);
            if new_elem != *elem {
                changed = true;
            }
            new_data.push(new_elem);
        }
        if changed {
            context.add(Expr::Matrix {
                rows,
                cols,
                data: new_data,
            })
        } else {
            original
        }
    }
}
