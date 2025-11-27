use crate::expression::{Expr, Constant};
use std::rc::Rc;

pub trait Visitor {
    fn visit_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Number(n) => self.visit_number(n),
            Expr::Constant(c) => self.visit_constant(c),
            Expr::Variable(name) => self.visit_variable(name),
            Expr::Add(l, r) => self.visit_add(l, r),
            Expr::Sub(l, r) => self.visit_sub(l, r),
            Expr::Mul(l, r) => self.visit_mul(l, r),
            Expr::Div(l, r) => self.visit_div(l, r),
            Expr::Pow(b, e) => self.visit_pow(b, e),
            Expr::Neg(e) => self.visit_neg(e),
            Expr::Function(name, args) => self.visit_function(name, args),
        }
    }

    fn visit_number(&mut self, _n: &num_rational::BigRational) {}
    fn visit_constant(&mut self, _c: &Constant) {}
    fn visit_variable(&mut self, _name: &str) {}

    fn visit_add(&mut self, left: &Expr, right: &Expr) {
        self.visit_expr(left);
        self.visit_expr(right);
    }

    fn visit_sub(&mut self, left: &Expr, right: &Expr) {
        self.visit_expr(left);
        self.visit_expr(right);
    }

    fn visit_mul(&mut self, left: &Expr, right: &Expr) {
        self.visit_expr(left);
        self.visit_expr(right);
    }

    fn visit_div(&mut self, left: &Expr, right: &Expr) {
        self.visit_expr(left);
        self.visit_expr(right);
    }

    fn visit_pow(&mut self, base: &Expr, exp: &Expr) {
        self.visit_expr(base);
        self.visit_expr(exp);
    }

    fn visit_neg(&mut self, expr: &Expr) {
        self.visit_expr(expr);
    }

    fn visit_function(&mut self, _name: &str, args: &[Rc<Expr>]) {
        for arg in args {
            self.visit_expr(arg);
        }
    }
}

pub trait MutVisitor {
    fn visit_expr(&mut self, expr: Rc<Expr>) -> Rc<Expr> {
        match expr.as_ref() {
            Expr::Number(_) => self.visit_number(expr),
            Expr::Constant(_) => self.visit_constant(expr),
            Expr::Variable(_) => self.visit_variable(expr),
            Expr::Add(l, r) => self.visit_add(expr.clone(), l, r),
            Expr::Sub(l, r) => self.visit_sub(expr.clone(), l, r),
            Expr::Mul(l, r) => self.visit_mul(expr.clone(), l, r),
            Expr::Div(l, r) => self.visit_div(expr.clone(), l, r),
            Expr::Pow(b, e) => self.visit_pow(expr.clone(), b, e),
            Expr::Neg(e) => self.visit_neg(expr.clone(), e),
            Expr::Function(name, args) => self.visit_function(expr.clone(), name, args),
        }
    }

    fn visit_number(&mut self, expr: Rc<Expr>) -> Rc<Expr> { expr }
    fn visit_constant(&mut self, expr: Rc<Expr>) -> Rc<Expr> { expr }
    fn visit_variable(&mut self, expr: Rc<Expr>) -> Rc<Expr> { expr }

    fn visit_add(&mut self, original: Rc<Expr>, l: &Rc<Expr>, r: &Rc<Expr>) -> Rc<Expr> {
        let new_l = self.visit_expr(l.clone());
        let new_r = self.visit_expr(r.clone());
        if new_l != *l || new_r != *r {
            Expr::add(new_l, new_r)
        } else {
            original
        }
    }

    fn visit_sub(&mut self, original: Rc<Expr>, l: &Rc<Expr>, r: &Rc<Expr>) -> Rc<Expr> {
        let new_l = self.visit_expr(l.clone());
        let new_r = self.visit_expr(r.clone());
        if new_l != *l || new_r != *r {
            Expr::sub(new_l, new_r)
        } else {
            original
        }
    }

    fn visit_mul(&mut self, original: Rc<Expr>, l: &Rc<Expr>, r: &Rc<Expr>) -> Rc<Expr> {
        let new_l = self.visit_expr(l.clone());
        let new_r = self.visit_expr(r.clone());
        if new_l != *l || new_r != *r {
            Expr::mul(new_l, new_r)
        } else {
            original
        }
    }

    fn visit_div(&mut self, original: Rc<Expr>, l: &Rc<Expr>, r: &Rc<Expr>) -> Rc<Expr> {
        let new_l = self.visit_expr(l.clone());
        let new_r = self.visit_expr(r.clone());
        if new_l != *l || new_r != *r {
            Expr::div(new_l, new_r)
        } else {
            original
        }
    }

    fn visit_pow(&mut self, original: Rc<Expr>, b: &Rc<Expr>, e: &Rc<Expr>) -> Rc<Expr> {
        let new_b = self.visit_expr(b.clone());
        let new_e = self.visit_expr(e.clone());
        if new_b != *b || new_e != *e {
            Expr::pow(new_b, new_e)
        } else {
            original
        }
    }

    fn visit_neg(&mut self, original: Rc<Expr>, e: &Rc<Expr>) -> Rc<Expr> {
        let new_e = self.visit_expr(e.clone());
        if new_e != *e {
            Expr::neg(new_e)
        } else {
            original
        }
    }

    fn visit_function(&mut self, original: Rc<Expr>, name: &str, args: &[Rc<Expr>]) -> Rc<Expr> {
        let mut new_args = Vec::new();
        let mut changed = false;
        for arg in args {
            let new_arg = self.visit_expr(arg.clone());
            if new_arg != *arg {
                changed = true;
            }
            new_args.push(new_arg);
        }
        if changed {
            Rc::new(Expr::Function(name.to_string(), new_args))
        } else {
            original
        }
    }
}
