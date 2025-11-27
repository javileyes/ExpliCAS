use cas_ast::{Expr, Visitor, Constant};
use std::rc::Rc;

pub trait Format {
    fn to_latex(&self) -> String;
}

impl Format for Expr {
    fn to_latex(&self) -> String {
        let mut visitor = LatexVisitor::new();
        visitor.visit_expr(self);
        visitor.output
    }
}

impl Format for Rc<Expr> {
    fn to_latex(&self) -> String {
        self.as_ref().to_latex()
    }
}

pub struct LatexVisitor {
    output: String,
}

impl LatexVisitor {
    pub fn new() -> Self {
        Self { output: String::new() }
    }
}

impl Visitor for LatexVisitor {
    fn visit_number(&mut self, n: &num_rational::BigRational) {
        if n.is_integer() {
            self.output.push_str(&n.to_integer().to_string());
        } else {
            self.output.push_str(&format!("\\frac{{{}}}{{{}}}", n.numer(), n.denom()));
        }
    }

    fn visit_constant(&mut self, c: &Constant) {
        match c {
            Constant::Pi => self.output.push_str("\\pi"),
            Constant::E => self.output.push_str("e"),
            Constant::Infinity => self.output.push_str("\\infty"),
            Constant::Undefined => self.output.push_str("\\text{undefined}"),
        }
    }

    fn visit_variable(&mut self, name: &str) {
        if name.len() > 1 {
            self.output.push_str(&format!("\\text{{{}}}", name));
        } else {
            self.output.push_str(name);
        }
    }

    fn visit_add(&mut self, l: &Expr, r: &Expr) {
        self.visit_expr(l);
        self.output.push_str(" + ");
        self.visit_expr(r);
    }

    fn visit_sub(&mut self, l: &Expr, r: &Expr) {
        self.visit_expr(l);
        self.output.push_str(" - ");
        // Simple paren logic for RHS of sub
        if matches!(r, Expr::Add(_, _) | Expr::Sub(_, _)) {
            self.output.push('(');
            self.visit_expr(r);
            self.output.push(')');
        } else {
            self.visit_expr(r);
        }
    }

    fn visit_mul(&mut self, l: &Expr, r: &Expr) {
        // Parentheses if lower precedence
        let l_needs_parens = matches!(l, Expr::Add(_, _) | Expr::Sub(_, _));
        if l_needs_parens { self.output.push('('); }
        self.visit_expr(l);
        if l_needs_parens { self.output.push(')'); }
        
        self.output.push_str(" \\cdot ");
        
        let r_needs_parens = matches!(r, Expr::Add(_, _) | Expr::Sub(_, _));
        if r_needs_parens { self.output.push('('); }
        self.visit_expr(r);
        if r_needs_parens { self.output.push(')'); }
    }

    fn visit_div(&mut self, l: &Expr, r: &Expr) {
        self.output.push_str("\\frac{");
        self.visit_expr(l);
        self.output.push_str("}{");
        self.visit_expr(r);
        self.output.push_str("}");
    }

    fn visit_pow(&mut self, b: &Expr, e: &Expr) {
        let b_needs_parens = !matches!(b, Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_));
        if b_needs_parens { self.output.push('('); }
        self.visit_expr(b);
        if b_needs_parens { self.output.push(')'); }
        
        self.output.push_str("^{");
        self.visit_expr(e);
        self.output.push_str("}");
    }

    fn visit_neg(&mut self, e: &Expr) {
        self.output.push('-');
        let needs_parens = matches!(e, Expr::Add(_, _) | Expr::Sub(_, _));
        if needs_parens { self.output.push('('); }
        self.visit_expr(e);
        if needs_parens { self.output.push(')'); }
    }

    fn visit_function(&mut self, name: &str, args: &[Rc<Expr>]) {
        match name {
            "sin" | "cos" | "tan" | "log" | "ln" => {
                self.output.push_str(&format!("\\{}(", name));
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { self.output.push_str(", "); }
                    self.visit_expr(arg);
                }
                self.output.push(')');
            },
            "sqrt" => {
                self.output.push_str("\\sqrt{");
                if !args.is_empty() {
                    self.visit_expr(&args[0]);
                }
                self.output.push('}');
            },
            _ => {
                self.output.push_str(&format!("\\operatorname{{{}}}(", name));
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { self.output.push_str(", "); }
                    self.visit_expr(arg);
                }
                self.output.push(')');
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Expr;

    #[test]
    fn test_latex_format() {
        // x^2 + \sin(x) / 2
        let e = Expr::add(
            Expr::pow(Expr::var("x"), Expr::num(2)),
            Expr::div(
                Expr::sin(Expr::var("x")),
                Expr::num(2)
            )
        );
        assert_eq!(e.to_latex(), "x^{2} + \\frac{\\sin(x)}{2}");
    }
}
