use cas_ast::{Constant, Context, Expr, ExprId, Visitor};

pub trait Format {
    fn to_latex(&self, ctx: &Context) -> String;
}

impl Format for ExprId {
    fn to_latex(&self, ctx: &Context) -> String {
        let mut visitor = LatexVisitor::new();
        visitor.visit_expr(ctx, *self);
        visitor.output
    }
}

pub struct LatexVisitor {
    output: String,
}

impl LatexVisitor {
    pub fn new() -> Self {
        Self {
            output: String::new(),
        }
    }
}

impl Visitor for LatexVisitor {
    fn visit_number(&mut self, n: &num_rational::BigRational) {
        if n.is_integer() {
            self.output.push_str(&n.to_integer().to_string());
        } else {
            self.output
                .push_str(&format!("\\frac{{{}}}{{{}}}", n.numer(), n.denom()));
        }
    }

    fn visit_constant(&mut self, c: &Constant) {
        match c {
            Constant::Pi => self.output.push_str("\\pi"),
            Constant::E => self.output.push_str("e"),
            Constant::Infinity => self.output.push_str("\\infty"),
            Constant::Undefined => self.output.push_str("\\text{undefined}"),
            Constant::I => self.output.push_str("i"),
        }
    }

    fn visit_variable(&mut self, name: &str) {
        if name.len() > 1 {
            self.output.push_str(&format!("\\text{{{}}}", name));
        } else {
            self.output.push_str(name);
        }
    }

    fn visit_add(&mut self, ctx: &Context, l: ExprId, r: ExprId) {
        self.visit_expr(ctx, l);
        self.output.push_str(" + ");
        self.visit_expr(ctx, r);
    }

    fn visit_sub(&mut self, ctx: &Context, l: ExprId, r: ExprId) {
        self.visit_expr(ctx, l);
        self.output.push_str(" - ");
        // Simple paren logic for RHS of sub
        let r_expr = ctx.get(r);
        if matches!(r_expr, Expr::Add(_, _) | Expr::Sub(_, _)) {
            self.output.push('(');
            self.visit_expr(ctx, r);
            self.output.push(')');
        } else {
            self.visit_expr(ctx, r);
        }
    }

    fn visit_mul(&mut self, ctx: &Context, l: ExprId, r: ExprId) {
        // Parentheses if lower precedence
        let l_expr = ctx.get(l);
        let l_needs_parens = matches!(l_expr, Expr::Add(_, _) | Expr::Sub(_, _));
        if l_needs_parens {
            self.output.push('(');
        }
        self.visit_expr(ctx, l);
        if l_needs_parens {
            self.output.push(')');
        }

        self.output.push_str(" \\cdot ");

        let r_expr = ctx.get(r);
        let r_needs_parens = matches!(r_expr, Expr::Add(_, _) | Expr::Sub(_, _));
        if r_needs_parens {
            self.output.push('(');
        }
        self.visit_expr(ctx, r);
        if r_needs_parens {
            self.output.push(')');
        }
    }

    fn visit_div(&mut self, ctx: &Context, l: ExprId, r: ExprId) {
        self.output.push_str("\\frac{");
        self.visit_expr(ctx, l);
        self.output.push_str("}{");
        self.visit_expr(ctx, r);
        self.output.push_str("}");
    }

    fn visit_pow(&mut self, ctx: &Context, b: ExprId, e: ExprId) {
        let b_expr = ctx.get(b);
        let b_needs_parens = !matches!(
            b_expr,
            Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_)
        );
        if b_needs_parens {
            self.output.push('(');
        }
        self.visit_expr(ctx, b);
        if b_needs_parens {
            self.output.push(')');
        }

        self.output.push_str("^{");
        self.visit_expr(ctx, e);
        self.output.push_str("}");
    }

    fn visit_neg(&mut self, ctx: &Context, e: ExprId) {
        self.output.push('-');
        let e_expr = ctx.get(e);
        let needs_parens = matches!(e_expr, Expr::Add(_, _) | Expr::Sub(_, _));
        if needs_parens {
            self.output.push('(');
        }
        self.visit_expr(ctx, e);
        if needs_parens {
            self.output.push(')');
        }
    }

    fn visit_function(&mut self, ctx: &Context, name: &str, args: &[ExprId]) {
        match name {
            "sin" | "cos" | "tan" | "log" | "ln" => {
                self.output.push_str(&format!("\\{}(", name));
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.visit_expr(ctx, *arg);
                }
                self.output.push(')');
            }
            "sqrt" => {
                self.output.push_str("\\sqrt{");
                if !args.is_empty() {
                    self.visit_expr(ctx, args[0]);
                }
                self.output.push('}');
            }
            // Matrix product: matmul(A, B) -> A \times B
            "matmul" => {
                if args.len() == 2 {
                    // Left operand (may need parens if it's complex)
                    self.visit_expr(ctx, args[0]);
                    self.output.push_str(" \\times ");
                    // Right operand
                    self.visit_expr(ctx, args[1]);
                } else {
                    // Fallback for wrong arity
                    self.output.push_str("\\operatorname{matmul}(");
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            self.output.push_str(", ");
                        }
                        self.visit_expr(ctx, *arg);
                    }
                    self.output.push(')');
                }
            }
            // Matrix transpose: transpose(A) -> A^{T}
            "transpose" | "T" => {
                if args.len() == 1 {
                    // Check if arg is simple (variable, number, matrix) or needs parens
                    let arg_expr = ctx.get(args[0]);
                    let needs_parens = matches!(
                        arg_expr,
                        Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Mul(_, _) | Expr::Div(_, _)
                    );
                    if needs_parens {
                        self.output.push('(');
                    }
                    self.visit_expr(ctx, args[0]);
                    if needs_parens {
                        self.output.push(')');
                    }
                    self.output.push_str("^{T}");
                } else {
                    self.output.push_str("\\operatorname{transpose}(");
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            self.output.push_str(", ");
                        }
                        self.visit_expr(ctx, *arg);
                    }
                    self.output.push(')');
                }
            }
            _ => {
                self.output
                    .push_str(&format!("\\operatorname{{{}}}(", name));
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.visit_expr(ctx, *arg);
                }
                self.output.push(')');
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{Context, Expr};

    #[test]
    fn test_latex_format() {
        let mut ctx = Context::new();
        // x^2 + \sin(x) / 2
        let x = ctx.var("x");
        let two = ctx.num(2);
        let pow = ctx.add(Expr::Pow(x, two));
        let sin = ctx.add(Expr::Function("sin".to_string(), vec![x]));
        let div = ctx.add(Expr::Div(sin, two));
        let expr = ctx.add(Expr::Add(pow, div));

        assert_eq!(expr.to_latex(&ctx), "x^{2} + \\frac{\\sin(x)}{2}");
    }
}
