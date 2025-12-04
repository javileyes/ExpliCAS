use crate::{Constant, Context, Expr, ExprId};

/// Converts an expression to LaTeX without converting fractional exponents to roots
/// Useful for showing exponent operations explicitly in rule descriptions
pub struct LatexNoRoots<'a> {
    pub context: &'a Context,
    pub id: ExprId,
}

impl<'a> LatexNoRoots<'a> {
    pub fn to_latex(&self) -> String {
        self.expr_to_latex(self.id, false)
    }

    fn expr_to_latex(&self, id: ExprId, parent_needs_parens: bool) -> String {
        match self.context.get(id) {
            Expr::Number(n) => {
                if n.is_integer() {
                    format!("{}", n.numer())
                } else {
                    format!("\\frac{{{}}}{{{}}}", n.numer(), n.denom())
                }
            }
            Expr::Variable(name) => name.clone(),
            Expr::Constant(c) => match c {
                Constant::Pi => "\\pi".to_string(),
                Constant::E => "e".to_string(),
                Constant::Infinity => "\\infty".to_string(),
                Constant::Undefined => "\\text{undefined}".to_string(),
            },
            Expr::Add(l, r) => {
                let left = self.expr_to_latex(*l, false);
                let right = self.expr_to_latex(*r, false);
                format!("{} + {}", left, right)
            }
            Expr::Sub(l, r) => {
                let left = self.expr_to_latex(*l, false);
                let right = self.expr_to_latex(*r, true);
                format!("{} - {}", left, right)
            }
            Expr::Mul(l, r) => {
                let left = self.expr_to_latex_mul(*l);
                let right = self.expr_to_latex_mul(*r);

                // Smart multiplication: omit * when appropriate
                let needs_cdot = self.needs_explicit_mult(*l, *r);
                if needs_cdot {
                    if parent_needs_parens {
                        format!("({}\\cdot {})", left, right)
                    } else {
                        format!("{}\\cdot {}", left, right)
                    }
                } else {
                    if parent_needs_parens {
                        format!("({}{})", left, right)
                    } else {
                        format!("{}{}", left, right)
                    }
                }
            }
            Expr::Div(l, r) => {
                let numer = self.expr_to_latex(*l, false);
                let denom = self.expr_to_latex(*r, false);
                format!("\\frac{{{}}}{{{}}}", numer, denom)
            }
            Expr::Pow(base, exp) => {
                // Always use power notation, never convert to roots
                let base_str = self.expr_to_latex_base(*base);
                let exp_str = self.expr_to_latex(*exp, false);
                format!("{{{}}}^{{{}}}", base_str, exp_str)
            }
            Expr::Neg(e) => {
                let inner = self.expr_to_latex(*e, true);
                format!("-{}", inner)
            }
            Expr::Function(name, args) => match name.as_str() {
                "sqrt" if args.len() == 1 => {
                    let arg = self.expr_to_latex(args[0], false);
                    format!("\\sqrt{{{}}}", arg)
                }
                "sqrt" if args.len() == 2 => {
                    let radicand = self.expr_to_latex(args[0], false);
                    let index = self.expr_to_latex(args[1], false);
                    format!("\\sqrt[{}]{{{}}}", index, radicand)
                }
                "sin" | "cos" | "tan" | "cot" | "sec" | "csc" => {
                    let arg = self.expr_to_latex(args[0], false);
                    format!("\\{}({})", name, arg)
                }
                "ln" => {
                    let arg = self.expr_to_latex(args[0], false);
                    format!("\\ln({})", arg)
                }
                "log" if args.len() == 2 => {
                    let base = self.expr_to_latex(args[0], false);
                    let arg = self.expr_to_latex(args[1], false);
                    format!("\\log_{{{}}}({})", base, arg)
                }
                "abs" => {
                    let arg = self.expr_to_latex(args[0], false);
                    format!("|{}|", arg)
                }
                _ => {
                    let args_str: Vec<String> = args
                        .iter()
                        .map(|&arg| self.expr_to_latex(arg, false))
                        .collect();
                    format!("\\text{{{}}}({})", name, args_str.join(", "))
                }
            },
        }
    }

    fn expr_to_latex_mul(&self, id: ExprId) -> String {
        match self.context.get(id) {
            Expr::Add(_, _) | Expr::Sub(_, _) => {
                format!("({})", self.expr_to_latex(id, false))
            }
            _ => self.expr_to_latex(id, false),
        }
    }

    fn expr_to_latex_base(&self, id: ExprId) -> String {
        match self.context.get(id) {
            Expr::Add(_, _)
            | Expr::Sub(_, _)
            | Expr::Mul(_, _)
            | Expr::Div(_, _)
            | Expr::Neg(_) => {
                format!("({})", self.expr_to_latex(id, false))
            }
            _ => self.expr_to_latex(id, false),
        }
    }

    fn needs_explicit_mult(&self, left: ExprId, right: ExprId) -> bool {
        // Check if we need explicit \cdot
        match (self.context.get(left), self.context.get(right)) {
            // Number * Number ALWAYS needs \cdot (5*5 = 5\cdot 5, not 55)
            (Expr::Number(_), Expr::Number(_)) => true,

            // Number * (complex expr) needs \cdot to avoid ambiguity
            (Expr::Number(_), Expr::Add(_, _)) => true,
            (Expr::Number(_), Expr::Sub(_, _)) => true,
            (Expr::Number(_), Expr::Div(_, _)) => true,

            // Anything * Number might need \cdot (e.g., x*2 is fine as x2, but (x+1)*2 needs parens already)
            (Expr::Add(_, _), Expr::Number(_)) => true,
            (Expr::Sub(_, _), Expr::Number(_)) => true,

            // Everything else can be implicit (2x, xy, x*sin(x), etc.)
            _ => false,
        }
    }
}
