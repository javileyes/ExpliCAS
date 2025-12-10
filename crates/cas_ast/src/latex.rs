use crate::{Constant, Context, Expr, ExprId};
use num_traits::Signed;

/// Converts an expression to LaTeX format for rendering with MathJax
pub struct LaTeXExpr<'a> {
    pub context: &'a Context,
    pub id: ExprId,
}

impl<'a> LaTeXExpr<'a> {
    pub fn to_latex(&self) -> String {
        let latex = self.expr_to_latex(self.id, false);
        Self::clean_latex_negatives(&latex)
    }

    /// Post-process LaTeX to fix negative sign patterns
    /// Handles cases like "+ -" → "-" and "- -" → "+"
    fn clean_latex_negatives(latex: &str) -> String {
        let mut result = latex.to_string();

        // Fix "+ -" → "-" in all contexts
        // Even "+ -(" is simplified to "-(" since +(-(x)) = -(x)
        result = result.replace("+ -\\", "- \\");
        result = result.replace("+ -{", "- {");
        result = result.replace("+ -(", "- (");

        // Fix "- -" → "+" (double negative)
        result = result.replace("- -\\", "+ \\");
        result = result.replace("- -{", "+ {");
        result = result.replace("- -(", "+ (");

        // Fix "+ -" before digits (e.g., "+ -4" → "- 4")
        // Need to use regex for this since we need to match digits
        use regex::Regex;
        let re_plus_minus_digit = Regex::new(r"\+ -(\d)").unwrap();
        result = re_plus_minus_digit.replace_all(&result, "- $1").to_string();

        // Fix "- -" before digits (e.g., "- -4" → "+ 4")
        let re_minus_minus_digit = Regex::new(r"- -(\d)").unwrap();
        result = re_minus_minus_digit
            .replace_all(&result, "+ $1")
            .to_string();

        result
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

                // Check if right side is a negative number, negation, or multiplication by negative
                // If so, render as subtraction instead of addition
                let (is_negative, right_str) = match self.context.get(*r) {
                    // Case 1: Negative number literal
                    Expr::Number(n) if n.is_negative() => {
                        let positive = -n;
                        let positive_str = if positive.is_integer() {
                            format!("{}", positive.numer())
                        } else {
                            format!("\\frac{{{}}}{{{}}}", positive.numer(), positive.denom())
                        };
                        (true, positive_str)
                    }
                    // Case 2: Neg(expr) - extract the inner expression
                    Expr::Neg(inner) => (true, self.expr_to_latex(*inner, true)),
                    // Case 3: Mul with negative leading coefficient (-1 * expr, -2 * expr, etc.)
                    Expr::Mul(ml, mr) => {
                        // Check if left factor is a negative number
                        if let Expr::Number(coef) = self.context.get(*ml) {
                            if coef.is_negative() {
                                // Extract positive coefficient and rest
                                let positive_coef = -coef;
                                let rest_latex = self.expr_to_latex_mul(*mr);

                                // Format as "positive_coef * rest" for subtraction
                                if positive_coef.is_integer() && *positive_coef.numer() == 1.into()
                                {
                                    // -1 * expr -> just "expr"
                                    (true, rest_latex)
                                } else {
                                    // -n * expr -> "n * expr" or "n expr"
                                    let coef_str = if positive_coef.is_integer() {
                                        format!("{}", positive_coef.numer())
                                    } else {
                                        format!(
                                            "\\frac{{{}}}{{{}}}",
                                            positive_coef.numer(),
                                            positive_coef.denom()
                                        )
                                    };

                                    // Check if we need explicit mult
                                    let needs_cdot = matches!(
                                        (self.context.get(*ml), self.context.get(*mr)),
                                        (Expr::Number(_), Expr::Number(_))
                                            | (Expr::Number(_), Expr::Add(_, _))
                                            | (Expr::Number(_), Expr::Sub(_, _))
                                    );

                                    if needs_cdot {
                                        (true, format!("{}\\cdot {}", coef_str, rest_latex))
                                    } else {
                                        (true, format!("{}{}", coef_str, rest_latex))
                                    }
                                }
                            } else {
                                // Positive coefficient, render normally
                                (false, self.expr_to_latex(*r, false))
                            }
                        } else {
                            // Left factor is not a number
                            (false, self.expr_to_latex(*r, false))
                        }
                    }
                    // Case 4: Regular positive expression
                    _ => (false, self.expr_to_latex(*r, false)),
                };

                if is_negative {
                    format!("{} - {}", left, right_str)
                } else {
                    format!("{} + {}", left, right_str)
                }
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
                // Pure power notation - no automatic sqrt conversion
                // DisplayContext handles root notation when appropriate
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
                    let base = args[0];
                    let arg = args[1];

                    // Check if base is constant e - if so, use ln instead
                    if let Expr::Constant(Constant::E) = self.context.get(base) {
                        let arg_latex = self.expr_to_latex(arg, false);
                        format!("\\ln({})", arg_latex)
                    } else {
                        let base_latex = self.expr_to_latex(base, false);
                        let arg_latex = self.expr_to_latex(arg, false);
                        format!("\\log_{{{}}}({})", base_latex, arg_latex)
                    }
                }
                "abs" => {
                    let arg = self.expr_to_latex(args[0], false);
                    format!("|{}|", arg)
                }
                // Matrix product: matmul(A, B) → A \times B
                "matmul" if args.len() == 2 => {
                    let left = self.expr_to_latex(args[0], false);
                    let right = self.expr_to_latex(args[1], false);
                    format!("{} \\times {}", left, right)
                }
                // Matrix transpose: transpose(A) → A^T
                "transpose" | "T" if args.len() == 1 => {
                    let arg_str = self.expr_to_latex(args[0], false);
                    // Check if arg needs parens (including matmul which is a Function)
                    let needs_parens = matches!(
                        self.context.get(args[0]),
                        Expr::Add(_, _)
                            | Expr::Sub(_, _)
                            | Expr::Mul(_, _)
                            | Expr::Div(_, _)
                            | Expr::Function(_, _)
                    );
                    if needs_parens {
                        format!("({})^{{T}}", arg_str)
                    } else {
                        format!("{}^{{T}}", arg_str)
                    }
                }
                _ => {
                    let args_str: Vec<String> = args
                        .iter()
                        .map(|&arg| self.expr_to_latex(arg, false))
                        .collect();
                    format!("\\text{{{}}}({})", name, args_str.join(", "))
                }
            },
            Expr::Matrix { rows, cols, data } => {
                // Render matrix as LaTeX bmatrix
                let mut result = String::from("\\begin{bmatrix}\n");
                for r in 0..*rows {
                    for c in 0..*cols {
                        if c > 0 {
                            result.push_str(" & ");
                        }
                        let idx = r * cols + c;
                        result.push_str(&self.expr_to_latex(data[idx], false));
                    }
                    if r < rows - 1 {
                        result.push_str(" \\\\\n");
                    }
                }
                result.push_str("\n\\end{bmatrix}");
                result
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latex_basic() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Mul(two, x));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        assert_eq!(latex.to_latex(), "2x");
    }

    #[test]
    fn test_latex_sqrt() {
        let mut ctx = Context::new();
        let five = ctx.num(5);
        let expr = ctx.add(Expr::Function("sqrt".to_string(), vec![five]));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        assert_eq!(latex.to_latex(), "\\sqrt{5}");
    }

    #[test]
    fn test_latex_fraction() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Div(x, two));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        assert_eq!(latex.to_latex(), "\\frac{x}{2}");
    }

    #[test]
    fn test_latex_power() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Pow(x, two));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        assert_eq!(latex.to_latex(), "{x}^{2}");
    }

    #[test]
    fn test_latex_sqrt_from_power() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let half = ctx.add(Expr::Div(one, two));
        let expr = ctx.add(Expr::Pow(x, half));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        // After simplification: x^(1/2) renders as power, not sqrt
        // sqrt() function is rendered as \sqrt, but Pow(x, 1/2) is power notation
        assert_eq!(latex.to_latex(), "{x}^{\\frac{1}{2}}");
    }

    #[test]
    fn test_latex_nth_root() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let three = ctx.num(3);
        let one_third = ctx.add(Expr::Div(one, three));
        let expr = ctx.add(Expr::Pow(x, one_third));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        assert_eq!(latex.to_latex(), "{x}^{\\frac{1}{3}}");
    }

    #[test]
    fn test_latex_fractional_power() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let two_thirds = ctx.add(Expr::Div(two, three));
        let expr = ctx.add(Expr::Pow(x, two_thirds));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        assert_eq!(latex.to_latex(), "{x}^{\\frac{2}{3}}");
    }

    #[test]
    fn test_latex_rational_exponent() {
        // Test with actual rational number (post-simplification)
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let rational = ctx.rational(1, 2); // 1/2 as a rational number
        let expr = ctx.add(Expr::Pow(x, rational));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        assert_eq!(latex.to_latex(), "{x}^{\\frac{1}{2}}");
    }

    #[test]
    fn test_latex_rational_exponent_2_3() {
        // Test with 2/3 as a rational number
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let rational = ctx.rational(2, 3);
        let expr = ctx.add(Expr::Pow(x, rational));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        assert_eq!(latex.to_latex(), "{x}^{\\frac{2}{3}}");
    }
}
