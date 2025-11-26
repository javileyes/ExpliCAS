use std::rc::Rc;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    Number(i64), // Using i64 for simplicity for now, will upgrade to BigInt later
    Variable(String),
    Add(Rc<Expr>, Rc<Expr>),
    Sub(Rc<Expr>, Rc<Expr>),
    Mul(Rc<Expr>, Rc<Expr>),
    Div(Rc<Expr>, Rc<Expr>),
    Pow(Rc<Expr>, Rc<Expr>),
    Neg(Rc<Expr>),
    Function(String, Vec<Rc<Expr>>), // e.g., sin(x), log(x, 10)
}

impl Expr {
    // Helper constructors for cleaner code
    pub fn num(n: i64) -> Rc<Self> {
        Rc::new(Expr::Number(n))
    }

    pub fn var(name: &str) -> Rc<Self> {
        Rc::new(Expr::Variable(name.to_string()))
    }

    pub fn add(lhs: Rc<Expr>, rhs: Rc<Expr>) -> Rc<Self> {
        Rc::new(Expr::Add(lhs, rhs))
    }

    pub fn sub(lhs: Rc<Expr>, rhs: Rc<Expr>) -> Rc<Self> {
        Rc::new(Expr::Sub(lhs, rhs))
    }

    pub fn mul(lhs: Rc<Expr>, rhs: Rc<Expr>) -> Rc<Self> {
        Rc::new(Expr::Mul(lhs, rhs))
    }

    pub fn div(lhs: Rc<Expr>, rhs: Rc<Expr>) -> Rc<Self> {
        Rc::new(Expr::Div(lhs, rhs))
    }

    pub fn pow(base: Rc<Expr>, exp: Rc<Expr>) -> Rc<Self> {
        Rc::new(Expr::Pow(base, exp))
    }
    
    pub fn neg(expr: Rc<Expr>) -> Rc<Self> {
        Rc::new(Expr::Neg(expr))
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Number(n) => write!(f, "{}", n),
            Expr::Variable(s) => write!(f, "{}", s),
            Expr::Add(l, r) => write!(f, "({} + {})", l, r),
            Expr::Sub(l, r) => write!(f, "({} - {})", l, r),
            Expr::Mul(l, r) => write!(f, "({} * {})", l, r),
            Expr::Div(l, r) => write!(f, "({} / {})", l, r),
            Expr::Pow(b, e) => write!(f, "({}^{})", b, e),
            Expr::Neg(e) => write!(f, "-{}", e),
            Expr::Function(name, args) => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display() {
        let e = Expr::add(Expr::num(1), Expr::mul(Expr::var("x"), Expr::num(2)));
        assert_eq!(format!("{}", e), "(1 + (x * 2))");
    }

    #[test]
    fn test_complex_display() {
        let e = Expr::pow(
            Expr::add(Expr::var("a"), Expr::var("b")),
            Expr::num(2)
        );
        assert_eq!(format!("{}", e), "((a + b)^2)");
    }
}
