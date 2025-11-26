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

impl Expr {
    fn precedence(&self) -> u8 {
        match self {
            Expr::Add(_, _) | Expr::Sub(_, _) => 1,
            Expr::Mul(_, _) | Expr::Div(_, _) => 2,
            Expr::Pow(_, _) => 3,
            Expr::Neg(_) => 4,
            Expr::Function(_, _) | Expr::Number(_) | Expr::Variable(_) => 5,
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Number(n) => write!(f, "{}", n),
            Expr::Variable(s) => write!(f, "{}", s),
            Expr::Add(l, r) => {
                let l_prec = l.precedence();
                let r_prec = r.precedence();
                let my_prec = self.precedence();
                
                if l_prec < my_prec { write!(f, "({})", l)? } else { write!(f, "{}", l)? }
                write!(f, " + ")?;
                if r_prec < my_prec { write!(f, "({})", r)? } else { write!(f, "{}", r)? }
                Ok(())
            },
            Expr::Sub(l, r) => {
                let l_prec = l.precedence();
                let r_prec = r.precedence();
                let my_prec = self.precedence();
                
                if l_prec < my_prec { write!(f, "({})", l)? } else { write!(f, "{}", l)? }
                write!(f, " - ")?;
                // Subtraction is non-associative, so if RHS has same precedence (e.g. a - (b - c)), it needs parens
                // But actually a - b - c is (a - b) - c. 
                // If we have a - (b + c), b+c is lower prec (1), so it gets parens.
                // If we have a - (b - c), b-c is same prec (1). Standard is left-assoc.
                // So if RHS is same precedence, we generally need parens for non-commutative ops?
                // For simplicity here: strict inequality for LHS, <= for RHS to enforce left-associativity visual?
                // Let's stick to simple precedence check first.
                if r_prec <= my_prec { write!(f, "({})", r)? } else { write!(f, "{}", r)? }
                Ok(())
            },
            Expr::Mul(l, r) => {
                let l_prec = l.precedence();
                let r_prec = r.precedence();
                let my_prec = self.precedence();
                
                if l_prec < my_prec { write!(f, "({})", l)? } else { write!(f, "{}", l)? }
                write!(f, " * ")?;
                if r_prec < my_prec { write!(f, "({})", r)? } else { write!(f, "{}", r)? }
                Ok(())
            },
            Expr::Div(l, r) => {
                let l_prec = l.precedence();
                let r_prec = r.precedence();
                let my_prec = self.precedence();
                
                if l_prec < my_prec { write!(f, "({})", l)? } else { write!(f, "{}", l)? }
                write!(f, " / ")?;
                if r_prec <= my_prec { write!(f, "({})", r)? } else { write!(f, "{}", r)? }
                Ok(())
            },
            Expr::Pow(b, e) => {
                let b_prec = b.precedence();
                let e_prec = e.precedence();
                let my_prec = self.precedence();
                if b_prec < my_prec { write!(f, "({})", b)? } else { write!(f, "{}", b)? }
                if e_prec < my_prec { write!(f, "^({})", e)? } else { write!(f, "^{}", e)? }
                Ok(())
            },
            Expr::Neg(e) => {
                let e_prec = e.precedence();
                let my_prec = self.precedence();
                write!(f, "-")?;
                if e_prec < my_prec { write!(f, "({})", e)? } else { write!(f, "{}", e)? }
                Ok(())
            },
            Expr::Function(name, args) => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
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
        assert_eq!(format!("{}", e), "1 + x * 2");
    }

    #[test]
    fn test_complex_display() {
        let e = Expr::pow(
            Expr::add(Expr::var("a"), Expr::var("b")),
            Expr::num(2)
        );
        assert_eq!(format!("{}", e), "(a + b)^2");
    }
}
