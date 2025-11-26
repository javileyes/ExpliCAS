use std::rc::Rc;
use std::fmt;
use num_rational::BigRational;
use num_bigint::BigInt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Constant {
    Pi,
    E,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    Number(BigRational),
    Constant(Constant),
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
        Rc::new(Expr::Number(BigRational::from_integer(BigInt::from(n))))
    }

    pub fn rational(num: i64, den: i64) -> Rc<Self> {
        Rc::new(Expr::Number(BigRational::new(BigInt::from(num), BigInt::from(den))))
    }

    pub fn var(name: &str) -> Rc<Self> {
        Rc::new(Expr::Variable(name.to_string()))
    }

    pub fn pi() -> Rc<Self> {
        Rc::new(Expr::Constant(Constant::Pi))
    }

    pub fn e() -> Rc<Self> {
        Rc::new(Expr::Constant(Constant::E))
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

    pub fn abs(expr: Rc<Expr>) -> Rc<Self> {
        Rc::new(Expr::Function("abs".to_string(), vec![expr]))
    }

    pub fn sin(expr: Rc<Expr>) -> Rc<Self> {
        Rc::new(Expr::Function("sin".to_string(), vec![expr]))
    }

    pub fn cos(expr: Rc<Expr>) -> Rc<Self> {
        Rc::new(Expr::Function("cos".to_string(), vec![expr]))
    }

    pub fn tan(expr: Rc<Expr>) -> Rc<Self> {
        Rc::new(Expr::Function("tan".to_string(), vec![expr]))
    }

    pub fn substitute(&self, var_name: &str, value: &Rc<Expr>) -> Rc<Self> {
        match self {
            Expr::Variable(name) if name == var_name => value.clone(),
            Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) => Rc::new(self.clone()),
            Expr::Add(l, r) => Expr::add(l.substitute(var_name, value), r.substitute(var_name, value)),
            Expr::Sub(l, r) => Expr::sub(l.substitute(var_name, value), r.substitute(var_name, value)),
            Expr::Mul(l, r) => Expr::mul(l.substitute(var_name, value), r.substitute(var_name, value)),
            Expr::Div(l, r) => Expr::div(l.substitute(var_name, value), r.substitute(var_name, value)),
            Expr::Pow(b, e) => Expr::pow(b.substitute(var_name, value), e.substitute(var_name, value)),
            Expr::Neg(e) => Expr::neg(e.substitute(var_name, value)),
            Expr::Function(name, args) => {
                let new_args = args.iter().map(|arg| arg.substitute(var_name, value)).collect();
                Rc::new(Expr::Function(name.clone(), new_args))
            }
        }
    }
}

impl Expr {
    fn precedence(&self) -> u8 {
        match self {
            Expr::Add(_, _) | Expr::Sub(_, _) => 1,
            Expr::Mul(_, _) | Expr::Div(_, _) => 2,
            Expr::Pow(_, _) => 3,
            Expr::Neg(_) => 4,
            Expr::Number(n) => if n.is_integer() { 5 } else { 2 },
            Expr::Function(_, _) | Expr::Variable(_) | Expr::Constant(_) => 5,
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Number(n) => {
                if n.is_integer() {
                    write!(f, "{}", n.to_integer())
                } else {
                    write!(f, "{}", n)
                }
            },
            Expr::Constant(c) => match c {
                Constant::Pi => write!(f, "pi"),
                Constant::E => write!(f, "e"),
            },
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

    #[test]
    fn test_substitute() {
        // x + 1, sub x = 2 -> 2 + 1
        let e = Expr::add(Expr::var("x"), Expr::num(1));
        let sub = e.substitute("x", &Expr::num(2));
        assert_eq!(format!("{}", sub), "2 + 1");

        // y + 1, sub x = 2 -> y + 1
        let e2 = Expr::add(Expr::var("y"), Expr::num(1));
        let sub2 = e2.substitute("x", &Expr::num(2));
        assert_eq!(format!("{}", sub2), "y + 1");

        // x^2, sub x = 3 -> 3^2
        let e3 = Expr::pow(Expr::var("x"), Expr::num(2));
        let sub3 = e3.substitute("x", &Expr::num(3));
        assert_eq!(format!("{}", sub3), "3^2");
    }

    #[test]
    fn test_substitute_nested() {
        // (x + 1) * x, sub x = 2 -> (2 + 1) * 2
        let e = Expr::mul(
            Expr::add(Expr::var("x"), Expr::num(1)),
            Expr::var("x")
        );
        let sub = e.substitute("x", &Expr::num(2));
        assert_eq!(format!("{}", sub), "(2 + 1) * 2");
    }

    #[test]
    fn test_substitute_complex() {
        // sqrt(x) + x^2, sub x = 4 -> sqrt(4) + 4^2
        let e = Expr::add(
            Rc::new(Expr::Function("sqrt".to_string(), vec![Expr::var("x")])),
            Expr::pow(Expr::var("x"), Expr::num(2))
        );
        let sub = e.substitute("x", &Expr::num(4));
        assert_eq!(format!("{}", sub), "sqrt(4) + 4^2");
    }
}
