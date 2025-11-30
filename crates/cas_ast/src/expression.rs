use std::fmt;
use num_rational::BigRational;
use num_traits::Zero;
use num_bigint::BigInt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Constant {
    Pi,
    E,
    Infinity,
    Undefined,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    Number(BigRational),
    Constant(Constant),
    Variable(String),
    Add(ExprId, ExprId),
    Sub(ExprId, ExprId),
    Mul(ExprId, ExprId),
    Div(ExprId, ExprId),
    Pow(ExprId, ExprId),
    Neg(ExprId),
    Function(String, Vec<ExprId>),
}

#[derive(Default, Clone)]
pub struct Context {
    pub nodes: Vec<Expr>,
}

impl Context {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add(&mut self, expr: Expr) -> ExprId {
        let id = ExprId(self.nodes.len() as u32);
        self.nodes.push(expr);
        id
    }

    pub fn get(&self, id: ExprId) -> &Expr {
        &self.nodes[id.0 as usize]
    }
    
    // Helper constructors that add to context immediately
    pub fn num(&mut self, n: i64) -> ExprId {
        self.add(Expr::Number(BigRational::from_integer(BigInt::from(n))))
    }

    pub fn rational(&mut self, num: i64, den: i64) -> ExprId {
        self.add(Expr::Number(BigRational::new(BigInt::from(num), BigInt::from(den))))
    }

    pub fn var(&mut self, name: &str) -> ExprId {
        self.add(Expr::Variable(name.to_string()))
    }
    
    // ... other helpers would need &mut self ...
}

impl fmt::Display for ExprId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Expr#{}", self.0)
    }
}

// We need a way to display Expr with Context, or just Expr if it doesn't recurse.
// But Expr DOES recurse via IDs. So we can't implement Display for Expr easily without Context.
// We can implement a helper struct for display.

pub struct DisplayExpr<'a> {
    pub context: &'a Context,
    pub id: ExprId,
}

impl<'a> fmt::Display for DisplayExpr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let expr = self.context.get(self.id);
        match expr {
            Expr::Number(n) => write!(f, "{}", n),
            Expr::Constant(c) => match c {
                Constant::Pi => write!(f, "pi"),
                Constant::E => write!(f, "e"),
                Constant::Infinity => write!(f, "infinity"),
                Constant::Undefined => write!(f, "undefined"),
            },
            Expr::Variable(s) => write!(f, "{}", s),
            Expr::Add(l, r) => {
                write!(f, "{} + {}", DisplayExpr { context: self.context, id: *l }, DisplayExpr { context: self.context, id: *r })
            },
            Expr::Sub(l, r) => {
                let rhs_prec = precedence(self.context, *r);
                let op_prec = 1; // Sub precedence
                if rhs_prec <= op_prec { // If RHS is Add/Sub, needs parens: a - (b + c)
                     write!(f, "{} - ({})", DisplayExpr { context: self.context, id: *l }, DisplayExpr { context: self.context, id: *r })
                } else {
                     write!(f, "{} - {}", DisplayExpr { context: self.context, id: *l }, DisplayExpr { context: self.context, id: *r })
                }
            },
            Expr::Mul(l, r) => {
                let lhs_prec = precedence(self.context, *l);
                let rhs_prec = precedence(self.context, *r);
                let op_prec = 2; // Mul precedence
                
                if lhs_prec < op_prec { write!(f, "({})", DisplayExpr { context: self.context, id: *l })? }
                else { write!(f, "{}", DisplayExpr { context: self.context, id: *l })? }
                
                write!(f, " * ")?;
                
                if rhs_prec < op_prec { write!(f, "({})", DisplayExpr { context: self.context, id: *r }) }
                else { write!(f, "{}", DisplayExpr { context: self.context, id: *r }) }
            },
            Expr::Div(l, r) => {
                let lhs_prec = precedence(self.context, *l);
                let rhs_prec = precedence(self.context, *r);
                let op_prec = 2; // Div precedence (same as Mul)
                
                if lhs_prec < op_prec { write!(f, "({})", DisplayExpr { context: self.context, id: *l })? }
                else { write!(f, "{}", DisplayExpr { context: self.context, id: *l })? }
                
                write!(f, " / ")?;
                
                // RHS of div always needs parens if it's Mul/Div or lower to be unambiguous? 
                // a / b * c -> (a / b) * c usually.
                // a / (b * c).
                // If RHS is Mul/Div, we need parens: a / (b * c) vs a / b * c.
                if rhs_prec <= op_prec { write!(f, "({})", DisplayExpr { context: self.context, id: *r }) }
                else { write!(f, "{}", DisplayExpr { context: self.context, id: *r }) }
            },
            Expr::Pow(b, e) => {
                let base_prec = precedence(self.context, *b);
                let op_prec = 3; // Pow precedence
                
                if base_prec < op_prec { write!(f, "({})", DisplayExpr { context: self.context, id: *b })? }
                else { write!(f, "{}", DisplayExpr { context: self.context, id: *b })? }
                
                write!(f, "^")?;
                
                // Exponent usually doesn't need parens if it's simple, but for clarity maybe?
                // x^(a+b) needs parens.
                // x^2 doesn't.
                // If exponent is complex, wrap in parens.
                let exp_prec = precedence(self.context, *e);
                let needs_parens = if exp_prec <= 4 {
                    true
                } else if let Expr::Number(n) = self.context.get(*e) {
                    !n.is_integer() || *n < num_rational::BigRational::zero() // If fraction or negative, add parens: x^(1/2), x^(-1)
                } else {
                    false
                };

                if needs_parens {
                    write!(f, "({})", DisplayExpr { context: self.context, id: *e })
                } else {
                    write!(f, "{}", DisplayExpr { context: self.context, id: *e })
                }
            },
            Expr::Neg(e) => {
                let inner_prec = precedence(self.context, *e);
                if inner_prec < 4 { // Neg precedence
                     write!(f, "-({})", DisplayExpr { context: self.context, id: *e })
                } else {
                     write!(f, "-{}", DisplayExpr { context: self.context, id: *e })
                }
            },
            Expr::Function(name, args) => {
                if name == "abs" && args.len() == 1 {
                    write!(f, "|{}|", DisplayExpr { context: self.context, id: args[0] })
                } else if name == "log" && args.len() == 2 {
                    // Check if base is 'e'
                    let base = self.context.get(args[0]);
                    if let Expr::Constant(Constant::E) = base {
                        write!(f, "ln({})", DisplayExpr { context: self.context, id: args[1] })
                    } else {
                        write!(f, "{}(", name)?;
                        for (i, arg) in args.iter().enumerate() {
                            if i > 0 { write!(f, ", ")?; }
                            write!(f, "{}", DisplayExpr { context: self.context, id: *arg })?;
                        }
                        write!(f, ")")
                    }
                } else if name == "factored" {
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 { write!(f, " * ")?; }
                        write!(f, "{}", DisplayExpr { context: self.context, id: *arg })?;
                    }
                    Ok(())
                } else if name == "factored_pow" && args.len() == 2 {
                    write!(f, "{}^{}", DisplayExpr { context: self.context, id: args[0] }, DisplayExpr { context: self.context, id: args[1] })
                } else {
                    write!(f, "{}(", name)?;
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{}", DisplayExpr { context: self.context, id: *arg })?;
                    }
                    write!(f, ")")
                }
            }
        }
    }
}

fn precedence(ctx: &Context, id: ExprId) -> i32 {
    match ctx.get(id) {
        Expr::Add(_, _) | Expr::Sub(_, _) => 1,
        Expr::Mul(_, _) | Expr::Div(_, _) => 2,
        Expr::Pow(_, _) => 3,
        Expr::Neg(_) => 4,
        Expr::Function(_, _) | Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) => 5,
    }
}
