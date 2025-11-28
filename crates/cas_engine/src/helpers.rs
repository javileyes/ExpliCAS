use cas_ast::Expr;
use std::rc::Rc;
use num_traits::{ToPrimitive, Signed, One};

pub fn is_trig_pow(expr: &Rc<Expr>, name: &str, power: i64) -> bool {
    if let Expr::Pow(base, exp) = expr.as_ref() {
        if let Expr::Number(n) = exp.as_ref() {
            if n.is_integer() && n.to_integer() == power.into() {
                if let Expr::Function(func_name, args) = base.as_ref() {
                    return func_name == name && args.len() == 1;
                }
            }
        }
    }
    false
}

pub fn get_trig_arg(expr: &Rc<Expr>) -> Option<Rc<Expr>> {
    if let Expr::Pow(base, _) = expr.as_ref() {
        if let Expr::Function(_, args) = base.as_ref() {
            if args.len() == 1 {
                return Some(args[0].clone());
            }
        }
    }
    None
}

pub fn get_square_root(expr: &Rc<Expr>) -> Option<Rc<Expr>> {
    match expr.as_ref() {
        Expr::Pow(b, e) => {
            if let Expr::Number(n) = e.as_ref() {
                if n.is_integer() {
                    let val = n.to_integer();
                    if &val % 2 == 0.into() {
                        let two = num_bigint::BigInt::from(2);
                        let new_exp = Expr::num((val / two).to_i64().unwrap());
                        // If new_exp is 1, simplify to b
                        if let Expr::Number(ne) = new_exp.as_ref() {
                            if ne.is_one() {
                                return Some(b.clone());
                            }
                        }
                        return Some(Expr::pow(b.clone(), new_exp));
                    }
                }
            }
            None
        },
        // Handle sin(x)^4 -> sin(x)^2
        // Handle 4 -> 2
        Expr::Number(n) => {
             // Check if n is a perfect square
             // For simplicity, only handle positive integers for now
             if n.is_integer() && !n.is_negative() {
                 let val = n.to_integer();
                 // Simple integer sqrt check
                 let sqrt = val.sqrt();
                 if &sqrt * &sqrt == val {
                     return Some(Expr::num(sqrt.to_i64().unwrap()));
                 }
             }
             None
        },
        _ => None
    }
}

pub fn extract_double_angle_arg(expr: &Expr) -> Option<Rc<Expr>> {
    if let Expr::Mul(lhs, rhs) = expr {
        if let Expr::Number(n) = lhs.as_ref() {
            if n.is_integer() && n.to_integer() == 2.into() {
                return Some(rhs.clone());
            }
        }
        if let Expr::Number(n) = rhs.as_ref() {
            if n.is_integer() && n.to_integer() == 2.into() {
                return Some(lhs.clone());
            }
        }
    }
    None
}
