use cas_ast::{Expr, ExprId, Context};
use num_traits::{ToPrimitive, Signed, One};

pub fn is_trig_pow(context: &Context, expr: ExprId, name: &str, power: i64) -> bool {
    if let Expr::Pow(base, exp) = context.get(expr) {
        if let Expr::Number(n) = context.get(*exp) {
            if n.is_integer() && n.to_integer() == power.into() {
                if let Expr::Function(func_name, args) = context.get(*base) {
                    return func_name == name && args.len() == 1;
                }
            }
        }
    }
    false
}

pub fn get_trig_arg(context: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Pow(base, _) = context.get(expr) {
        if let Expr::Function(_, args) = context.get(*base) {
            if args.len() == 1 {
                return Some(args[0]);
            }
        }
    }
    None
}

pub fn get_square_root(context: &mut Context, expr: ExprId) -> Option<ExprId> {
    // We need to clone the expression to avoid borrowing issues if we need to inspect it deeply
    // But context.get returns reference.
    // We can't hold reference to context while mutating it.
    // So we should extract necessary data first.
    
    let expr_data = context.get(expr).clone();
    
    match expr_data {
        Expr::Pow(b, e) => {
            if let Expr::Number(n) = context.get(e) {
                if n.is_integer() {
                    let val = n.to_integer();
                    if &val % 2 == 0.into() {
                        let two = num_bigint::BigInt::from(2);
                        let new_exp_val = (val / two).to_i64()?;
                        let new_exp = context.num(new_exp_val);
                        
                        // If new_exp is 1, simplify to b
                        if let Expr::Number(ne) = context.get(new_exp) {
                            if ne.is_one() {
                                return Some(b);
                            }
                        }
                        return Some(context.add(Expr::Pow(b, new_exp)));
                    }
                }
            }
            None
        },
        Expr::Number(n) => {
             if n.is_integer() && !n.is_negative() {
                 let val = n.to_integer();
                 let sqrt = val.sqrt();
                 if &sqrt * &sqrt == val {
                     return Some(context.num(sqrt.to_i64()?));
                 }
             }
             None
        },
        _ => None
    }
}

pub fn extract_double_angle_arg(context: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Mul(lhs, rhs) = context.get(expr) {
        if let Expr::Number(n) = context.get(*lhs) {
            if n.is_integer() && n.to_integer() == 2.into() {
                return Some(*rhs);
            }
        }
        if let Expr::Number(n) = context.get(*rhs) {
            if n.is_integer() && n.to_integer() == 2.into() {
                return Some(*lhs);
            }
        }
    }
    None
}

pub fn flatten_add(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            flatten_add(ctx, *l, terms);
            flatten_add(ctx, *r, terms);
        }
        _ => terms.push(expr),
    }
}

pub fn get_parts(context: &mut Context, e: ExprId) -> (num_rational::BigRational, ExprId) {
    match context.get(e) {
        Expr::Mul(a, b) => {
            if let Expr::Number(n) = context.get(*a) {
                (n.clone(), *b)
            } else if let Expr::Number(n) = context.get(*b) {
                (n.clone(), *a)
            } else {
                (num_rational::BigRational::one(), e)
            }
        }
        Expr::Number(n) => (n.clone(), context.num(1)), // Treat constant as c * 1
        _ => (num_rational::BigRational::one(), e),
    }
}
