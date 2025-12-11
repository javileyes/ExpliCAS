use cas_ast::{Context, Expr, ExprId};
use num_traits::{One, Signed, ToPrimitive};

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
        }
        Expr::Number(n) => {
            if n.is_integer() && !n.is_negative() {
                let val = n.to_integer();
                let sqrt = val.sqrt();
                if &sqrt * &sqrt == val {
                    return Some(context.num(sqrt.to_i64()?));
                }
            }
            None
        }
        _ => None,
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

/// Flatten an Add/Sub chain into a list of terms, converting subtractions to Neg.
/// This is used by collect and grouping modules for like-term collection.
///
/// Unlike `flatten_add`, this handles:
/// - `Add(a, b)` → [a, b]
/// - `Sub(a, b)` → [a, Neg(b)]
/// - `Neg(Neg(x))` → [x]
pub fn flatten_add_sub_chain(ctx: &mut Context, expr: ExprId) -> Vec<ExprId> {
    let mut terms = Vec::new();
    flatten_add_sub_recursive(ctx, expr, &mut terms, false);
    terms
}

fn flatten_add_sub_recursive(
    ctx: &mut Context,
    expr: ExprId,
    terms: &mut Vec<ExprId>,
    negate: bool,
) {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Add(l, r) => {
            flatten_add_sub_recursive(ctx, l, terms, negate);
            flatten_add_sub_recursive(ctx, r, terms, negate);
        }
        Expr::Sub(l, r) => {
            flatten_add_sub_recursive(ctx, l, terms, negate);
            flatten_add_sub_recursive(ctx, r, terms, !negate);
        }
        Expr::Neg(inner) => {
            // Handle double negation: Neg(Neg(x)) -> x
            flatten_add_sub_recursive(ctx, inner, terms, !negate);
        }
        _ => {
            if negate {
                terms.push(ctx.add(Expr::Neg(expr)));
            } else {
                terms.push(expr);
            }
        }
    }
}

pub fn flatten_mul(ctx: &Context, expr: ExprId, factors: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            flatten_mul(ctx, *l, factors);
            flatten_mul(ctx, *r, factors);
        }
        _ => factors.push(expr),
    }
}

/// Flatten a Mul chain into a list of factors, handling Neg as -1 multiplication.
/// Returns Vec<ExprId> where Neg(e) is converted to [num(-1), ...factors of e...]
pub fn flatten_mul_chain(ctx: &mut Context, expr: ExprId) -> Vec<ExprId> {
    let mut factors = Vec::new();
    flatten_mul_recursive(ctx, expr, &mut factors);
    factors
}

fn flatten_mul_recursive(ctx: &mut Context, expr: ExprId, factors: &mut Vec<ExprId>) {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Mul(l, r) => {
            flatten_mul_recursive(ctx, l, factors);
            flatten_mul_recursive(ctx, r, factors);
        }
        Expr::Neg(e) => {
            // Treat Neg(e) as -1 * e
            let neg_one = ctx.num(-1);
            factors.push(neg_one);
            flatten_mul_recursive(ctx, e, factors);
        }
        _ => {
            factors.push(expr);
        }
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

// ========== Pi Helpers ==========

/// Check if expression equals π/n for a given denominator (handles both Div and Mul forms)
pub fn is_pi_over_n(ctx: &Context, expr: ExprId, denom: i32) -> bool {
    // Handle Div form: pi/n
    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Expr::Constant(c) = ctx.get(*num) {
            if matches!(c, cas_ast::Constant::Pi) {
                if let Expr::Number(n) = ctx.get(*den) {
                    return *n == num_rational::BigRational::from_integer(denom.into());
                }
            }
        }
    }

    // Handle Mul form: (1/n) * pi
    if let Expr::Mul(l, r) = ctx.get(expr) {
        let (num_part, const_part) = if let Expr::Constant(_) = ctx.get(*l) {
            (*r, *l)
        } else if let Expr::Constant(_) = ctx.get(*r) {
            (*l, *r)
        } else {
            return false;
        };

        if let Expr::Constant(c) = ctx.get(const_part) {
            if matches!(c, cas_ast::Constant::Pi) {
                if let Expr::Number(n) = ctx.get(num_part) {
                    return *n == num_rational::BigRational::new(1.into(), denom.into());
                }
            }
        }
    }

    false
}

/// Check if expression is exactly π
pub fn is_pi(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Constant(cas_ast::Constant::Pi))
}

/// Check if expression equals a specific numeric value
pub fn is_numeric_value(ctx: &Context, expr: ExprId, numer: i32, denom: i32) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        *n == num_rational::BigRational::new(numer.into(), denom.into())
    } else {
        false
    }
}

/// Build π/n expression
pub fn build_pi_over_n(ctx: &mut Context, denom: i64) -> ExprId {
    let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
    let d = ctx.num(denom);
    ctx.add(Expr::Div(pi, d))
}

/// Check if expression equals 1/2
pub fn is_half(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        *n.numer() == 1.into() && *n.denom() == 2.into()
    } else {
        false
    }
}

// ========== Common Expression Predicates ==========
// These functions were previously duplicated across multiple files.
// Now consolidated here for consistency and maintainability.

/// Check if expression is the number 1
pub fn is_one(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        n.is_one()
    } else {
        false
    }
}

/// Check if expression is the number 0
pub fn is_zero(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        num_traits::Zero::is_zero(n)
    } else {
        false
    }
}

/// Check if expression is a negative number
pub fn is_negative(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        n.is_negative()
    } else if let Expr::Neg(_) = ctx.get(expr) {
        true
    } else {
        false
    }
}

/// Try to extract an integer value from an expression
pub fn get_integer(ctx: &Context, expr: ExprId) -> Option<i64> {
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_integer() {
            n.to_integer().to_i64()
        } else {
            None
        }
    } else {
        None
    }
}

/// Get the variant name of an expression (for debugging/display)
pub fn get_variant_name(expr: &Expr) -> &'static str {
    match expr {
        Expr::Number(_) => "Number",
        Expr::Variable(_) => "Variable",
        Expr::Constant(_) => "Constant",
        Expr::Add(_, _) => "Add",
        Expr::Sub(_, _) => "Sub",
        Expr::Mul(_, _) => "Mul",
        Expr::Div(_, _) => "Div",
        Expr::Pow(_, _) => "Pow",
        Expr::Neg(_) => "Neg",
        Expr::Function(_, _) => "Function",
        Expr::Matrix { .. } => "Matrix",
    }
}
