use crate::rule::Rewrite;
use crate::define_rule;
use cas_ast::{Expr, ExprId, Context};
use num_traits::{Zero, One, Signed};
use num_integer::Integer;
use num_bigint::BigInt;
use num_rational::BigRational;

define_rule!(
    NumberTheoryRule,
    "Number Theory Operations",
    |ctx, expr| {
        let (name, args) = if let Expr::Function(name, args) = ctx.get(expr) {
            (name.clone(), args.clone())
        } else {
            return None;
        };

        match name.as_str() {
            "gcd" => {
                if args.len() == 2 {
                    if let Some(res) = compute_gcd(ctx, args[0], args[1]) {
                        return Some(Rewrite {
                            new_expr: res,
                            description: format!("gcd({:?}, {:?})", args[0], args[1]),
                        });
                    }
                }
            },
            "lcm" => {
                if args.len() == 2 {
                    if let Some(res) = compute_lcm(ctx, args[0], args[1]) {
                        return Some(Rewrite {
                            new_expr: res,
                            description: format!("lcm({:?}, {:?})", args[0], args[1]),
                        });
                    }
                }
            },
            "mod" => {
                if args.len() == 2 {
                    if let Some(res) = compute_mod(ctx, args[0], args[1]) {
                        return Some(Rewrite {
                            new_expr: res,
                            description: format!("mod({:?}, {:?})", args[0], args[1]),
                        });
                    }
                }
            },
            "prime_factors" | "factors" => {
                if args.len() == 1 {
                    if let Some(res) = compute_prime_factors(ctx, args[0]) {
                        return Some(Rewrite {
                            new_expr: res,
                            description: format!("prime_factors({:?})", args[0]),
                        });
                    }
                }
            },
            "fact" | "factorial" => {
                if args.len() == 1 {
                    if let Some(res) = compute_factorial(ctx, args[0]) {
                        return Some(Rewrite {
                            new_expr: res,
                            description: format!("fact({:?})", args[0]),
                        });
                    }
                }
            },
            "choose" | "nCr" => {
                if args.len() == 2 {
                    if let Some(res) = compute_choose(ctx, args[0], args[1]) {
                        return Some(Rewrite {
                            new_expr: res,
                            description: format!("choose({:?}, {:?})", args[0], args[1]),
                        });
                    }
                }
            },
            "perm" | "nPr" => {
                if args.len() == 2 {
                    if let Some(res) = compute_perm(ctx, args[0], args[1]) {
                        return Some(Rewrite {
                            new_expr: res,
                            description: format!("perm({:?}, {:?})", args[0], args[1]),
                        });
                    }
                }
            },
            _ => {}
        }
        None
    }
);

fn compute_gcd(ctx: &mut Context, a: ExprId, b: ExprId) -> Option<ExprId> {
    let val_a = get_integer(ctx, a)?;
    let val_b = get_integer(ctx, b)?;
    let gcd = val_a.gcd(&val_b);
    Some(ctx.add(Expr::Number(BigRational::from_integer(gcd))))
}

fn compute_lcm(ctx: &mut Context, a: ExprId, b: ExprId) -> Option<ExprId> {
    let val_a = get_integer(ctx, a)?;
    let val_b = get_integer(ctx, b)?;
    if val_a.is_zero() && val_b.is_zero() {
        return Some(ctx.num(0));
    }
    let lcm = val_a.lcm(&val_b);
    Some(ctx.add(Expr::Number(BigRational::from_integer(lcm))))
}

fn compute_mod(ctx: &mut Context, a: ExprId, n: ExprId) -> Option<ExprId> {
    let val_a = get_integer(ctx, a)?;
    let val_n = get_integer(ctx, n)?;
    if val_n.is_zero() {
        return None; // Undefined
    }
    // Euclidean remainder (always positive)
    let rem = ((val_a % &val_n) + &val_n) % &val_n;
    Some(ctx.add(Expr::Number(BigRational::from_integer(rem))))
}

fn compute_prime_factors(ctx: &mut Context, n: ExprId) -> Option<ExprId> {
    let val = get_integer(ctx, n)?;
    if val.is_zero() { return Some(ctx.num(0)); }
    if val.is_one() { return Some(ctx.num(1)); }
    
    let sign = if val.is_negative() { -1 } else { 1 };
    let mut n_abs = val.abs();
    
    let mut factors = Vec::new();
    
    // Simple trial division
    let one = BigInt::one();
    
    // Optimization: check 2 separately
    while n_abs.is_even() {
        factors.push(BigInt::from(2));
        n_abs /= 2;
    }
    
    let mut d = BigInt::from(3);
    while &d * &d <= n_abs {
        while (&n_abs % &d).is_zero() {
            factors.push(d.clone());
            n_abs /= &d;
        }
        d += 2;
    }
    if n_abs > one {
        factors.push(n_abs);
    }
    
    // Group factors: 2, 2, 3 -> 2^2 * 3
    // Since factors are sorted, we can just iterate
    let mut grouped = Vec::new();
    if !factors.is_empty() {
        let mut current = factors[0].clone();
        let mut count = 1;
        for f in factors.iter().skip(1) {
            if *f == current {
                count += 1;
            } else {
                grouped.push((current, count));
                current = f.clone();
                count = 1;
            }
        }
        grouped.push((current, count));
    }
    
    // Construct expression
    let mut exprs = Vec::new();
    if sign == -1 {
        exprs.push(ctx.num(-1));
    }
    
    for (base, exp) in grouped {
        let base_expr = ctx.add(Expr::Number(BigRational::from_integer(base)));
        if exp == 1 {
            exprs.push(base_expr);
        } else {
            let exp_expr = ctx.num(exp);
            // Use "factored_pow" to prevent EvaluateNumericPower from simplifying 2^2 -> 4
            exprs.push(ctx.add(Expr::Function("factored_pow".to_string(), vec![base_expr, exp_expr])));
        }
    }
    
    if exprs.is_empty() {
        return Some(ctx.num(1));
    }
    
    // Return as a "factored" function to prevent CombineConstants from undoing it
    Some(ctx.add(Expr::Function("factored".to_string(), exprs)))
}

fn compute_factorial(ctx: &mut Context, n: ExprId) -> Option<ExprId> {
    let val = get_integer(ctx, n)?;
    if val.is_negative() {
        return None; // Undefined for negative integers
    }
    
    // Limit factorial size to prevent hanging
    if val > BigInt::from(1000) {
        return None; // Too large to compute
    }
    
    let mut res = BigInt::one();
    let mut i = BigInt::one();
    while i <= val {
        res = res * &i;
        i = i + 1;
    }
    
    Some(ctx.add(Expr::Number(BigRational::from_integer(res))))
}

fn compute_choose(ctx: &mut Context, n: ExprId, k: ExprId) -> Option<ExprId> {
    let val_n = get_integer(ctx, n)?;
    let val_k = get_integer(ctx, k)?;
    
    if val_k.is_negative() || val_k > val_n {
        return Some(ctx.num(0));
    }
    
    // Optimization: nC0 = 1, nCn = 1
    if val_k.is_zero() || val_k == val_n {
        return Some(ctx.num(1));
    }
    
    // Symmetry: nCk = nC(n-k)
    let k_eff = if &val_k * 2 > val_n {
        &val_n - &val_k
    } else {
        val_k.clone()
    };
    
    // Compute: n * (n-1) * ... * (n-k+1) / k!
    let mut num = BigInt::one();
    let mut den = BigInt::one();
    
    let mut i = BigInt::zero();
    while i < k_eff {
        num = num * (&val_n - &i);
        den = den * (&i + 1);
        i = i + 1;
    }
    
    let res = num / den;
    Some(ctx.add(Expr::Number(BigRational::from_integer(res))))
}

fn compute_perm(ctx: &mut Context, n: ExprId, k: ExprId) -> Option<ExprId> {
    let val_n = get_integer(ctx, n)?;
    let val_k = get_integer(ctx, k)?;
    
    if val_k.is_negative() || val_k > val_n {
        return Some(ctx.num(0));
    }
    
    if val_k.is_zero() {
        return Some(ctx.num(1));
    }
    
    // Compute: n * (n-1) * ... * (n-k+1)
    let mut res = BigInt::one();
    let mut i = BigInt::zero();
    while i < val_k {
        res = res * (&val_n - &i);
        i = i + 1;
    }
    
    Some(ctx.add(Expr::Number(BigRational::from_integer(res))))
}

fn get_integer(ctx: &Context, expr: ExprId) -> Option<BigInt> {
    match ctx.get(expr) {
        Expr::Number(n) => {
            if n.is_integer() {
                Some(n.to_integer())
            } else {
                None
            }
        },
        Expr::Neg(e) => {
            get_integer(ctx, *e).map(|n| -n)
        },
        _ => None
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(NumberTheoryRule));
}
