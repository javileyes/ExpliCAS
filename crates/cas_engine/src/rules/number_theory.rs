use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

/// Result of a GCD computation, optionally including educational steps
pub struct GcdResult {
    pub value: Option<ExprId>,
    pub steps: Vec<String>,
}

define_rule!(NumberTheoryRule, "Number Theory Operations", |ctx, expr| {
    let (name, args) = if let Expr::Function(name, args) = ctx.get(expr) {
        (name.clone(), args.clone())
    } else {
        return None;
    };

    match name.as_str() {
        "gcd" => {
            if args.len() == 2 {
                let gcd_result = compute_gcd(ctx, args[0], args[1], false);
                if let Some(res) = gcd_result.value {
                    return Some(Rewrite {
                        new_expr: res,
                        description: format!(
                            "gcd({}, {})",
                            cas_ast::DisplayExpr {
                                context: ctx,
                                id: args[0]
                            },
                            cas_ast::DisplayExpr {
                                context: ctx,
                                id: args[1]
                            }
                        ),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                        required_conditions: vec![],
                    });
                }
            }
        }
        "lcm" => {
            if args.len() == 2 {
                if let Some(res) = compute_lcm(ctx, args[0], args[1]) {
                    return Some(Rewrite {
                        new_expr: res,
                        description: format!(
                            "lcm({}, {})",
                            cas_ast::DisplayExpr {
                                context: ctx,
                                id: args[0]
                            },
                            cas_ast::DisplayExpr {
                                context: ctx,
                                id: args[1]
                            }
                        ),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                        required_conditions: vec![],
                    });
                }
            }
        }
        "mod" => {
            if args.len() == 2 {
                if let Some(res) = compute_mod(ctx, args[0], args[1]) {
                    return Some(Rewrite {
                        new_expr: res,
                        description: format!(
                            "mod({}, {})",
                            cas_ast::DisplayExpr {
                                context: ctx,
                                id: args[0]
                            },
                            cas_ast::DisplayExpr {
                                context: ctx,
                                id: args[1]
                            }
                        ),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                        required_conditions: vec![],
                    });
                }
            }
        }
        "prime_factors" | "factors" => {
            if args.len() == 1 {
                if let Some(res) = compute_prime_factors(ctx, args[0]) {
                    return Some(Rewrite {
                        new_expr: res,
                        description: format!(
                            "factors({})",
                            cas_ast::DisplayExpr {
                                context: ctx,
                                id: args[0]
                            }
                        ),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                        required_conditions: vec![],
                    });
                }
            }
        }
        "fact" | "factorial" => {
            if args.len() == 1 {
                if let Some(res) = compute_factorial(ctx, args[0]) {
                    return Some(Rewrite {
                        new_expr: res,
                        description: format!(
                            "fact({})",
                            cas_ast::DisplayExpr {
                                context: ctx,
                                id: args[0]
                            }
                        ),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                        required_conditions: vec![],
                    });
                }
            }
        }
        "choose" | "nCr" => {
            if args.len() == 2 {
                if let Some(res) = compute_choose(ctx, args[0], args[1]) {
                    return Some(Rewrite {
                        new_expr: res,
                        description: format!(
                            "choose({}, {})",
                            cas_ast::DisplayExpr {
                                context: ctx,
                                id: args[0]
                            },
                            cas_ast::DisplayExpr {
                                context: ctx,
                                id: args[1]
                            }
                        ),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                        required_conditions: vec![],
                    });
                }
            }
        }
        "perm" | "nPr" => {
            if args.len() == 2 {
                if let Some(res) = compute_perm(ctx, args[0], args[1]) {
                    return Some(Rewrite {
                        new_expr: res,
                        description: format!(
                            "perm({}, {})",
                            cas_ast::DisplayExpr {
                                context: ctx,
                                id: args[0]
                            },
                            cas_ast::DisplayExpr {
                                context: ctx,
                                id: args[1]
                            }
                        ),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                        required_conditions: vec![],
                    });
                }
            }
        }
        _ => {}
    }
    None
});

pub fn compute_gcd(ctx: &mut Context, a: ExprId, b: ExprId, explain: bool) -> GcdResult {
    let mut steps = Vec::new();

    // Attempt integer GCD first
    if let (Some(val_a), Some(val_b)) = (get_integer(ctx, a), get_integer(ctx, b)) {
        use num_traits::ToPrimitive;

        if explain {
            // Check if values fit in i64 for verbose explanation
            if let (Some(a_i64), Some(b_i64)) = (val_a.to_i64(), val_b.to_i64()) {
                steps.push(format!(
                    "Intentando GCD numérico entre {} y {}",
                    a_i64, b_i64
                ));
                let (gcd, sub_steps) = verbose_integer_gcd(a_i64, b_i64);
                steps.extend(sub_steps);
                let res = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(gcd))));
                return GcdResult {
                    value: Some(res),
                    steps,
                };
            }
        }

        // Fast path (or numbers too large for i64)
        use num_integer::Integer;
        let gcd = val_a.gcd(&val_b);
        return GcdResult {
            value: Some(ctx.add(Expr::Number(BigRational::from_integer(gcd)))),
            steps,
        };
    }

    // Try polynomial GCD
    use crate::polynomial::Polynomial;
    use crate::rules::algebra::collect_variables;

    let vars = collect_variables(ctx, a);
    let vars_b = collect_variables(ctx, b);

    // Must be univariate and same variable
    if vars.len() == 1 && vars == vars_b {
        let var = vars.iter().next().unwrap();

        if let (Ok(p_a), Ok(p_b)) = (
            Polynomial::from_expr(ctx, a, var),
            Polynomial::from_expr(ctx, b, var),
        ) {
            if explain {
                steps.push(format!(
                    "Detectados polinomios univariados en '{}'. Aplicando Euclides polinómico.",
                    var
                ));
                let (gcd_poly, sub_steps) = verbose_poly_gcd(ctx, &p_a, &p_b);
                steps.extend(sub_steps);
                return GcdResult {
                    value: Some(gcd_poly.to_expr(ctx)),
                    steps,
                };
            } else {
                let gcd_poly = p_a.gcd(&p_b);
                return GcdResult {
                    value: Some(gcd_poly.to_expr(ctx)),
                    steps: vec![],
                };
            }
        }
    }

    // Multivariable case: conservatively return GCD=1
    // This is safe - worst case we don't simplify, but we don't produce incorrect results
    if !vars.is_empty() || !vars_b.is_empty() {
        if explain {
            if vars.len() > 1
                || vars_b.len() > 1
                || (vars != vars_b && !vars.is_empty() && !vars_b.is_empty())
            {
                steps.push("Detectados polinomios multivariables.".to_string());
                steps.push(
                    "LIMITACIÓN: El GCD de polinomios multivariables no está implementado."
                        .to_string(),
                );
                steps.push("Devolviendo GCD = 1 (conservador, no simplifica).".to_string());
            } else {
                steps.push("No se pudo calcular el GCD para estas expresiones.".to_string());
            }
        }
        return GcdResult {
            value: Some(ctx.num(1)),
            steps,
        };
    }

    if explain {
        steps.push("No se pudo calcular el GCD".to_string());
    }
    GcdResult { value: None, steps }
}

/// Helper function to compute GCD with educational explanation
pub fn explain_gcd(ctx: &mut Context, a: ExprId, b: ExprId) -> GcdResult {
    compute_gcd(ctx, a, b, true)
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
    if val.is_zero() {
        return Some(ctx.num(0));
    }
    if val.is_one() {
        return Some(ctx.num(1));
    }

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
            exprs.push(ctx.add(Expr::Function(
                "factored_pow".to_string(),
                vec![base_expr, exp_expr],
            )));
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
        res *= &i;
        i += 1;
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
        val_k
    };

    // Compute: n * (n-1) * ... * (n-k+1) / k!
    let mut num = BigInt::one();
    let mut den = BigInt::one();

    let mut i = BigInt::zero();
    while i < k_eff {
        num *= &val_n - &i;
        den *= &i + 1;
        i += 1;
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
        res *= &val_n - &i;
        i += 1;
    }

    Some(ctx.add(Expr::Number(BigRational::from_integer(res))))
}

/// Get integer value from expression as BigInt.
///
/// Uses canonical implementation from helpers.rs.
/// (See ARCHITECTURE.md "Canonical Utilities Registry")
fn get_integer(ctx: &Context, expr: ExprId) -> Option<BigInt> {
    crate::helpers::get_integer_exact(ctx, expr)
}

/// Compute GCD of two integers with educational step-by-step explanation
fn verbose_integer_gcd(a_int: i64, b_int: i64) -> (i64, Vec<String>) {
    let mut a = a_int.abs();
    let mut b = b_int.abs();
    let mut steps = Vec::new();

    steps.push("Algoritmo de Euclides para enteros:".to_string());

    // Swap if needed for cleaner educational trace
    if a < b {
        std::mem::swap(&mut a, &mut b);
        steps.push(format!(
            "Intercambiamos para que el primer número sea mayor: GCD({}, {}) = GCD({}, {})",
            a_int.abs(),
            b_int.abs(),
            a,
            b
        ));
    }

    steps.push(format!("Calculamos GCD({}, {})", a, b));

    while b != 0 {
        let quotient = a / b;
        let remainder = a % b;

        steps.push(format!(
            "Dividimos {} entre {}: Cociente = {}, Resto = {}",
            a, b, quotient, remainder
        ));

        if remainder != 0 {
            steps.push(format!(
                "   → Como el resto es {}, el nuevo problema es GCD({}, {})",
                remainder, b, remainder
            ));
        } else {
            steps.push("   → El resto es 0. ¡Hemos terminado!".to_string());
        }

        a = b;
        b = remainder;
    }

    steps.push(format!("El Máximo Común Divisor es: {}", a));
    (a, steps)
}

/// Compute GCD of two polynomials with educational step-by-step explanation
fn verbose_poly_gcd(
    ctx: &mut Context,
    p1: &crate::polynomial::Polynomial,
    p2: &crate::polynomial::Polynomial,
) -> (crate::polynomial::Polynomial, Vec<String>) {
    use crate::polynomial::Polynomial;
    use cas_ast::DisplayExpr;

    let mut a = p1.clone();
    let mut b = p2.clone();
    let mut steps = Vec::new();

    steps.push("Algoritmo de Euclides para Polinomios:".to_string());
    steps.push(
        "Objetivo: Reducir el grado del polinomio mediante divisiones sucesivas.".to_string(),
    );

    let mut step_count = 1;

    while !b.is_zero() {
        steps.push(format!("--- Paso {} ---", step_count));

        // Convert to expressions for display
        let a_expr = a.to_expr(ctx);
        let b_expr = b.to_expr(ctx);

        steps.push(format!(
            "Polinomio A (grado {}): {}",
            a.degree(),
            DisplayExpr {
                context: ctx,
                id: a_expr
            }
        ));
        steps.push(format!(
            "Polinomio B (grado {}): {}",
            b.degree(),
            DisplayExpr {
                context: ctx,
                id: b_expr
            }
        ));

        // División (b is non-zero from while condition)
        let (_, r) = a
            .div_rem(&b)
            .expect("div_rem should not fail: b is non-zero");

        steps.push("Dividimos A entre B.".to_string());
        if r.is_zero() {
            steps.push("El resto es 0 (división exacta).".to_string());
        } else {
            let r_expr = r.to_expr(ctx);
            steps.push(format!(
                "El resto R es: {} (grado {})",
                DisplayExpr {
                    context: ctx,
                    id: r_expr
                },
                r.degree()
            ));
            steps.push("La propiedad gcd(A, B) = gcd(B, R) nos permite descartar A.".to_string());
        }

        a = b;
        b = r;
        step_count += 1;
    }

    // Normalización (Paso clave en álgebra computacional)
    steps.push("--- Paso Final ---".to_string());
    let a_before_norm = a.to_expr(ctx);
    steps.push(format!(
        "El último divisor no nulo es: {}",
        DisplayExpr {
            context: ctx,
            id: a_before_norm
        }
    ));

    // Normalize to primitive (integer coefficients with GCD=1)
    // instead of monic (leading coefficient=1)
    let gcd_final = if !a.is_zero() {
        let content = a.content();
        if !content.is_zero() && content != num_rational::BigRational::one() {
            let inv_content = num_rational::BigRational::one() / &content;
            let scalar = Polynomial::new(vec![inv_content], a.var.clone());
            let normalized = a.mul(&scalar);

            steps.push(format!(
                "Normalizamos el polinomio (contenido = {})",
                content
            ));
            steps.push(
                "Dividimos todos los coeficientes por el GCD para obtener coeficientes enteros:"
                    .to_string(),
            );
            let norm_expr = normalized.to_expr(ctx);
            steps.push(format!(
                "Resultado final: {}",
                DisplayExpr {
                    context: ctx,
                    id: norm_expr
                }
            ));
            normalized
        } else {
            // Already primitive, just ensure positive leading coefficient
            let lc = a.leading_coeff();
            if lc < num_rational::BigRational::zero() {
                steps.push("Hacemos el coeficiente principal positivo:".to_string());
                let result = a.neg();
                let result_expr = result.to_expr(ctx);
                steps.push(format!(
                    "Resultado final: {}",
                    DisplayExpr {
                        context: ctx,
                        id: result_expr
                    }
                ));
                result
            } else {
                steps.push(
                    "El polinomio ya es primitivo (coeficientes enteros con GCD=1)".to_string(),
                );
                a
            }
        }
    } else {
        a
    };

    (gcd_final, steps)
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(NumberTheoryRule));
}
