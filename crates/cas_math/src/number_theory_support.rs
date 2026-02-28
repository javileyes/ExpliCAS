use crate::poly_gcd_mode::GcdMode;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumberTheoryDispatch {
    Simple(NumberTheorySimpleRewrite),
    PolyGcd {
        lhs: ExprId,
        rhs: ExprId,
        mode: GcdMode,
    },
}

/// Simple number-theory rewrite produced from a named function call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumberTheorySimpleRewrite {
    Unary {
        name: &'static str,
        arg: ExprId,
        result: ExprId,
    },
    Binary {
        name: &'static str,
        lhs: ExprId,
        rhs: ExprId,
        result: ExprId,
    },
}

impl NumberTheorySimpleRewrite {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Unary { name, .. } | Self::Binary { name, .. } => name,
        }
    }

    pub fn result(&self) -> ExprId {
        match self {
            Self::Unary { result, .. } | Self::Binary { result, .. } => *result,
        }
    }
}

/// Render canonical textual description for a simple number-theory rewrite.
///
/// Caller provides expression rendering (to keep this crate independent from
/// formatting crates).
pub fn render_number_theory_desc_with<F>(
    call: NumberTheorySimpleRewrite,
    mut render_expr: F,
) -> String
where
    F: FnMut(ExprId) -> String,
{
    match call {
        NumberTheorySimpleRewrite::Unary { name, arg, .. } => {
            format!("{}({})", name, render_expr(arg))
        }
        NumberTheorySimpleRewrite::Binary { name, lhs, rhs, .. } => {
            format!("{}({}, {})", name, render_expr(lhs), render_expr(rhs))
        }
    }
}

/// Evaluate simple named number-theory calls.
///
/// This handles integer-only/safe operations:
/// - `gcd` (integer case only; polynomial fallback lives in engine)
/// - `lcm`, `mod`
/// - `prime_factors` / `factors`
/// - `fact` / `factorial`
/// - `choose` / `nCr`
/// - `perm` / `nPr`
pub fn try_eval_simple_number_theory_call(
    ctx: &mut Context,
    name: &str,
    args: &[ExprId],
) -> Option<NumberTheorySimpleRewrite> {
    match name {
        "gcd" if args.len() >= 2 => {
            let result = compute_integer_gcd_expr(ctx, args[0], args[1])?;
            Some(NumberTheorySimpleRewrite::Binary {
                name: "gcd",
                lhs: args[0],
                rhs: args[1],
                result,
            })
        }
        "lcm" if args.len() == 2 => {
            let result = compute_lcm_expr(ctx, args[0], args[1])?;
            Some(NumberTheorySimpleRewrite::Binary {
                name: "lcm",
                lhs: args[0],
                rhs: args[1],
                result,
            })
        }
        "mod" if args.len() == 2 => {
            let result = compute_mod_expr(ctx, args[0], args[1])?;
            Some(NumberTheorySimpleRewrite::Binary {
                name: "mod",
                lhs: args[0],
                rhs: args[1],
                result,
            })
        }
        "prime_factors" | "factors" if args.len() == 1 => {
            let result = compute_prime_factors_expr(ctx, args[0])?;
            Some(NumberTheorySimpleRewrite::Unary {
                name: "factors",
                arg: args[0],
                result,
            })
        }
        "fact" | "factorial" if args.len() == 1 => {
            let result = compute_factorial_expr(ctx, args[0])?;
            Some(NumberTheorySimpleRewrite::Unary {
                name: "fact",
                arg: args[0],
                result,
            })
        }
        "choose" | "nCr" if args.len() == 2 => {
            let result = compute_choose_expr(ctx, args[0], args[1])?;
            Some(NumberTheorySimpleRewrite::Binary {
                name: "choose",
                lhs: args[0],
                rhs: args[1],
                result,
            })
        }
        "perm" | "nPr" if args.len() == 2 => {
            let result = compute_perm_expr(ctx, args[0], args[1])?;
            Some(NumberTheorySimpleRewrite::Binary {
                name: "perm",
                lhs: args[0],
                rhs: args[1],
                result,
            })
        }
        _ => None,
    }
}

/// Dispatch a number-theory function call into either:
/// - a fully solved simple rewrite, or
/// - a polynomial-gcd fallback request (`gcd` symbolic/poly case).
pub fn dispatch_number_theory_call(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<NumberTheoryDispatch> {
    let (name, args) = match ctx.get(expr) {
        Expr::Function(fn_id, args) => (ctx.sym_name(*fn_id).to_string(), args.clone()),
        _ => return None,
    };

    if let Some(simple) = try_eval_simple_number_theory_call(ctx, name.as_str(), &args) {
        return Some(NumberTheoryDispatch::Simple(simple));
    }

    if name == "gcd" && args.len() >= 2 {
        use crate::poly_gcd_mode::parse_gcd_mode;
        let explicit_mode = if args.len() >= 3 {
            Some(parse_gcd_mode(ctx, args[2]))
        } else {
            None
        };
        let mode = select_poly_gcd_mode(ctx, args[0], args[1], explicit_mode);
        return Some(NumberTheoryDispatch::PolyGcd {
            lhs: args[0],
            rhs: args[1],
            mode,
        });
    }

    None
}

/// Check if expression contains a `poly_result(...)` reference.
pub fn contains_poly_result(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            if ctx.is_builtin(*fn_id, BuiltinFn::PolyResult) {
                return true;
            }
            if ctx.is_builtin(*fn_id, BuiltinFn::Hold) && !args.is_empty() {
                return contains_poly_result(ctx, args[0]);
            }
            args.iter().any(|&arg| contains_poly_result(ctx, arg))
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            contains_poly_result(ctx, *l) || contains_poly_result(ctx, *r)
        }
        Expr::Pow(base, exp) => contains_poly_result(ctx, *base) || contains_poly_result(ctx, *exp),
        Expr::Neg(inner) => contains_poly_result(ctx, *inner),
        _ => false,
    }
}

/// Extract integer exponent from expression.
pub fn get_integer_exponent(ctx: &Context, exp: ExprId) -> Option<i64> {
    match ctx.get(exp) {
        Expr::Number(n) => {
            if n.is_integer() {
                n.to_integer().try_into().ok()
            } else {
                None
            }
        }
        Expr::Neg(inner) => get_integer_exponent(ctx, *inner).map(|k| -k),
        _ => None,
    }
}

/// Check if expression has large unexpanded powers (exponent > 2) over non-atomic bases.
pub fn has_large_unexpanded_power(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            if let Some(n) = get_integer_exponent(ctx, *exp) {
                if n > 2 && !matches!(ctx.get(*base), Expr::Variable(_) | Expr::Number(_)) {
                    return true;
                }
            }
            has_large_unexpanded_power(ctx, *base) || has_large_unexpanded_power(ctx, *exp)
        }
        Expr::Function(fn_id, args) => {
            if ctx.is_builtin(*fn_id, BuiltinFn::PolyResult) {
                return false;
            }
            args.iter().any(|&arg| has_large_unexpanded_power(ctx, arg))
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            has_large_unexpanded_power(ctx, *l) || has_large_unexpanded_power(ctx, *r)
        }
        Expr::Neg(inner) => has_large_unexpanded_power(ctx, *inner),
        _ => false,
    }
}

/// Select effective polynomial GCD mode, honoring explicit mode when provided.
///
/// Auto-mode heuristic:
/// - use `Modp` when expression references `poly_result(...)` or contains
///   large unexpanded powers over non-atomic bases;
/// - otherwise use `Structural`.
pub fn select_poly_gcd_mode(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    explicit_mode: Option<GcdMode>,
) -> GcdMode {
    if let Some(mode) = explicit_mode {
        return mode;
    }

    let needs_modp = contains_poly_result(ctx, a)
        || contains_poly_result(ctx, b)
        || has_large_unexpanded_power(ctx, a)
        || has_large_unexpanded_power(ctx, b);

    if needs_modp {
        GcdMode::Modp
    } else {
        GcdMode::Structural
    }
}

/// Extract exact integer value from expression.
pub fn extract_integer_bigint(ctx: &Context, expr: ExprId) -> Option<BigInt> {
    crate::expr_extract::extract_integer_exact(ctx, expr)
}

/// Compute integer GCD expression when both inputs are exact integers.
pub fn compute_integer_gcd_expr(ctx: &mut Context, a: ExprId, b: ExprId) -> Option<ExprId> {
    let val_a = extract_integer_bigint(ctx, a)?;
    let val_b = extract_integer_bigint(ctx, b)?;
    let gcd = val_a.gcd(&val_b);
    Some(ctx.add(Expr::Number(BigRational::from_integer(gcd))))
}

/// Compute `lcm(a, b)` when both inputs are exact integers.
pub fn compute_lcm_expr(ctx: &mut Context, a: ExprId, b: ExprId) -> Option<ExprId> {
    let val_a = extract_integer_bigint(ctx, a)?;
    let val_b = extract_integer_bigint(ctx, b)?;
    if val_a.is_zero() && val_b.is_zero() {
        return Some(ctx.num(0));
    }
    let lcm = val_a.lcm(&val_b);
    Some(ctx.add(Expr::Number(BigRational::from_integer(lcm))))
}

/// Compute Euclidean remainder `a mod n` when both inputs are exact integers.
pub fn compute_mod_expr(ctx: &mut Context, a: ExprId, n: ExprId) -> Option<ExprId> {
    let val_a = extract_integer_bigint(ctx, a)?;
    let val_n = extract_integer_bigint(ctx, n)?;
    if val_n.is_zero() {
        return None;
    }
    let rem = ((val_a % &val_n) + &val_n) % &val_n;
    Some(ctx.add(Expr::Number(BigRational::from_integer(rem))))
}

/// Compute prime factorization expression for exact integer input.
///
/// Returns a `factored(...)` expression with optional `factored_pow(base, exp)` nodes.
pub fn compute_prime_factors_expr(ctx: &mut Context, n: ExprId) -> Option<ExprId> {
    let val = extract_integer_bigint(ctx, n)?;
    if val.is_zero() {
        return Some(ctx.num(0));
    }
    if val.is_one() {
        return Some(ctx.num(1));
    }

    let sign = if val.is_negative() { -1 } else { 1 };
    let mut n_abs = val.abs();

    let mut factors = Vec::new();
    let one = BigInt::one();

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
            exprs.push(ctx.call("factored_pow", vec![base_expr, exp_expr]));
        }
    }

    if exprs.is_empty() {
        return Some(ctx.num(1));
    }

    Some(ctx.call("factored", exprs))
}

/// Compute `n!` for exact non-negative integer inputs, bounded by `n <= 1000`.
pub fn compute_factorial_expr(ctx: &mut Context, n: ExprId) -> Option<ExprId> {
    let val = extract_integer_bigint(ctx, n)?;
    if val.is_negative() {
        return None;
    }
    if val > BigInt::from(1000) {
        return None;
    }

    let mut res = BigInt::one();
    let mut i = BigInt::one();
    while i <= val {
        res *= &i;
        i += 1;
    }
    Some(ctx.add(Expr::Number(BigRational::from_integer(res))))
}

/// Compute binomial coefficient `n choose k` for exact integer inputs.
pub fn compute_choose_expr(ctx: &mut Context, n: ExprId, k: ExprId) -> Option<ExprId> {
    let val_n = extract_integer_bigint(ctx, n)?;
    let val_k = extract_integer_bigint(ctx, k)?;

    if val_k.is_negative() || val_k > val_n {
        return Some(ctx.num(0));
    }
    if val_k.is_zero() || val_k == val_n {
        return Some(ctx.num(1));
    }

    let k_eff = if &val_k * 2 > val_n {
        &val_n - &val_k
    } else {
        val_k
    };

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

/// Compute permutations `nPk` for exact integer inputs.
pub fn compute_perm_expr(ctx: &mut Context, n: ExprId, k: ExprId) -> Option<ExprId> {
    let val_n = extract_integer_bigint(ctx, n)?;
    let val_k = extract_integer_bigint(ctx, k)?;

    if val_k.is_negative() || val_k > val_n {
        return Some(ctx.num(0));
    }
    if val_k.is_zero() {
        return Some(ctx.num(1));
    }

    let mut res = BigInt::one();
    let mut i = BigInt::zero();
    while i < val_k {
        res *= &val_n - &i;
        i += 1;
    }

    Some(ctx.add(Expr::Number(BigRational::from_integer(res))))
}

/// Compute GCD of two integers with an educational Euclidean trace.
///
/// Returns `(gcd, steps)` where steps are human-readable Spanish explanations.
pub fn verbose_integer_gcd_steps(a_int: i64, b_int: i64) -> (i64, Vec<String>) {
    let mut a = a_int.abs();
    let mut b = b_int.abs();
    let mut steps = Vec::new();

    steps.push("Algoritmo de Euclides para enteros:".to_string());

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

/// Result of a (possibly explained) GCD computation.
#[derive(Debug, Clone)]
pub struct GcdResult {
    pub value: Option<ExprId>,
    pub steps: Vec<String>,
}

/// Compute GCD for integers/polynomials, with optional educational trace.
///
/// Behavior:
/// - Integer inputs: Euclidean GCD (exact).
/// - Univariate polynomial inputs on same variable: polynomial Euclidean GCD.
/// - Multivariate or unsupported symbolic inputs: conservative fallback to `1`.
pub fn compute_gcd(ctx: &mut Context, a: ExprId, b: ExprId, explain: bool) -> GcdResult {
    let mut steps = Vec::new();

    // Integer fast path.
    if let (Some(val_a), Some(val_b)) = (
        extract_integer_bigint(ctx, a),
        extract_integer_bigint(ctx, b),
    ) {
        use num_traits::ToPrimitive;

        if explain {
            if let (Some(a_i64), Some(b_i64)) = (val_a.to_i64(), val_b.to_i64()) {
                steps.push(format!(
                    "Intentando GCD numérico entre {} y {}",
                    a_i64, b_i64
                ));
                let (gcd, sub_steps) = verbose_integer_gcd_steps(a_i64, b_i64);
                steps.extend(sub_steps);
                let res = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(gcd))));
                return GcdResult {
                    value: Some(res),
                    steps,
                };
            }
        }

        let gcd = compute_integer_gcd_expr(ctx, a, b).expect("integer gcd after integer extract");
        return GcdResult {
            value: Some(gcd),
            steps,
        };
    }

    // Try univariate polynomial GCD.
    use crate::polynomial::Polynomial;

    let vars_a = cas_ast::collect_variables(ctx, a);
    let vars_b = cas_ast::collect_variables(ctx, b);

    if vars_a.len() == 1 && vars_a == vars_b {
        if let Some(var) = vars_a.iter().next() {
            if let (Ok(p_a), Ok(p_b)) = (
                Polynomial::from_expr(ctx, a, var),
                Polynomial::from_expr(ctx, b, var),
            ) {
                if explain {
                    steps.push(format!(
                        "Detectados polinomios univariados en '{}'. Aplicando Euclides polinómico.",
                        var
                    ));
                    let (gcd_poly, sub_steps) = verbose_poly_gcd(&p_a, &p_b);
                    steps.extend(sub_steps);
                    return GcdResult {
                        value: Some(gcd_poly.to_expr(ctx)),
                        steps,
                    };
                }

                let gcd_poly = p_a.gcd(&p_b);
                return GcdResult {
                    value: Some(gcd_poly.to_expr(ctx)),
                    steps,
                };
            }
        }
    }

    // Multivariable fallback: conservative, never incorrect.
    if !vars_a.is_empty() || !vars_b.is_empty() {
        if explain {
            if vars_a.len() > 1
                || vars_b.len() > 1
                || (vars_a != vars_b && !vars_a.is_empty() && !vars_b.is_empty())
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

/// Educational wrapper around [`compute_gcd`].
pub fn explain_gcd(ctx: &mut Context, a: ExprId, b: ExprId) -> GcdResult {
    compute_gcd(ctx, a, b, true)
}

fn format_rational_for_step(r: &BigRational) -> String {
    if r.is_integer() {
        r.to_integer().to_string()
    } else {
        format!("{}/{}", r.numer(), r.denom())
    }
}

fn format_polynomial_for_step(poly: &crate::polynomial::Polynomial) -> String {
    use num_traits::Signed;

    if poly.is_zero() {
        return "0".to_string();
    }

    let mut out = String::new();
    for (i, coeff) in poly.coeffs.iter().enumerate().rev() {
        if coeff.is_zero() {
            continue;
        }

        let sign_neg = coeff.is_negative();
        let abs_coeff = coeff.abs();
        let var = &poly.var;
        let term = if i == 0 {
            format_rational_for_step(&abs_coeff)
        } else {
            let var_part = if i == 1 {
                var.to_string()
            } else {
                format!("{}^{}", var, i)
            };
            if abs_coeff.is_one() {
                var_part
            } else {
                format!("{}*{}", format_rational_for_step(&abs_coeff), var_part)
            }
        };

        if out.is_empty() {
            if sign_neg {
                out.push('-');
            }
            out.push_str(&term);
        } else if sign_neg {
            out.push_str(" - ");
            out.push_str(&term);
        } else {
            out.push_str(" + ");
            out.push_str(&term);
        }
    }

    out
}

/// Compute GCD of two polynomials with educational Euclidean trace.
fn verbose_poly_gcd(
    p1: &crate::polynomial::Polynomial,
    p2: &crate::polynomial::Polynomial,
) -> (crate::polynomial::Polynomial, Vec<String>) {
    use crate::polynomial::Polynomial;

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
        steps.push(format!(
            "Polinomio A (grado {}): {}",
            a.degree(),
            format_polynomial_for_step(&a)
        ));
        steps.push(format!(
            "Polinomio B (grado {}): {}",
            b.degree(),
            format_polynomial_for_step(&b)
        ));

        let (_, r) = a
            .div_rem(&b)
            .unwrap_or_else(|_| (Polynomial::zero(a.var.clone()), a.clone()));

        steps.push("Dividimos A entre B.".to_string());
        if r.is_zero() {
            steps.push("El resto es 0 (división exacta).".to_string());
        } else {
            steps.push(format!(
                "El resto R es: {} (grado {})",
                format_polynomial_for_step(&r),
                r.degree()
            ));
            steps.push("La propiedad gcd(A, B) = gcd(B, R) nos permite descartar A.".to_string());
        }

        a = b;
        b = r;
        step_count += 1;
    }

    steps.push("--- Paso Final ---".to_string());
    steps.push(format!(
        "El último divisor no nulo es: {}",
        format_polynomial_for_step(&a)
    ));

    let gcd_final = if !a.is_zero() {
        let content = a.content();
        if !content.is_zero() && content != BigRational::one() {
            let inv_content = BigRational::one() / &content;
            let scalar = Polynomial::new(vec![inv_content], a.var.clone());
            let normalized = a.mul(&scalar);

            steps.push(format!(
                "Normalizamos el polinomio (contenido = {})",
                format_rational_for_step(&content)
            ));
            steps.push(
                "Dividimos todos los coeficientes por el GCD para obtener coeficientes enteros:"
                    .to_string(),
            );
            steps.push(format!(
                "Resultado final: {}",
                format_polynomial_for_step(&normalized)
            ));
            normalized
        } else {
            let lc = a.leading_coeff();
            if lc < BigRational::zero() {
                steps.push("Hacemos el coeficiente principal positivo:".to_string());
                let result = a.neg();
                steps.push(format!(
                    "Resultado final: {}",
                    format_polynomial_for_step(&result)
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

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn detects_poly_result_even_inside_hold() {
        let mut ctx = Context::new();
        let poly = parse("poly_result(7)", &mut ctx).expect("poly");
        let held = ctx.call_builtin(BuiltinFn::Hold, vec![poly]);
        let one = ctx.num(1);
        let wrapped = ctx.add(Expr::Add(one, held));
        assert!(contains_poly_result(&ctx, wrapped));
    }

    #[test]
    fn large_unexpanded_power_detection_matches_intent() {
        let mut ctx = Context::new();
        let large_compound = parse("(x+1)^3", &mut ctx).expect("compound");
        let large_atomic = parse("x^3", &mut ctx).expect("atomic");
        let poly_power = parse("(poly_result(3))^8", &mut ctx).expect("poly");

        assert!(has_large_unexpanded_power(&ctx, large_compound));
        assert!(!has_large_unexpanded_power(&ctx, large_atomic));
        assert!(has_large_unexpanded_power(&ctx, poly_power));
    }

    #[test]
    fn integer_exponent_extraction_handles_negation() {
        let mut ctx = Context::new();
        let exp = parse("-5", &mut ctx).expect("exp");
        assert_eq!(get_integer_exponent(&ctx, exp), Some(-5));
    }

    #[test]
    fn computes_integer_number_theory_primitives() {
        let mut ctx = Context::new();
        let a = ctx.num(12);
        let b = ctx.num(18);
        let n = ctx.num(5);
        let two = ctx.num(2);
        let minus_seven = ctx.num(-7);
        let five = ctx.num(5);

        let lcm = compute_lcm_expr(&mut ctx, a, b).expect("lcm");
        assert_eq!(extract_integer_bigint(&ctx, lcm), Some(BigInt::from(36)));

        let gcd = compute_integer_gcd_expr(&mut ctx, a, b).expect("gcd");
        assert_eq!(extract_integer_bigint(&ctx, gcd), Some(BigInt::from(6)));

        let modu = compute_mod_expr(&mut ctx, minus_seven, five).expect("mod");
        assert_eq!(extract_integer_bigint(&ctx, modu), Some(BigInt::from(3)));

        let fact = compute_factorial_expr(&mut ctx, n).expect("fact");
        assert_eq!(extract_integer_bigint(&ctx, fact), Some(BigInt::from(120)));

        let choose = compute_choose_expr(&mut ctx, five, two).expect("choose");
        assert_eq!(extract_integer_bigint(&ctx, choose), Some(BigInt::from(10)));

        let perm = compute_perm_expr(&mut ctx, five, two).expect("perm");
        assert_eq!(extract_integer_bigint(&ctx, perm), Some(BigInt::from(20)));
    }

    #[test]
    fn selects_poly_gcd_mode_explicit_or_auto() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let x2 = ctx.add(Expr::Pow(x, two));
        let x3 = ctx.add(Expr::Pow(x, three));

        // Explicit mode always wins
        let explicit = select_poly_gcd_mode(&ctx, x2, x3, Some(GcdMode::Structural));
        assert_eq!(explicit, GcdMode::Structural);

        // Auto mode: large unexpanded power over non-atomic base forces Modp
        let one = ctx.num(1);
        let five = ctx.num(5);
        let non_atomic_base = ctx.add(Expr::Add(x, one));
        let big_pow = ctx.add(Expr::Pow(non_atomic_base, five));
        let auto_modp = select_poly_gcd_mode(&ctx, big_pow, x2, None);
        assert_eq!(auto_modp, GcdMode::Modp);

        // Auto mode: small/simple cases stay structural
        let auto_structural = select_poly_gcd_mode(&ctx, x2, x3, None);
        assert_eq!(auto_structural, GcdMode::Structural);
    }

    #[test]
    fn verbose_integer_gcd_trace_is_consistent() {
        let (gcd, steps) = verbose_integer_gcd_steps(48, 18);
        assert_eq!(gcd, 6);
        assert!(!steps.is_empty());
        assert!(steps.iter().any(|s| s.contains("Máximo Común Divisor")));
    }

    #[test]
    fn evaluates_simple_named_number_theory_calls() {
        let mut ctx = Context::new();
        let a = ctx.num(12);
        let b = ctx.num(18);
        let five = ctx.num(5);
        let two = ctx.num(2);

        let gcd_call =
            try_eval_simple_number_theory_call(&mut ctx, "gcd", &[a, b]).expect("gcd call");
        assert_eq!(gcd_call.name(), "gcd");
        assert_eq!(
            extract_integer_bigint(&ctx, gcd_call.result()),
            Some(BigInt::from(6))
        );

        let choose_call =
            try_eval_simple_number_theory_call(&mut ctx, "nCr", &[five, two]).expect("choose call");
        assert_eq!(choose_call.name(), "choose");
        assert_eq!(
            extract_integer_bigint(&ctx, choose_call.result()),
            Some(BigInt::from(10))
        );

        let factor_call =
            try_eval_simple_number_theory_call(&mut ctx, "factorial", &[five]).expect("fact");
        assert_eq!(factor_call.name(), "fact");
        assert_eq!(
            extract_integer_bigint(&ctx, factor_call.result()),
            Some(BigInt::from(120))
        );

        let x = ctx.var("x");
        assert!(try_eval_simple_number_theory_call(&mut ctx, "gcd", &[x, five]).is_none());
    }

    #[test]
    fn dispatches_simple_and_poly_gcd_paths() {
        let mut ctx = Context::new();

        let gcd_int = parse("gcd(12, 18)", &mut ctx).expect("gcd int");
        match dispatch_number_theory_call(&mut ctx, gcd_int).expect("dispatch int") {
            NumberTheoryDispatch::Simple(call) => {
                assert_eq!(call.name(), "gcd");
                assert_eq!(
                    extract_integer_bigint(&ctx, call.result()),
                    Some(BigInt::from(6))
                );
            }
            NumberTheoryDispatch::PolyGcd { .. } => panic!("expected simple gcd path"),
        }

        let gcd_poly = parse("gcd(x^2-1, x-1)", &mut ctx).expect("gcd poly");
        match dispatch_number_theory_call(&mut ctx, gcd_poly).expect("dispatch poly") {
            NumberTheoryDispatch::PolyGcd { lhs, rhs, .. } => {
                assert_eq!(lhs, parse("x^2 - 1", &mut ctx).expect("lhs"));
                assert_eq!(rhs, parse("x - 1", &mut ctx).expect("rhs"));
            }
            NumberTheoryDispatch::Simple(_) => panic!("expected poly gcd fallback"),
        }
    }

    #[test]
    fn renders_simple_number_theory_description_with_callback() {
        let mut ctx = Context::new();
        let twelve = ctx.num(12);
        let eighteen = ctx.num(18);
        let rewrite = NumberTheorySimpleRewrite::Binary {
            name: "gcd",
            lhs: twelve,
            rhs: eighteen,
            result: ctx.num(6),
        };
        let desc = render_number_theory_desc_with(rewrite, |id| {
            if id == twelve {
                "12".to_string()
            } else if id == eighteen {
                "18".to_string()
            } else {
                "?".to_string()
            }
        });
        assert_eq!(desc, "gcd(12, 18)");
    }
}
