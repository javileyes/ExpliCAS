//! Number theory helpers exposed by solver facade.

use cas_ast::{Context, ExprId};
use cas_math::number_theory_support::{compute_integer_gcd_expr, extract_integer_bigint};
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

pub use cas_math::number_theory_support::compute_gcd;

/// Result of an explained GCD computation.
#[derive(Debug, Clone)]
pub struct GcdResult {
    pub value: Option<ExprId>,
    pub steps: Vec<String>,
}

/// Educational wrapper around pure GCD computation.
pub fn explain_gcd(ctx: &mut Context, a: ExprId, b: ExprId) -> GcdResult {
    let mut steps = Vec::new();

    if let (Some(val_a), Some(val_b)) = (
        extract_integer_bigint(ctx, a),
        extract_integer_bigint(ctx, b),
    ) {
        if let (Some(a_i64), Some(b_i64)) = (val_a.to_i64(), val_b.to_i64()) {
            steps.push(format!(
                "Intentando GCD numérico entre {} y {}",
                a_i64, b_i64
            ));
            let (gcd, sub_steps) = verbose_integer_gcd_steps(a_i64, b_i64);
            steps.extend(sub_steps);
            let value = ctx.num(gcd);
            return GcdResult {
                value: Some(value),
                steps,
            };
        }

        return GcdResult {
            value: compute_integer_gcd_expr(ctx, a, b),
            steps,
        };
    }

    let vars_a = cas_ast::collect_variables(ctx, a);
    let vars_b = cas_ast::collect_variables(ctx, b);
    if vars_a.len() == 1 && vars_a == vars_b {
        if let Some(var) = vars_a.iter().next() {
            if let (Ok(p_a), Ok(p_b)) = (
                Polynomial::from_expr(ctx, a, var),
                Polynomial::from_expr(ctx, b, var),
            ) {
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
        }
    }

    if !vars_a.is_empty() || !vars_b.is_empty() {
        if vars_a.len() > 1
            || vars_b.len() > 1
            || (vars_a != vars_b && !vars_a.is_empty() && !vars_b.is_empty())
        {
            steps.push("Detectados polinomios multivariables.".to_string());
            steps.push(
                "LIMITACIÓN: El GCD de polinomios multivariables no está implementado.".to_string(),
            );
            steps.push("Devolviendo GCD = 1 (conservador, no simplifica).".to_string());
        } else {
            steps.push("No se pudo calcular el GCD para estas expresiones.".to_string());
        }
        return GcdResult {
            value: Some(ctx.num(1)),
            steps,
        };
    }

    steps.push("No se pudo calcular el GCD".to_string());
    GcdResult { value: None, steps }
}

fn verbose_integer_gcd_steps(a_int: i64, b_int: i64) -> (i64, Vec<String>) {
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

fn format_rational_for_step(r: &BigRational) -> String {
    if r.is_integer() {
        r.to_integer().to_string()
    } else {
        format!("{}/{}", r.numer(), r.denom())
    }
}

fn format_polynomial_for_step(poly: &Polynomial) -> String {
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

fn verbose_poly_gcd(p1: &Polynomial, p2: &Polynomial) -> (Polynomial, Vec<String>) {
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
