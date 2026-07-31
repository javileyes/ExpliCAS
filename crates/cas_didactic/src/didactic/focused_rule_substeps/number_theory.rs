//! `focused_rule_substeps`: familia `number_theory`.
//!
//! Ver la cabecera de `focused_rule_substeps.rs` para el contexto.

use super::*;

pub(super) fn generate_number_theory_operation_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    // choose()/nCr keeps its dedicated binomial-coefficient substeps.
    if let Some((n, k)) = choose_integer_args(ctx, before) {
        let Some(value) = integer_value(ctx, after) else {
            return Vec::new();
        };
        if n < 0 || k < 0 || k > n {
            return Vec::new();
        }

        let complement = n - k;
        let quotient_plain = format!("{n}! / ({k}! · {complement}!)");
        let quotient_latex = format!("\\frac{{{n}!}}{{{k}!\\cdot {complement}!}}");

        return vec![
            formula_substep(
                format!("Usar C({n},{k}) = {n}! / ({k}! · {complement}!)"),
                &binom_plain(n, k),
                &quotient_plain,
                &binom_latex(n, k),
                &quotient_latex,
            ),
            formula_substep(
                format!("Calcular {n}! / ({k}! · {complement}!) = {value}"),
                &quotient_plain,
                &value.to_string(),
                &quotient_latex,
                &latex_expr(ctx, after),
            ),
        ];
    }

    // Other number-theory operations: explain HOW the result is reached with one concrete substep
    // (generic Spanish title localized via the locale fallback; the math lines carry the specifics).
    let Some((name, args)) = number_theory_integer_call(ctx, before) else {
        return Vec::new();
    };
    match (name.as_str(), args.as_slice()) {
        ("gcd", [a, b]) => number_theory_gcd_substeps(*a, *b),
        ("lcm", [a, b]) => number_theory_lcm_substeps(*a, *b),
        ("totient" | "eulerphi" | "phi", [n]) => number_theory_totient_substeps(*n),
        ("divisors", [n]) => number_theory_divisors_substeps(*n),
        ("sigma", [n]) => number_theory_sigma_substeps(*n),
        ("fibonacci" | "fib", [n]) => number_theory_fibonacci_substeps(*n),
        _ => Vec::new(),
    }
}

/// `(name, integer_args)` for a number-theory call like `gcd(48, 36)` or `totient(12)`.
fn number_theory_integer_call(ctx: &Context, expr: ExprId) -> Option<(String, Vec<i64>)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    let name = ctx.sym_name(*fn_id).to_string();
    let ints: Option<Vec<i64>> = args.iter().map(|&a| integer_value(ctx, a)).collect();
    Some((name, ints?))
}

fn number_theory_substep_latex(s: &str) -> String {
    s.replace(" · ", " \\cdot ")
        .replace('·', "\\cdot ")
        .replace(" → ", " \\to ")
        .replace('→', "\\to ")
        .replace("φ(", "\\varphi(")
        .replace("σ(", "\\sigma(")
}

fn number_theory_substep(title: &'static str, before: String, after: String) -> SubStep {
    let before_latex = number_theory_substep_latex(&before);
    let after_latex = number_theory_substep_latex(&after);
    formula_substep(title, &before, &after, &before_latex, &after_latex)
}

fn nt_gcd_value(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

pub(super) fn number_theory_gcd_substeps(a: i64, b: i64) -> Vec<SubStep> {
    if a.abs() > NUMBER_THEORY_SUBSTEP_MAX || b.abs() > NUMBER_THEORY_SUBSTEP_MAX {
        return Vec::new();
    }
    let (mut x, mut y) = (a.abs(), b.abs());
    let mut chain = vec![format!("gcd({x}, {y})")];
    while y != 0 {
        let r = x % y;
        x = y;
        y = r;
        chain.push(format!("gcd({x}, {y})"));
    }
    if chain.len() < 2 {
        return Vec::new();
    }
    let after = format!("{} = {x}", chain[1..].join(" = "));
    vec![number_theory_substep(
        "Aplicar el algoritmo de Euclides (restos sucesivos)",
        chain[0].clone(),
        after,
    )]
}

pub(super) fn number_theory_lcm_substeps(a: i64, b: i64) -> Vec<SubStep> {
    if a.abs() > NUMBER_THEORY_SUBSTEP_MAX || b.abs() > NUMBER_THEORY_SUBSTEP_MAX {
        return Vec::new();
    }
    let g = nt_gcd_value(a, b);
    let Some(prod) = a.abs().checked_mul(b.abs()) else {
        return Vec::new();
    };
    if g == 0 {
        return Vec::new();
    }
    let l = prod / g;
    let after = format!(
        "({} · {}) / gcd({}, {}) = {prod} / {g} = {l}",
        a.abs(),
        b.abs(),
        a.abs(),
        b.abs()
    );
    vec![number_theory_substep(
        "Usar lcm(a, b) = a · b / gcd(a, b)",
        format!("lcm({a}, {b})"),
        after,
    )]
}

pub(super) fn number_theory_totient_substeps(n: i64) -> Vec<SubStep> {
    if !(1..=NUMBER_THEORY_SUBSTEP_MAX).contains(&n) {
        return Vec::new();
    }
    let factors = nt_prime_factors(n);
    if factors.is_empty() {
        // totient(1) = 1; nothing to factor.
        return Vec::new();
    }
    let phi: i64 = factors.iter().fold(n, |acc, &(p, _)| acc / p * (p - 1));
    let terms: Vec<String> = factors
        .iter()
        .map(|&(p, _)| format!("(1 - 1/{p})"))
        .collect();
    vec![
        number_theory_substep(
            "Factorizar n en potencias de primos",
            format!("φ({n})"),
            format!("{n} = {}", nt_factorization_string(&factors)),
        ),
        number_theory_substep(
            "Aplicar la fórmula de Euler φ(n) = n · ∏(1 - 1/p)",
            format!("φ({n})"),
            format!("{n} · {} = {phi}", terms.join(" · ")),
        ),
    ]
}

pub(super) fn number_theory_divisors_substeps(n: i64) -> Vec<SubStep> {
    let n = n.abs();
    if !(1..=NUMBER_THEORY_SUBSTEP_MAX).contains(&n) {
        return Vec::new();
    }
    let factors = nt_prime_factors(n);
    let divs = nt_divisors_list(n);
    let divs_str = format!(
        "[{}]",
        divs.iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    vec![number_theory_substep(
        "Combinar las potencias de los factores primos",
        format!("{n} = {}", nt_factorization_string(&factors)),
        divs_str,
    )]
}

pub(super) fn number_theory_sigma_substeps(n: i64) -> Vec<SubStep> {
    let n = n.abs();
    if !(1..=NUMBER_THEORY_SUBSTEP_MAX).contains(&n) {
        return Vec::new();
    }
    let divs = nt_divisors_list(n);
    let sum: i64 = divs.iter().sum();
    let after = format!(
        "{} = {sum}",
        divs.iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(" + ")
    );
    vec![number_theory_substep(
        "Sumar todos los divisores",
        format!("σ({n})"),
        after,
    )]
}

pub(super) fn number_theory_fibonacci_substeps(n: i64) -> Vec<SubStep> {
    // F(90) is the largest Fibonacci number that fits in i64.
    if !(0..=90).contains(&n) {
        return Vec::new();
    }
    let len = n as usize + 1;
    let mut seq = vec![0i64, 1i64];
    while seq.len() < len {
        let next = seq[seq.len() - 1] + seq[seq.len() - 2];
        seq.push(next);
    }
    let value = seq[n as usize];
    let shown = seq[..len]
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    vec![number_theory_substep(
        "Aplicar la recurrencia F(n) = F(n-1) + F(n-2)",
        format!("F({n})"),
        format!("{shown} → {value}"),
    )]
}

pub(super) fn gcd_usize(a: usize, b: usize) -> usize {
    let mut a = a;
    let mut b = b;
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}
