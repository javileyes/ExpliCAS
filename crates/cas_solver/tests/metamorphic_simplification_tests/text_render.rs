//! `metamorphic_simplification_tests`: familia `text_render`.
//!
//! Ver la cabecera de `metamorphic_simplification_tests.rs` para el contexto.

use super::*;

pub(super) fn expr_text(ctx: &Context, expr: ExprId) -> String {
    DisplayExpr {
        context: ctx,
        id: expr,
    }
    .to_string()
}

pub(super) fn normalize_pair_text(expr: &str) -> String {
    expr.chars().filter(|c| !c.is_whitespace()).collect()
}

pub(super) fn alpha_normalize_pair_text(expr: &str) -> Option<String> {
    fn inner(ctx: &Context, expr: ExprId, vars: &mut HashMap<String, usize>) -> String {
        match ctx.get(expr) {
            Expr::Number(n) => format!("N({}/{})", n.numer(), n.denom()),
            Expr::Variable(sym) => {
                let name = ctx.sym_name(*sym).to_string();
                let idx = match vars.get(&name) {
                    Some(idx) => *idx,
                    None => {
                        let idx = vars.len();
                        vars.insert(name, idx);
                        idx
                    }
                };
                format!("V({idx})")
            }
            Expr::Constant(c) => format!("C({c:?})"),
            Expr::Add(l, r) => {
                let mut parts = [inner(ctx, *l, vars), inner(ctx, *r, vars)];
                parts.sort_unstable();
                format!("Add({},{})", parts[0], parts[1])
            }
            Expr::Sub(l, r) => format!("Sub({},{})", inner(ctx, *l, vars), inner(ctx, *r, vars)),
            Expr::Mul(l, r) => {
                let mut parts = [inner(ctx, *l, vars), inner(ctx, *r, vars)];
                parts.sort_unstable();
                format!("Mul({},{})", parts[0], parts[1])
            }
            Expr::Div(l, r) => format!("Div({},{})", inner(ctx, *l, vars), inner(ctx, *r, vars)),
            Expr::Pow(b, e) => format!("Pow({},{})", inner(ctx, *b, vars), inner(ctx, *e, vars)),
            Expr::Neg(e) => format!("Neg({})", inner(ctx, *e, vars)),
            Expr::Function(name, args) => {
                let args = args
                    .iter()
                    .map(|arg| inner(ctx, *arg, vars))
                    .collect::<Vec<_>>()
                    .join(",");
                format!("Fn({};{})", ctx.sym_name(*name), args)
            }
            Expr::Matrix { rows, cols, data } => {
                let data = data
                    .iter()
                    .map(|cell| inner(ctx, *cell, vars))
                    .collect::<Vec<_>>()
                    .join(",");
                format!("Mat({rows}x{cols};{data})")
            }
            Expr::SessionRef(id) => format!("Ref({id})"),
            Expr::Hold(e) => format!("Hold({})", inner(ctx, *e, vars)),
        }
    }

    let mut ctx = Context::new();
    let expr = parse(expr, &mut ctx).ok()?;
    let mut vars = HashMap::new();
    Some(inner(&ctx, expr, &mut vars))
}

pub(super) fn split_top_level_mul_factors_text(text: &str) -> Vec<&str> {
    let mut factors = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;

    for (idx, ch) in text.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            '*' if depth == 0 => {
                factors.push(&text[start..idx]);
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
    }

    if start <= text.len() {
        factors.push(&text[start..]);
    }
    factors
}

fn split_top_level_add_terms_text(text: &str) -> Vec<&str> {
    let mut terms = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;

    for (idx, ch) in text.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            '+' if depth == 0 => {
                terms.push(&text[start..idx]);
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
    }

    if start <= text.len() {
        terms.push(&text[start..]);
    }
    terms
}

fn is_simple_ident_text(text: &str) -> bool {
    let mut chars = text.chars();
    matches!(chars.next(), Some(ch) if ch.is_ascii_alphabetic() || ch == '_')
        && chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn extract_three_linear_shift_anchor_base_text(factor_text: &str) -> Option<String> {
    let inner = strip_wrapping_parens(factor_text);
    let factors = split_top_level_mul_factors_text(inner);
    if factors.len() != 3 {
        return None;
    }

    let mut base = None::<&str>;
    let mut constants = Vec::with_capacity(3);
    for factor in factors {
        let factor = strip_wrapping_parens(factor);
        let plus_idx = factor.rfind('+')?;
        let factor_base = &factor[..plus_idx];
        let constant = &factor[plus_idx + 1..];
        if !is_simple_ident_text(factor_base) {
            return None;
        }
        let constant = constant.parse::<i64>().ok()?;
        if let Some(expected_base) = base {
            if expected_base != factor_base {
                return None;
            }
        } else {
            base = Some(factor_base);
        }
        constants.push(constant);
    }

    constants.sort_unstable();
    (constants == [1, 2, 3]).then(|| base.unwrap().to_string())
}

fn extract_three_linear_shift_expanded_base_text(factor_text: &str) -> Option<String> {
    let inner = strip_wrapping_parens(factor_text);
    let marker = "^3+6*";
    let idx = inner.find(marker)?;
    let base = &inner[..idx];
    if !is_simple_ident_text(base) {
        return None;
    }
    let expected = format!("{base}^3+6*{base}^2+11*{base}+6");
    (inner == expected).then(|| base.to_string())
}

pub(super) fn extract_three_linear_shift_anchor_and_partner_text(
    side: &str,
) -> Option<(String, String)> {
    let factors = split_top_level_mul_factors_text(strip_wrapping_parens(side));
    if factors.len() != 2 {
        return None;
    }

    for anchor_index in 0..2 {
        let partner_index = 1 - anchor_index;
        let Some(base) = extract_three_linear_shift_anchor_base_text(factors[anchor_index]) else {
            continue;
        };
        return Some((
            base,
            strip_wrapping_parens(factors[partner_index]).to_string(),
        ));
    }

    None
}

pub(super) fn extract_three_linear_shift_expanded_and_partner_text(
    side: &str,
) -> Option<(String, String)> {
    let factors = split_top_level_mul_factors_text(strip_wrapping_parens(side));
    if factors.len() != 2 {
        return None;
    }

    for cubic_index in 0..2 {
        let partner_index = 1 - cubic_index;
        let Some(base) = extract_three_linear_shift_expanded_base_text(factors[cubic_index]) else {
            continue;
        };
        return Some((
            base,
            strip_wrapping_parens(factors[partner_index]).to_string(),
        ));
    }

    None
}

pub(super) fn matches_double_angle_arcsin_partner_text(
    factored_partner: &str,
    expanded_partner: &str,
) -> bool {
    let factored_partner = strip_wrapping_parens(factored_partner);
    let Some(inner) = factored_partner
        .strip_prefix("sin(2*arcsin(")
        .and_then(|s| s.strip_suffix("))"))
    else {
        return false;
    };
    let expected = format!("2*{inner}*sqrt(1-{inner}^2)");
    strip_wrapping_parens(expanded_partner) == strip_wrapping_parens(&expected)
}

pub(super) fn matches_small_radical_product_partner_text(
    factored_partner: &str,
    expanded_partner: &str,
) -> bool {
    let factors = split_top_level_mul_factors_text(strip_wrapping_parens(factored_partner));
    if factors.len() != 2 {
        return false;
    }

    for lhs_index in 0..2 {
        let rhs_index = 1 - lhs_index;
        let Some(inner) = strip_wrapping_parens(factors[lhs_index])
            .strip_prefix("sqrt(")
            .and_then(|s| s.strip_suffix(')'))
        else {
            continue;
        };
        let expected_other = format!("sqrt(4*{inner})");
        if strip_wrapping_parens(factors[rhs_index]) != strip_wrapping_parens(&expected_other) {
            continue;
        }

        let expected = format!("2*{inner}");
        if strip_wrapping_parens(expanded_partner) == strip_wrapping_parens(&expected) {
            return true;
        }
    }

    false
}

fn looks_like_square_of_simple_ident_text(text: &str) -> bool {
    let text = strip_wrapping_parens(text);
    let Some(base) = text.strip_suffix("^2") else {
        return false;
    };
    is_simple_ident_text(strip_wrapping_parens(base))
}

fn looks_like_sum_of_two_squares_text(text: &str) -> bool {
    let terms = split_top_level_add_terms_text(strip_wrapping_parens(text));
    terms.len() == 2
        && terms
            .iter()
            .all(|term| looks_like_square_of_simple_ident_text(term))
}

fn looks_like_sum_of_squares_product_anchor_text(text: &str) -> bool {
    let factors = split_top_level_mul_factors_text(strip_wrapping_parens(text));
    factors.len() == 2
        && factors
            .iter()
            .all(|factor| looks_like_sum_of_two_squares_text(factor))
}

pub(super) fn extract_sum_of_squares_anchor_and_partner_text(
    side: &str,
) -> Option<(String, String)> {
    let factors = split_top_level_mul_factors_text(strip_wrapping_parens(side));
    if factors.len() != 2 {
        return None;
    }

    for anchor_index in 0..2 {
        let partner_index = 1 - anchor_index;
        if !looks_like_sum_of_squares_product_anchor_text(factors[anchor_index]) {
            continue;
        }
        return Some((
            strip_wrapping_parens(factors[anchor_index]).to_string(),
            strip_wrapping_parens(factors[partner_index]).to_string(),
        ));
    }

    None
}

pub(super) fn safe_window_parametrized_pair_texts(
    lhs_text: &str,
    rhs_text: &str,
) -> Option<(String, String)> {
    let lhs = normalize_metamorphic_text(lhs_text);
    let rhs = normalize_metamorphic_text(rhs_text);
    let pair_matches = |a: &str, b: &str| (lhs == a && rhs == b) || (lhs == b && rhs == a);

    let replacements: Vec<(&str, &str)> = if pair_matches("ln((-u)^2)", "2*ln((-u))") {
        vec![("u", "-exp(safe_t)")]
    } else if pair_matches("ln((2*u)^2)", "2*ln((2*u))") {
        vec![("u", "exp(safe_t)/2")]
    } else if pair_matches("ln((1-u)^2)", "2*ln((1-u))") {
        vec![("u", "1-exp(safe_t)")]
    } else if pair_matches(
        "((exp(x)-exp(-x))/2)*(sin(2*arcsin(u)))",
        "(sinh(x))*(2*u*sqrt(1-u^2))",
    ) || pair_matches(
        "(tanh(x))*(sin(2*arcsin(u)))",
        "((exp(x)-exp(-x))/(exp(x)+exp(-x)))*(2*u*sqrt(1-u^2))",
    ) {
        vec![("u", "sin(safe_theta)")]
    } else if pair_matches(
        "(sin(2*arcsin(x)))*(abs(sin(u/2)))",
        "(2*x*sqrt(1-x^2))*(sqrt((1-cos(u))/2))",
    ) {
        vec![("x", "sin(safe_theta)"), ("u", "2*safe_phi")]
    } else if pair_matches(
        "(cos(3*pi/8))*(sqrt(u)*sqrt(4*u))",
        "(sqrt(2-sqrt(2))/2)*(2*u)",
    ) || pair_matches(
        "(sin(2*arcsin(x)))*(sqrt(u)*sqrt(4*u))",
        "(2*x*sqrt(1-x^2))*(2*u)",
    ) {
        vec![("u", "exp(safe_t)")]
    } else {
        return None;
    };

    let mut lhs_param = lhs_text.to_string();
    let mut rhs_param = rhs_text.to_string();
    for (var, replacement) in replacements {
        lhs_param = text_substitute(&lhs_param, var, replacement);
        rhs_param = text_substitute(&rhs_param, var, replacement);
    }

    Some((lhs_param, rhs_param))
}

#[test]
fn metamorphic_texts_use_simplified_variants_for_curated_pairs() {
    let lhs = "(1/(u - 1) + 1/(u + 1)) + ((u+1)*(u+1))";
    let rhs = "(2*u/(u^2 - 1)) + (u^2 + 2*u + 1)";

    let mut simplifier = Simplifier::with_default_rules();
    let lhs_expr = parse(lhs, &mut simplifier.context).expect("lhs parses");
    let rhs_expr = parse(rhs, &mut simplifier.context).expect("rhs parses");
    let (lhs_simp_raw, _) = simplifier.simplify(lhs_expr);
    let lhs_simp = fold_constants_safe(&mut simplifier.context, lhs_simp_raw);
    let (rhs_simp_raw, _) = simplifier.simplify(rhs_expr);
    let rhs_simp = fold_constants_safe(&mut simplifier.context, rhs_simp_raw);

    assert!(prove_zero_from_metamorphic_texts(
        &mut simplifier,
        lhs,
        rhs,
        lhs_simp,
        rhs_simp
    ));
}

#[test]
fn metamorphic_texts_use_power_merged_variants_for_curated_pairs() {
    let lhs = "(1/u + 1/(u+1)) + ((u-1)^2*(u-1)^3)";
    let rhs = "((2*u+1)/(u*(u+1))) + (u^5 - 5*u^4 + 10*u^3 - 10*u^2 + 5*u - 1)";

    let mut simplifier = Simplifier::with_default_rules();
    let lhs_expr = parse(lhs, &mut simplifier.context).expect("lhs parses");
    let rhs_expr = parse(rhs, &mut simplifier.context).expect("rhs parses");
    let (lhs_simp_raw, _) = simplifier.simplify(lhs_expr);
    let lhs_simp = fold_constants_safe(&mut simplifier.context, lhs_simp_raw);
    let (rhs_simp_raw, _) = simplifier.simplify(rhs_expr);
    let rhs_simp = fold_constants_safe(&mut simplifier.context, rhs_simp_raw);

    assert!(prove_zero_from_metamorphic_texts(
        &mut simplifier,
        lhs,
        rhs,
        lhs_simp,
        rhs_simp
    ));
}

#[test]
fn raw_pressure_proof_can_use_original_engine_texts_for_curated_pair() {
    let lhs = "sec((1/(u - 1) + 1/(u + 1)))^2 - tan((1/(u - 1) + 1/(u + 1)))^2";
    let rhs = "1";

    assert!(prove_zero_from_curated_pair_corpus_text(lhs, rhs));
    assert!(prove_zero_from_engine_texts(lhs, rhs));

    let mut simplifier = Simplifier::with_default_rules();
    let lhs_expr = parse(lhs, &mut simplifier.context).expect("lhs parses");
    let rhs_expr = parse(rhs, &mut simplifier.context).expect("rhs parses");
    let (lhs_simp_raw, _) = simplifier.simplify(lhs_expr);
    let lhs_simp = fold_constants_safe(&mut simplifier.context, lhs_simp_raw);
    let (rhs_simp_raw, _) = simplifier.simplify(rhs_expr);
    let rhs_simp = fold_constants_safe(&mut simplifier.context, rhs_simp_raw);

    assert!(prove_zero_from_metamorphic_texts_with_flavor(
        &mut simplifier,
        lhs,
        rhs,
        lhs_simp,
        rhs_simp,
        MetamorphicProofFlavor::RawPressure
    ));
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI keeps engine and torture coverage for this raw-pressure polynomial identity smoke"
)]
fn raw_pressure_proof_can_use_original_engine_texts() {
    let lhs = "((u/(u + 1))+1)^4";
    let rhs = "(u/(u + 1))^4 + 4*(u/(u + 1))^3 + 6*(u/(u + 1))^2 + 4*(u/(u + 1)) + 1";

    assert!(!prove_zero_from_curated_pair_corpus_text(lhs, rhs));
    assert!(prove_zero_from_engine_texts(lhs, rhs));

    let mut simplifier = Simplifier::with_default_rules();
    let lhs_expr = parse(lhs, &mut simplifier.context).expect("lhs parses");
    let rhs_expr = parse(rhs, &mut simplifier.context).expect("rhs parses");
    let (lhs_simp_raw, _) = simplifier.simplify(lhs_expr);
    let lhs_simp = fold_constants_safe(&mut simplifier.context, lhs_simp_raw);
    let (rhs_simp_raw, _) = simplifier.simplify(rhs_expr);
    let rhs_simp = fold_constants_safe(&mut simplifier.context, rhs_simp_raw);

    assert!(prove_zero_from_metamorphic_texts_with_flavor(
        &mut simplifier,
        lhs,
        rhs,
        lhs_simp,
        rhs_simp,
        MetamorphicProofFlavor::RawPressure
    ));
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI keeps direct engine regressions for this special-angle double-angle child-process smoke"
)]
fn raw_pressure_child_process_can_use_engine_direct_pair_texts_for_special_angle_double_angle_pair()
{
    let lhs = "((cot(5*pi/12)) * (sin(2*x)))";
    let rhs = "(((2 - 3^(1/2))) * (2*sin(x)*cos(x)))";

    assert!(prove_zero_from_engine_texts_in_child_process(lhs, rhs));
    assert!(prove_zero_from_engine_texts_in_child_process(rhs, lhs));
}

pub(super) fn parse_text_probe(text: &str) -> Result<(), String> {
    let mut simplifier = Simplifier::with_default_rules();
    parse(text, &mut simplifier.context)
        .map(|_| ())
        .map_err(|err| format!("{err:?}"))
}
