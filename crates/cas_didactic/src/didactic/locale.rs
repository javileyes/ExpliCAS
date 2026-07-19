//! Localization for didactic sub-step titles (and other keyed strings).
//!
//! The step-by-step is generated in Spanish (the source language). To support other languages
//! without threading a `Language` into every generator, a generator may emit a sub-step with an
//! i18n KEY plus positional ARGS (`SubStep::keyed`); the wire layer then renders the title for the
//! requested language via [`translate`]. The key table below is the single source of truth for every
//! language, so adding a third language is just another arm.
//!
//! Args are already-rendered, language-neutral fragments (math, numbers) substituted into the
//! template's `{0}`, `{1}`, … placeholders. A key with no template arm falls back to its Spanish
//! template (or, failing that, the raw key) so a partially-migrated table never breaks output.

use cas_solver_core::eval_option_axes::Language;

/// Render the i18n `key` for `lang`, substituting `args` into the template placeholders.
pub(crate) fn translate(key: &str, args: &[&str], lang: Language) -> String {
    let template = template_for(key, lang)
        .or_else(|| template_for(key, Language::Es))
        .unwrap_or(key);
    fill(template, args)
}

/// Substitute `{0}`, `{1}`, … in `template` with `args` (out-of-range placeholders are left as-is).
fn fill(template: &str, args: &[&str]) -> String {
    if args.is_empty() || !template.contains('{') {
        return template.to_string();
    }
    let mut out = String::with_capacity(template.len());
    let mut chars = template.char_indices().peekable();
    while let Some((i, ch)) = chars.next() {
        if ch == '{' {
            // Read the digits until '}'.
            let rest = &template[i + 1..];
            if let Some(end) = rest.find('}') {
                if let Ok(idx) = rest[..end].parse::<usize>() {
                    if let Some(arg) = args.get(idx) {
                        out.push_str(arg);
                    } else {
                        out.push('{');
                        out.push_str(&rest[..end]);
                        out.push('}');
                    }
                    // Advance past the consumed `{..}`.
                    for _ in 0..(end + 1) {
                        chars.next();
                    }
                    continue;
                }
            }
            out.push(ch);
        } else {
            out.push(ch);
        }
    }
    out
}

/// The localized template for `key`, or `None` if the key/language pair is not in the table.
fn template_for(key: &str, lang: Language) -> Option<&'static str> {
    let (es, en) = match key {
        // Integration by parts.
        "by_parts.repeated" => (
            "Usar integración por partes repetida",
            "Use repeated integration by parts",
        ),
        "by_parts.use" => ("Usar integración por partes", "Use integration by parts"),
        "by_parts.choose_u_dv" => ("Elegir u y dv", "Choose u and dv"),
        "by_parts.compute_du_v" => ("Calcular du y v", "Compute du and v"),
        "by_parts.apply_formula" => (
            "Aplicar la fórmula de integración por partes",
            "Apply the integration-by-parts formula",
        ),
        "by_parts.integrate_remaining" => (
            "Integrar el término restante",
            "Integrate the remaining term",
        ),
        "partial_fractions.decompose" => ("Descomponer en fracciones parciales", "Decompose into partial fractions"),
        "integral.integrate_simple_terms" => ("Integrar los términos simples", "Integrate the simple terms"),
        "usub.identify_u_du" => ("Identificar u y du", "Identify u and du"),
        "usub.adjust_constant_factor" => ("Ajustar el factor constante", "Adjust the constant factor"),
        "integral.find_antiderivative" => ("Hallar la antiderivada", "Find the antiderivative"),
        "integral.evaluate_antiderivative_at_bounds" => ("Evaluar la antiderivada en los límites", "Evaluate the antiderivative at the bounds"),
        "polynomial.identify_distributive_products" => ("Identificar los productos que genera la distributiva", "Identify the products produced by the distributive law"),
        "polynomial.write_products_with_original_signs" => ("Escribir los productos con los signos originales", "Write the products with their original signs"),
        "usub.use_substitution" => ("Usar sustitución", "Use substitution"),
        "usub.identify_affine_argument" => ("Identificar el argumento afín", "Identify the affine argument"),
        "usub.identify_affine_denominator" => ("Identificar el denominador afín", "Identify the affine denominator"),
        "integral.use_linearity" => ("Usar linealidad de la integral", "Use linearity of the integral"),
        "integral.integrate_each_term" => ("Integrar cada término", "Integrate each term"),
        "integral.reduce_positive_quadratic_to_square" => ("Reducir el cuadrático positivo al cuadrado", "Reduce the positive quadratic to a square"),
        "integral.integrate_arctan_and_rational_parts" => ("Integrar la parte arctan y la parte racional", "Integrate the arctan part and the rational part"),
        "usub.rule_cos_to_sin" => ("Usar la regla de cos(u) -> sin(u)", "Use the rule cos(u) -> sin(u)"),
        "usub.rule_exp_inner_derivative" => ("Usar la regla de exp con derivada interna", "Use the exp rule with inner derivative"),
        "usub.rule_ln_abs_inner_derivative" => ("Usar la regla de ln|u| con derivada interna", "Use the ln|u| rule with inner derivative"),
        "usub.rule_exp_to_exp" => ("Usar la regla de exp(u) -> exp(u)", "Use the rule exp(u) -> exp(u)"),
        "usub.rule_arctan_inner_derivative" => ("Usar la regla de arctan con derivada interna", "Use the arctan rule with inner derivative"),
        "usub.rule_tan_to_neg_ln_abs_cos" => ("Usar la regla de tan(u) -> -ln|cos(u)|", "Use the rule tan(u) -> -ln|cos(u)|"),
        "usub.rule_sin_inner_derivative" => ("Usar la regla de sin con derivada interna", "Use the sin rule with inner derivative"),
        "usub.rule_cos_inner_derivative" => ("Usar la regla de cos con derivada interna", "Use the cos rule with inner derivative"),
        "usub.rule_sec_squared_to_tan" => ("Usar la regla de 1/cos(u)^2 -> tan(u)", "Use the rule 1/cos(u)^2 -> tan(u)"),
        "usub.rule_udu_over_u_to_ln_abs" => ("Usar la regla de u'/u -> ln|u|", "Use the rule u'/u -> ln|u|"),
        "usub.rule_arcsin_inner_derivative" => ("Usar la regla de arcsin con derivada interna", "Use the arcsin rule with inner derivative"),
        "usub.rule_cot_to_ln_abs_sin" => ("Usar la regla de cot(u) -> ln|sin(u)|", "Use the rule cot(u) -> ln|sin(u)|"),
        "usub.rule_power_to_power_plus_one" => ("Usar la regla de u'·u^p -> u^(p+1)/(p+1)", "Use the rule u'·u^p -> u^(p+1)/(p+1)"),
        "limit.direct_substitution_0_0" => ("La sustitución directa da la indeterminación 0/0", "Direct substitution gives the indeterminate form 0/0"),
        "limit.numerator_denominator_inf_over_inf" => ("Numerador y denominador → ∞: indeterminación ∞/∞", "Numerator and denominator → ∞: indeterminate form ∞/∞"),
        "limit.base_to_1_exponent_to_inf_1_pow_inf" => ("La base tiende a 1 y el exponente a ∞: indeterminación 1^∞", "The base tends to 1 and the exponent to ∞: indeterminate form 1^∞"),
        "limit.notable_sin_u_over_u" => ("Aplicar el límite notable: lím(u→0) sin(u)/u = 1", "Apply the standard limit: lim(u→0) sin(u)/u = 1"),
        "limit.notable_one_minus_cos_over_u2" => ("Aplicar el límite notable: lím(u→0) (1 − cos(u))/u² = 1/2", "Apply the standard limit: lim(u→0) (1 − cos(u))/u² = 1/2"),
        "limit.notable_a_pow_u_minus_1_over_u" => ("Aplicar el límite notable: lím(u→0) (aᵘ − 1)/u = ln(a)", "Apply the standard limit: lim(u→0) (aᵘ − 1)/u = ln(a)"),
        "limit.notable_ln_1_plus_u_over_u" => ("Aplicar el límite notable: lím(u→0) ln(1+u)/u = 1", "Apply the standard limit: lim(u→0) ln(1+u)/u = 1"),
        "limit.notable_exp_u_minus_1_over_u" => ("Aplicar el límite notable: lím(u→0) (e^u − 1)/u = 1", "Apply the standard limit: lim(u→0) (e^u − 1)/u = 1"),
        "limit.notable_tan_u_over_u" => ("Aplicar el límite notable: lím(u→0) tan(u)/u = 1", "Apply the standard limit: lim(u→0) tan(u)/u = 1"),
        "limit.notable_u_over_sin_u" => ("Aplicar el límite notable: lím(u→0) u/sin(u) = 1", "Apply the standard limit: lim(u→0) u/sin(u) = 1"),
        "limit.notable_one_plus_u_pow_recip_eq_e" => ("Aplicar el límite notable: lím(u→0) (1 + u)^(1/u) = e", "Apply the standard limit: lim(u→0) (1 + u)^(1/u) = e"),
        "limit.notable_one_plus_recip_pow_x_eq_e" => ("Aplicar el límite notable: lím(x→∞) (1 + 1/x)^x = e", "Apply the standard limit: lim(x→∞) (1 + 1/x)^x = e"),
        "limit.notable_first_order_scaled_over_u" => ("Aplicar el límite notable: lím(u→0) {0}({1}·u)/u = {1}", "Apply the standard limit: lim(u→0) {0}({1}·u)/u = {1}"),
        "limit.notable_first_order_cross_ratio" => ("Aplicar el límite notable: lím(u→0) {0}/{1} = {2}", "Apply the standard limit: lim(u→0) {0}/{1} = {2}"),
        "limit.notable_binomial_first_order" => ("Aplicar el límite notable: lím(u→0) ((1+u)^({0}) − 1)/u = {0}  (equivalente de primer orden (1+u)^a ~ 1 + a·u)", "Apply the standard limit: lim(u→0) ((1+u)^({0}) − 1)/u = {0}  (first-order equivalent (1+u)^a ~ 1 + a·u)"),
        "limit.lhopital_first_iteration" => ("Indeterminación 0/0 en {0} = {1}: aplica L'Hôpital (deriva numerador y denominador)", "Indeterminate form 0/0 at {0} = {1}: apply L'Hôpital (differentiate numerator and denominator)"),
        "limit.lhopital_still_0_0_again" => ("Sigue siendo 0/0: aplica L'Hôpital otra vez", "Still 0/0: apply L'Hôpital again"),
        "limit.lhopital_denominator_nonzero_substitute" => ("El denominador ya no se anula; sustituye {0} = {1}", "The denominator no longer vanishes; substitute {0} = {1}"),
        "limit.generic_0_0_lhopital_or_taylor" => ("Indeterminación 0/0 en {0}={1}: aplica la regla de L'Hôpital (deriva numerador y denominador) o el desarrollo de Taylor", "Indeterminate form 0/0 at {0}={1}: apply L'Hôpital's rule (differentiate numerator and denominator) or the Taylor expansion"),
        "limit.factor_numerator_denominator" => ("Factoriza numerador y denominador", "Factor the numerator and denominator"),
        "limit.cancel_common_factor" => ("Cancela el factor común ({0})", "Cancel the common factor ({0})"),
        "collect.add_literal_coefficients" => ("Sumar los coeficientes que acompañan a {0}", "Add the coefficients of {0}"),
        "nested.rewrite_denominator_common_factor" => ("Reescribir el denominador sacando factor común {0}", "Rewrite the denominator by factoring out {0}"),
        "nested.rewrite_numerator_common_factor" => ("Reescribir el numerador sacando factor común {0}", "Rewrite the numerator by factoring out {0}"),
        "limit.substitute_in_simplified" => ("Sustituye {0} = {1} en la expresión simplificada", "Substitute {0} = {1} into the simplified expression"),
        "limit.factor_cancel_before_evaluate" => ("Factorizar numerador y denominador y cancelar el factor común antes de evaluar", "Factor the numerator and denominator and cancel the common factor before evaluating"),
        "limit.direct_substitution_polynomial_continuous_at" => ("Sustitución directa: el polinomio es continuo, así que el límite es su valor en {0} = {1}", "Direct substitution: the polynomial is continuous, so the limit is its value at {0} = {1}"),
        "limit.direct_substitution_polynomial_continuity" => ("Sustitución directa: el límite de un polinomio es su valor en el punto (continuidad)", "Direct substitution: the limit of a polynomial is its value at the point (continuity)"),
        "limit.squeeze_bound_oscillator" => ("Acota el factor oscilante: |{0}| ≤ 1, luego |{1}| ≤ {2}", "Bound the oscillating factor: |{0}| ≤ 1, so |{1}| ≤ {2}"),
        "limit.squeeze_infinitesimal_conclusion" => ("El infinitésimo {0} → 0, así que por el teorema del sándwich el límite es 0", "The infinitesimal {0} → 0, so by the squeeze theorem the limit is 0"),
        "limit.squeeze_theorem_bounded_times_infinitesimal" => ("Aplicar el teorema del sándwich: factor acotado × infinitésimo → 0", "Apply the squeeze theorem: bounded factor × infinitesimal → 0"),
        "limit.dominance_slower_quotient_to_0" => ("Dominancia: {0} crece más despacio que {1} (jerarquía ln ≪ potencia ≪ exp), así que el cociente → 0", "Dominance: {0} grows more slowly than {1} (hierarchy ln ≪ power ≪ exp), so the quotient → 0"),
        "limit.dominance_faster_quotient_to_inf" => ("Dominancia: {0} crece más rápido que {1} (jerarquía ln ≪ potencia ≪ exp), así que el cociente → ±∞", "Dominance: {0} grows faster than {1} (hierarchy ln ≪ power ≪ exp), so the quotient → ±∞"),
        "limit.dominance_denominator_higher_degree_to_0" => ("Dominancia: el denominador tiene mayor grado, así que el cociente → 0", "Dominance: the denominator has higher degree, so the quotient → 0"),
        "limit.dominance_equal_degree_leading_coeff_ratio" => ("Dominancia: grados iguales, el límite es el cociente de los coeficientes líderes", "Dominance: equal degrees, the limit is the ratio of the leading coefficients"),
        "limit.dominance_numerator_higher_degree_to_inf" => ("Dominancia: el numerador tiene mayor grado, así que el cociente → ±∞", "Dominance: the numerator has higher degree, so the quotient → ±∞"),
        "limit.dominance_polynomial_diverges" => ("Dominancia: un polinomio de grado ≥ 1 tiende a ±∞", "Dominance: a polynomial of degree ≥ 1 tends to ±∞"),
        "limit.dominance_exponential_decay_beats_power_product_to_0" => ("Dominancia: la exponencial decae más rápido de lo que crece la potencia, así que el producto → 0", "Dominance: the exponential decays faster than the power grows, so the product → 0"),
        "limit.residual_one_sided_method" => ("La política segura no decide este límite. Para investigarlo, calcula los límites laterales en {0} = {1} (por la izquierda y por la derecha): si coinciden, ese es el valor del límite; si difieren, el límite no existe", "The safe policy does not decide this limit. To investigate it, compute the one-sided limits at {0} = {1} (from the left and from the right): if they agree, that is the value of the limit; if they differ, the limit does not exist"),
        "limit.inf_minus_inf_indeterminate" => ("La sustitución directa da la indeterminación ∞−∞", "Direct substitution gives the indeterminate form ∞−∞"),
        "limit.combine_over_common_denominator" => ("Combina las fracciones sobre un común denominador", "Combine the fractions over a common denominator"),
        "limit.multiply_divide_by_conjugate" => ("Multiplica y divide por el conjugado ({0}) para racionalizar", "Multiply and divide by the conjugate ({0}) to rationalize"),
        "limit.divide_by_dominant_power_evaluate" => ("Divide numerador y denominador entre {1} (la potencia dominante): {0}/{1} → 1, así que el límite es {2}", "Divide numerator and denominator by {1} (the dominant power): {0}/{1} → 1, so the limit is {2}"),
        "radical.denest_identify_form" => ("Identificar la forma √(a ± c·√d)", "Identify the form √(a ± c·√d)"),
        "radical.denest_compute_delta" => ("Calcular Δ = a² - c²d", "Compute Δ = a² - c²d"),
        "radical.denest_apply" => ("Δ es cuadrado perfecto: aplicar desanidación", "Δ is a perfect square: apply denesting"),
        "rationalize.form_conjugate" => ("Cambiar el signo para formar el conjugado", "Change the sign to form the conjugate"),
        "rationalize.multiply_by_conjugate_both" => ("Multiplicar numerador y denominador por ese conjugado", "Multiply numerator and denominator by that conjugate"),
        "rationalize.denominator_difference_of_squares" => ("En el denominador aparece una diferencia de cuadrados", "The denominator becomes a difference of squares"),
        "rationalize.multiply_by_conjugate" => ("Multiplicar por el conjugado", "Multiply by the conjugate"),
        "rationalize.difference_of_squares" => ("Diferencia de cuadrados", "Difference of squares"),
        "rationalize.group_denominator_terms" => ("Agrupar términos del denominador", "Group the terms of the denominator"),
        "rationalize.denominator_radical_product" => ("Denominador con producto de radical", "Denominator with a radical factor"),
        "rationalize.multiply_by_root_over_root" => ("Multiplicar por \\sqrt{n}/\\sqrt{n}", "Multiply by \\sqrt{n}/\\sqrt{n}"),
        "rationalize.multiply_by_cube_conjugate" => ("Multiplicar por el conjugado cúbico", "Multiply by the cubic conjugate"),
        "rationalize.sum_of_cubes_denominator" => ("Aplicar suma de cubos en el denominador", "Apply the sum of cubes in the denominator"),
        "rationalize.cube_exact_quotient_identity" => ("Usar (u^3 - 1) / (u - 1) = u^2 + u + 1", "Use (u^3 - 1) / (u - 1) = u^2 + u + 1"),
        "rationalize.factor_numerator_sum_of_cubes" => ("Factorizar el numerador como suma de cubos", "Factor the numerator as a sum of cubes"),
        "rationalize.factor_numerator_difference_of_cubes" => ("Factorizar el numerador como diferencia de cubos", "Factor the numerator as a difference of cubes"),
        "rationalize.numerator_equals_denominator_quotient_one" => ("Numerador y denominador quedan iguales, así que el cociente vale 1", "Numerator and denominator are equal, so the quotient is 1"),
        "polynomial.simplify_nested_fraction" => ("Simplificar fracción anidada", "Simplify nested fraction"),
        "generic.simplify_expression" => ("Simplificar expresión", "Simplify expression"),
        "polynomial.invert_denominator_fraction" => ("Invertir la fracción del denominador", "Invert the fraction in the denominator"),
        "polynomial.simplify_resulting_product" => ("Simplificar el producto resultante", "Simplify the resulting product"),
        "polynomial.common_denominator_within_denominator" => ("Llevar a denominador común dentro del denominador", "Put over a common denominator inside the denominator"),
        "polynomial.divide_by_fraction_is_multiply_inverse" => ("Dividir entre una fracción es multiplicar por su inversa", "Dividing by a fraction is multiplying by its reciprocal"),
        "polynomial.define_cube_bases" => ("Definimos las bases de los cubos", "Define the bases of the cubes"),
        "polynomial.verify_sum_zero" => ("Verificamos que x + y + z = 0", "Verify that x + y + z = 0"),
        "polynomial.apply_sum_of_cubes_identity" => ("Aplicamos la identidad: si x+y+z=0, entonces x³+y³+z³=3xyz", "Apply the identity: if x+y+z=0, then x³+y³+z³=3xyz"),
        "polynomial.substitution_to_simplify" => ("Sustitución para simplificar", "Substitution to simplify"),
        "polynomial.substituted_expression" => ("Expresión sustituida", "Substituted expression"),
        "polynomial.all_terms_cancel" => ("Todos los términos se cancelan", "All terms cancel"),
        "polynomial.to_normal_form" => ("Convertir a forma normal polinómica", "Convert to polynomial normal form"),
        "polynomial.cancel_like_terms" => ("Cancelar términos semejantes", "Cancel like terms"),
        "polynomial.expand_left_side" => ("Expandir lado izquierdo", "Expand the left-hand side"),
        "polynomial.expand_right_side" => ("Expandir lado derecho", "Expand the right-hand side"),
        "polynomial.compare_normal_forms" => ("Comparar formas normales", "Compare normal forms"),
        "polynomial.factor_numerator_sum_or_difference_of_cubes" => ("Factorizar el numerador como suma o diferencia de cubos", "Factor the numerator as a sum or difference of cubes"),
        "polynomial.cancel_factor" => ("Ahora se cancela el factor {0}", "Now the factor {0} cancels"),
        "polynomial.replace_block_in_expression" => ("Reemplazar ese bloque en la expresión", "Replace that block in the expression"),
        "polynomial.cancel_exact_opposite_terms" => ("Cancelar términos opuestos exactos", "Cancel exact opposite terms"),
        "polynomial.factor_appears_when_equal" => ("Si {0} = {1}, aparece el factor {0} - {1}", "If {0} = {1}, the factor {0} - {1} appears"),
        "polynomial.remaining_quotient" => ("El cociente restante es {0}", "The remaining quotient is {0}"),
        "polynomial.check_factor_vanishing" => ("Comprobar anulación de factores", "Check that the factors vanish"),
        "series.telescope_partial_fraction_unit_gap" => ("Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)", "Use 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)"),
        "series.telescope_partial_fraction_general_gap" => ("Usar 1 / (u · (u + g)) = 1 / g · (1 / u - 1 / (u + g))", "Use 1 / (u · (u + g)) = 1 / g · (1 / u - 1 / (u + g))"),
        "series.telescoping_sum_cancels_middle_terms" => ("La suma telescópica cancela los términos intermedios", "The telescoping sum cancels the intermediate terms"),
        "series.telescope_product_square_difference" => ("Usar (u^2 - 1) / u^2 = ((u - 1) · (u + 1)) / u^2", "Use (u^2 - 1) / u^2 = ((u - 1) · (u + 1)) / u^2"),
        "series.product_factors_cancel_telescopically" => ("Los factores (u + 1) y (u - 1) se cancelan telescópicamente", "The (u + 1) and (u - 1) factors cancel telescopically"),
        "series.product_only_first_and_last_factor_remain" => ("Solo quedan el primer factor u - 1 y el último factor u + 1", "Only the first factor u - 1 and the last factor u + 1 remain"),
        "series.write_first_and_last_product_factors" => ("Escribir los primeros y últimos factores del producto", "Write the first and last factors of the product"),
        "series.intermediate_factors_cancel_in_pairs" => ("Los factores intermedios se cancelan por parejas", "The intermediate factors cancel in pairs"),
        "series.only_last_numerator_and_first_denominator_remain" => ("Solo quedan el último numerador y el primer denominador", "Only the last numerator and the first denominator remain"),
        "series.write_sum_with_endpoints" => ("Escribir la suma con sus extremos", "Write the sum with its endpoints"),
        "series.write_product_with_endpoints" => ("Escribir el producto con sus extremos", "Write the product with its endpoints"),
        "series.closed_form_sum_of_integers" => ("Usar la fórmula cerrada para la suma de enteros", "Use the closed-form formula for the sum of integers"),
        "series.closed_form_sum_of_squares" => ("Usar la fórmula cerrada para la suma de cuadrados", "Use the closed-form formula for the sum of squares"),
        "series.closed_form_sum_of_cubes" => ("Usar la fórmula cerrada para la suma de cubos", "Use the closed-form formula for the sum of cubes"),
        "series.closed_form_geometric_sum" => ("Usar la fórmula cerrada para la suma geométrica", "Use the closed-form formula for the geometric sum"),
        "series.count_equal_terms_in_sum" => ("Contar términos iguales en la suma", "Count the equal terms in the sum"),
        "series.product_of_consecutive_integers_as_factorial" => ("Usar factorial para el producto de enteros consecutivos", "Use a factorial for the product of consecutive integers"),
        "series.product_of_powers_as_power_of_factorials" => ("Convertir el producto de potencias en potencia de factoriales", "Convert the product of powers into a power of factorials"),
        "series.count_equal_factors_in_product" => ("Contar factores iguales en el producto", "Count the equal factors in the product"),
        "fraction.common_denominator" => ("Llevar a denominador común", "Put over a common denominator"),
        "fraction.simplify_numerator_and_denominator" => ("Simplificar el numerador y el denominador", "Simplify the numerator and the denominator"),
        "gradient.component" => ("Derivar respecto de {0}, tratando las demás variables como constantes", "Differentiate with respect to {0}, treating the other variables as constants"),
        "lineintegral.formula_vector" => ("∮F·dr: sustituir la parametrización en F, derivar r(t), ensamblar Σ Fᵢ·rᵢ\u{2032} e integrar en [{0}, {1}]", "∮F·dr: substitute the parametrization into F, differentiate r(t), assemble Σ Fᵢ·rᵢ\u{2032} and integrate over [{0}, {1}]"),
        "lineintegral.formula_scalar" => ("∫f·ds: sustituir la parametrización en f, calcular ‖r\u{2032}(t)‖ e integrar f·‖r\u{2032}‖ en [{0}, {1}]", "∫f·ds: substitute the parametrization into f, compute ‖r\u{2032}(t)‖ and integrate f·‖r\u{2032}‖ over [{0}, {1}]"),
        "surfaceintegral.formula_vector" => ("∫∫F·dS: sustituir la parametrización en F, ensamblar r_u×r_v e integrar F·(r_u×r_v) sobre el dominio de parámetros", "∫∫F·dS: substitute the parametrization into F, assemble r_u×r_v and integrate F·(r_u×r_v) over the parameter domain"),
        "surfaceintegral.formula_scalar" => ("∫∫f·dS: sustituir la parametrización en f, calcular ‖r_u×r_v‖ e integrar f·‖r_u×r_v‖ sobre el dominio de parámetros", "∫∫f·dS: substitute the parametrization into f, compute ‖r_u×r_v‖ and integrate f·‖r_u×r_v‖ over the parameter domain"),
        "taylor.formula" => ("Serie de Taylor de grado total ≤ {0} en {1} variables: Σ_(|α|≤{0}) ∂^α f(a)/α! · (x−a)^α — derivar cada multi-índice, evaluar en el punto y dividir por α!", "Taylor series of total degree ≤ {0} in {1} variables: Σ_(|α|≤{0}) ∂^α f(a)/α! · (x−a)^α — differentiate each multi-index, evaluate at the point and divide by α!"),
        "divergence.formula" => ("∇·F = Σ ∂Fᵢ/∂xᵢ: derivar cada componente respecto de su propia variable y sumar", "∇·F = Σ ∂Fᵢ/∂xᵢ: differentiate each component with respect to its own variable and add"),
        "laplacian.formula" => ("Δf = Σ ∂²f/∂xᵢ²: sumar las segundas derivadas respecto de cada variable", "Δf = Σ ∂²f/∂xᵢ²: add the second derivatives with respect to each variable"),
        "curl.formula3d" => ("∇×F por filas: [∂F₃/∂y − ∂F₂/∂z, ∂F₁/∂z − ∂F₃/∂x, ∂F₂/∂x − ∂F₁/∂y]", "∇×F row by row: [∂F₃/∂y − ∂F₂/∂z, ∂F₁/∂z − ∂F₃/∂x, ∂F₂/∂x − ∂F₁/∂y]"),
        "curl.formula2d" => ("Rotacional 2D (escalar): ∂Q/∂x − ∂P/∂y", "2D curl (scalar): ∂Q/∂x − ∂P/∂y"),
        "jacobian.row" => ("Fila {0}: derivar la componente respecto de cada variable", "Row {0}: differentiate the component with respect to each variable"),
        "hessian.row" => ("Fila {0}: derivar ∂f/∂{1} respecto de cada variable", "Row {0}: differentiate ∂f/∂{1} with respect to each variable"),
        "derivative.use_logarithmic_diff" => ("Usar derivación logarítmica", "Use logarithmic differentiation"),
        "derivative.product_rule" => ("Usar regla del producto", "Use the product rule"),
        "derivative.differentiate_first_factor" => ("Derivar el primer factor", "Differentiate the first factor"),
        "derivative.differentiate_second_factor" => ("Derivar el segundo factor", "Differentiate the second factor"),
        "derivative.linearity" => ("Usar linealidad de la derivada", "Use linearity of the derivative"),
        "derivative.constant_multiple" => ("Usar factor constante de la derivada", "Use the constant multiple rule of the derivative"),
        "derivative.quotient_rule" => ("Usar regla del cociente", "Use the quotient rule"),
        "derivative.differentiate_numerator" => ("Derivar el numerador", "Differentiate the numerator"),
        "derivative.differentiate_denominator" => ("Derivar el denominador", "Differentiate the denominator"),
        "derivative.power_rule" => ("Usar regla de la potencia", "Use the power rule"),
        "derivative.power_rule_with_chain" => ("Usar regla de la potencia con cadena", "Use the power rule with the chain rule"),
        "derivative.exponential_rule" => ("Usar regla exponencial", "Use the exponential rule"),
        "derivative.chain_rule" => ("Usar regla de la cadena", "Use the chain rule"),
        "derivative.rule_sin_u" => ("Usar regla de sin(u)", "Use the sin(u) rule"),
        "derivative.rule_cos_u" => ("Usar regla de cos(u)", "Use the cos(u) rule"),
        "derivative.rule_tan_u" => ("Usar regla de tan(u)", "Use the tan(u) rule"),
        "derivative.rule_ln_u" => ("Usar regla de ln(u)", "Use the ln(u) rule"),
        "derivative.rule_exp_u" => ("Usar regla de exp(u)", "Use the exp(u) rule"),
        "derivative.rule_sqrt_u" => ("Usar regla de sqrt(u)", "Use the sqrt(u) rule"),
        "derivative.rule_arctan_u" => ("Usar regla de arctan(u)", "Use the arctan(u) rule"),
        "derivative.rule_arcsin_u" => ("Usar regla de arcsin(u)", "Use the arcsin(u) rule"),
        "derivative.rule_arccos_u" => ("Usar regla de arccos(u)", "Use the arccos(u) rule"),
        "derivative.rule_sec_u" => ("Usar regla de sec(u)", "Use the sec(u) rule"),
        "derivative.rule_csc_u" => ("Usar regla de csc(u)", "Use the csc(u) rule"),
        "derivative.rule_cot_u" => ("Usar regla de cot(u)", "Use the cot(u) rule"),
        "derivative.rule_sinh_u" => ("Usar regla de sinh(u)", "Use the sinh(u) rule"),
        "derivative.rule_cosh_u" => ("Usar regla de cosh(u)", "Use the cosh(u) rule"),
        "derivative.rule_tanh_u" => ("Usar regla de tanh(u)", "Use the tanh(u) rule"),
        "derivative.rule_sign_u_away_from_zero" => ("Usar derivada de sign(u) fuera de u = 0", "Use the derivative of sign(u) away from u = 0"),
        "factorial.expand_upper_until_lower" => ("Expandir el factorial superior hasta llegar al factorial inferior", "Expand the upper factorial down to the lower factorial"),
        "factorial.write_upper_as_next_times_previous" => ("Escribir el factorial superior como el siguiente número por el factorial anterior", "Write the upper factorial as the next number times the previous factorial"),
        "factorial.cancel_common" => ("Cancelar el factorial común", "Cancel the common factorial"),
        "factorial.binom_as_factorial_quotient" => ("Usar C({0},{1}) = {0}! / ({1}! · {2}!)", "Use C({0},{1}) = {0}! / ({1}! · {2}!)"),
        "factorial.compute_binom_quotient" => ("Calcular {0}! / ({1}! · {2}!) = {3}", "Compute {0}! / ({1}! · {2}!) = {3}"),
        "factorial.pascal_identity" => ("Usar C({0},{1}) + C({0},{2}) = C({3},{4})", "Use C({0},{1}) + C({0},{2}) = C({3},{4})"),
        "factorial.binom_symmetry" => ("Usar C({0},{1}) = C({0},{0}-{1})", "Use C({0},{1}) = C({0},{0}-{1})"),
        "factorial.compute_symmetry_complement" => ("Calcular {0}-{1} = {2}", "Compute {0}-{1} = {2}"),
        _ => return None,
    };
    Some(match lang {
        Language::Es => es,
        Language::En => en,
    })
}

/// English for a STATIC Spanish sub-step description (no embedded values). Unkeyed sub-steps whose
/// title is a fixed Spanish string are translated here at the wire boundary (the same pivot-through-
/// Spanish approach as rule names); dynamic titles use the keyed path. Unmapped strings pass through.
pub(crate) fn description_en(es: &str) -> &str {
    match es {
        "Descomponer en fracciones parciales" => "Decompose into partial fractions",
        "Integrar los términos simples" => "Integrate the simple terms",
        "Identificar u y du" => "Identify u and du",
        "Ajustar el factor constante" => "Adjust the constant factor",
        "Hallar la antiderivada" => "Find the antiderivative",
        "Evaluar la antiderivada en los límites" => "Evaluate the antiderivative at the bounds",
        "Identificar los productos que genera la distributiva" => "Identify the products produced by the distributive law",
        "Escribir los productos con los signos originales" => "Write the products with their original signs",
        "Usar sustitución" => "Use substitution",
        "Identificar el argumento afín" => "Identify the affine argument",
        "Identificar el denominador afín" => "Identify the affine denominator",
        "Usar linealidad de la integral" => "Use linearity of the integral",
        "Integrar cada término" => "Integrate each term",
        "Reducir el cuadrático positivo al cuadrado" => "Reduce the positive quadratic to a square",
        "Integrar la parte arctan y la parte racional" => "Integrate the arctan part and the rational part",
        "Usar la regla de cos(u) -> sin(u)" => "Use the rule cos(u) -> sin(u)",
        "Usar la regla de exp con derivada interna" => "Use the exp rule with inner derivative",
        "Usar la regla de ln|u| con derivada interna" => "Use the ln|u| rule with inner derivative",
        "Usar la regla de exp(u) -> exp(u)" => "Use the rule exp(u) -> exp(u)",
        "Usar la regla de arctan con derivada interna" => "Use the arctan rule with inner derivative",
        "Usar la regla de tan(u) -> -ln|cos(u)|" => "Use the rule tan(u) -> -ln|cos(u)|",
        "Usar la regla de sin con derivada interna" => "Use the sin rule with inner derivative",
        "Usar la regla de cos con derivada interna" => "Use the cos rule with inner derivative",
        "Usar la regla de 1/cos(u)^2 -> tan(u)" => "Use the rule 1/cos(u)^2 -> tan(u)",
        "Usar la regla de u'/u -> ln|u|" => "Use the rule u'/u -> ln|u|",
        "Usar la regla de arcsin con derivada interna" => "Use the arcsin rule with inner derivative",
        "Usar la regla de cot(u) -> ln|sin(u)|" => "Use the rule cot(u) -> ln|sin(u)|",
        "Usar la regla de u'·u^p -> u^(p+1)/(p+1)" => "Use the rule u'·u^p -> u^(p+1)/(p+1)",
        "La sustitución directa da la indeterminación 0/0" => "Direct substitution gives the indeterminate form 0/0",
        "Numerador y denominador → ∞: indeterminación ∞/∞" => "Numerator and denominator → ∞: indeterminate form ∞/∞",
        "La base tiende a 1 y el exponente a ∞: indeterminación 1^∞" => "The base tends to 1 and the exponent to ∞: indeterminate form 1^∞",
        "Aplicar el límite notable: lím(u→0) sin(u)/u = 1" => "Apply the standard limit: lim(u→0) sin(u)/u = 1",
        "Aplicar el límite notable: lím(u→0) (1 − cos(u))/u² = 1/2" => "Apply the standard limit: lim(u→0) (1 − cos(u))/u² = 1/2",
        "Aplicar el límite notable: lím(u→0) (aᵘ − 1)/u = ln(a)" => "Apply the standard limit: lim(u→0) (aᵘ − 1)/u = ln(a)",
        "Aplicar el límite notable: lím(u→0) ln(1+u)/u = 1" => "Apply the standard limit: lim(u→0) ln(1+u)/u = 1",
        "Aplicar el límite notable: lím(u→0) (e^u − 1)/u = 1" => "Apply the standard limit: lim(u→0) (e^u − 1)/u = 1",
        "Aplicar el límite notable: lím(u→0) tan(u)/u = 1" => "Apply the standard limit: lim(u→0) tan(u)/u = 1",
        "Aplicar el límite notable: lím(u→0) u/sin(u) = 1" => "Apply the standard limit: lim(u→0) u/sin(u) = 1",
        "Aplicar el límite notable: lím(u→0) (1 + u)^(1/u) = e" => "Apply the standard limit: lim(u→0) (1 + u)^(1/u) = e",
        "Aplicar el límite notable: lím(x→∞) (1 + 1/x)^x = e" => "Apply the standard limit: lim(x→∞) (1 + 1/x)^x = e",
        "Sigue siendo 0/0: aplica L'Hôpital otra vez" => "Still 0/0: apply L'Hôpital again",
        "Factoriza numerador y denominador" => "Factor the numerator and denominator",
        "Factorizar numerador y denominador y cancelar el factor común antes de evaluar" => "Factor the numerator and denominator and cancel the common factor before evaluating",
        "Sustitución directa: el límite de un polinomio es su valor en el punto (continuidad)" => "Direct substitution: the limit of a polynomial is its value at the point (continuity)",
        "Aplicar el teorema del sándwich: factor acotado × infinitésimo → 0" => "Apply the squeeze theorem: bounded factor × infinitesimal → 0",
        "Dominancia: el denominador tiene mayor grado, así que el cociente → 0" => "Dominance: the denominator has higher degree, so the quotient → 0",
        "Dominancia: grados iguales, el límite es el cociente de los coeficientes líderes" => "Dominance: equal degrees, the limit is the ratio of the leading coefficients",
        "Dominancia: el numerador tiene mayor grado, así que el cociente → ±∞" => "Dominance: the numerator has higher degree, so the quotient → ±∞",
        "Dominancia: un polinomio de grado ≥ 1 tiende a ±∞" => "Dominance: a polynomial of degree ≥ 1 tends to ±∞",
        "Dominancia: la exponencial decae más rápido de lo que crece la potencia, así que el producto → 0" => "Dominance: the exponential decays faster than the power grows, so the product → 0",
        "Identificar la forma √(a ± c·√d)" => "Identify the form √(a ± c·√d)",
        "Calcular Δ = a² - c²d" => "Compute Δ = a² - c²d",
        "Δ es cuadrado perfecto: aplicar desanidación" => "Δ is a perfect square: apply denesting",
        "Cambiar el signo para formar el conjugado" => "Change the sign to form the conjugate",
        "Multiplicar numerador y denominador por ese conjugado" => "Multiply numerator and denominator by that conjugate",
        "En el denominador aparece una diferencia de cuadrados" => "The denominator becomes a difference of squares",
        "Multiplicar por el conjugado" => "Multiply by the conjugate",
        "Diferencia de cuadrados" => "Difference of squares",
        "Agrupar términos del denominador" => "Group the terms of the denominator",
        "Denominador con producto de radical" => "Denominator with a radical factor",
        "Multiplicar por el conjugado cúbico" => "Multiply by the cubic conjugate",
        "Aplicar suma de cubos en el denominador" => "Apply the sum of cubes in the denominator",
        "Usar (u^3 - 1) / (u - 1) = u^2 + u + 1" => "Use (u^3 - 1) / (u - 1) = u^2 + u + 1",
        "Factorizar el numerador como suma de cubos" => "Factor the numerator as a sum of cubes",
        "Factorizar el numerador como diferencia de cubos" => "Factor the numerator as a difference of cubes",
        "Numerador y denominador quedan iguales, así que el cociente vale 1" => "Numerator and denominator are equal, so the quotient is 1",
        "Simplificar fracción anidada" => "Simplify nested fraction",
        "Simplificar expresión" => "Simplify expression",
        "Invertir la fracción del denominador" => "Invert the fraction in the denominator",
        "Simplificar el producto resultante" => "Simplify the resulting product",
        "Llevar a denominador común dentro del denominador" => "Put over a common denominator inside the denominator",
        "Dividir entre una fracción es multiplicar por su inversa" => "Dividing by a fraction is multiplying by its reciprocal",
        "Definimos las bases de los cubos" => "Define the bases of the cubes",
        "Verificamos que x + y + z = 0" => "Verify that x + y + z = 0",
        "Aplicamos la identidad: si x+y+z=0, entonces x³+y³+z³=3xyz" => "Apply the identity: if x+y+z=0, then x³+y³+z³=3xyz",
        "Sustitución para simplificar" => "Substitution to simplify",
        "Expresión sustituida" => "Substituted expression",
        "Todos los términos se cancelan" => "All terms cancel",
        "Convertir a forma normal polinómica" => "Convert to polynomial normal form",
        "Cancelar términos semejantes" => "Cancel like terms",
        "Expandir lado izquierdo" => "Expand the left-hand side",
        "Expandir lado derecho" => "Expand the right-hand side",
        "Comparar formas normales" => "Compare normal forms",
        "Factorizar el numerador como suma o diferencia de cubos" => "Factor the numerator as a sum or difference of cubes",
        "Reemplazar ese bloque en la expresión" => "Replace that block in the expression",
        "Cancelar términos opuestos exactos" => "Cancel exact opposite terms",
        "Comprobar anulación de factores" => "Check that the factors vanish",
        "Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)" => "Use 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)",
        "Usar 1 / (u · (u + g)) = 1 / g · (1 / u - 1 / (u + g))" => "Use 1 / (u · (u + g)) = 1 / g · (1 / u - 1 / (u + g))",
        "La suma telescópica cancela los términos intermedios" => "The telescoping sum cancels the intermediate terms",
        "Usar (u^2 - 1) / u^2 = ((u - 1) · (u + 1)) / u^2" => "Use (u^2 - 1) / u^2 = ((u - 1) · (u + 1)) / u^2",
        "Los factores (u + 1) y (u - 1) se cancelan telescópicamente" => "The (u + 1) and (u - 1) factors cancel telescopically",
        "Solo quedan el primer factor u - 1 y el último factor u + 1" => "Only the first factor u - 1 and the last factor u + 1 remain",
        "Escribir los primeros y últimos factores del producto" => "Write the first and last factors of the product",
        "Los factores intermedios se cancelan por parejas" => "The intermediate factors cancel in pairs",
        "Solo quedan el último numerador y el primer denominador" => "Only the last numerator and the first denominator remain",
        "Escribir la suma con sus extremos" => "Write the sum with its endpoints",
        "Escribir el producto con sus extremos" => "Write the product with its endpoints",
        "Usar la fórmula cerrada para la suma de enteros" => "Use the closed-form formula for the sum of integers",
        "Usar la fórmula cerrada para la suma de cuadrados" => "Use the closed-form formula for the sum of squares",
        "Usar la fórmula cerrada para la suma de cubos" => "Use the closed-form formula for the sum of cubes",
        "Usar la fórmula cerrada para la suma geométrica" => "Use the closed-form formula for the geometric sum",
        "Contar términos iguales en la suma" => "Count the equal terms in the sum",
        "Usar factorial para el producto de enteros consecutivos" => "Use a factorial for the product of consecutive integers",
        "Convertir el producto de potencias en potencia de factoriales" => "Convert the product of powers into a power of factorials",
        "Contar factores iguales en el producto" => "Count the equal factors in the product",
        "Llevar a denominador común" => "Put over a common denominator",
        "Simplificar el numerador y el denominador" => "Simplify the numerator and the denominator",
        "Usar derivación logarítmica" => "Use logarithmic differentiation",
        "Usar regla del producto" => "Use the product rule",
        "Derivar el primer factor" => "Differentiate the first factor",
        "Derivar el segundo factor" => "Differentiate the second factor",
        "Usar linealidad de la derivada" => "Use linearity of the derivative",
        "Usar factor constante de la derivada" => "Use the constant multiple rule of the derivative",
        "Usar regla del cociente" => "Use the quotient rule",
        "Derivar el numerador" => "Differentiate the numerator",
        "Derivar el denominador" => "Differentiate the denominator",
        "Usar regla de la potencia" => "Use the power rule",
        "Usar regla de la potencia con cadena" => "Use the power rule with the chain rule",
        "Usar regla exponencial" => "Use the exponential rule",
        "Usar regla de la cadena" => "Use the chain rule",
        "Usar regla de sin(u)" => "Use the sin(u) rule",
        "Usar regla de cos(u)" => "Use the cos(u) rule",
        "Usar regla de tan(u)" => "Use the tan(u) rule",
        "Usar regla de ln(u)" => "Use the ln(u) rule",
        "Usar regla de exp(u)" => "Use the exp(u) rule",
        "Usar regla de sqrt(u)" => "Use the sqrt(u) rule",
        "Usar regla de arctan(u)" => "Use the arctan(u) rule",
        "Usar regla de arcsin(u)" => "Use the arcsin(u) rule",
        "Usar regla de arccos(u)" => "Use the arccos(u) rule",
        "Usar regla de sec(u)" => "Use the sec(u) rule",
        "Usar regla de csc(u)" => "Use the csc(u) rule",
        "Usar regla de cot(u)" => "Use the cot(u) rule",
        "Usar regla de sinh(u)" => "Use the sinh(u) rule",
        "Usar regla de cosh(u)" => "Use the cosh(u) rule",
        "Usar regla de tanh(u)" => "Use the tanh(u) rule",
        "Usar derivada de sign(u) fuera de u = 0" => "Use the derivative of sign(u) away from u = 0",
        "Expandir el factorial superior hasta llegar al factorial inferior" => "Expand the upper factorial down to the lower factorial",
        "Escribir el factorial superior como el siguiente número por el factorial anterior" => "Write the upper factorial as the next number times the previous factorial",
        "Cancelar el factorial común" => "Cancel the common factorial",
        "Agrupar los términos del mismo grado" => "Group the terms of the same degree",
        "Distribuir cada término del producto" => "Distribute each term of the product",
        "Dominancia: el logaritmo crece más despacio que la exponencial (jerarquía ln ≪ potencia ≪ exp), así que el cociente → 0" => "Dominance: the logarithm grows more slowly than the exponential (hierarchy ln ≪ power ≪ exp), so the quotient → 0",
        "Dominancia: la potencia crece más despacio que la exponencial (jerarquía ln ≪ potencia ≪ exp), así que el cociente → 0" => "Dominance: the power grows more slowly than the exponential (hierarchy ln ≪ power ≪ exp), so the quotient → 0",
        "Los términos intermedios se cancelan por parejas" => "The intermediate terms cancel in pairs",
        "Usar integración por partes" => "Use integration by parts",
        "Usar integración por partes repetida" => "Use repeated integration by parts",
        "Usar regla de potencia para integrales" => "Use the power rule for integrals",
        "Aplicar el algoritmo de Euclides (restos sucesivos)" => {
            "Apply the Euclidean algorithm (successive remainders)"
        }
        "Usar lcm(a, b) = a · b / gcd(a, b)" => "Use lcm(a, b) = a · b / gcd(a, b)",
        "Factorizar n en potencias de primos" => "Factor n into prime powers",
        "Aplicar la fórmula de Euler φ(n) = n · ∏(1 - 1/p)" => {
            "Apply Euler's totient formula φ(n) = n · ∏(1 - 1/p)"
        }
        "Combinar las potencias de los factores primos" => "Combine the powers of the prime factors",
        "Sumar todos los divisores" => "Sum all divisors",
        "Aplicar la recurrencia F(n) = F(n-1) + F(n-2)" => {
            "Apply the Fibonacci recurrence F(n) = F(n-1) + F(n-2)"
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_static_key_both_languages() {
        assert_eq!(
            translate("by_parts.choose_u_dv", &[], Language::Es),
            "Elegir u y dv"
        );
        assert_eq!(
            translate("by_parts.choose_u_dv", &[], Language::En),
            "Choose u and dv"
        );
    }

    #[test]
    fn unknown_key_passes_through() {
        assert_eq!(translate("not.a.key", &[], Language::En), "not.a.key");
    }

    #[test]
    fn fill_substitutes_positional_args() {
        assert_eq!(fill("Compute Δ = {0}", &["a^2 - b"]), "Compute Δ = a^2 - b");
        assert_eq!(fill("{0} over {1}", &["x", "y"]), "x over y");
        assert_eq!(fill("no args here", &["unused"]), "no args here");
    }
}
