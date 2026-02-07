use crate::step::Step;
use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Signed;

use super::SubStep;

/// Pattern classification for nested fractions
#[derive(Debug)]
pub(crate) enum NestedFractionPattern {
    /// P1: 1/(a + 1/b) → b/(a·b + 1)
    OneOverSumWithUnitFraction,
    /// P2: 1/(a + b/c) → c/(a·c + b)
    OneOverSumWithFraction,
    /// P3: A/(B + C/D) → A·D/(B·D + C)
    FractionOverSumWithFraction,
    /// P4: (A + 1/B)/C → (A·B + 1)/(B·C)
    SumWithFractionOverScalar,
    /// Fallback for complex patterns
    General,
}

/// Check if expression contains a division (nested fraction)
pub(crate) fn contains_div(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Div(_, _) => true,
        Expr::Add(l, r) | Expr::Sub(l, r) => contains_div(ctx, *l) || contains_div(ctx, *r),
        Expr::Mul(l, r) => contains_div(ctx, *l) || contains_div(ctx, *r),
        Expr::Pow(b, e) => {
            // Check for negative exponent (b^(-1) = 1/b)
            if let Expr::Neg(_) = ctx.get(*e) {
                return true;
            }
            if let Expr::Number(n) = ctx.get(*e) {
                if n.is_negative() {
                    return true;
                }
            }
            contains_div(ctx, *b) || contains_div(ctx, *e)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => contains_div(ctx, *inner),
        Expr::Function(_, args) => args.iter().any(|a| contains_div(ctx, *a)),
        Expr::Matrix { data, .. } => data.iter().any(|e| contains_div(ctx, *e)),
        // Leaves
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => false,
    }
}

/// Find and return the first Div node within an expression
fn find_div_in_expr(ctx: &Context, id: ExprId) -> Option<ExprId> {
    match ctx.get(id) {
        Expr::Div(_, _) => Some(id),
        Expr::Add(l, r) | Expr::Sub(l, r) => {
            find_div_in_expr(ctx, *l).or_else(|| find_div_in_expr(ctx, *r))
        }
        Expr::Mul(l, r) => find_div_in_expr(ctx, *l).or_else(|| find_div_in_expr(ctx, *r)),
        Expr::Neg(inner) | Expr::Hold(inner) => find_div_in_expr(ctx, *inner),
        Expr::Pow(b, e) => find_div_in_expr(ctx, *b).or_else(|| find_div_in_expr(ctx, *e)),
        Expr::Function(_, args) => args.iter().find_map(|a| find_div_in_expr(ctx, *a)),
        Expr::Matrix { data, .. } => data.iter().find_map(|e| find_div_in_expr(ctx, *e)),
        // Leaves
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => None,
    }
}

/// Classify a nested fraction expression and return the pattern and extracted components
pub(crate) fn classify_nested_fraction(
    ctx: &Context,
    expr: ExprId,
) -> Option<NestedFractionPattern> {
    // Helper to check if expression is 1
    let is_one = |id: ExprId| -> bool {
        matches!(ctx.get(id), Expr::Number(n) if n.is_integer() && *n.numer() == BigInt::from(1))
    };

    // Helper to extract a fraction (1/x or a/b) from Add terms
    let find_fraction_in_add = |id: ExprId| -> Option<ExprId> {
        match ctx.get(id) {
            Expr::Add(l, r) => {
                if matches!(ctx.get(*l), Expr::Div(_, _)) {
                    Some(*l)
                } else if matches!(ctx.get(*r), Expr::Div(_, _)) {
                    Some(*r)
                } else {
                    None
                }
            }
            _ => None,
        }
    };

    if let Expr::Div(num, den) = ctx.get(expr) {
        // P1/P2/P3: Something/(... + .../...)
        if let Some(inner_frac) = find_fraction_in_add(*den) {
            if is_one(*num) {
                // P1 or P2: 1/(a + ?/?)
                if let Expr::Div(n, _) = ctx.get(inner_frac) {
                    if is_one(*n) {
                        return Some(NestedFractionPattern::OneOverSumWithUnitFraction);
                    }
                }
                return Some(NestedFractionPattern::OneOverSumWithFraction);
            } else {
                // P3: A/(B + C/D)
                return Some(NestedFractionPattern::FractionOverSumWithFraction);
            }
        }

        // P4: (A + 1/B)/C - numerator contains fraction
        if contains_div(ctx, *num) && !contains_div(ctx, *den) {
            return Some(NestedFractionPattern::SumWithFractionOverScalar);
        }

        // General nested: denominator has nested structure
        if contains_div(ctx, *den) {
            return Some(NestedFractionPattern::General);
        }
    }

    None
}

/// Extract the combined fraction string from an Add expression containing a fraction.
/// For example: 1 + 1/x → "\frac{x + 1}{x}" in LaTeX
pub(crate) fn extract_combined_fraction_str(ctx: &Context, add_expr: ExprId) -> String {
    use cas_ast::display_context::DisplayContext;
    use cas_ast::LaTeXExprWithHints;

    // Helper to convert expression to LaTeX
    let hints = DisplayContext::default();
    let to_latex = |id: ExprId| -> String {
        LaTeXExprWithHints {
            context: ctx,
            id,
            hints: &hints,
        }
        .to_latex()
    };

    // Find the fraction term and non-fraction term
    if let Expr::Add(l, r) = ctx.get(add_expr) {
        let (frac_id, other_id) = if matches!(ctx.get(*l), Expr::Div(_, _)) {
            (*l, *r)
        } else if matches!(ctx.get(*r), Expr::Div(_, _)) {
            (*r, *l)
        } else {
            // No fraction found, return generic
            return "\\text{(combinado)}".to_string();
        };

        // Extract numerator and denominator of the fraction
        if let Expr::Div(frac_num, frac_den) = ctx.get(frac_id) {
            let frac_num_latex = to_latex(*frac_num);
            let frac_den_latex = to_latex(*frac_den);
            let other_latex = to_latex(other_id);

            // Build the combined expression in LaTeX: \frac{other·den + num}{den}
            return format!(
                "\\frac{{{} \\cdot {} + {}}}{{{}}}",
                other_latex, frac_den_latex, frac_num_latex, frac_den_latex
            );
        }
    }

    "\\text{(combinado)}".to_string()
}

/// Generate sub-steps explaining nested fraction simplification
/// For example: 1/(1 + 1/x) shows:
///   1. Combine terms in denominator: 1 + 1/x → (x+1)/x
///   2. Invert the fraction: 1/((x+1)/x) → x/(x+1)
pub(crate) fn generate_nested_fraction_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    use cas_ast::display_context::DisplayContext;
    use cas_ast::LaTeXExprWithHints;

    let mut sub_steps = Vec::new();

    // Get the before expression (which should be a nested Div)
    let before_expr = step.before;
    let after_expr = step.after;

    // Classify the pattern
    let pattern = match classify_nested_fraction(ctx, before_expr) {
        Some(p) => p,
        None => return sub_steps, // Not a nested fraction pattern we handle
    };

    // Build display hints for proper notation
    let hints = DisplayContext::default();

    // Helper to convert expression to LaTeX
    let to_latex = |id: ExprId| -> String {
        LaTeXExprWithHints {
            context: ctx,
            id,
            hints: &hints,
        }
        .to_latex()
    };

    let before_str = to_latex(before_expr);
    let after_str = to_latex(after_expr);

    // Generate pattern-specific sub-steps
    match pattern {
        NestedFractionPattern::OneOverSumWithUnitFraction
        | NestedFractionPattern::OneOverSumWithFraction => {
            // P1/P2: 1/(a + b/c) → c/(a·c + b)
            // Extract denominator for display
            if let Expr::Div(_, den) = ctx.get(before_expr) {
                let den_str = to_latex(*den);

                // Try to extract inner fraction to show real intermediate
                // For 1/(a + b/c), the inner fraction is b/c, and combined = (a*c + b)/c
                let intermediate_str = extract_combined_fraction_str(ctx, *den);

                // Sub-step 1: Common denominator in the denominator
                sub_steps.push(SubStep {
                    description: "Combinar términos del denominador (denominador común)"
                        .to_string(),
                    before_expr: den_str.clone(),
                    after_expr: intermediate_str.clone(),
                });

                // Sub-step 2: Invert the fraction (use intermediate_str from step 1)
                sub_steps.push(SubStep {
                    description: "Invertir la fracción: 1/(a/b) = b/a".to_string(),
                    before_expr: format!("\\frac{{1}}{{{}}}", intermediate_str),
                    after_expr: after_str,
                });
            }
        }

        NestedFractionPattern::FractionOverSumWithFraction => {
            // P3: A/(B + C/D) → A·D/(B·D + C)
            if let Expr::Div(num, den) = ctx.get(before_expr) {
                let num_str = to_latex(*num);
                let den_str = to_latex(*den);

                // Try to extract inner fraction to show real intermediate
                let intermediate_str = extract_combined_fraction_str(ctx, *den);

                // Sub-step 1: Common denominator in the denominator
                sub_steps.push(SubStep {
                    description: "Combinar términos del denominador (denominador común)"
                        .to_string(),
                    before_expr: den_str,
                    after_expr: intermediate_str,
                });

                // Sub-step 2: Multiply numerator by D and simplify
                sub_steps.push(SubStep {
                    description: format!("Multiplicar {} por el denominador interno", num_str),
                    before_expr: before_str,
                    after_expr: after_str,
                });
            }
        }

        NestedFractionPattern::SumWithFractionOverScalar => {
            // P4: (A + 1/B)/C → (A·B + 1)/(B·C)
            if let Expr::Div(num, den) = ctx.get(before_expr) {
                let num_str = to_latex(*num);
                let den_str = to_latex(*den);

                // Sub-step 1: Combine the numerator
                sub_steps.push(SubStep {
                    description: "Combinar términos del numerador (denominador común)".to_string(),
                    before_expr: num_str,
                    after_expr: "(numerador combinado) / B".to_string(),
                });

                // Sub-step 2: Divide by C (multiply denominators)
                sub_steps.push(SubStep {
                    description: format!("Dividir por {}: multiplicar denominadores", den_str),
                    before_expr: before_str,
                    after_expr: after_str,
                });
            }
        }

        NestedFractionPattern::General => {
            // General nested fraction: try to show meaningful intermediate steps
            // by extracting the inner structure
            if let Expr::Div(num, den) = ctx.get(before_expr) {
                let num_str = to_latex(*num);
                let _den_str = to_latex(*den);

                // Try to find an inner fraction in the denominator
                if let Some(inner_frac) = find_div_in_expr(ctx, *den) {
                    if let Expr::Div(inner_num, inner_den) = ctx.get(inner_frac) {
                        let inner_num_str = to_latex(*inner_num);
                        let inner_den_str = to_latex(*inner_den);

                        // Sub-step 1: Identify the inner fraction structure
                        sub_steps.push(SubStep {
                            description: "Identificar la fracción anidada en el denominador"
                                .to_string(),
                            before_expr: format!(
                                "\\frac{{{}}}{{\\text{{...}} + \\frac{{{}}}{{{}}}}}",
                                num_str, inner_num_str, inner_den_str
                            ),
                            after_expr: format!("\\text{{Multiplicar por }} {}", inner_den_str),
                        });

                        // Sub-step 2: Show the actual rule applied
                        sub_steps.push(SubStep {
                            description: "Simplificar: 1/(a/b) = b/a".to_string(),
                            before_expr: before_str.clone(),
                            after_expr: after_str,
                        });
                    } else {
                        // Fallback: single step with real expressions
                        sub_steps.push(SubStep {
                            description:
                                "Simplificar fracción compleja (multiplicar por denominador común)"
                                    .to_string(),
                            before_expr: before_str.clone(),
                            after_expr: after_str,
                        });
                    }
                } else {
                    // No inner fraction found, single step
                    sub_steps.push(SubStep {
                        description: "Simplificar fracción anidada".to_string(),
                        before_expr: before_str.clone(),
                        after_expr: after_str,
                    });
                }
            } else {
                // Not a Div, shouldn't happen but handle gracefully
                sub_steps.push(SubStep {
                    description: "Simplificar expresión".to_string(),
                    before_expr: before_str.clone(),
                    after_expr: after_str,
                });
            }
        }
    }

    sub_steps
}

/// Generate sub-steps explaining rationalization process
/// Uses LaTeXExprWithHints for proper sqrt notation rendering
pub(crate) fn generate_rationalization_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    use cas_ast::display_context::DisplayContext;
    use cas_ast::LaTeXExprWithHints;

    let mut sub_steps = Vec::new();

    // Build display hints for sqrt notation
    let hints = DisplayContext::with_root_index(2);

    // Helper to convert expression to LaTeX with hints
    let to_latex = |id: ExprId| -> String {
        LaTeXExprWithHints {
            context: ctx,
            id,
            hints: &hints,
        }
        .to_latex()
    };

    // Helper to collect additive terms from an expression
    fn collect_add_terms(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
        let mut terms = Vec::new();
        collect_add_terms_recursive(ctx, expr, &mut terms);
        terms
    }

    fn collect_add_terms_recursive(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
        match ctx.get(expr) {
            Expr::Add(l, r) => {
                collect_add_terms_recursive(ctx, *l, terms);
                collect_add_terms_recursive(ctx, *r, terms);
            }
            _ => terms.push(expr),
        }
    }

    // Extract before/after expressions
    // Use before_local/after_local if available (the focused sub-expression)
    // Otherwise fall back to global before/after
    let before = step.before_local.unwrap_or(step.before);
    let after = step.after_local.unwrap_or(step.after);

    // Check if it's a generalized rationalization (3+ terms)
    if step.description.contains("group") {
        // Generalized rationalization: a/(x + y + z) -> a(x+y-z)/[(x+y)² - z²]
        if let Expr::Div(num, den) = ctx.get(before) {
            let num_latex = to_latex(*num);
            let den_latex = to_latex(*den);

            // Collect terms from denominator
            let den_terms = collect_add_terms(ctx, *den);

            if den_terms.len() >= 3 {
                // Group first n-1 terms as "group", last term as "c"
                let group_terms: Vec<String> = den_terms[..den_terms.len() - 1]
                    .iter()
                    .map(|t| to_latex(*t))
                    .collect();
                let last_term = to_latex(den_terms[den_terms.len() - 1]);

                let group_str = group_terms.join(" + ");

                // Sub-step 1: Show the original fraction and grouping
                sub_steps.push(SubStep {
                    description: "Agrupar términos del denominador".to_string(),
                    before_expr: format!("\\frac{{{}}}{{{}}}", num_latex, den_latex),
                    after_expr: if group_terms.len() > 1 {
                        format!("\\frac{{{}}}{{({}) + {}}}", num_latex, group_str, last_term)
                    } else {
                        format!("\\frac{{{}}}{{{} + {}}}", num_latex, group_str, last_term)
                    },
                });

                // Sub-step 2: Identify conjugate with specific terms
                let conjugate = if group_terms.len() > 1 {
                    format!("({}) - {}", group_str, last_term)
                } else {
                    format!("{} - {}", group_str, last_term)
                };

                sub_steps.push(SubStep {
                    description: "Multiplicar por el conjugado".to_string(),
                    before_expr: if group_terms.len() > 1 {
                        format!("({}) + {}", group_str, last_term)
                    } else {
                        format!("{} + {}", group_str, last_term)
                    },
                    after_expr: conjugate.clone(),
                });

                // Sub-step 3: Apply difference of squares with specific terms
                if let Expr::Div(_new_num, new_den) = ctx.get(after) {
                    let after_den_latex = to_latex(*new_den);
                    sub_steps.push(SubStep {
                        description: "Diferencia de cuadrados".to_string(),
                        before_expr: if group_terms.len() > 1 {
                            format!("({})^2 - ({})^2", group_str, last_term)
                        } else {
                            format!("{}^2 - {}^2", group_str, last_term)
                        },
                        after_expr: after_den_latex,
                    });
                }
            }
        }
    } else if step.description.contains("product") {
        // Product rationalization: a/(b·√c) -> a·√c/(b·c)
        if let Expr::Div(num, den) = ctx.get(before) {
            let num_latex = to_latex(*num);
            let den_latex = to_latex(*den);

            // Sub-step 1: Identify the radical in denominator
            sub_steps.push(SubStep {
                description: "Denominador con producto de radical".to_string(),
                before_expr: format!("\\frac{{{}}}{{{}}}", num_latex, den_latex),
                after_expr: "\\frac{a}{k \\cdot \\sqrt{n}}".to_string(),
            });

            // Sub-step 2: Multiply by √n/√n
            if let Expr::Div(new_num, new_den) = ctx.get(after) {
                let after_num_latex = to_latex(*new_num);
                let after_den_latex = to_latex(*new_den);
                sub_steps.push(SubStep {
                    description: "Multiplicar por \\sqrt{n}/\\sqrt{n}".to_string(),
                    before_expr: format!(
                        "\\frac{{{} \\cdot \\sqrt{{n}}}}{{{} \\cdot \\sqrt{{n}}}}",
                        num_latex, den_latex
                    ),
                    after_expr: format!("\\frac{{{}}}{{{}}}", after_num_latex, after_den_latex),
                });
            }
        }
    } else {
        // Binary rationalization (difference of squares with 2 terms)
        if let Expr::Div(num, den) = ctx.get(before) {
            let num_latex = to_latex(*num);
            let den_latex = to_latex(*den);

            // Try to extract the actual terms (a ± b) from denominator
            // For √x - 1 (stored as Add(√x, Neg(1))), terms are √x and 1, conjugate is √x + 1
            // For √x + 1, conjugate is √x - 1
            let (term_a, term_b, is_original_minus) = match ctx.get(*den) {
                Expr::Add(l, r) => {
                    // Check if r is negative (could be Neg(x) or Number(-n))
                    match ctx.get(*r) {
                        Expr::Neg(inner) => {
                            // a + (-b) => original is "a - b", conjugate is "a + b"
                            (to_latex(*l), to_latex(*inner), true)
                        }
                        Expr::Number(n) if n.is_negative() => {
                            // a + (-1) stored as Add(a, Number(-1))
                            // original is "a - 1", conjugate is "a + 1"
                            // Format the absolute value directly
                            let abs_n = -n;
                            let abs_str = if abs_n.is_integer() {
                                format!("{}", abs_n.numer())
                            } else {
                                format!("\\frac{{{}}}{{{}}}", abs_n.numer(), abs_n.denom())
                            };
                            (to_latex(*l), abs_str, true)
                        }
                        _ => {
                            // a + b => conjugate is "a - b"
                            (to_latex(*l), to_latex(*r), false)
                        }
                    }
                }
                Expr::Sub(l, r) => (to_latex(*l), to_latex(*r), true),
                _ => (den_latex.clone(), String::new(), false),
            };

            // Build conjugate string (flip the sign)
            let conjugate = if term_b.is_empty() {
                den_latex.clone()
            } else if is_original_minus {
                // Original was a - b, conjugate is a + b
                format!("{} + {}", term_a, term_b)
            } else {
                // Original was a + b, conjugate is a - b
                format!("{} - {}", term_a, term_b)
            };

            // Sub-step 1: Identify binomial and conjugate
            sub_steps.push(SubStep {
                description: "Denominador binomial con radical".to_string(),
                before_expr: format!("\\frac{{{}}}{{{}}}", num_latex, den_latex),
                after_expr: format!("\\text{{Conjugado: }} {}", conjugate),
            });

            // Sub-step 2: Apply difference of squares
            if let Expr::Div(new_num, new_den) = ctx.get(after) {
                let after_num_latex = to_latex(*new_num);
                let after_den_latex = to_latex(*new_den);

                sub_steps.push(SubStep {
                    description: "(a+b)(a-b) = a² - b²".to_string(),
                    before_expr: format!(
                        "\\frac{{({}) \\cdot ({})}}{{{}  \\cdot ({})}}",
                        num_latex, conjugate, den_latex, conjugate
                    ),
                    after_expr: format!("\\frac{{{}}}{{{}}}", after_num_latex, after_den_latex),
                });
            }
        }
    }

    sub_steps
}

/// Generate sub-steps explaining polynomial identity normalization (PolyZero airbag)
///
/// When a polynomial identity like `(a+b)^2 - a^2 - 2ab - b^2` is detected to equal 0,
/// this generates explanatory sub-steps:
///   1. "Convert to polynomial normal form" - shows the expanded/normalized polynomial or stats
///   2. "All coefficients cancel → 0" - explains the cancellation
///
/// The proof data attached to the step contains:
/// - monomials: count of monomials in the polynomial (0 for identity = 0)
/// - degree: maximum degree of the polynomial
/// - vars: list of variable names
/// - normal_form_expr: the normalized expression if small enough to display
pub(crate) fn generate_polynomial_identity_substeps(
    ctx: &Context,
    step: &crate::step::Step,
) -> Vec<SubStep> {
    use cas_ast::DisplayExpr;

    let mut sub_steps = Vec::new();

    // Get the proof data (caller should have checked is_some())
    let proof = match &step.poly_proof {
        Some(p) => p,
        None => return sub_steps,
    };

    // Helper to format polynomial stats
    let format_poly_stats = |stats: &crate::multipoly_display::PolyNormalFormStats| -> String {
        if let Some(expr_id) = stats.expr {
            format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: expr_id
                }
            )
        } else {
            format!("{} monomios, grado {}", stats.monomials, stats.degree)
        }
    };

    // Check if we have LHS/RHS split (better for identities)
    if let (Some(lhs_stats), Some(rhs_stats)) = (&proof.lhs_stats, &proof.rhs_stats) {
        // Sub-step 1: Show LHS normal form
        sub_steps.push(SubStep {
            description: "Expandir lado izquierdo".to_string(),
            before_expr: "(a + b + c)³".to_string(), // Placeholder, will be overwritten
            after_expr: format_poly_stats(lhs_stats),
        });

        // Sub-step 2: Show RHS normal form
        sub_steps.push(SubStep {
            description: "Expandir lado derecho".to_string(),
            before_expr: "a³ + b³ + c³ + ...".to_string(), // Placeholder
            after_expr: format_poly_stats(rhs_stats),
        });

        // Sub-step 3: Compare and show they match
        sub_steps.push(SubStep {
            description: "Comparar formas normales".to_string(),
            before_expr: format!(
                "LHS: {} monomios | RHS: {} monomios",
                lhs_stats.monomials, rhs_stats.monomials
            ),
            after_expr: "Coinciden ⇒ diferencia = 0".to_string(),
        });
    } else {
        // Fallback: single normal form display
        let normal_form_description = if let Some(expr_id) = proof.normal_form_expr {
            format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: expr_id
                }
            )
        } else {
            let vars_str = if proof.vars.is_empty() {
                "constante".to_string()
            } else {
                proof.vars.join(", ")
            };
            format!(
                "{} monomios, grado {}, vars: {}",
                proof.monomials, proof.degree, vars_str
            )
        };

        sub_steps.push(SubStep {
            description: "Convertir a forma normal polinómica".to_string(),
            before_expr: format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: step.before
                }
            ),
            after_expr: normal_form_description,
        });

        sub_steps.push(SubStep {
            description: "Cancelar términos semejantes".to_string(),
            before_expr: "todos los coeficientes".to_string(),
            after_expr: "0".to_string(),
        });
    }

    sub_steps
}

/// Generate sub-steps explaining the Sum of Three Cubes identity
/// When x + y + z = 0, we have x³ + y³ + z³ = 3xyz
///
/// For (a-b)³ + (b-c)³ + (c-a)³, the substeps are:
///   1. Define x = (a-b), y = (b-c), z = (c-a)
///   2. Verify x + y + z = 0
///   3. Apply the identity x³ + y³ + z³ = 3xyz
pub(crate) fn generate_sum_three_cubes_substeps(
    ctx: &Context,
    step: &crate::step::Step,
) -> Vec<SubStep> {
    use crate::helpers::flatten_add;
    use cas_ast::DisplayExpr;

    let mut sub_steps = Vec::new();

    // Extract the three cubed bases from the before expression
    let before_expr = step.before;

    // Flatten the sum to get individual terms
    let mut terms = Vec::new();
    flatten_add(ctx, before_expr, &mut terms);

    if terms.len() != 3 {
        return sub_steps; // Not the expected pattern
    }

    // Extract bases from each cube
    let mut bases: Vec<ExprId> = Vec::new();
    for &term in &terms {
        let base = match ctx.get(term).clone() {
            Expr::Pow(b, e) => {
                if let Expr::Number(n) = ctx.get(e).clone() {
                    if n.is_integer() && n.to_integer() == BigInt::from(3) {
                        Some(b)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            Expr::Neg(inner) => {
                if let Expr::Pow(_b, e) = ctx.get(inner).clone() {
                    if let Expr::Number(n) = ctx.get(e).clone() {
                        if n.is_integer() && n.to_integer() == BigInt::from(3) {
                            // The base is negated: -(x³) - just use inner directly
                            // We'll handle the negation in display
                            Some(inner) // Return the Pow node, we'll detect negation later
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(b) = base {
            bases.push(b);
        } else {
            return sub_steps; // Not the expected pattern
        }
    }

    if bases.len() != 3 {
        return sub_steps;
    }

    // Helper to format expression
    let fmt = |id: ExprId| -> String { format!("{}", DisplayExpr { context: ctx, id }) };

    let x_str = fmt(bases[0]);
    let y_str = fmt(bases[1]);
    let z_str = fmt(bases[2]);

    // Sub-step 1: Define x, y, z
    sub_steps.push(SubStep {
        description: "Definimos las bases de los cubos".to_string(),
        before_expr: format!("x = {}, \\quad y = {}, \\quad z = {}", x_str, y_str, z_str),
        after_expr: "x^3 + y^3 + z^3".to_string(),
    });

    // Sub-step 2: Show that x + y + z = 0
    sub_steps.push(SubStep {
        description: "Verificamos que x + y + z = 0".to_string(),
        before_expr: format!("({}) + ({}) + ({})", x_str, y_str, z_str),
        after_expr: "0 \\quad \\checkmark".to_string(),
    });

    // Sub-step 3: Apply the identity
    sub_steps.push(SubStep {
        description: "Aplicamos la identidad: si x+y+z=0, entonces x³+y³+z³=3xyz".to_string(),
        before_expr: format!("{}^3 + {}^3 + {}^3", x_str, y_str, z_str),
        after_expr: format!("3 \\cdot ({}) \\cdot ({}) \\cdot ({})", x_str, y_str, z_str),
    });

    sub_steps
}

/// Generate sub-steps explaining root denesting process
/// For √(a + c·√d), the substeps show:
///   1. Identify the form √(a + c·√d) with values
///   2. Calculate Δ = a² - c²d
///   3. Verify Δ is a perfect square and apply the formula
pub(crate) fn generate_root_denesting_substeps(
    ctx: &Context,
    step: &crate::step::Step,
) -> Vec<SubStep> {
    use cas_ast::display_context::DisplayContext;
    use cas_ast::LaTeXExprWithHints;
    use num_traits::Signed;

    let mut sub_steps = Vec::new();

    // Get the before expression (should be sqrt(a + c·√d))
    let before_expr = step.before_local.unwrap_or(step.before);

    // Build display hints for proper sqrt notation
    let hints = DisplayContext::with_root_index(2);

    // Helper for LaTeX display (for timeline)
    let to_latex = |id: ExprId| -> String {
        LaTeXExprWithHints {
            context: ctx,
            id,
            hints: &hints,
        }
        .to_latex()
    };

    // Helper to extract sqrt radicand
    let get_sqrt_inner = |id: ExprId| -> Option<ExprId> {
        match ctx.get(id) {
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 =>
            {
                Some(args[0])
            }
            Expr::Pow(base, exp) => {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if *n.numer() == BigInt::from(1) && *n.denom() == BigInt::from(2) {
                        return Some(*base);
                    }
                }
                None
            }
            _ => None,
        }
    };

    // Get the inner expression of the sqrt
    let inner = match get_sqrt_inner(before_expr) {
        Some(id) => id,
        None => return sub_steps,
    };

    // Extract a ± c·√d pattern
    let (a_term, b_term, is_add) = match ctx.get(inner) {
        Expr::Add(l, r) => (*l, *r, true),
        Expr::Sub(l, r) => (*l, *r, false),
        _ => return sub_steps,
    };

    // Try to identify which is the rational part and which is the surd part
    // The surd should be c·√d or just √d
    fn analyze_surd(ctx: &Context, e: ExprId) -> Option<(BigRational, ExprId)> {
        match ctx.get(e) {
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 =>
            {
                Some((BigRational::from_integer(BigInt::from(1)), args[0]))
            }
            Expr::Pow(base, exp) => {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if *n.numer() == BigInt::from(1) && *n.denom() == BigInt::from(2) {
                        return Some((BigRational::from_integer(BigInt::from(1)), *base));
                    }
                }
                None
            }
            Expr::Mul(l, r) => {
                // Check both orderings
                if let Expr::Number(coef) = ctx.get(*l) {
                    if let Some((_, d)) = analyze_surd(ctx, *r) {
                        return Some((coef.clone(), d));
                    }
                }
                if let Expr::Number(coef) = ctx.get(*r) {
                    if let Some((_, d)) = analyze_surd(ctx, *l) {
                        return Some((coef.clone(), d));
                    }
                }
                None
            }
            _ => None,
        }
    }

    // Determine which term is a (rational) and which is c·√d (surd)
    let (a_val, c_val, d_val, _d_id) = if let Expr::Number(a) = ctx.get(a_term) {
        if let Some((c, d_id)) = analyze_surd(ctx, b_term) {
            if let Expr::Number(d) = ctx.get(d_id) {
                (a.clone(), c, d.clone(), d_id)
            } else {
                return sub_steps;
            }
        } else {
            return sub_steps;
        }
    } else if let Expr::Number(a) = ctx.get(b_term) {
        if let Some((c, d_id)) = analyze_surd(ctx, a_term) {
            if let Expr::Number(d) = ctx.get(d_id) {
                (a.clone(), c, d.clone(), d_id)
            } else {
                return sub_steps;
            }
        } else {
            return sub_steps;
        }
    } else {
        return sub_steps;
    };

    // Calculate the discriminant: Δ = a² - c²d
    let a2 = &a_val * &a_val;
    let c2 = &c_val * &c_val;
    let c2d = &c2 * &d_val;
    let delta = &a2 - &c2d;

    // Get z = √Δ if it's a perfect square
    if delta.is_negative() || !delta.is_integer() {
        return sub_steps;
    }
    let delta_int = delta.to_integer();
    let z = delta_int.sqrt();
    if &z * &z != delta_int {
        return sub_steps;
    }

    let op_sign = if is_add { "+" } else { "-" };

    // Sub-step 1: Identify the pattern - using LaTeX format
    sub_steps.push(SubStep {
        description: "Reconocer patrón √(a + c·√d)".to_string(),
        before_expr: to_latex(before_expr),
        after_expr: format!("a = {}, c = {}, d = {}", a_val, c_val.abs(), d_val),
    });

    // Sub-step 2: Calculate discriminant - LaTeX for timeline
    sub_steps.push(SubStep {
        description: "Calcular Δ = a² − c²·d".to_string(),
        before_expr: format!("{}^2 - {}^2 \\cdot {}", a_val, c_val.abs(), d_val),
        after_expr: format!("{} - {} = {}", a2, c2d, delta_int),
    });

    // Sub-step 3: Verify perfect square and apply formula
    let az = &a_val + BigRational::from_integer(z.clone());
    let az_half = &az / BigRational::from_integer(BigInt::from(2));
    let amz = &a_val - BigRational::from_integer(z.clone());
    let amz_half = &amz / BigRational::from_integer(BigInt::from(2));

    sub_steps.push(SubStep {
        description: "Δ es cuadrado perfecto → aplicar sqrt((a+z)/2) ± sqrt((a−z)/2)".to_string(),
        before_expr: format!("\\Delta = {} = {}^2 \\Rightarrow z = {}", delta_int, z, z),
        after_expr: format!(
            "\\sqrt{{\\frac{{{}+{}}}{{2}}}} {} \\sqrt{{\\frac{{{}-{}}}{{2}}}} = \\sqrt{{{}}} {} \\sqrt{{{}}}",
            a_val, z, op_sign, a_val, z, az_half, op_sign, amz_half
        ),
    });

    sub_steps
}
