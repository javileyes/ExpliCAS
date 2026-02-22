//! Quadratic formula didactic substep construction.
//!
//! This module builds the symbolic derivation steps (completing the square)
//! independent of engine-specific step types/renderers.

use cas_ast::{Context, Equation, Expr, ExprId, RelOp};

/// Main didactic narration for quadratic-strategy activation.
pub const QUADRATIC_FORMULA_MAIN_STEP_DESCRIPTION: &str =
    "Detected quadratic equation. Applying quadratic formula.";

/// Narration for factored zero-product entrypoint.
pub fn factorized_equation_message(expr_display: &str) -> String {
    format!("Factorized equation: {} = 0", expr_display)
}

/// Narration for solving one zero-product factor.
pub fn solve_factor_message(factor_display: &str) -> String {
    format!("Solve factor: {} = 0", factor_display)
}

/// Core didactic step payload for quadratic derivations.
#[derive(Debug, Clone, PartialEq)]
pub struct DidacticSubstep {
    pub description: String,
    pub equation_after: Equation,
}

/// Check if an expression is a literal zero (Number(0)).
fn is_zero_literal(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if *n == num_rational::BigRational::from_integer(0.into()))
}

/// Check if an expression is a numeric constant (Number).
fn is_numeric(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(_))
}

/// Build didactic substeps for solving a quadratic equation.
///
/// `render_expr` is injected by caller (engine/CLI) so this module remains
/// independent of formatting crates.
pub fn build_quadratic_substeps_with<F>(
    ctx: &mut Context,
    var: &str,
    a: ExprId,
    b: ExprId,
    c: ExprId,
    is_real_only: bool,
    mut render_expr: F,
) -> Vec<DidacticSubstep>
where
    F: FnMut(&Context, ExprId) -> String,
{
    let mut steps = Vec::new();
    let zero = ctx.num(0);
    let one = ctx.num(1);
    let two = ctx.num(2);
    let four = ctx.num(4);

    // Check if all coefficients are numeric - if so, don't show algebraic derivation.
    let all_numeric = is_numeric(ctx, a) && is_numeric(ctx, b) && is_numeric(ctx, c);

    // Peephole optimization: check if c is literally zero.
    let c_is_zero = is_zero_literal(ctx, c);

    // Build x variable.
    let x = ctx.var(var);

    // Build x².
    let x2 = ctx.add(Expr::Pow(x, two));

    // =========================================================================
    // NUMERIC CASE: when a, b, c are all constants, show natural algebraic steps.
    // =========================================================================
    if all_numeric {
        // Build the polynomial expression for display.
        let ax2 = ctx.add(Expr::Mul(a, x2));
        let bx = ctx.add(Expr::Mul(b, x));
        let ax2_plus_bx = ctx.add(Expr::Add(ax2, bx));
        let _poly = ctx.add(Expr::Add(ax2_plus_bx, c));

        // CASE 1: c = 0 -> factor out x.
        if c_is_zero {
            // x(ax + b) = 0.
            let ax = ctx.add(Expr::Mul(a, x));
            let inner = ctx.add(Expr::Add(ax, b)); // ax + b
            let factored_form = ctx.add(Expr::Mul(x, inner)); // x(ax + b)

            steps.push(DidacticSubstep {
                description: "Factorizar x común".to_string(),
                equation_after: Equation {
                    lhs: factored_form,
                    rhs: zero,
                    op: RelOp::Eq,
                },
            });

            steps.push(DidacticSubstep {
                description: "Producto igual a cero: algún factor es cero".to_string(),
                equation_after: Equation {
                    lhs: x,
                    rhs: zero,
                    op: RelOp::Eq,
                },
            });

            let neg_b = ctx.add(Expr::Neg(b));
            let sol2 = ctx.add(Expr::Div(neg_b, a));

            steps.push(DidacticSubstep {
                description: format!("Resolver {} = 0", render_expr(ctx, inner)),
                equation_after: Equation {
                    lhs: x,
                    rhs: sol2,
                    op: RelOp::Eq,
                },
            });

            return steps;
        }
        // General numeric case (c != 0): fall through to full derivation.
    }

    // =========================================================================
    // Step 1: Identify the equation form.
    // =========================================================================
    let ax2 = ctx.add(Expr::Mul(a, x2));
    let bx = ctx.add(Expr::Mul(b, x));
    let ax2_plus_bx = ctx.add(Expr::Add(ax2, bx));
    let poly = ctx.add(Expr::Add(ax2_plus_bx, c));

    steps.push(DidacticSubstep {
        description: format!(
            "Identificar forma cuadrática: a·x² + b·x + c = 0 con a = {}, b = {}, c = {}",
            render_expr(ctx, a),
            render_expr(ctx, b),
            render_expr(ctx, c)
        ),
        equation_after: Equation {
            lhs: poly,
            rhs: zero,
            op: RelOp::Eq,
        },
    });

    // =========================================================================
    // Step 2: Divide both sides by a.
    // =========================================================================
    let b_over_a = ctx.add(Expr::Div(b, a));
    let c_over_a = if c_is_zero {
        zero
    } else {
        ctx.add(Expr::Div(c, a))
    };
    let bax = ctx.add(Expr::Mul(b_over_a, x));
    let x2_plus_bax = ctx.add(Expr::Add(x2, bax));
    let step2_lhs = if c_is_zero {
        x2_plus_bax
    } else {
        ctx.add(Expr::Add(x2_plus_bax, c_over_a))
    };

    steps.push(DidacticSubstep {
        description: "Dividir ambos lados por a (requiere a ≠ 0)".to_string(),
        equation_after: Equation {
            lhs: step2_lhs,
            rhs: zero,
            op: RelOp::Eq,
        },
    });

    // =========================================================================
    // Step 3: Move constant to RHS.
    // =========================================================================
    let step3_rhs = if c_is_zero {
        zero
    } else {
        ctx.add(Expr::Neg(c_over_a))
    };
    let step3_lhs = x2_plus_bax;

    steps.push(DidacticSubstep {
        description: "Mover término constante al lado derecho".to_string(),
        equation_after: Equation {
            lhs: step3_lhs,
            rhs: step3_rhs,
            op: RelOp::Eq,
        },
    });

    // =========================================================================
    // Step 4: Complete the square.
    // =========================================================================
    let two_a = ctx.add(Expr::Mul(two, a));
    let b_over_2a = ctx.add(Expr::Div(b, two_a));
    let b_over_2a_sq = ctx.add(Expr::Pow(b_over_2a, two));
    let step4_lhs = ctx.add(Expr::Add(step3_lhs, b_over_2a_sq));

    let step4_rhs = if c_is_zero {
        b_over_2a_sq
    } else {
        let neg_c_a = ctx.add(Expr::Neg(c_over_a));
        ctx.add(Expr::Add(b_over_2a_sq, neg_c_a))
    };

    steps.push(DidacticSubstep {
        description: "Completar el cuadrado: sumar (b/2a)² a ambos lados".to_string(),
        equation_after: Equation {
            lhs: step4_lhs,
            rhs: step4_rhs,
            op: RelOp::Eq,
        },
    });

    // =========================================================================
    // Step 5: Factor LHS as perfect square.
    // =========================================================================
    let x_plus_b_over_2a = ctx.add(Expr::Add(x, b_over_2a));
    let step5_lhs = ctx.add(Expr::Pow(x_plus_b_over_2a, two));

    let b2 = ctx.add(Expr::Pow(b, two));
    let discriminant = if c_is_zero {
        b2
    } else {
        let four_a = ctx.add(Expr::Mul(four, a));
        let four_ac = ctx.add(Expr::Mul(four_a, c));
        ctx.add(Expr::Sub(b2, four_ac))
    };
    let a2 = ctx.add(Expr::Pow(a, two));
    let four_a2 = ctx.add(Expr::Mul(four, a2));
    let step5_rhs = ctx.add(Expr::Div(discriminant, four_a2));

    steps.push(DidacticSubstep {
        description: "Escribir lado izquierdo como cuadrado perfecto".to_string(),
        equation_after: Equation {
            lhs: step5_lhs,
            rhs: step5_rhs,
            op: RelOp::Eq,
        },
    });

    // =========================================================================
    // Step 6: Take square root.
    // =========================================================================
    let half = ctx.add(Expr::Div(one, two));
    let sqrt_rhs = ctx.add(Expr::Pow(step5_rhs, half));
    let abs_lhs = ctx.call_builtin(cas_ast::BuiltinFn::Abs, vec![x_plus_b_over_2a]);

    steps.push(DidacticSubstep {
        description: "Tomar raíz cuadrada en ambos lados".to_string(),
        equation_after: Equation {
            lhs: abs_lhs,
            rhs: sqrt_rhs,
            op: RelOp::Eq,
        },
    });

    // =========================================================================
    // Step 7: Split absolute-value equation and isolate x.
    // =========================================================================
    let neg_b = ctx.add(Expr::Neg(b));
    let neg_b_over_2a = ctx.add(Expr::Div(neg_b, two_a));
    let plus_minus = ctx.call("PlusMinus", vec![neg_b_over_2a, sqrt_rhs]);

    let description = if is_real_only {
        "|u| = a se descompone en u = a y u = -a. Despejando x (requiere Δ ≥ 0)".to_string()
    } else {
        "|u| = a se descompone en u = a y u = -a. Despejando x".to_string()
    };

    steps.push(DidacticSubstep {
        description,
        equation_after: Equation {
            lhs: x,
            rhs: plus_minus,
            op: RelOp::Eq,
        },
    });

    steps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_quadratic_substeps_numeric_general_case() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(2);
        let c = ctx.num(1);

        let steps = build_quadratic_substeps_with(&mut ctx, "x", a, b, c, true, |_ctx, id| {
            format!("{:?}", id)
        });

        assert_eq!(steps.len(), 7);
        assert!(steps[0].description.contains("Identificar"));
        assert!(steps[1].description.contains("Dividir"));
        assert!(steps[2].description.contains("Mover"));
        assert!(steps[3].description.contains("Completar"));
        assert!(steps[4].description.contains("cuadrado perfecto"));
        assert!(steps[5].description.contains("raíz cuadrada"));
        assert!(steps[6].description.contains("descompone"));
    }

    #[test]
    fn build_quadratic_substeps_numeric_c_zero_factoring_path() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(2);
        let c = ctx.num(0);

        let steps = build_quadratic_substeps_with(&mut ctx, "x", a, b, c, true, |_ctx, id| {
            format!("{:?}", id)
        });

        assert_eq!(steps.len(), 3);
        assert!(steps[0].description.contains("Factorizar"));
        assert!(steps[1].description.contains("Producto"));
        assert!(steps[2].description.contains("Resolver"));
    }

    #[test]
    fn quadratic_main_step_description_stays_stable() {
        assert_eq!(
            QUADRATIC_FORMULA_MAIN_STEP_DESCRIPTION,
            "Detected quadratic equation. Applying quadratic formula."
        );
    }

    #[test]
    fn zero_product_messages_format_expected_text() {
        assert_eq!(
            factorized_equation_message("x*(x-1)"),
            "Factorized equation: x*(x-1) = 0"
        );
        assert_eq!(solve_factor_message("x-1"), "Solve factor: x-1 = 0");
    }
}
