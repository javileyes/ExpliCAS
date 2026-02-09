// Quadratic Formula Didactic Substeps
//
// This module generates step-by-step derivation of the quadratic formula
// using the "completing the square" method for educational purposes.

use cas_ast::{Context, Equation, Expr, ExprId};

use crate::engine::Simplifier;
use crate::solver::SolveSubStep;
use crate::step::ImportanceLevel;

// Note: Simplification of substep expressions (e.g., c/a when c=0) is deferred
// to future work due to borrow checker complexity with nested simplifier calls.

/// Check if an expression is a literal zero (Number(0))
fn is_zero_literal(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if *n == num_rational::BigRational::from_integer(0.into()))
}

/// Check if an expression is a numeric constant (Number)
fn is_numeric(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(_))
}

/// Apply didactic simplification to all substeps (post-pass).
/// This runs AFTER construction to avoid borrow conflicts.
/// Only applies "safe" simplifications: 0+x→x, 1*x→x, 0*x→0, x/1→x, etc.
pub(crate) fn didactic_simplify_substeps(
    simplifier: &mut Simplifier,
    substeps: &mut [SolveSubStep],
) {
    // Disable step collection during simplification
    let was_collecting = simplifier.collect_steps();
    simplifier.set_collect_steps(false);

    for substep in substeps.iter_mut() {
        // Simplify both LHS and RHS of equation_after
        let (simplified_lhs, _) = simplifier.simplify(substep.equation_after.lhs);
        let (simplified_rhs, _) = simplifier.simplify(substep.equation_after.rhs);

        substep.equation_after.lhs = simplified_lhs;
        substep.equation_after.rhs = simplified_rhs;
    }

    // Restore step collection setting
    simplifier.set_collect_steps(was_collecting);
}

/// Build the didactic substeps for solving a quadratic equation.
///
/// Given ax² + bx + c = 0 with known coefficients a, b, c (ExprIds),
/// generates 6 substeps showing the completing-the-square derivation:
///
/// 1. Identify coefficients a, b, c
/// 2. Divide both sides by a
/// 3. Move constant term to RHS
/// 4. Complete the square (add (b/2a)² to both sides)
/// 5. Factor LHS as perfect square
/// 6. Take square root and isolate x
pub(crate) fn build_quadratic_substeps(
    simplifier: &mut Simplifier,
    var: &str,
    a: ExprId,
    b: ExprId,
    c: ExprId,
    is_real_only: bool,
) -> Vec<SolveSubStep> {
    use cas_ast::DisplayExpr;

    let ctx = &mut simplifier.context;
    let mut steps = Vec::new();
    let zero = ctx.num(0);
    let one = ctx.num(1);
    let two = ctx.num(2);
    let four = ctx.num(4);

    // Check if all coefficients are numeric - if so, don't show algebraic derivation
    let all_numeric = is_numeric(ctx, a) && is_numeric(ctx, b) && is_numeric(ctx, c);

    // Peephole optimization: check if c is literally zero
    let c_is_zero = is_zero_literal(ctx, c);

    // Helper for display
    let disp =
        |ctx: &Context, id: ExprId| -> String { format!("{}", DisplayExpr { context: ctx, id }) };

    // Build x variable
    let x = ctx.var(var);

    // Build x²
    let x2 = ctx.add(Expr::Pow(x, two));

    // =========================================================================
    // NUMERIC CASE: When a, b, c are all constants, show natural algebraic steps
    // Instead of abstract discriminant, use intuitive operations
    // =========================================================================
    if all_numeric {
        // Build the polynomial expression for display
        let ax2 = ctx.add(Expr::Mul(a, x2));
        let bx = ctx.add(Expr::Mul(b, x));
        let ax2_plus_bx = ctx.add(Expr::Add(ax2, bx));
        let _poly = ctx.add(Expr::Add(ax2_plus_bx, c));

        // CASE 1: c = 0 → Factor out x
        // x(ax + b) = 0 → x = 0 or ax + b = 0 → x = -b/a
        if c_is_zero {
            // Build x(ax + b) = 0
            let ax = ctx.add(Expr::Mul(a, x));
            let inner = ctx.add(Expr::Add(ax, b)); // ax + b
            let factored_form = ctx.add(Expr::Mul(x, inner)); // x(ax + b)

            steps.push(SolveSubStep {
                description: "Factorizar x común".to_string(),
                equation_after: Equation {
                    lhs: factored_form,
                    rhs: zero,
                    op: cas_ast::RelOp::Eq,
                },
                importance: ImportanceLevel::Low,
            });

            // Step 2: Product = 0 means one factor = 0
            steps.push(SolveSubStep {
                description: "Producto igual a cero: algún factor es cero".to_string(),
                equation_after: Equation {
                    lhs: x,
                    rhs: zero,
                    op: cas_ast::RelOp::Eq,
                },
                importance: ImportanceLevel::Low,
            });

            // Step 3: Other factor = 0 → ax + b = 0 → x = -b/a
            let neg_b = ctx.add(Expr::Neg(b));
            let sol2 = ctx.add(Expr::Div(neg_b, a));

            steps.push(SolveSubStep {
                description: format!("Resolver {} = 0", disp(ctx, inner)),
                equation_after: Equation {
                    lhs: x,
                    rhs: sol2,
                    op: cas_ast::RelOp::Eq,
                },
                importance: ImportanceLevel::Low,
            });

            return steps;
        }
        // General numeric case (c ≠ 0): fall through to full 6-step symbolic derivation
    }

    // =========================================================================
    // Step 1: Identify the equation form
    // ax² + bx + c = 0 with a, b, c values
    // =========================================================================

    // Build the standard form: ax² + bx + c
    let ax2 = ctx.add(Expr::Mul(a, x2));
    let bx = ctx.add(Expr::Mul(b, x));
    let ax2_plus_bx = ctx.add(Expr::Add(ax2, bx));
    let poly = ctx.add(Expr::Add(ax2_plus_bx, c));

    steps.push(SolveSubStep {
        description: format!(
            "Identificar forma cuadrática: a·x² + b·x + c = 0 con a = {}, b = {}, c = {}",
            disp(ctx, a),
            disp(ctx, b),
            disp(ctx, c)
        ),
        equation_after: Equation {
            lhs: poly,
            rhs: zero,
            op: cas_ast::RelOp::Eq,
        },
        importance: ImportanceLevel::Low,
    });

    // =========================================================================
    // Step 2: Divide both sides by a
    // x² + (b/a)x + c/a = 0
    // When c=0: x² + (b/a)x = 0
    // =========================================================================

    let b_over_a = ctx.add(Expr::Div(b, a));
    // c/a: when c=0, use 0 directly
    let c_over_a = if c_is_zero {
        zero
    } else {
        ctx.add(Expr::Div(c, a))
    };
    let bax = ctx.add(Expr::Mul(b_over_a, x));
    let x2_plus_bax = ctx.add(Expr::Add(x2, bax));
    // When c=0, step2_lhs is just x² + (b/a)x (skip the +0)
    let step2_lhs = if c_is_zero {
        x2_plus_bax
    } else {
        ctx.add(Expr::Add(x2_plus_bax, c_over_a))
    };

    steps.push(SolveSubStep {
        description: "Dividir ambos lados por a (requiere a ≠ 0)".to_string(),
        equation_after: Equation {
            lhs: step2_lhs,
            rhs: zero,
            op: cas_ast::RelOp::Eq,
        },
        importance: ImportanceLevel::Low,
    });

    // =========================================================================
    // Step 3: Move constant to RHS
    // x² + (b/a)x = -c/a
    // When c=0: x² + (b/a)x = 0
    // =========================================================================

    // -c/a: when c=0, RHS is just 0
    let step3_rhs = if c_is_zero {
        zero
    } else {
        ctx.add(Expr::Neg(c_over_a))
    };
    let step3_lhs = x2_plus_bax; // x² + (b/a)x

    steps.push(SolveSubStep {
        description: "Mover término constante al lado derecho".to_string(),
        equation_after: Equation {
            lhs: step3_lhs,
            rhs: step3_rhs,
            op: cas_ast::RelOp::Eq,
        },
        importance: ImportanceLevel::Low,
    });

    // =========================================================================
    // Step 4: Complete the square
    // Add (b/2a)² to both sides
    // x² + (b/a)x + (b/2a)² = (b/2a)² - c/a
    // When c=0: x² + (b/a)x + (b/2a)² = (b/2a)²
    // =========================================================================

    // (b/2a)
    let two_a = ctx.add(Expr::Mul(two, a));
    let b_over_2a = ctx.add(Expr::Div(b, two_a));

    // (b/2a)²
    let b_over_2a_sq = ctx.add(Expr::Pow(b_over_2a, two));

    // LHS: x² + (b/a)x + (b/2a)²
    let step4_lhs = ctx.add(Expr::Add(step3_lhs, b_over_2a_sq));

    // RHS: when c=0, just (b/2a)², otherwise (b/2a)² - c/a
    let step4_rhs = if c_is_zero {
        b_over_2a_sq
    } else {
        let neg_c_a = ctx.add(Expr::Neg(c_over_a));
        ctx.add(Expr::Add(b_over_2a_sq, neg_c_a))
    };

    steps.push(SolveSubStep {
        description: "Completar el cuadrado: sumar (b/2a)² a ambos lados".to_string(),
        equation_after: Equation {
            lhs: step4_lhs,
            rhs: step4_rhs,
            op: cas_ast::RelOp::Eq,
        },
        importance: ImportanceLevel::Low,
    });

    // =========================================================================
    // Step 5: Factor LHS as perfect square
    // (x + b/2a)² = (b² - 4ac) / 4a²
    // When c=0: (x + b/2a)² = b² / 4a²
    // =========================================================================

    // LHS: (x + b/2a)²
    let x_plus_b_over_2a = ctx.add(Expr::Add(x, b_over_2a));
    let step5_lhs = ctx.add(Expr::Pow(x_plus_b_over_2a, two));

    // RHS: (b² - 4ac) / 4a²
    // When c=0: b² / 4a² (discriminant = b²)
    let b2 = ctx.add(Expr::Pow(b, two));
    let discriminant = if c_is_zero {
        b2 // Just b² when c=0
    } else {
        let four_a = ctx.add(Expr::Mul(four, a));
        let four_ac = ctx.add(Expr::Mul(four_a, c));
        ctx.add(Expr::Sub(b2, four_ac)) // b² - 4ac
    };
    let a2 = ctx.add(Expr::Pow(a, two));
    let four_a2 = ctx.add(Expr::Mul(four, a2));
    let step5_rhs = ctx.add(Expr::Div(discriminant, four_a2));

    steps.push(SolveSubStep {
        description: "Escribir lado izquierdo como cuadrado perfecto".to_string(),
        equation_after: Equation {
            lhs: step5_lhs,
            rhs: step5_rhs,
            op: cas_ast::RelOp::Eq,
        },
        importance: ImportanceLevel::Low,
    });

    // =========================================================================
    // Step 6: Take square root → |x + b/2a| = √(RHS)
    // =========================================================================

    let half = ctx.add(Expr::Div(one, two));
    let sqrt_rhs = ctx.add(Expr::Pow(step5_rhs, half));
    // |x + b/2a|
    let abs_lhs = ctx.call_builtin(cas_ast::BuiltinFn::Abs, vec![x_plus_b_over_2a]);

    steps.push(SolveSubStep {
        description: "Tomar raíz cuadrada en ambos lados".to_string(),
        equation_after: Equation {
            lhs: abs_lhs,
            rhs: sqrt_rhs,
            op: cas_ast::RelOp::Eq,
        },
        importance: ImportanceLevel::Low,
    });

    // =========================================================================
    // Step 7: |u| = a se descompone en u = a y u = -a → x = -b/2a ± √(...)
    // For numeric case: show the actual computed solutions
    // =========================================================================

    // Use the same sqrt_rhs from step 6 for consistency
    // x = -b/2a ± sqrt_rhs (where sqrt_rhs = √((b²-4ac)/4a²) = √Δ/2a)
    let neg_b = ctx.add(Expr::Neg(b));
    let neg_b_over_2a = ctx.add(Expr::Div(neg_b, two_a));

    // For both numeric and symbolic: show formula with ± notation
    // x = -b/2a ± √((b²-4ac)/4a²)
    // The post-pass simplifier will clean up numeric values
    let plus_minus = ctx.call("PlusMinus", vec![neg_b_over_2a, sqrt_rhs]);
    // No need to divide by 2a again - it's already incorporated in both terms

    let description = if is_real_only {
        "|u| = a se descompone en u = a y u = -a. Despejando x (requiere Δ ≥ 0)".to_string()
    } else {
        "|u| = a se descompone en u = a y u = -a. Despejando x".to_string()
    };

    // Show x = (-b ± √Δ) / 2a
    steps.push(SolveSubStep {
        description,
        equation_after: Equation {
            lhs: x,
            rhs: plus_minus,
            op: cas_ast::RelOp::Eq,
        },
        importance: ImportanceLevel::Low,
    });

    steps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Simplifier;

    #[test]
    fn test_build_quadratic_substeps() {
        let mut simplifier = Simplifier::with_default_rules();

        // Test with numeric coefficients: x² + 2x + 1 = 0
        // General numeric case (c≠0) uses full 6-step derivation
        let a = simplifier.context.num(1);
        let b = simplifier.context.num(2);
        let c = simplifier.context.num(1);

        let steps = build_quadratic_substeps(&mut simplifier, "x", a, b, c, true);

        assert_eq!(
            steps.len(),
            7,
            "General numeric case should generate 7 substeps"
        );
        assert!(steps[0].description.contains("Identificar"));
        assert!(steps[1].description.contains("Dividir"));
        assert!(steps[2].description.contains("Mover"));
        assert!(steps[3].description.contains("Completar"));
        assert!(steps[4].description.contains("cuadrado perfecto"));
        assert!(steps[5].description.contains("raíz cuadrada"));
        assert!(steps[6].description.contains("descompone"));
    }
}
