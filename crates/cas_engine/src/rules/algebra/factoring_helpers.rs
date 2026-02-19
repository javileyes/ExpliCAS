//! Helper functions for factoring rules.
//!
//! Contains conjugate pair detection, negation checking, and structural zero
//! verification utilities used by the factoring rules.

use cas_ast::Expr;

pub(super) use cas_math::expr_relations::conjugate_add_sub_pair as is_conjugate_pair;
pub(super) use cas_math::expr_relations::conjugate_nary_add_sub_pair as is_nary_conjugate_pair;

/// Check if an expression is structurally zero after simplification
/// This handles cases like (a-b) + (b-c) + (c-a) = 0
pub(super) fn is_structurally_zero(ctx: &mut cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    use num_traits::Zero;

    // First, simple check: is it literally 0?
    if let Expr::Number(n) = ctx.get(expr) {
        return n.is_zero();
    }

    // Flatten the sum and collect terms with signs
    // For (a-b) + (b-c) + (c-a), we expect:
    // +a, -b, +b, -c, +c, -a → all cancel
    let mut atomic_terms: std::collections::HashMap<String, i32> = std::collections::HashMap::new();

    fn collect_atoms(
        ctx: &cas_ast::Context,
        expr: cas_ast::ExprId,
        sign: i32,
        atoms: &mut std::collections::HashMap<String, i32>,
    ) {
        match ctx.get(expr) {
            Expr::Add(l, r) => {
                let (l, r) = (*l, *r);
                collect_atoms(ctx, l, sign, atoms);
                collect_atoms(ctx, r, sign, atoms);
            }
            Expr::Sub(l, r) => {
                let (l, r) = (*l, *r);
                collect_atoms(ctx, l, sign, atoms);
                collect_atoms(ctx, r, -sign, atoms);
            }
            Expr::Neg(inner) => {
                collect_atoms(ctx, *inner, -sign, atoms);
            }
            _ => {
                // Use display string as key for structural comparison
                let key = format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: ctx,
                        id: expr
                    }
                );
                *atoms.entry(key).or_insert(0) += sign;
            }
        }
    }

    collect_atoms(ctx, expr, 1, &mut atomic_terms);

    // Check if all coefficients are zero
    atomic_terms.values().all(|&coef| coef == 0)
}
