use cas_ast::{Context, Expr, ExprId, RelOp, SolutionSet};

/// Classification of a variable-free equation residual `diff = lhs - rhs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarFreeDiffKind {
    /// `diff == 0`: identity (`0 = 0`)
    IdentityZero,
    /// `diff != 0` as a concrete number: contradiction (`c = 0`)
    ContradictionNonZero,
    /// Non-numeric residual over other symbols: constraint on parameters
    Constraint,
}

/// Classify the simplified variable-free residual.
pub fn classify_var_free_difference(ctx: &Context, diff: ExprId) -> VarFreeDiffKind {
    match ctx.get(diff) {
        Expr::Number(n) if *n == num_rational::BigRational::from_integer(0.into()) => {
            VarFreeDiffKind::IdentityZero
        }
        Expr::Number(_) => VarFreeDiffKind::ContradictionNonZero,
        _ => VarFreeDiffKind::Constraint,
    }
}

/// Solve outcome for `B^E op RHS` when `E` is even and `RHS` is proven negative.
pub fn even_power_negative_rhs_outcome(op: RelOp) -> SolutionSet {
    match op {
        RelOp::Eq => SolutionSet::Empty,
        RelOp::Gt | RelOp::Geq | RelOp::Neq => SolutionSet::AllReals,
        RelOp::Lt | RelOp::Leq => SolutionSet::Empty,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_zero_number() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        assert_eq!(
            classify_var_free_difference(&ctx, zero),
            VarFreeDiffKind::IdentityZero
        );
    }

    #[test]
    fn classify_nonzero_number() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        assert_eq!(
            classify_var_free_difference(&ctx, two),
            VarFreeDiffKind::ContradictionNonZero
        );
    }

    #[test]
    fn classify_symbolic_constraint() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        assert_eq!(
            classify_var_free_difference(&ctx, y),
            VarFreeDiffKind::Constraint
        );
    }

    #[test]
    fn even_power_negative_rhs_outcome_for_eq_is_empty() {
        assert!(matches!(
            even_power_negative_rhs_outcome(RelOp::Eq),
            SolutionSet::Empty
        ));
    }

    #[test]
    fn even_power_negative_rhs_outcome_for_neq_is_all_reals() {
        assert!(matches!(
            even_power_negative_rhs_outcome(RelOp::Neq),
            SolutionSet::AllReals
        ));
    }
}
