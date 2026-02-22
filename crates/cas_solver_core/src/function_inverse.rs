use cas_ast::{Constant, Context, Equation, Expr, ExprId, RelOp};

/// Supported unary function inversion kinds used by equation isolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryInverseKind {
    Sqrt,
    Ln,
    Exp,
    Sin,
    Cos,
    Tan,
}

/// Concrete rewrite plan for unary inverse isolation.
#[derive(Debug, Clone, PartialEq)]
pub struct UnaryInverseRewritePlan {
    pub equation: Equation,
    pub step_description: &'static str,
    pub needs_rhs_cleanup: bool,
}

impl UnaryInverseKind {
    /// Classify a unary function name into an inversion kind.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "sqrt" => Some(Self::Sqrt),
            "ln" => Some(Self::Ln),
            "exp" => Some(Self::Exp),
            "sin" => Some(Self::Sin),
            "cos" => Some(Self::Cos),
            "tan" => Some(Self::Tan),
            _ => None,
        }
    }

    /// Build the transformed RHS when inverting `f(arg) = rhs`.
    pub fn build_rhs(self, ctx: &mut Context, rhs: ExprId) -> ExprId {
        match self {
            Self::Sqrt => {
                let two = ctx.num(2);
                ctx.add(Expr::Pow(rhs, two))
            }
            Self::Ln => {
                let e = ctx.add(Expr::Constant(Constant::E));
                ctx.add(Expr::Pow(e, rhs))
            }
            Self::Exp => ctx.call("ln", vec![rhs]),
            Self::Sin => ctx.call("arcsin", vec![rhs]),
            Self::Cos => ctx.call("arccos", vec![rhs]),
            Self::Tan => ctx.call("arctan", vec![rhs]),
        }
    }

    /// Whether the inverted RHS should be simplified before recursive isolation.
    ///
    /// Inverse trig functions often produce rewrite-friendly forms.
    pub fn needs_rhs_cleanup(self) -> bool {
        matches!(self, Self::Sin | Self::Cos | Self::Tan)
    }

    /// Human-readable action used by solver step narration.
    pub fn step_description(self) -> &'static str {
        match self {
            Self::Ln => "Exponentiate both sides with base e",
            Self::Exp => "Take natural log of both sides",
            Self::Sqrt => "Square both sides",
            Self::Sin => "Take arcsin of both sides",
            Self::Cos => "Take arccos of both sides",
            Self::Tan => "Take arctan of both sides",
        }
    }

    /// Whether this inverse is allowed in `UnwrapStrategy`.
    pub fn supports_unwrap_strategy(self) -> bool {
        matches!(self, Self::Sqrt | Self::Ln | Self::Exp)
    }

    /// Step text used by `UnwrapStrategy`.
    pub fn unwrap_step_description(self) -> Option<&'static str> {
        match self {
            Self::Sqrt => Some("Square both sides"),
            Self::Ln => Some("Exponentiate (base e)"),
            Self::Exp => Some("Take natural log"),
            _ => None,
        }
    }
}

/// Rewrite a unary-function equation by applying its inverse:
/// `f(arg) op other` -> `arg op inverse_f(other)` (when `is_lhs=true`),
/// or symmetrically when the function is on RHS.
pub fn rewrite_unary_inverse_equation(
    ctx: &mut Context,
    fn_name: &str,
    arg: ExprId,
    other: ExprId,
    op: RelOp,
    is_lhs: bool,
) -> Option<(Equation, UnaryInverseKind)> {
    let inverse_kind = UnaryInverseKind::from_name(fn_name)?;
    let transformed_other = inverse_kind.build_rhs(ctx, other);
    let equation = if is_lhs {
        Equation {
            lhs: arg,
            rhs: transformed_other,
            op,
        }
    } else {
        Equation {
            lhs: transformed_other,
            rhs: arg,
            op,
        }
    };
    Some((equation, inverse_kind))
}

/// Same as [`rewrite_unary_inverse_equation`] but restricted to unwrap-safe inverses.
pub fn rewrite_unary_inverse_equation_for_unwrap(
    ctx: &mut Context,
    fn_name: &str,
    arg: ExprId,
    other: ExprId,
    op: RelOp,
    is_lhs: bool,
) -> Option<(Equation, UnaryInverseKind)> {
    let (equation, inverse_kind) =
        rewrite_unary_inverse_equation(ctx, fn_name, arg, other, op, is_lhs)?;
    if inverse_kind.supports_unwrap_strategy() {
        Some((equation, inverse_kind))
    } else {
        None
    }
}

/// Build an unwrap-safe unary inverse rewrite plan.
///
/// Returns `None` when the function is unsupported by `UnwrapStrategy`.
pub fn plan_unary_inverse_rewrite_for_unwrap(
    ctx: &mut Context,
    fn_name: &str,
    arg: ExprId,
    other: ExprId,
    op: RelOp,
    is_lhs: bool,
) -> Option<(Equation, &'static str)> {
    let (equation, inverse_kind) =
        rewrite_unary_inverse_equation_for_unwrap(ctx, fn_name, arg, other, op, is_lhs)?;
    let description = inverse_kind
        .unwrap_step_description()
        .expect("unwrap filter guarantees supported inverse description");
    Some((equation, description))
}

/// Build a full unary inverse rewrite plan with narration metadata.
pub fn plan_unary_inverse_rewrite(
    ctx: &mut Context,
    fn_name: &str,
    arg: ExprId,
    other: ExprId,
    op: RelOp,
    is_lhs: bool,
) -> Option<UnaryInverseRewritePlan> {
    let (equation, inverse_kind) =
        rewrite_unary_inverse_equation(ctx, fn_name, arg, other, op, is_lhs)?;
    Some(UnaryInverseRewritePlan {
        equation,
        step_description: inverse_kind.step_description(),
        needs_rhs_cleanup: inverse_kind.needs_rhs_cleanup(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_supported_names() {
        assert_eq!(
            UnaryInverseKind::from_name("sqrt"),
            Some(UnaryInverseKind::Sqrt)
        );
        assert_eq!(
            UnaryInverseKind::from_name("ln"),
            Some(UnaryInverseKind::Ln)
        );
        assert_eq!(
            UnaryInverseKind::from_name("exp"),
            Some(UnaryInverseKind::Exp)
        );
        assert_eq!(
            UnaryInverseKind::from_name("sin"),
            Some(UnaryInverseKind::Sin)
        );
        assert_eq!(
            UnaryInverseKind::from_name("cos"),
            Some(UnaryInverseKind::Cos)
        );
        assert_eq!(
            UnaryInverseKind::from_name("tan"),
            Some(UnaryInverseKind::Tan)
        );
        assert_eq!(UnaryInverseKind::from_name("unknown"), None);
    }

    #[test]
    fn build_ln_inverse_rhs_is_e_pow_rhs() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        let rhs = UnaryInverseKind::Ln.build_rhs(&mut ctx, y);
        match ctx.get(rhs) {
            Expr::Pow(base, exp) => {
                assert_eq!(*exp, y);
                assert!(matches!(ctx.get(*base), Expr::Constant(Constant::E)));
            }
            other => panic!("Expected Pow(e, y), got {:?}", other),
        }
    }

    #[test]
    fn trig_inverses_require_cleanup() {
        assert!(UnaryInverseKind::Sin.needs_rhs_cleanup());
        assert!(UnaryInverseKind::Cos.needs_rhs_cleanup());
        assert!(UnaryInverseKind::Tan.needs_rhs_cleanup());
        assert!(!UnaryInverseKind::Ln.needs_rhs_cleanup());
        assert!(!UnaryInverseKind::Exp.needs_rhs_cleanup());
        assert!(!UnaryInverseKind::Sqrt.needs_rhs_cleanup());
    }

    #[test]
    fn step_descriptions_match_expected_wording() {
        assert_eq!(
            UnaryInverseKind::Ln.step_description(),
            "Exponentiate both sides with base e"
        );
        assert_eq!(
            UnaryInverseKind::Exp.step_description(),
            "Take natural log of both sides"
        );
        assert_eq!(
            UnaryInverseKind::Sqrt.step_description(),
            "Square both sides"
        );
        assert_eq!(
            UnaryInverseKind::Sin.step_description(),
            "Take arcsin of both sides"
        );
        assert_eq!(
            UnaryInverseKind::Cos.step_description(),
            "Take arccos of both sides"
        );
        assert_eq!(
            UnaryInverseKind::Tan.step_description(),
            "Take arctan of both sides"
        );
    }

    #[test]
    fn unwrap_strategy_support_is_restricted_to_ln_exp_sqrt() {
        assert!(UnaryInverseKind::Ln.supports_unwrap_strategy());
        assert!(UnaryInverseKind::Exp.supports_unwrap_strategy());
        assert!(UnaryInverseKind::Sqrt.supports_unwrap_strategy());
        assert!(!UnaryInverseKind::Sin.supports_unwrap_strategy());
        assert!(!UnaryInverseKind::Cos.supports_unwrap_strategy());
        assert!(!UnaryInverseKind::Tan.supports_unwrap_strategy());
    }

    #[test]
    fn unwrap_step_descriptions_match_expected_wording() {
        assert_eq!(
            UnaryInverseKind::Ln.unwrap_step_description(),
            Some("Exponentiate (base e)")
        );
        assert_eq!(
            UnaryInverseKind::Exp.unwrap_step_description(),
            Some("Take natural log")
        );
        assert_eq!(
            UnaryInverseKind::Sqrt.unwrap_step_description(),
            Some("Square both sides")
        );
        assert_eq!(UnaryInverseKind::Sin.unwrap_step_description(), None);
        assert_eq!(UnaryInverseKind::Cos.unwrap_step_description(), None);
        assert_eq!(UnaryInverseKind::Tan.unwrap_step_description(), None);
    }

    #[test]
    fn rewrite_unary_inverse_equation_lhs_orientation() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let (eq, kind) = rewrite_unary_inverse_equation(&mut ctx, "sqrt", x, y, RelOp::Eq, true)
            .expect("sqrt inverse should rewrite");
        assert_eq!(kind, UnaryInverseKind::Sqrt);
        assert_eq!(eq.lhs, x);
        assert!(matches!(ctx.get(eq.rhs), Expr::Pow(base, _) if *base == y));
    }

    #[test]
    fn rewrite_unary_inverse_equation_rhs_orientation() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let (eq, kind) = rewrite_unary_inverse_equation(&mut ctx, "exp", x, y, RelOp::Eq, false)
            .expect("exp inverse should rewrite");
        assert_eq!(kind, UnaryInverseKind::Exp);
        assert_eq!(eq.rhs, x);
        assert!(
            matches!(ctx.get(eq.lhs), Expr::Function(_, args) if args.len() == 1 && args[0] == y)
        );
    }

    #[test]
    fn rewrite_unary_inverse_equation_for_unwrap_rejects_trig() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        assert!(
            rewrite_unary_inverse_equation_for_unwrap(&mut ctx, "sin", x, y, RelOp::Eq, true)
                .is_none()
        );
    }

    #[test]
    fn plan_unary_inverse_rewrite_carries_step_metadata() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let plan = plan_unary_inverse_rewrite(&mut ctx, "exp", x, y, RelOp::Eq, true)
            .expect("exp inverse should build plan");
        assert_eq!(plan.step_description, "Take natural log of both sides");
        assert!(!plan.needs_rhs_cleanup);
        assert_eq!(plan.equation.lhs, x);
    }

    #[test]
    fn plan_unary_inverse_rewrite_for_unwrap_returns_description() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let (eq, desc) =
            plan_unary_inverse_rewrite_for_unwrap(&mut ctx, "ln", x, y, RelOp::Eq, true)
                .expect("ln should be unwrap-safe");
        assert_eq!(desc, "Exponentiate (base e)");
        assert_eq!(eq.lhs, x);
    }

    #[test]
    fn plan_unary_inverse_rewrite_for_unwrap_rejects_trig() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        assert!(
            plan_unary_inverse_rewrite_for_unwrap(&mut ctx, "sin", x, y, RelOp::Eq, true).is_none()
        );
    }
}
