use cas_ast::{Constant, Context, Expr, ExprId};

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
}
