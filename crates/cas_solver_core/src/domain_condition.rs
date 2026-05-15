//! Implicit-domain condition model shared by engine/solver layers.
//!
//! This module owns:
//! - condition vocabulary (`ImplicitCondition`)
//! - display policy (`RequiresDisplayLevel`)
//! - inferred condition set container (`ImplicitDomain`)

use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_extract::extract_abs_argument_view;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::collections::HashSet;

/// An implicit condition inferred from expression structure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ImplicitCondition {
    /// x ≥ 0 (from sqrt(x) or x^(1/2))
    NonNegative(ExprId),
    /// x ≥ c, for real-domain lower bounds such as acosh(x), where c = 1.
    LowerBound(ExprId, BigRational),
    /// x > 0 (from ln(x) or log(x))
    Positive(ExprId),
    /// x ≠ 0 (from 1/x or Div(_, x))
    NonZero(ExprId),
}

impl ImplicitCondition {
    /// Human-readable display for REPL/UI.
    pub fn display(&self, ctx: &Context) -> String {
        use cas_formatter::DisplayExpr;

        match self {
            Self::NonNegative(e) => {
                if let Some(bound) = display_unit_interval_nonnegative(ctx, *e) {
                    bound
                } else if let Some(bound) = display_quadratic_interval_nonnegative(ctx, *e) {
                    bound
                } else if let Some(bound) =
                    display_affine_lower_bound(ctx, *e, BigRational::zero(), false)
                {
                    bound
                } else {
                    format!(
                        "{} ≥ 0",
                        DisplayExpr {
                            context: ctx,
                            id: *e
                        }
                    )
                }
            }
            Self::LowerBound(e, lower) => display_affine_lower_bound(ctx, *e, lower.clone(), false)
                .unwrap_or_else(|| {
                    format!(
                        "{} ≥ {}",
                        DisplayExpr {
                            context: ctx,
                            id: *e
                        },
                        lower
                    )
                }),
            Self::Positive(e) => {
                if let Some(arg) = extract_abs_argument_view(ctx, *e) {
                    format!(
                        "{} ≠ 0",
                        DisplayExpr {
                            context: ctx,
                            id: arg
                        }
                    )
                } else if let Some(bound) = display_quadratic_interval_positive(ctx, *e) {
                    bound
                } else if let Some(bound) =
                    display_affine_lower_bound(ctx, *e, BigRational::zero(), true)
                {
                    bound
                } else {
                    format!(
                        "{} > 0",
                        DisplayExpr {
                            context: ctx,
                            id: *e
                        }
                    )
                }
            }
            Self::NonZero(e) => {
                let display_expr = extract_abs_argument_view(ctx, *e).unwrap_or(*e);
                display_affine_nonzero(ctx, display_expr).unwrap_or_else(|| {
                    format!(
                        "{} ≠ 0",
                        DisplayExpr {
                            context: ctx,
                            id: display_expr
                        }
                    )
                })
            }
        }
    }

    /// Check if this condition is trivial (always true or on a constant expression).
    pub fn is_trivial(&self, ctx: &Context) -> bool {
        let expr = match self {
            Self::NonNegative(e)
            | Self::LowerBound(e, _)
            | Self::Positive(e)
            | Self::NonZero(e) => *e,
        };

        if matches!(self, Self::NonZero(_))
            && cas_math::prove_nonzero::prove_nonzero_depth_with(
                ctx,
                expr,
                crate::predicate_proofs::DEFAULT_PROOF_DEPTH,
                |_ctx, _expr| cas_math::tri_proof::TriProof::Unknown,
                |_ctx, _expr| None,
            )
            .is_proven()
        {
            return true;
        }

        if matches!(self, Self::NonZero(_))
            && cas_math::prove_sign::prove_positive_depth_with(
                ctx,
                expr,
                crate::predicate_proofs::DEFAULT_PROOF_DEPTH,
                true,
                |_ctx, _expr, _depth| cas_math::tri_proof::TriProof::Unknown,
            )
            .is_proven()
        {
            return true;
        }

        if matches!(self, Self::Positive(_))
            && cas_math::prove_sign::prove_positive_depth_with(
                ctx,
                expr,
                crate::predicate_proofs::DEFAULT_PROOF_DEPTH,
                true,
                |_ctx, _expr, _depth| cas_math::tri_proof::TriProof::Unknown,
            )
            .is_proven()
        {
            return true;
        }

        if matches!(self, Self::NonNegative(_))
            && cas_math::prove_sign::prove_nonnegative_depth_with(
                ctx,
                expr,
                crate::predicate_proofs::DEFAULT_PROOF_DEPTH,
                true,
                |_ctx, _expr, _depth| cas_math::tri_proof::TriProof::Unknown,
            )
            .is_proven()
        {
            return true;
        }

        if let Self::LowerBound(_, lower) = self {
            if let Expr::Number(n) = ctx.get(expr) {
                return n >= lower;
            }
        }

        // Fully numeric expressions are trivial in this context.
        if !cas_math::expr_predicates::contains_variable(ctx, expr) {
            return true;
        }

        // x^2 >= 0-like predicates are always true and not useful to display.
        if let Self::NonNegative(e) = self {
            if cas_math::expr_predicates::is_always_nonnegative_expr(ctx, *e) {
                return true;
            }
        }

        false
    }

    /// Check if this condition's witness survives in the output expression.
    pub fn witness_survives_in(&self, ctx: &Context, output: ExprId) -> bool {
        use cas_math::expr_witness::{witness_survives, WitnessKind};

        match self {
            Self::NonNegative(e) => witness_survives(ctx, *e, output, WitnessKind::Sqrt),
            Self::LowerBound(_, _) => false,
            Self::Positive(e) => witness_survives(ctx, *e, output, WitnessKind::Log),
            Self::NonZero(e) => witness_survives(ctx, *e, output, WitnessKind::Division),
        }
    }
}

fn display_unit_interval_nonnegative(ctx: &Context, expr: ExprId) -> Option<String> {
    use cas_formatter::DisplayExpr;

    let base = unit_interval_base(ctx, expr)?;
    if !cas_math::expr_predicates::contains_variable(ctx, base) {
        return None;
    }

    if let Some(denominator) = reciprocal_denominator(ctx, base) {
        if let Some(bound) = display_affine_exterior_interval(ctx, denominator) {
            return Some(bound);
        }

        return Some(format!(
            "{} ≤ -1 or {} ≥ 1",
            DisplayExpr {
                context: ctx,
                id: denominator
            },
            DisplayExpr {
                context: ctx,
                id: denominator
            }
        ));
    }

    Some(format!(
        "-1 ≤ {} ≤ 1",
        DisplayExpr {
            context: ctx,
            id: base
        }
    ))
}

#[derive(Clone)]
struct AffineForm {
    variable: Option<usize>,
    slope: BigRational,
    intercept: BigRational,
}

impl AffineForm {
    fn constant(value: BigRational) -> Self {
        Self {
            variable: None,
            slope: BigRational::zero(),
            intercept: value,
        }
    }

    fn variable(variable: usize) -> Self {
        Self {
            variable: Some(variable),
            slope: BigRational::one(),
            intercept: BigRational::zero(),
        }
    }

    fn is_constant(&self) -> bool {
        self.variable.is_none() && self.slope.is_zero()
    }

    fn scale(mut self, factor: BigRational) -> Self {
        self.slope *= factor.clone();
        self.intercept *= factor;
        if self.slope.is_zero() {
            self.variable = None;
        }
        self
    }

    fn add(self, other: Self) -> Option<Self> {
        let variable = match (self.variable, other.variable) {
            (None, variable) | (variable, None) => variable,
            (Some(left), Some(right)) if left == right => Some(left),
            _ => return None,
        };

        let slope = self.slope + other.slope;
        Some(Self {
            variable: if slope.is_zero() { None } else { variable },
            slope,
            intercept: self.intercept + other.intercept,
        })
    }
}

fn display_affine_exterior_interval(ctx: &Context, expr: ExprId) -> Option<String> {
    let affine = affine_form(ctx, expr)?;
    let variable = affine.variable?;
    if affine.slope.is_zero() {
        return None;
    }

    let negative_one = BigRational::from_integer((-1).into());
    let one = BigRational::one();
    let left_endpoint = (negative_one - affine.intercept.clone()) / affine.slope.clone();
    let right_endpoint = (one - affine.intercept) / affine.slope;
    let (lower, upper) = if left_endpoint <= right_endpoint {
        (left_endpoint, right_endpoint)
    } else {
        (right_endpoint, left_endpoint)
    };
    let variable_name = ctx.sym_name(variable);

    Some(format!(
        "{} ≤ {} or {} ≥ {}",
        variable_name,
        display_rational_bound(&lower),
        variable_name,
        display_rational_bound(&upper)
    ))
}

fn display_affine_lower_bound(
    ctx: &Context,
    expr: ExprId,
    lower: BigRational,
    strict: bool,
) -> Option<String> {
    let affine = affine_form(ctx, expr)?;
    let variable = affine.variable?;
    if affine.slope.is_zero() {
        return None;
    }

    let bound = (lower - affine.intercept) / affine.slope.clone();
    let variable_name = ctx.sym_name(variable);
    let bound = display_rational_bound(&bound);

    if affine.slope.is_positive() {
        Some(if strict {
            format!("{variable_name} > {bound}")
        } else {
            format!("{variable_name} ≥ {bound}")
        })
    } else {
        Some(if strict {
            format!("{variable_name} < {bound}")
        } else {
            format!("{variable_name} ≤ {bound}")
        })
    }
}

fn display_affine_nonzero(ctx: &Context, expr: ExprId) -> Option<String> {
    let affine = affine_form(ctx, expr)?;
    let variable = affine.variable?;
    if affine.slope.is_zero() {
        return None;
    }

    let bound = -affine.intercept / affine.slope;
    Some(format!(
        "{} ≠ {}",
        ctx.sym_name(variable),
        display_rational_bound(&bound)
    ))
}

#[derive(Clone)]
struct QuadraticForm {
    variable: Option<usize>,
    constant: BigRational,
    linear: BigRational,
    quadratic: BigRational,
}

impl QuadraticForm {
    fn constant(value: BigRational) -> Self {
        Self {
            variable: None,
            constant: value,
            linear: BigRational::zero(),
            quadratic: BigRational::zero(),
        }
    }

    fn variable(variable: usize) -> Self {
        Self {
            variable: Some(variable),
            constant: BigRational::zero(),
            linear: BigRational::one(),
            quadratic: BigRational::zero(),
        }
    }

    fn is_constant(&self) -> bool {
        self.variable.is_none() && self.linear.is_zero() && self.quadratic.is_zero()
    }

    fn scale(mut self, factor: BigRational) -> Self {
        self.constant *= factor.clone();
        self.linear *= factor.clone();
        self.quadratic *= factor;
        if self.linear.is_zero() && self.quadratic.is_zero() {
            self.variable = None;
        }
        self
    }

    fn add(self, other: Self) -> Option<Self> {
        let variable = combine_single_variable(self.variable, other.variable)?;
        let linear = self.linear + other.linear;
        let quadratic = self.quadratic + other.quadratic;

        Some(Self {
            variable: if linear.is_zero() && quadratic.is_zero() {
                None
            } else {
                variable
            },
            constant: self.constant + other.constant,
            linear,
            quadratic,
        })
    }

    fn mul(self, other: Self) -> Option<Self> {
        let variable = combine_single_variable(self.variable, other.variable)?;
        let zero = BigRational::zero();
        let degree_self = polynomial_degree(&[&self.constant, &self.linear, &self.quadratic]);
        let degree_other = polynomial_degree(&[&other.constant, &other.linear, &other.quadratic]);
        if degree_self + degree_other > 2 {
            return None;
        }

        let constant = self.constant.clone() * other.constant.clone();
        let linear = self.constant.clone() * other.linear.clone()
            + self.linear.clone() * other.constant.clone();
        let quadratic = self.constant * other.quadratic
            + self.linear * other.linear
            + self.quadratic * other.constant;

        Some(Self {
            variable: if linear == zero && quadratic == zero {
                None
            } else {
                variable
            },
            constant,
            linear,
            quadratic,
        })
    }
}

fn display_quadratic_interval_nonnegative(ctx: &Context, expr: ExprId) -> Option<String> {
    if let Some((variable, lower, upper)) = concave_quadratic_interval(ctx, expr) {
        let variable_name = ctx.sym_name(variable);

        return Some(format!(
            "{} ≤ {} ≤ {}",
            display_rational_bound(&lower),
            variable_name,
            display_rational_bound(&upper)
        ));
    }

    if let Some((variable, center, offset)) = concave_quadratic_surd_interval(ctx, expr) {
        let variable_name = ctx.sym_name(variable);

        return Some(format!(
            "{} ≤ {} ≤ {}",
            display_center_minus_offset(&center, &offset),
            variable_name,
            display_center_plus_offset(&center, &offset)
        ));
    }

    if let Some((variable, bound)) = symmetric_convex_quadratic_surd_bound(ctx, expr) {
        let variable_name = ctx.sym_name(variable);

        return Some(format!(
            "{} ≤ -{} or {} ≥ {}",
            variable_name, bound, variable_name, bound
        ));
    }

    if let Some((variable, center, offset)) = shifted_convex_quadratic_surd_bounds(ctx, expr) {
        let variable_name = ctx.sym_name(variable);

        return Some(format!(
            "{} ≤ {} or {} ≥ {}",
            variable_name,
            display_center_minus_offset(&center, &offset),
            variable_name,
            display_center_plus_offset(&center, &offset)
        ));
    }

    let (variable, lower, upper) = convex_quadratic_interval(ctx, expr)?;
    if lower == upper {
        return None;
    }
    let variable_name = ctx.sym_name(variable);

    Some(format!(
        "{} ≤ {} or {} ≥ {}",
        variable_name,
        display_rational_bound(&lower),
        variable_name,
        display_rational_bound(&upper)
    ))
}

fn display_quadratic_interval_positive(ctx: &Context, expr: ExprId) -> Option<String> {
    if let Some((variable, lower, upper)) = concave_quadratic_interval(ctx, expr) {
        if lower == upper {
            return None;
        }
        let variable_name = ctx.sym_name(variable);

        return Some(format!(
            "{} < {} < {}",
            display_rational_bound(&lower),
            variable_name,
            display_rational_bound(&upper)
        ));
    }

    if let Some((variable, center, offset)) = concave_quadratic_surd_interval(ctx, expr) {
        let variable_name = ctx.sym_name(variable);

        return Some(format!(
            "{} < {} < {}",
            display_center_minus_offset(&center, &offset),
            variable_name,
            display_center_plus_offset(&center, &offset)
        ));
    }

    if let Some((variable, bound)) = symmetric_convex_quadratic_surd_bound(ctx, expr) {
        let variable_name = ctx.sym_name(variable);

        return Some(format!(
            "{} < -{} or {} > {}",
            variable_name, bound, variable_name, bound
        ));
    }

    if let Some((variable, center, offset)) = shifted_convex_quadratic_surd_bounds(ctx, expr) {
        let variable_name = ctx.sym_name(variable);

        return Some(format!(
            "{} < {} or {} > {}",
            variable_name,
            display_center_minus_offset(&center, &offset),
            variable_name,
            display_center_plus_offset(&center, &offset)
        ));
    }

    let (variable, lower, upper) = convex_quadratic_interval(ctx, expr)?;
    let variable_name = ctx.sym_name(variable);
    if lower == upper {
        return Some(format!(
            "{} ≠ {}",
            variable_name,
            display_rational_bound(&lower)
        ));
    }

    Some(format!(
        "{} < {} or {} > {}",
        variable_name,
        display_rational_bound(&lower),
        variable_name,
        display_rational_bound(&upper)
    ))
}

fn concave_quadratic_interval(
    ctx: &Context,
    expr: ExprId,
) -> Option<(usize, BigRational, BigRational)> {
    let quadratic = quadratic_form(ctx, expr)?;
    let variable = quadratic.variable?;
    if !quadratic.quadratic.is_negative() {
        return None;
    }

    let four = BigRational::from_integer(4.into());
    let two = BigRational::from_integer(2.into());
    let discriminant = quadratic.linear.clone() * quadratic.linear.clone()
        - four * quadratic.quadratic.clone() * quadratic.constant;
    let sqrt_discriminant = rational_sqrt_nonnegative(&discriminant)?;
    let denominator = two * quadratic.quadratic;
    let left = (-quadratic.linear.clone() - sqrt_discriminant.clone()) / denominator.clone();
    let right = (-quadratic.linear + sqrt_discriminant) / denominator;
    let (lower, upper) = if left <= right {
        (left, right)
    } else {
        (right, left)
    };

    Some((variable, lower, upper))
}

fn concave_quadratic_surd_interval(ctx: &Context, expr: ExprId) -> Option<(usize, String, String)> {
    let quadratic = quadratic_form(ctx, expr)?;
    let variable = quadratic.variable?;
    if !quadratic.quadratic.is_negative() {
        return None;
    }

    let four = BigRational::from_integer(4.into());
    let two = BigRational::from_integer(2.into());
    let discriminant = quadratic.linear.clone() * quadratic.linear.clone()
        - four.clone() * quadratic.quadratic.clone() * quadratic.constant;
    if !discriminant.is_positive() || rational_sqrt_nonnegative(&discriminant).is_some() {
        return None;
    }

    let center = -quadratic.linear / (two * quadratic.quadratic.clone());
    let offset_radicand = discriminant / (four * quadratic.quadratic.clone() * quadratic.quadratic);

    Some((
        variable,
        display_rational_bound(&center),
        display_sqrt_bound(&offset_radicand)?,
    ))
}

fn symmetric_convex_quadratic_surd_bound(ctx: &Context, expr: ExprId) -> Option<(usize, String)> {
    let quadratic = quadratic_form(ctx, expr)?;
    let variable = quadratic.variable?;
    if !quadratic.quadratic.is_positive()
        || !quadratic.linear.is_zero()
        || !quadratic.constant.is_negative()
    {
        return None;
    }

    let radicand = -quadratic.constant / quadratic.quadratic;
    if rational_sqrt_nonnegative(&radicand).is_some() {
        return None;
    }

    Some((variable, display_sqrt_bound(&radicand)?))
}

fn shifted_convex_quadratic_surd_bounds(
    ctx: &Context,
    expr: ExprId,
) -> Option<(usize, String, String)> {
    let quadratic = quadratic_form(ctx, expr)?;
    let variable = quadratic.variable?;
    if !quadratic.quadratic.is_positive() || quadratic.linear.is_zero() {
        return None;
    }

    let four = BigRational::from_integer(4.into());
    let two = BigRational::from_integer(2.into());
    let discriminant = quadratic.linear.clone() * quadratic.linear.clone()
        - four.clone() * quadratic.quadratic.clone() * quadratic.constant;
    if !discriminant.is_positive() || rational_sqrt_nonnegative(&discriminant).is_some() {
        return None;
    }

    let center = -quadratic.linear / (two * quadratic.quadratic.clone());
    let offset_radicand = discriminant / (four * quadratic.quadratic.clone() * quadratic.quadratic);

    Some((
        variable,
        display_rational_bound(&center),
        display_sqrt_bound(&offset_radicand)?,
    ))
}

fn convex_quadratic_interval(
    ctx: &Context,
    expr: ExprId,
) -> Option<(usize, BigRational, BigRational)> {
    let quadratic = quadratic_form(ctx, expr)?;
    if !quadratic.quadratic.is_positive() {
        return None;
    }

    quadratic_rational_roots(quadratic)
}

fn quadratic_rational_roots(quadratic: QuadraticForm) -> Option<(usize, BigRational, BigRational)> {
    let variable = quadratic.variable?;
    let four = BigRational::from_integer(4.into());
    let two = BigRational::from_integer(2.into());
    let discriminant = quadratic.linear.clone() * quadratic.linear.clone()
        - four * quadratic.quadratic.clone() * quadratic.constant;
    let sqrt_discriminant = rational_sqrt_nonnegative(&discriminant)?;
    let denominator = two * quadratic.quadratic;
    let left = (-quadratic.linear.clone() - sqrt_discriminant.clone()) / denominator.clone();
    let right = (-quadratic.linear + sqrt_discriminant) / denominator;
    let (lower, upper) = if left <= right {
        (left, right)
    } else {
        (right, left)
    };

    Some((variable, lower, upper))
}

fn quadratic_form(ctx: &Context, expr: ExprId) -> Option<QuadraticForm> {
    match ctx.get(expr) {
        Expr::Number(n) => Some(QuadraticForm::constant(n.clone())),
        Expr::Variable(variable) => Some(QuadraticForm::variable(*variable)),
        Expr::Neg(inner) => quadratic_form(ctx, *inner).map(|form| form.scale(negative_one())),
        Expr::Add(left, right) => quadratic_form(ctx, *left)?.add(quadratic_form(ctx, *right)?),
        Expr::Sub(left, right) => {
            quadratic_form(ctx, *left)?.add(quadratic_form(ctx, *right)?.scale(negative_one()))
        }
        Expr::Mul(left, right) => quadratic_form(ctx, *left)?.mul(quadratic_form(ctx, *right)?),
        Expr::Div(left, right) => {
            let numerator = quadratic_form(ctx, *left)?;
            let denominator = quadratic_form(ctx, *right)?;
            if !denominator.is_constant() || denominator.constant.is_zero() {
                return None;
            }
            Some(numerator.scale(BigRational::one() / denominator.constant))
        }
        Expr::Pow(base, exponent) if is_integer_literal(ctx, *exponent, 2) => {
            let base_form = quadratic_form(ctx, *base)?;
            base_form.clone().mul(base_form)
        }
        _ => None,
    }
}

fn combine_single_variable(left: Option<usize>, right: Option<usize>) -> Option<Option<usize>> {
    match (left, right) {
        (None, variable) | (variable, None) => Some(variable),
        (Some(left), Some(right)) if left == right => Some(Some(left)),
        _ => None,
    }
}

fn polynomial_degree(coefficients: &[&BigRational]) -> usize {
    coefficients
        .iter()
        .enumerate()
        .rev()
        .find_map(|(degree, coefficient)| (!coefficient.is_zero()).then_some(degree))
        .unwrap_or(0)
}

fn rational_sqrt_nonnegative(value: &BigRational) -> Option<BigRational> {
    if value.is_negative() {
        return None;
    }
    let num_root = value.numer().sqrt();
    let den_root = value.denom().sqrt();
    if num_root.pow(2) == *value.numer() && den_root.pow(2) == *value.denom() {
        Some(BigRational::new(num_root, den_root))
    } else {
        None
    }
}

fn affine_form(ctx: &Context, expr: ExprId) -> Option<AffineForm> {
    match ctx.get(expr) {
        Expr::Number(n) => Some(AffineForm::constant(n.clone())),
        Expr::Variable(variable) => Some(AffineForm::variable(*variable)),
        Expr::Neg(inner) => affine_form(ctx, *inner).map(|form| form.scale(negative_one())),
        Expr::Add(left, right) => affine_form(ctx, *left)?.add(affine_form(ctx, *right)?),
        Expr::Sub(left, right) => {
            affine_form(ctx, *left)?.add(affine_form(ctx, *right)?.scale(negative_one()))
        }
        Expr::Mul(left, right) => {
            let left_form = affine_form(ctx, *left)?;
            let right_form = affine_form(ctx, *right)?;
            if left_form.is_constant() {
                Some(right_form.scale(left_form.intercept))
            } else if right_form.is_constant() {
                Some(left_form.scale(right_form.intercept))
            } else {
                None
            }
        }
        Expr::Div(left, right) => {
            let left_form = affine_form(ctx, *left)?;
            let right_form = affine_form(ctx, *right)?;
            if !right_form.is_constant() || right_form.intercept.is_zero() {
                return None;
            }
            Some(left_form.scale(BigRational::one() / right_form.intercept))
        }
        _ => None,
    }
}

fn display_rational_bound(value: &BigRational) -> String {
    if value.denom() == &BigInt::one() {
        value.numer().to_string()
    } else {
        format!("{}/{}", value.numer(), value.denom())
    }
}

fn display_sqrt_bound(value: &BigRational) -> Option<String> {
    if value.is_negative() || value.is_zero() {
        return None;
    }

    let radicand = if value.denom() == &BigInt::one() {
        value.numer().to_string()
    } else {
        format!("{}/{}", value.numer(), value.denom())
    };

    Some(format!("sqrt({radicand})"))
}

fn display_center_minus_offset(center: &str, offset: &str) -> String {
    if center == "0" {
        return format!("-{offset}");
    }

    format!("{center} - {offset}")
}

fn display_center_plus_offset(center: &str, offset: &str) -> String {
    if center == "0" {
        return offset.to_string();
    }

    format!("{center} + {offset}")
}

fn negative_one() -> BigRational {
    BigRational::from_integer((-1).into())
}

fn unit_interval_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Sub(left, right) if is_integer_literal(ctx, *left, 1) => squared_base(ctx, *right),
        Expr::Add(left, right) if is_integer_literal(ctx, *left, 1) => {
            negated_squared_base(ctx, *right)
        }
        Expr::Add(left, right) if is_integer_literal(ctx, *right, 1) => {
            negated_squared_base(ctx, *left)
        }
        _ => None,
    }
}

fn squared_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if is_integer_literal(ctx, *exponent, 2) {
        Some(*base)
    } else {
        None
    }
}

fn negated_squared_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Neg(inner) => squared_base(ctx, *inner),
        Expr::Mul(left, right) if is_integer_literal(ctx, *left, -1) => squared_base(ctx, *right),
        Expr::Mul(left, right) if is_integer_literal(ctx, *right, -1) => squared_base(ctx, *left),
        _ => None,
    }
}

fn reciprocal_denominator(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Div(numerator, denominator) if is_integer_literal(ctx, *numerator, 1) => {
            Some(*denominator)
        }
        Expr::Pow(base, exponent) if is_integer_literal(ctx, *exponent, -1) => Some(*base),
        _ => None,
    }
}

fn is_integer_literal(ctx: &Context, expr: ExprId, value: i64) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n == &BigRational::from_integer(value.into()))
}

/// Display level for required conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RequiresDisplayLevel {
    /// Show only essential requires (witness consumed or from equation derivation)
    #[default]
    Essential,
    /// Show all requires including implicit ones (witness survives)
    All,
}

/// Filter required conditions based on display level and witness survival.
pub fn filter_requires_for_display<'a>(
    requires: &'a [ImplicitCondition],
    ctx: &Context,
    result: ExprId,
    level: RequiresDisplayLevel,
) -> Vec<&'a ImplicitCondition> {
    requires
        .iter()
        .filter(|cond| {
            if level == RequiresDisplayLevel::All {
                return true;
            }
            !cond.witness_survives_in(ctx, result)
        })
        .collect()
}

/// Set of implicit conditions inferred from an expression.
#[derive(Debug, Clone, Default)]
pub struct ImplicitDomain {
    conditions: HashSet<ImplicitCondition>,
}

impl ImplicitDomain {
    /// Create an empty implicit domain.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Check if domain is empty.
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }

    /// Check if an expression has an implicit NonNegative constraint.
    pub fn contains_nonnegative(&self, expr: ExprId) -> bool {
        self.conditions
            .contains(&ImplicitCondition::NonNegative(expr))
    }

    /// Check if an expression has an implicit Positive constraint.
    pub fn contains_positive(&self, expr: ExprId) -> bool {
        self.conditions.contains(&ImplicitCondition::Positive(expr))
    }

    /// Check if an expression has an implicit NonZero constraint.
    pub fn contains_nonzero(&self, expr: ExprId) -> bool {
        self.conditions.contains(&ImplicitCondition::NonZero(expr))
    }

    /// Add a NonNegative condition.
    pub fn add_nonnegative(&mut self, expr: ExprId) {
        self.conditions.insert(ImplicitCondition::NonNegative(expr));
    }

    /// Add a lower-bound condition.
    pub fn add_lower_bound(&mut self, expr: ExprId, lower: BigRational) {
        self.conditions
            .insert(ImplicitCondition::LowerBound(expr, lower));
    }

    /// Add a Positive condition.
    pub fn add_positive(&mut self, expr: ExprId) {
        self.conditions.insert(ImplicitCondition::Positive(expr));
    }

    /// Add a NonZero condition.
    pub fn add_nonzero(&mut self, expr: ExprId) {
        self.conditions.insert(ImplicitCondition::NonZero(expr));
    }

    /// Get all conditions (for iteration/comparison).
    pub fn conditions(&self) -> &HashSet<ImplicitCondition> {
        &self.conditions
    }

    /// Get mutable access to conditions.
    pub fn conditions_mut(&mut self) -> &mut HashSet<ImplicitCondition> {
        &mut self.conditions
    }

    /// Check if this domain is a superset of another (contains all its conditions).
    pub fn contains_all(&self, other: &ImplicitDomain) -> bool {
        other.conditions.is_subset(&self.conditions)
    }

    /// Get conditions that are in self but not in other (dropped conditions).
    pub fn dropped_from<'a>(&'a self, other: &'a ImplicitDomain) -> Vec<&'a ImplicitCondition> {
        self.conditions.difference(&other.conditions).collect()
    }

    /// Merge conditions from another domain into this one.
    pub fn extend(&mut self, other: &ImplicitDomain) {
        for cond in &other.conditions {
            self.conditions.insert(cond.clone());
        }
    }

    /// Convert to a ConditionSet for solver use.
    pub fn to_condition_set(&self) -> cas_ast::ConditionSet {
        let predicates: Vec<cas_ast::ConditionPredicate> =
            self.conditions.iter().map(|c| c.into()).collect();
        cas_ast::ConditionSet::from_predicates(predicates)
    }
}

impl crate::domain_env::RequiredDomainSet for ImplicitDomain {
    fn contains_positive(&self, expr: ExprId) -> bool {
        self.contains_positive(expr)
    }

    fn contains_nonnegative(&self, expr: ExprId) -> bool {
        self.contains_nonnegative(expr)
    }

    fn contains_nonzero(&self, expr: ExprId) -> bool {
        self.contains_nonzero(expr)
    }

    fn to_condition_set(&self) -> cas_ast::ConditionSet {
        self.to_condition_set()
    }
}

impl From<&ImplicitCondition> for cas_ast::ConditionPredicate {
    fn from(cond: &ImplicitCondition) -> Self {
        match cond {
            ImplicitCondition::NonNegative(e) => cas_ast::ConditionPredicate::NonNegative(*e),
            ImplicitCondition::LowerBound(e, lower) => cas_ast::ConditionPredicate::LowerBound {
                expr: *e,
                lower: lower.clone(),
            },
            ImplicitCondition::Positive(e) => cas_ast::ConditionPredicate::Positive(*e),
            ImplicitCondition::NonZero(e) => cas_ast::ConditionPredicate::NonZero(*e),
        }
    }
}

impl From<ImplicitCondition> for cas_ast::ConditionPredicate {
    fn from(cond: ImplicitCondition) -> Self {
        (&cond).into()
    }
}

impl TryFrom<&cas_ast::ConditionPredicate> for ImplicitCondition {
    type Error = ();

    fn try_from(pred: &cas_ast::ConditionPredicate) -> Result<Self, Self::Error> {
        match pred {
            cas_ast::ConditionPredicate::NonNegative(e) => Ok(ImplicitCondition::NonNegative(*e)),
            cas_ast::ConditionPredicate::LowerBound { expr, lower } => {
                Ok(ImplicitCondition::LowerBound(*expr, lower.clone()))
            }
            cas_ast::ConditionPredicate::Positive(e) => Ok(ImplicitCondition::Positive(*e)),
            cas_ast::ConditionPredicate::NonZero(e) => Ok(ImplicitCondition::NonZero(*e)),
            _ => Err(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ImplicitCondition;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn nonzero_exponential_condition_is_trivial() {
        let mut ctx = Context::new();
        let exp_x = parse("exp(x)", &mut ctx).expect("parse exp(x)");

        assert!(ImplicitCondition::NonZero(exp_x).is_trivial(&ctx));
    }

    #[test]
    fn nonzero_strictly_positive_quadratic_condition_is_trivial() {
        let mut ctx = Context::new();
        let expr = parse("x^2+1", &mut ctx).expect("parse x^2+1");

        assert!(ImplicitCondition::NonZero(expr).is_trivial(&ctx));
    }

    #[test]
    fn positive_strictly_positive_quadratic_condition_is_trivial() {
        let mut ctx = Context::new();
        let expr = parse("x^2+1", &mut ctx).expect("parse x^2+1");

        assert!(ImplicitCondition::Positive(expr).is_trivial(&ctx));
    }

    #[test]
    fn nonnegative_shifted_square_plus_constant_condition_is_trivial() {
        let mut ctx = Context::new();
        let expr = parse("(2*x+1)^2+3", &mut ctx).expect("parse shifted square");

        assert!(ImplicitCondition::NonNegative(expr).is_trivial(&ctx));
    }

    #[test]
    fn positive_abs_displays_as_nonzero_inner_expression() {
        let mut ctx = Context::new();
        let abs_x = parse("abs(x)", &mut ctx).expect("parse abs(x)");

        assert_eq!(ImplicitCondition::Positive(abs_x).display(&ctx), "x ≠ 0");
    }

    #[test]
    fn nonzero_abs_displays_as_nonzero_inner_expression() {
        let mut ctx = Context::new();
        let abs_x = parse("abs(x)", &mut ctx).expect("parse abs(x)");

        assert_eq!(ImplicitCondition::NonZero(abs_x).display(&ctx), "x ≠ 0");
    }
}
