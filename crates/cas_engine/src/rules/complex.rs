//! Complex-number rewrite rules.
//!
//! Rule matching remains in engine, while rewrite math lives in `cas_math`.

use crate::define_rule;
use crate::rule::Rewrite;
pub use cas_math::complex_support::{extract_gaussian, GaussianRational};
use cas_math::complex_support::{
    try_rewrite_gaussian_add_expr, try_rewrite_gaussian_div_expr, try_rewrite_gaussian_mul_expr,
    try_rewrite_i_squared_mul_identity_expr, try_rewrite_imaginary_power_expr,
    try_rewrite_sqrt_negative_expr,
};

define_rule!(
    ImaginaryPowerRule,
    "Imaginary Power",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }

        let rewrite = try_rewrite_imaginary_power_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(ISquaredMulRule, "i * i = -1", |ctx, expr, parent_ctx| {
    if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
        return None;
    }

    let rewrite = try_rewrite_i_squared_mul_identity_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});

define_rule!(
    GaussianMulRule,
    "Gaussian Multiplication",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }

        let rewrite = try_rewrite_gaussian_mul_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(
    GaussianAddRule,
    "Gaussian Addition",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }

        let rewrite = try_rewrite_gaussian_add_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(
    GaussianDivRule,
    "Gaussian Division",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }

        let rewrite = try_rewrite_gaussian_div_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(
    SqrtNegativeRule,
    "Square Root of Negative",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }

        let rewrite = try_rewrite_sqrt_negative_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(ImaginaryPowerRule));
    simplifier.add_rule(Box::new(ISquaredMulRule));
    simplifier.add_rule(Box::new(GaussianMulRule));
    simplifier.add_rule(Box::new(GaussianAddRule));
    simplifier.add_rule(Box::new(GaussianDivRule));
    simplifier.add_rule(Box::new(SqrtNegativeRule));
}
