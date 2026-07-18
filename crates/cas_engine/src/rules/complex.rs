//! Complex-number rewrite rules.
//!
//! Rule matching remains in engine, while rewrite math lives in `cas_math`.

use crate::define_rule;
use crate::rule::Rewrite;
pub use cas_math::complex_support::{extract_gaussian, GaussianRational};
use cas_math::complex_support::{
    try_rewrite_arg_expr, try_rewrite_complex_general_power_expr, try_rewrite_conjugate_expr,
    try_rewrite_euler_expr, try_rewrite_gaussian_abs_expr, try_rewrite_gaussian_add_expr,
    try_rewrite_gaussian_div_expr, try_rewrite_gaussian_mul_expr, try_rewrite_gaussian_power_expr,
    try_rewrite_gaussian_sqrt_expr, try_rewrite_i_squared_mul_identity_expr, try_rewrite_im_expr,
    try_rewrite_imaginary_power_expr, try_rewrite_negative_base_half_power_expr,
    try_rewrite_re_expr, try_rewrite_sqrt_negative_expr, ComplexRewriteKind,
};

fn format_complex_rewrite_desc(kind: ComplexRewriteKind) -> &'static str {
    match kind {
        ComplexRewriteKind::ImaginaryPower => "Imaginary power (using i⁴ = 1)",
        ComplexRewriteKind::ISquaredMul => "i · i = -1",
        ComplexRewriteKind::GaussianMul => {
            "Gaussian multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i"
        }
        ComplexRewriteKind::GaussianAdd => "Gaussian addition: (a+bi) + (c+di) = (a+c) + (b+d)i",
        ComplexRewriteKind::GaussianDiv => "Gaussian division: (a+bi)/(c+di) using conjugate",
        ComplexRewriteKind::GaussianPower => {
            "Gaussian power: (a+bi)^n by repeated multiplication (using i² = -1)"
        }
        ComplexRewriteKind::SqrtNegative => "sqrt(-n) = i·√n (complex mode)",
        ComplexRewriteKind::GaussianAbs => "Complex modulus: |a+bi| = √(a²+b²)",
        ComplexRewriteKind::Conjugate => "Complex conjugate: conjugate(a+bi) = a-bi",
        ComplexRewriteKind::RealPart => "Real part: Re(a+bi) = a",
        ComplexRewriteKind::ImagPart => "Imaginary part: Im(a+bi) = b",
        ComplexRewriteKind::Euler => "Euler's formula: e^(iθ) = cos θ + i·sin θ",
        ComplexRewriteKind::PrincipalArg => "Principal argument: arg(a+bi) via exact atan2",
        ComplexRewriteKind::PrincipalLog => "Principal logarithm: ln(z) = ln|z| + i·Arg(z)",
        ComplexRewriteKind::GaussianSqrt => {
            "Principal square root: sqrt(a+bi) = sqrt((|z|+a)/2) + i·sign(b)·sqrt((|z|-a)/2)"
        }
        ComplexRewriteKind::ComplexGeneralPower => {
            "Complex power: z^w = e^(w·ln z) (principal branch)"
        }
    }
}

define_rule!(
    ImaginaryPowerRule,
    "Imaginary Power",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }

        let rewrite = try_rewrite_imaginary_power_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_complex_rewrite_desc(rewrite.kind)))
    }
);

define_rule!(ISquaredMulRule, "i * i = -1", |ctx, expr, parent_ctx| {
    if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
        return None;
    }

    let rewrite = try_rewrite_i_squared_mul_identity_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(format_complex_rewrite_desc(rewrite.kind)))
});

define_rule!(
    GaussianMulRule,
    "Gaussian Multiplication",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }

        let rewrite = try_rewrite_gaussian_mul_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_complex_rewrite_desc(rewrite.kind)))
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
        Some(Rewrite::new(rewrite.rewritten).desc(format_complex_rewrite_desc(rewrite.kind)))
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
        Some(Rewrite::new(rewrite.rewritten).desc(format_complex_rewrite_desc(rewrite.kind)))
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
        Some(Rewrite::new(rewrite.rewritten).desc(format_complex_rewrite_desc(rewrite.kind)))
    }
);

define_rule!(
    NegativeBaseHalfPowerRule,
    "Negative Base Square Root Power",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }

        // The canonical Pow form `(-n)^(1/2)` -> `i·√n` (complements SqrtNegativeRule,
        // which only matches the `sqrt(...)` call form).
        let rewrite = try_rewrite_negative_base_half_power_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_complex_rewrite_desc(rewrite.kind)))
    }
);

define_rule!(
    GaussianPowRule,
    "Gaussian Power",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }

        // `(a+bi)^n` (true binomial base, integer n ≥ 2) folds to its exact
        // Gaussian value. Real / pure-imaginary bases decline inside the helper
        // (owned by ordinary arithmetic and power-of-a-product + i^n).
        let rewrite = try_rewrite_gaussian_power_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_complex_rewrite_desc(rewrite.kind)))
    }
);

define_rule!(
    GaussianAbsRule,
    "Complex Modulus",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }

        // `abs(a+bi)` with b ≠ 0 folds to the exact modulus √(a²+b²) (perfect
        // squares fold to a rational). Real arguments decline inside the
        // helper: the real-`abs` machinery keeps its ownership.
        let rewrite = try_rewrite_gaussian_abs_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_complex_rewrite_desc(rewrite.kind)))
    }
);

define_rule!(
    ConjugateRule,
    "Complex Conjugate",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }

        let rewrite = try_rewrite_conjugate_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_complex_rewrite_desc(rewrite.kind)))
    }
);

define_rule!(RealPartRule, "Real Part", |ctx, expr, parent_ctx| {
    if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
        return None;
    }

    let rewrite = try_rewrite_re_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(format_complex_rewrite_desc(rewrite.kind)))
});

define_rule!(ImagPartRule, "Imaginary Part", |ctx, expr, parent_ctx| {
    if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
        return None;
    }

    let rewrite = try_rewrite_im_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(format_complex_rewrite_desc(rewrite.kind)))
});

define_rule!(
    UnimodularAbsRule,
    "Unimodular Absolute Value",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }
        // `|cos θ ± i·sin θ| = 1` ONLY when θ is a DECIDABLE real constant
        // (`provable_const_sign`: rationals, surds, e/π combos). A bare symbol must
        // DECLINE: under ComplexEnabled it may hold a complex value and unimodularity
        // is then FALSE (x:=i gives |e^(i·i)| = 1/e) — the V0 sticky-fold discipline.
        let theta = cas_math::complex_support::try_match_unimodular_abs(ctx, expr)?;
        cas_math::const_sign::provable_const_sign(ctx, theta)?;
        Some(
            Rewrite::new(ctx.num(1))
                .desc("valor absoluto unimodular: |cos θ + i·sen θ| = 1 con θ real"),
        )
    }
);

define_rule!(
    TrigOfImaginaryRule,
    "Trig of Imaginary Argument",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }
        // The entire-function bridge (sin(iy) = i·sinh(y) and 5 sisters): valid for
        // ARBITRARY complex y — no realness guard needed, unlike UnimodularAbsRule.
        // ONE-DIRECTION on purpose (no inverse rule: ping-pong).
        let (rewritten, desc) =
            cas_math::complex_support::try_rewrite_trig_of_imaginary(ctx, expr)?;
        Some(Rewrite::new(rewritten).desc(desc))
    }
);

define_rule!(
    ComplexAngleSumRule,
    "Complex Angle Sum",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }
        // Mixed argument `re + i·θ`: the entire angle-sum bridge (sin(1+i) →
        // sin(1)·cosh(1) + i·cos(1)·sinh(1)). Pure-imaginary stays with
        // TrigOfImaginaryRule; tan/tanh decline honestly. ONE-DIRECTION — the trig
        // contraction rules pattern-match cos/sin pairs, never cosh/sinh, so no
        // ping-pong side exists.
        let (rewritten, desc) =
            cas_math::complex_support::try_rewrite_trig_complex_angle_sum(ctx, expr)?;
        Some(Rewrite::new(rewritten).desc(desc))
    }
);

define_rule!(
    GaussianSurdAbsRule,
    "Gaussian Surd Modulus",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }
        // |a+b·i| with decidable-real surd/transcendental components — the guard is
        // provable_const_sign on BOTH parts (V0 discipline: symbols decline). The
        // plain-rational case stays with the exact GaussianRational modulus.
        let (rewritten, desc) =
            cas_math::complex_support::try_rewrite_gaussian_surd_abs(ctx, expr)?;
        Some(Rewrite::new(rewritten).desc(desc))
    }
);

define_rule!(
    ReciprocalCisRule,
    "Reciprocal Cis",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }
        // n/(cos u ± i·sin u) → n·(cos u ∓ i·sin u): entire (Pythagorean cis·cis̄ = 1
        // holds on all of ℂ), so symbolic u carries no realness guard. Closes the
        // e^(-i·x) reciprocal residual (B2). ONE-DIRECTION.
        let (rewritten, desc) = cas_math::complex_support::try_rewrite_reciprocal_cis(ctx, expr)?;
        Some(Rewrite::new(rewritten).desc(desc))
    }
);

define_rule!(EulerRule, "Euler Formula", |ctx, expr, parent_ctx| {
    if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
        return None;
    }

    // `e^(a + i·θ)` -> `e^a · (cos θ + i·sin θ)` (Pow(E,·) primary — the
    // parser desugars exp() at parse time — plus a defensive Function arm
    // inside the helper). ONE-DIRECTION: no inverse rule exists on purpose.
    let rewrite = try_rewrite_euler_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(format_complex_rewrite_desc(rewrite.kind)))
});

define_rule!(ArgRule, "Principal Argument", |ctx, expr, parent_ctx| {
    if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
        return None;
    }

    // `arg(a+bi)` (closed Gaussian) -> exact principal value in (-π, π] via
    // the 9-case sign table; `arg(0)` -> Undefined explicitly.
    let rewrite = try_rewrite_arg_expr(ctx, expr)?;
    let mut out = Rewrite::new(rewrite.rewritten).desc(format_complex_rewrite_desc(rewrite.kind));
    // An undefined verdict is not a branch choice: only defined values
    // carry the structured principal-branch condition.
    if !matches!(
        ctx.get(rewrite.rewritten),
        cas_ast::Expr::Constant(cas_ast::Constant::Undefined)
    ) {
        if let cas_ast::Expr::Function(_, args) = ctx.get(expr) {
            out = out.requires(crate::ImplicitCondition::PrincipalBranch {
                func: "arg",
                arg: args[0],
            });
        }
    }
    Some(out)
});

define_rule!(
    GaussianSqrtRule,
    "Gaussian Square Root",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }

        // `sqrt(a+bi)` with a perfect-square norm folds to the EXACT
        // principal root (sqrt(3+4i) -> 2+i). Pure-real radicands and
        // non-perfect squares decline inside the helper.
        let rewrite = try_rewrite_gaussian_sqrt_expr(ctx, expr)?;
        let radicand = match ctx.get(expr) {
            cas_ast::Expr::Pow(base, _) => *base,
            cas_ast::Expr::Function(_, args) => args[0],
            _ => return None,
        };
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc(format_complex_rewrite_desc(rewrite.kind))
                .requires(crate::ImplicitCondition::PrincipalBranch {
                    func: "sqrt",
                    arg: radicand,
                }),
        )
    }
);

define_rule!(
    ComplexGeneralPowerRule,
    "Complex General Power",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }

        // `z^w = e^(w·ln z)` for closed Gaussians (i^i, 2^i, (1+i)^(1/3)).
        // The helper declines every owned form — including base == e, which
        // doubles as the anti-churn guard for the rule's own Pow(E,·) output.
        let rewrite = try_rewrite_complex_general_power_expr(ctx, expr)?;
        let base = match ctx.get(expr) {
            cas_ast::Expr::Pow(base, _) => *base,
            _ => return None,
        };
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc(format_complex_rewrite_desc(rewrite.kind))
                .requires(crate::ImplicitCondition::PrincipalBranch {
                    func: "pow",
                    arg: base,
                }),
        )
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(ImaginaryPowerRule));
    simplifier.add_rule(Box::new(ISquaredMulRule));
    simplifier.add_rule(Box::new(GaussianMulRule));
    simplifier.add_rule(Box::new(GaussianAddRule));
    simplifier.add_rule(Box::new(GaussianDivRule));
    simplifier.add_rule(Box::new(GaussianPowRule));
    simplifier.add_rule(Box::new(SqrtNegativeRule));
    simplifier.add_rule(Box::new(NegativeBaseHalfPowerRule));
    simplifier.add_rule(Box::new(GaussianAbsRule));
    simplifier.add_rule(Box::new(ConjugateRule));
    simplifier.add_rule(Box::new(RealPartRule));
    simplifier.add_rule(Box::new(ImagPartRule));
    simplifier.add_rule(Box::new(UnimodularAbsRule));
    simplifier.add_rule(Box::new(TrigOfImaginaryRule));
    simplifier.add_rule(Box::new(ComplexAngleSumRule));
    simplifier.add_rule(Box::new(GaussianSurdAbsRule));
    simplifier.add_rule(Box::new(ReciprocalCisRule));
    simplifier.add_rule(Box::new(EulerRule));
    simplifier.add_rule(Box::new(ArgRule));
    simplifier.add_rule(Box::new(GaussianSqrtRule));
    simplifier.add_rule(Box::new(ComplexGeneralPowerRule));
}
