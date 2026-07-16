//! Numeric (f64) evaluation of expressions containing `root_sum` nodes
//! (G1 Cap. E-iv-d). The exact `root_sum(R(t), t, summand)` stays the primary
//! answer; this module powers the on-demand `approx(...)` consumption: the
//! roots of `R` are found numerically (Durand-Kerner over complex f64), the
//! summand is evaluated at each root in COMPLEX arithmetic (the imaginary
//! parts of conjugate pairs cancel — that cancellation carries the arctan
//! content, which is why the summand uses the plain complex-analytic `ln`),
//! and the real part is returned once the residual imaginary part is
//! negligible. Everything else delegates to the existing [`eval_f64`]
//! real-valued evaluator.
//!
//! This is a PRESENTATION/consumption surface: display values, never
//! keep/drop decisions (those stay exact per the soundness contract).

use crate::evaluator_f64::eval_f64;
use crate::polynomial::Polynomial;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_traits::ToPrimitive;
use std::collections::HashMap;

/// Minimal complex arithmetic over f64 (kept local: no new dependency).
#[derive(Clone, Copy, Debug)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    fn new(re: f64, im: f64) -> Self {
        C { re, im }
    }
    fn real(re: f64) -> Self {
        C { re, im: 0.0 }
    }
    fn add(self, o: C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: C) -> C {
        C::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: C) -> C {
        C::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    fn div(self, o: C) -> Option<C> {
        let d = o.re * o.re + o.im * o.im;
        if d == 0.0 || !d.is_finite() {
            return None;
        }
        Some(C::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        ))
    }
    fn neg(self) -> C {
        C::new(-self.re, -self.im)
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    /// Principal branch of the complex logarithm.
    fn ln(self) -> Option<C> {
        let m = self.abs();
        if m == 0.0 || !m.is_finite() {
            return None;
        }
        Some(C::new(m.ln(), self.im.atan2(self.re)))
    }
    fn powi(self, mut e: u32) -> C {
        let mut acc = C::real(1.0);
        let mut base = self;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(base);
            }
            base = base.mul(base);
            e >>= 1;
        }
        acc
    }
}

/// All complex roots of a ℚ-coefficient univariate polynomial, via the
/// Durand-Kerner (Weierstrass) simultaneous iteration in complex f64.
/// Returns `None` on degenerate input or non-convergence. Display-grade
/// accuracy (~1e-12 relative), NOT a decision procedure.
fn durand_kerner_roots(poly: &Polynomial) -> Option<Vec<C>> {
    let n = poly.degree();
    if n == 0 || poly.is_zero() {
        return None;
    }
    let coeffs: Vec<f64> = poly
        .coeffs
        .iter()
        .map(|c| c.to_f64())
        .collect::<Option<_>>()?;
    let lead = *coeffs.last()?;
    if lead == 0.0 || !lead.is_finite() {
        return None;
    }
    let monic: Vec<f64> = coeffs.iter().map(|c| c / lead).collect();
    let eval = |z: C| -> C {
        let mut acc = C::real(0.0);
        for &c in monic.iter().rev() {
            acc = acc.mul(z).add(C::real(c));
        }
        acc
    };
    // Standard starting points: powers of a non-real seed on a radius that
    // bounds the roots (1 + max |coeff|).
    let radius = 1.0 + monic.iter().fold(0.0f64, |m, c| m.max(c.abs()));
    let seed = C::new(0.4, 0.9);
    let mut roots: Vec<C> = (0..n)
        .map(|k| seed.powi(k as u32 + 1).mul(C::real(radius.min(2.0))))
        .collect();
    for _ in 0..200 {
        let mut moved = 0.0f64;
        for i in 0..n {
            let mut denom = C::real(1.0);
            for j in 0..n {
                if i != j {
                    denom = denom.mul(roots[i].sub(roots[j]));
                }
            }
            let delta = eval(roots[i]).div(denom)?;
            roots[i] = roots[i].sub(delta);
            moved = moved.max(delta.abs());
        }
        if moved < 1e-14 {
            return Some(roots);
        }
    }
    // Accept if the final residuals are tiny even without step convergence.
    if roots.iter().all(|&r| eval(r).abs() < 1e-9) {
        Some(roots)
    } else {
        None
    }
}

/// Evaluate an expression in complex arithmetic with a single bound variable.
/// Supports the shapes a `root_sum` summand contains (`t·ln(x − w(t))` and
/// friends): numbers, the bound variable, +, −, ·, /, unary minus, integer
/// powers and `ln`.
fn eval_complex(ctx: &Context, expr: ExprId, bound: &str, value: C, depth: usize) -> Option<C> {
    if depth == 0 {
        return None;
    }
    match ctx.get(expr) {
        Expr::Number(n) => Some(C::real(n.to_f64()?)),
        Expr::Variable(sym) => {
            if ctx.sym_name(*sym) == bound {
                Some(value)
            } else {
                None
            }
        }
        Expr::Add(l, r) => Some(
            eval_complex(ctx, *l, bound, value, depth - 1)?.add(eval_complex(
                ctx,
                *r,
                bound,
                value,
                depth - 1,
            )?),
        ),
        Expr::Sub(l, r) => Some(
            eval_complex(ctx, *l, bound, value, depth - 1)?.sub(eval_complex(
                ctx,
                *r,
                bound,
                value,
                depth - 1,
            )?),
        ),
        Expr::Mul(l, r) => Some(
            eval_complex(ctx, *l, bound, value, depth - 1)?.mul(eval_complex(
                ctx,
                *r,
                bound,
                value,
                depth - 1,
            )?),
        ),
        Expr::Div(l, r) => eval_complex(ctx, *l, bound, value, depth - 1)?.div(eval_complex(
            ctx,
            *r,
            bound,
            value,
            depth - 1,
        )?),
        Expr::Neg(inner) => Some(eval_complex(ctx, *inner, bound, value, depth - 1)?.neg()),
        Expr::Pow(base, exp) => {
            let e = match ctx.get(*exp) {
                Expr::Number(n) if n.is_integer() => n.to_integer().to_u32()?,
                _ => return None,
            };
            Some(eval_complex(ctx, *base, bound, value, depth - 1)?.powi(e))
        }
        Expr::Function(fn_id, args) if args.len() == 1 => {
            if ctx.is_builtin(*fn_id, BuiltinFn::Ln) {
                eval_complex(ctx, args[0], bound, value, depth - 1)?.ln()
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Numeric value of a `root_sum(R(t), t, summand)` node: `Σ summand(t = c)`
/// over the numeric roots `c` of `R`, in complex arithmetic. The result must
/// come out real (conjugate imaginary parts cancel); a non-negligible
/// residual imaginary part means the input was not a genuine real-valued
/// root sum and evaluation declines.
pub fn eval_rootsum_node_f64(ctx: &Context, args: &[ExprId]) -> Option<f64> {
    if args.len() != 3 {
        return None;
    }
    let bound = match ctx.get(args[1]) {
        Expr::Variable(sym) => ctx.sym_name(*sym).to_string(),
        _ => return None,
    };
    let r_poly = Polynomial::from_expr(ctx, args[0], &bound).ok()?;
    let roots = durand_kerner_roots(&r_poly)?;
    let mut total = C::real(0.0);
    for root in roots {
        total = total.add(eval_complex(ctx, args[2], &bound, root, 200)?);
    }
    if !total.re.is_finite() || total.im.abs() > 1e-8 * (1.0 + total.re.abs()) {
        return None;
    }
    Some(total.re)
}

/// Whether the expression contains a `root_sum` node anywhere.
fn contains_root_sum(ctx: &Context, root: ExprId) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(fn_id, args) => {
                if ctx.sym_name(*fn_id) == "root_sum" {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

/// Numeric evaluation of a closed (variable-free apart from `root_sum`
/// binders) expression, `root_sum`-aware: the additive/multiplicative spine
/// is walked here, `root_sum` nodes are summed over their numeric roots, and
/// any root_sum-free subtree delegates to the existing real [`eval_f64`].
pub fn numeric_eval_with_rootsum(ctx: &Context, expr: ExprId) -> Option<f64> {
    numeric_eval_depth(ctx, expr, 200)
}

fn numeric_eval_depth(ctx: &Context, expr: ExprId, depth: usize) -> Option<f64> {
    if depth == 0 {
        return None;
    }
    if !contains_root_sum(ctx, expr) {
        return eval_f64(ctx, expr, &HashMap::new()).filter(|v| v.is_finite());
    }
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if ctx.sym_name(*fn_id) == "root_sum" => {
            eval_rootsum_node_f64(ctx, args)
        }
        Expr::Add(l, r) => {
            Some(numeric_eval_depth(ctx, *l, depth - 1)? + numeric_eval_depth(ctx, *r, depth - 1)?)
        }
        Expr::Sub(l, r) => {
            Some(numeric_eval_depth(ctx, *l, depth - 1)? - numeric_eval_depth(ctx, *r, depth - 1)?)
        }
        Expr::Mul(l, r) => {
            Some(numeric_eval_depth(ctx, *l, depth - 1)? * numeric_eval_depth(ctx, *r, depth - 1)?)
        }
        Expr::Div(l, r) => {
            let denom = numeric_eval_depth(ctx, *r, depth - 1)?;
            if denom == 0.0 {
                return None;
            }
            Some(numeric_eval_depth(ctx, *l, depth - 1)? / denom)
        }
        Expr::Neg(inner) => Some(-numeric_eval_depth(ctx, *inner, depth - 1)?),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn rootsum_numeric_matches_reference_values() {
        let mut ctx = Context::new();
        // ∫1/(x^3-x-1) evaluated at x=2 MINUS at x... use the antiderivative
        // difference F(3) − F(2) = ∫_2^3 dx/(x^3-x-1) ≈ 0.0985572476935.
        let f3 = parse(
            "root_sum(1-23*t^3-3*t, t, t*ln(3 - (46/9*t^2 + 23/9*t + 4/9)))",
            &mut ctx,
        )
        .expect("parse F(3)");
        let f2 = parse(
            "root_sum(1-23*t^3-3*t, t, t*ln(2 - (46/9*t^2 + 23/9*t + 4/9)))",
            &mut ctx,
        )
        .expect("parse F(2)");
        let v3 = numeric_eval_with_rootsum(&ctx, f3).expect("F(3)");
        let v2 = numeric_eval_with_rootsum(&ctx, f2).expect("F(2)");
        // Reference: mpmath 30-digit quadrature of ∫_2^3 dx/(x^3-x-1).
        assert!(
            (v3 - v2 - 0.094_426_569_058_380_1).abs() < 1e-12,
            "got {}",
            v3 - v2
        );
    }

    #[test]
    fn rootsum_numeric_declines_uneval_shapes() {
        let mut ctx = Context::new();
        // Free variable x inside the summand: not a closed value.
        let open = parse("root_sum(1-23*t^3-3*t, t, t*ln(x - t))", &mut ctx).expect("parse");
        assert!(numeric_eval_with_rootsum(&ctx, open).is_none());
        // Degenerate polynomial (constant in t).
        let degenerate = parse("root_sum(5, t, t*ln(2 - t))", &mut ctx).expect("parse");
        assert!(numeric_eval_with_rootsum(&ctx, degenerate).is_none());
    }

    #[test]
    fn plain_constants_delegate_to_eval_f64() {
        let mut ctx = Context::new();
        let expr = parse("pi + sqrt(2)", &mut ctx).expect("parse");
        let v = numeric_eval_with_rootsum(&ctx, expr).expect("value");
        assert!((v - (std::f64::consts::PI + std::f64::consts::SQRT_2)).abs() < 1e-12);
    }
}
