//! Complex f64 evaluation — the numeric REFUTE-ONLY net of the complex domain.
//!
//! `Complex64` is the minimal hand-rolled complex arithmetic promoted out of
//! `rootsum_numeric.rs` (G1 E-iv-d1), which now delegates here. The deliberate
//! no-new-dependency decision is inherited from that module: everything is
//! `f64` + `std` (`atan2`, `hypot`, `exp`, `ln`), no `num-complex`.
//!
//! CONTRACT (same asymmetry as `eval/actions.rs`: "a probe may REFUTE, never
//! CONFIRM"): values computed here are display/refutation-grade (~1e-12), NOT
//! a decision procedure. Consumers may return "not equivalent" / decline from
//! a clearly non-zero probe, but must NEVER confirm an identity from a
//! near-zero probe. Exact confirmation stays in the exact layers.
//!
//! BRANCH CONVENTION: principal branch everywhere (`ln` via `atan2` into
//! `(-π, π]`, `sqrt`/`powc` derived from it). This intentionally DIFFERS from
//! the real-domain odd-root convention of `evaluator_f64::pow_real`
//! (`(-8)^(1/3) = -2` there, `1 + i·√3` here): never mix the two evaluators
//! across domain semantics or a true identity will be falsely refuted.

use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_traits::ToPrimitive;
use std::collections::HashMap;

/// Minimal complex number over `f64` (principal-branch transcendentals).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub fn new(re: f64, im: f64) -> Self {
        Complex64 { re, im }
    }
    pub fn real(re: f64) -> Self {
        Complex64 { re, im: 0.0 }
    }
    pub const I: Complex64 = Complex64 { re: 0.0, im: 1.0 };

    /// Checked division (`None` on a zero/non-finite denominator) — named
    /// like `i32::checked_div`; the `std::ops` traits cover the exact ops.
    pub fn checked_div(self, o: Complex64) -> Option<Complex64> {
        let d = o.re * o.re + o.im * o.im;
        if d == 0.0 || !d.is_finite() {
            return None;
        }
        Some(Complex64::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        ))
    }
    /// Modulus `|z|` (real, non-negative).
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    /// Principal argument `Arg(z) ∈ (-π, π]`; `None` for `z = 0`.
    pub fn arg(self) -> Option<f64> {
        if self.re == 0.0 && self.im == 0.0 {
            return None;
        }
        Some(self.im.atan2(self.re))
    }
    /// Principal branch of the complex logarithm.
    pub fn ln(self) -> Option<Complex64> {
        let m = self.abs();
        if m == 0.0 || !m.is_finite() {
            return None;
        }
        Some(Complex64::new(m.ln(), self.im.atan2(self.re)))
    }
    /// Complex exponential `e^z = e^re · (cos im + i sin im)`.
    pub fn exp(self) -> Complex64 {
        let r = self.re.exp();
        Complex64::new(r * self.im.cos(), r * self.im.sin())
    }
    /// Complex sine `sin(a+bi) = sin a cosh b + i cos a sinh b`.
    pub fn sin(self) -> Complex64 {
        Complex64::new(
            self.re.sin() * self.im.cosh(),
            self.re.cos() * self.im.sinh(),
        )
    }
    /// Complex cosine `cos(a+bi) = cos a cosh b - i sin a sinh b`.
    pub fn cos(self) -> Complex64 {
        Complex64::new(
            self.re.cos() * self.im.cosh(),
            -self.re.sin() * self.im.sinh(),
        )
    }
    /// Principal square root (`Re ≥ 0` half-plane; derived from `powc`).
    pub fn sqrt(self) -> Option<Complex64> {
        if self.re == 0.0 && self.im == 0.0 {
            return Some(Complex64::real(0.0));
        }
        self.powc(Complex64::real(0.5))
    }
    /// Non-negative integer power by repeated squaring.
    pub fn powi(self, mut e: u32) -> Complex64 {
        let mut acc = Complex64::real(1.0);
        let mut base = self;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc * base;
            }
            base = base * base;
            e >>= 1;
        }
        acc
    }
    /// Principal complex power `z^w = e^(w · Log z)`; `None` for `z = 0`
    /// with non-real-positive `w` semantics kept simple: `0^w` declines.
    pub fn powc(self, w: Complex64) -> Option<Complex64> {
        Some((w * self.ln()?).exp())
    }
}

impl std::ops::Add for Complex64 {
    type Output = Complex64;
    fn add(self, o: Complex64) -> Complex64 {
        Complex64::new(self.re + o.re, self.im + o.im)
    }
}

impl std::ops::Sub for Complex64 {
    type Output = Complex64;
    fn sub(self, o: Complex64) -> Complex64 {
        Complex64::new(self.re - o.re, self.im - o.im)
    }
}

impl std::ops::Mul for Complex64 {
    type Output = Complex64;
    fn mul(self, o: Complex64) -> Complex64 {
        Complex64::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

impl std::ops::Neg for Complex64 {
    type Output = Complex64;
    fn neg(self) -> Complex64 {
        Complex64::new(-self.re, -self.im)
    }
}

const MAX_EVAL_DEPTH: usize = 256;

/// Evaluate an expression to a `Complex64` under COMPLEX (principal-branch)
/// semantics. Variables come from `var_map`; `i`, `π`, `e` are built in.
/// Returns `None` for anything unsupported (honest decline, never a guess).
pub fn eval_complex(
    ctx: &Context,
    expr: ExprId,
    var_map: &HashMap<String, Complex64>,
) -> Option<Complex64> {
    eval_complex_depth(ctx, expr, var_map, MAX_EVAL_DEPTH)
}

fn eval_complex_depth(
    ctx: &Context,
    expr: ExprId,
    var_map: &HashMap<String, Complex64>,
    depth: usize,
) -> Option<Complex64> {
    if depth == 0 {
        return None;
    }
    let rec = |e: ExprId| eval_complex_depth(ctx, e, var_map, depth - 1);
    match ctx.get(expr) {
        Expr::Number(n) => Some(Complex64::real(n.to_f64()?)),
        Expr::Constant(Constant::I) => Some(Complex64::I),
        Expr::Constant(Constant::Pi) => Some(Complex64::real(std::f64::consts::PI)),
        Expr::Constant(Constant::E) => Some(Complex64::real(std::f64::consts::E)),
        Expr::Constant(_) => None,
        Expr::Variable(sym) => var_map.get(ctx.sym_name(*sym)).copied(),
        Expr::Add(l, r) => Some(rec(*l)? + rec(*r)?),
        Expr::Sub(l, r) => Some(rec(*l)? - rec(*r)?),
        Expr::Mul(l, r) => Some(rec(*l)? * rec(*r)?),
        Expr::Div(l, r) => rec(*l)?.checked_div(rec(*r)?),
        Expr::Neg(inner) => Some(-rec(*inner)?),
        Expr::Pow(base, exp) => {
            let b = rec(*base)?;
            // Integer exponents use exact repeated squaring (sound at z = 0
            // and cheaper); everything else goes through the principal branch.
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() {
                    let int = n.to_integer();
                    if let Some(e) = int.to_u32() {
                        return Some(b.powi(e));
                    }
                    if let Some(e) = (-int).to_u32() {
                        return Complex64::real(1.0).checked_div(b.powi(e));
                    }
                    return None;
                }
            }
            b.powc(rec(*exp)?)
        }
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let builtin = ctx.builtin_of(*fn_id)?;
            let a = rec(args[0])?;
            match builtin {
                BuiltinFn::Ln => a.ln(),
                BuiltinFn::Exp => Some(a.exp()),
                BuiltinFn::Sin => Some(a.sin()),
                BuiltinFn::Cos => Some(a.cos()),
                BuiltinFn::Tan => a.sin().checked_div(a.cos()),
                BuiltinFn::Sqrt => a.sqrt(),
                BuiltinFn::Abs => Some(Complex64::real(a.abs())),
                // Real-argument atan only (what the exact Arg table emits);
                // complex atan is out of scope — honest decline.
                BuiltinFn::Atan if a.im == 0.0 => Some(Complex64::real(a.re.atan())),
                BuiltinFn::Re => Some(Complex64::real(a.re)),
                BuiltinFn::Im => Some(Complex64::real(a.im)),
                BuiltinFn::Conjugate => Some(Complex64::new(a.re, -a.im)),
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{eval_complex, Complex64};
    use cas_ast::Context;
    use std::collections::HashMap;

    const EPS: f64 = 1e-12;

    fn close(z: Complex64, re: f64, im: f64) -> bool {
        (z.re - re).abs() < EPS && (z.im - im).abs() < EPS
    }

    fn eval_str(src: &str) -> Option<Complex64> {
        let mut ctx = Context::new();
        let expr = cas_parser::parse(src, &mut ctx).expect("parse");
        eval_complex(&ctx, expr, &HashMap::new())
    }

    #[test]
    fn euler_identity_numeric() {
        // e^(iπ) = -1 — the flagship value-level identity B2 will emit.
        let z = eval_str("e^(i*pi)").expect("eval");
        assert!(close(z, -1.0, 0.0), "e^(i*pi) -> {z:?}");
        let z = eval_str("e^(i*pi/2)").expect("eval");
        assert!(close(z, 0.0, 1.0), "e^(i*pi/2) -> {z:?}");
    }

    #[test]
    fn principal_log_and_arg() {
        // ln(-1) = iπ, ln(i) = iπ/2 (principal branch).
        let z = eval_str("ln(-1)").expect("eval");
        assert!(close(z, 0.0, std::f64::consts::PI), "ln(-1) -> {z:?}");
        let z = eval_str("ln(i)").expect("eval");
        assert!(close(z, 0.0, std::f64::consts::FRAC_PI_2), "ln(i) -> {z:?}");
        // Arg boundary: Arg(-1) = +π (not -π).
        assert!((Complex64::real(-1.0).arg().unwrap() - std::f64::consts::PI).abs() < EPS);
        assert!(Complex64::real(0.0).arg().is_none());
    }

    #[test]
    fn principal_powers_and_sqrt() {
        // i^i = e^(-π/2) (real!).
        let z = eval_str("i^i").expect("eval");
        assert!(
            close(z, (-std::f64::consts::FRAC_PI_2).exp(), 0.0),
            "i^i -> {z:?}"
        );
        // sqrt(i) = (1+i)/√2 (principal, Re ≥ 0).
        let z = eval_str("sqrt(i)").expect("eval");
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!(close(z, s, s), "sqrt(i) -> {z:?}");
        // sqrt(3+4i) = 2+i.
        let z = eval_str("sqrt(3+4*i)").expect("eval");
        assert!(close(z, 2.0, 1.0), "sqrt(3+4i) -> {z:?}");
        // PRINCIPAL convention (differs from the real odd-root evaluator):
        // (-8)^(1/3) = 1 + i·√3, NOT -2.
        let z = eval_str("(-8)^(1/3)").expect("eval");
        assert!(close(z, 1.0, 3.0f64.sqrt()), "(-8)^(1/3) -> {z:?}");
    }

    #[test]
    fn a2_builtins_and_gaussian_arithmetic() {
        let z = eval_str("abs(3+4*i)").expect("eval");
        assert!(close(z, 5.0, 0.0));
        let z = eval_str("Re(3+4*i)").expect("eval");
        assert!(close(z, 3.0, 0.0));
        let z = eval_str("Im(3+4*i)").expect("eval");
        assert!(close(z, 4.0, 0.0));
        let z = eval_str("conjugate(3+4*i)").expect("eval");
        assert!(close(z, 3.0, -4.0));
        let z = eval_str("(3+4*i)/(1-2*i)").expect("eval");
        assert!(close(z, -1.0, 2.0));
    }

    #[test]
    fn honest_declines() {
        // 0^i, ln(0), Arg(0), unknown variable, unsupported function.
        assert!(eval_str("ln(0)").is_none());
        assert!(eval_str("0^i").is_none());
        assert!(eval_str("x + i").is_none());
        assert!(eval_str("floor(i)").is_none());
    }
}
