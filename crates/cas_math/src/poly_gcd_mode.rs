//! Shared mode/option parsing for polynomial GCD entry points.

use crate::expr_extract::{extract_symbol_name, extract_usize_integer};
use crate::gcd_zippel_modp::ZippelPreset;
use cas_ast::{Context, Expr, ExprId};

/// Mode for poly_gcd computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcdMode {
    /// Structural GCD (HoldAll, no expansion) - default.
    Structural,
    /// Auto-select: structural → exact → modp.
    Auto,
    /// Force exact GCD over ℚ[x].
    Exact,
    /// Force modular GCD over 𝔽p[x].
    Modp,
}

/// Goal/context for GCD computation - determines allowed methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcdGoal {
    /// User explicitly asked for GCD (full pipeline allowed including modp).
    UserPolyGcd,
    /// Simplifier canceling fractions (safe methods only: Structural → Exact).
    /// Modp is blocked for soundness.
    CancelFraction,
}

/// Parsed `poly_gcd`/`pgcd` function call payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsedPolyGcdCall {
    pub lhs: ExprId,
    pub rhs: ExprId,
    pub mode: GcdMode,
    pub modp_preset: Option<ZippelPreset>,
    pub modp_main_var: Option<usize>,
}

/// Parse `GcdMode` from expression (variable token).
pub fn parse_gcd_mode(ctx: &Context, expr: ExprId) -> GcdMode {
    if let Expr::Variable(sym_id) = ctx.get(expr) {
        let s = ctx.sym_name(*sym_id);
        match s.to_lowercase().as_str() {
            "auto" => GcdMode::Auto,
            "exact" | "rational" | "algebraic" | "q" => GcdMode::Exact,
            "modp" | "mod_p" | "fast" | "zippel" => GcdMode::Modp,
            _ => GcdMode::Structural,
        }
    } else {
        GcdMode::Structural
    }
}

/// Parse modp options (preset symbol and/or `main_var` integer) from arguments.
pub fn parse_modp_options(ctx: &Context, args: &[ExprId]) -> (Option<ZippelPreset>, Option<usize>) {
    let mut preset: Option<ZippelPreset> = None;
    let mut main_var: Option<usize> = None;

    for &arg in args {
        if let Some(v) = extract_usize_integer(ctx, arg) {
            if v <= 64 {
                main_var = Some(v);
                continue;
            }
        }

        if let Some(s) = extract_symbol_name(ctx, arg) {
            if let Some(p) = ZippelPreset::parse(s) {
                preset = Some(p);
            }
        }
    }

    (preset, main_var)
}

/// Try parsing `poly_gcd(a, b [, mode [, modp_options...]])` / `pgcd(...)`.
///
/// Returns `None` when expression is not a matching call shape.
pub fn try_parse_poly_gcd_call(ctx: &Context, expr: ExprId) -> Option<ParsedPolyGcdCall> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    let fn_id = *fn_id;
    let args = args.clone();
    let name = ctx.sym_name(fn_id);

    if name != "poly_gcd" && name != "pgcd" {
        return None;
    }
    if !(2..=4).contains(&args.len()) {
        return None;
    }

    let lhs = args[0];
    let rhs = args[1];
    let mode = if args.len() >= 3 {
        parse_gcd_mode(ctx, args[2])
    } else {
        GcdMode::Structural
    };

    let (modp_preset, modp_main_var) = if args.len() >= 4 {
        parse_modp_options(ctx, &args[3..])
    } else {
        (None, None)
    };

    Some(ParsedPolyGcdCall {
        lhs,
        rhs,
        mode,
        modp_preset,
        modp_main_var,
    })
}

#[cfg(test)]
mod tests {
    use super::{try_parse_poly_gcd_call, GcdMode};
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn parses_basic_poly_gcd_call() {
        let mut ctx = Context::new();
        let expr = parse("poly_gcd(a, b)", &mut ctx).expect("parse");
        let parsed = try_parse_poly_gcd_call(&ctx, expr).expect("parsed");
        assert_eq!(parsed.mode, GcdMode::Structural);
        assert!(parsed.modp_preset.is_none());
        assert!(parsed.modp_main_var.is_none());
    }

    #[test]
    fn parses_alias_with_mode_and_modp_main_var() {
        let mut ctx = Context::new();
        let expr = parse("pgcd(a, b, modp, 2)", &mut ctx).expect("parse");
        let parsed = try_parse_poly_gcd_call(&ctx, expr).expect("parsed");
        assert_eq!(parsed.mode, GcdMode::Modp);
        assert_eq!(parsed.modp_main_var, Some(2));
    }

    #[test]
    fn rejects_non_poly_gcd_calls() {
        let mut ctx = Context::new();
        let expr = parse("foo(a, b)", &mut ctx).expect("parse");
        assert!(try_parse_poly_gcd_call(&ctx, expr).is_none());
    }
}
