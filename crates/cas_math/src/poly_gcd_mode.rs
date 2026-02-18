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
