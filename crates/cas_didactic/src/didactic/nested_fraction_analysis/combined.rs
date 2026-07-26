use cas_ast::{Context, Expr, ExprId};

/// Extract the combined fraction string from an `Add` expression containing a fraction.
/// Example: `1 + 1/x -> "\\frac{x + 1}{x}"` in LaTeX.
/// The combined-fraction intermediate has NO node behind it (the design's §5
/// rama (b) class), so it exists only as a render — and it must exist ONCE per
/// surface. The walk is generic over the surface; the two public wrappers pin
/// plain and LaTeX so they cannot drift, and neither can leak into the other's
/// hole.
struct FractionSurface {
    render: fn(&Context, ExprId) -> String,
    fraction: fn(&str, &str) -> String,
    product: fn(&str, &str) -> String,
    fallback: &'static str,
}

const LATEX_SURFACE: FractionSurface = FractionSurface {
    render: |ctx, id| {
        let hints = cas_formatter::DisplayContext::default();
        cas_formatter::LaTeXExprWithHints {
            context: ctx,
            id,
            hints: &hints,
        }
        .to_latex()
    },
    fraction: |num, den| format!("\\frac{{{num}}}{{{den}}}"),
    product: |lhs, rhs| match (lhs.trim(), rhs.trim()) {
        ("1", other) => other.to_string(),
        (other, "1") => other.to_string(),
        (left, right) => format!("{left} \\cdot {right}"),
    },
    fallback: "\\text{(combinado)}",
};

const PLAIN_SURFACE: FractionSurface = FractionSurface {
    render: |ctx, id| format!("{}", cas_formatter::DisplayExpr { context: ctx, id }),
    fraction: |num, den| format!("({num})/({den})"),
    product: |lhs, rhs| match (lhs.trim(), rhs.trim()) {
        ("1", other) => other.to_string(),
        (other, "1") => other.to_string(),
        (left, right) => format!("{left} · {right}"),
    },
    fallback: "(combinado)",
};

pub(crate) fn extract_combined_fraction_str(ctx: &Context, add_expr: ExprId) -> String {
    extract_combined_fraction_surface(ctx, add_expr, &LATEX_SURFACE)
}

pub(crate) fn extract_combined_fraction_plain(ctx: &Context, add_expr: ExprId) -> String {
    extract_combined_fraction_surface(ctx, add_expr, &PLAIN_SURFACE)
}

fn extract_combined_fraction_surface(
    ctx: &Context,
    add_expr: ExprId,
    surface: &FractionSurface,
) -> String {
    if let Expr::Add(l, r) = ctx.get(add_expr) {
        if let (Expr::Div(left_num, left_den), Expr::Div(right_num, right_den)) =
            (ctx.get(*l), ctx.get(*r))
        {
            let left_num_str = (surface.render)(ctx, *left_num);
            let left_den_str = (surface.render)(ctx, *left_den);
            let right_num_str = (surface.render)(ctx, *right_num);
            let right_den_str = (surface.render)(ctx, *right_den);

            let left_scaled = (surface.product)(&left_num_str, &right_den_str);
            let right_scaled = (surface.product)(&right_num_str, &left_den_str);
            let common_den = (surface.product)(&left_den_str, &right_den_str);

            return (surface.fraction)(&format!("{left_scaled} + {right_scaled}"), &common_den);
        }

        let (frac_id, other_id) = if matches!(ctx.get(*l), Expr::Div(_, _)) {
            (*l, *r)
        } else if matches!(ctx.get(*r), Expr::Div(_, _)) {
            (*r, *l)
        } else {
            return surface.fallback.to_string();
        };

        if let Expr::Div(frac_num, frac_den) = ctx.get(frac_id) {
            let frac_num_str = (surface.render)(ctx, *frac_num);
            let frac_den_str = (surface.render)(ctx, *frac_den);
            let other_str = (surface.render)(ctx, other_id);

            let scaled = (surface.product)(&other_str, &frac_den_str);
            return (surface.fraction)(&format!("{scaled} + {frac_num_str}"), &frac_den_str);
        }
    }

    surface.fallback.to_string()
}
