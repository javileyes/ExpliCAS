use super::strong_target_match;
use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeriveIntegratePrepRewriteKind {
    CosProductTelescoping,
    DirichletKernelIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeriveIntegratePrepRewrite {
    pub(crate) rewritten: ExprId,
    pub(crate) assume_nonzero_expr: ExprId,
    pub(crate) kind: DeriveIntegratePrepRewriteKind,
}

impl DeriveIntegratePrepRewriteKind {
    pub(crate) fn description(self) -> &'static str {
        match self {
            Self::CosProductTelescoping => "Apply Morrie's law",
            Self::DirichletKernelIdentity => "Apply the Dirichlet kernel identity",
        }
    }

    pub(crate) fn rule_name(self) -> &'static str {
        match self {
            Self::CosProductTelescoping => "Cos Product Telescoping",
            Self::DirichletKernelIdentity => "Dirichlet Kernel Identity",
        }
    }
}

pub(crate) fn try_rewrite_integrate_prep_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveIntegratePrepRewrite> {
    try_rewrite_cos_product_telescoping_target_aware(ctx, expr, target_expr)
        .or_else(|| try_rewrite_dirichlet_kernel_target_aware(ctx, expr, target_expr))
}

fn try_rewrite_cos_product_telescoping_target_aware(
    ctx: &mut Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveIntegratePrepRewrite> {
    try_rewrite_cos_product_telescoping_forward(ctx, expr, target_expr).or_else(|| {
        try_rewrite_cos_product_telescoping_reverse_with_roles(ctx, target_expr, expr, target_expr)
    })
}

fn try_rewrite_cos_product_telescoping_forward(
    ctx: &mut Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveIntegratePrepRewrite> {
    let rewrite = cas_engine::try_rewrite_cos_product_telescoping_expr(ctx, expr)?;
    if !strong_target_match(ctx, rewrite.rewritten, target_expr) {
        return None;
    }
    Some(DeriveIntegratePrepRewrite {
        rewritten: rewrite.rewritten,
        assume_nonzero_expr: rewrite.assume_nonzero_expr,
        kind: DeriveIntegratePrepRewriteKind::CosProductTelescoping,
    })
}

fn try_rewrite_cos_product_telescoping_reverse_with_roles(
    ctx: &mut Context,
    product_expr: ExprId,
    quotient_expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveIntegratePrepRewrite> {
    let rewrite = cas_engine::try_rewrite_cos_product_telescoping_expr(ctx, product_expr)?;
    if !strong_target_match(ctx, rewrite.rewritten, quotient_expr) {
        return None;
    }

    let Expr::Div(_, denominator) = ctx.get(quotient_expr) else {
        return None;
    };

    Some(DeriveIntegratePrepRewrite {
        rewritten: target_expr,
        assume_nonzero_expr: *denominator,
        kind: DeriveIntegratePrepRewriteKind::CosProductTelescoping,
    })
}

fn try_rewrite_dirichlet_kernel_target_aware(
    ctx: &mut Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveIntegratePrepRewrite> {
    try_rewrite_dirichlet_kernel_with_roles(ctx, expr, target_expr, target_expr)
        .or_else(|| try_rewrite_dirichlet_kernel_with_roles(ctx, target_expr, expr, target_expr))
}

fn try_rewrite_dirichlet_kernel_with_roles(
    ctx: &mut Context,
    cosine_sum_expr: ExprId,
    quotient_expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveIntegratePrepRewrite> {
    let difference = ctx.add(Expr::Sub(cosine_sum_expr, quotient_expr));
    cas_engine::try_dirichlet_kernel_identity_pub(ctx, difference)?;

    let Expr::Div(_, denominator) = ctx.get(quotient_expr) else {
        return None;
    };

    Some(DeriveIntegratePrepRewrite {
        rewritten: target_expr,
        assume_nonzero_expr: *denominator,
        kind: DeriveIntegratePrepRewriteKind::DirichletKernelIdentity,
    })
}
