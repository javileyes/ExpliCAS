use super::super::additive_search::collect_additive_focus_paths;
use crate::runtime::{pathsteps_to_expr_path, Step};
use cas_ast::{Context, ExprId, ExprPath};
use cas_formatter::path::{
    diff_find_path_to_expr, diff_find_paths_by_structure, navigate_to_subexpr,
};
use cas_math::expr_nary::{add_terms_signed, Sign};

pub(super) fn collect_additive_focus_paths_with_scope(
    context: &Context,
    step: &Step,
    global_before_expr: ExprId,
    focus_before: ExprId,
    focus_terms: &[ExprId],
) -> Vec<ExprPath> {
    let step_path_prefix = pathsteps_to_expr_path(step.path());
    let subexpr_at_path = navigate_to_subexpr(context, global_before_expr, &step_path_prefix);
    let before_local_path = diff_find_path_to_expr(context, subexpr_at_path, focus_before)
        .or_else(|| {
            diff_find_paths_by_structure(context, subexpr_at_path, focus_before)
                .into_iter()
                .next()
        })
        .or_else(|| {
            find_additive_scope_path_by_signed_terms(context, subexpr_at_path, focus_before)
        });

    let (search_scope, scope_path_prefix) = if let Some(path_to_local) = &before_local_path {
        let local_scope = navigate_to_subexpr(context, subexpr_at_path, path_to_local);
        let mut full_prefix = step_path_prefix.clone();
        full_prefix.extend(path_to_local.clone());
        (local_scope, full_prefix)
    } else if focus_terms.len() > 1 {
        // Some engine steps point to the first changed term instead of the full additive
        // cancellation scope. In that case, search the whole global snapshot so repeated
        // terms like a - a can still be highlighted as one local cancellation.
        (global_before_expr, Vec::new())
    } else {
        (subexpr_at_path, step_path_prefix.clone())
    };

    collect_additive_focus_paths(context, search_scope, &scope_path_prefix, focus_terms)
}

fn find_additive_scope_path_by_signed_terms(
    context: &Context,
    root: ExprId,
    target: ExprId,
) -> Option<ExprPath> {
    let target_signature = additive_signature(context, target);
    let mut path = Vec::new();
    find_additive_scope_path_by_signed_terms_rec(context, root, &target_signature, &mut path)
}

fn find_additive_scope_path_by_signed_terms_rec(
    context: &Context,
    current: ExprId,
    target_signature: &[String],
    path: &mut ExprPath,
) -> Option<ExprPath> {
    if matches!(
        context.get(current),
        cas_ast::Expr::Add(_, _) | cas_ast::Expr::Sub(_, _)
    ) && additive_signature(context, current) == target_signature
    {
        return Some(path.clone());
    }

    match context.get(current) {
        cas_ast::Expr::Add(l, r)
        | cas_ast::Expr::Sub(l, r)
        | cas_ast::Expr::Mul(l, r)
        | cas_ast::Expr::Div(l, r)
        | cas_ast::Expr::Pow(l, r) => {
            path.push(0);
            if let Some(found) =
                find_additive_scope_path_by_signed_terms_rec(context, *l, target_signature, path)
            {
                return Some(found);
            }
            path.pop();

            path.push(1);
            if let Some(found) =
                find_additive_scope_path_by_signed_terms_rec(context, *r, target_signature, path)
            {
                return Some(found);
            }
            path.pop();
            None
        }
        cas_ast::Expr::Neg(inner) | cas_ast::Expr::Hold(inner) => {
            path.push(0);
            let found = find_additive_scope_path_by_signed_terms_rec(
                context,
                *inner,
                target_signature,
                path,
            );
            path.pop();
            found
        }
        cas_ast::Expr::Function(_, args) => {
            for (i, arg) in args.iter().enumerate() {
                path.push(i as u8);
                if let Some(found) = find_additive_scope_path_by_signed_terms_rec(
                    context,
                    *arg,
                    target_signature,
                    path,
                ) {
                    return Some(found);
                }
                path.pop();
            }
            None
        }
        cas_ast::Expr::Matrix { data, .. } => {
            for (i, elem) in data.iter().enumerate() {
                path.push(i as u8);
                if let Some(found) = find_additive_scope_path_by_signed_terms_rec(
                    context,
                    *elem,
                    target_signature,
                    path,
                ) {
                    return Some(found);
                }
                path.pop();
            }
            None
        }
        cas_ast::Expr::Number(_)
        | cas_ast::Expr::Variable(_)
        | cas_ast::Expr::Constant(_)
        | cas_ast::Expr::SessionRef(_) => None,
    }
}

fn additive_signature(context: &Context, expr: ExprId) -> Vec<String> {
    let mut signature: Vec<String> = add_terms_signed(context, expr)
        .into_iter()
        .map(|(term, sign)| {
            let sign_prefix = match sign {
                Sign::Pos => "+",
                Sign::Neg => "-",
            };
            let display =
                cas_formatter::clean_display_string(&crate::didactic::latex_to_plain_text(
                    &cas_formatter::LaTeXExpr { context, id: term }.to_latex(),
                ));
            format!("{sign_prefix}:{display}")
        })
        .collect();
    signature.sort();
    signature
}
