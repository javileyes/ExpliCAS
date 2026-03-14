//! Structural support for proving polynomial identities equal to zero.

use cas_ast::ordering::compare_expr;
use cas_ast::{Constant, Context, Expr, ExprId};
use cas_math::multipoly::{MultiPoly, PolyBudget};
use cas_math::multipoly_display::PolynomialProofData;
use cas_math::opaque_atoms::{
    collect_exp_exponents, collect_function_calls, dedup_expr_ids,
    extract_opaque_rational_power_atom, extract_opaque_reciprocal_power_base, find_exp_base,
    is_polynomial_candidate, substitute_exp_atoms,
};
use cas_math::poly_convert::try_multipoly_from_expr_with_var_limit;
use cas_math::substitute::{substitute_power_aware, SubstituteOptions};
use num_rational::BigRational;
use num_traits::{One, Zero};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolynomialIdentityProofKind {
    Direct,
    OpaqueSubstitution,
    OpaqueRootRelation,
}

#[derive(Debug, Clone)]
pub struct PolynomialIdentityZeroPlan {
    pub kind: PolynomialIdentityProofKind,
    pub proof_data: PolynomialProofData,
}

#[derive(Debug, Clone)]
pub struct PolynomialIdentityPolicy {
    pub max_nodes: usize,
    pub max_vars: usize,
    pub max_atoms: usize,
    pub var_limit: usize,
    pub max_scan_depth: usize,
    pub max_pow_exp_scan: u32,
    pub poly_budget: PolyBudget,
}

impl Default for PolynomialIdentityPolicy {
    fn default() -> Self {
        Self {
            max_nodes: 100,
            max_vars: 4,
            max_atoms: 4,
            var_limit: 4,
            max_scan_depth: 30,
            max_pow_exp_scan: 6,
            poly_budget: PolyBudget {
                max_terms: 50,
                max_total_degree: 6,
                max_pow_exp: 6,
            },
        }
    }
}

/// Try to prove that `expr` is identically zero as a polynomial expression.
///
/// Returns proof metadata for didactic rendering when successful.
pub fn try_prove_polynomial_identity_zero_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PolynomialIdentityZeroPlan> {
    try_prove_polynomial_identity_zero_with_policy(ctx, expr, &PolynomialIdentityPolicy::default())
}

pub fn try_prove_polynomial_identity_zero_with_policy(
    ctx: &mut Context,
    expr: ExprId,
    policy: &PolynomialIdentityPolicy,
) -> Option<PolynomialIdentityZeroPlan> {
    if !matches!(ctx.get(expr), Expr::Sub(_, _) | Expr::Add(_, _)) {
        return None;
    }

    if cas_ast::count_nodes(ctx, expr) > policy.max_nodes {
        return None;
    }

    if !is_polynomial_candidate(ctx, expr, policy.max_scan_depth, policy.max_pow_exp_scan) {
        return None;
    }

    let mut vars = Vec::new();
    let poly_opt = expr_to_multipoly(ctx, expr, &mut vars, &policy.poly_budget, policy.var_limit);

    let poly = match poly_opt {
        Some(poly) => poly,
        None => {
            return try_opaque_zero(ctx, expr, policy);
        }
    };

    if vars.len() > policy.max_vars || !poly.is_zero() {
        return None;
    }

    let proof_data =
        build_direct_identity_proof_data(ctx, expr, &vars, &policy.poly_budget, policy.var_limit);
    Some(PolynomialIdentityZeroPlan {
        kind: PolynomialIdentityProofKind::Direct,
        proof_data,
    })
}

fn expr_to_multipoly(
    ctx: &Context,
    id: ExprId,
    vars: &mut Vec<String>,
    budget: &PolyBudget,
    var_limit: usize,
) -> Option<MultiPoly> {
    let poly = try_multipoly_from_expr_with_var_limit(ctx, id, budget, var_limit)?;
    *vars = poly.vars.clone();
    Some(poly)
}

fn try_reduce_by_reciprocal_power_relation(
    poly: &MultiPoly,
    opaque_var_name: &str,
    root_index: u32,
    base_poly: &MultiPoly,
    budget: &PolyBudget,
) -> Option<MultiPoly> {
    let mut all_vars = poly.vars.clone();
    for var in &base_poly.vars {
        if !all_vars.iter().any(|existing| existing == var) {
            all_vars.push(var.clone());
        }
    }

    let t_idx = all_vars.iter().position(|v| v == opaque_var_name)?;
    let mut current = poly.align_vars(&all_vars);
    let base_aligned = base_poly.align_vars(&all_vars);
    if base_aligned.degree_in(t_idx) != 0 {
        return None;
    }

    let mut relation_map = base_aligned.neg().to_map();
    let mut root_mono = vec![0; all_vars.len()];
    root_mono[t_idx] = root_index;
    let entry = relation_map
        .entry(root_mono)
        .or_insert_with(BigRational::zero);
    *entry = entry.clone() + BigRational::one();
    let relation = MultiPoly::from_map(all_vars.clone(), relation_map);

    let mut reduction_steps = 0usize;
    loop {
        let candidate_idx = current
            .terms
            .iter()
            .enumerate()
            .filter(|(_, (_, mono))| mono[t_idx] >= root_index)
            .max_by_key(|(_, (_, mono))| mono[t_idx])
            .map(|(idx, _)| idx);

        let Some(idx) = candidate_idx else {
            break;
        };

        let (coeff, mono) = current.terms[idx].clone();
        let mut q_mono = mono;
        q_mono[t_idx] -= root_index;

        let shifted = relation.mul_monomial(&q_mono).ok()?.mul_scalar(&coeff);
        current = current.sub(&shifted).ok()?;

        reduction_steps += 1;
        if reduction_steps > 256
            || current.num_terms() > budget.max_terms.saturating_mul(8)
            || current.total_degree() > budget.max_total_degree.saturating_mul(8)
        {
            return None;
        }
    }

    Some(current)
}

fn try_opaque_zero(
    ctx: &mut Context,
    expr: ExprId,
    policy: &PolynomialIdentityPolicy,
) -> Option<PolynomialIdentityZeroPlan> {
    let calls = collect_function_calls(ctx, expr, policy.max_atoms);
    let unique_calls = dedup_expr_ids(ctx, &calls);

    let exp_exponents = collect_exp_exponents(ctx, expr, policy.max_scan_depth);
    let exp_base = find_exp_base(ctx, &exp_exponents, policy.max_pow_exp_scan);

    let total_atoms = unique_calls.len() + usize::from(exp_base.is_some());
    if total_atoms == 0 || total_atoms > policy.max_atoms {
        return None;
    }

    const SUBSCRIPTS: [char; 10] = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
    let display_name = |i: usize| -> String {
        if total_atoms == 1 {
            "t".to_string()
        } else if i < 10 {
            format!("t{}", SUBSCRIPTS[i])
        } else {
            format!("t{}", i)
        }
    };

    let mut tmp_ctx = ctx.clone();
    let mut sub_expr = expr;
    let mut substitutions: Vec<(String, ExprId)> = Vec::new();
    let mut atom_idx = 0;
    let mut reciprocal_power_relations: Vec<(usize, ExprId, u32)> = Vec::new();

    if let Some(base) = exp_base {
        let temp_name = format!("__opq{}", atom_idx);
        let temp_var = tmp_ctx.var(&temp_name);
        sub_expr = substitute_exp_atoms(
            &mut tmp_ctx,
            sub_expr,
            base,
            temp_var,
            policy.max_scan_depth,
            policy.max_pow_exp_scan,
        );
        let e_const = ctx.add(Expr::Constant(Constant::E));
        let exp_display = ctx.add(Expr::Pow(e_const, base));
        substitutions.push((display_name(atom_idx), exp_display));
        atom_idx += 1;
    }

    for &call_id in &unique_calls {
        if extract_opaque_reciprocal_power_base(ctx, call_id).is_none() {
            continue;
        }
        let temp_name = format!("__opq{}", atom_idx);
        let temp_var = tmp_ctx.var(&temp_name);
        sub_expr = substitute_power_aware(
            &mut tmp_ctx,
            sub_expr,
            call_id,
            temp_var,
            SubstituteOptions {
                power_aware: true,
                ..Default::default()
            },
        );
        if let Some((base_expr, root_index)) = extract_opaque_reciprocal_power_base(ctx, call_id) {
            reciprocal_power_relations.push((atom_idx, base_expr, root_index));
        }
        substitutions.push((display_name(atom_idx), call_id));
        atom_idx += 1;
    }

    for &call_id in &unique_calls {
        if extract_opaque_reciprocal_power_base(ctx, call_id).is_some() {
            continue;
        }
        if let Some((base_expr, numer, denom)) = extract_opaque_rational_power_atom(ctx, call_id) {
            if let Some((root_atom_idx, _, _)) =
                reciprocal_power_relations
                    .iter()
                    .find(|(_, rel_base, root_index)| {
                        *root_index == denom
                            && compare_expr(ctx, *rel_base, base_expr) == std::cmp::Ordering::Equal
                    })
            {
                let root_var = tmp_ctx.var(&format!("__opq{}", root_atom_idx));
                let replacement = if numer == 1 {
                    root_var
                } else {
                    let numer_expr = tmp_ctx.num(numer as i64);
                    tmp_ctx.add(Expr::Pow(root_var, numer_expr))
                };
                sub_expr = substitute_power_aware(
                    &mut tmp_ctx,
                    sub_expr,
                    call_id,
                    replacement,
                    SubstituteOptions::exact(),
                );
                continue;
            }
        }

        let temp_name = format!("__opq{}", atom_idx);
        let temp_var = tmp_ctx.var(&temp_name);
        sub_expr = substitute_power_aware(
            &mut tmp_ctx,
            sub_expr,
            call_id,
            temp_var,
            SubstituteOptions {
                power_aware: true,
                ..Default::default()
            },
        );
        substitutions.push((display_name(atom_idx), call_id));
        atom_idx += 1;
    }

    let mut vars = Vec::new();
    let poly = expr_to_multipoly(
        &tmp_ctx,
        sub_expr,
        &mut vars,
        &policy.poly_budget,
        policy.var_limit,
    )?;
    let proof_kind = if poly.is_zero() {
        PolynomialIdentityProofKind::OpaqueSubstitution
    } else if reciprocal_power_relations.len() == 1 {
        let (opaque_atom_idx, base_expr, root_index) = &reciprocal_power_relations[0];
        let mut base_vars = Vec::new();
        let base_poly = expr_to_multipoly(
            &tmp_ctx,
            *base_expr,
            &mut base_vars,
            &policy.poly_budget,
            policy.var_limit,
        )?;
        let reduced = try_reduce_by_reciprocal_power_relation(
            &poly,
            &format!("__opq{}", opaque_atom_idx),
            *root_index,
            &base_poly,
            &policy.poly_budget,
        )?;
        if !reduced.is_zero() {
            return None;
        }
        PolynomialIdentityProofKind::OpaqueRootRelation
    } else {
        return None;
    };

    let mut display_expr = expr;
    let mut disp_idx = 0;

    if let Some(base) = exp_base {
        let disp_var = ctx.var(&display_name(disp_idx));
        display_expr = substitute_exp_atoms(
            ctx,
            display_expr,
            base,
            disp_var,
            policy.max_scan_depth,
            policy.max_pow_exp_scan,
        );
        disp_idx += 1;
    }

    for &call_id in &unique_calls {
        if extract_opaque_reciprocal_power_base(ctx, call_id).is_none() {
            continue;
        }
        let disp_var = ctx.var(&display_name(disp_idx));
        display_expr = substitute_power_aware(
            ctx,
            display_expr,
            call_id,
            disp_var,
            SubstituteOptions {
                power_aware: true,
                ..Default::default()
            },
        );
        disp_idx += 1;
    }

    for &call_id in &unique_calls {
        if extract_opaque_reciprocal_power_base(ctx, call_id).is_some() {
            continue;
        }
        if let Some((base_expr, numer, denom)) = extract_opaque_rational_power_atom(ctx, call_id) {
            if let Some((root_atom_idx, _, _)) =
                reciprocal_power_relations
                    .iter()
                    .find(|(_, rel_base, root_index)| {
                        *root_index == denom
                            && compare_expr(ctx, *rel_base, base_expr) == std::cmp::Ordering::Equal
                    })
            {
                let root_var = ctx.var(&display_name(*root_atom_idx));
                let replacement = if numer == 1 {
                    root_var
                } else {
                    let numer_expr = ctx.num(numer as i64);
                    ctx.add(Expr::Pow(root_var, numer_expr))
                };
                display_expr = substitute_power_aware(
                    ctx,
                    display_expr,
                    call_id,
                    replacement,
                    SubstituteOptions::exact(),
                );
                continue;
            }
        }

        let disp_var = ctx.var(&display_name(disp_idx));
        display_expr = substitute_power_aware(
            ctx,
            display_expr,
            call_id,
            disp_var,
            SubstituteOptions {
                power_aware: true,
                ..Default::default()
            },
        );
        disp_idx += 1;
    }

    let display_vars: Vec<String> = vars
        .iter()
        .map(|v| {
            if let Some(idx_str) = v.strip_prefix("__opq") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    return display_name(idx);
                }
            }
            v.clone()
        })
        .collect();

    let expanded_form_expr =
        cas_math::multipoly_display::expand_additive_terms(ctx, display_expr, &display_vars);

    let mut proof_data = PolynomialProofData {
        monomials: 0,
        degree: 0,
        vars: display_vars,
        normal_form_expr: Some(display_expr),
        expanded_form_expr,
        lhs_stats: None,
        rhs_stats: None,
        opaque_substitutions: Vec::new(),
    };
    proof_data.opaque_substitutions = substitutions;
    Some(PolynomialIdentityZeroPlan {
        kind: proof_kind,
        proof_data,
    })
}

fn build_direct_identity_proof_data(
    ctx: &mut Context,
    expr: ExprId,
    vars: &[String],
    budget: &PolyBudget,
    var_limit: usize,
) -> PolynomialProofData {
    let (positive_terms, negative_terms) = split_additive_terms(ctx, expr);
    let zero = ctx.num(0);

    if positive_terms.is_empty() || negative_terms.is_empty() {
        return PolynomialProofData {
            monomials: 0,
            degree: 0,
            vars: vars.to_vec(),
            normal_form_expr: Some(zero),
            expanded_form_expr: None,
            lhs_stats: None,
            rhs_stats: None,
            opaque_substitutions: Vec::new(),
        };
    }

    let mut lhs_poly = MultiPoly::zero(vars.to_vec());
    for &term in &positive_terms {
        let mut term_vars = vars.to_vec();
        if let Some(term_poly) = expr_to_multipoly(ctx, term, &mut term_vars, budget, var_limit) {
            if term_poly.vars == lhs_poly.vars {
                if let Ok(sum) = lhs_poly.add(&term_poly) {
                    lhs_poly = sum;
                }
            }
        }
    }

    let mut rhs_poly = MultiPoly::zero(vars.to_vec());
    for &term in &negative_terms {
        let mut term_vars = vars.to_vec();
        if let Some(term_poly) = expr_to_multipoly(ctx, term, &mut term_vars, budget, var_limit) {
            if term_poly.vars == rhs_poly.vars {
                if let Ok(sum) = rhs_poly.add(&term_poly) {
                    rhs_poly = sum;
                }
            }
        }
    }

    PolynomialProofData::from_identity(ctx, &lhs_poly, &rhs_poly, vars.to_vec())
}

fn split_additive_terms(ctx: &Context, expr: ExprId) -> (Vec<ExprId>, Vec<ExprId>) {
    fn collect_terms(ctx: &Context, e: ExprId, pos: &mut Vec<ExprId>, neg: &mut Vec<ExprId>) {
        match ctx.get(e) {
            Expr::Add(a, b) => {
                collect_terms(ctx, *a, pos, neg);
                collect_terms(ctx, *b, pos, neg);
            }
            Expr::Sub(a, b) => {
                collect_terms(ctx, *a, pos, neg);
                neg.push(*b);
            }
            Expr::Neg(inner) => {
                neg.push(*inner);
            }
            _ => pos.push(e),
        }
    }

    let mut pos = Vec::new();
    let mut neg = Vec::new();
    collect_terms(ctx, expr, &mut pos, &mut neg);
    (pos, neg)
}

#[cfg(test)]
mod tests {
    use super::{
        try_prove_polynomial_identity_zero_expr, PolynomialIdentityPolicy,
        PolynomialIdentityProofKind,
    };
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn proves_direct_polynomial_identity() {
        let mut ctx = Context::new();
        let expr = parse("(x+1)^2 - (x^2 + 2*x + 1)", &mut ctx).unwrap_or_else(|_| panic!("parse"));
        let plan = try_prove_polynomial_identity_zero_expr(&mut ctx, expr)
            .unwrap_or_else(|| panic!("plan"));
        assert_eq!(plan.kind, PolynomialIdentityProofKind::Direct);
    }

    #[test]
    fn rejects_non_identity() {
        let mut ctx = Context::new();
        let expr = parse("(x+1)^2 - x^2", &mut ctx).unwrap_or_else(|_| panic!("parse"));
        assert!(try_prove_polynomial_identity_zero_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn proves_opaque_identity_by_substitution() {
        let mut ctx = Context::new();
        let expr = parse("sin(x) + 1 - (sin(x) + 1)", &mut ctx).unwrap_or_else(|_| panic!("parse"));
        let plan = try_prove_polynomial_identity_zero_expr(&mut ctx, expr)
            .unwrap_or_else(|| panic!("plan"));
        assert_eq!(plan.kind, PolynomialIdentityProofKind::OpaqueSubstitution);
    }

    #[test]
    fn proves_opaque_fractional_power_identity_by_substitution() {
        let mut ctx = Context::new();
        let expr = parse(
            "(x^2 + 1)^(1/2)*((x^2 + 1)^(1/2) + 1) - (((x^2 + 1)^(1/2))^2 + (x^2 + 1)^(1/2))",
            &mut ctx,
        )
        .unwrap_or_else(|_| panic!("parse"));
        let plan = try_prove_polynomial_identity_zero_expr(&mut ctx, expr)
            .unwrap_or_else(|| panic!("plan"));
        assert_eq!(plan.kind, PolynomialIdentityProofKind::OpaqueSubstitution);
    }

    #[test]
    fn proves_collapsed_root_base_identity_by_relation() {
        let mut ctx = Context::new();
        let expr = parse(
            "(x^2 + 1)^(1/2)*((x^2 + 1)^(1/2) + 1) - ((x^2 + 1)^(1/2) + x^2 + 1)",
            &mut ctx,
        )
        .unwrap_or_else(|_| panic!("parse"));
        let plan = try_prove_polynomial_identity_zero_expr(&mut ctx, expr)
            .unwrap_or_else(|| panic!("plan"));
        assert_eq!(plan.kind, PolynomialIdentityProofKind::OpaqueRootRelation);
    }

    #[test]
    fn proves_collapsed_root_cubic_identity_by_relation() {
        let mut ctx = Context::new();
        let expr = parse(
            "(x^2 + 1)^(3/2) + 1 - (((x^2 + 1)^(1/2) + 1)*(x^2 + 2 - (x^2 + 1)^(1/2)))",
            &mut ctx,
        )
        .unwrap_or_else(|_| panic!("parse"));
        let plan = try_prove_polynomial_identity_zero_expr(&mut ctx, expr)
            .unwrap_or_else(|| panic!("plan"));
        assert_eq!(plan.kind, PolynomialIdentityProofKind::OpaqueRootRelation);
    }

    #[test]
    fn policy_max_nodes_blocks_large_expr() {
        let mut ctx = Context::new();
        let expr = parse("(x+1)^2 - (x^2 + 2*x + 1)", &mut ctx).unwrap_or_else(|_| panic!("parse"));
        let policy = PolynomialIdentityPolicy {
            max_nodes: 1,
            ..PolynomialIdentityPolicy::default()
        };
        assert!(
            super::try_prove_polynomial_identity_zero_with_policy(&mut ctx, expr, &policy)
                .is_none()
        );
    }
}
