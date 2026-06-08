use cas_ast::{Context, ExprId};

pub(super) use super::atanh_open_interval_domain::{
    atanh_known_empty_open_interval_gap, atanh_open_interval_condition,
    atanh_self_normalized_surd_quotient_positive_gap, collect_atanh_open_interval_conditions,
};
pub(super) use super::bounded_inverse_trig_open_interval_domain::{
    bounded_inverse_trig_known_empty_open_interval_gap,
    inverse_reciprocal_trig_bounded_trig_empty_open_interval_gap,
};
pub(crate) use super::diff_domain_undefined_targets::diff_target_known_undefined_over_reals;
pub(super) use super::diff_required_conditions::{
    inverse_reciprocal_trig_affine_abs_required_conditions,
    reciprocal_trig_and_log_diff_required_conditions,
};
pub(super) use super::root_required_conditions::{
    positive_polynomial_radicand_and_nonzero_required_conditions,
    positive_polynomial_radicand_required_conditions, shifted_sqrt_product_required_conditions,
};

pub(super) fn positive_required_conditions_first(
    required_positive: ExprId,
    mut required_conditions: Vec<crate::ImplicitCondition>,
) -> Vec<crate::ImplicitCondition> {
    required_conditions.insert(0, crate::ImplicitCondition::Positive(required_positive));
    required_conditions
}

pub(super) fn append_positive_required_conditions(
    target: &mut Vec<crate::ImplicitCondition>,
    required_positive: ExprId,
    required_conditions: impl IntoIterator<Item = crate::ImplicitCondition>,
) {
    target.push(crate::ImplicitCondition::Positive(required_positive));
    target.extend(required_conditions);
}

pub(super) fn diff_target_known_undefined_or_empty_domain_over_reals(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    diff_target_known_undefined_over_reals(ctx, target, var_name)
        || atanh_known_empty_open_interval_gap(ctx, target).is_some()
        || inverse_reciprocal_trig_bounded_trig_empty_open_interval_gap(ctx, target, var_name)
            .is_some()
        || bounded_inverse_trig_known_empty_open_interval_gap(ctx, target, var_name).is_some()
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::{
        atanh_known_empty_open_interval_gap, atanh_open_interval_condition,
        bounded_inverse_trig_known_empty_open_interval_gap, diff_target_known_undefined_over_reals,
    };

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn bounded_inverse_trig_empty_open_interval_gap_detects_root_quadratic() {
        let mut ctx = Context::new();
        let target = parse("arcsin(sqrt(x^2+1))", &mut ctx).unwrap();
        let gap = bounded_inverse_trig_known_empty_open_interval_gap(&mut ctx, target, "x")
            .expect("empty open interval gap");

        assert_eq!(rendered(&ctx, gap), "-(x^2)");

        let negated_target = parse("-arcsin(sqrt(x^2+1))", &mut ctx).unwrap();
        let negated_gap =
            bounded_inverse_trig_known_empty_open_interval_gap(&mut ctx, negated_target, "x")
                .expect("negated empty open interval gap");
        assert_eq!(rendered(&ctx, negated_gap), "-(x^2)");

        let shifted_target = parse("arcsin((x+1)^2+1)", &mut ctx).unwrap();
        assert!(
            bounded_inverse_trig_known_empty_open_interval_gap(&mut ctx, shifted_target, "x")
                .is_some()
        );

        let shifted_negative_target = parse("arccos(-((x+1)^2+1))", &mut ctx).unwrap();
        assert!(bounded_inverse_trig_known_empty_open_interval_gap(
            &mut ctx,
            shifted_negative_target,
            "x"
        )
        .is_some());

        for source in ["arcsin(-1)", "arccos(-1)"] {
            let finite_boundary_target = parse(source, &mut ctx).unwrap();
            assert!(
                bounded_inverse_trig_known_empty_open_interval_gap(
                    &mut ctx,
                    finite_boundary_target,
                    "x"
                )
                .is_none(),
                "source: {source}"
            );
        }

        let symbolic_constant_target = parse("arcsin(pi)", &mut ctx).unwrap();
        let symbolic_gap = bounded_inverse_trig_known_empty_open_interval_gap(
            &mut ctx,
            symbolic_constant_target,
            "x",
        )
        .expect("symbolic constant outside unit interval");
        assert_eq!(rendered(&ctx, symbolic_gap), "1 - pi^2");

        let negative_symbolic_constant_target = parse("arccos(-e)", &mut ctx).unwrap();
        assert!(bounded_inverse_trig_known_empty_open_interval_gap(
            &mut ctx,
            negative_symbolic_constant_target,
            "x"
        )
        .is_some());

        let shifted_symbolic_constant_target = parse("pi - arccos(e)", &mut ctx).unwrap();
        assert!(bounded_inverse_trig_known_empty_open_interval_gap(
            &mut ctx,
            shifted_symbolic_constant_target,
            "x"
        )
        .is_some());
    }

    #[test]
    fn atanh_empty_open_interval_gap_detects_direct_quadratic_and_wrappers() {
        let mut ctx = Context::new();
        let target = parse("atanh(x^2+1)", &mut ctx).unwrap();
        let gap =
            atanh_known_empty_open_interval_gap(&mut ctx, target).expect("empty interval gap");

        assert_eq!(rendered(&ctx, gap), "-x^4 - 2 * x^2");

        let scaled_target = parse("2*atanh(x^2+1)", &mut ctx).unwrap();
        let scaled_gap = atanh_known_empty_open_interval_gap(&mut ctx, scaled_target)
            .expect("scaled empty interval gap");
        assert_eq!(rendered(&ctx, scaled_gap), "-x^4 - 2 * x^2");

        let shifted_target = parse("atanh((x+1)^2+1)", &mut ctx).unwrap();
        assert!(atanh_known_empty_open_interval_gap(&mut ctx, shifted_target).is_some());

        let symbolic_constant_target = parse("atanh(pi)", &mut ctx).unwrap();
        let symbolic_gap = atanh_known_empty_open_interval_gap(&mut ctx, symbolic_constant_target)
            .expect("symbolic constant outside open interval");
        assert_eq!(rendered(&ctx, symbolic_gap), "1 - pi^2");

        let symbolic_offset_target = parse("atanh(phi+x^2)", &mut ctx).unwrap();
        assert!(
            atanh_known_empty_open_interval_gap(&mut ctx, symbolic_offset_target).is_some(),
            "named constant offset outside open interval should be rejected early"
        );
    }

    #[test]
    fn atanh_open_interval_condition_compacts_scaled_shifted_sqrt_arg() {
        let mut ctx = Context::new();
        let arg = parse("a*sqrt(x+1)", &mut ctx).unwrap();
        let condition = atanh_open_interval_condition(&mut ctx, arg);

        assert_eq!(rendered(&ctx, condition), "1 - a^2 * x - a^2");
    }

    #[test]
    fn atanh_open_interval_condition_compacts_symbolic_denominator_scaled_sqrt_arg() {
        let mut ctx = Context::new();
        let arg = parse("sqrt(x+1)/a", &mut ctx).unwrap();
        let condition = atanh_open_interval_condition(&mut ctx, arg);

        assert_eq!(rendered(&ctx, condition), "a^2 - x - 1");
    }

    #[test]
    fn atanh_open_interval_condition_compacts_external_denominator_scaled_sqrt_arg() {
        let mut ctx = Context::new();
        let arg = parse("2*sqrt(x+1)/a", &mut ctx).unwrap();
        let condition = atanh_open_interval_condition(&mut ctx, arg);

        assert_eq!(rendered(&ctx, condition), "a^2 - 4 * x - 4");
    }

    #[test]
    fn diff_target_known_undefined_over_reals_detects_nonfinite_log_and_root_domains() {
        let mut ctx = Context::new();
        for source in [
            "-infinity",
            "ln(0)",
            "log2(0)",
            "log(2, 0)",
            "log(1, 2)",
            "log(1, x)",
            "log(-2, x)",
        ] {
            let target = parse(source, &mut ctx).unwrap();
            assert!(
                diff_target_known_undefined_over_reals(&mut ctx, target, "x"),
                "source: {source}"
            );
        }

        for source in ["sqrt(-1)", "sqrt(-x^2-1)", "sqrt(-x^2)"] {
            let target = parse(source, &mut ctx).unwrap();
            assert!(
                diff_target_known_undefined_over_reals(&mut ctx, target, "x"),
                "source: {source}"
            );
        }

        for source in ["sqrt(0)", "sqrt(4)", "ln(1)", "log2(1)", "log(2, 1)"] {
            let target = parse(source, &mut ctx).unwrap();
            assert!(
                !diff_target_known_undefined_over_reals(&mut ctx, target, "x"),
                "source: {source}"
            );
        }
    }
}
