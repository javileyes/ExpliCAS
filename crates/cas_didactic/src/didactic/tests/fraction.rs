use super::super::fraction_steps::generate_fraction_sum_substeps;
use super::super::fraction_sum_analysis::FractionSumInfo;
use cas_ast::{Context, Expr};
use cas_math::expr_predicates::contains_division_like_term;
use num_bigint::BigInt;
use num_rational::BigRational;

#[test]
fn test_fraction_sum_substeps() {
    let fractions = vec![
        BigRational::new(BigInt::from(1), BigInt::from(24)),
        BigRational::new(BigInt::from(1), BigInt::from(2)),
        BigRational::new(BigInt::from(1), BigInt::from(6)),
    ];
    let result: BigRational = fractions.iter().cloned().sum();

    let info = FractionSumInfo {
        fractions,
        result: result.clone(),
    };

    let substeps = generate_fraction_sum_substeps(&info);
    assert!(!substeps.is_empty());
    assert_eq!(result, BigRational::new(BigInt::from(17), BigInt::from(24)));
}

#[test]
fn test_contains_div_simple() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(1))));

    assert!(!contains_division_like_term(&ctx, x));

    let div = ctx.add(Expr::Div(one, x));
    assert!(contains_division_like_term(&ctx, div));

    let add = ctx.add(Expr::Add(one, div));
    assert!(contains_division_like_term(&ctx, add));
}
