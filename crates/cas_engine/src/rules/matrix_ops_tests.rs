use super::matrix_ops::{MatrixAddRule, MatrixMultiplyRule, ScalarMatrixRule};
use crate::rule::SimpleRule;
use cas_ast::{Context, Expr};

#[test]
fn test_matrix_add_rule() {
    let mut ctx = Context::new();
    let rule = MatrixAddRule;

    // [[1, 2], [3, 4]] + [[5, 6], [7, 8]]
    let n1 = ctx.num(1);
    let n2 = ctx.num(2);
    let n3 = ctx.num(3);
    let n4 = ctx.num(4);
    let n5 = ctx.num(5);
    let n6 = ctx.num(6);
    let n7 = ctx.num(7);
    let n8 = ctx.num(8);

    let m1 = ctx.matrix(2, 2, vec![n1, n2, n3, n4]).unwrap();
    let m2 = ctx.matrix(2, 2, vec![n5, n6, n7, n8]).unwrap();
    let add_expr = ctx.add(Expr::Add(m1, m2));

    let result = rule.apply_simple(&mut ctx, add_expr);
    assert!(result.is_some());
}

#[test]
fn test_scalar_matrix_rule() {
    let mut ctx = Context::new();
    let rule = ScalarMatrixRule;

    // 2 * [[1, 2], [3, 4]]
    let scalar = ctx.num(2);
    let n1 = ctx.num(1);
    let n2 = ctx.num(2);
    let n3 = ctx.num(3);
    let n4 = ctx.num(4);

    let matrix = ctx.matrix(2, 2, vec![n1, n2, n3, n4]).unwrap();
    let mul_expr = ctx.add(Expr::Mul(scalar, matrix));

    let result = rule.apply_simple(&mut ctx, mul_expr);
    assert!(result.is_some());
}

#[test]
fn test_matrix_multiply_rule() {
    let mut ctx = Context::new();
    let rule = MatrixMultiplyRule;

    // [[1, 2], [3, 4]] * [[5, 6], [7, 8]]
    let n1 = ctx.num(1);
    let n2 = ctx.num(2);
    let n3 = ctx.num(3);
    let n4 = ctx.num(4);
    let n5 = ctx.num(5);
    let n6 = ctx.num(6);
    let n7 = ctx.num(7);
    let n8 = ctx.num(8);

    let m1 = ctx.matrix(2, 2, vec![n1, n2, n3, n4]).unwrap();
    let m2 = ctx.matrix(2, 2, vec![n5, n6, n7, n8]).unwrap();
    let mul_expr = ctx.add(Expr::Mul(m1, m2));

    let result = rule.apply_simple(&mut ctx, mul_expr);
    assert!(result.is_some());
}

#[test]
fn test_matrix_multiply_rule_non_square_is_budget_exempt() {
    // 2x2 * 2x3: each output entry is a 2-term unfolded sum, so the rewrite grows the
    // tree past the anti-worsen budget. The rule must mark it budget-exempt so the valid
    // product is not rejected by the simplifier (which would fall through to the buggy
    // scalar-broadcast rule).
    let mut ctx = Context::new();
    let rule = MatrixMultiplyRule;
    let n: Vec<_> = (1..=10).map(|i| ctx.num(i)).collect();
    let m1 = ctx.matrix(2, 2, vec![n[0], n[1], n[2], n[3]]).unwrap();
    let m2 = ctx
        .matrix(2, 3, vec![n[4], n[5], n[6], n[7], n[8], n[9]])
        .unwrap();
    let mul_expr = ctx.add(Expr::Mul(m1, m2));

    let rewrite = rule
        .apply_simple(&mut ctx, mul_expr)
        .expect("non-square product");
    assert!(
        rewrite.budget_exempt,
        "matrix multiply rewrite must be budget-exempt"
    );
    let result = cas_math::matrix::Matrix::from_expr(&ctx, rewrite.new_expr).expect("matrix");
    assert_eq!(result.rows, 2);
    assert_eq!(result.cols, 3);
}

#[test]
fn test_matrix_multiply_rule_declines_dimension_mismatch() {
    // 2x3 * 2x2: inner dimensions 3 != 2, so there is no product. The rule must decline
    // (leaving an honest residual) rather than fabricate a result.
    let mut ctx = Context::new();
    let rule = MatrixMultiplyRule;
    let n: Vec<_> = (1..=10).map(|i| ctx.num(i)).collect();
    let m1 = ctx
        .matrix(2, 3, vec![n[0], n[1], n[2], n[3], n[4], n[5]])
        .unwrap();
    let m2 = ctx.matrix(2, 2, vec![n[6], n[7], n[8], n[9]]).unwrap();
    let mul_expr = ctx.add(Expr::Mul(m1, m2));

    assert!(rule.apply_simple(&mut ctx, mul_expr).is_none());
}

#[test]
fn test_scalar_matrix_rule_declines_matrix_times_matrix() {
    // Both operands are matrices: this is matrix-matrix multiplication, not a scalar
    // broadcast. ScalarMatrixRule must decline so MatrixMultiplyRule (or an honest
    // residual on mismatch) handles it instead of corrupting it into a matrix-of-matrices.
    let mut ctx = Context::new();
    let rule = ScalarMatrixRule;
    let n: Vec<_> = (1..=8).map(|i| ctx.num(i)).collect();
    let m1 = ctx.matrix(2, 2, vec![n[0], n[1], n[2], n[3]]).unwrap();
    let m2 = ctx.matrix(2, 2, vec![n[4], n[5], n[6], n[7]]).unwrap();
    let mul_expr = ctx.add(Expr::Mul(m1, m2));

    assert!(rule.apply_simple(&mut ctx, mul_expr).is_none());
}
