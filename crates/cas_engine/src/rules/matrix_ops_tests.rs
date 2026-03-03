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
