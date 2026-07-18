use super::{
    format_matrix_function_desc, try_eval_matrix_add_expr, try_eval_matrix_function_expr,
    try_eval_matrix_mul_expr, try_eval_scalar_matrix_mul_expr,
    try_rewrite_matrix_function_rule_expr, try_rewrite_transpose_product_expr,
    try_rewrite_transpose_product_identity_expr, MatrixFunctionEval, ScalarMatrixSide,
};
use cas_ast::Expr;

#[test]
fn matrix_add_eval_extracts_shapes_and_result() {
    let mut ctx = cas_ast::Context::new();
    let one = ctx.num(1);
    let two = ctx.num(2);
    let three = ctx.num(3);
    let four = ctx.num(4);
    let m1 = ctx.matrix(2, 2, vec![one, two, three, four]).expect("m1");
    let m2 = ctx.matrix(2, 2, vec![one, two, three, four]).expect("m2");
    let expr = ctx.add(Expr::Add(m1, m2));

    let eval = try_eval_matrix_add_expr(&mut ctx, expr).expect("matrix add");
    assert_eq!(eval.left.rows, 2);
    assert_eq!(eval.right.cols, 2);
    assert_eq!(eval.result.rows, 2);
    assert_eq!(eval.result.cols, 2);
}

#[test]
fn matrix_mul_eval_extracts_shapes_and_result() {
    let mut ctx = cas_ast::Context::new();
    let one = ctx.num(1);
    let two = ctx.num(2);
    let three = ctx.num(3);
    let four = ctx.num(4);
    let m1 = ctx.matrix(2, 2, vec![one, two, three, four]).expect("m1");
    let m2 = ctx.matrix(2, 2, vec![one, two, three, four]).expect("m2");
    let expr = ctx.add(Expr::Mul(m1, m2));

    let eval = try_eval_matrix_mul_expr(&mut ctx, expr).expect("matrix mul");
    assert_eq!(eval.result.rows, 2);
    assert_eq!(eval.result.cols, 2);
}

#[test]
fn scalar_matrix_eval_detects_side() {
    let mut ctx = cas_ast::Context::new();
    let scalar = ctx.num(2);
    let one = ctx.num(1);
    let two = ctx.num(2);
    let three = ctx.num(3);
    let four = ctx.num(4);
    let matrix = ctx
        .matrix(2, 2, vec![one, two, three, four])
        .expect("matrix");
    let expr = ctx.add(Expr::Mul(scalar, matrix));

    let eval = try_eval_scalar_matrix_mul_expr(&mut ctx, expr).expect("scalar*matrix");
    assert_eq!(eval.side, ScalarMatrixSide::ScalarLeft);
    assert_eq!(eval.matrix.rows, 2);
}

#[test]
fn matrix_function_eval_handles_transpose() {
    let mut ctx = cas_ast::Context::new();
    let one = ctx.num(1);
    let two = ctx.num(2);
    let three = ctx.num(3);
    let four = ctx.num(4);
    let matrix = ctx
        .matrix(2, 2, vec![one, two, three, four])
        .expect("matrix");
    let expr = ctx.call("transpose", vec![matrix]);

    let eval = try_eval_matrix_function_expr(&mut ctx, expr, false).expect("transpose");
    match eval {
        MatrixFunctionEval::Transpose { from, to, .. } => {
            assert_eq!(from.rows, 2);
            assert_eq!(to.cols, 2);
        }
        _ => panic!("expected transpose"),
    }
}

#[test]
fn transpose_product_rewrite_builds_expected_shape() {
    let mut ctx = cas_ast::Context::new();
    let a = ctx.var("A");
    let b = ctx.var("B");
    let matmul = ctx.call("matmul", vec![a, b]);
    let expr = ctx.call("transpose", vec![matmul]);

    let rewritten = try_rewrite_transpose_product_expr(&mut ctx, expr).expect("expected rewrite");
    let Expr::Function(fn_id, args) = ctx.get(rewritten) else {
        panic!("expected function");
    };
    assert_eq!(ctx.sym_name(*fn_id), "matmul");
    assert_eq!(args.len(), 2);
}

#[test]
fn transpose_product_identity_rewrite_has_desc() {
    let mut ctx = cas_ast::Context::new();
    let a = ctx.var("A");
    let b = ctx.var("B");
    let matmul = ctx.call("matmul", vec![a, b]);
    let expr = ctx.call("transpose", vec![matmul]);
    let rewrite = try_rewrite_transpose_product_identity_expr(&mut ctx, expr).expect("rewrite");
    assert_eq!(rewrite.desc, "(AB)^T = B^T·A^T");
}

#[test]
fn matrix_function_rule_rewrite_materializes_transpose_matrix() {
    let mut ctx = cas_ast::Context::new();
    let one = ctx.num(1);
    let two = ctx.num(2);
    let three = ctx.num(3);
    let four = ctx.num(4);
    let matrix = ctx
        .matrix(2, 2, vec![one, two, three, four])
        .expect("matrix");
    let expr = ctx.call("transpose", vec![matrix]);

    let rewrite = try_rewrite_matrix_function_rule_expr(&mut ctx, expr, false).expect("rewrite");
    let materialized =
        cas_math::matrix::Matrix::from_expr(&ctx, rewrite.rewritten).expect("matrix");
    assert_eq!(materialized.rows, 2);
    assert_eq!(materialized.cols, 2);
    assert!(rewrite.desc.contains("transpose"));
}

#[test]
fn description_builders_include_shapes() {
    let mut ctx = cas_ast::Context::new();
    let one = ctx.num(1);
    let two = ctx.num(2);
    let three = ctx.num(3);
    let four = ctx.num(4);
    let matrix = ctx
        .matrix(2, 2, vec![one, two, three, four])
        .expect("matrix");
    let scalar = ctx.num(2);
    let expr_mul = ctx.add(Expr::Mul(scalar, matrix));
    let scalar_eval = try_eval_scalar_matrix_mul_expr(&mut ctx, expr_mul).expect("eval");
    assert!(scalar_eval.desc().contains("2x2 matrix") || scalar_eval.desc().contains("2×2 matrix"));

    let m1 = ctx.matrix(2, 2, vec![one, two, three, four]).expect("m1");
    let m2 = ctx.matrix(2, 2, vec![one, two, three, four]).expect("m2");
    let add_expr = ctx.add(Expr::Add(m1, m2));
    let add_eval = try_eval_matrix_add_expr(&mut ctx, add_expr).expect("matrix add");
    assert!(add_eval.add_desc().contains("2x2 + 2x2") || add_eval.add_desc().contains("2×2 + 2×2"));

    let mul_expr = ctx.add(Expr::Mul(m1, m2));
    let mul_eval = try_eval_matrix_mul_expr(&mut ctx, mul_expr).expect("matrix mul");
    assert!(mul_eval.mul_desc().contains("= 2x2") || mul_eval.mul_desc().contains("= 2×2"));

    let transposed = ctx.call("transpose", vec![m1]);
    let function_eval =
        try_eval_matrix_function_expr(&mut ctx, transposed, false).expect("transpose");
    assert!(format_matrix_function_desc(&function_eval).contains("transpose(2"));
}

#[test]
fn map_matrix_components_is_all_or_nothing() {
    // The V1 primitive: one failing component declines the WHOLE map — a half-transformed
    // matrix is never emitted.
    let mut ctx = cas_ast::Context::new();
    let one = ctx.num(1);
    let two = ctx.num(2);
    let v = ctx.matrix(2, 1, vec![one, two]).expect("vector");

    let doubled = super::map_matrix_components(&mut ctx, v, |ctx, e| {
        let two = ctx.num(2);
        Some(ctx.add(Expr::Mul(two, e)))
    })
    .expect("total map succeeds");
    let m = cas_math::matrix::Matrix::from_expr(&ctx, doubled).expect("matrix result");
    assert_eq!((m.rows, m.cols), (2, 1));

    let partial = super::map_matrix_components(&mut ctx, v, |ctx, e| {
        // Fail exactly on the `2` entry.
        if cas_math::numeric_eval::as_rational_const(ctx, e)
            .is_some_and(|r| r == num_rational::BigRational::from_integer(2.into()))
        {
            None
        } else {
            Some(e)
        }
    });
    assert!(partial.is_none(), "partial map must decline entirely");
}

#[test]
fn componentwise_diff_matrix_derives_each_entry() {
    // diff([x^2, x^3], x) → [2x, 3x^2] via the shared support differentiator.
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let two = ctx.num(2);
    let three = ctx.num(3);
    let x2 = ctx.add(Expr::Pow(x, two));
    let x3 = ctx.add(Expr::Pow(x, three));
    let v = ctx.matrix(2, 1, vec![x2, x3]).expect("vector");

    let derived = super::try_componentwise_diff_matrix(&mut ctx, v, "x").expect("derives");
    let m = cas_math::matrix::Matrix::from_expr(&ctx, derived).expect("matrix result");
    assert_eq!((m.rows, m.cols), (2, 1));
    // Sanity on the first entry: d/dx x² must mention x and be a product/power form,
    // not the untouched x².
    assert_ne!(m.data[0], x2);
}

#[test]
fn matmul_dispatch_multiplies_and_declines_mismatch() {
    // `matmul` was in the eval gate but missing from the dispatch (silent-residual gotcha).
    let mut ctx = cas_ast::Context::new();
    let one = ctx.num(1);
    let two = ctx.num(2);
    let three = ctx.num(3);
    let four = ctx.num(4);
    let a = ctx.matrix(1, 2, vec![one, two]).expect("a");
    let b = ctx.matrix(2, 1, vec![three, four]).expect("b");
    let call = ctx.call("matmul", vec![a, b]);
    let rewrite =
        super::try_rewrite_matrix_binary_function_expr(&mut ctx, call).expect("matmul evaluates");
    assert!(rewrite.desc.contains("matrix multiplication"));

    // 1×2 · 1×2 is dimension-mismatched → decline (honest residual), not undefined.
    let c = ctx.matrix(1, 2, vec![three, four]).expect("c");
    let bad = ctx.call("matmul", vec![a, c]);
    assert!(super::try_rewrite_matrix_binary_function_expr(&mut ctx, bad).is_none());
}

#[test]
fn gradient_expr_assembles_column_and_declines_dishonest_shapes() {
    // V3: gradient of a SCALAR field is an n×1 COLUMN; a Matrix field (jacobian
    // territory) and an over-cap var list decline honestly.
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");
    let two = ctx.num(2);
    let x2 = ctx.add(Expr::Pow(x, two));
    let field = ctx.add(Expr::Mul(x2, y));
    let vars: Vec<String> = vec!["x".into(), "y".into()];

    let grad = super::try_gradient_expr(&mut ctx, field, &vars).expect("gradient");
    let m = cas_math::matrix::Matrix::from_expr(&ctx, grad).expect("matrix");
    assert_eq!((m.rows, m.cols), (2, 1), "gradient must be an n×1 column");

    let mfield = ctx.matrix(2, 1, vec![x, y]).expect("vector field");
    assert!(
        super::try_gradient_expr(&mut ctx, mfield, &vars).is_none(),
        "matrix field is jacobian territory (V4) — decline"
    );

    let too_many: Vec<String> = (0..9).map(|i| format!("v{i}")).collect();
    assert!(
        super::try_gradient_expr(&mut ctx, field, &too_many).is_none(),
        "var cap (VERB_MAX_VARS=8) must decline honestly"
    );
}

#[test]
fn jacobian_orientation_rows_are_functions() {
    // V4 orientation pin (BEFORE the fixtures existed): J[i][j] = ∂fᵢ/∂vⱼ —
    // rows = functions, columns = variables. Hash-consing makes ExprId equality
    // a sound cell-level check.
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");
    let two = ctx.num(2);
    let x2 = ctx.add(Expr::Pow(x, two));
    let f0 = ctx.add(Expr::Mul(x2, y));
    let f1 = ctx.add(Expr::Add(x, y));
    let fields = ctx.matrix(2, 1, vec![f0, f1]).expect("vector");
    let vars: Vec<String> = vec!["x".into(), "y".into()];

    let jac = super::try_jacobian_expr(&mut ctx, fields, &vars).expect("jacobian");
    let m = cas_math::matrix::Matrix::from_expr(&ctx, jac).expect("matrix");
    assert_eq!((m.rows, m.cols), (2, 2));
    use cas_math::symbolic_differentiation_support::differentiate_symbolic_expr;
    let df0_dy = differentiate_symbolic_expr(&mut ctx, f0, "y").unwrap();
    let df1_dx = differentiate_symbolic_expr(&mut ctx, f1, "x").unwrap();
    assert_eq!(
        m.data[1], df0_dy,
        "cell (0,1) must be ∂f₀/∂y (rows=functions)"
    );
    assert_eq!(m.data[2], df1_dx, "cell (1,0) must be ∂f₁/∂x");

    // Scalar target declines (gradient owns scalars); 2×2 matrix target declines.
    assert!(super::try_jacobian_expr(&mut ctx, f0, &vars).is_none());
    let one = ctx.num(1);
    let sq = ctx.matrix(2, 2, vec![one, one, one, one]).expect("2x2");
    assert!(super::try_jacobian_expr(&mut ctx, sq, &vars).is_none());
}

#[test]
fn hessian_is_symmetric_and_declines_matrix_field() {
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");
    let two = ctx.num(2);
    let x2 = ctx.add(Expr::Pow(x, two));
    let field = ctx.add(Expr::Mul(x2, y));
    let vars: Vec<String> = vec!["x".into(), "y".into()];

    let hes = super::try_hessian_expr(&mut ctx, field, &vars).expect("hessian");
    let m = cas_math::matrix::Matrix::from_expr(&ctx, hes).expect("matrix");
    assert_eq!((m.rows, m.cols), (2, 2));
    assert_eq!(
        m.data[1], m.data[2],
        "H must be symmetric (Clairaut) cell-for-cell"
    );

    let mfield = ctx.matrix(2, 1, vec![x, y]).expect("vector");
    assert!(super::try_hessian_expr(&mut ctx, mfield, &vars).is_none());
}

#[test]
fn divergence_requires_component_var_match_and_laplacian_declines_matrix() {
    // V5: ∇·F needs #components == #vars (mismatch → honest residual); Δ of a vector
    // field is a named scope-out (vector-laplacian), not an error.
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");
    let two = ctx.num(2);
    let x2 = ctx.add(Expr::Pow(x, two));
    let y2 = ctx.add(Expr::Pow(y, two));
    let field = ctx.matrix(2, 1, vec![x2, y2]).expect("vector");
    let vars2: Vec<String> = vec!["x".into(), "y".into()];
    let vars3: Vec<String> = vec!["x".into(), "y".into(), "z".into()];

    assert!(super::try_divergence_expr(&mut ctx, field, &vars2).is_some());
    assert!(
        super::try_divergence_expr(&mut ctx, field, &vars3).is_none(),
        "2 components with 3 vars must decline"
    );
    assert!(
        super::try_laplacian_expr(&mut ctx, field, &vars2).is_none(),
        "vector-laplacian is a scope-out: Matrix target declines"
    );
    let scalar_sum = ctx.add(Expr::Add(x2, y2));
    assert!(super::try_laplacian_expr(&mut ctx, scalar_sum, &vars2).is_some());
}
