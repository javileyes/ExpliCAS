use cas_ast::{Context, DisplayExpr};
use cas_engine::matrix;
use cas_parser::parse;

fn main() {
    let mut ctx = Context::new();

    println!("Testing determinant for different matrix sizes:\n");

    // 2x2 matrix
    println!("2×2 Matrix:");
    let m2 = parse("[[1, 2], [3, 4]]", &mut ctx).unwrap();
    if let Some(det) =
        matrix::Matrix::from_expr(&ctx, m2).and_then(|m| matrix::Matrix::determinant(&mut ctx, m2))
    {
        println!(
            " det([[1, 2], [3, 4]]) = {}",
            DisplayExpr {
                context: &ctx,
                id: det
            }
        );
    }

    // 3x3 matrix
    println!("\n3×3 Matrix:");
    let m3 = parse("[[1, 2, 3], [4, 5, 6], [7, 8, 9]]", &mut ctx).unwrap();
    if let Some(det) =
        matrix::Matrix::from_expr(&ctx, m3).and_then(|m| matrix::Matrix::determinant(&mut ctx, m3))
    {
        println!(
            " det([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) = {}",
            DisplayExpr {
                context: &ctx,
                id: det
            }
        );
    }

    // 4x4 matrix
    println!("\n4×4 Matrix:");
    let m4 = parse(
        "[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]",
        &mut ctx,
    )
    .unwrap();
    if let Some(det) =
        matrix::Matrix::from_expr(&ctx, m4).and_then(|m| matrix::Matrix::determinant(&mut ctx, m4))
    {
        println!(
            " det(4×4) = {}",
            DisplayExpr {
                context: &ctx,
                id: det
            }
        );
    }

    // 4x4 identity matrix (should be 1)
    println!("\n4×4 Identity Matrix:");
    let m4_identity = parse(
        "[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]",
        &mut ctx,
    )
    .unwrap();
    if let Some(det) = matrix::Matrix::from_expr(&ctx, m4_identity)
        .and_then(|m| matrix::Matrix::determinant(&mut ctx, m4_identity))
    {
        println!(
            " det(I_4) = {}",
            DisplayExpr {
                context: &ctx,
                id: det
            }
        );
    }

    // 5x5 identity (should be 1)
    println!("\n5×5 Identity Matrix:");
    let m5_identity = parse(
        "[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]",
        &mut ctx,
    )
    .unwrap();
    if let Some(det) = matrix::Matrix::from_expr(&ctx, m5_identity)
        .and_then(|m| matrix::Matrix::determinant(&mut ctx, m5_identity))
    {
        println!(
            " det(I_5) = {}",
            DisplayExpr {
                context: &ctx,
                id: det
            }
        );
    }
}
