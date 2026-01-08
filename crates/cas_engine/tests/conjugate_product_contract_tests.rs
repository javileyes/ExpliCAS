//! Contract tests for conjugate product detection and Sophie Germain identity.

use cas_ast::Context;
use cas_engine::ordering::compare_expr;
use cas_parser::parse;
use std::cmp::Ordering;

/// Debug test to verify structural equality of terms.
#[test]
fn test_conjugate_compare_terms() {
    use cas_engine::nary::{add_terms_signed, Sign};

    let mut ctx = Context::new();

    let l = parse("a^2 + 2*b^2 + 2*a*b", &mut ctx).expect("parse L");
    let r = parse("a^2 + 2*b^2 - 2*a*b", &mut ctx).expect("parse R");

    let l_terms = add_terms_signed(&ctx, l);
    let r_terms = add_terms_signed(&ctx, r);

    // Check structural equality between corresponding terms
    println!("\n=== Term Comparison ===");
    for (i, (l_term, l_sign)) in l_terms.iter().enumerate() {
        for (j, (r_term, r_sign)) in r_terms.iter().enumerate() {
            let eq = compare_expr(&ctx, *l_term, *r_term);
            let l_s = if matches!(l_sign, Sign::Pos) {
                "+"
            } else {
                "-"
            };
            let r_s = if matches!(r_sign, Sign::Pos) {
                "+"
            } else {
                "-"
            };
            let l_disp = cas_ast::display::DisplayExpr {
                context: &ctx,
                id: *l_term,
            };
            let r_disp = cas_ast::display::DisplayExpr {
                context: &ctx,
                id: *r_term,
            };
            if eq == Ordering::Equal {
                println!(
                    "L[{}] ({}{}) == R[{}] ({}{})",
                    i, l_s, l_disp, j, r_s, r_disp
                );
            }
        }
    }

    // The test: L[2] should equal R[2] structurally (both are 2*a*b)
    // Even though one has Sign::Pos and other has Sign::Neg
    let l2_term = l_terms[2].0;
    let r2_term = r_terms[2].0;
    let cmp = compare_expr(&ctx, l2_term, r2_term);
    println!("\nDirect comparison L[2] vs R[2]: {:?}", cmp);
    assert_eq!(cmp, Ordering::Equal, "2*a*b should be structurally equal");
}
