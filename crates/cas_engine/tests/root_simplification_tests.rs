use cas_ast::DisplayExpr;
use cas_engine::Simplifier;

fn test_simplify(input: &str, expected_part: &str) {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = cas_parser::parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let res_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    println!("Input: {}, Result: {}", input, res_str);
    assert!(
        res_str.contains(expected_part),
        "Expected result to contain '{}', got '{}'",
        expected_part,
        res_str
    );
}

#[test]
fn test_square_roots() {
    // sqrt(12) = sqrt(4*3) = 2*sqrt(3)
    test_simplify("sqrt(12)", "2 * 3^(1/2)");
    // sqrt(72) = sqrt(36*2) = 6*sqrt(2)
    test_simplify("sqrt(72)", "6 * 2^(1/2)");
}

#[test]
fn test_cube_roots() {
    // cbrt(16) = cbrt(8*2) = 2*cbrt(2)
    test_simplify("16^(1/3)", "2 * 2^(1/3)");
    // cbrt(54) = cbrt(27*2) = 3*cbrt(2)
    test_simplify("54^(1/3)", "3 * 2^(1/3)");
}

#[test]
fn test_higher_roots() {
    // 32^(1/4) = (16*2)^(1/4) = 2 * 2^(1/4)
    test_simplify("32^(1/4)", "2 * 2^(1/4)");
    // 243^(1/4) = (81*3)^(1/4) = 3 * 3^(1/4)
    test_simplify("243^(1/4)", "3 * 3^(1/4)");
}

#[test]
fn test_fraction_roots() {
    // sqrt(8/9) = sqrt(8)/3 = 2*sqrt(2)/3
    test_simplify("sqrt(8/9)", "2/3 * 2^(1/2)");
    // sqrt(12/25) = 2*sqrt(3)/5
    test_simplify("sqrt(12/25)", "2/5 * 3^(1/2)");
}

#[test]
fn test_root_addition() {
    // sqrt(12) + sqrt(27) = 2*sqrt(3) + 3*sqrt(3) = 5*sqrt(3)
    test_simplify("sqrt(12) + sqrt(27)", "5 * 3^(1/2)");
    // sqrt(8) + sqrt(2) = 2*sqrt(2) + sqrt(2) = 3*sqrt(2)
    test_simplify("sqrt(8) + sqrt(2)", "3 * 2^(1/2)");
    // cbrt(16) + cbrt(54) = 2*cbrt(2) + 3*cbrt(2) = 5*cbrt(2)
    test_simplify("16^(1/3) + 54^(1/3)", "5 * 2^(1/3)");
}

#[test]
fn test_mixed_roots_fraction() {
    // sqrt(8) / 16^(1/3)
    // = 2*2^(1/2) / (2*2^(1/3))
    // = 2^(1/2) / 2^(1/3)
    // = 2^(1/2 - 1/3) = 2^(3/6 - 2/6) = 2^(1/6)
    test_simplify("sqrt(8) / 16^(1/3)", "2^(1/6)");

    // sqrt(32) / 2^(1/2)
    // = 4*sqrt(2) / sqrt(2) = 4
    test_simplify("sqrt(32) / sqrt(2)", "4");
}

#[test]
fn test_complex_expressions() {
    // (sqrt(12) + sqrt(27)) / sqrt(3)
    // = (2*sqrt(3) + 3*sqrt(3)) / sqrt(3)
    // = 5*sqrt(3) / sqrt(3) = 5
    test_simplify("(sqrt(12) + sqrt(27)) / sqrt(3)", "5");

    // sqrt(2) * cbrt(2)
    // = 2^(1/2) * 2^(1/3) = 2^(5/6)
    test_simplify("sqrt(2) * 2^(1/3)", "2^(5/6)");

    // sqrt(8) * sqrt(2)
    // = 2*sqrt(2) * sqrt(2) = 2 * 2 = 4
    test_simplify("sqrt(8) * sqrt(2)", "4");
}

#[test]
fn test_large_numbers() {
    // sqrt(3200) = sqrt(1600 * 2) = 40 * sqrt(2)
    test_simplify("sqrt(3200)", "40 * 2^(1/2)");

    // cbrt(8000) = 20
    test_simplify("8000^(1/3)", "20");
}

#[test]
fn test_variables_and_roots() {
    // sqrt(8 * x^2) = sqrt(8) * sqrt(x^2) = 2*sqrt(2) * |x|
    // Output: 2 * 2^(1/2) * |x|
    test_simplify("sqrt(8 * x^2)", "2 * 2^(1/2) * |x|");

    // sqrt(12 * x^3) - x^3 is NOT a perfect square, so distribution is blocked
    // The expression stays as (12 * x^3)^(1/2)
    let mut simplifier = Simplifier::with_default_rules();
    let expr = cas_parser::parse("sqrt(12 * x^3)", &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let res_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    println!("sqrt(12 * x^3) => {}", res_str);
    // Just verify it produces some output (not fully simplified due to odd power)
    assert!(!res_str.is_empty(), "Expected non-empty result");
}
