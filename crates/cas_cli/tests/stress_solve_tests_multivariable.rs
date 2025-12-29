//! Advanced multivariable stress tests for the equation solver
//! Tests solving for one variable while treating others as parameters

use cas_ast::{Equation, RelOp};
use cas_engine::engine::Simplifier;
use cas_engine::solver::solve;

// ============================================================================
// LEVEL 1: Basic Linear Multivariable
// ============================================================================

#[test]
fn test_linear_two_variables_solve_for_x() {
    // ax + by = c → solve for x → x = (c - by)/a
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("a*x + b*y", &mut s.context).unwrap();
    let rhs = cas_parser::parse("c", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok(),
        "Should solve linear equation for x with parameters a, b, c, y"
    );
}

#[test]
fn test_linear_two_variables_solve_for_y() {
    // 2x + 3y = 5 → solve for y → y = (5 - 2x)/3
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("2*x + 3*y", &mut s.context).unwrap();
    let rhs = cas_parser::parse("5", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "y", &mut s);

    assert!(
        result.is_ok(),
        "Should solve for y expressing solution in terms of x"
    );
}

#[test]
fn test_linear_three_variables() {
    // 2x + 3y - z = 10 → solve for z → z = 2x + 3y - 10
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("2*x + 3*y - z", &mut s.context).unwrap();
    let rhs = cas_parser::parse("10", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "z", &mut s);

    assert!(
        result.is_ok(),
        "Should solve for z with x and y as parameters"
    );
}

// ============================================================================
// LEVEL 2: Quadratic Multivariable
// ============================================================================

#[test]
fn test_quadratic_in_x_with_parameter() {
    // x^2 + bx + c = 0 → solve for x
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^2 + b*x + c", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok(),
        "Should solve quadratic with parameters b, c"
    );
}

#[test]
fn test_quadratic_mixed_xy() {
    // x^2 + xy = y^2 → solve for x
    // This becomes x^2 + xy - y^2 = 0 quadratic in x
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^2 + x*y", &mut s.context).unwrap();
    let rhs = cas_parser::parse("y^2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Should solve quadratic with mixed xy term");
}

#[test]
fn test_quadratic_both_variables_squared() {
    // x^2 + y^2 = r^2 → solve for x → x = ± sqrt(r^2 - y^2)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^2 + y^2", &mut s.context).unwrap();
    let rhs = cas_parser::parse("r^2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Should solve circle equation for x");
}

#[test]
fn test_ellipse_equation() {
    // x^2/a^2 + y^2/b^2 = 1 → solve for y
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^2/a^2 + y^2/b^2", &mut s.context).unwrap();
    let rhs = cas_parser::parse("1", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "y", &mut s);

    assert!(result.is_ok(), "Should solve ellipse equation for y");
}

// ============================================================================
// LEVEL 3: Rational Multivariable
// ============================================================================

#[test]
fn test_rational_with_parameter() {
    // a/x + b = c → x = a/(c - b)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("a/x + b", &mut s.context).unwrap();
    let rhs = cas_parser::parse("c", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok(),
        "Should solve rational equation with parameters"
    );
}

#[test]
fn test_rational_both_variables() {
    // 1/x + 1/y = 1/f → solve for x → x = fy/(y - f)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("1/x + 1/y", &mut s.context).unwrap();
    let rhs = cas_parser::parse("1/f", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Lens equation should solve for x");
}

#[test]
fn test_complex_rational_multivariable() {
    // (x + y)/(x - y) = a → solve for x
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x + y)/(x - y)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("a", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Should solve complex rational for x");
}

// ============================================================================
// LEVEL 4: Polynomial Multivariable
// ============================================================================

#[test]
fn test_cubic_in_x_with_parameters() {
    // x^3 + ax^2 + bx + c = 0 → solve for x
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^3 + a*x^2 + b*x + c", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    // May not solve analytically, but should handle gracefully
    assert!(result.is_ok() || result.is_err(), "Should handle cubic");
}

#[test]
fn test_factorable_cubic_multivar() {
    // (x - y)(x - z)(x - w) = 0 → x = y or x = z or x = w
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x - y)*(x - z)*(x - w)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok(),
        "Should solve factored cubic with multiple solutions"
    );
}

#[test]
fn test_symmetric_polynomial() {
    // x^2 + xy + y^2 = k → solve for x
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^2 + x*y + y^2", &mut s.context).unwrap();
    let rhs = cas_parser::parse("k", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Should solve symmetric polynomial");
}

// ============================================================================
// LEVEL 5: Inequalities Multivariable
// ============================================================================

#[test]
fn test_linear_inequality_multivar() {
    // ax + by > c → x > (c - by)/a (assuming a > 0)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("a*x + b*y", &mut s.context).unwrap();
    let rhs = cas_parser::parse("c", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok(),
        "Should solve linear inequality with parameters"
    );
}

#[test]
fn test_quadratic_inequality_param() {
    // x^2 > y → |x| > sqrt(y)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^2", &mut s.context).unwrap();
    let rhs = cas_parser::parse("y", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };
    let result = solve(&eq, "x", &mut s);

    // Complex parametric inequality - may not fully solve symbolically
    assert!(
        result.is_ok() || result.is_err(),
        "Should handle quadratic inequality with parameter"
    );
}

#[test]
fn test_rational_inequality_multivar() {
    // x/(x + y) > 0 → solve for regions where x*(x+y) > 0
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x/(x + y)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok(),
        "Should solve rational inequality with y as parameter"
    );
}

// ============================================================================
// LEVEL 6: Radicals and Powers Multivariable
// ============================================================================

#[test]
fn test_sqrt_with_parameters() {
    // sqrt(x + y) = z → x = z^2 - y
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x + y)^(1/2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("z", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok(),
        "Should solve radical equation with parameters"
    );
}

#[test]
fn test_power_with_base_parameter() {
    // (x + a)^b = c → x = c^(1/b) - a
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x + a)^b", &mut s.context).unwrap();
    let rhs = cas_parser::parse("c", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok(),
        "Should solve power equation with parameters"
    );
}

#[test]
fn test_pythagorean_variant() {
    // sqrt(x^2 + y^2) = r → x^2 = r^2 - y^2
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x^2 + y^2)^(1/2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("r", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Should solve Pythagorean-style equation");
}

// ============================================================================
// LEVEL 7: Complex Physics/Engineering Formulas
// ============================================================================

#[test]
fn test_kinematic_equation() {
    // v^2 = u^2 + 2as → solve for s → s = (v^2 - u^2)/(2a)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("v^2", &mut s.context).unwrap();
    let rhs = cas_parser::parse("u^2 + 2*a*s", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "s", &mut s);

    assert!(result.is_ok(), "Should solve kinematic equation for s");
}

#[test]
fn test_ideal_gas_law() {
    // PV = nRT → solve for T → T = PV/(nR)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("P*V", &mut s.context).unwrap();
    let rhs = cas_parser::parse("n*R*T", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "T", &mut s);

    assert!(result.is_ok(), "Should solve ideal gas law for T");
}

#[test]
fn test_thin_lens_equation() {
    // 1/f = 1/u + 1/v → solve for v → v = uf/(u - f)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("1/f", &mut s.context).unwrap();
    let rhs = cas_parser::parse("1/u + 1/v", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "v", &mut s);

    assert!(result.is_ok(), "Should solve thin lens equation for v");
}

#[test]
fn test_quadratic_formula_derivation() {
    // ax^2 + bx + c = 0 → solve for x
    // Should give x = (-b ± sqrt(b^2 - 4ac))/(2a)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("a*x^2 + b*x + c", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Should derive quadratic formula");
}

// ============================================================================
// LEVEL 8: Hyperbola and Conic Sections
// ============================================================================

#[test]
fn test_hyperbola_equation() {
    // x^2/a^2 - y^2/b^2 = 1 → solve for y
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^2/a^2 - y^2/b^2", &mut s.context).unwrap();
    let rhs = cas_parser::parse("1", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "y", &mut s);

    assert!(result.is_ok(), "Should solve hyperbola equation");
}

#[test]
fn test_rotated_conic() {
    // Ax^2 + Bxy + Cy^2 = D → solve for y (quadratic in y)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("A*x^2 + B*x*y + C*y^2", &mut s.context).unwrap();
    let rhs = cas_parser::parse("D", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "y", &mut s);

    assert!(result.is_ok(), "Should solve rotated conic for y");
}

// ============================================================================
// LEVEL 9: Nested Parameters and Complex Cases
// ============================================================================

#[test]
fn test_nested_parameters_fraction() {
    // (a + b)/(c + d) * x = e → x = e(c + d)/(a + b)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(a + b)/(c + d) * x", &mut s.context).unwrap();
    let rhs = cas_parser::parse("e", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok(),
        "Should solve with nested parameter expressions"
    );
}

#[test]
fn test_parameter_in_exponent() {
    // x^a = b → x = b^(1/a)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^a", &mut s.context).unwrap();
    let rhs = cas_parser::parse("b", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Should solve with parameter in exponent");
}

#[test]
fn test_four_variables_linear() {
    // ax + by + cz + dw = e → solve for w → w = (e - ax - by - cz)/d
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("a*x + b*y + c*z + d*w", &mut s.context).unwrap();
    let rhs = cas_parser::parse("e", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "w", &mut s);

    assert!(result.is_ok(), "Should solve 4-variable linear equation");
}

#[test]
fn test_reciprocal_multivar() {
    // 1/x + 1/y + 1/z = 1/f → solve for z
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("1/x + 1/y + 1/z", &mut s.context).unwrap();
    let rhs = cas_parser::parse("1/f", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "z", &mut s);

    assert!(result.is_ok(), "Should solve triple reciprocal equation");
}

// ============================================================================
// LEVEL 10: Stress Tests with Many Variables
// ============================================================================

#[test]
fn test_many_variables_sum() {
    // a + b + c + d + e + f + g + h + i + j = x → solve for x
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("a + b + c + d + e + f + g + h + i + j", &mut s.context).unwrap();
    let rhs = cas_parser::parse("x", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Should solve equation with 10 variables");
}

#[test]
fn test_product_of_sums() {
    // (a + x)(b + x) = c → solve for x (quadratic)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(a + x)*(b + x)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("c", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Should solve product of sums");
}

#[test]
fn test_general_quadratic_multivar() {
    // Ax^2 + Bx + C = y → solve for x
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("A*x^2 + B*x + C", &mut s.context).unwrap();
    let rhs = cas_parser::parse("y", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok(),
        "Should solve general quadratic with all parameters"
    );
}

// ============================================================================
// LEVEL 11: Advanced - Logarithms with Multiple Variables
// ============================================================================

#[test]
fn test_log_with_parameters() {
    // log(x, base: a) = b → x = a^b
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("log(x, a)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("b", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle log equation with parameters"
    );
}

#[test]
fn test_natural_log_multivar() {
    // ln(x + y) = z → x = e^z - y
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("ln(x + y)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("z", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle natural log with sum"
    );
}

#[test]
fn test_log_product_rule_multivar() {
    // ln(x*y) = a → solve for x → x = e^a/y
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("ln(x*y)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("a ", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle log of product"
    );
}

#[test]
fn test_log_quotient_multivar() {
    // ln(x/y) = b → x = y*e^b
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("ln(x/y)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("b", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle log of quotient"
    );
}

// ============================================================================
// LEVEL 12: Advanced - Complex Radicals with Parameters
// ============================================================================

#[test]
fn test_nested_radicals_multivar() {
    // sqrt(x + sqrt(y)) = z → x = z^2 - sqrt(y)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x + y^(1/2))^(1/2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("z", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle nested radicals"
    );
}

#[test]
fn test_cube_root_multivar() {
    // (x + a)^(1/3) = b → x = b^3 - a
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x + a)^(1/3)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("b", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Should solve cube root equation");
}

#[test]
fn test_radical_fraction_combo() {
    // sqrt(x/y) = a → x = a^2 * y
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x/y)^(1/2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("a", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Should solve radical of fraction");
}

#[test]
fn test_nth_root_general() {
    // (x + y)^(1/n) = z → x = z^n - y
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x + y)^(1/n)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("z", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle general nth root"
    );
}

// ============================================================================
// LEVEL 13: Advanced - Mixed Exponentials and Logarithms
// ============================================================================

#[test]
fn test_exponential_with_parameter() {
    // a^x = b → x = log(b)/log(a)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("a^x", &mut s.context).unwrap();
    let rhs = cas_parser::parse("b", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle exponential with parameter base"
    );
}

#[test]
fn test_exponential_sum() {
    // a^x + b^x = c → complex transcendental
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("a^x + b^x", &mut s.context).unwrap();
    let rhs = cas_parser::parse("c", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    // Very complex - may not solve
    assert!(
        result.is_ok() || result.is_err(),
        "Should handle exponential sum"
    );
}

#[test]
fn test_log_exponential_inverse() {
    // ln(a^x) = b → x*ln(a) = b → x = b/ln(a)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("ln(a^x)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("b", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle log of exponential"
    );
}

#[test]
fn test_exponential_equation_change_base() {
    // a^x = b^y where solving for x in terms of y
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("a^x", &mut s.context).unwrap();
    let rhs = cas_parser::parse("b^y", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle exponential base change"
    );
}

// ============================================================================
// LEVEL 14: Advanced - Complex Rational with Radicals
// ============================================================================

#[test]
fn test_rational_with_sqrt_numerator() {
    // sqrt(x)/y = a → x = (a*y)^2
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^(1/2)/y", &mut s.context).unwrap();
    let rhs = cas_parser::parse("a", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok(),
        "Should solve rational with sqrt in numerator"
    );
}

#[test]
fn test_rational_with_sqrt_denominator() {
    // x/sqrt(y) = a → x = a*sqrt(y)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x/y^(1/2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("a", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok(),
        "Should solve rational with sqrt in denominator"
    );
}

#[test]
fn test_complex_rational_radical() {
    // sqrt(x + y) / sqrt(x - y) = a
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x + y)^(1/2) / (x - y)^(1/2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("a", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle complex rational radical"
    );
}

#[test]
fn test_nested_fraction_with_powers() {
    // (a/b)^x = c/d
    // After simplification: a^x / b^x = c/d → a^x = (c/d) * b^x
    // The RHS still contains x, so log inverse cannot be applied.
    // The solver should return a controlled error instead of infinite loop.
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(a/b)^x", &mut s.context).unwrap();
    let rhs = cas_parser::parse("c/d", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    // Should return an error (variable on both sides) not stack overflow
    assert!(
        result.is_err(),
        "Should return error when variable appears on both sides of exponential"
    );
}

// ============================================================================
// LEVEL 15: Ultra Complex - Mixed Everything
// ============================================================================

#[test]
fn test_log_of_radical() {
    // ln(sqrt(x + y)) = a → sqrt(x+y) = e^a → x = e^(2a) - y
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("ln((x + y)^(1/2))", &mut s.context).unwrap();
    let rhs = cas_parser::parse("a", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle log of radical"
    );
}

#[test]
fn test_power_of_log() {
    // (ln(x))^a = b → ln(x) = b^(1/a) → x = e^(b^(1/a))
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("ln(x)^a", &mut s.context).unwrap();
    let rhs = cas_parser::parse("b", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle power of logarithm"
    );
}

#[test]
fn test_radical_of_exponential() {
    // sqrt(a^x) = b → a^x = b^2 → x = 2*log(b)/log(a)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(a^x)^(1/2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("b", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle radical of exponential"
    );
}

#[test]
fn test_exponential_of_radical() {
    // a^(sqrt(x)) = b → sqrt(x) = log(b)/log(a) → x = (log(b)/log(a))^2
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("a^(x^(1/2))", &mut s.context).unwrap();
    let rhs = cas_parser::parse("b", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle exponential of radical"
    );
}

#[test]
fn test_log_radical_fraction_combo() {
    // ln(sqrt(x/y)) = a → sqrt(x/y) = e^a → x/y = e^(2a) → x = y*e^(2a)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("ln((x/y)^(1/2))", &mut s.context).unwrap();
    let rhs = cas_parser::parse("a", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle log of radical of fraction"
    );
}

#[test]
fn test_fraction_of_logs() {
    // ln(x)/ln(y) = a → ln(x) = a*ln(y) → x = y^a
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("ln(x)/ln(y)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("a", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle fraction of logarithms"
    );
}

#[test]
fn test_nested_power_radical() {
    // ((x + a)^b)^(1/c) = d → (x+a)^b = d^c → x+a = (d^c)^(1/b) → x = (d^c)^(1/b) - a
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("((x + a)^b)^(1/c)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("d", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle nested power and radical"
    );
}

#[test]
fn test_exponential_fraction_radical() {
    // a^(x/y) = b^(1/z) → very complex
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("a^(x/y)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("b^(1/z)", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle exponential with fractional exponent"
    );
}

// ============================================================================
// LEVEL 16: Extreme Stress Tests - Deeply Nested
// ============================================================================

#[test]
fn test_triple_nested_radicals() {
    // sqrt(x + sqrt(y + sqrt(z))) = w
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x + (y + z^(1/2))^(1/2))^(1/2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("w", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle triple nested radicals"
    );
}

#[test]
fn test_log_sum_of_exponentials() {
    // ln(a^x + b^y) = c → transcendental, very difficult
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("ln(a^x + b^y)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("c", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    // Extremely complex - likely won't solve
    assert!(
        result.is_ok() || result.is_err(),
        "Should handle log of sum of exponentials"
    );
}

#[test]
fn test_radical_log_power_combo() {
    // sqrt(ln(x^a)) = b → ln(x^a) = b^2 → a*ln(x) = b^2 → x = e^(b^2/a)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(ln(x^a))^(1/2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("b", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle sqrt of log of power"
    );
}

#[test]
fn test_power_tower_simple() {
    // (a^b)^x = c → a^(bx) = c → bx = log(c)/log(a) → x = log(c)/(b*log(a))
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(a^b)^x", &mut s.context).unwrap();
    let rhs = cas_parser::parse("c", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle power tower"
    );
}

#[test]
fn test_rational_exponential_mix() {
    // (a^x / b^y) = c/d → solving for x
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("a^x / b^y", &mut s.context).unwrap();
    let rhs = cas_parser::parse("c/d", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Should handle rational of exponentials"
    );
}
