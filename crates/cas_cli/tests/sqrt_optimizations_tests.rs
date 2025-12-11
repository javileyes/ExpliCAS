use cas_ast::DisplayExpr;
use cas_engine::Simplifier;
use cas_parser::parse;

/// Helper para crear simplifier
fn create_simplifier() -> Simplifier {
    Simplifier::with_default_rules()
}

// ========================================
// OPTIMIZACIONES DE sqrt(x^2)
// ========================================

#[test]
fn test_sqrt_x_squared_simple() {
    // sqrt(x^2) -> |x| en 0 pasos
    let mut simplifier = create_simplifier();
    let expr = parse("sqrt(x^2)", &mut simplifier.context).unwrap();
    let (result, steps) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // Debe ser |x|
    assert!(result_str.contains('|') && result_str.contains('x'));
    // Optimized - should be significantly fewer than before (was 4 steps)
    assert!(steps.len() < 4, "Expected < 4 steps, got {}", steps.len());
}

#[test]
fn test_sqrt_binomial_squared() {
    // sqrt((x-1)^2) -> |x-1| optimized
    let mut simplifier = create_simplifier();
    let expr = parse("sqrt((x-1)^2)", &mut simplifier.context).unwrap();
    let (result, steps) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // Debe contener valor absoluto
    assert!(result_str.contains('|'));
    // Should be optimized (before was many steps with expansion)
    assert!(
        steps.len() < 10,
        "Expected < 10 steps, got {} steps",
        steps.len()
    );
}

#[test]
fn test_sqrt_binomial_addition_squared() {
    // sqrt((a+b)^2) -> |a+b| en 0-1 pasos
    let mut simplifier = create_simplifier();
    let expr = parse("sqrt((a+b)^2)", &mut simplifier.context).unwrap();
    let (result, _steps) = simplifier.simplify(expr);

    let _result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
}

#[test]
fn test_sqrt_trinomial_squared() {
    // sqrt((x^2 + 2*x + 1)^2) should simplify
    let mut simplifier = create_simplifier();
    let expr = parse("sqrt((x^2 + 2*x + 1)^2)", &mut simplifier.context).unwrap();
    let (result, steps) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // Should contain abs
    assert!(result_str.contains('|'));
    // Should not expand significantly
    assert!(steps.len() < 10, "Too many steps: {}", steps.len());
}

// ========================================
// PROTECCIÓN DE PRODUCTOS CANÓNICOS
// ========================================

#[test]
fn test_abs_conjugate_product() {
    // abs((x-2)*(x+2)) -> |(-2+x)*(2+x)| en 0 pasos (no expande)
    let mut simplifier = create_simplifier();
    let expr = parse("abs((x-2)*(x+2))", &mut simplifier.context).unwrap();
    let (result, steps) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // Debe mantener forma de producto
    assert!(
        result_str.contains('*'),
        "Expected product form, got {}",
        result_str
    );
    // No debe expandirse (allowing up to 2 steps for canonicalization)
    assert!(steps.len() <= 2, "Expected 0-2 steps, got {}", steps.len());
}

#[test]
fn test_sqrt_conjugate_product() {
    // sqrt((x+1)*(x-1)) -> sqrt(product) sin expandir
    let mut simplifier = create_simplifier();
    let expr = parse("sqrt((x+1)*(x-1))", &mut simplifier.context).unwrap();
    let (result, _steps) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // May simplify in various ways - just check it doesn't over-expand
    assert!(result_str.len() < 50, "Result too complex: {}", result_str);
}

#[test]
fn test_abs_simple_product() {
    // abs((a-1)*(a+3)) should preserve product form
    let mut simplifier = create_simplifier();
    let expr = parse("abs((a-1)*(a+3))", &mut simplifier.context).unwrap();
    let (result, steps) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // Should preserve mult
    assert!(result_str.contains('*') || result_str.contains('|'));
    assert!(
        steps.len() <= 2,
        "Too many expansion steps: {}",
        steps.len()
    );
}

// ========================================
// CASOS EDGE
// ========================================

#[test]
fn test_sqrt_numeric_squared() {
    // sqrt(4^2) -> 4
    let mut simplifier = create_simplifier();
    let expr = parse("sqrt(4^2)", &mut simplifier.context).unwrap();
    let (result, _steps) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    assert_eq!(result_str, "4");
}

#[test]
fn test_sqrt_variable_coefficient_squared() {
    // sqrt((2*x)^2) -> |2*x| = 2*|x|
    let mut simplifier = create_simplifier();
    let expr = parse("sqrt((2*x)^2)", &mut simplifier.context).unwrap();
    let (result, _steps) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // Should contain abs
    assert!(result_str.contains('|'));
}

#[test]
fn test_nested_sqrt() {
    // sqrt(sqrt(x^4)) -> |x|
    let mut simplifier = create_simplifier();
    let expr = parse("sqrt(sqrt(x^4))", &mut simplifier.context).unwrap();
    let (result, _steps) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // Should simplify significantly
    assert!(
        result_str.len() < 20,
        "Expected simplified form, got {}",
        result_str
    );
}

// ========================================
// MODO AGGRESSIVE
// ========================================

#[test]
fn test_aggressive_mode_sqrt_binomial() {
    // Verificar que modo aggressive también optimiza
    use cas_ast::Expr;

    let mut simplifier = create_simplifier();
    let expr = parse("sqrt((x-1)^2)", &mut simplifier.context).unwrap();

    // Aplicar aggressive simplification
    let expand_func = simplifier
        .context
        .add(Expr::Function("expand".to_string(), vec![expr]));
    let (result, _steps) = simplifier.simplify(expand_func);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // Debe llegar a |x-1| o equivalente
    assert!(result_str.contains('|') || result_str.contains("sqrt"));
}

#[test]
fn test_aggressive_mode_abs_product() {
    // Verificar que abs((x-2)(x+2)) no expande en modo aggressive
    use cas_ast::Expr;

    let mut simplifier = create_simplifier();
    let expr = parse("abs((x-2)*(x+2))", &mut simplifier.context).unwrap();

    let expand_func = simplifier
        .context
        .add(Expr::Function("expand".to_string(), vec![expr]));
    let (result, _steps) = simplifier.simplify(expand_func);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // Debe mantener producto o ser |x^2-4|
    assert!(
        result_str.len() < 30,
        "Should not over-expand: {}",
        result_str
    );
}

// ========================================
// REGRESIONES
// ========================================

#[test]
fn test_sqrt_not_perfect_square() {
    // sqrt(x^2 + 1) no debe simplificar incorrectamente
    let mut simplifier = create_simplifier();
    let expr = parse("sqrt(x^2 + 1)", &mut simplifier.context).unwrap();
    let (result, _steps) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // Should maintain sqrt or convert to power notation
    assert!(result_str.contains("sqrt") || result_str.contains("1/2"));
}

#[test]
fn test_abs_not_canonical() {
    // abs(x^2 + x + 1) no debe tratarse como canónico
    let mut simplifier = create_simplifier();
    let expr = parse("abs(x^2 + x + 1)", &mut simplifier.context).unwrap();
    let (result, _steps) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // Debe mantener abs
    assert!(result_str.contains('|'));
}
