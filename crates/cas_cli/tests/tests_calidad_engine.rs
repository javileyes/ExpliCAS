//! # Tests de Calidad del Motor CAS
//!
//! Esta suite contiene tests que verifican la calidad y robustez del motor
//! de simplificación. Estos tests representan casos "Boss Final" que requieren
//! un manejo muy fino de la simplificación algebraica.

use cas_engine::Simplifier;
use cas_format::Format;
use cas_parser::parse;

/// Helper para simplificar una expresión y obtener su representación como string
fn simplify_expr(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input, &mut simplifier.context).expect("Failed to parse");
    let (result, _) = simplifier.simplify(expr);
    result.to_latex(&simplifier.context)
}

/// Test #64: Suma Cíclica Racional
///
/// $$\frac{1}{(a-b)(a-c)} + \frac{1}{(b-c)(b-a)} + \frac{1}{(c-a)(c-b)}$$
///
/// Este test es el "Boss Final" de las fracciones algebraicas.
/// Para que dé 0, el CAS tiene que darse cuenta de que $(a-c) = -(c-a)$.
///
/// Si el sistema de Canonical Ordering está bien hecho, normalizará
/// automáticamente $(c-a)$ vs $(a-c)$ (extrayendo un -1 para que la
/// variable alfabéticamente menor vaya primero).
///
/// - Sin Canonical Ordering: El denominador común será gigante (producto de 6 términos).
/// - Con Canonical Ordering: El sistema verá que solo hay 3 términos únicos (con cambios de signo).
#[test]
fn test_cyclic_rational_sum_to_zero() {
    let result = simplify_expr("1/((a-b)*(a-c)) + 1/((b-c)*(b-a)) + 1/((c-a)*(c-b))");
    assert_eq!(
        result, "0",
        "Suma cíclica racional debe simplificar a 0. Resultado: {}",
        result
    );
}

/// Variante del test cíclico con expansión explícita
#[test]
fn test_cyclic_rational_sum_expanded() {
    // Esta variante usa la forma expandida donde los signos ya están más visibles
    let result = simplify_expr("1/((a-b)*(a-c)) - 1/((b-c)*(a-b)) + 1/((a-c)*(b-c))");
    assert_eq!(
        result, "0",
        "Suma cíclica (forma 2) debe simplificar a 0. Resultado: {}",
        result
    );
}

/// Test de verificación: El motor debe reconocer que (a-b) = -(b-a)
#[test]
fn test_opposite_subtraction_recognition() {
    // (a - b) + (b - a) debe ser 0
    let result = simplify_expr("(a - b) + (b - a)");
    assert_eq!(
        result, "0",
        "Términos opuestos (a-b) + (b-a) deben cancelarse. Resultado: {}",
        result
    );
}

/// Test de fracciones con denominadores opuestos simples
#[test]
fn test_opposite_denominator_fractions() {
    // 1/(a-b) + 1/(b-a) = 1/(a-b) - 1/(a-b) = 0
    let result = simplify_expr("1/(a-b) + 1/(b-a)");
    assert_eq!(
        result, "0",
        "Fracciones con denominadores opuestos deben cancelarse. Resultado: {}",
        result
    );
}

/// Test de productos con un factor opuesto (stepping stone hacia Boss Final)
#[test]
fn test_product_fraction_with_opposite_factor() {
    // 1/((a-b)*(a-c)) + 1/((a-c)*(b-a)) = 0
    // Ambos comparten (a-c), y (a-b) vs (b-a) son opuestos
    let result = simplify_expr("1/((a-b)*(a-c)) + 1/((a-c)*(b-a))");
    assert_eq!(
        result, "0",
        "Productos con factor común y factor opuesto deben cancelarse. Resultado: {}",
        result
    );
}
