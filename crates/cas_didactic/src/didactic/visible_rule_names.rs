use std::borrow::Cow;

pub(crate) fn visible_rule_name(rule_name: &str) -> &str {
    match rule_name {
        "Rationalize Linear Sqrt Denominator" => "Racionalizar el denominador",
        "Subtraction Self-Cancel" => "Restar dos expresiones iguales",
        "Identity Property of Addition" => "Quitar el 0",
        "Identity Property of Multiplication" => "Quitar el factor 1",
        "Evaluate Numeric Power" => "Calcular potencia numérica",
        "Cancel Reciprocal Exponents" => "Deshacer raíz y potencia",
        "Canonicalize Nested Power" => "Reescribir potencia de una raíz",
        "Canonicalize Trig Function Names" => "Usar el nombre arctan",
        "Canonicalize Negation" => "Quitar paréntesis tras el signo menos",
        "Canonicalize" => "Reordenar la expresión",
        "Polynomial Identity" => "Cancelar una identidad exacta",
        "Pre-order Common Factor Cancel" => "Cancelar un factor común",
        "Pre-order Difference of Squares Cancel" => {
            "Factorizar una diferencia de cuadrados y cancelar"
        }
        "Pre-order Sum/Difference of Cubes" => "Factorizar suma o diferencia de cubos",
        "Pre-order Sum/Difference of Cubes Cancel" => "Cancelar factor tras factorizar cubos",
        "Inverse Tan Relations" => "Aplicar identidad de arctangentes",
        "Sqrt Perfect Square" | "Simplify Square Root" => {
            "Reconocer un cuadrado perfecto bajo la raíz"
        }
        "Combine Like Terms" => "Agrupar términos semejantes",
        "Combine Same Denominator Fractions" => "Sumar fracciones con mismo denominador",
        "Common Denominator" => "Llevar a denominador común",
        "Add Fractions" => "Sumar fracciones",
        "Simplify Complex Fraction" => "Simplificar fracción anidada",
        "Auto Expand Power Sum" => "Expandir binomio",
        "Polynomial Product Normalize" => "Expandir y reagrupar un producto polinómico",
        "Pythagorean Chain Identity" => "Aplicar la identidad pitagórica",
        _ => rule_name,
    }
}

pub(crate) fn visible_rule_name_for_step<'a>(
    rule_name: &'a str,
    description: &str,
) -> Cow<'a, str> {
    match rule_name {
        "Rationalize Linear Sqrt Denominator" if description.contains("opaque substitution") => {
            Cow::Borrowed("Reconocer un cociente notable")
        }
        _ => Cow::Borrowed(visible_rule_name(rule_name)),
    }
}

pub(crate) fn visible_step_description<'a>(description: &'a str) -> Cow<'a, str> {
    match description {
        "Rationalize: multiply by conjugate" => Cow::Borrowed("Multiplicar por el conjugado"),
        "1 * x = x" => Cow::Borrowed("Quitar el factor 1"),
        "Evaluate literal power" => Cow::Borrowed("Calcular potencia numérica"),
        "a - a = 0" => Cow::Borrowed("Restar dos expresiones iguales"),
        "Add fractions: a/b + c/d -> (ad+bc)/bd" => Cow::Borrowed("Sumar fracciones"),
        "Common denominator: k + p/q → (k·q + p)/q" => {
            Cow::Borrowed("Llevar a denominador común")
        }
        "Simplify nested fraction" => Cow::Borrowed("Simplificar fracción anidada"),
        "Simplify Complex Fraction" => Cow::Borrowed("Simplificar fracción anidada"),
        "0 + x = x" => Cow::Borrowed("Quitar el 0"),
        "Combine like terms" => Cow::Borrowed("Agrupar términos semejantes"),
        "(x^k)^r = x^(k·r)" => Cow::Borrowed("Reescribir potencia de una raíz"),
        "(u^y)^(1/y) = u" => Cow::Borrowed("Deshacer una raíz con su potencia"),
        "Cancel common factor" => Cow::Borrowed("Cancelar factor común"),
        "atan -> arctan" => Cow::Borrowed("Usar el nombre arctan"),
        "arctan(x) + arctan(1/x) = π/2" => Cow::Borrowed("Aplicar identidad de arctangentes"),
        "Canonicalization" => Cow::Borrowed("Reordenar la expresión"),
        description if description.starts_with("-(") && description.contains(") = -") => {
            Cow::Borrowed("Quitar paréntesis tras el signo menos")
        }
        "Polynomial division with opaque substitution" => {
            Cow::Borrowed("Reconocer un cociente notable")
        }
        "Polynomial identity (opaque substitution): cancel to 0" => {
            Cow::Borrowed("Cancelar una identidad exacta")
        }
        "Auto-expand (a+b)^2" => Cow::Borrowed("Expandir el binomio"),
        "Expand and combine polynomial product" => {
            Cow::Borrowed("Expandir y reagrupar un producto polinómico")
        }
        "Cancel opposite terms" => Cow::Borrowed("Cancelar términos opuestos"),
        "sin²(x) + cos²(x) = 1" => Cow::Borrowed("Aplicar la identidad pitagórica"),
        "sqrt(A^2 ± 2AB + B^2) = |A ± B|" => {
            Cow::Borrowed("Reconocer un cuadrado perfecto bajo la raíz")
        }
        _ => Cow::Borrowed(description),
    }
}
