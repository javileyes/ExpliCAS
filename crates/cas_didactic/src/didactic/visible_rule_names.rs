use std::borrow::Cow;

pub(crate) fn visible_rule_name(rule_name: &str) -> &str {
    match rule_name {
        "Collect Terms" => "Agrupar términos por variable",
        "Factor Out With Division" => "Sacar factor usando división",
        "Factorization" => "Factorizar",
        "Binomial Expansion" => "Expandir binomio",
        "Small Multinomial Expansion" => "Expandir binomio",
        "Expand" => "Expandir la expresión",
        "Distributive Property" => "Expandir la expresión",
        "expand_log" => "Expandir logaritmos",
        "Expand Log Abs Mul/Div" => "Expandir logaritmos",
        "Factor Perfect Square in Logarithm" => "Sacar un exponente fuera del logaritmo",
        "Log Contraction" => "Contraer logaritmos",
        "Finite Product" => "Evaluar producto finito",
        "Finite Summation" => "Evaluar suma finita",
        "Cos Product Telescoping" => "Aplicar telescopado de cosenos",
        "Dirichlet Kernel Identity" => "Aplicar identidad del núcleo de Dirichlet",
        "Complete the Square" => "Completar el cuadrado",
        "Product-to-Sum Identity" => "Aplicar producto a suma",
        "Product-to-Sum and Triple-Angle Identity" => "Aplicar producto a suma y ángulo triple",
        "Hyperbolic Product-to-Sum and Triple-Angle Identity" => {
            "Aplicar producto a suma y ángulo triple hiperbólico"
        }
        "Sum-to-Product Identity" => "Aplicar suma a producto",
        "Sum-to-Product Identity Cancellation Bridge" => "Aplicar suma a producto",
        "Phase Shift Identity" => "Aplicar identidad de desfase",
        "Double Angle Expansion" => "Expandir ángulo doble",
        "Double Angle Contraction" => "Contraer ángulo doble",
        "Angle Consistency (Half-Angle)" => "Expandir coseno de ángulo doble",
        "Cos 2x Additive Contraction" => "Contraer ángulo doble",
        "Trig Square Identity" => "Aplicar identidad del cuadrado trigonométrico",
        "Power Reduction Identity" => "Aplicar reducción de potencias",
        "Expand Secant Squared" => "Expandir secante cuadrada",
        "Expand Cosecant Squared" => "Expandir cosecante cuadrada",
        "Recognize Secant Squared" => "Reconocer secante cuadrada",
        "Recognize Cosecant Squared" => "Reconocer cosecante cuadrada",
        "Half-Angle Square Identity" => "Aplicar identidad de ángulo mitad",
        "Reciprocal Trig Identity" => "Aplicar identidad trigonométrica recíproca",
        "Reciprocal Product Identity" => "Cancelar funciones trigonométricas recíprocas",
        "Reciprocal Pythagorean Identity" => "Aplicar identidad pitagórica recíproca",
        "Triple Angle Identity" => "Reescribir ángulo triple",
        "Half-Angle Tangent Identity" => "Aplicar identidad de tangente de ángulo mitad",
        "Trig Expansion" => "Expandir una identidad trigonométrica",
        "Trig Quotient" => "Convertir un cociente trigonométrico en tangente",
        "Cos-Diff / Sin-Diff Quotient" => "Convertir un cociente trigonométrico en tangente",
        "Pythagorean Factor Form" => "Aplicar identidad pitagórica",
        "Pythagorean High-Power Factor" => "Aplicar identidad pitagórica y reagrupar",
        "Hyperbolic Pythagorean Identity Cancellation Bridge" => {
            "Aplicar la identidad pitagórica hiperbólica"
        }
        "Hyperbolic Pythagorean Identity Cancellation Bridge Residual" => {
            "Aplicar la identidad pitagórica hiperbólica"
        }
        "Consecutive Factorial Ratio" => "Cancelar factoriales consecutivos",
        "Rationalize" | "Rationalize Linear Sqrt Denominator" | "Rationalize Denominator" => {
            "Racionalizar el denominador"
        }
        "Distribute Division" => "Repartir el denominador común",
        "Mixed Fraction Split" => "Separar parte entera y resto",
        "Mixed Fraction Combine" => "Unir parte entera y fracción",
        "Telescoping Fraction Combine" => "Recomponer fracción telescópica",
        "Telescoping Fraction Split" => "Descomponer en fracciones telescópicas",
        "Canonicalize Roots" => "Reescribir la raíz como potencia fraccionaria",
        "Combine powers with same base (n-ary)" => "Sumar exponentes de la misma base",
        "Expand Odd Half Power" => "Reescribir potencia semientera impar",
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
        "Polynomial division with opaque substitution" => "Reconocer un cociente notable",
        "Pre-order Common Factor Cancel" => "Cancelar un factor común",
        "Pre-order Difference of Squares Cancel" => {
            "Factorizar una diferencia de cuadrados y cancelar"
        }
        "Pre-order Sum/Difference of Cubes" => "Factorizar suma o diferencia de cubos",
        "Pre-order Sum/Difference of Cubes Cancel" => "Cancelar factor tras factorizar cubos",
        "Cancel Sum/Difference of Cubes Fraction" => "Factorizar cubos y cancelar",
        "Inverse Tan Relations" => "Aplicar identidad de arctangentes",
        "Inverse Trig Composition" => "Aplicar composición trigonométrica inversa",
        "Sqrt Perfect Square" | "Simplify Square Root" | "Simplify perfect square root" => {
            "Reconocer un cuadrado perfecto bajo la raíz"
        }
        "Combine Like Terms" => "Agrupar términos semejantes",
        "Combine Same Denominator Fractions" => "Sumar fracciones con mismo denominador",
        "Combine Same Denominator Sub" => "Restar fracciones con mismo denominador",
        "Common Denominator" => "Llevar a denominador común",
        "Add Fractions" => "Sumar fracciones",
        "Subtract Fractions" => "Restar fracciones",
        "Simplify Nested Fraction" => "Cancelar factores en una fracción",
        "Simplify Complex Fraction" => "Simplificar fracción anidada",
        "Auto Expand Power Sum" => "Expandir binomio",
        "Polynomial Product Normalize" => "Expandir y reagrupar un producto polinómico",
        "Difference of Squares" => "Expandir la expresión",
        "Difference of Squares (Product to Difference)" => "Expandir la expresión",
        "Sophie Germain Identity" => "Expandir la expresión",
        "Sum/Difference of Cubes Contraction" => "Expandir la expresión",
        "Pythagorean Chain Identity" => "Aplicar la identidad pitagórica",
        _ => rule_name,
    }
}

pub(crate) fn visible_rule_name_for_step<'a>(
    rule_name: &'a str,
    description: &str,
) -> Cow<'a, str> {
    if rule_name == "Collect Terms" && description.starts_with("Collect terms by ") {
        let focus = &description["Collect terms by ".len()..];
        if is_simple_collect_focus(focus) {
            return Cow::Borrowed("Agrupar términos por variable");
        }
        return Cow::Borrowed("Agrupar términos por factor común");
    }

    if rule_name == "Finite Product"
        && (description.starts_with("Telescoping product:")
            || description.starts_with("Factorized telescoping product:"))
    {
        return Cow::Borrowed("Evaluar producto telescópico finito");
    }
    if rule_name == "Finite Summation" && description.starts_with("Telescoping sum:") {
        return Cow::Borrowed("Evaluar suma telescópica finita");
    }

    match rule_name {
        "Expand Log Product Power"
            if description == "Log expansion followed by exact cancellation" =>
        {
            Cow::Borrowed("Expandir logaritmos y cancelar términos iguales")
        }
        "Angle Consistency (Half-Angle)" if description == "Half-Angle Expansion" => {
            Cow::Borrowed("Expandir coseno de ángulo doble")
        }
        "Expand Log Abs Mul/Div"
            if description == "Log expansion followed by exact cancellation" =>
        {
            Cow::Borrowed("Expandir logaritmos y cancelar términos iguales")
        }
        "Collapse Exact Zero Additive Subexpression"
            if description == "Expand sine sum to product"
                || description == "Expand sine difference to product"
                || description == "Expand cosine sum to product"
                || description == "Expand cosine difference to product" =>
        {
            Cow::Borrowed("Aplicar suma a producto")
        }
        "Collapse Exact Zero Additive Subexpression"
            if description == "Product-to-Sum Identity" =>
        {
            Cow::Borrowed("Aplicar producto a suma")
        }
        "Collapse Exact Zero Additive Subexpression"
            if description == "Angle Sum/Diff Identity" =>
        {
            Cow::Borrowed("Angle Sum/Diff Identity")
        }
        "Collapse Exact Zero Additive Subexpression"
            if description == "Half-Angle Square Identity" =>
        {
            Cow::Borrowed("Aplicar identidad de ángulo mitad")
        }
        "Collapse Exact Zero Additive Subexpression"
            if description == "Power Reduction Identity" =>
        {
            Cow::Borrowed("Aplicar reducción de potencias")
        }
        "Collapse Exact Zero Additive Subexpression"
            if description == "Recognize Secant Squared" =>
        {
            Cow::Borrowed("Reconocer secante cuadrada")
        }
        "Collapse Exact Zero Additive Subexpression"
            if description == "Recognize Cosecant Squared" =>
        {
            Cow::Borrowed("Reconocer cosecante cuadrada")
        }
        "Collapse Exact Zero Additive Subexpression" if description == "Phase Shift Identity" => {
            Cow::Borrowed("Aplicar identidad de desfase")
        }
        "Collapse Exact Zero Additive Subexpression"
            if description == "Expand hyperbolic angle sum/difference" =>
        {
            Cow::Borrowed("Hyperbolic Angle Sum/Difference Identity")
        }
        "Collapse Exact Zero Additive Subexpression" if description == "Complete the Square" => {
            Cow::Borrowed("Completar el cuadrado")
        }
        "Collapse Exact Zero Additive Subexpression"
            if description == "Log expansion followed by exact cancellation" =>
        {
            Cow::Borrowed("Expandir logaritmos y cancelar términos iguales")
        }
        "Evaluate Logarithms" if description == "log(b, x^y) = y * log(b, x)" => {
            Cow::Borrowed("Sacar un exponente fuera del logaritmo")
        }
        "Rationalize Linear Sqrt Denominator" if description.contains("opaque substitution") => {
            Cow::Borrowed("Reconocer un cociente notable")
        }
        _ => Cow::Borrowed(visible_rule_name(rule_name)),
    }
}

fn is_simple_collect_focus(focus: &str) -> bool {
    !focus.is_empty()
        && focus
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

pub(crate) fn visible_step_description<'a>(description: &'a str) -> Cow<'a, str> {
    if description.starts_with("Telescoping product:") {
        return Cow::Borrowed("Evaluar producto telescópico");
    }
    if description.starts_with("Factorized telescoping product:") {
        return Cow::Borrowed("Evaluar producto telescópico factorizado");
    }
    if description.starts_with("Telescoping sum:") {
        return Cow::Borrowed("Evaluar suma telescópica");
    }

    match description {
        "Rationalize: multiply by conjugate" => Cow::Borrowed("Multiplicar por el conjugado"),
        "Factorization" => Cow::Borrowed("Factorizar"),
        "Log expansion" => Cow::Borrowed("Expandir logaritmos"),
        "Apply Morrie's law to telescope the cosine product" | "Apply Morrie's law" => {
            Cow::Borrowed("Aplicar la ley de Morrie")
        }
        "Apply the Dirichlet kernel identity to rewrite the cosine sum"
        | "Apply the Dirichlet kernel identity" => Cow::Borrowed("Aplicar el núcleo de Dirichlet"),
        "Complete the square to rewrite the quadratic" => {
            Cow::Borrowed("Completar el cuadrado para reescribir la cuadrática")
        }
        "log(b, x^y) = y * log(b, x)" => Cow::Borrowed("Sacar un exponente fuera del logaritmo"),
        "Distribute a sum over the common denominator" => {
            Cow::Borrowed("Repartir el denominador entre los sumandos")
        }
        "Split a fraction into a whole part plus remainder" => {
            Cow::Borrowed("Separar la fracción en parte entera y resto")
        }
        "Combine the whole part with the remaining fraction" => {
            Cow::Borrowed("Unir la parte entera con la fracción restante")
        }
        "Recompose the telescoping partial fractions into a single fraction" => {
            Cow::Borrowed("Recomponer las fracciones telescópicas en una sola fracción")
        }
        "Split into telescoping partial fractions" => {
            Cow::Borrowed("Descomponer en fracciones telescópicas")
        }
        "Rewrite an odd half-integer power using a square root" => {
            Cow::Borrowed("Reescribir la potencia semientera con una raíz")
        }
        "sqrt(x) = x^(1/2)" => Cow::Borrowed("Reescribir la raíz como potencia fraccionaria"),
        "Combine powers with same base (n-ary)" => {
            Cow::Borrowed("Sumar exponentes de la misma base")
        }
        "ln(x^(2k)) = 2·ln(|x^k|)" => Cow::Borrowed("Sacar un exponente fuera del logaritmo"),
        "Expand double-angle sine" => Cow::Borrowed("Expandir seno de ángulo doble"),
        "Expand double-angle cosine" => Cow::Borrowed("Expandir coseno de ángulo doble"),
        "Expand sin²(u) as (1 - cos(2u))/2" => {
            Cow::Borrowed("Expandir seno cuadrado con ángulo mitad")
        }
        "Half-Angle Expansion" => Cow::Borrowed("Expandir coseno de ángulo doble"),
        "Expand cos²(u) as (1 + cos(2u))/2" => {
            Cow::Borrowed("Expandir coseno cuadrado con ángulo mitad")
        }
        "Recognize (1 - cos(2u))/2 as sin²(u)" => {
            Cow::Borrowed("Reconocer seno cuadrado desde ángulo mitad")
        }
        "Recognize (1 + cos(2u))/2 as cos²(u)" => {
            Cow::Borrowed("Reconocer coseno cuadrado desde ángulo mitad")
        }
        "Expand cosine double-angle as 1 - 2·sin(u)^2" => {
            Cow::Borrowed("Expandir coseno de ángulo doble como 1 - 2·sin²")
        }
        "Expand cosine double-angle as 2·cos(u)^2 - 1" => {
            Cow::Borrowed("Expandir coseno de ángulo doble como 2·cos² - 1")
        }
        "Expand tangent to sine over cosine" => {
            Cow::Borrowed("Expandir tangente como seno entre coseno")
        }
        "Expand sec(u) as 1 / cos(u)" => {
            Cow::Borrowed("Reescribir secante como recíproco del coseno")
        }
        "Expand csc(u) as 1 / sin(u)" => {
            Cow::Borrowed("Reescribir cosecante como recíproco del seno")
        }
        "Expand cot(u) as cos(u) / sin(u)" => {
            Cow::Borrowed("Reescribir cotangente como coseno entre seno")
        }
        "Recognize 1 / cos(u) as sec(u)" => Cow::Borrowed("Reconocer secante desde un recíproco"),
        "Recognize 1 / sin(u) as csc(u)" => Cow::Borrowed("Reconocer cosecante desde un recíproco"),
        "Recognize cos(u) / sin(u) as cot(u)" => {
            Cow::Borrowed("Reconocer cotangente desde un cociente")
        }
        "Recognize tan(u) · cot(u) = 1" => {
            Cow::Borrowed("Cancelar funciones trigonométricas recíprocas")
        }
        "Recognize sec²(u) - tan²(u) = 1" => {
            Cow::Borrowed("Aplicar identidad pitagórica recíproca")
        }
        "Recognize csc²(u) - cot²(u) = 1" => {
            Cow::Borrowed("Aplicar identidad pitagórica recíproca")
        }
        "Expand 2·sin(A)·cos(B) into sin(A+B) + sin(A-B)" => {
            Cow::Borrowed("Convertir producto seno-coseno en suma de senos")
        }
        "Expand 2·cos(A)·sin(B) into sin(A+B) - sin(A-B)" => {
            Cow::Borrowed("Convertir producto coseno-seno en diferencia de senos")
        }
        "Expand 2·cos(A)·cos(B) into cos(A+B) + cos(A-B)" => {
            Cow::Borrowed("Convertir producto de cosenos en suma")
        }
        "Expand 2·sin(A)·sin(B) into cos(A-B) - cos(A+B)" => {
            Cow::Borrowed("Convertir producto de senos en diferencia de cosenos")
        }
        "Expand sine sum to product" => Cow::Borrowed("Convertir suma de senos en producto"),
        "Expand sine difference to product" => {
            Cow::Borrowed("Convertir diferencia de senos en producto")
        }
        "Expand cosine sum to product" => Cow::Borrowed("Convertir suma de cosenos en producto"),
        "Expand cosine difference to product" => {
            Cow::Borrowed("Convertir diferencia de cosenos en producto")
        }
        "Contract half-angle tangent quotient" => {
            Cow::Borrowed("Aplicar identidad de tangente de ángulo mitad")
        }
        "Half-Angle Square Identity" => Cow::Borrowed("Aplicar identidad de ángulo mitad"),
        "1 * x = x" => Cow::Borrowed("Quitar el factor 1"),
        "Evaluate literal power" => Cow::Borrowed("Calcular potencia numérica"),
        "Simplify perfect square root" => {
            Cow::Borrowed("Reconocer un cuadrado perfecto bajo la raíz")
        }
        "a - a = 0" => Cow::Borrowed("Restar dos expresiones iguales"),
        "Add fractions: a/b + c/d -> (ad+bc)/bd" => Cow::Borrowed("Sumar fracciones"),
        "Subtract fractions: a/b - c/d -> (ad-bc)/bd" => Cow::Borrowed("Restar fracciones"),
        "Common denominator: k + p/q → (k·q + p)/q" => {
            Cow::Borrowed("Llevar a denominador común")
        }
        "Common denominator: a - b/a → (a² - b)/a" => {
            Cow::Borrowed("Llevar a denominador común en una resta")
        }
        "Simplify nested fraction" => Cow::Borrowed("Simplificar fracción anidada"),
        "Simplify Nested Fraction" => Cow::Borrowed("Cancelar factores en una fracción"),
        "Simplify Complex Fraction" => Cow::Borrowed("Simplificar fracción anidada"),
        "Pre-order Perfect Square Minus Cancel" => {
            Cow::Borrowed("Cancelar un cuadrado perfecto con el mismo binomio")
        }
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
        description if description.starts_with("Collect terms by ") => Cow::Owned(format!(
            "Agrupar términos por {}",
            &description["Collect terms by ".len()..]
        )),
        _ => Cow::Borrowed(description),
    }
}
