use cas_solver_core::rule_names::{
    RULE_CANCEL_EXACT_ADDITIVE_PAIRS, RULE_EVALUATE_NUMERIC_POWER, RULE_EXPAND_LOG_ABS_MUL_DIV,
};
use std::borrow::Cow;

pub(crate) fn visible_rule_name(rule_name: &str) -> &str {
    match rule_name {
        "Collect Terms" => "Agrupar términos por variable",
        "Factor Out With Division" => "Sacar factor usando división",
        "Factorization" => "Factorizar",
        "Factor Polynomial" => "Factorizar el polinomio",
        "Evaluate Meta Functions" => "Evaluar la operación solicitada",
        "Binomial Expansion" => "Expandir binomio",
        "Small Multinomial Expansion" => "Expandir binomio",
        "Expand" => "Expandir la expresión",
        "Log Even Power" => "Sacar la potencia par fuera del logaritmo",
        "Power of a Quotient" => "Potencia de un cociente",
        "Trig Fourth Power Difference" => "Diferencia de cuartas potencias trigonométricas",
        "Pythagorean with Generic Coefficient" => "Aplicar la identidad pitagórica con coeficiente",
        "Quartic Pythagorean Identity" => "Aplicar la identidad pitagórica (cuarta potencia)",
        "Collapse Shifted Quotient of Equivalent Expressions" => {
            "Cancelar el cociente de expresiones equivalentes"
        }
        "Collapse Exact Zero Additive Subexpression" => {
            "Cancelar la subexpresión idénticamente nula"
        }
        "Distributive Property" => "Expandir la expresión",
        "expand_log" => "Expandir logaritmos",
        RULE_EXPAND_LOG_ABS_MUL_DIV => "Expandir logaritmos",
        "Factor Perfect Square in Logarithm" => "Sacar un exponente fuera del logaritmo",
        "Log Contraction" => "Contraer logaritmos",
        "Change of Base" => "Aplicar cambio de base",
        "Log-Exp Inverse" => "Cancelar logaritmo natural y exponencial inversos",
        "Exponential-Log Inverse" => "Cancelar exponencial y logaritmo inversos",
        "Exponential-Log Power Inverse" => {
            "Cancelar exponencial con logaritmo y conservar exponente"
        }
        "Log Inverse Power" => "Convertir potencia logarítmica inversa",
        "Exponential Sum/Difference Identity" => "Reescribir exponenciales",
        "Exponential Reciprocal Identity" => "Reescribir recíproco exponencial",
        "Exponential Power Identity" => "Reescribir potencia exponencial",
        "Power of a Power" => "Multiplicar exponentes",
        "Finite Product" => "Evaluar producto finito",
        "Finite Summation" => "Evaluar suma finita",
        "Number Theory Operations" => "Evaluar operación de teoría de números",
        "Pascal's Identity" => "Aplicar identidad de Pascal",
        "Binomial Coefficient Symmetry" => "Aplicar simetría del coeficiente binomial",
        "Cos Product Telescoping" => "Aplicar telescopado de cosenos",
        "Dirichlet Kernel Identity" => "Aplicar identidad del núcleo de Dirichlet",
        "Complete the Square" => "Completar el cuadrado",
        "Product-to-Sum Identity" => "Aplicar producto a suma",
        "Product-to-Sum and Triple-Angle Identity" => "Aplicar producto a suma y ángulo triple",
        "Hyperbolic Angle Sum/Difference Identity" => {
            "Aplicar identidad hiperbólica de suma/diferencia de ángulos"
        }
        "Hyperbolic Product-to-Sum Identity" => "Aplicar identidad hiperbólica de producto a suma",
        "Hyperbolic Product-to-Sum and Triple-Angle Identity" => {
            "Aplicar producto a suma y ángulo triple hiperbólico"
        }
        "Hyperbolic Double-Angle Identity" => "Aplicar identidad hiperbólica de ángulo doble",
        "Hyperbolic Triple-Angle Identity" => "Aplicar identidad hiperbólica de ángulo triple",
        "Hyperbolic Half-Angle Squares" => "Aplicar identidad hiperbólica de ángulo mitad",
        "Hyperbolic Exponential Identity" => "Aplicar identidad exponencial hiperbólica",
        "Hyperbolic Pythagorean Identity" => "Aplicar identidad pitagórica hiperbólica",
        "Hyperbolic Quotient Identity" => "Aplicar identidad hiperbólica de cociente",
        "Evaluate Hyperbolic Functions" => "Evaluar valor hiperbólico especial",
        "Evaluate Trigonometric Functions" => "Evaluar valor trigonométrico especial",
        "Evaluate Trigonometric Functions (Table)" => "Evaluar valor trigonométrico especial",
        "Evaluate Trig at Integer Multiple of π" => "Evaluar en múltiplo entero de π",
        "Hyperbolic Composition" => "Cancelar funciones hiperbólicas inversas",
        "Inverse Hyperbolic Log Identity" => "Convertir tangente hiperbólica inversa en logaritmo",
        "Hyperbolic Parity (Odd/Even)" => "Aplicar paridad hiperbólica",
        "Sum-to-Product Identity" => "Aplicar suma a producto",
        "Sum-to-Product Identity Cancellation Bridge" => "Aplicar suma a producto",
        "Cofunction Identity" => "Aplicar identidad de cofunción",
        "Angle Sum/Diff Identity" => "Aplicar suma/diferencia de ángulos",
        "Phase Shift Identity" => "Aplicar identidad de desfase",
        "Double Angle Expansion" => "Expandir ángulo doble",
        "Double Angle Contraction" => "Contraer ángulo doble",
        "Square Double Angle Contraction" => "Contraer cuadrado de ángulo doble",
        "Tangent Double-Angle Identity" => "Aplicar identidad de tangente de ángulo doble",
        "Tangent Angle Sum/Diff Identity" => {
            "Aplicar identidad de tangente de suma/diferencia de ángulos"
        }
        "Angle Consistency (Half-Angle)" => "Expandir identidad de ángulo doble",
        "Cos 2x Additive Contraction" => "Contraer ángulo doble",
        "Trig Square Identity" => "Aplicar identidad del cuadrado trigonométrico",
        "Power Reduction Identity" => "Aplicar reducción de potencias",
        "Quadruple Angle Expansion" => "Reescribir ángulo cuádruple",
        "Expand Secant Squared" => "Expandir secante cuadrada",
        "Expand Cosecant Squared" => "Expandir cosecante cuadrada",
        "Recognize Secant Squared" => "Reconocer secante cuadrada",
        "Recognize Cosecant Squared" => "Reconocer cosecante cuadrada",
        "Half-Angle Square Identity" => "Aplicar identidad de ángulo mitad",
        "Reciprocal Trig Identity" => "Aplicar identidad trigonométrica recíproca",
        "Reciprocal Product Identity" => "Cancelar funciones trigonométricas recíprocas",
        "Reciprocal Pythagorean Identity" => "Aplicar identidad pitagórica recíproca",
        "Quintuple Angle Identity" => "Reescribir ángulo quíntuple",
        "Triple Angle Expansion" | "Triple Angle Identity" => "Reescribir ángulo triple",
        "Half-Angle Tangent Identity" => "Aplicar identidad de tangente de ángulo mitad",
        "Trig Parity (Odd/Even)" => "Aplicar paridad trigonométrica",
        "Trig Expansion" => "Expandir una identidad trigonométrica",
        "Tan to Sin/Cos" => "Expandir tangente como seno entre coseno",
        "Secant to Reciprocal Cosine" => "Expandir secante como recíproco de coseno",
        "Cosecant to Reciprocal Sine" => "Expandir cosecante como recíproco de seno",
        "Cotangent to Cosine over Sine" => "Expandir cotangente como coseno entre seno",
        "Trig Quotient" => "Convertir un cociente trigonométrico en tangente",
        "Cos-Diff / Sin-Diff Quotient" => "Convertir un cociente trigonométrico en tangente",
        "Pythagorean Identity" => "Aplicar la identidad pitagórica",
        "Pythagorean Factor Form" => "Aplicar identidad pitagórica",
        "Pythagorean High-Power Factor" => "Aplicar identidad pitagórica y reagrupar",
        "Hyperbolic Pythagorean Identity Cancellation Bridge" => {
            "Aplicar la identidad pitagórica hiperbólica"
        }
        "Hyperbolic Pythagorean Identity Cancellation Bridge Residual" => {
            "Aplicar la identidad pitagórica hiperbólica"
        }
        "Consecutive Factorial Ratio" => "Cancelar factoriales consecutivos",
        "Rationalize"
        | "Rationalize Cube Root Denominator"
        | "Rationalize Linear Sqrt Denominator"
        | "Rationalize Denominator" => "Racionalizar el denominador",
        "Distribute Division" => "Repartir el denominador común",
        "Distribute Division Into Sum" => "Repartir el denominador entre los sumandos",
        "Mixed Fraction Split" => "Separar parte entera y resto",
        "Mixed Fraction Combine" => "Unir parte entera y fracción",
        "Telescoping Fraction Combine" => "Recomponer fracción telescópica",
        "Telescoping Fraction Split" => "Descomponer en fracciones telescópicas",
        "Canonicalize Roots" => "Reescribir la raíz como potencia fraccionaria",
        "Combine powers with same base (n-ary)" => "Sumar exponentes de la misma base",
        "N-ary Mul Combine Powers" => "Sumar exponentes de la misma base",
        "Cancel Same-Base Powers" => "Cancelar potencias de la misma base",
        "Identity Power" => "Simplificar una potencia con exponente 0 o 1",
        "Normalize Negative Exponent" => "Reescribir un exponente negativo",
        "Combine Constants" => "Combinar las constantes",
        "Expand Odd Half Power" => "Reescribir potencia semientera impar",
        "Merge Sqrt Product" => "Combinar raíces en un producto",
        "Merge Sqrt Quotient" => "Combinar raíces en un cociente",
        "Subtraction Self-Cancel" => "Restar dos expresiones iguales",
        "Identity Property of Addition" => "Quitar el 0",
        "Identity Property of Multiplication" => "Quitar el factor 1",
        "Negative Base Power" => "Simplificar potencia con base negativa",
        "Canonicalize Even Power Base" => "Invertir una resta dentro de una potencia par",
        RULE_EVALUATE_NUMERIC_POWER => "Calcular potencia numérica",
        "Evaluate Logarithms" => "Calcular el logaritmo",
        "Cancel Reciprocal Exponents" => "Deshacer raíz y potencia",
        "Square of Square Root" => "Deshacer raíz y potencia",
        "Canonicalize Nested Power" => "Reescribir potencia de una raíz",
        "Canonicalize Trig Function Names" => "Usar el nombre arctan",
        "Canonicalize Negation" => "Quitar paréntesis tras el signo menos",
        "Canonicalize" => "Reordenar la expresión",
        "Polynomial Identity" => "Cancelar una identidad exacta",
        "Polynomial division with opaque substitution" => "Reconocer un cociente notable",
        "Pre-order Common Factor Cancel" => "Cancelar un factor común",
        "Pre-order Difference of Squares" => "Factorizar una diferencia de cuadrados",
        "Pre-order Difference of Squares Cancel" => {
            "Factorizar una diferencia de cuadrados y cancelar"
        }
        "Pre-order Perfect Square Minus Cancel" => {
            "Cancelar un cuadrado perfecto con el mismo binomio"
        }
        "Pre-order Sum/Difference of Cubes" => "Factorizar suma o diferencia de cubos",
        "Pre-order Sum/Difference of Cubes Cancel" => "Cancelar factor tras factorizar cubos",
        "Cancel Sum/Difference of Cubes Fraction" => "Factorizar cubos y cancelar",
        "Inverse Tan Relations" => "Aplicar identidad de arctangentes",
        "Inverse Trig Sum Identity" => "Aplicar identidad complementaria arcsin/arccos",
        "Inverse Trig Composition" => "Aplicar composición trigonométrica inversa",
        "Sqrt Perfect Square" | "Simplify Square Root" | "Simplify perfect square root" => {
            "Reconocer un cuadrado perfecto bajo la raíz"
        }
        "Abs Of Sum Of Squares" => "Quitar valor absoluto de una expresión no negativa",
        "Abs Of Even Power" => "Quitar el valor absoluto de una potencia par",
        "Combine Like Terms" => "Agrupar términos semejantes",
        "Combine Same Denominator Fractions" => "Sumar fracciones con mismo denominador",
        "Combine Same Denominator Sub" => "Restar fracciones con mismo denominador",
        "Cancel Equal Fractions Difference" => "Cancelar fracciones iguales",
        RULE_CANCEL_EXACT_ADDITIVE_PAIRS => "Cancelar términos opuestos",
        "Common Denominator" => "Llevar a denominador común",
        "Add Fractions" => "Sumar fracciones",
        "Subtract Fractions" => "Restar fracciones",
        "Symbolic Differentiation" => "Calcular la derivada",
        "Symbolic Integration" => "Calcular la integral",
        "Vector Gradient" => "Calcular el gradiente",
        "Present calculus result in compact form" => {
            "Presentar resultado de cálculo en forma compacta"
        }
        "Pull Constant From Fraction" => "Sacar constante de una fracción",
        "Simplify Multiplication with Division" => "Combinar fracciones en una multiplicación",
        "Simplify Nested Fraction" => "Cancelar factores en una fracción",
        "Simplify Complex Fraction" => "Simplificar fracción anidada",
        "Auto Expand Power Sum" => "Expandir binomio",
        "Polynomial Product Normalize" => "Expandir y reagrupar un producto polinómico",
        "Difference of Squares" => "Expandir la expresión",
        "Difference of Squares (Product to Difference)" => "Expandir la expresión",
        "Sophie Germain Identity" => "Expandir la expresión",
        "Sum/Difference of Cubes Contraction" => "Expandir la expresión",
        "Pythagorean Chain Identity" => "Aplicar la identidad pitagórica",
        // Didactic round: Spanish names for rules that previously leaked their raw English
        // identifier into the visible `rule` field of the step-by-step (the `_` fallthrough below).
        "AutoExpandLogRule" => "Expandir el logaritmo de un producto",
        "Cancel Common Factors" => "Cancelar factores comunes",
        "Cancel Identical Numerator/Denominator" => "Cancelar numerador y denominador iguales",
        "Cancel Opposite Fractions" => "Cancelar fracciones opuestas",
        "Canonicalize Division" => "Reescribir la división",
        "Canonicalize Multiplication" => "Reescribir el producto",
        "Canonicalize Reciprocal Sqrt" => "Reescribir el inverso de una raíz",
        "Division by Infinity" => "Dividir entre infinito",
        "Evaluate Absolute Value" => "Evaluar el valor absoluto",
        "Extract Common Multiplicative Factor" => "Sacar factor común",
        "Extract Perfect Square from Radicand" => "Sacar el cuadrado perfecto de la raíz",
        "Higher-Order Differentiation" => "Derivar de orden superior",
        "Infinity Absorption in Addition" => "El infinito domina la suma",
        "Matrix Functions" => "Operación con matrices",
        "Matrix Multiplication" => "Multiplicar matrices",
        "Matrix Reciprocal/Inverse" => "Potencia o inversa de la matriz",
        "Partial Fraction Decomposition" => "Descomponer en fracciones parciales",
        "Power of a Product" => "Distribuir la potencia sobre el producto",
        "Product of Powers" => "Sumar exponentes de la misma base",
        "Quotient of Powers" => "Restar exponentes de la misma base",
        "Recognize Hyperbolic from Exponential" => {
            "Reconocer una función hiperbólica en las exponenciales"
        }
        "Root Denesting" => "Desanidar el radical",
        "Taylor Series" => "Desarrollar en serie de Taylor",
        "Zero Property of Division" => "Cero dividido entre cualquier valor es cero",
        "sinh(x)/cosh(x) = tanh(x)" => "Reconocer la tangente hiperbólica",
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
    if rule_name == "Finite Product" {
        if description.starts_with("Product of first integers:") {
            return Cow::Borrowed("Aplicar producto factorial");
        }
        if description.starts_with("Product of powers:") {
            return Cow::Borrowed("Aplicar producto de potencias");
        }
        if description.starts_with("Product of constant factor:") {
            return Cow::Borrowed("Aplicar producto de constante");
        }
    }
    if rule_name == "Finite Summation" && description.starts_with("Telescoping sum:") {
        return Cow::Borrowed("Evaluar suma telescópica finita");
    }
    if rule_name == "Finite Summation" {
        if description.starts_with("Sum of first integers:") {
            return Cow::Borrowed("Aplicar fórmula de suma de enteros");
        }
        if description.starts_with("Sum of squares:") {
            return Cow::Borrowed("Aplicar fórmula de suma de cuadrados");
        }
        if description.starts_with("Sum of cubes:") {
            return Cow::Borrowed("Aplicar fórmula de suma de cubos");
        }
        if description.starts_with("Sum of constant term:") {
            return Cow::Borrowed("Aplicar suma de constante");
        }
        if description.starts_with("Geometric sum:") {
            return Cow::Borrowed("Aplicar fórmula de suma geométrica");
        }
    }
    if rule_name == "Number Theory Operations" && description.starts_with("choose(") {
        return Cow::Borrowed("Calcular coeficiente binomial");
    }

    match rule_name {
        "Change of Base" if description == "Expand the logarithm using a change-of-base chain" => {
            Cow::Borrowed("Expandir cambio de base")
        }
        "Expand Log Product Power"
            if description == "Log expansion followed by exact cancellation" =>
        {
            Cow::Borrowed("Expandir logaritmos y cancelar términos iguales")
        }
        "Angle Consistency (Half-Angle)" if description == "Half-Angle Expansion" => {
            Cow::Borrowed("Expandir identidad de ángulo doble")
        }
        RULE_EXPAND_LOG_ABS_MUL_DIV
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
            Cow::Borrowed("Aplicar suma/diferencia de ángulos")
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
            Cow::Borrowed("Aplicar identidad hiperbólica de suma/diferencia de ángulos")
        }
        "Hyperbolic Quotient Identity"
            if description == "Recognize sinh(u) / cosh(u) as tanh(u)" =>
        {
            Cow::Borrowed("Reconocer tangente hiperbólica desde un cociente")
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
        "Trig Expansion" if description == "Expand tangent to sine over cosine" => {
            Cow::Borrowed("Expandir tangente como seno entre coseno")
        }
        "Trig Quotient" | "Trig Quotient to Named Function"
            if description == "cos(x)/sin(x) → cot(x)" =>
        {
            Cow::Borrowed("Reconocer cotangente desde un cociente")
        }
        "Trig Quotient" | "Trig Quotient to Named Function"
            if description == "1/sin(x) → csc(x)" =>
        {
            Cow::Borrowed("Reconocer cosecante desde un recíproco")
        }
        "Trig Quotient" | "Trig Quotient to Named Function"
            if description == "1/cos(x) → sec(x)" =>
        {
            Cow::Borrowed("Reconocer secante desde un recíproco")
        }
        "Trig Quotient" | "Trig Quotient to Named Function"
            if description == "1/tan(x) → cot(x)" =>
        {
            Cow::Borrowed("Reconocer cotangente desde un recíproco")
        }
        "Reciprocal Trig Identity" if description == "Expand sec(u) as 1 / cos(u)" => {
            Cow::Borrowed("Reescribir secante como recíproco del coseno")
        }
        "Reciprocal Trig Identity" if description == "Expand csc(u) as 1 / sin(u)" => {
            Cow::Borrowed("Reescribir cosecante como recíproco del seno")
        }
        "Reciprocal Trig Identity" if description == "Expand cot(u) as cos(u) / sin(u)" => {
            Cow::Borrowed("Reescribir cotangente como coseno entre seno")
        }
        "Reciprocal Trig Identity" if description == "Recognize 1 / cos(u) as sec(u)" => {
            Cow::Borrowed("Reconocer secante desde un recíproco")
        }
        "Reciprocal Trig Identity" if description == "Recognize 1 / sin(u) as csc(u)" => {
            Cow::Borrowed("Reconocer cosecante desde un recíproco")
        }
        "Reciprocal Trig Identity" if description == "Recognize cos(u) / sin(u) as cot(u)" => {
            Cow::Borrowed("Reconocer cotangente desde un cociente")
        }
        "Reciprocal Product Identity" if description == "Recognize tan(u) · cot(u) = 1" => {
            Cow::Borrowed("Reconocer tangente por cotangente como 1")
        }
        "Reciprocal Product Identity" if description == "Recognize sin(u) · csc(u) = 1" => {
            Cow::Borrowed("Reconocer seno por cosecante como 1")
        }
        "Reciprocal Product Identity" if description == "Recognize cos(u) · sec(u) = 1" => {
            Cow::Borrowed("Reconocer coseno por secante como 1")
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
        "Pre-order Difference of Squares" => {
            Cow::Borrowed("Factorizar una diferencia de cuadrados")
        }
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
        "Cancel equal fractions" => Cow::Borrowed("Cancelar fracciones iguales"),
        "ln(x^(2k)) = 2·ln(|x^k|)" => Cow::Borrowed("Sacar un exponente fuera del logaritmo"),
        "Expand double-angle sine" => Cow::Borrowed("Expandir seno de ángulo doble"),
        "Expand double-angle cosine" => Cow::Borrowed("Expandir coseno de ángulo doble"),
        "Expand sin²(u) as (1 - cos(2u))/2" => {
            Cow::Borrowed("Expandir seno cuadrado con ángulo mitad")
        }
        "Half-Angle Expansion" => Cow::Borrowed("Expandir identidad de ángulo doble"),
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
        "arcsin(x) + arccos(x) = π/2" => {
            Cow::Borrowed("Aplicar identidad complementaria arcsin/arccos")
        }
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

/// Translate a Spanish (default) visible rule name to English. The didactic layer builds the
/// step-by-step in Spanish (the source language); the `En` localization pivots through that output,
/// so this is the single Spanish -> English rule-name table. An unmapped name passes through
/// unchanged (e.g. an already-English internal name, or a dynamic name covered later).
pub(crate) fn rule_name_es_to_en(es: &str) -> &str {
    match es {
        "Agrupar términos semejantes" => "Group like terms",
        "Aplicar fórmula de suma de cuadrados" => "Apply the sum of squares formula",
        "Aplicar fórmula de suma de cubos" => "Apply the sum of cubes formula",
        "Aplicar fórmula de suma geométrica" => "Apply the geometric sum formula",
        "Aplicar la identidad pitagórica" => "Apply the Pythagorean identity",
        "Aplicar la identidad pitagórica con coeficiente" => {
            "Pythagorean with generic coefficient"
        }
        "Aplicar la identidad pitagórica (cuarta potencia)" => "Quartic Pythagorean identity",
        "Cancelar el cociente de expresiones equivalentes" => {
            "Collapse the quotient of equivalent expressions"
        }
        "Cancelar la subexpresión idénticamente nula" => {
            "Collapse the identically-zero subexpression"
        }
        "Diferencia de cuartas potencias trigonométricas" => "Trig fourth-power difference",
        "Potencia de un cociente" => "Power of a quotient",
        "Sacar la potencia par fuera del logaritmo" => "Take the even power out of the logarithm",
        "Aplicar producto factorial" => "Apply the factorial product",
        "Calcular coeficiente binomial" => "Compute the binomial coefficient",
        "Calcular el logaritmo" => "Compute the logarithm",
        "Calcular el gradiente" => "Compute the gradient",
        "Calcular la derivada" => "Compute the derivative",
        "Calcular la integral" => "Compute the integral",
        "Calcular potencia numérica" => "Compute the numeric power",
        "Cancelar factores comunes" => "Cancel common factors",
        "Cancelar factores en una fracción" => "Cancel factors in a fraction",
        "Cancelar factoriales consecutivos" => "Cancel consecutive factorials",
        "Cancelar fracciones iguales" => "Cancel equal fractions",
        "Cancelar fracciones opuestas" => "Cancel opposite fractions",
        "Cancelar numerador y denominador iguales" => "Cancel identical numerator and denominator",
        "Cancelar términos opuestos" => "Cancel opposite terms",
        "Cero dividido entre cualquier valor es cero" => "Zero divided by any value is zero",
        "Combinar las constantes" => "Combine the constants",
        "Derivar de orden superior" => "Differentiate to a higher order",
        "Desanidar el radical" => "Denest the radical",
        "Desarrollar en serie de Taylor" => "Expand as a Taylor series",
        "Descomponer en fracciones parciales" => "Decompose into partial fractions",
        "Deshacer raíz y potencia" => "Cancel the root with the power",
        "Distribuir la potencia sobre el producto" => "Distribute the power over the product",
        "Dividir entre infinito" => "Divide by infinity",
        "El infinito domina la suma" => "Infinity dominates the sum",
        "Evaluar el valor absoluto" => "Evaluate the absolute value",
        "Evaluar la operación solicitada" => "Evaluate the requested operation",
        "Evaluar límite en infinito" => "Evaluate the limit at infinity",
        "Evaluar límite finito" => "Evaluate the finite limit",
        "Evaluar operación de teoría de números" => "Evaluate the number-theory operation",
        "Evaluar producto telescópico finito" => "Evaluate the finite telescoping product",
        "Evaluar suma finita" => "Evaluate the finite sum",
        "Evaluar suma telescópica finita" => "Evaluate the finite telescoping sum",
        "Evaluar valor trigonométrico especial" => "Evaluate the special trigonometric value",
        "Expandir el logaritmo de un producto" => "Expand the logarithm of a product",
        "Expandir la expresión" => "Expand the expression",
        "Expandir tangente como seno entre coseno" => "Expand tangent as sine over cosine",
        "Expandir y reagrupar un producto polinómico" => "Expand and regroup a polynomial product",
        "Expandir ángulo doble" => "Expand the double angle",
        "Factorizar cubos y cancelar" => "Factor the cubes and cancel",
        "Factorizar el polinomio" => "Factor the polynomial",
        "Llevar a denominador común" => "Bring to a common denominator",
        "Multiplicar matrices" => "Multiply matrices",
        "Operación con matrices" => "Matrix operation",
        "Potencia o inversa de la matriz" => "Matrix power or inverse",
        "Calcular el determinante de la matriz" => "Compute the matrix determinant",
        "Calcular la traza de la matriz" => "Compute the matrix trace",
        "Transponer la matriz" => "Transpose the matrix",
        "Calcular la inversa de la matriz" => "Compute the matrix inverse",
        "Calcular el polinomio característico" => "Compute the characteristic polynomial",
        "Reducir la matriz a forma escalonada" => "Reduce the matrix to row echelon form",
        "Calcular los autovalores" => "Compute the eigenvalues",
        "Calcular los autovectores" => "Compute the eigenvectors",
        "Calcular el rango de la matriz" => "Compute the matrix rank",
        "Calcular el producto escalar" => "Compute the dot product",
        "Calcular el producto vectorial" => "Compute the cross product",
        "Resolver el sistema lineal" => "Solve the linear system",
        "Elevar la matriz a una potencia" => "Raise the matrix to a power",
        "Multiplicar por la inversa de la matriz" => "Multiply by the matrix inverse",
        "Presentar resultado de cálculo en forma compacta" => {
            "Present the calculus result in compact form"
        }
        "Quitar paréntesis tras el signo menos" => "Remove the parentheses after the minus sign",
        "Racionalizar el denominador" => "Rationalize the denominator",
        "Reconocer la tangente hiperbólica" => "Recognize the hyperbolic tangent",
        "Reconocer una función hiperbólica en las exponenciales" => {
            "Recognize a hyperbolic function in the exponentials"
        }
        "Repartir el denominador entre los sumandos" => "Split the denominator over the terms",
        "Restar exponentes de la misma base" => "Subtract exponents of the same base",
        "Sacar constante de una fracción" => "Factor a constant out of the fraction",
        "Sacar el cuadrado perfecto de la raíz" => "Take the perfect square out of the root",
        "Sacar factor común" => "Factor out the common term",
        "Sacar un exponente fuera del logaritmo" => "Bring an exponent out of the logarithm",
        "Simplificar fracción anidada" => "Simplify the nested fraction",
        "Simplificar una potencia con exponente 0 o 1" => "Simplify a power with exponent 0 or 1",
        "Sumar exponentes de la misma base" => "Add exponents of the same base",
        "Sumar fracciones" => "Add fractions",
        other => other,
    }
}
