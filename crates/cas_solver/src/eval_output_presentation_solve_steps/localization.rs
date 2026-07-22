//! Localization (es/en) of solver step/sub-step descriptions at the wire boundary.
//!
//! The solver builds `solve_steps[].description` (and sub-step descriptions) as fixed English
//! templates (and a few Spanish ones for the completing-the-square sub-steps), and `--lang` never
//! reached them, so a `solve()` trace was mixed EN/ES. This module translates each rendered
//! description back to the requested language by matching it against a table of source templates
//! and re-filling the captured variable parts into the target-language template. The capture/refill
//! is identity on the variable parts (both templates share the same `{N}` layout), so only the
//! surrounding wording changes. Unmatched descriptions pass through unchanged.

use cas_solver_core::eval_option_axes::Language;

struct SolveDesc {
    template: &'static str,
    es: &'static str,
    en: &'static str,
}

enum Part {
    Lit(String),
    Ph(usize),
}

fn parse_template(t: &str) -> Vec<Part> {
    let mut parts = Vec::new();
    let mut lit = String::new();
    let mut rest = t;
    while let Some(open) = rest.find('{') {
        if let Some(close_rel) = rest[open + 1..].find('}') {
            let inner = &rest[open + 1..open + 1 + close_rel];
            if !inner.is_empty() && inner.bytes().all(|b| b.is_ascii_digit()) {
                lit.push_str(&rest[..open]);
                if !lit.is_empty() {
                    parts.push(Part::Lit(std::mem::take(&mut lit)));
                }
                parts.push(Part::Ph(inner.parse().unwrap()));
                rest = &rest[open + 1 + close_rel + 1..];
                continue;
            }
        }
        // `{` that does not open a numeric placeholder: keep as literal.
        lit.push_str(&rest[..=open]);
        rest = &rest[open + 1..];
    }
    lit.push_str(rest);
    if !lit.is_empty() {
        parts.push(Part::Lit(lit));
    }
    parts
}

/// Match `input` against a parsed template, capturing `(index, value)` for each placeholder.
fn match_template(input: &str, parts: &[Part]) -> Option<Vec<(usize, String)>> {
    let mut caps = Vec::new();
    let mut pos = 0usize;
    for (i, part) in parts.iter().enumerate() {
        match part {
            Part::Lit(lit) => {
                if i == 0 {
                    if !input[pos..].starts_with(lit.as_str()) {
                        return None;
                    }
                    pos += lit.len();
                } else {
                    let rel = input[pos..].find(lit.as_str())?;
                    pos += rel + lit.len();
                }
            }
            Part::Ph(n) => match parts.get(i + 1) {
                Some(Part::Lit(next)) => {
                    let rel = input[pos..].find(next.as_str())?;
                    caps.push((*n, input[pos..pos + rel].to_string()));
                    pos += rel;
                }
                Some(Part::Ph(_)) => return None,
                None => {
                    caps.push((*n, input[pos..].to_string()));
                    pos = input.len();
                }
            },
        }
    }
    if matches!(parts.last(), Some(Part::Lit(_))) && pos != input.len() {
        return None;
    }
    Some(caps)
}

fn fill(template: &str, caps: &[(usize, String)]) -> String {
    let mut out = String::new();
    for part in parse_template(template) {
        match part {
            Part::Lit(s) => out.push_str(&s),
            Part::Ph(n) => match caps.iter().find(|(i, _)| *i == n) {
                Some((_, v)) => out.push_str(v),
                None => out.push_str(&format!("{{{n}}}")),
            },
        }
    }
    out
}

/// Translate a solver step/sub-step description into `language`. Returns the input unchanged when no
/// template matches (so an unknown description never gets garbled).
pub(crate) fn localize_solve_description(desc: &str, language: Language) -> String {
    for entry in SOLVE_DESCRIPTIONS {
        let parts = parse_template(entry.template);
        if let Some(caps) = match_template(desc, &parts) {
            let target = match language {
                Language::En => entry.en,
                Language::Es => entry.es,
            };
            return fill(target, &caps);
        }
    }
    desc.to_string()
}

#[rustfmt::skip]
static SOLVE_DESCRIPTIONS: &[SolveDesc] = &[
    // dsolve (Fase 4) — second-order characteristic narration (D13, O4).
    SolveDesc { template: "Plantear la ecuación característica: a·r² + b·r + c = 0 con a = {0}, b = {1}, c = {2}", es: "Plantear la ecuación característica: a·r² + b·r + c = 0 con a = {0}, b = {1}, c = {2}", en: "Set up the characteristic equation: a·r² + b·r + c = 0 with a = {0}, b = {1}, c = {2}" },
    SolveDesc { template: "Calcular el discriminante de la característica: Δ = {0}", es: "Calcular el discriminante de la característica: Δ = {0}", en: "Compute the characteristic discriminant: Δ = {0}" },
    SolveDesc { template: "Raíces reales distintas (Δ > 0): la base es {e^(r1·x), e^(r2·x)}", es: "Raíces reales distintas (Δ > 0): la base es {e^(r1·x), e^(r2·x)}", en: "Distinct real roots (Δ > 0): the basis is {e^(r1·x), e^(r2·x)}" },
    SolveDesc { template: "Raíz real doble (Δ = 0): la base es {e^(r·x), x·e^(r·x)}", es: "Raíz real doble (Δ = 0): la base es {e^(r·x), x·e^(r·x)}", en: "Repeated real root (Δ = 0): the basis is {e^(r·x), x·e^(r·x)}" },
    SolveDesc { template: "Raíces complejas conjugadas (Δ < 0): la base es {e^(α·x)·cos(β·x), e^(α·x)·sin(β·x)}", es: "Raíces complejas conjugadas (Δ < 0): la base es {e^(α·x)·cos(β·x), e^(α·x)·sin(β·x)}", en: "Complex conjugate roots (Δ < 0): the basis is {e^(α·x)·cos(β·x), e^(α·x)·sin(β·x)}" },
    SolveDesc { template: "Solución general: combinación lineal de la base con C1 y C2", es: "Solución general: combinación lineal de la base con C1 y C2", en: "General solution: linear combination of the basis with C1 and C2" },
    SolveDesc { template: "Verificar por sustitución: cada función de la base anula la EDO", es: "Verificar por sustitución: cada función de la base anula la EDO", en: "Verify by substitution: each basis function annihilates the ODE" },
    SolveDesc { template: "Aplicar las condiciones iniciales en {0} = {1}: resolver el sistema 2×2 en {2}, {3}", es: "Aplicar las condiciones iniciales en {0} = {1}: resolver el sistema 2×2 en {2}, {3}", en: "Apply the initial conditions at {0} = {1}: solve the 2×2 system in {2}, {3}" },
    // dsolve (Fase 4) — initial-condition narration (D13, O3).
    SolveDesc { template: "Aplicar la condición inicial {0}({1}) = {2}: sustituir el punto y fijar la constante", es: "Aplicar la condición inicial {0}({1}) = {2}: sustituir el punto y fijar la constante", en: "Apply the initial condition {0}({1}) = {2}: substitute the point and pin the constant" },
    SolveDesc { template: "Solución particular con la condición aplicada", es: "Solución particular con la condición aplicada", en: "Particular solution with the condition applied" },
    // dsolve (Fase 4) — exact method narration (D13, O2).
    SolveDesc { template: "Identificar forma exacta: M + N·y' = 0 con M = {0}, N = {1}", es: "Identificar forma exacta: M + N·y' = 0 con M = {0}, N = {1}", en: "Identify exact form: M + N·y' = 0 with M = {0}, N = {1}" },
    SolveDesc { template: "Comprobar exactitud: ∂M/∂y = ∂N/∂x (el campo (M, N) es conservativo)", es: "Comprobar exactitud: ∂M/∂y = ∂N/∂x (el campo (M, N) es conservativo)", en: "Check exactness: ∂M/∂y = ∂N/∂x (the field (M, N) is conservative)" },
    SolveDesc { template: "Reconstruir el potencial: φ = ∫M dx + h(y) con h'(y) ajustando ∂φ/∂y = N", es: "Reconstruir el potencial: φ = ∫M dx + h(y) con h'(y) ajustando ∂φ/∂y = N", en: "Reconstruct the potential: φ = ∫M dx + h(y) with h'(y) matching ∂φ/∂y = N" },
    SolveDesc { template: "Verificar el potencial: ∂φ/∂x = M y ∂φ/∂y = N (residuos exactos a 0)", es: "Verificar el potencial: ∂φ/∂x = M y ∂φ/∂y = N (residuos exactos a 0)", en: "Verify the potential: ∂φ/∂x = M and ∂φ/∂y = N (exact residues to 0)" },
    // dsolve (Fase 4) — linear first-order method narration (D13, O1).
    SolveDesc { template: "Identificar forma lineal: y' + p·y = q con p = {0}, q = {1}", es: "Identificar forma lineal: y' + p·y = q con p = {0}, q = {1}", en: "Identify linear form: y' + p·y = q with p = {0}, q = {1}" },
    SolveDesc { template: "Calcular el factor integrante: μ = e^(∫p dx) = {0}", es: "Calcular el factor integrante: μ = e^(∫p dx) = {0}", en: "Compute the integrating factor: μ = e^(∫p dx) = {0}" },
    SolveDesc { template: "Multiplicar por μ: el lado izquierdo se vuelve la derivada del producto μ·y", es: "Multiplicar por μ: el lado izquierdo se vuelve la derivada del producto μ·y", en: "Multiply by μ: the left-hand side becomes the derivative of the product μ·y" },
    SolveDesc { template: "Integrar ambos lados: μ·y = ∫ μ·q dx + C", es: "Integrar ambos lados: μ·y = ∫ μ·q dx + C", en: "Integrate both sides: μ·y = ∫ μ·q dx + C" },
    // dsolve (Fase 4) — separable method narration (D13).
    SolveDesc { template: "Identificar EDO separable: y' = f(x)·g(y) con f = {0}, g = {1}", es: "Identificar EDO separable: y' = f(x)·g(y) con f = {0}, g = {1}", en: "Identify separable ODE: y' = f(x)·g(y) with f = {0}, g = {1}" },
    SolveDesc { template: "Separar las variables: dy/g(y) = f(x)·dx", es: "Separar las variables: dy/g(y) = f(x)·dx", en: "Separate the variables: dy/g(y) = f(x)·dx" },
    SolveDesc { template: "Integrar ambos lados de la ecuación separada", es: "Integrar ambos lados de la ecuación separada", en: "Integrate both sides of the separated equation" },
    SolveDesc { template: "Despejar la incógnita de la relación integrada", es: "Despejar la incógnita de la relación integrada", en: "Solve the integrated relation for the unknown" },
    SolveDesc { template: "Combinar en una solución implícita φ(x,y) = C", es: "Combinar en una solución implícita φ(x,y) = C", en: "Combine into an implicit solution φ(x,y) = C" },
    SolveDesc { template: "Verificar por sustitución: el residuo de la EDO se reduce a 0", es: "Verificar por sustitución: el residuo de la EDO se reduce a 0", en: "Verify by substitution: the ODE residue reduces to 0" },
    SolveDesc { template: "Combine fractions on RHS (common denominator)", es: "Combina las fracciones del lado derecho (común denominador)", en: "Combine fractions on RHS (common denominator)" },
    SolveDesc { template: "Take reciprocal", es: "Toma el recíproco", en: "Take reciprocal" },
    SolveDesc { template: "Case 1: Assume {0} > 0. Multiply by {1} (positive). Inequality direction preserved (flipped from isolation logic).", es: "Caso 1: Supón {0} > 0. Multiplica por {1} (positivo). Se preserva el sentido de la desigualdad (invertido respecto a la lógica de aislamiento).", en: "Case 1: Assume {0} > 0. Multiply by {1} (positive). Inequality direction preserved (flipped from isolation logic)." },
    SolveDesc { template: "Power Equals Base with symbolic base '{0}': case split -> a=1: AllReals, a=0: x>0, otherwise: x=1", es: "Potencia igual a base con base simbólica '{0}': separación de casos -> a=1: todos los reales, a=0: x>0, en otro caso: x=1", en: "Power Equals Base with symbolic base '{0}': case split -> a=1: AllReals, a=0: x>0, otherwise: x=1" },
    SolveDesc { template: "Power Equals Base Shortcut: 0^{0} = 0 -> {1} > 0 (0^0 undefined, 0^t for t<0 undefined)", es: "Atajo potencia igual a base: 0^{0} = 0 -> {1} > 0 (0^0 indefinido, 0^t para t<0 indefinido)", en: "Power Equals Base Shortcut: 0^{0} = 0 -> {1} > 0 (0^0 undefined, 0^t for t<0 undefined)" },
    SolveDesc { template: "Variable '{0}' canceled during simplification. Solution depends on constraint: {1} = 0", es: "La variable '{0}' se canceló durante la simplificación. La solución depende de la restricción: {1} = 0", en: "Variable '{0}' canceled during simplification. Solution depends on constraint: {1} = 0" },
    SolveDesc { template: "Applied SolveTactic normalization (Assume mode) to enable logarithm isolation", es: "Se aplicó la normalización de SolveTactic (modo Asumir) para habilitar el aislamiento por logaritmo", en: "Applied SolveTactic normalization (Assume mode) to enable logarithm isolation" },
    SolveDesc { template: "Pattern: {0}^{1} = {2}^{3} -> {4} = {5} (equal bases imply equal exponents when base != 0, 1)", es: "Patrón: {0}^{1} = {2}^{3} -> {4} = {5} (bases iguales implican exponentes iguales cuando la base != 0, 1)", en: "Pattern: {0}^{1} = {2}^{3} -> {4} = {5} (equal bases imply equal exponents when base != 0, 1)" },
    SolveDesc { template: "Case 2: Assume {0} < 0. Multiply by negative denominator (flips inequality).", es: "Caso 2: Supón {0} < 0. Multiplica por el denominador negativo (invierte la desigualdad).", en: "Case 2: Assume {0} < 0. Multiply by negative denominator (flips inequality)." },
    SolveDesc { template: "|u| = a se descompone en u = a y u = -a. Despejando x (requiere Δ ≥ 0)", es: "|u| = a se descompone en u = a y u = -a. Despejando x (requiere Δ ≥ 0)", en: "|u| = a splits into u = a and u = -a. Solving for x (requires Δ ≥ 0)" },
    SolveDesc { template: "Identificar forma cuadrática: a·x² + b·x + c = 0 con a = {0}, b = {1}, c = {2}", es: "Identificar forma cuadrática: a·x² + b·x + c = 0 con a = {0}, b = {1}, c = {2}", en: "Identify the quadratic form: a·x² + b·x + c = 0 with a = {0}, b = {1}, c = {2}" },
    SolveDesc { template: "Case 2: Assume {0} < 0. Multiply by {1} (negative). Inequality flips.", es: "Caso 2: Supón {0} < 0. Multiplica por {1} (negativo). La desigualdad se invierte.", en: "Case 2: Assume {0} < 0. Multiply by {1} (negative). Inequality flips." },
    SolveDesc { template: "Power Equals Base Shortcut: {0}^{1} = {2} -> {3} = 1 (B^1 = B always holds)", es: "Atajo potencia igual a base: {0}^{1} = {2} -> {3} = 1 (B^1 = B siempre se cumple)", en: "Power Equals Base Shortcut: {0}^{1} = {2} -> {3} = 1 (B^1 = B always holds)" },
    SolveDesc { template: "Take {0}-th root of both sides (even root implies absolute value)", es: "Toma la raíz {0}-ésima en ambos lados (la raíz par implica valor absoluto)", en: "Take {0}-th root of both sides (even root implies absolute value)" },
    SolveDesc { template: "Raise both sides to power {0} to eliminate fractional exponent", es: "Eleva ambos lados a la potencia {0} para eliminar el exponente fraccionario", en: "Raise both sides to power {0} to eliminate fractional exponent" },
    SolveDesc { template: "Raise both sides to power {0} to eliminate rational exponent", es: "Eleva ambos lados a la potencia {0} para eliminar el exponente racional", en: "Raise both sides to power {0} to eliminate rational exponent" },
    SolveDesc { template: "Detected quadratic equation. Applying quadratic formula.", es: "Se detectó una ecuación cuadrática. Aplicando la fórmula cuadrática.", en: "Detected quadratic equation. Applying quadratic formula." },
    SolveDesc { template: "Power Equals Base: {0}^{1} = {2} -> {3} = 1 (assuming base != 0, 1)", es: "Potencia igual a base: {0}^{1} = {2} -> {3} = 1 (suponiendo base != 0, 1)", en: "Power Equals Base: {0}^{1} = {2} -> {3} = 1 (assuming base != 0, 1)" },
    SolveDesc { template: "Case 1: Assume {0} > 0. Multiply by positive denominator.", es: "Caso 1: Supón {0} > 0. Multiplica por el denominador positivo.", en: "Case 1: Assume {0} > 0. Multiply by positive denominator." },
    SolveDesc { template: "|u| = a se descompone en u = a y u = -a. Despejando x", es: "|u| = a se descompone en u = a y u = -a. Despejando x", en: "|u| = a splits into u = a and u = -a. Solving for x" },
    SolveDesc { template: "Applied Rational Root Theorem to degree-{0} polynomial", es: "Se aplicó el Teorema de las Raíces Racionales al polinomio de grado {0}", en: "Applied Rational Root Theorem to degree-{0} polynomial" },
    SolveDesc { template: "1^x = 1 for all x -> any real number is a solution", es: "1^x = 1 para todo x -> cualquier número real es solución", en: "1^x = 1 for all x -> any real number is a solution" },
    SolveDesc { template: "Completar el cuadrado: sumar (b/2a)² a ambos lados", es: "Completar el cuadrado: sumar (b/2a)² a ambos lados", en: "Complete the square: add (b/2a)² to both sides" },
    SolveDesc { template: "1^x = 1 for all x, but RHS = {0} != 1 -> no solution", es: "1^x = 1 para todo x, pero el lado derecho = {0} != 1 -> sin solución", en: "1^x = 1 for all x, but RHS = {0} != 1 -> no solution" },
    SolveDesc { template: "Escribir lado izquierdo como cuadrado perfecto", es: "Escribir lado izquierdo como cuadrado perfecto", en: "Write the left-hand side as a perfect square" },
    SolveDesc { template: "Multiply both sides by -1 (flips inequality)", es: "Multiplica ambos lados por -1 (invierte la desigualdad)", en: "Multiply both sides by -1 (flips inequality)" },
    SolveDesc { template: "Take log base {0} of both sides (under guard: {1})", es: "Toma logaritmo en base {0} en ambos lados (bajo la condición: {1})", en: "Take log base {0} of both sides (under guard: {1})" },
    SolveDesc { template: "Producto igual a cero: algún factor es cero", es: "Producto igual a cero: algún factor es cero", en: "Product equal to zero: some factor is zero" },
    SolveDesc { template: "Move {0} and multiply by -1 (flips inequality)", es: "Pasa {0} al otro lado y multiplica por -1 (invierte la desigualdad)", en: "Move {0} and multiply by -1 (flips inequality)" },
    SolveDesc { template: "Dividir ambos lados por a (requiere a ≠ 0)", es: "Dividir ambos lados por a (requiere a ≠ 0)", en: "Divide both sides by a (requires a ≠ 0)" },
    SolveDesc { template: "Mover término constante al lado derecho", es: "Mover término constante al lado derecho", en: "Move the constant term to the right-hand side" },
    SolveDesc { template: "Collect terms in {0} and factor: {1} · {0} = {2}", es: "Agrupa los términos en {0} y factoriza: {1} · {0} = {2}", en: "Collect terms in {0} and factor: {1} · {0} = {2}" },
    SolveDesc { template: "Exponentiate both sides with base e", es: "Exponencia ambos lados con base e", en: "Exponentiate both sides with base e" },
    SolveDesc { template: "Tomar raíz cuadrada en ambos lados", es: "Tomar raíz cuadrada en ambos lados", en: "Take the square root of both sides" },
    SolveDesc { template: "Even power cannot be negative ({0} {1} {2})", es: "Una potencia par no puede ser negativa ({0} {1} {2})", en: "Even power cannot be negative ({0} {1} {2})" },
    SolveDesc { template: "Swap sides to put variable on LHS", es: "Intercambia los lados para dejar la variable a la izquierda", en: "Swap sides to put variable on LHS" },
    SolveDesc { template: "Take natural log of both sides", es: "Toma logaritmo natural en ambos lados", en: "Take natural log of both sides" },
    SolveDesc { template: "Take log base e of both sides", es: "Toma logaritmo en base e en ambos lados", en: "Take log base e of both sides" },
    SolveDesc { template: "{0} (residual, budget exhausted)", es: "{0} (residual, presupuesto agotado)", en: "{0} (residual, budget exhausted)" },
    SolveDesc { template: "Take log base {0} of both sides", es: "Toma logaritmo en base {0} en ambos lados", en: "Take log base {0} of both sides" },
    SolveDesc { template: "Take {0}-th root of both sides", es: "Toma la raíz {0}-ésima en ambos lados", en: "Take {0}-th root of both sides" },
    SolveDesc { template: "Detected substitution: u = {0}", es: "Sustitución detectada: u = {0}", en: "Detected substitution: u = {0}" },
    SolveDesc { template: "Split absolute value ({0}): {1} {2} {3}", es: "Descompón el valor absoluto ({0}): {1} {2} {3}", en: "Split absolute value ({0}): {1} {2} {3}" },
    SolveDesc { template: "Power isolation terminated", es: "Aislamiento de la potencia terminado", en: "Power isolation terminated" },
    SolveDesc { template: "Take arcsin of both sides", es: "Toma arcoseno en ambos lados", en: "Take arcsin of both sides" },
    SolveDesc { template: "Take arccos of both sides", es: "Toma arcocoseno en ambos lados", en: "Take arccos of both sides" },
    SolveDesc { template: "Take arctan of both sides", es: "Toma arcotangente en ambos lados", en: "Take arctan of both sides" },
    SolveDesc { template: "Subtract {0} from both sides", es: "Resta {0} en ambos lados", en: "Subtract {0} from both sides" },
    SolveDesc { template: "Factorized equation: {0} = 0", es: "Ecuación factorizada: {0} = 0", en: "Factorized equation: {0} = 0" },
    SolveDesc { template: "Collect and factor {0} terms", es: "Agrupa y factoriza los términos en {0}", en: "Collect and factor {0} terms" },
    SolveDesc { template: "Substituted equation: {0} {1} {2}", es: "Ecuación sustituida: {0} {1} {2}", en: "Substituted equation: {0} {1} {2}" },
    SolveDesc { template: "Expand distributive law", es: "Aplica la propiedad distributiva", en: "Expand distributive law" },
    SolveDesc { template: "Multiply both sides by {0}", es: "Multiplica ambos lados por {0}", en: "Multiply both sides by {0}" },
    SolveDesc { template: "Move {0} terms to one side", es: "Pasa los términos en {0} a un lado", en: "Move {0} terms to one side" },
    SolveDesc { template: "Move terms to one side", es: "Mueve los términos a un lado", en: "Move terms to one side" },
    SolveDesc { template: "Conditional solution: {0}", es: "Solución condicional: {0}", en: "Conditional solution: {0}" },
    SolveDesc { template: "Raise both sides to 1/{0}", es: "Eleva ambos lados a 1/{0}", en: "Raise both sides to 1/{0}" },
    SolveDesc { template: "Exponentiate (base e)", es: "Exponencia (base e)", en: "Exponentiate (base e)" },
    SolveDesc { template: "Divide both sides by {0}", es: "Divide ambos lados entre {0}", en: "Divide both sides by {0}" },
    SolveDesc { template: "--- End of Case {0} ---", es: "--- Fin del Caso {0} ---", en: "--- End of Case {0} ---" },
    SolveDesc { template: "Back-substitute: {0} = {1}", es: "Sustitución inversa: {0} = {1}", en: "Back-substitute: {0} = {1}" },
    SolveDesc { template: "Factorizar x común", es: "Factorizar x común", en: "Factor out the common x" },
    SolveDesc { template: "Add {0} to both sides", es: "Suma {0} en ambos lados", en: "Add {0} to both sides" },
    SolveDesc { template: "Solve factor: {0} = 0", es: "Resuelve el factor: {0} = 0", en: "Solve factor: {0} = 0" },
    SolveDesc { template: "Square both sides", es: "Eleva al cuadrado ambos lados", en: "Square both sides" },
    SolveDesc { template: "Collect terms in {0}", es: "Agrupa los términos en {0}", en: "Collect terms in {0}" },
    SolveDesc { template: "Take natural log", es: "Toma logaritmo natural", en: "Take natural log" },
    SolveDesc { template: "Resolver {0} = 0", es: "Resolver {0} = 0", en: "Solve {0} = 0" },
    SolveDesc { template: "{0} (residual)", es: "{0} (residual)", en: "{0} (residual)" },
    SolveDesc { template: "Factor out {0}", es: "Factoriza {0}", en: "Factor out {0}" },
    SolveDesc { template: "Divide by {0}", es: "Divide entre {0}", en: "Divide by {0}" },
    SolveDesc { template: "Case 1", es: "Caso 1", en: "Case 1" },
    SolveDesc { template: "Case 2", es: "Caso 2", en: "Case 2" },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_english_top_level_into_spanish() {
        assert_eq!(
            localize_solve_description("Subtract 3 from both sides", Language::Es),
            "Resta 3 en ambos lados"
        );
        assert_eq!(
            localize_solve_description("Divide both sides by 2", Language::Es),
            "Divide ambos lados entre 2"
        );
        assert_eq!(
            localize_solve_description(
                "Detected quadratic equation. Applying quadratic formula.",
                Language::Es
            ),
            "Se detectó una ecuación cuadrática. Aplicando la fórmula cuadrática."
        );
        assert_eq!(
            localize_solve_description(
                "Applied Rational Root Theorem to degree-3 polynomial",
                Language::Es
            ),
            "Se aplicó el Teorema de las Raíces Racionales al polinomio de grado 3"
        );
    }

    #[test]
    fn keeps_english_under_en_and_translates_spanish_substeps() {
        // English source stays English under En (identity).
        assert_eq!(
            localize_solve_description("Subtract 3 from both sides", Language::En),
            "Subtract 3 from both sides"
        );
        // Spanish completing-the-square sub-step becomes English under En.
        assert_eq!(
            localize_solve_description("Tomar raíz cuadrada en ambos lados", Language::En),
            "Take the square root of both sides"
        );
        // ...and stays Spanish under Es (identity).
        assert_eq!(
            localize_solve_description("Tomar raíz cuadrada en ambos lados", Language::Es),
            "Tomar raíz cuadrada en ambos lados"
        );
    }

    #[test]
    fn multi_placeholder_round_trips_variable_parts() {
        let out = localize_solve_description("Split absolute value (Case 1): x = 3", Language::Es);
        assert!(
            out.starts_with("Descompón el valor absoluto ("),
            "got {out}"
        );
        assert!(out.ends_with(": x = 3"), "got {out}");
    }

    #[test]
    fn unknown_description_passes_through() {
        assert_eq!(
            localize_solve_description("Totally unknown narration", Language::Es),
            "Totally unknown narration"
        );
    }
}
