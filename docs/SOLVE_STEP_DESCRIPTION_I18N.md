# Solve-step description i18n — catalog + implementation plan

Status: SCOPED, not yet implemented. Cycle D of the didactic-quality round.

## Problem

`solve_steps[].description` (top-level) is hard-coded ENGLISH and `--lang` never touches it; the completing-the-square substeps are hard-coded SPANISH. So a `solve()` trace is mixed EN/ES in BOTH `--lang es` (default) and `--lang en`. Fix: localize at the boundary `crates/cas_solver/src/eval_output_presentation_solve_steps.rs::collect_output_solve_steps` (line ~64 description clone), which keeps cas_solver_core + its assert_eq! unit tests untouched.

## Threading (Language = cas_solver_core::Language)

Add a `language` param through: `collect_output_solve_steps` <- `collect_eval_artifacts` (present/collect.rs) <- `finalize_eval_run` (present.rs) <- `evaluate_eval_with_session` (eval_command_runtime.rs, PUBLIC) <- `evaluate_eval_command_pretty_with_session` (cas_session, PUBLIC) <- cas_cli eval.rs (pass `args.lang.to_language()`). Also update derive_didactic_audit.rs test + cas_session/benches/frontend_session.rs (pass Language::Es). cas_api_models can't hold cas_solver_core::Language, so thread the param explicitly (do NOT route via EvalSessionRunConfig).

## Localizer

A `localize_solve_description(desc, lang)` in cas_solver: ordered-segment template matcher over the table below. For en-source entries: es-mode translates, en-mode keeps. For es-source entries (substeps): en-mode translates, es-mode keeps. Unmatched -> return as-is.

### Safe entries (68) — static or single/literal-bounded placeholders (implement first)

| source_string | source | surfaces | es | en |
|---|---|---|---|---|
| `1^x = 1 for all x -> any real number is a solution` | en | top | 1^x = 1 para todo x -> cualquier número real es solución | 1^x = 1 for all x -> any real number is a solution |
| `1^x = 1 for all x, but RHS = {0} != 1 -> no solution` | en | top | 1^x = 1 para todo x, pero el lado derecho = {0} != 1 -> sin solución | 1^x = 1 for all x, but RHS = {0} != 1 -> no solution |
| `Power Equals Base Shortcut: 0^{0} = 0 -> {1} > 0 (0^0 undefined, 0^t for t<0 undefined)` | en | top | Atajo potencia igual a base: 0^{0} = 0 -> {1} > 0 (0^0 indefinido, 0^t para t<0 indefinido) | Power Equals Base Shortcut: 0^{0} = 0 -> {1} > 0 (0^0 undefined, 0^t for t<0 undefined) |
| `Power Equals Base Shortcut: {0}^{1} = {2} -> {3} = 1 (B^1 = B always holds)` | en | top | Atajo potencia igual a base: {0}^{1} = {2} -> {3} = 1 (B^1 = B siempre se cumple) | Power Equals Base Shortcut: {0}^{1} = {2} -> {3} = 1 (B^1 = B always holds) |
| `Power Equals Base: {0}^{1} = {2} -> {3} = 1 (assuming base != 0, 1)` | en | top | Potencia igual a base: {0}^{1} = {2} -> {3} = 1 (suponiendo base != 0, 1) | Power Equals Base: {0}^{1} = {2} -> {3} = 1 (assuming base != 0, 1) |
| `Power Equals Base with symbolic base '{0}': case split -> a=1: AllReals, a=0: x>0, otherwise: x=1` | en | top | Potencia igual a base con base simbólica '{0}': separación de casos -> a=1: todos los reales, a=0: x>0, en otro caso: x=1 | Power Equals Base with symbolic base '{0}': case split -> a=1: AllReals, a=0: x>0, otherwise: x=1 |
| `Pattern: {0}^{1} = {2}^{3} -> {4} = {5} (equal bases imply equal exponents when base != 0, 1)` | en | top | Patrón: {0}^{1} = {2}^{3} -> {4} = {5} (bases iguales implican exponentes iguales cuando la base != 0, 1) | Pattern: {0}^{1} = {2}^{3} -> {4} = {5} (equal bases imply equal exponents when base != 0, 1) |
| `Power isolation terminated` | en | top | Aislamiento de la potencia terminado | Power isolation terminated |
| `Take {0}-th root of both sides (even root implies absolute value)` | en | top | Toma la raíz {0}-ésima en ambos lados (la raíz par implica valor absoluto) | Take {0}-th root of both sides (even root implies absolute value) |
| `Take {0}-th root of both sides` | en | top | Toma la raíz {0}-ésima en ambos lados | Take {0}-th root of both sides |
| `Subtract {0} from both sides` | en | top | Resta {0} en ambos lados | Subtract {0} from both sides |
| `Add {0} to both sides` | en | top | Suma {0} en ambos lados | Add {0} to both sides |
| `Move {0} and multiply by -1 (flips inequality)` | en | top | Pasa {0} al otro lado y multiplica por -1 (invierte la desigualdad) | Move {0} and multiply by -1 (flips inequality) |
| `Multiply both sides by -1 (flips inequality)` | en | top | Multiplica ambos lados por -1 (invierte la desigualdad) | Multiply both sides by -1 (flips inequality) |
| `Swap sides to put variable on LHS` | en | top | Intercambia los lados para dejar la variable a la izquierda | Swap sides to put variable on LHS |
| `Applied SolveTactic normalization (Assume mode) to enable logarithm isolation` | en | top | Se aplicó la normalización de SolveTactic (modo Asumir) para habilitar el aislamiento por logaritmo | Applied SolveTactic normalization (Assume mode) to enable logarithm isolation |
| `Divide both sides by {0}` | en | top | Divide ambos lados entre {0} | Divide both sides by {0} |
| `Multiply both sides by {0}` | en | top | Multiplica ambos lados por {0} | Multiply both sides by {0} |
| `--- End of Case {0} ---` | en | top | --- Fin del Caso {0} --- | --- End of Case {0} --- |
| `Case 1: Assume {0} > 0. Multiply by positive denominator.` | en | top | Caso 1: Supón {0} > 0. Multiplica por el denominador positivo. | Case 1: Assume {0} > 0. Multiply by positive denominator. |
| `Case 2: Assume {0} < 0. Multiply by negative denominator (flips inequality).` | en | top | Caso 2: Supón {0} < 0. Multiplica por el denominador negativo (invierte la desigualdad). | Case 2: Assume {0} < 0. Multiply by negative denominator (flips inequality). |
| `Case 1: Assume {0} > 0. Multiply by {1} (positive). Inequality direction preserved (flipped from isolation logic).` | en | top | Caso 1: Supón {0} > 0. Multiplica por {1} (positivo). Se preserva el sentido de la desigualdad (invertido respecto a la lógica de aislamiento). | Case 1: Assume {0} > 0. Multiply by {1} (positive). Inequality direction preserved (flipped from isolation logic). |
| `Case 2: Assume {0} < 0. Multiply by {1} (negative). Inequality flips.` | en | top | Caso 2: Supón {0} < 0. Multiplica por {1} (negativo). La desigualdad se invierte. | Case 2: Assume {0} < 0. Multiply by {1} (negative). Inequality flips. |
| `Take log base {0} of both sides` | en | top | Toma logaritmo en base {0} en ambos lados | Take log base {0} of both sides |
| `Take log base {0} of both sides (under guard: {1})` | en | top | Toma logaritmo en base {0} en ambos lados (bajo la condición: {1}) | Take log base {0} of both sides (under guard: {1}) |
| `Conditional solution: {0}` | en | top | Solución condicional: {0} | Conditional solution: {0} |
| `{0} (residual)` | en | top | {0} (residual) | {0} (residual) |
| `{0} (residual, budget exhausted)` | en | top | {0} (residual, presupuesto agotado) | {0} (residual, budget exhausted) |
| `Raise both sides to power {0} to eliminate fractional exponent` | en | top | Eleva ambos lados a la potencia {0} para eliminar el exponente fraccionario | Raise both sides to power {0} to eliminate fractional exponent |
| `Raise both sides to power {0} to eliminate rational exponent` | en | top | Eleva ambos lados a la potencia {0} para eliminar el exponente racional | Raise both sides to power {0} to eliminate rational exponent |
| `Variable '{0}' canceled during simplification. Solution depends on constraint: {1} = 0` | en | top | La variable '{0}' se canceló durante la simplificación. La solución depende de la restricción: {1} = 0 | Variable '{0}' canceled during simplification. Solution depends on constraint: {1} = 0 |
| `Case 1` | en | top | Caso 1 | Case 1 |
| `Case 2` | en | top | Caso 2 | Case 2 |
| `Back-substitute: {0} = {1}` | en | substep | Sustitución inversa: {0} = {1} | Back-substitute: {0} = {1} |
| `Raise both sides to 1/{0}` | en | top | Eleva ambos lados a 1/{0} | Raise both sides to 1/{0} |
| `Detected quadratic equation. Applying quadratic formula.` | en | top | Se detectó una ecuación cuadrática. Aplicando la fórmula cuadrática. | Detected quadratic equation. Applying quadratic formula. |
| `Factorized equation: {0} = 0` | en | top | Ecuación factorizada: {0} = 0 | Factorized equation: {0} = 0 |
| `Solve factor: {0} = 0` | en | top | Resuelve el factor: {0} = 0 | Solve factor: {0} = 0 |
| `Factorizar x común` | es | substep | Factorizar x común | Factor out the common x |
| `Producto igual a cero: algún factor es cero` | es | substep | Producto igual a cero: algún factor es cero | Product equal to zero: some factor is zero |
| `Resolver {0} = 0` | es | substep | Resolver {0} = 0 | Solve {0} = 0 |
| `Identificar forma cuadrática: a·x² + b·x + c = 0 con a = {0}, b = {1}, c = {2}` | es | substep | Identificar forma cuadrática: a·x² + b·x + c = 0 con a = {0}, b = {1}, c = {2} | Identify the quadratic form: a·x² + b·x + c = 0 with a = {0}, b = {1}, c = {2} |
| `Dividir ambos lados por a (requiere a ≠ 0)` | es | substep | Dividir ambos lados por a (requiere a ≠ 0) | Divide both sides by a (requires a ≠ 0) |
| `Mover término constante al lado derecho` | es | substep | Mover término constante al lado derecho | Move the constant term to the right-hand side |
| `Completar el cuadrado: sumar (b/2a)² a ambos lados` | es | substep | Completar el cuadrado: sumar (b/2a)² a ambos lados | Complete the square: add (b/2a)² to both sides |
| `Escribir lado izquierdo como cuadrado perfecto` | es | substep | Escribir lado izquierdo como cuadrado perfecto | Write the left-hand side as a perfect square |
| `Tomar raíz cuadrada en ambos lados` | es | substep | Tomar raíz cuadrada en ambos lados | Take the square root of both sides |
| `\|u\| = a se descompone en u = a y u = -a. Despejando x (requiere Δ ≥ 0)` | es | substep | \|u\| = a se descompone en u = a y u = -a. Despejando x (requiere Δ ≥ 0) | \|u\| = a splits into u = a and u = -a. Solving for x (requires Δ ≥ 0) |
| `\|u\| = a se descompone en u = a y u = -a. Despejando x` | es | substep | \|u\| = a se descompone en u = a y u = -a. Despejando x | \|u\| = a splits into u = a and u = -a. Solving for x |
| `Applied Rational Root Theorem to degree-{0} polynomial` | en | top | Se aplicó el Teorema de las Raíces Racionales al polinomio de grado {0} | Applied Rational Root Theorem to degree-{0} polynomial |
| `Collect and factor {0} terms` | en | top | Agrupa y factoriza los términos en {0} | Collect and factor {0} terms |
| `Take log base e of both sides` | en | top | Toma logaritmo en base e en ambos lados | Take log base e of both sides |
| `Exponentiate both sides with base e` | en | top | Exponencia ambos lados con base e | Exponentiate both sides with base e |
| `Take natural log of both sides` | en | top | Toma logaritmo natural en ambos lados | Take natural log of both sides |
| `Square both sides` | en | top | Eleva al cuadrado ambos lados | Square both sides |
| `Take arcsin of both sides` | en | top | Toma arcoseno en ambos lados | Take arcsin of both sides |
| `Take arccos of both sides` | en | top | Toma arcocoseno en ambos lados | Take arccos of both sides |
| `Take arctan of both sides` | en | top | Toma arcotangente en ambos lados | Take arctan of both sides |
| `Exponentiate (base e)` | en | top | Exponencia (base e) | Exponentiate (base e) |
| `Take natural log` | en | top | Toma logaritmo natural | Take natural log |
| `Move terms to one side` | en | top | Mueve los términos a un lado | Move terms to one side |
| `Collect terms in {0} and factor: {1} · {0} = {2}` | en | top | Agrupa los términos en {0} y factoriza: {1} · {0} = {2} | Collect terms in {0} and factor: {1} · {0} = {2} |
| `Divide by {0}` | en | top | Divide entre {0} | Divide by {0} |
| `Collect terms in {0}` | en | top | Agrupa los términos en {0} | Collect terms in {0} |
| `Detected substitution: u = {0}` | en | top | Sustitución detectada: u = {0} | Detected substitution: u = {0} |
| `Expand distributive law` | en | top | Aplica la propiedad distributiva | Expand distributive law |
| `Move {0} terms to one side` | en | top | Pasa los términos en {0} a un lado | Move {0} terms to one side |
| `Factor out {0}` | en | top | Factoriza {0} | Factor out {0} |

### Fragile entries (4) — space-separated multi-placeholders; re-template (merge trailing placeholders) or defer

| source_string | es | en |
|---|---|---|
| `Even power cannot be negative ({0} {1} {2})` | Una potencia par no puede ser negativa ({0} {1} {2}) | Even power cannot be negative ({0} {1} {2}) |
| `{0}{1}` | {0}{1} | {0}{1} |
| `Split absolute value ({0}): {1} {2} {3}` | Descompón el valor absoluto ({0}): {1} {2} {3} | Split absolute value ({0}): {1} {2} {3} |
| `Substituted equation: {0} {1} {2}` | Ecuación sustituida: {0} {1} {2} | Substituted equation: {0} {1} {2} |

## Wire-test surface

Contract tests that assert `solve_steps[].description` (cas_cli, cas_solver) will flip to Spanish under the default `es` and need updating in lockstep. cas_solver_core unit tests are NOT affected (source strings unchanged).
