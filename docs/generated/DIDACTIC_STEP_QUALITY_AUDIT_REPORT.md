# Didactic Step Quality Audit Report

Generated from [didactic_step_quality_cases.csv](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/didactic_step_quality_cases.csv).

Command: `cargo test -p cas_didactic --test didactic_step_quality_audit didactic_step_quality_audit_generates_markdown_report -- --ignored --exact --nocapture`

## Summary

| id | category | steps | wire substeps | flags |
| --- | --- | ---: | ---: | --- |
| `rationalize_linear_root` | `rationalization` | 6 | 3 | none |
| `nested_fraction_one_over_sum` | `nested_fraction` | 2 | 2 | none |
| `nested_fraction_fraction_over_sum` | `nested_fraction` | 2 | 2 | none |
| `combine_like_terms_basic` | `combine` | 2 | 1 | none |
| `same_denominator_fraction_focus` | `fractions` | 2 | 2 | none |
| `cancel_factors_fraction` | `cancellation` | 1 | 1 | none |
| `difference_of_squares_quotient` | `quotient` | 1 | 2 | none |
| `pythagorean_identity` | `trig` | 1 | 1 | none |
| `inverse_trig_identity` | `inverse_trig` | 5 | 2 | none |
| `polynomial_expansion_cancel` | `polynomial` | 6 | 3 | none |
| `perfect_square_root` | `radicals` | 2 | 2 | none |
| `cube_quotient_radical` | `quotient` | 4 | 5 | none |

## rationalize_linear_root (rationalization)

- Input: `1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)`
- Focus: `web_substeps_latex_and_human_explanation`
- Final result: `0`
- Step count: `6`
- Wire substep count: `3`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Multiplicar por el conjugado
   Before: 1/(sqrt(x) - 1) - (sqrt(x) + 1)/(x - 1)
   [Simplificación de fracción compleja]
      → Cambiar el signo para formar el conjugado
        sqrt(x) - 1 → sqrt(x) + 1
      → Multiplicar numerador y denominador por ese conjugado
        1/(sqrt(x) - 1) → (sqrt(x) + 1)/((sqrt(x) - 1)  ·  (sqrt(x) + 1))
      → En el denominador aparece una diferencia de cuadrados
        (sqrt(x) + 1)/((sqrt(x) - 1)  ·  (sqrt(x) + 1)) → (sqrt(x) + 1)/(x - 1^2)
   After: (sqrt(x) + 1)/(x - 1^2) - (sqrt(x) + 1)/(x - 1)
2. Calcular potencia numérica
   Before: (sqrt(x) + 1)/(x - 1^2) - (sqrt(x) + 1)/(x - 1)
   After: (sqrt(x) + 1)/(x - 1) - (sqrt(x) + 1)/(x - 1)
3. Restar dos expresiones iguales
   Before: (sqrt(x) + 1)/(x - 1) - (sqrt(x) + 1)/(x - 1)
   After: 0
```

### Wire / Web Steps

1. `Racionalizar el denominador`
   - before: `1/(sqrt(x) - 1) - (sqrt(x) + 1)/(x - 1)`
   - after: `(sqrt(x) + 1)/(x - 1^2) - (sqrt(x) + 1)/(x - 1)`
   - before_latex: `{\color{red}{\frac{1}{\sqrt{x} - 1}}} - \frac{1 + \sqrt{x}}{x - 1}`
   - after_latex: `{\color{green}{\frac{\sqrt{x} + 1}{x - {(-1)}^{2}}}} - \frac{1 + \sqrt{x}}{x - 1}`
   - substeps:
     1. `Cambiar el signo para formar el conjugado`
        - before_latex: `\sqrt{x} - 1`
        - after_latex: `\sqrt{x} + 1`
     2. `Multiplicar numerador y denominador por ese conjugado`
        - before_latex: `\frac{1}{\sqrt{x} - 1}`
        - after_latex: `\frac{(\sqrt{x} + 1)}{(\sqrt{x} - 1) \cdot (\sqrt{x} + 1)}`
     3. `En el denominador aparece una diferencia de cuadrados`
        - before_latex: `\frac{(\sqrt{x} + 1)}{(\sqrt{x} - 1) \cdot (\sqrt{x} + 1)}`
        - after_latex: `\frac{\sqrt{x} + 1}{x - 1^{2}}`
2. `Restar dos expresiones iguales`
   - before: `(sqrt(x) + 1)/(x - 1) - (sqrt(x) + 1)/(x - 1)`
   - after: `0`
   - before_latex: `{\color{red}{\frac{1 + \sqrt{x}}{x - 1} - \frac{1 + \sqrt{x}}{x - 1}}}`
   - after_latex: `{\color{green}{0}}`
   - substeps: none

## nested_fraction_one_over_sum (nested_fraction)

- Input: `1/(1/x + 1/y)`
- Focus: `narrate_common_denominator_without_magic`
- Final result: `x * y / (x + y)`
- Step count: `2`
- Wire substep count: `2`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Sumar fracciones
   Before: 1/(1/x + 1/y)
   [Simplificación de fracción compleja]
      → Juntar todo en una sola fracción
        1/x + 1/y → (x + y)/(x ·  y)
   After: 1/((x + y)/(x ·  y))
2. Simplificar fracción anidada
   Before: 1/((x + y)/(x ·  y))
   [Simplificación de fracción compleja]
      → Dividir entre una fracción equivale a invertirla
        1/((x + y)/(x ·  y)) → (x ·  y)/(x + y)
   After: (x ·  y)/(x + y)
```

### Wire / Web Steps

1. `Sumar fracciones`
   - before: `1/(1/x + 1/y)`
   - after: `1/((x + y)/(x ·  y))`
   - before_latex: `\frac{1}{{\color{red}{\frac{1}{x} + \frac{1}{y}}}}`
   - after_latex: `\frac{1}{{\color{green}{\frac{x + y}{x\cdot y}}}}`
   - substeps:
     1. `Juntar todo en una sola fracción`
        - before_latex: `\frac{1}{x} + \frac{1}{y}`
        - after_latex: `\frac{x + y}{x\cdot y}`
2. `Simplificar fracción anidada`
   - before: `1/((x + y)/(x ·  y))`
   - after: `(x ·  y)/(x + y)`
   - before_latex: `{\color{red}{\frac{1}{\frac{x + y}{x\cdot y}}}}`
   - after_latex: `{\color{green}{\frac{x\cdot y}{x + y}}}`
   - substeps:
     1. `Dividir entre una fracción equivale a invertirla`
        - before_latex: `\frac{1}{\frac{x + y}{x\cdot y}}`
        - after_latex: `\frac{x\cdot y}{x + y}`

## nested_fraction_fraction_over_sum (nested_fraction)

- Input: `(1/x)/(1/y + 1/z)`
- Focus: `explain_inner_fraction_then_outer_division`
- Final result: `y * z / (x * y + x * z)`
- Step count: `2`
- Wire substep count: `2`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Sumar fracciones
   Before: (1/x)/(1/y + 1/z)
   [Simplificación de fracción compleja]
      → Juntar todo en una sola fracción
        1/y + 1/z → (y + z)/(y ·  z)
   After: (1/x)/((y + z)/(y ·  z))
2. Simplificar fracción anidada
   Before: (1/x)/((y + z)/(y ·  z))
   [Simplificación de fracción compleja]
      → Dividir entre una fracción equivale a invertirla
        (1/x)/((y + z)/(y ·  z)) → (y ·  z)/((y + z) ·  x)
   After: (y ·  z)/((y + z) ·  x)
```

### Wire / Web Steps

1. `Sumar fracciones`
   - before: `(1/x)/(1/y + 1/z)`
   - after: `(1/x)/((y + z)/(y ·  z))`
   - before_latex: `\frac{\frac{1}{x}}{{\color{red}{\frac{1}{y} + \frac{1}{z}}}}`
   - after_latex: `\frac{\frac{1}{x}}{{\color{green}{\frac{y + z}{y\cdot z}}}}`
   - substeps:
     1. `Juntar todo en una sola fracción`
        - before_latex: `\frac{1}{y} + \frac{1}{z}`
        - after_latex: `\frac{y + z}{y\cdot z}`
2. `Simplificar fracción anidada`
   - before: `(1/x)/((y + z)/(y ·  z))`
   - after: `(y ·  z)/((y + z) ·  x)`
   - before_latex: `{\color{red}{\frac{\frac{1}{x}}{\frac{y + z}{y\cdot z}}}}`
   - after_latex: `{\color{green}{\frac{y\cdot z}{(y + z)\cdot x}}}`
   - substeps:
     1. `Dividir entre una fracción equivale a invertirla`
        - before_latex: `\frac{\frac{1}{x}}{\frac{y + z}{y\cdot z}}`
        - after_latex: `\frac{y\cdot z}{(y + z)\cdot x}`

## combine_like_terms_basic (combine)

- Input: `2*x + 3*x + 0`
- Focus: `avoid_noop_and_show_collection`
- Final result: `5 * x`
- Step count: `2`
- Wire substep count: `1`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Quitar el 0
   Before: 2 ·  x + 3 ·  x + 0
   After: 2 ·  x + 3 ·  x
2. Agrupar términos semejantes
   Before: 2 ·  x + 3 ·  x
      → Agrupar términos semejantes y sumar coeficientes
        2 ·  x + 3 ·  x → 5 ·  x
   After: 5 ·  x
```

### Wire / Web Steps

1. `Agrupar términos semejantes`
   - before: `2 ·  x + 3 ·  x`
   - after: `5 ·  x`
   - before_latex: `{\color{red}{2\cdot x + 3\cdot x}}`
   - after_latex: `{\color{green}{5\cdot x}}`
   - substeps:
     1. `Agrupar términos semejantes y sumar coeficientes`
        - before_latex: `2\cdot x + 3\cdot x`
        - after_latex: `5\cdot x`

## same_denominator_fraction_focus (fractions)

- Input: `1 + a/d + b/d`
- Focus: `focus_only_on_fraction_part`
- Final result: `(a + b + d) / d`
- Step count: `2`
- Wire substep count: `2`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Llevar a denominador común
   Before: a/d + b/d + 1
   [Simplificación de fracción compleja]
      → Poner ambos términos sobre el mismo denominador
        a/d + 1 → (a + d)/d
   After: b/d + (a + d)/d
2. Sumar fracciones
   Before: b/d + (a + d)/d
   [Simplificación de fracción compleja]
      → Juntar todo en una sola fracción
        b/d + (a + d)/d → (a + b + d)/d
   After: (a + b + d)/d
```

### Wire / Web Steps

1. `Llevar a denominador común`
   - before: `a/d + b/d + 1`
   - after: `b/d + (a + d)/d`
   - before_latex: `{\color{red}{1 + \frac{a}{d}}} + \frac{b}{d}`
   - after_latex: `{\color{green}{\frac{a + d}{d}}} + \frac{b}{d}`
   - substeps:
     1. `Poner ambos términos sobre el mismo denominador`
        - before_latex: `\frac{a}{d} + 1`
        - after_latex: `\frac{a + d}{d}`
2. `Sumar fracciones`
   - before: `b/d + (a + d)/d`
   - after: `(a + b + d)/d`
   - before_latex: `{\color{red}{\frac{a + d}{d} + \frac{b}{d}}}`
   - after_latex: `{\color{green}{\frac{a + b + d}{d}}}`
   - substeps:
     1. `Juntar todo en una sola fracción`
        - before_latex: `\frac{b}{d} + \frac{a + d}{d}`
        - after_latex: `\frac{a + b + d}{d}`

## cancel_factors_fraction (cancellation)

- Input: `(2*x)/(4*x)`
- Focus: `show_factorization_and_cancellation`
- Final result: `1/2`
- Step count: `1`
- Wire substep count: `1`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Cancelar factor común
   Before: (2 ·  x)/(4 ·  x)
      → Como x aparece arriba y abajo, se cancela
        (2 ·  x)/(4 ·  x) → 2/4
   After: 2/4
```

### Wire / Web Steps

1. `Cancelar un factor común`
   - before: `(2 ·  x)/(4 ·  x)`
   - after: `2/4`
   - before_latex: `{\color{red}{\frac{2\cdot x}{4\cdot x}}}`
   - after_latex: `{\color{green}{\frac{2}{4}}}`
   - substeps:
     1. `Como x aparece arriba y abajo, se cancela`
        - before_latex: `\frac{2\cdot x}{4\cdot x}`
        - after_latex: `\frac{2}{4}`

## difference_of_squares_quotient (quotient)

- Input: `(x^2 - 1)/(x - 1)`
- Focus: `factor_then_cancel_exactly`
- Final result: `x + 1`
- Step count: `1`
- Wire substep count: `2`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Cancelar factor común
   Before: ((x + 1) ·  (x - 1))/(x - 1)
   [Factorización de polinomios]
      → Reescribir el numerador como diferencia de cuadrados
        x^2 - 1 → (x + 1) ·  (x - 1)
      → Ahora se cancela el factor x - 1
        ((x + 1) ·  (x - 1))/(x - 1) → x + 1
   After: x + 1
```

### Wire / Web Steps

1. `Factorizar una diferencia de cuadrados y cancelar`
   - before: `((x + 1) ·  (x - 1))/(x - 1)`
   - after: `x + 1`
   - before_latex: `{\color{red}{\frac{(1 + x)\cdot (x - 1)}{x - 1}}}`
   - after_latex: `{\color{green}{1 + x}}`
   - substeps:
     1. `Reescribir el numerador como diferencia de cuadrados`
        - before_latex: `{x}^{2} - 1`
        - after_latex: `(x + 1)\cdot (x - 1)`
     2. `Ahora se cancela el factor x - 1`
        - before_latex: `\frac{(x + 1)\cdot (x - 1)}{x - 1}`
        - after_latex: `x + 1`

## pythagorean_identity (trig)

- Input: `sin(x)^2 + cos(x)^2`
- Focus: `avoid_identity_as_magic_jump`
- Final result: `1`
- Step count: `1`
- Wire substep count: `1`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Aplicar la identidad pitagórica
   Before: sin(x)^2 + cos(x)^2
      → Sin²(u) y cos²(u) del mismo ángulo suman 1
        sin(x)^2 + cos(x)^2 → 1
   After: 1
```

### Wire / Web Steps

1. `Aplicar la identidad pitagórica`
   - before: `sin(x)^2 + cos(x)^2`
   - after: `1`
   - before_latex: `{\color{red}{{\sin(x)}^{2} + {\cos(x)}^{2}}}`
   - after_latex: `{\color{green}{1}}`
   - substeps:
     1. `Sin²(u) y cos²(u) del mismo ángulo suman 1`
        - before_latex: `{\sin(x)}^{2} + {\cos(x)}^{2}`
        - after_latex: `1`

## inverse_trig_identity (inverse_trig)

- Input: `atan(3) + (atan(1/3) - pi/2)`
- Focus: `show_known_identity_clearly`
- Final result: `0`
- Step count: `5`
- Wire substep count: `2`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Aplicar identidad de arctangentes
   Before: arctan(1/3) + arctan(3) - 1/2 ·  pi
      → Juntar la pareja que encaja con la identidad
        arctan(1/3) + arctan(3) - 1/2 ·  pi → arctan(1/3) + arctan(3)
      → Esa pareja vale pi/2
        arctan(1/3) + arctan(3) → pi/2
   After: pi/2 - 1/2 ·  pi
2. Cancelar una identidad exacta
   Before: pi/2 - 1/2 ·  pi
      → Las dos partes se compensan exactamente
        pi/2 - 1/2 ·  pi → 0
   After: 0
```

### Wire / Web Steps

1. `Aplicar identidad de arctangentes`
   - before: `arctan(1/3) + arctan(3) - 1/2 ·  pi`
   - after: `pi/2 - 1/2 ·  pi`
   - before_latex: `{\color{red}{\text{arctan}(3) + \text{arctan}(\frac{1}{3}) - \frac{1}{2}\cdot \pi}}`
   - after_latex: `{\color{green}{\frac{\pi}{2} - \frac{1}{2}\cdot \pi}}`
   - substeps:
     1. `Juntar la pareja que encaja con la identidad`
        - before_latex: `\arctan(\frac{1}{3}) + \arctan(3) - \frac{1}{2}\cdot \pi`
        - after_latex: `\arctan(\frac{1}{3}) + \arctan(3)`
     2. `Esa pareja vale pi/2`
        - before_latex: `\arctan(\frac{1}{3}) + \arctan(3)`
        - after_latex: `\frac{\pi}{2}`

## polynomial_expansion_cancel (polynomial)

- Input: `(a+b)^2 - a^2 - 2*a*b`
- Focus: `expansion_and_collection_should_be_readable`
- Final result: `b^2`
- Step count: `6`
- Wire substep count: `3`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Expandir el binomio
   Before: ((a + b))^2 - a^2 - (2 ·  a ·  b)
      → Aplicar la fórmula (A + B)^2 = A^2 + 2AB + B^2
        ((a + b))^2 → b^2 + 2 ·  a ·  b + a^2
   After: b^2 + 2 ·  a ·  b + a^2 - a^2 - (2 ·  a ·  b)
2. Cancelar términos opuestos
   Before: a^2 + b^2 + 2 ·  a ·  b - a^2 - (2 ·  a ·  b)
      → Estos dos términos se anulan entre sí
        a^2 - a^2 → 0
   After: b^2 + 2 ·  a ·  b - (2 ·  a ·  b)
3. Cancelar términos opuestos
   Before: b^2 + 2 ·  a ·  b - 2 ·  a ·  b
      → Estos dos términos se anulan entre sí
        2 ·  a ·  b - 2 ·  a ·  b → 0
   After: b^2
```

### Wire / Web Steps

1. `Expandir binomio`
   - before: `((a + b))^2 - a^2 - (2 ·  a ·  b)`
   - after: `b^2 + 2 ·  a ·  b + a^2 - a^2 - (2 ·  a ·  b)`
   - before_latex: `{\color{red}{{(a + b)}^{2}}} - {a}^{2} - (2\cdot a\cdot b)`
   - after_latex: `{\color{green}{{b}^{2} + 2\cdot a\cdot b + {a}^{2}}} - {a}^{2} - (2\cdot a\cdot b)`
   - substeps:
     1. `Aplicar la fórmula (A + B)^2 = A^2 + 2AB + B^2`
        - before_latex: `{(a + b)}^{2}`
        - after_latex: `{b}^{2} + 2\cdot a\cdot b + {a}^{2}`
2. `Agrupar términos semejantes`
   - before: `a^2 + b^2 + 2 ·  a ·  b - a^2 - (2 ·  a ·  b)`
   - after: `b^2 + 2 ·  a ·  b - (2 ·  a ·  b)`
   - before_latex: `{\color{red}{{b}^{2} + 2\cdot a\cdot b + {a}^{2} - {a}^{2}}} - (2\cdot a\cdot b)`
   - after_latex: `{\color{green}{{b}^{2} + 2\cdot a\cdot b}} - (2\cdot a\cdot b)`
   - substeps:
     1. `Estos dos términos se anulan entre sí`
        - before_latex: `{a}^{2} - {a}^{2}`
        - after_latex: `0`
3. `Agrupar términos semejantes`
   - before: `b^2 + 2 ·  a ·  b - 2 ·  a ·  b`
   - after: `b^2`
   - before_latex: `{\color{red}{{b}^{2} + 2\cdot a\cdot b - 2\cdot a\cdot b}}`
   - after_latex: `{\color{green}{{b}^{2}}}`
   - substeps:
     1. `Estos dos términos se anulan entre sí`
        - before_latex: `2\cdot a\cdot b - 2\cdot a\cdot b`
        - after_latex: `0`

## perfect_square_root (radicals)

- Input: `sqrt(x^2 + 2*x + 1)`
- Focus: `show_square_pattern_and_abs`
- Final result: `|x + 1|`
- Step count: `2`
- Wire substep count: `2`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Reconocer un cuadrado perfecto bajo la raíz
   Before: sqrt(x^2 + 2 ·  x + 1)
      → Reescribir el radicando como un cuadrado perfecto
        x^2 + 2 ·  x + 1 → (x + 1)^2
      → La raíz de un cuadrado da un valor absoluto
        sqrt((x + 1)^2) → |x + 1|
   After: |x + 1|
```

### Wire / Web Steps

1. `Reconocer un cuadrado perfecto bajo la raíz`
   - before: `sqrt(x^2 + 2 ·  x + 1)`
   - after: `|x + 1|`
   - before_latex: `{\color{red}{\sqrt{1 + {x}^{2} + 2\cdot x}}}`
   - after_latex: `{\color{green}{|1 + x|}}`
   - substeps:
     1. `Reescribir el radicando como un cuadrado perfecto`
        - before_latex: `{x}^{2} + 2\cdot x + 1`
        - after_latex: `{x + 1}^{2}`
     2. `La raíz de un cuadrado da un valor absoluto`
        - before_latex: `\sqrt{{x + 1}^{2}}`
        - after_latex: `|x + 1|`

## cube_quotient_radical (quotient)

- Input: `((sqrt(x))^3 - 1)/(sqrt(x) - 1)`
- Focus: `show_exact_quotient_reason`
- Final result: `x^(1/2) + x + 1`
- Step count: `4`
- Wire substep count: `5`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Reescribir potencia de una raíz
   Before: (sqrt(x)^3 - 1)/(sqrt(x) - 1)
      → Pasar la potencia al interior de la raíz
        sqrt(x)^3 → sqrt(x^3)
   After: (sqrt(x^3) - 1)/(sqrt(x) - 1)
2. Reconocer un cociente notable
   Before: (sqrt(x^3) - 1)/(sqrt(x) - 1)
      → Llamar t = sqrt(x) para reconocer la forma
        sqrt(x) → t
      → Ese cociente notable se convierte en t^2 + t + 1
        (t^3 - 1)/(t - 1) → t^2 + t + 1
      → Volver a poner t = sqrt(x)
        t^2 + t + 1 → sqrt(x) + sqrt(x)^2 + 1
   After: sqrt(x) + sqrt(x)^2 + 1
3. Deshacer una raíz con su potencia
   Before: sqrt(x) + sqrt(x)^2 + 1
      → El cuadrado deshace la raíz
        sqrt(x)^2 → x
      → Reemplazar ese bloque en la expresión
        sqrt(x) + sqrt(x)^2 + 1 → sqrt(x) + x + 1
   After: sqrt(x) + x + 1
   ℹ️ Requires: x > 0
```

### Wire / Web Steps

1. `Reconocer un cociente notable`
   - before: `(sqrt(x^3) - 1)/(sqrt(x) - 1)`
   - after: `sqrt(x) + sqrt(x)^2 + 1`
   - before_latex: `{\color{red}{\frac{\sqrt{{x}^{3}} - 1}{\sqrt{x} - 1}}}`
   - after_latex: `{\color{green}{1 + \sqrt{x} + {\sqrt{x}}^{2}}}`
   - substeps:
     1. `Llamar t = sqrt(x) para reconocer la forma`
        - before_latex: `\sqrt{x}`
        - after_latex: `t`
     2. `Ese cociente notable se convierte en t^2 + t + 1`
        - before_latex: `\frac{t^{3} - 1}{t - 1}`
        - after_latex: `t^{2} + t + 1`
     3. `Volver a poner t = sqrt(x)`
        - before_latex: `t^{2} + t + 1`
        - after_latex: `\sqrt{x} + {\sqrt{x}}^{2} + 1`
2. `Deshacer raíz y potencia`
   - before: `sqrt(x) + sqrt(x)^2 + 1`
   - after: `sqrt(x) + x + 1`
   - before_latex: `1 + \sqrt{x} + {\color{red}{{\sqrt{x}}^{2}}}`
   - after_latex: `1 + \sqrt{x} + {\color{green}{x}}`
   - substeps:
     1. `El cuadrado deshace la raíz`
        - before_latex: `{\sqrt{x}}^{2}`
        - after_latex: `x`
     2. `Reemplazar ese bloque en la expresión`
        - before_latex: `\sqrt{x} + {\sqrt{x}}^{2} + 1`
        - after_latex: `\sqrt{x} + x + 1`
