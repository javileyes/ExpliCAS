# Didactic Step Quality Audit Report

Generated from [didactic_step_quality_cases.csv](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/didactic_step_quality_cases.csv).

Command: `cargo test -p cas_didactic --test didactic_step_quality_audit didactic_step_quality_audit_generates_markdown_report -- --ignored --exact --nocapture`

## Summary

| id | category | steps | wire substeps | flags |
| --- | --- | ---: | ---: | --- |
| `rationalize_linear_root` | `rationalization` | 5 | 3 | none |
| `nested_fraction_one_over_sum` | `nested_fraction` | 2 | 2 | none |
| `nested_fraction_fraction_over_sum` | `nested_fraction` | 2 | 2 | none |
| `combine_like_terms_basic` | `combine` | 2 | 1 | none |
| `log_product_cancellation` | `logs` | 1 | 0 | no wire substeps emitted; single step with no didactic substeps |
| `same_denominator_fraction_focus` | `fractions` | 2 | 1 | none |
| `cancel_factors_fraction` | `cancellation` | 1 | 0 | none |
| `difference_of_squares_quotient` | `quotient` | 2 | 2 | none |
| `pythagorean_identity` | `trig` | 1 | 2 | none |
| `inverse_trig_identity` | `inverse_trig` | 6 | 1 | none |
| `polynomial_expansion_cancel` | `polynomial` | 3 | 2 | none |
| `perfect_square_root` | `radicals` | 1 | 2 | none |
| `cube_quotient_radical` | `quotient` | 1 | 3 | none |
| `geometric_product_cancellation` | `polynomial` | 1 | 3 | none |

## rationalize_linear_root (rationalization)

- Input: `1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)`
- Focus: `web_substeps_latex_and_human_explanation`
- Final result: `0`
- Step count: `5`
- Wire substep count: `3`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Multiplicar por el conjugado
   Before: 1/(sqrt(x) - 1) - (sqrt(x) + 1)/(x - 1)
   [Simplificación de fracción compleja]
      [Cambiar el signo para formar el conjugado]
        sqrt(x) - 1
        ->
        sqrt(x) + 1
      [Multiplicar numerador y denominador por ese conjugado]
        1/(sqrt(x) - 1)
        ->
        (sqrt(x) + 1)/((sqrt(x) - 1)  ·  (sqrt(x) + 1))
      [En el denominador aparece una diferencia de cuadrados]
        (sqrt(x) + 1)/((sqrt(x) - 1)  ·  (sqrt(x) + 1))
        ->
        (sqrt(x) + 1)/(x - 1^2)
   After: (sqrt(x) + 1)/(x - 1^2) - (sqrt(x) + 1)/(x - 1)
2. Calcular potencia numérica
   Before: (sqrt(x) + 1)/(x - 1^2) - (sqrt(x) + 1)/(x - 1)
   After: (sqrt(x) + 1)/(x - 1) - (sqrt(x) + 1)/(x - 1)
3. Cancelar fracciones iguales
   Before: (sqrt(x) + 1)/(x - 1) - (sqrt(x) + 1)/(x - 1)
   Cambio local: (sqrt(x) + 1)/(x - 1) - (sqrt(x) + 1)/(x - 1) -> 0
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
2. `Cancelar fracciones iguales`
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
   After: 1/((x + y)/(x · y))
2. Simplificar fracción anidada
   Before: 1/((x + y)/(x · y))
   [Simplificación de fracción compleja]
      [Invertir la fracción del denominador]
        1/((x + y)/(x ·  y))
        ->
        1 ·  (x ·  y)/(x + y)
      [Simplificar el producto resultante]
        1 ·  (x ·  y)/(x + y)
        ->
        (x ·  y)/(x + y)
   After: (x · y)/(x + y)
```

### Wire / Web Steps

1. `Sumar fracciones`
   - before: `1/(1/x + 1/y)`
   - after: `1/((x + y)/(x · y))`
   - before_latex: `\frac{1}{{\color{red}{\frac{1}{x} + \frac{1}{y}}}}`
   - after_latex: `\frac{1}{{\color{green}{\frac{x + y}{x\cdot y}}}}`
   - substeps: none
2. `Simplificar fracción anidada`
   - before: `1/((x + y)/(x · y))`
   - after: `(x · y)/(x + y)`
   - before_latex: `{\color{red}{\frac{1}{\frac{x + y}{x\cdot y}}}}`
   - after_latex: `{\color{green}{\frac{x\cdot y}{x + y}}}`
   - substeps:
     1. `Invertir la fracción del denominador`
        - before_latex: `\frac{1}{\frac{x + y}{x\cdot y}}`
        - after_latex: `1\cdot \frac{x\cdot y}{x + y}`
     2. `Simplificar el producto resultante`
        - before_latex: `1\cdot \frac{x\cdot y}{x + y}`
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
   After: (1/x)/((y + z)/(y · z))
2. Simplificar fracción anidada
   Before: (1/x)/((y + z)/(y · z))
   [Simplificación de fracción compleja]
      [Invertir la fracción del denominador]
        (1/x)/((y + z)/(y ·  z))
        ->
        (1/x) ·  (y ·  z)/(y + z)
      [Simplificar el producto resultante]
        (1/x) ·  (y ·  z)/(y + z)
        ->
        (y ·  z)/((y + z) ·  x)
   After: (y · z)/((y + z) · x)
```

### Wire / Web Steps

1. `Sumar fracciones`
   - before: `(1/x)/(1/y + 1/z)`
   - after: `1/(x · (y + z)/(y · z))`
   - before_latex: `\frac{\frac{1}{x}}{{\color{red}{\frac{1}{y} + \frac{1}{z}}}}`
   - after_latex: `\frac{\frac{1}{x}}{{\color{green}{\frac{y + z}{y\cdot z}}}}`
   - substeps: none
2. `Simplificar fracción anidada`
   - before: `1/(x · (y + z)/(y · z))`
   - after: `(y · z)/(x · (y + z))`
   - before_latex: `{\color{red}{\frac{\frac{1}{x}}{\frac{y + z}{y\cdot z}}}}`
   - after_latex: `{\color{green}{\frac{y\cdot z}{x\cdot (y + z)}}}`
   - substeps:
     1. `Invertir la fracción del denominador`
        - before_latex: `\frac{\frac{1}{x}}{\frac{y + z}{y\cdot z}}`
        - after_latex: `\left(\frac{1}{x}\right)\cdot \frac{y\cdot z}{y + z}`
     2. `Simplificar el producto resultante`
        - before_latex: `\left(\frac{1}{x}\right)\cdot \frac{y\cdot z}{y + z}`
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
1. Cancelar términos opuestos
   Before: 2 · x + 3 · x + 0
2. Agrupar términos semejantes
   Before: 2 · x + 3 · x
      [Sumar los coeficientes que acompañan a x]
        2 + 3
        ->
        5
   After: 5 · x
```

### Wire / Web Steps

1. `Agrupar términos semejantes`
   - before: `2 · x + 3 · x`
   - after: `5 · x`
   - before_latex: `{\color{red}{2\cdot x + 3\cdot x}}`
   - after_latex: `{\color{green}{5\cdot x}}`
   - substeps:
     1. `Sumar los coeficientes que acompañan a x`
        - before_latex: `2 + 3`
        - after_latex: `5`

## log_product_cancellation (logs)

- Input: `ln(x^3) + ln(y^2) - ln(x^3 * y^2)`
- Focus: `avoid_single_redundant_cancellation_substep`
- Final result: `0`
- Step count: `1`
- Wire substep count: `0`
- Flags: no wire substeps emitted; single step with no didactic substeps

### CLI Step By Step

```text
Steps:
1.
   Before: ln(x^3) + ln(y^2) - ln(x^3 · y^2)
   Cambio local: ln(x^3) + ln(y^2) - ln(x^3 · y^2) -> 0
   After: 0
```

### Wire / Web Steps

1. `Cancelar la subexpresión idénticamente nula`
   - before: `ln(x^3) + ln(y^2) - ln(x^3 · y^2)`
   - after: `0`
   - before_latex: `{\color{red}{\ln({x}^{3}) + \ln({y}^{2}) - \ln({x}^{3}\cdot {y}^{2})}}`
   - after_latex: `{\color{green}{0}}`
   - substeps: none

## same_denominator_fraction_focus (fractions)

- Input: `1 + a/d + b/d`
- Focus: `focus_only_on_fraction_part`
- Final result: `(a + b + d) / d`
- Step count: `2`
- Wire substep count: `1`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Llevar a denominador común
   Before: a/d + b/d + 1
   After: b/d + (a + d)/d
2. Sumar fracciones
   Before: b/d + (a + d)/d
   [Simplificación de fracción compleja]
      [Llevar a denominador común]
        b/d + (a + d)/d
        ->
        (d ·  (a + d) + b ·  d)/(d ·  d)
   After: (a + b + d)/d
```

### Wire / Web Steps

1. `Llevar a denominador común`
   - before: `a/d + b/d + 1`
   - after: `b/d + (a + d)/d`
   - before_latex: `{\color{red}{1 + \frac{a}{d}}} + \frac{b}{d}`
   - after_latex: `\frac{b}{d} + {\color{green}{\frac{a + d}{d}}}`
   - substeps: none
2. `Sumar fracciones`
   - before: `b/d + (a + d)/d`
   - after: `(a + b + d)/d`
   - before_latex: `{\color{red}{\frac{b}{d} + \frac{a + d}{d}}}`
   - after_latex: `{\color{green}{\frac{a + b + d}{d}}}`
   - substeps:
     1. `Llevar a denominador común`
        - before_latex: `\frac{b}{d} + \frac{a + d}{d}`
        - after_latex: `\frac{d\cdot (a + d) + b\cdot d}{d\cdot d}`

## cancel_factors_fraction (cancellation)

- Input: `(2*x)/(4*x)`
- Focus: `show_factorization_and_cancellation`
- Final result: `2 / 4`
- Step count: `1`
- Wire substep count: `0`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Cancelar factor común
   Before: (2 · x)/(4 · x)
   After: 2/4
```

### Wire / Web Steps

1. `Cancelar un factor común`
   - before: `(2 · x)/(4 · x)`
   - after: `2/4`
   - before_latex: `{\color{red}{\frac{2\cdot x}{4\cdot x}}}`
   - after_latex: `{\color{green}{\frac{2}{4}}}`
   - substeps: none

## difference_of_squares_quotient (quotient)

- Input: `(x^2 - 1)/(x - 1)`
- Focus: `factor_then_cancel_exactly`
- Final result: `x + 1`
- Step count: `2`
- Wire substep count: `2`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Factor: A² - B² = (A-B)(A+B)
   Before: (x^2 - 1)/(x - 1)
   Cambio local: x^2 - 1 -> (x + 1) * (x - 1)
   After: ((x + 1) · (x - 1))/(x - 1)
2. Cancelar factor común
   Before: ((x + 1) · (x - 1))/(x - 1)
   [Factorización de polinomios]
      [Usar la diferencia de cuadrados: a^2 - b^2 = (a - b)(a + b)]
        x^2 - 1^2
        ->
        (x + 1) ·  (x - 1)
      [Ahora se cancela el factor x - 1]
        (x - 1  ·  x + 1)/(x - 1)
        ->
        x + 1
   After: x + 1
```

### Wire / Web Steps

1. `Factorizar una diferencia de cuadrados`
   - before: `(x^2 - 1)/(x - 1)`
   - after: `((x + 1) · (x - 1))/(x - 1)`
   - before_latex: `\frac{{\color{red}{{x}^{2} - 1}}}{x - 1}`
   - after_latex: `\frac{{\color{green}{(1 + x)\cdot (x - 1)}}}{x - 1}`
   - substeps: none
2. `Factorizar una diferencia de cuadrados y cancelar`
   - before: `((x + 1) · (x - 1))/(x - 1)`
   - after: `x + 1`
   - before_latex: `{\color{red}{\frac{(1 + x)\cdot (x - 1)}{x - 1}}}`
   - after_latex: `{\color{green}{1 + x}}`
   - substeps:
     1. `Usar la diferencia de cuadrados: a^2 - b^2 = (a - b)(a + b)`
        - before_latex: `{x}^{2} - {1}^{2}`
        - after_latex: `(x + 1)\cdot (x - 1)`
     2. `Ahora se cancela el factor x - 1`
        - before_latex: `\frac{x - 1 \cdot x + 1}{x - 1}`
        - after_latex: `x + 1`

## pythagorean_identity (trig)

- Input: `sin(x)^2 + cos(x)^2`
- Focus: `avoid_identity_as_magic_jump`
- Final result: `1`
- Step count: `1`
- Wire substep count: `2`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Pythagorean Identity
   Before: sin(x)^2 + cos(x)^2
      [Reescribir cos(x)^2 como 1 - sin(x)^2]
        sin(x)^2 + cos(x)^2
        ->
        sin(x)^2 + 1 - sin(x)^2
      [Cancelar sin(x)^2 - sin(x)^2]
        sin(x)^2 + 1 - sin(x)^2
        ->
        1
   After: 1
```

### Wire / Web Steps

1. `Aplicar la identidad pitagórica`
   - before: `sin(x)^2 + cos(x)^2`
   - after: `1`
   - before_latex: `{\color{red}{{\sin(x)}^{2} + {\cos(x)}^{2}}}`
   - after_latex: `{\color{green}{1}}`
   - substeps:
     1. `Reescribir cos(x)^2 como 1 - sin(x)^2`
        - before_latex: `{\sin(x)}^{2} + {\cos(x)}^{2}`
        - after_latex: `{\sin(x)}^{2} + 1 - {\sin(x)}^{2}`
     2. `Cancelar sin(x)^2 - sin(x)^2`
        - before_latex: `{\sin(x)}^{2} + 1 - {\sin(x)}^{2}`
        - after_latex: `1`

## inverse_trig_identity (inverse_trig)

- Input: `atan(3) + (atan(1/3) - pi/2)`
- Focus: `show_known_identity_clearly`
- Final result: `0`
- Step count: `6`
- Wire substep count: `1`
- Flags: none

### CLI Step By Step

```text
Steps:
1. arctan(x) + arctan(1/x) = (π/2)·sign(x)
   Before: arctan(1/3) + arctan(3) - 1/2 · pi
      [Juntar la pareja que encaja con la identidad]
        arctan(1/3) + arctan(3) - 1/2 ·  pi
        ->
        arctan(1/3) + arctan(3)
   After: pi/2 · sign(1/3) - 1/2 · pi
2. sign(constant) = 1
   Before: pi/2 · sign(1/3) - 1/2 · pi
   Cambio local: sign(1/3) -> 1
   After: pi/2 - 1/2 · pi
3. Cancel exact additive pairs
   Before: pi/2 - 1/2 · pi
   Cambio local: pi/2 - 1/2 · pi -> 0
   After: 0
```

### Wire / Web Steps

1. `Aplicar identidad de arctangentes`
   - before: `arctan(1/3) + arctan(3) - 1/2 · pi`
   - after: `sign(1/3) · pi/2 - 1/2 · pi`
   - before_latex: `{\color{red}{\text{arctan}(\frac{1}{3}) + \text{arctan}(3)}} - \frac{1}{2}\cdot \pi`
   - after_latex: `{\color{green}{\text{sign}(\frac{1}{3})\cdot \frac{\pi}{2}}} - \frac{1}{2}\cdot \pi`
   - substeps:
     1. `Juntar la pareja que encaja con la identidad`
        - before_latex: `\arctan(\frac{1}{3}) + \arctan(3) - \frac{1}{2}\cdot \pi`
        - after_latex: `\arctan(\frac{1}{3}) + \arctan(3)`
2. `Evaluate Sign`
   - before: `sign(1/3) · pi/2 - 1/2 · pi`
   - after: `pi/2 - 1/2 · pi`
   - before_latex: `{\color{red}{\text{sign}(\frac{1}{3})\cdot \frac{\pi}{2}}} - \frac{1}{2}\cdot \pi`
   - after_latex: `\frac{\pi}{2} - \frac{1}{2}\cdot \pi`
   - substeps: none
3. `Cancelar fracciones opuestas`
   - before: `pi/2 - 1/2 · pi`
   - after: `0`
   - before_latex: `{\color{red}{\frac{\pi}{2} - \frac{1}{2}\cdot \pi}}`
   - after_latex: `{\color{green}{0}}`
   - substeps: none

## polynomial_expansion_cancel (polynomial)

- Input: `(a+b)^2 - a^2 - 2*a*b`
- Focus: `expansion_and_collection_should_be_readable`
- Final result: `b^2`
- Step count: `3`
- Wire substep count: `2`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Expandir el binomio
   Before: (a + b)^2 - a^2 - 2 · a · b
   After: b^2 + 2 · a · b + a^2 - a^2 - 2 · a · b
2. Cancel exact additive pairs
   Before: b^2 + 2 · a · b + a^2 - a^2 - 2 · a · b
      [Cancelar términos opuestos exactos]
        a^2 - a^2
        ->
        0
   After: b^2 + 2 · a · b - 2 · a · b
3. Cancel exact additive pairs
   Before: b^2 + 2 · a · b - 2 · a · b
      [Cancelar términos opuestos exactos]
        2 ·  a ·  b - 2 ·  a ·  b
        ->
        0
   After: b^2
```

### Wire / Web Steps

1. `Expandir binomio`
   - before: `(a + b)^2 - a^2 - 2 · a · b`
   - after: `a^2 + b^2 + 2 · a · b - a^2 - 2 · a · b`
   - before_latex: `{\color{red}{{(a + b)}^{2}}} - {a}^{2} - 2\cdot a\cdot b`
   - after_latex: `{\color{green}{{a}^{2} + {b}^{2} + 2\cdot a\cdot b}} - {a}^{2} - 2\cdot a\cdot b`
   - substeps: none
2. `Cancelar términos opuestos`
   - before: `a^2 + b^2 + 2 · a · b - a^2 - 2 · a · b`
   - after: `b^2`
   - before_latex: `{\color{red}{{a}^{2} + {b}^{2} + 2\cdot a\cdot b - {a}^{2}}} - 2\cdot a\cdot b`
   - after_latex: `{\color{green}{{b}^{2}}}`
   - substeps:
     1. `Cancelar términos opuestos exactos`
        - before_latex: `{a}^{2} - {a}^{2}`
        - after_latex: `0`
     2. `Cancelar términos opuestos exactos`
        - before_latex: `2\cdot a\cdot b - 2\cdot a\cdot b`
        - after_latex: `0`

## perfect_square_root (radicals)

- Input: `sqrt(x^2 + 2*x + 1)`
- Focus: `show_square_pattern_and_abs`
- Final result: `|x + 1|`
- Step count: `1`
- Wire substep count: `2`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Reconocer un cuadrado perfecto bajo la raíz
   Before: sqrt(x^2 + 2 · x + 1)
      [Reescribir el radicando como un cuadrado perfecto]
        x^2 + 2 ·  x + 1
        ->
        (x + 1)^2
      [La raíz de un cuadrado da un valor absoluto]
        sqrt((x + 1)^2)
        ->
        |x + 1|
   After: |x + 1|
```

### Wire / Web Steps

1. `Reconocer un cuadrado perfecto bajo la raíz`
   - before: `sqrt(x^2 + 2 · x + 1)`
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
- Final result: `sqrt(x) + 1 + sqrt(x)^2`
- Step count: `1`
- Wire substep count: `3`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Factor: a³ - b³ = (a-b)(a² + ab + b²)
   Before: (sqrt(x)^3 - 1)/(sqrt(x) - 1)
   [Factorización de polinomios]
      [Factorizar el numerador como suma o diferencia de cubos]
        sqrt(x)^3 - 1^3
        ->
        (sqrt(x) - 1) ·  (sqrt(x)^2 + sqrt(x) ·  1 + 1^2)
      [Ahora se cancela el factor (sqrt(x) - 1)]
        ((sqrt(x) - 1) ·  (sqrt(x)^2 + sqrt(x) ·  1 + 1^2))/(sqrt(x) - 1)
        ->
        sqrt(x) + 1 + sqrt(x)^2
      [Reemplazar ese bloque en la expresión]
        (sqrt(x)^3 - 1)/(sqrt(x) - 1)
        ->
        sqrt(x) + 1 + sqrt(x)^2
   After: sqrt(x) + 1 + sqrt(x)^2
```

### Wire / Web Steps

1. `Factorizar cubos y cancelar`
   - before: `(sqrt(x)^3 - 1)/(sqrt(x) - 1)`
   - after: `sqrt(x) + 1 + sqrt(x)^2`
   - before_latex: `{\color{red}{\frac{{\sqrt{x}}^{3} - 1}{\sqrt{x} - 1}}}`
   - after_latex: `{\color{green}{\sqrt{x} + 1 + {\sqrt{x}}^{2}}}`
   - substeps:
     1. `Factorizar el numerador como suma o diferencia de cubos`
        - before_latex: `\sqrt{x}^3 - 1^3`
        - after_latex: `\left(\sqrt{x} - 1\right)\cdot \left({\sqrt{x}}^{2} + \sqrt{x}\cdot 1 + {1}^{2}\right)`
     2. `Ahora se cancela el factor (sqrt(x) - 1)`
        - before_latex: `\frac{\left(\sqrt{x} - 1\right)\cdot \left({\sqrt{x}}^{2} + \sqrt{x}\cdot 1 + {1}^{2}\right)}{\left(\sqrt{x} - 1\right)}`
        - after_latex: `\sqrt{x} + 1 + {\sqrt{x}}^{2}`
     3. `Reemplazar ese bloque en la expresión`
        - before_latex: `\frac{{\sqrt{x}}^{3} - 1}{\sqrt{x} - 1}`
        - after_latex: `\sqrt{x} + 1 + {\sqrt{x}}^{2}`

## geometric_product_cancellation (polynomial)

- Input: `(x - 1)*(x^5 + x^4 + x^3 + x^2 + x + 1)`
- Focus: `show_small_full_expansion_then_group_and_cancel`
- Final result: `x^6 - 1`
- Step count: `1`
- Wire substep count: `3`
- Flags: none

### CLI Step By Step

```text
Steps:
1. Expandir y reagrupar un producto polinómico
   Before: (x - 1) · (x^5 + x^4 + x^3 + x^2 + x + 1)
      [Distribuir cada término del producto]
        (x - 1) ·  (x^5 + x^4 + x^3 + x^2 + x + 1)
        ->
        x^6 + x^5 + x^4 + x^3 + x^2 + x - x^5 - x^4 - x^3 - x^2 - x - 1
      [Agrupar los términos del mismo grado]
        x^6 + x^5 + x^4 + x^3 + x^2 + x - x^5 - x^4 - x^3 - x^2 - x - 1
        ->
        x^6 + (x^5 - x^5) + (x^4 - x^4) + (x^3 - x^3) + (x^2 - x^2) + (x - x) - 1
      [Los términos intermedios se cancelan por parejas]
        x^6 + (x^5 - x^5) + (x^4 - x^4) + (x^3 - x^3) + (x^2 - x^2) + (x - x) - 1
        ->
        x^6 - 1
   After: x^6 - 1
```

### Wire / Web Steps

1. `Expandir y reagrupar un producto polinómico`
   - before: `(x - 1) · (x^5 + x^4 + x^3 + x^2 + x + 1)`
   - after: `x^6 - 1`
   - before_latex: `{\color{red}{(x - 1)\cdot (1 + x + {x}^{2} + {x}^{3} + {x}^{4} + {x}^{5})}}`
   - after_latex: `{\color{green}{{x}^{6} - 1}}`
   - substeps:
     1. `Distribuir cada término del producto`
        - before_latex: `(x - 1)\cdot ({x}^{5} + {x}^{4} + {x}^{3} + {x}^{2} + x + 1)`
        - after_latex: `{x}^{6} + {x}^{5} + {x}^{4} + {x}^{3} + {x}^{2} + x - {x}^{5} - {x}^{4} - {x}^{3} - {x}^{2} - x - 1`
     2. `Agrupar los términos del mismo grado`
        - before_latex: `{x}^{6} + {x}^{5} + {x}^{4} + {x}^{3} + {x}^{2} + x - {x}^{5} - {x}^{4} - {x}^{3} - {x}^{2} - x - 1`
        - after_latex: `{x}^{6} + \left({x}^{5} - {x}^{5}\right) + \left({x}^{4} - {x}^{4}\right) + \left({x}^{3} - {x}^{3}\right) + \left({x}^{2} - {x}^{2}\right) + \left(x - x\right) - 1`
     3. `Los términos intermedios se cancelan por parejas`
        - before_latex: `{x}^{6} + \left({x}^{5} - {x}^{5}\right) + \left({x}^{4} - {x}^{4}\right) + \left({x}^{3} - {x}^{3}\right) + \left({x}^{2} - {x}^{2}\right) + \left(x - x\right) - 1`
        - after_latex: `{x}^{6} - 1`
