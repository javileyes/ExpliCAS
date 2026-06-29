# Auditoría de paso a paso + condiciones de dominio

Corpus: 266 expresiones (18 temas). Motor real-domain, modo generic. Workflow ultracode: 18 auditores + verificación adversarial por tema.

## Resumen ejecutivo

- **Soundness: 0 errores.** Ningún resultado incorrecto ni condición de dominio FALTANTE. Toda condición conservadora que el motor mantiene (`|x|`, `x≠1`, `x≥0`, `cos(x)≠0`, `x=0` para sqrt(x)·sqrt(-x), holes de cancelación) es genuinamente necesaria — los verificadores adversariales lo confirmaron caso por caso. El comportamiento conservador de los bloques 'anti-simplificación' (dejar `sqrt(x^2)-x` sin tocar, mantener `x≠1`) es CORRECTO.
- **46 hallazgos de RUIDO DIDÁCTICO** (0 high, 30 medium, 16 low), dominados por **25 condiciones de dominio tautológicas/redundantes** — justo lo que pediste revisar.

| Categoría | N |
|---|---|
| Condiciones de dominio tautológicas/redundantes | 25 |
| Steps mágicos (saltos sin substep) | 8 |
| Steps redundantes / no-op | 6 |
| Canonicalización ruidosa | 6 |
| Otros | 1 |
| Fugas de nombre de regla en inglés (hallazgo propio) | 6+ |

---
## 1. Condiciones de dominio tautológicas y redundantes (25) — PRIORIDAD

Son condiciones que SIEMPRE se cumplen (o están implicadas por otra ya listada) y deberían omitirse. Agrupadas por causa raíz:

### A. Cuadrática definida-positiva → `... ≠ 0` siempre cierto
Un denominador/factor cuadrático con discriminante negativo nunca se anula en ℝ; emitir `q ≠ 0` es ruido.

- `(x^4+1)/(x^2+sqrt(2)*x+1)  →  `x·√2 + x² + 1 ≠ 0` (Δ<0)`
- `(x^4+1)/(x^2-sqrt(2)*x+1)  →  `-x·√2 + x² + 1 ≠ 0` (Δ<0)`
- `(x^10-1)/(x^5-1)  →  `x⁴+x³+x²+x+1 ≠ 0` (sin raíces reales)`
- `((x^3-y^3)/(x-y))/(x^2+x*y+y^2)  →  `x²+xy+y² ≠ 0` (forma def-positiva)`
- `1/(x+1/(x+1))  →  `1/(x+1)+x ≠ 0` ⇒ (x²+x+1)/(x+1)≠0 (numerador def-positivo)`

### B. Denominador idénticamente constante → `... ≠ 0` siempre cierto (bloque 14)
El denominador `sqrt(x^2)-abs(x)+1` = |x|-|x|+1 = 1 idénticamente; `1 ≠ 0` es tautología. Aparece 4 veces.

- `(log(x^2)-2*log(abs(x)))/(sqrt(x^2)-abs(x)+1)`
- `(exp(log(x^2))-x^2)/(sqrt(x^2)-abs(x)+1)`
- `(sin(x)^2+cos(x)^2-1)/(sqrt(x^2)-abs(x)+1)`
- `((x^2-1)/(x-1)-x-1)/(sqrt(x^2)-abs(x)+1)  →  todos emiten `sqrt(x^2)-|x|+1 ≠ 0``

### C. Base potencia-2/3 (raíz cúbica de cuadrado) → `base ≥ 0` siempre cierto
El motor interpreta x^(2/3) = (∛x)² ≥ 0 incondicionalmente; `x^(2/3) ≥ 0` es ruido.

- `(x^(2/3))^(3/2)  →  `x^(2/3) ≥ 0``
- `((x^2-1)^(2/3))^(3/2)  →  `(x²-1)^(2/3) ≥ 0``
- `(x^(2/3))^(3/2)/abs(x)  →  `x^(2/3) ≥ 0``

### D. Radicando |x|±x o ≡0 → `... ≥ 0` siempre cierto + DUPLICADO (forma sqrt(x²) y forma |x|)
|x|+x ≥ 0 y |x|-x ≥ 0 son identidades; además se listan DOS veces (forma `sqrt(x^2)+x` y forma `|x|+x`).

- `sqrt(x+sqrt(x^2))  →  `sqrt(x^2)+x ≥ 0`, `|x|+x ≥ 0`  (ambas siempre ciertas y duplicadas)`
- `sqrt(sqrt(x^2)+x), sqrt(sqrt(x^2)-x)  →  análogo`
- `sqrt(x^2-sqrt(x^4))  →  `x²-sqrt(x⁴) ≥ 0` (radicando ≡ 0)`
- `sqrt((sqrt(x^2)-x)/(sqrt(x^2)+x)), sqrt((x-sqrt(x^2))/(x+sqrt(x^2)))  →  4 condiciones, 2 tautológicas + 2 duplicadas`

### E. Condición redundante implicada por otra ya listada
Una condición se deduce de otra del mismo conjunto.

- `((x^4-y^4)/(x^2-y^2))/(x^2+y^2)  →  `x²+y² ≠ 0` redundante (ya están x-y≠0 y x+y≠0)`
- `log(x/y)  →  `y ≠ 0` redundante (ya está x/y > 0)`
- `sqrt(x^2-1)/sqrt(x-1)  →  `x ≥ -1` redundante (ya está x > 1)  [hallazgo propio]`
- `(sqrt(x^2-1)-(x-1))/x  →  `x ≠ 0` redundante (ya está |x|≥1, que excluye 0)`

### F. Composición trig con dominio total → `cos(...) ≠ 0` siempre cierto
cos(atan(x)) > 0 para todo x real; `cos(atan(x)) ≠ 0` es ruido.

- `tan(atan(x))  →  `cos(atan(x)) ≠ 0``

**Fix recomendado (transversal):** antes de emitir una condición en `required_display`, pasarla por un *probador de tautología en ℝ*: (1) formas `q ≠ 0` con `q` cuadrática/polinómica de discriminante<0 o def-positiva; (2) `r ≥ 0` con `r` reducible a `|·|±·`, a una constante ≥0, o ≡0; (3) deduplicar la forma `sqrt(x^2)` contra `|x|`; (4) eliminar condiciones implicadas por otra ya presente (subsunción). Es el mismo *backstop domain-aware exacto* que ya usáis para keep/drop, aplicado al display.

---
## 2. Steps mágicos (8) — saltos sin substep que lo explique
- `sin(x)^6+3*sin(x)^4*cos(x)^2+3*sin(x)^2*cos(x)^4+cos(x)^6` — Step 1 'Pythagorean with Generic Coefficient' collapses 3·sin(x)^2·cos(x)^4 + 3·sin(x)^4·cos(x)^2 -> 3·sin(x)^2·cos(x)^2 in a single line with NO substep. This is a two-move jump (factor out 3·sin(x)^
- `sin(x)^4+2*sin(x)^2*cos(x)^2+cos(x)^4` — Step 1 'Quartic Pythagorean Identity' rewrites sin(x)^4 + cos(x)^4 -> 1 - 2·sin(x)^2·cos(x)^2 as an unexplained jump. It is a deliberate detour: it injects a -2·sin^2·cos^2 term so it cancels the +2·s
- `(x^4-y^4)/(x-y)` — Final result x^3+y^3+x·y^2+y·x^2 is CORRECT (verified: difference from x^3+x^2 y+x y^2+y^3 is 0). But the single step is a rabbit-out-of-a-hat: rule 'Cancelar factores en una fracción' shows before='(
- `(x^5-y^5)/(x-y)` — Result x^4+y^4+x·y^3+y·x^3+x^2·y^2 is CORRECT (verified difference = 0). Same defect as the x^4 case: the only step ('Cancelar factores en una fracción') shows before='(y^5 - x^5)/(y - x)' (unexplaine
- `((x+y)^3-(x-y)^3)/(2*y)` — Result y^2 + 3·x^2 is CORRECT (expand: (x+y)^3-(x-y)^3 = 6x^2 y + 2y^3, /2y = 3x^2 + y^2). But step 1 is mislabeled and garbled: rule is 'Cancelar términos opuestos' yet its green output is '{y}^{3} +
- `(x^2-y^2)/(x-y)` — Result x+y and condition x-y≠0 are both CORRECT. The defect is a garbled substep LaTeX: the substep 'Ahora se cancela el factor x - y' has before_latex='\frac{x - y \cdot x + y}{x - y}', which renders
- `((x+y)^4-(x-y)^4)/(8*x*y)` — Result x^2 + y^2 is CORRECT ((x+y)^4-(x-y)^4 = 8xy(x^2+y^2)) with correct conditions x≠0, y≠0. Step 1 is labeled 'Cancelar términos opuestos' but actually performs a full quartic-binomial expand and c
- `log(x/y)-log(x)+log(y)` — Collapses 'ln(y) + ln(x/y) - ln(x)' -> '0' in one step whose rule is the untranslated internal name 'Collapse Exact Zero Additive Subexpression', with NO substeps exposing the intermediate (the log-qu

---
## 3. Steps redundantes / no-op (6) — before == after
- `sqrt(abs(x))` — The single emitted step is a no-op: rule 'Reescribir la raíz como potencia fraccionaria' has before == after (before: 'sqrt(|x|)', after: 'sqrt(|x|)') and even its rule_latex is th
- `(log(x)-log(x))/(x-x)` — Result is correctly 'undefined' (0/0: numerator log(x)-log(x)=0, denominator x-x=0). The emitted condition 'x > 0' is itself mathematically correct (it is log(x)'s real domain), so
- `(sqrt(1+x)-1)/x` — The single emitted step, rule 'Reescribir la raíz como potencia fraccionaria' (rewrite the root as a fractional power), is a pure no-op: before = '(sqrt(x + 1) - 1)/x' and after =
- `((x+y)^2-(x-y)^2)/(4*x*y)` — Result 1 and conditions x≠0, y≠0 are CORRECT. Minor: step 1 rule 'Cancelar términos opuestos' shows a 'before' (display form) '(x^2 + y^2 + 2xy - (x^2 + y^2 - 2xy))/(4xy)' that doe
- `asin(sin(x))-x` — The single emitted step has rule 'Usar el nombre arctan' with before == after: before='arcsin(sin(x)) - x', after='arcsin(sin(x)) - x'. It is a pure no-op (the only change is the c
- `acos(cos(x))-x` — Same defect as the asin case: the lone step is rule 'Usar el nombre arctan' with before == after ('arccos(cos(x)) - x' -> 'arccos(cos(x)) - x'), a no-op whose only effect is the di

Patrón dominante: `Reescribir la raíz como potencia fraccionaria` y `Usar el nombre arctan` emitidos como pasos cuyo before==after (no cambian nada visible).

---
## 4. Canonicalización ruidosa (6)
- `sqrt(exp(2*x))` — The result is correct (e^x, no conditions) but the single step's `after` is "e^(2/2 · x)" with an unreduced coefficient 2/2. The step trace ends at e^(2/2·x) while the reported res
- `(exp(x)-1-x)/x^2` — The single step, rule 'Quitar paréntesis tras el signo menos' (remove parentheses after the minus sign), only reorders commutative addition terms: before '(e^x - 1 - x)/x^2' -> aft
- `(x)/(sqrt(1+x)-1)` — The domain condition is reported in unreduced form as 'sqrt(x + 1) - 1 ≠ 0' rather than the equivalent simplified 'x ≠ 0'. It is mathematically correct (it does exclude the removab
- `sqrt(x^2*y^2)/(x*y)` — Result |x·y|/(x·y) and conditions x≠0, y≠0 are CORRECT. Minor LaTeX bracketing defect in the substep 'Reescribir el radicando como un cuadrado perfecto': after_latex='{x\cdot y}^{2
- `sqrt(x^2*y^2)/(abs(x)*abs(y))` — Result 1 and conditions x≠0, y≠0 are CORRECT. Two cosmetic issues: (1) same '{x\cdot y}^{2}' substep latex renders as x·y^2 instead of (x·y)^2; (2) a heuristic 'cycle detected' dia
- `atan(tan(x))-x` — The lone step rule 'Usar el nombre arctan' has before == after ('arctan(tan(x)) - x' -> 'arctan(tan(x)) - x'); it only performs the cosmetic display rename atan->arctan (before_lat
- `(sqrt(x)-sqrt(x))/(sqrt(x)-sqrt(x))` — Result is correctly 'undefined' (this is 0/0: numerator and denominator are both literally sqrt(x)-sqrt(x)=0 for every x in the sqrt domain). But a domain condition 'x > 0' is stil

---
## 5. Fugas de nombre de regla en inglés (hallazgo propio, recurrente)

Nombres de regla VISIBLES sin traducir en la traza española (residual de las rondas previas):
- `Log Even Power (×2 en log(x^2)-2·log(x))`
- `Power of a Quotient (en 1+tan(x)^2)`
- `Trig Fourth Power Difference (en sin(x)^4-cos(x)^4)`
- `Pythagorean with Generic Coefficient / Quartic Pythagorean Identity (bloque 8)`
- `Collapse Shifted Quotient of Equivalent Expressions ((x^2+1)/(x^2+1))`
- `Collapse Exact Zero Additive Subexpression (log(x/y)-log(x)+log(y))`

**Fix:** añadir estos nombres a `rule_name_es_to_en` (o, si emergen del path de eventos/inglés, al mapeo visible).

---
## 6. Otros (cosméticos)
- **Paso `Usar el nombre arctan` MAL ETIQUETADO** para asin/acos: en `asin(sin(x))` el paso dice 'Usar el nombre arctan' aunque renombra arcsin (no arctan). Nombre de paso genérico/incorrecto.
- **Display en forma potencia vs raíz:** resultados como `(x+1)^(1/2)`, `(x·y)^(1/2)`, `(|x|+x)^(1/2)` en vez de `sqrt(...)` cuando la entrada usaba `sqrt`. Inconsistencia de notación entrada→salida.
- **Coeficiente sin reducir:** `sqrt(exp(2*x))` → paso con `e^(2/2·x)` (2/2 sin reducir a 1).
- **Condición sin reducir:** `(x)/(sqrt(1+x)-1)` → `sqrt(x+1)-1 ≠ 0` en vez de `x ≠ 0`.
- **Condición espuria en no-op:** `(sqrt(x)-sqrt(x))/(sqrt(x)-sqrt(x))` → `undefined` (0/0) pero emite `x>0`; y `log(x*y)` (no simplifica) emite `x·y>0`.

---
## Recomendación priorizada de ciclos
1. **[alto ROI] Probador de tautología/subsunción en `required_display`** — elimina ~25 ruidos de un golpe (secciones 1A–1F). El de mayor impacto y el que pediste.
2. **[medio] Traducir las fugas de regla en inglés** (sección 5) — barato, cierra un residual de las rondas previas.
3. **[medio] Suprimir los no-op `Reescribir la raíz como potencia fraccionaria` / `Usar el nombre arctan`** (sección 3) cuando before==after; y renombrar el paso arctan para asin/acos.
4. **[bajo] Pulido:** forma sqrt vs potencia en salida, reducir coeficientes (2/2), reducir forma de condiciones, no emitir dominio en no-ops.
5. **[bajo] Steps mágicos:** añadir substeps a las expansiones (x^n-y^n)/(x-y) y a las identidades pitagóricas de grado alto.