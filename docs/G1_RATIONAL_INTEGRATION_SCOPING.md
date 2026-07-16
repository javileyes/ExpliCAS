# G1 — Integración racional universal: scoping en sub-ciclos acotados

- **Fecha:** 2026-07-14
- **HEAD:** `901cf595a`
- **Clase:** L (gatekeeper). Se entra como **secuencia de sub-ciclos**, nunca como un solo ciclo.
- **Método:** scoping workflow READ-ONLY (4 mapeadores + síntesis); todos los anclajes `file:line` verificados a mano contra el árbol.
- **Relacionado:** `docs/CALCULUS_ENGINE_DEVELOPMENT_PHASES.md` (G1, líneas ~58‑64 + criterio de salida #1), `docs/ENGINE_VS_SYMPY_ASSESSMENT_2026-07-14.md` (por qué G1 es el bloqueador top-leverage), `docs/GENERAL_INTEGRATION_BACKEND_ROADMAP.md`.

Cerrar G1 gradúa el **criterio de salida #1 de la Fase 1** y desbloquea formalmente la Fase 2.

---

## La frontera exacta (probes verificados)

> **Nota (2026-07-15):** esta tabla es el snapshot del ARRANQUE del scoping. `1/(x^4-4)`, `1/(x^6+1)` y `1/(x^8-1)` ya NO son residual — se graduaron en Cap.A/B (ver los ☑ Sub-ciclos 1/2 abajo, con sus hashes). Siguen residual de verdad solo `1/(x^5-1)` (Cap.C) y `1/(x^3-2)` (Cap.D).

| Input | Estado | Punto de decline |
|---|---|---|
| `1/(x^6-1)` | ✅ funciona | factoriza sobre ℚ → lineales + cuadráticas irreducibles-racionales |
| `1/(x^2+x+1)` | ✅ funciona | cuadrática irreducible → arctan con √3 (el builder YA emite surds de racional) |
| `1/(x^4-4)` | ❌ residual | `split_even_residual` `methods.rs:1033`: raíz `u0=+2` positiva pero `√2` irracional → `return None` |
| `1/(x^6+1)` | ❌ residual | `even_quartic_descent` `methods.rs:1084`: `x⁴−x²+1` necesita `a=√3` irracional |
| `1/(x^8-1)` | ❌ residual | `x⁴+1` necesita `a=√2` (misma vía que x⁶+1) |
| `1/(x^5-1)` | ❌ residual | `split_general_quartic` `methods.rs:~1003`: Φ₅ necesita cuadráticas con `√5`, resolvente sin raíz cuadrado-perfecta |
| `1/(x^3-2)` | ❌ residual | `split_squarefree_factors` `methods.rs:918`: sobrante grado-3 no-par, necesita `∛2` |

**El único cuello de botella:** el denominador **solo se factoriza sobre ℚ** (`split_squarefree_factors`, `methods.rs:893` → `factor_rational_roots`, Rational Root Theorem). Los builders de log/arctan (`build_multi_quadratic_term_antiderivative`, `methods.rs:266`) **ya emiten coeficientes surd** (`√3`) para cuadráticas irreducibles-racionales; falta factorizar/renderizar cuando los factores llevan datos **irracionales**.

---

## Resumen de arquitectura (lo reusable)

Todo vive en `crates/cas_math/src/general_integration_backend/methods.rs`. El pipeline ya es un integrador factor-then-residue correcto sobre ℚ:

- `squarefree_split` (`:811`) — Yun-lite `gcd(D,D')`.
- `ostrogradsky_reduce` (`:828`) — Horowitz-Ostrogradsky para la parte racional. **Ya universal, sin factorizar.**
- `split_squarefree_factors` (`:893`) — **el techo ℚ**.
- `mixed_partial_fraction_terms` (`:1239`) — resuelve el sistema lineal exacto (BigRational) → `MultiQuadraticFactorTerm`.
- `build_multi_quadratic_term_antiderivative` (`:266`) — render; ya emite `√racional` para cuadráticas irreducibles-racionales.

**Clave que abarata todo:** el residue-solve **se queda sobre ℚ** incluso en los casos duros, porque la cuártica/cuadrática irreducible se mantiene como **un único factor de coeficientes racionales**; los números algebraicos aparecen **solo en el RENDER**. Templates ya funcionando para reusar: `polynomial_square_minus_constant_log_antiderivative` (`symbolic_integration_support.rs:17289`, log-ratio real de una cuadrática) y `symmetric_surd_even_quartic_antiderivative` (`methods.rs:1105`, salida con coeficientes surd). Verificador diferenciar-hacia-atrás (`verification*.rs`) garantiza que una factorización mala **degrada a residual**, no a respuesta incorrecta. `SquarefreeFactor` (`:338`) es BigRational-only → es el punto de extensión.

---

## Secuencia de sub-ciclos (cada uno = un ciclo /auto-mejora, un commit)

### ☑ Sub-ciclo 1 — Cap. A: cuadrática racional con discriminante positivo → log-ratio real **[S] — HECHO** *(2026-07-14 `d557556ea`)*
- **Graduado:** `1/(x^4-4)` (+ `1/(x^4-9)`, `1/((x^2-2)(x^2+1))`, `1/((x^2-2)(x^2-3))`, `x/(x^4-4)`). Verificado por diferenciar-atrás (√2 interno; √3 por finite-diff independiente). Δ<0 byte-idéntico; huella sólo +2 tests. Fix extra: guard `leading.is_negative()→None` (answer+narration) para el bug latente de narración de signo en denominadores leading-negativo (`4-x^4`) que además secuestraba la narración u-sub. Residuales honestos pendientes: leading-negativo (narración con signo), surd-impar self-verify (pliegue `√n·√n`).
- **Inserción:** `methods.rs:266` (rama `radius_square<0`, i.e. Δ>0) + `methods.rs:1033` (el brazo `u0` positivo-no-cuadrado que hoy hace `return None`).
- **Reuso:** `polynomial_square_minus_constant_log_antiderivative` (`symbolic_integration_support.rs:17289`), `build_numeric_radius_expr` (`:1362`), `rational_positive_square_root` (`:2514`); `SquarefreeFactor::Quadratic` sin cambios (ya guarda `x²−2` como `{linear_b:0, constant_c:−2}`); verificador diferenciar-atrás.
- **Net-new:** cuando `Δ=b²−4c>0`, emitir `(α/2)·ln|q| + (β−αb/2)/√Δ · ln|(2x+b−√Δ)/(2x+b+√Δ)|`; reemplazar el `return None` de `:1033` por `factors.push(Quadratic{...})` para el `+u0`.
- **Blast:** el assembler `:266` lo comparten Route A y B; la rama nueva está **estrictamente gateada a Δ>0**, así que todo fixture Δ<0 queda byte-idéntico (huella 0-delta). Verificar con stash-regenerate antes de commit.
- **Depende:** nada.
- **Retención:** `cargo test --workspace` verde; el verificador confirma `d/dx(F)=1/(x⁴−4)` simbólicamente (misma forma ya probada para `1/(x²−2)`); contadores de scorecard iguales en todos los probes previos; si la verificación falla → residual, nunca respuesta sin verificar.

### ☑ Sub-ciclo 2 — Cap. B: cuártica par irreducible como FACTOR → término surd conjugado **[M] — HECHO** *(2026-07-14 `6c4d59afc`)*
- **Graduado:** `1/(x^6+1)` (`x⁴−x²+1`, `a=√3`) y `1/(x^8-1)` (`x⁴+1`, `a=√2`) + variantes numerador-no-constante verificadas (`x^3/(x^4-x^2+1)`, `x/(x^6+1)`). Nuevo `SquarefreeFactor::EvenQuartic{p,r}` (factor entero sobre ℚ), bloque de 4 columnas en `mixed_partial_fraction_terms`, render `even_quartic_factor_antiderivative` por descenso de PARIDAD (ODD via u=x² racional-en-u; EVEN via split surd simétrico). Gate de propiedad: standalone constante lo conserva `symmetric_surd` (byte-idéntico). Ladder de factores repetidos declina EvenQuartic. Verificado: targets por diferenciar-atrás; variantes por finite-diff sympy. Δ<0 byte-idéntico; huella 0-delta suite-level. **Residual documentado:** numeradores generales sobre la cuártica par (`(x^3+5)/(x^6+1)`) producen render CORRECTO que el verificador √3-débil no pliega → declina honesto (residual de simplificador, no wrong-answer). `1/(x^8+1)` (resolvente u⁴+1 con surd) queda para más adelante.
- **Inserción:** `methods.rs:338` (añadir `SquarefreeFactor::EvenQuartic{p,q}` racional); `even_quartic_descent` (`:1053`, empujar `EvenQuartic` en vez del `return None` de `:1084`); `mixed_partial_fraction_terms` (`:1284`, bloque cofactor de 4 columnas); un builder generalizando `symmetric_surd_even_quartic_antiderivative` (`:1105`).
- **Reuso:** `symmetric_surd_even_quartic_antiderivative` como plantilla cerrada; `build_numeric_radius_expr`; `solve_rational_linear_system`; código even_substitution/resolvente de `split_even_residual`.
- **Net-new:** variante `EvenQuartic{p,q}`; contribución de 4 columnas al sistema lineal; builder que integra `(αx³+βx²+γx+δ)/(x⁴+px²+q)` partiendo en `(x²+ax+s)(x²−ax+s)`, `a=√(2s−p)` surd (ambas subcuadráticas Δ<0 → solo arctan).
- **Blast:** el brazo `EvenQuartic` solo dispara en inputs que hoy declinan; residue-solve sigue sobre ℚ. Guard: solo cuárticas pares irreducibles-sobre-ℚ (sin raíz racional) con resolvente Δ<0.
- **Depende:** nada.
- **Retención:** workspace verde; diferenciar-atrás confirma ambas antiderivadas; contadores previos iguales; si el split surd o la verificación falla → todo el factor devuelve `None` (residual honesto), nunca surd parcial/sin verificar.

### ⛔ Sub-ciclo 3 — Cap. C: cuártica general (impar), raíz de resolvente irracional → par de cuadráticas algebraicas **[M→L, BLOQUEADO]**
> **BLOQUEADO (scoping 2026-07-15).** Φ₅ es un par cuadrático ASIMÉTRICO con coeficientes en ℚ(√5) y radios arctan ANIDADOS `√((5∓√5)/2)`. Requiere dos capacidades ausentes de *nivel-2* (elemento ℚ(√n) ligero + torre de radicales anidados en el verificador). Plan de nivel-2 desglosado abajo en **"Nivel-2: el prerequisito ℚ(√n) + radical-anidado"**. Este sub-ciclo se convierte en el **C-iii** (cableado) de esa secuencia.
- **Gradúa:** `1/(x^5-1)` (Φ₅ = `x⁴+x³+x²+x+1` → dos cuadráticas conjugadas con `√5`).
- **Inserción:** `methods.rs:338` (`SquarefreeFactor::GeneralQuartic{c3,c2,c1,c0}` racional); `split_general_quartic` (`:932‑1003`, levantar el requisito de cuadrado-perfecto en `:974` para que `a=√(raíz resolvente)` pueda ser irracional, y rutear irreducibles a `GeneralQuartic`); bloque de 4 columnas de `mixed_partial_fraction_terms` (reusa el del ciclo 2); builder para `(cúbica)/(cuártica irreducible general)`.
- **Reuso:** admisión de 4 columnas del ciclo 2; depresión/recuperación de la cúbica resolvente ya en `split_general_quartic`; kernels de signo surd de `root_forms.rs` (`provable_sign_vs_zero`, `sign_of_linear_surd`) para keep/drop e irreducibilidad; `build_numeric_radius_expr`; builder arctan.
- **Net-new:** variante `GeneralQuartic`; llevar `a=√(raíz resolvente)` como radio algebraico cuando la raíz es racional no-cuadrada; recuperar el par asimétrico `(x²+ax+b)(x²−ax+c)` con `b,c` algebraicos; render de `(cúbica)/Φ₅` con coeficientes `√5`.
- **Blast:** solo cambia el brazo general-quartic que hoy declina; residue sobre ℚ (Φ₅ racional). El camino Δ<0-con-resolvente-cuadrado-perfecto racional debe quedar byte-idéntico (gate la rama nueva a raíz de resolvente no-cuadrada).
- **Depende:** **ciclo 2** (la admisión de 4 columnas en `mixed_partial_fraction_terms`).
- **Retención:** workspace verde; diferenciar-atrás confirma `d/dx=1/(x⁵−1)` con el átomo `√5`; contadores previos iguales; factorización reducible/fallida → residual.

### ☑ Sub-ciclo 4 — Cap. D: factor lineal con raíz cúbica (`x^3-2`) **[L] — HECHO** *(2026-07-15 `816ab8c1a`)*
- **Graduado:** `1/(x^3-2)` (+ `x/(x^3-2)`, `(x^2+1)/(x^3-2)`, `1/(x^3-5)`, compuesto `1/((x-1)(x^3-2))`) — EMITEN con condición de polo real `x≠∛k`, verificados por la torre degree-aware (`t³=k`, `s²=3` — radios PLANOS por construcción: `√3·c`, nunca `√(3c²)`) + numéricamente vs sympy (30 dígitos). Verificador: `radical_like_radicand` √/∛, reducer `t^e≥d→t^(e−d)·r`, nonneg SOLO grado-2 (raíz impar = sound incondicional). Render: triples exactos ℚ(c) `[1,c,c²]` con identidad de cierre chequeada; `Cbrt(k)` como nodo builtin (estanco a la canonicalización de Pow). Residuales honestos: `1/(x^3+2)` (k<0, peldaño — **cerrado en R1**, ver "Residuales post-G1" abajo), `1/(x^3-x-1)`. **CON ESTO LOS 5 PROBES DEL CRITERIO DE SALIDA #1 ESTÁN VERDES → G1 GRADÚA; y con los criterios #2 y #3 ya cumplidos, el umbral de la FASE 1 queda CRUZADO.**
- **Gradúa:** `1/(x^3-2)` (→ `(x−∛2)` + cuadrática irreducible `x²+∛2·x+∛4`).
- **Inserción:** `methods.rs:338` (representación lineal-∛ + cuadrática-∛); `split_squarefree_factors` (`:911‑918`, el brazo grado-3 `return None`, gate a `x³−k` con `rational_cbrt_exact`); `verification_algebraic.rs` `collect_variable_free_radicands` (extender de solo-√ a átomos ∛ con relación `t³=k` para `algebraic_rational_zero_test`).
- **Reuso:** `rational_cbrt_exact` (`root_forms.rs:2847`); builder log para el lineal real y arctan para la cuadrática; `reduce_by_relations` en `algebraic_rational_zero_test` (ya general una vez colectado el átomo/radicando ∛).
- **Net-new:** variante factor ∛ llevando `k` con `t=∛k`; coeficientes de residue en ℚ(∛2) (`1/(3∛4)` para el término lineal); el átomo ∛ + relación `t³=k` en el colector de radicandos del verificador.
- **Blast:** el más net-new y menor reuso: nueva clase de extensión grado-3 (∛) separada de los átomos √, más un cambio al `algebraic_rational_zero_test` compartido — **acotar** el cambio de colección a átomos ∛ aditivos para que la verificación √ existente quede intacta.
- **Depende:** nada (extensión independiente; **hacer al final** por ser lo más net-new y tocar el verificador).
- **Retención:** workspace verde; verificador extendido confirma `d/dx=1/(x³−2)` con el átomo `∛2`; regresión de todos los probes √ intactos; fallo de verificación → residual.

### ☐ Sub-ciclo 5 — Cierre universal (terminal opcional): parte logarítmica Lazard-Rioboo-Trager **[L]**
- **Gradúa:** los cinco uniforme + cualquier denominador numérico racional.
- **Inserción:** `crates/cas_math/src/polynomial.rs` (añadir resultante/subresultante sobre ℚ y ℚ[t] + aislamiento de raíces reales que devuelva raíces, no solo `count_real_roots` de `:435`); `split_squarefree_factors` (`:893`) reemplazado por un driver de parte-logarítmica LRT que alimenta los builders surd log/arctan existentes.
- **Reuso:** `squarefree_split`, `ostrogradsky_reduce` (parte racional ya universal), capa de render surd de A‑D, `count_real_roots` Sturm como oráculo, verificador diferenciar-atrás.
- **Net-new:** primitiva resultante/subresultante (**no existe en el repo**); aritmética de polinomios con coeficientes paramétricos en ℚ[t]; factorización squarefree de `R(t)` en `t` (cada bloque conjugado grado-2 → un arctan+log; cada bloque real grado-1 → un log real); aislamiento de raíces reales.
- **Blast:** el mayor; introduce maquinaria de resultantes + nuevo driver. **Correr como camino paralelo gateado tras la ruta de factorización existente hasta probarse**, no un rewrite.
- **Depende:** ciclos 2‑4 (construyen los targets de render surd/algebraico que LRT emite). **No adelantarlo**; es el reemplazo eventual, no un prerequisito.
- **Retención:** workspace verde; diferenciar-atrás confirma cada término; los cinco probes + un fuzz set de racionales aleatorios verifican; cualquier término no confirmable → residual.

---

## Orden recomendado y primer ciclo

**Ejecutar Cap. A primero (sub-ciclo 1)** — gradúa `1/(x^4-4)`. Es el incremento más pequeño y autocontenido: **enteramente sobre ℚ** (sin tipo de número algebraico), toca solo la rama de render `:266` (gateada a Δ>0 → todo fixture Δ<0 byte-idéntico) más un `push` en `split_even_residual` (`:1033`). Reusa la forma log-ratio real de una cuadrática ya verificada (`:17289`), así que el verificador diferenciar-atrás la confirma con certeza. Blast casi-cero, un commit, un probe nombrado graduado, y establece el render Δ>0 que los ciclos posteriores aprovechan sin necesitarlo como dependencia dura.

**Orden:** 1 (A) → 2 (B) → 3 (C, depende de 2) → 4 (D, independiente, al final por net-new) → 5 (LRT, terminal opcional). Los ciclos 1‑4 gradúan los 5 probes nombrados; el 5 los subsume y hace universal la integración racional para cualquier denominador numérico.

## Riesgos (trampas a evitar)

- **Soundness:** el render Δ>0 debe usar `ln|·|` (log-ratio real principal), nunca `arctanh` fuera de dominio; un desliz de signo en `(β−αb/2)/√Δ` solo lo caza diferenciar-atrás → mantener ese gate obligatorio (error → residual, no respuesta incorrecta).
- **Huella:** `build_multi_quadratic_term_antiderivative` (`:266`) lo comparten Route A y B; la rama nueva DEBE gatearse por signo de `radius_square` y **no reordenar/renombrar/re-foldar** la salida Δ<0 existente, o los bytes de cada fixture cuadrático previo se mueven (falso huella-delta). Verificar con stash-regenerate antes de commit.
- **Contrato de residual honesto:** en B/C/D una factorización surd/∛ parcial o sin verificar debe devolver `None` (residual), nunca emitir antiderivada sin chequear; la extensión del verificador (átomos surd en B/C, átomo ∛ en D) **es parte del coste de ese ciclo** — un término no confirmable es un FALSO DECLINE (seguro), un término emitido-pero-sin-verificar es la trampa.
- **Admisión de 4 columnas (B/C):** gatear la precondición de fracción propia (grado numerador < 4) y admitir solo cuárticas irreducibles-sobre-ℚ con el signo de discriminante-resolvente correcto; reusar los gates de irreducibilidad existentes, no inventar nuevos.
- **Declines intencionales pinneados** (`1/(2x²−3)`, `(x+1)/(x²−2)`, cli_contract_tests): son cuadráticas grado-2 bajo el MIN grado-3 de la ruta general → nunca entran a este pipeline; pero si algún ciclo voltea globalmente `push_quadratic_or_bail` (`:1209`) a admitir Δ>0, re-verificar que siguen declinados y que la ventana de grado los excluye.
- **No recursar al entrypoint** ni resetear cycle-guards desde el render nuevo (lección de hang previa); mantener cada builder local y no-recursivo.
- **Verificador ∛ (D)** edita el `algebraic_rational_zero_test` compartido — acotarlo a átomos ∛ aditivos con relación `t³=k` para que toda verificación √ existente quede confirmable byte-a-byte; regresión explícita de los probes √ antes de commit.
- **LRT (ciclo 5)** aterrizar como camino paralelo gateado tras la ruta existente hasta que verifique en los 5 probes + fuzz set; un big-bang de `split_squarefree_factors` arriesga mover greens en silencio.

---

## Cómo ejecutar

Cada sub-ciclo es un `/auto-mejora 1` (o encadenar `/auto-mejora N`). Marca aquí `☑` con el hash del commit al graduar. Los 5 probes nombrados son el criterio de salida #1 de la Fase 1; cuando `1/(x^5-1)`, `1/(x^6+1)`, `1/(x^8-1)`, `1/(x^4-4)`, `1/(x^3-2)` estén todos verdes (ciclos 1‑4) + verificados por diferenciar-atrás, G1 gradúa y la Fase 2 queda desbloqueada.

---

## Nivel-2: el prerequisito ℚ(√n) + radical-anidado (desbloquea Cap. C)

Scopeado 2026-07-15 vía workflow READ-ONLY (4 mappers convergentes; el sintetizador falló al schema → síntesis a mano). Anclajes verificados. Cap. C `1/(x^5-1)` está BLOQUEADO por dos capacidades ausentes que este nivel-2 provee. **Cap. D `1/(x^3-2)` NO se desbloquea aquí** (es ℚ(∛2), extensión grado-3, relación `t³=k`; QuadSurd no ayuda — sigue independiente y al final, como ya ordena el sub-ciclo 4).

### La matemática exacta de `1/(x^5-1)` (verificada)

- **PF externa sobre ℚ (reusa el bloque 4-columnas de Cap. B):** `1/(x^5-1) = 1/(5(x-1)) − (x³+2x²+3x+4)/(5·Φ₅)`, `Φ₅=x⁴+x³+x²+x+1`. Enteramente dentro de la maquinaria Cap. B ya verde.
- **PF interna sobre el par asimétrico** `Φ₅=(x²+φx+1)(x²+ψx+1)`, `φ=(1+√5)/2`, `ψ=(1−√5)/2`: `N/Φ₅ = (P₁x+Q₁)/(x²+φx+1) + (P₂x+Q₂)/(x²+ψx+1)` con **solo las pendientes en ℚ(√5)**: `P₁=−1/10−√5/10`, `P₂=conj(P₁)=−1/10+√5/10`, y **`Q₁=Q₂=−2/5` RACIONALES**.
- **Hallazgo clave que abarata (mapper #4):** calcular P₁,Q₁ necesita **SOLO álgebra lineal RACIONAL** vía el ansatz-conjugado (represent + mul + separación parte-racional/parte-√5 → 8 ecuaciones racionales, eliminación gaussiana sobre ℚ). **La división general en ℚ(√5) NO se necesita** para los coeficientes → el tipo de número surd es LIGERO (represent/add/sub/mul/neg/conj/split), no un QuadSurd completo con `/`.
- **Radios arctan ANIDADOS e irreducibles:** `√(4−φ²)=√((5−√5)/2)`, `√(4−ψ²)=√((5+√5)/2)` — no denestan a ningún `a+b√5` (ni el engine ni `sqrtdenest` de sympy). Se renderizan tal cual. Handles estructurales que el engine YA maneja: `(√((5−√5)/2))²→(5−√5)/2` (fold plano) y `d₁·d₂=√5` (el producto de radios conjugados es un surd PLANO).

### Los dos bloqueadores (por qué un render CORRECTO hoy DECLINA)

**B1 — El verificador no certifica el radical anidado.** `algebraic_rational_zero_test` (`verification_algebraic.rs:53`) YA es un reductor multi-relación: `reduce_by_relations` (`:385`) reduce módulo CADA `tᵢ²=radicandᵢ` — el motor de reducción YA soporta una TORRE. Lo que bloquea:
- `relation_polys` (`:264`, guard `:278‑283`) construye `multipoly_from_expr` sobre el radicando CRUDO `(5−√5)/2`, que contiene `sqrt(5)` → `BadExponent` → `None`; **y prohíbe activamente** radicandos con otro átomo (`degree_in(other_index) != 0 → None`).
- Phase A (`:76‑106`) sustituye `√r→t` solo en el residual, NO dentro de los exprs de radicando guardados en `relations`.
- `ALGEBRAIC_ZERO_TEST_MAX_RELATIONS = 2` (`:26`); Cap. C necesita ≥3 (s=√5, t₁, t₂).
- Sign-anchoring: {s²−5, t₁²−(5−s)/2, t₂²−(5+s)/2} solo fija `(t₁t₂)²=5`, dejando el signo de `t₁t₂=+√5` ambiguo — hay que anclarlo (radios conjugados ambos positivos, vía los kernels de signo surd existentes).

**B2 — El colapso del átomo φ (mapper #4).** `(1+√5)/2` auto-colapsa a un átomo golden-ratio `phi` (con `φ²→φ+1`), que NO reconcilia con la forma √5: `√(3−φ)−√((5−√5)/2)` no simplifica a 0 → el render DEBE construir todo coeficiente algebraico en forma explícita `√5/2+1/2`, NUNCA `(1+√5)/2`. Es un hazard sintáctico evitable (`1/2+√5/2` NO colapsa).

### Secuencia de nivel-2 (cada uno = un /auto-mejora, un commit)

#### ☑ C-i — Elemento ℚ(√n) ligero (`quad_surd.rs`) **[S] — HECHO** *(2026-07-15 `1f89e1b74`)*
- **Graduado:** módulo `crates/cas_math/src/quad_surd.rs` (standalone, wired-to-nothing, 7 tests). Tipo canónico con invariantes en construcción (`n≥0`; `surd==0⟹n==0`; radicando cuadrado-perfecto se pliega vía `perfect_square_support::rational_sqrt`); ops `add/sub/mul/neg/conj` + `from_expr` bridge + `to_expr` round-trip; mismatch de radicandos → `None` honesto. Tests: φ²=φ+1, φψ=−1, **coeficientes de Φ₅ = (1,1,1,1,1) vía aritmética QuadSurd** (la operación exacta de C-iii), fronteras adversariales. Huella byte-idéntica (cero callers). Workspace verde, clippy limpio.
- **Gradúa:** nada visible aún (unidad autocontenida, wired-to-nothing). Módulo nuevo `crates/cas_math/src/quad_surd.rs`.
- **Diseño:** tipo `{rat, surd, n: BigRational}` = struct-ificación del triple que `as_linear_surd` ya devuelve. Ops: represent, add, sub, mul `(ac+n·bd)+(ad+bc)√n`, neg, conj, is_rational, is_zero, split (parte-racional/parte-surd), to_expr; `from_expr` puenteando `as_linear_surd` (`root_forms.rs:1739`) — una sola ruta de parseo. **SIN división** (mapper #4: los coeficientes no la necesitan; añadirla sería código muerto que clippy `-D warnings` caza).
- **Tests:** axiomas de campo-ligero, `φ²=φ+1`, `φψ=−1`, `(x²+φx+1)(x²+ψx+1)=Φ₅` vía mul. Todo método ejercido por un test (evita el dead-code detector en crate de dominio — ver la lección `dead-code-detector-domain-vs-plumbing`).
- **Blast:** CERO (sin caller → huella byte-idéntica). Reusable más allá de Cap. C (cualquier familia con coeficientes-surd; consolida los kernels `(rat,surd,n)` dispersos de `root_forms.rs`).
- **Depende:** nada. **PRIMER CICLO si se continúa G1.**

#### ☑ C-ii — Torre de radicales anidados en el verificador **[M] — HECHO** *(2026-07-15 `e54372c2a`)*
- **Graduado:** `algebraic_rational_zero_test` confirma `√((5−√5)/2)² ≡ (5−√5)/2` y la forma real del término Cap. C (`d/dx[(1/d)arctan(x/d)] ≡ 1/(x²+(5−√5)/2)`, `d` anidado) vía torre `t₁²=(5−t₂)/2, t₂²=5`. Implementado: orden outer-first por node-count descendente (el DFS de colección NO garantiza orden — trampa cazada), relaciones con radicando SUSTITUIDO (ningún sqrt llega a multipoly), guard triangular (solo self-reference prohibida), nonneg surd exacta vía `provable_sign_vs_zero`, MAX_RELATIONS 2→3. `reduce_by_relations` SIN CAMBIOS (ya era motor de torre). Honestidad: radicando negativo declina; cross-product `t₁t₂=√5` queda None honesto (residual documentado: cada radio aparece a grados PARES en el residual real de Φ₅, así que C-iii no lo necesita). **Byte-compare explícito de los 9 probes Cap.A/B/residuales IDÉNTICOS** + workspace verde + huella 0-delta.
- **Gradúa:** capacidad de CONFIRMAR una antiderivada con radical anidado (probada por fixture, sin render aún).
- **Inserción:** `verification_algebraic.rs` — (a) ordenar radicandos inner-first por CONTENCIÓN ESTRUCTURAL (no heurística float — lección `soundness-gates-must-be-exact`); (b) sustituir cada átomo interno `√rⱼ→tⱼ` dentro de los exprs de radicando externos ANTES de `multipoly_from_expr` (así el externo se vuelve `(5−t₂)/2`, un multipoly genuino, y `sqrt(5)` nunca llega a la capa poly); (c) relajar el guard `:278‑283` de "libre de TODO átomo" a "triangular" (permite átomos estrictamente más profundos, prohíbe self/forward-reference → preserva terminación); (d) `MAX_RELATIONS` 2→3; (e) anclar el signo `t₁t₂=+√5` vía `sign_of_sum_two_surds`/`provable_sign_vs_zero` (`root_forms.rs:1919`/`:1876`); (f) nonnegatividad del radicando externo `(5−√5)/2≥0` vía sign-eval surd o condición emitida por el render.
- **Reuso:** `reduce_by_relations` (`:385`) NO cambia (ya maneja la torre); kernels de signo surd existentes.
- **Blast:** toca una decisión-procedure COMPARTIDA (superficie de soundness) → **regresión explícita de TODOS los probes √ de Cap. A/B byte-idéntica antes de commit**. `None`-en-lo-indecidible se preserva (nunca refutación falsa). `multipoly/conversion.rs:131‑150` (`BadExponent` en `Pow(_,1/2)`) queda intacto: es correctamente estricto; el fix quita el √ interno del radicando ANTES de esa capa, no la afloja.
- **Depende:** nada duro (independiente de C-i), pero su valor se realiza en C-iii.

#### ☑ C-iii — Cablear Cap. C (`1/(x^5-1)`) **[M] — HECHO** *(2026-07-15 `69a215bf6`)*
- **Graduado:** `1/(x^5-1)`, `1/(x^5+1)` (Φ₁₀), `x/(x^5-1)`, `(x^3+1)/(x^5-1)`, `1/Φ₅` standalone — EMITEN y el verificador C-ii los CONFIRMA (+ oráculo sympy independiente: residual 0 fuera del polo). `SquarefreeFactor::GeneralQuartic` (cuártica ENTERA sobre ℚ + raíz resolvente t₀ no-cuadrada), segundo pase en `split_general_quartic` (pase racional intacto → byte-identidad por construcción), split conjugado verificado con QuadSurd (C-i), PF interna por sistema 4×4 RACIONAL de identidades de traza (sin división ℚ(√t₀)), render con radios anidados `√(5/2∓√(5/4))` donde el coeficiente lleva UN factor del radio (residual PAR → torre C-ii). BONUS: RED heredado del sweep (wedged >14min desde Cap.B por Double-Angle-off vs surds √3) arreglado — suite 360/360 en 5.42s. **Con esto 4 de los 5 probes del criterio #1 están verdes; solo queda `1/(x^3-2)` (Cap. D).**
- **Gradúa:** `1/(x^5-1)` + la familia general del par cuadrático asimétrico.
- **Inserción:** `split_general_quartic` (`methods.rs:~932‑1003`, levantar la precondición de cuadrado-perfecto en `:974`) + bloque 4-columnas en `mixed_partial_fraction_terms` + `SquarefreeFactor::GeneralQuartic` (`methods.rs:338`).
- **Diseño:** PF externa por Cap. B; PF interna resolviendo P₁,Q₁ por eliminación gaussiana RACIONAL con ansatz-conjugado (usa C-i para el setup del producto); render φ-libre (`√5/2+1/2`, nunca `(1+√5)/2`) con radios `√((5∓√5)/2)`; **gateado tras el verificador C-ii** → si no confirma, DECLINA honesto (contrato residual).
- **Reuso:** builders arctan/log existentes; C-i (elemento surd); C-ii (verificador torre).
- **Depende:** **C-i + C-ii.**
- **Retención:** `1/(x^5-1)` verifica por diferenciar-atrás; workspace verde; huella +tests.

### Veredicto honesto (¿vale el prerequisito vs. diferir la cola G1?)

- **Coste:** 3 sub-ciclos acotados (C-i ~S zero-blast, C-ii ~M soundness-sensible, C-iii ~M) para graduar UN probe nombrado (`1/(x^5-1)`) + la familia par-asimétrico. Cap. D (`1/(x^3-2)`, ∛2) es un prerequisito SEPARADO (~2 ciclos más: verificador `t³=k` + ℚ(∛2)), no cubierto aquí. Cap. E (LRT) terminal.
- **Leverage:** C-i y C-ii son ampliamente REUSABLES (elemento ℚ(√n) general; confirmación de CUALQUIER antiderivada nested-radical) — no son gadgets de un solo probe. Pero `1/(x^5-1)` en sí es una cola estrecha: el par cuadrático asimétrico con coeficientes algebraicos es raro en un currículo real-univariable-elemental.
- **Contexto north-star:** G1 ya consumió 3 ciclos seguidos (Cap.A + Cap.B + budget-lift). La Fase 1 tiene DOS gatekeepers y G1 es solo uno; **el otro — narrativa educativa de límites (L'Hôpital iterado / límite notable / sándwich / jerarquía ∞/∞ / `e` / ∞−∞) — ya está MADURO** (graduado `bd64bfa3a`+`d600876e6`; criterio de salida #2 CUMPLIDO 2026-07-15), **NO a ~0%**. *(Esta línea decía originalmente "a ~0%" — creencia STALE falsificada al imprimir los SUBSTEPS del CLI, no el `rule` de nivel-1.)* Sus residuales son estrechos: VALOR de 0·∞/0^0 por política/dominio (bilátero de `x·ln x`, `x^x` — unilaterales YA resuelven), el algoritmo de valor general tipo Gruntz, y la NARRACIÓN del ∞−∞ común-denom en casos log/exp. Además hay wins P1 baratos (`diff(x,n)`/`diff(x,y)`, u-sub transcendente general, `taylor()`/`series()` + linealidad de sumatorios).
- **Recomendación:** el prerequisito es REAL y ACOTADO (3 ciclos limpios) y C-i/C-ii son reusables — NO es trabajo tirable. PERO por el propio guardrail de la skill ("alterna frentes cuando uno acumula varios ciclos seguidos") y dado que G1 ya lleva 3 ciclos consecutivos, la jugada de mayor ROI para el north-star es **alternar a un win P1 barato** (el gatekeeper de límites educativos ya está sustancialmente cerrado; solo quedan sus residuales estrechos), y volver a C-i→C-ii→C-iii después. Si se continúa G1 ahora, **C-i es el primer incremento correcto**: el más pequeño, zero-blast, reusable, y de-risquea el álgebra de coeficientes antes de tocar el verificador compartido o el pipeline de integración.
- **Primer ciclo si se continúa G1:** **C-i** (`quad_surd.rs` ligero, standalone, unit-tested, cero blast).

---

## Residuales post-G1 (rematados tras cruzar el umbral, 2026-07-15)

Con los 5 probes del criterio #1 verdes, quedaban tres residuales menores nombrados en Cap. D. Secuencia R1→R2→R3 (un ciclo cada uno):

### ☑ R1 — `1/(x^3+2)` (k<0 en CbrtCubic) **[S] — HECHO** *(2026-07-15 `0d2e0f35f`)*
- **Graduado:** `1/(x^3+2)`, `x/(x^3+2)` — el álgebra de Cap. D es SIGN-INVARIANTE: `c=∛k<0` existe, los triples ℚ(c) llevan el k con signo, `x²+cx+c²` es positivo-definido para todo c≠0, y `(2/w)arctan(u/w)` es PAR en `w=√3·c`. Fix: relajar los gates `!k.is_positive()` a `k.is_zero()` en `split_cubic_cbrt` + render. Verificado numéricamente (sympy 30 dígitos).

### ☑ R2 — `1/(x^4-5)` (cuártica par con resolvente de raíces REALES irracionales) **[M] — HECHO** *(2026-07-15 `4fd29a4ca`, estampado desde el ciclo R3)*
- **Graduado:** `1/(x^4-5)`, `1/(x^4-2)`, `(x^2+1)/(x^4-5)`, `(x^3+x)/(x^4-5)`, `1/(x^4+3x^2+1)`, `1/(x^4+6x^2+4)` (ambas raíces negativas → par arctan puro), y el compuesto `1/(x^6-2x^4-5x^2+10)`.
- **Diseño:** `SquarefreeFactor::EvenQuarticRealResolvent{p,r}` — Δ=p²−4r>0 NO-cuadrado (el fallthrough de `even_quartic_descent` tras fallar `√r` racional); factor ENTERO sobre ℚ; split conjugado `(x²−u₁)(x²−u₂)` con `u_i=(−p±√Δ)/2 ∈ ℚ(√Δ)` SOLO en el render. PF interna CERRADA en QuadSurd (sin división de campo: `1/√Δ=√Δ/Δ`, `1/u=conj(u)/r` con `norm(u)=u₁u₂=r`); identidades de traza racionales chequeadas (defensa en profundidad). Por factor: signo de `u` EXACTO vía `sign_of_linear_surd`; `u>0` → log-ratio real alrededor del radio ANIDADO `√u` + 2 polos reales (condiciones NonZero, pasa el gate ≤2); `u<0` → arctan alrededor de `√(−u)`. Cada coeficiente lleva UN factor de su radio → residual diferenciar-atrás PAR en cada átomo-radio → la torre C-ii (3 relaciones: `t²=Δ`, radios anidados) lo confirma. SIN cambios de verificador.
- **Re-pin:** `x^4-2` sale de `stay_residual`; `x^4-5` deja de ser el residual del test de numerador general; el pin simétrico-surd `x^4+3x^2+1` (que fijaba "residual" para esta forma EXACTA) gradúa con arctan sobre ℚ(√5). Byte-compare 16 probes (Φ₅, ∛, Cap.B, u-sub `x^3/(x^4-5)`, log-derivative, `x^4±4`, Φ₁₂) IDÉNTICOS.

### ☑ R3 — `1/(x^8+1)` (óctica doblemente-par) **[M] — HECHO** *(2026-07-15, hash en el ledger)*
- **Graduado:** `1/(x^8+1)` (¡la integral "monstruo"! — `(√(2+√2)/16)·ln-ratio + (√(2−√2)/16)·ln-ratio + (√(2+√2)/8)·par-arctan + (√(2−√2)/8)·par-arctan`, en ~0.9s), `2/(x^8+1)`, `1/(x^8+16)`, `1/(x^8-x^4+1)` (Φ₂₄), `1/(x^8+5x^4+16)`. **sympy 1.14 devuelve literalmente `0` para `1/(x^8+1)` y `1/(x^8+16)` — un wrong-answer VIVO; este render lo SUPERA** (los 4 verificados numéricamente a 30 dígitos).
- **Diseño:** probe standalone `doubly_even_octic_antiderivative` (patrón Cap.B symmetric_surd; 4º bloque del Err-arm — solo reclama lo que multi_quadratic, general_rational Y symmetric_surd declinaron); gates exactos S=√R, s=√S ∈ ℚ, A²=2S−P positivo NO-cuadrado, `2s>A` vía `sign_of_linear_surd`; PF nivel-1 CERRADA en ℚ(A) (γ=c/(2SA) como `cA/(2SA²)`, δ=c/(2S); identidades re-chequeadas exactas); nivel-2 = fórmulas EVEN de Cap.B con coeficientes QuadSurd, radios ANIDADOS `√(2s∓A)` (el D de una familia ES el a de la otra), recíprocos racionalizados por conjugado+norma racional `4s²−A²`. Cuadráticas positivo-definidas → SIN abs, SIN condiciones (emisión incondicional). Torre C-ii sin cambios estructurales.
- **Bumps medidos del verificador:** `REDUCTION_STEPS` 512→**16384** (medido REAL: hasta 8916 pasos en `x^8+5x^4+16` — la emulación del plan (~1950) subestimó 4.5×; headroom 1.8×) y `MAX_NODES` 500→1024. Sound: el step-cap solo acota trabajo antes de None honesto.
- **BONUS chokepoint (capa de signo):** el pipeline público adjuntaba condiciones `Positive` SIEMPRE-verdaderas a los `ln(x²±ax+s)` con `a` surd (el extractor racional de `prove_sign` no representa coeficientes surd). Extendido `is_strictly_positive/nonnegative_univariate_quadratic` con vía de INTERVALOS EXACTOS (`const_value_bounds`, BigRational outward-safe, jamás f64): cuadrática de coeficientes constantes-surd (anidados incl.) con `a_lo>0 ∧ disc_hi<0` = PRUEBA. Limpia retroactivamente las condiciones espurias de Cap.B/C-iii (`x^6+1`, `x^8-1`, `x^5±1`, Φ₁₂, Ostrogradsky) con RESULTADOS byte-idénticos y polos NonZero preservados; probes de solve intactos (bordes disc>0/disc=0 declinan honestos).
- **Re-pin:** `x^8+1` sale de `stay_residual` y del pin séxtico (reemplazado por `x^7-2`); rejects nuevos: `x^8+3x^4+1` (A²<0), `x^8+4` (s irracional), numeradores no-constantes (dueño u-sub `x^3/(x^8+1)` byte-idéntico).

**Nota de disciplina (hash-stamps):** los stamps de Cap.A/B/D y C-ii/C-iii apuntaban a los hashes PRE-amend (commits colgantes, no en main) — corregidos 2026-07-15 a los hashes reales (`d557556ea`, `6c4d59afc`, `816ab8c1a`, `e54372c2a`, `69a215bf6`). Regla nueva: nunca estampar el hash del PROPIO commit vía amend (el amend lo invalida); estampar el hash de un ciclo en el commit del ciclo SIGUIENTE, o citar "hash en el ledger".

---

## Cap. E — LRT universal (Lazard-Rioboo-Trager): scoping de sub-ciclos (2026-07-16)

Scopeado vía workflow READ-ONLY (5 mappers convergentes, 0 errores; el mapper de render-layer devolvió placeholder pero su contenido está cubierto por los otros cuatro). Informe crudo: `scratchpad/cap_e_scoping_journal.jsonl`. HEAD `6fc3f0cf9`.

### Estado de la maquinaria (qué existe, qué es NET-NEW)

**`crates/cas_math/src/polynomial.rs`** es un tipo DENSO UNIVARIADO sobre ℚ (`coeffs: Vec<BigRational>`). No hay parámetro `t`; los coeficientes no son polinomios → **ℚ[t][x] NO es representable** en `Polynomial`. `MultiPoly` (sparse) puede tener 2 variables y expone `degree_in`/`leading_coeff_in`, pero solo tiene `div_exact` (sin resto, sin pseudo-división).
- **Reusable:** `derivative` (`:403`), `div_rem` (`:274`, división exacta ℚ), `gcd` (`:305`, primitivo), `factor_rational_roots` (`:497`, SOLO raíces racionales), `eval` (`:415`), `content` (`:339`).
- **NET-NEW (no existe NADA en el repo, verificado con rg word-boundary):** RESULTANTE (`res_x`), SUBRESULTANT PRS (el primitivo núcleo de LRT), pseudo-división (prem/pquo), Yun squarefree multiplicidad-separado (el `squarefree_split` de `methods.rs:1196` es un solo `gcd(D,D')`), factorización-ℚ completa de `R(t)` más allá de raíces racionales, y un tipo de NÚMERO ALGEBRAICO general. `count_real_roots` (`:435`) devuelve SOLO un `usize`, nunca las raíces ni intervalos aislantes. Único álgebra de número algebraico existente: `QuadSurd` (grado 2) y los triples ℚ(∛k) de Cap.D — ningún RootOf/RootSum.

**Punto de inserción (anti-hijack por construcción):** el driver LRT entra como probe standalone GATEADO en el Err-arm de `try_rational_reciprocal_affine_probe` (`methods.rs:13`), DESPUÉS de `general_rational_partial_fraction_antiderivative` (`:40-56`) y junto a R2 (`:57`) / R3 (`:75`) — la ruta de factorización existente se intenta PRIMERO y solo devuelve None al topar el techo ℚ, así que todo input que hoy funciona nunca llega al driver → byte-identidad garantizada, CERO cambios a los 6 match sites del enum `SquarefreeFactor`.

**Techo público (constraint dura):** `public_rational_backend_conditions_allowed` (`cas_engine .../integration.rs:131`) cap-ea a **≤2** condiciones NonZero de polo en el límite público. Un output LRT con muchos polos reales queda silenciado salvo que se ensanche ese gate en el mismo cambio — es el techo de PUBLICACIÓN real, no `split_squarefree_factors`.

### La frontera real son DOS clases (probes verificados, HEAD, ≤0.5s, CERO hangs)

- **Track R — "graduación barata" (NO es LRT):** `1/(x^10-1)`, `1/(x^12-1)`, `x^8+x^4+1` siguen residual **aunque cada factor irreducible-ℚ ya emite standalone** (Φ5, Φ10, Φ12 emiten solos). El bloqueo es puramente factorización-ℚ incompleta: `factor_rational_roots` pela raíces racionales y deja el resto como UN blob reducible que el pipeline no parte. Reusa TODAS las plantillas A-D. GOTCHA: el normalizador EXPANDE el input pre-factorizado → la factorización DEBE correr dentro del path de integración, no asumirse de la forma superficial.
- **Track L — LRT genuino:** `1/(x^3-x-1)`, `1/(x^4+x+1)`, `1/(x^5-x-1)`, Φ9 (`x^9-1`), `1/(x^7-1)`, `x^7-2`, `x^6-2`, y el caso Rioboo `(x^4-3x^2+6)/(x^6-5x^4+5x^2+4)`. Numeradores NUNCA son el bloqueo (`(x^2+1)/(x^5-1)`, `x/(x^4+1)` emiten) — solo la forma del factor irreducible del denominador decide.

### El algoritmo (Rothstein-Trager / LRT / Rioboo)

Para `N/D` con `D` squarefree sobre ℚ, SIN factorizar `D`: forma `R(t) = res_x(N − t·D', D) ∈ ℚ[t]`; sus raíces `c_i` son los coeficientes logarítmicos y `v_i = gcd(N − c_i·D', D)` los argumentos. La mejora LRT calcula todos los `v_i` de UN subresultant PRS sobre ℚ[t] + squarefree de `R`, todo sobre ℚ hasta el render. **Forma REAL (Rioboo):** raíces reales → logs reales; cada par conjugado `c = a±bi` con `gcd = A(x)+iB(x)` → `a·ln(A²+B²) − 2b·LogToAtan(A,B)`, donde la recursión signed-remainder `LogToAtan` de Rioboo (Bronstein SI-I §2.8) da una suma CONTINUA de arctangentes (evita logs complejos y los saltos de `atan(B/A)`). Subsume A-D porque cada uno hard-codea una rebanada {#raíces reales, #pares, cuerpo-de-raíz} de `R(t)` alimentando los MISMOS dos builders.

**Techo de clausura radical (caveat de soundness):** LRT prueba que la primitiva elemental EXISTE, pero el RENDER necesita las raíces de cada factor-ℚ de `R(t)` como algebraicos concretos. Factores no-resolubles (`1/(x^5-x-1)` = Galois S₅ canónico; los factores deg-6 de `1/(x^7-1)` y `1/(x^6+x^3+1)`) **NO tienen forma cerrada con radicales** → E-ii DEBE gatearlos a residual honesto (o a una forma RootSum parametrizada si/ cuando se soporte), NUNCA forzar un surd incorrecto.

**Oráculo (sympy 1.14):** las 7 integrales objetivo son elementales; sympy las da bien EXCEPTO `1/(x^8+1)` → `0` (bug vivo, ya batido por R3). Varias las deja como `RootSum` sin evaluar (`1/(x^5-x-1)` es el caso elemental-pero-no-radical canónico).

### Secuencia de sub-ciclos (cada uno = un /auto-mejora, un commit)

- **☐ Track R [M] — factorización-ℚ completa del denominador squarefree** *(paralelo, más barato, NO LRT).* Parte el blob reducible de `factor_rational_roots` en sus factores irreducibles-ℚ → alimenta las plantillas A-D existentes. Gradúa `x^10-1`, `x^12-1`, `x^8+x^4+1` con render EXISTENTE. De-risquea el chokepoint. Correr DENTRO del path de integración (el normalizador expande). Se puede intercalar; es factorización-completeness, no maquinaria LRT.
- **☑ E-i [L] — HECHO 2026-07-16 `b612ba34a` — Subresultant PRS + resultante sobre ℚ[t].** Módulo `crates/cas_math/src/subresultant_prs.rs` (wired-to-nothing): `PolyTX`=ℚ[t][x] (reusa `Polynomial` como anillo ℚ[t] — el fork temido era trivial), `pseudo_remainder` fraction-free, `subresultant_prs` (cadena Brown/GCL), `rothstein_trager_resultant`/`_prs`. 11 resultantes RT pinneadas EXACTAS vs sympy 1.14 + cadena subresultante del cúbico. Lección: la resultante = último elemento de la PRS sin bookkeeping; el anillo ℚ[t][x] es `Vec<Polynomial>`. **ORIGINAL:** Subresultant PRS + resultante sobre ℚ[t]. Módulo standalone nuevo (p.ej. `crates/cas_math/src/subresultant_prs.rs`, wired-to-nothing como `quad_surd.rs`): (a) aritmética ℚ[t][x] (coeffs = polys univariados en `t`); (b) el subresultant PRS por la FÓRMULA de coeficientes subresultantes (NO pseudo-remainder naive — evita el blowup de coeficientes, el riesgo #1); (c) `R(t)=S₀` como byproduct; exponer la cadena `S₀..S_k` para que E-ii saque el subresultant de grado j. Unit-test cada entrada contra `resultant()`/`subresultants()` de sympy en los 7 denominadores oráculo (coeficientes exactos). **PRIMER CICLO — el único primitivo genuinamente duro y ampliamente reusable; aterriza en aislamiento, huella byte-idéntica, antes de tocar el integrador.** Es el análogo exacto de C-i.
- **☐ E-ii [M→L] — Driver Rothstein-Trager + render real de Rioboo.** Consume E-i. (1) `R(t)=res_x(N−t·D',D)` vía E-i; (2) squarefree-factor `R` sobre ℚ; (3) por factor squarefree `Q_j`, extraer `v_j` del PRS; (4) raíces REALES → log real (reusa `polynomial_square_minus_constant_log_antiderivative`, `symbolic_integration_support.rs:17289`); (5) `Q_j` grado-2 disc<0 → par conjugado → `a·ln(A²+B²) − 2b·LogToAtan(A,B)` (reusa builder arctan; coeficientes vía `quad_surd.rs`). NET-NEW dentro: (i) aislamiento de raíces reales que DEVUELVA raíces/intervalos (upgrade de `count_real_roots:435`, no reemplazo); (ii) la recursión `LogToAtan(A,B)` de Rioboo (arctan continua — el naive `atan(B/A)` es discontinuo donde `A=0`, riesgo #3). GATE: emite SOLO si cada `Q_j` es grado≤2 o sus raíces son surds representables; grado≥3 no-resoluble → residual honesto. Todo término pasa `algebraic_rational_zero_test` (`verification_algebraic.rs`) o el factor entero devuelve None.
- **☐ E-iii [M] — Wiring + gating.** Inserta el driver E-ii como path PARALELO gateado en el Err-arm (junto a R2/R3), corriendo SOLO en inputs que A-D/R2/R3 declinan; ensancha `public_rational_backend_conditions_allowed` si el conteo de polos reales lo exige; re-pins + byte-compare de todos los probes A-D/R1-R3.
- **☐ E-iv [L, terminal opcional] — Tipo RootSum/RootOf para factores no-resolubles.** La clausura VERDADERAMENTE universal: `1/(x^5-x-1)` y compañía. Grande, terminal, opcional (las familias curriculares están cubiertas sin esto).

**Orden recomendado:** Track R (win barato, intercalable) → **E-i (PRIMERO del track LRT)** → E-ii → E-iii → E-iv (opcional). Riesgos top: (1) blowup de coeficientes del PRS ℚ[t] si se usa pseudo-remainder naive → usar la fórmula subresultante; (2) techo de clausura radical → gatear no-resolubles a residual honesto, jamás surd incorrecto; (3) continuidad de Rioboo → `LogToAtan` obligatorio, no `atan(B/A)`.

### Cap. E-iv — RootSum (universalidad real): scoping de sub-ciclos (2026-07-16)

**Decisión del usuario (2026-07-16):** tras el análisis de R(t) (casi todos los targets tienen R(t) irreducible grado ≥3 → sin forma radical; E-ii radical solo alcanza Rioboo), se elige **E-iv RootSum** — la clausura universal de verdad.

**Insight que lo hace tractable Y un diferenciador sobre sympy:** la forma limpia universal es
`∫ N/D = RootSum(R(t), t ↦ t·log(x − w(t)))` con `R(t) = res_x(N − t·D', D)` (ya lo da **E-i**) y `w(t)` un polinomio en t de grado < deg R. Se calcula del subresultante de **grado-1-en-x** de la cadena de E-i (`S₁ = a(t)·x + b(t)`) como `w(t) = −b(t)·a(t)⁻¹ mod R(t)`. **VERIFICADO numéricamente** (err ~1e-165) y `w(t)` coincide EXACTO con sympy para `1/(x^3-x-1)`, `1/(x^5-x-1)`, `x/(x^4+x+1)`. Para `1/(x^3-x-1)` (grado 3) emitimos el RootSum LIMPIO mientras **sympy devuelve la basura de Cardano (casus irreducibilis, miles de chars)** → superamos a sympy, no solo igualamos.

**Alcance (acotado):** D squarefree, R(t) squarefree, argumento log LINEAL en x (el subresultante grado-1 es el gcd). Gradúa toda la clase "genuinamente irreducible": `1/(x^3-x-1)`, `1/(x^5-x-1)`, `1/(x^7-1)`, `x/(x^4+x+1)`, `1/(x^4+x+1)`… Casos con argumento no-lineal (raíces múltiples de D colapsando a un c) o R no-squarefree → residual honesto.

**Representación AST (bajo-blast):** el residual `integrate(...)` es un `Expr::Function` opaco con símbolo "integrate" (`symbolic_calculus_call_support.rs:29`) — RootSum sigue el MISMO patrón: `Function("RootSum", [R(t)_expr, summand_expr, t_var])` opaco a eval, renderizado especial. Sin variante nueva de `Expr`.

**Sub-ciclos:**
- **☑ E-iv-a [S] — HECHO 2026-07-16 `c4b8a5458` — inverso modular ℚ[t] + extracción de w(t).** Extiende `subresultant_prs.rs`: `modular_inverse(a, m)` (Euclides extendido en ℚ[t]) + `rothstein_trager_log_argument(N, D) -> (R(t), w(t))` que localiza el S₁ grado-1 en la cadena de E-i y calcula `w = −b·a⁻¹ mod R`. Unit-test contra el w(t) exacto de sympy (3 casos ya verificados). Wired-to-nothing, huella byte-idéntica. **PRIMER CICLO.**
- **☑ E-iv-b [S/M] — HECHO 2026-07-16 (hash en el ledger) — trazas de Newton + verificador EXACTO del RootSum.** *(Re-scopeado: los primitivos puros van antes que el nodo/render.)* `power_sums` (identidades de Newton — sumas de potencias de las raíces desde los coeficientes, sin extraer raíces), `trace_mod(h,R)=Σh(c)`, `rootsum_log_derivative_at(R,w,x₀)=Σc/(x₀−w(c))` = traza de `t·(x₀−w(t))⁻¹ mod R` (usa el inverso modular de E-iv-a), y `verify_rootsum_antiderivative(N,D,R,w)`: ambos lados son funciones racionales de grado acotado → igualdad exacta BigRational en 2·bound+1 puntos = **PRUEBA de identidad, no muestreo numérico**. Verifica los 5 casos oráculo (incl. S₅ y `1/(x^7-1)`) y REFUTA un w desplazado (adversarial). Sustituye al diferenciar-atrás surd (que no aplica a RootSum) como gate de emisión.
- **☐ E-iv-c [M] — nodo RootSum + driver + wiring + gating.** `Function("root_sum", …)` opaco (patrón residual-integrate); display `RootSum(R(t), t → t·ln(x − w(t)))`; driver ensambla vía E-iv-a; inserta en el Err-arm (junto a R2/R3, tras general_rational); gate: D+R squarefree, arg lineal en x, **emisión gateada por el verificador exacto de E-iv-b**; si no → residual honesto. Gradúa `1/(x^3-x-1)`, `1/(x^5-x-1)`, `1/(x^7-1)`, `x/(x^4+x+1)` etc.
