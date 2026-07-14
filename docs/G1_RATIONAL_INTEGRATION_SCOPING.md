# G1 — Integración racional universal: scoping en sub-ciclos acotados

- **Fecha:** 2026-07-14
- **HEAD:** `901cf595a`
- **Clase:** L (gatekeeper). Se entra como **secuencia de sub-ciclos**, nunca como un solo ciclo.
- **Método:** scoping workflow READ-ONLY (4 mapeadores + síntesis); todos los anclajes `file:line` verificados a mano contra el árbol.
- **Relacionado:** `docs/CALCULUS_ENGINE_DEVELOPMENT_PHASES.md` (G1, líneas ~58‑64 + criterio de salida #1), `docs/ENGINE_VS_SYMPY_ASSESSMENT_2026-07-14.md` (por qué G1 es el bloqueador top-leverage), `docs/GENERAL_INTEGRATION_BACKEND_ROADMAP.md`.

Cerrar G1 gradúa el **criterio de salida #1 de la Fase 1** y desbloquea formalmente la Fase 2.

---

## La frontera exacta (probes verificados)

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

### ☑ Sub-ciclo 1 — Cap. A: cuadrática racional con discriminante positivo → log-ratio real **[S] — HECHO** *(2026-07-14 `300f03ef38f4bebf55a26fad077eeeb6eee00fbb`)*
- **Graduado:** `1/(x^4-4)` (+ `1/(x^4-9)`, `1/((x^2-2)(x^2+1))`, `1/((x^2-2)(x^2-3))`, `x/(x^4-4)`). Verificado por diferenciar-atrás (√2 interno; √3 por finite-diff independiente). Δ<0 byte-idéntico; huella sólo +2 tests. Fix extra: guard `leading.is_negative()→None` (answer+narration) para el bug latente de narración de signo en denominadores leading-negativo (`4-x^4`) que además secuestraba la narración u-sub. Residuales honestos pendientes: leading-negativo (narración con signo), surd-impar self-verify (pliegue `√n·√n`).
- **Inserción:** `methods.rs:266` (rama `radius_square<0`, i.e. Δ>0) + `methods.rs:1033` (el brazo `u0` positivo-no-cuadrado que hoy hace `return None`).
- **Reuso:** `polynomial_square_minus_constant_log_antiderivative` (`symbolic_integration_support.rs:17289`), `build_numeric_radius_expr` (`:1362`), `rational_positive_square_root` (`:2514`); `SquarefreeFactor::Quadratic` sin cambios (ya guarda `x²−2` como `{linear_b:0, constant_c:−2}`); verificador diferenciar-atrás.
- **Net-new:** cuando `Δ=b²−4c>0`, emitir `(α/2)·ln|q| + (β−αb/2)/√Δ · ln|(2x+b−√Δ)/(2x+b+√Δ)|`; reemplazar el `return None` de `:1033` por `factors.push(Quadratic{...})` para el `+u0`.
- **Blast:** el assembler `:266` lo comparten Route A y B; la rama nueva está **estrictamente gateada a Δ>0**, así que todo fixture Δ<0 queda byte-idéntico (huella 0-delta). Verificar con stash-regenerate antes de commit.
- **Depende:** nada.
- **Retención:** `cargo test --workspace` verde; el verificador confirma `d/dx(F)=1/(x⁴−4)` simbólicamente (misma forma ya probada para `1/(x²−2)`); contadores de scorecard iguales en todos los probes previos; si la verificación falla → residual, nunca respuesta sin verificar.

### ☑ Sub-ciclo 2 — Cap. B: cuártica par irreducible como FACTOR → término surd conjugado **[M] — HECHO** *(2026-07-14 `5892e7660b2db09183557349ae1751ffdd91e0a0`)*
- **Graduado:** `1/(x^6+1)` (`x⁴−x²+1`, `a=√3`) y `1/(x^8-1)` (`x⁴+1`, `a=√2`) + variantes numerador-no-constante verificadas (`x^3/(x^4-x^2+1)`, `x/(x^6+1)`). Nuevo `SquarefreeFactor::EvenQuartic{p,r}` (factor entero sobre ℚ), bloque de 4 columnas en `mixed_partial_fraction_terms`, render `even_quartic_factor_antiderivative` por descenso de PARIDAD (ODD via u=x² racional-en-u; EVEN via split surd simétrico). Gate de propiedad: standalone constante lo conserva `symmetric_surd` (byte-idéntico). Ladder de factores repetidos declina EvenQuartic. Verificado: targets por diferenciar-atrás; variantes por finite-diff sympy. Δ<0 byte-idéntico; huella 0-delta suite-level. **Residual documentado:** numeradores generales sobre la cuártica par (`(x^3+5)/(x^6+1)`) producen render CORRECTO que el verificador √3-débil no pliega → declina honesto (residual de simplificador, no wrong-answer). `1/(x^8+1)` (resolvente u⁴+1 con surd) queda para más adelante.
- **Inserción:** `methods.rs:338` (añadir `SquarefreeFactor::EvenQuartic{p,q}` racional); `even_quartic_descent` (`:1053`, empujar `EvenQuartic` en vez del `return None` de `:1084`); `mixed_partial_fraction_terms` (`:1284`, bloque cofactor de 4 columnas); un builder generalizando `symmetric_surd_even_quartic_antiderivative` (`:1105`).
- **Reuso:** `symmetric_surd_even_quartic_antiderivative` como plantilla cerrada; `build_numeric_radius_expr`; `solve_rational_linear_system`; código even_substitution/resolvente de `split_even_residual`.
- **Net-new:** variante `EvenQuartic{p,q}`; contribución de 4 columnas al sistema lineal; builder que integra `(αx³+βx²+γx+δ)/(x⁴+px²+q)` partiendo en `(x²+ax+s)(x²−ax+s)`, `a=√(2s−p)` surd (ambas subcuadráticas Δ<0 → solo arctan).
- **Blast:** el brazo `EvenQuartic` solo dispara en inputs que hoy declinan; residue-solve sigue sobre ℚ. Guard: solo cuárticas pares irreducibles-sobre-ℚ (sin raíz racional) con resolvente Δ<0.
- **Depende:** nada.
- **Retención:** workspace verde; diferenciar-atrás confirma ambas antiderivadas; contadores previos iguales; si el split surd o la verificación falla → todo el factor devuelve `None` (residual honesto), nunca surd parcial/sin verificar.

### ☐ Sub-ciclo 3 — Cap. C: cuártica general (impar), raíz de resolvente irracional → par de cuadráticas algebraicas **[M]**
- **Gradúa:** `1/(x^5-1)` (Φ₅ = `x⁴+x³+x²+x+1` → dos cuadráticas conjugadas con `√5`).
- **Inserción:** `methods.rs:338` (`SquarefreeFactor::GeneralQuartic{c3,c2,c1,c0}` racional); `split_general_quartic` (`:932‑1003`, levantar el requisito de cuadrado-perfecto en `:974` para que `a=√(raíz resolvente)` pueda ser irracional, y rutear irreducibles a `GeneralQuartic`); bloque de 4 columnas de `mixed_partial_fraction_terms` (reusa el del ciclo 2); builder para `(cúbica)/(cuártica irreducible general)`.
- **Reuso:** admisión de 4 columnas del ciclo 2; depresión/recuperación de la cúbica resolvente ya en `split_general_quartic`; kernels de signo surd de `root_forms.rs` (`provable_sign_vs_zero`, `sign_of_linear_surd`) para keep/drop e irreducibilidad; `build_numeric_radius_expr`; builder arctan.
- **Net-new:** variante `GeneralQuartic`; llevar `a=√(raíz resolvente)` como radio algebraico cuando la raíz es racional no-cuadrada; recuperar el par asimétrico `(x²+ax+b)(x²−ax+c)` con `b,c` algebraicos; render de `(cúbica)/Φ₅` con coeficientes `√5`.
- **Blast:** solo cambia el brazo general-quartic que hoy declina; residue sobre ℚ (Φ₅ racional). El camino Δ<0-con-resolvente-cuadrado-perfecto racional debe quedar byte-idéntico (gate la rama nueva a raíz de resolvente no-cuadrada).
- **Depende:** **ciclo 2** (la admisión de 4 columnas en `mixed_partial_fraction_terms`).
- **Retención:** workspace verde; diferenciar-atrás confirma `d/dx=1/(x⁵−1)` con el átomo `√5`; contadores previos iguales; factorización reducible/fallida → residual.

### ☐ Sub-ciclo 4 — Cap. D: factor lineal con raíz cúbica (`x^3-2`) **[L]**
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
