# D1a — Inventario del motor de cancelación (2026-08-02)

Peldaño (a) de la fase D1 de `PLAN_DESACOPLO_2026-08.md`, medido con el arnés
de D0 (`engine_decoupling_closure.py --per-entry`) sobre
`rules/arithmetic/` en `346d2c630`. Los 25 entries `define_rule!` están
**todos registrados** (verificado contra `pub fn register` — ninguna regla
muerta). El JSON del cierre por entry es regenerable con el comando de arriba.

## Los 25 entries caen en CUATRO estratos

| estrato | entries | cierre | papel en D1 |
|---|---|---:|---|
| **Núcleo** (mega-entries) | CollapseExactOneShiftedQuotient, CollapseExactZeroThreeTermSubset, CollapseExactZeroCommonScaledDifference, CollapseExactZeroProductFactor, DivZero | 595–635 c/u | SON el motor — su cierre es el fichero entero. No migran: son el destino del módulo `cancellation`. |
| **Disparadores familiares** | 12 reglas trig/hyperbolic/log/exponents/algebra (tabla abajo) | 3–119 | Los que D1c migra a «detectar y delegar». |
| **Par aditivo** | CancelExactAdditivePairs | 18 | Migrable, casi puro API. |
| **Triviales** | AddZero, MulOne, MulZero, CombineConstants, SubSelfToZero, AddInverse, SimplifyNumericExponents, NormalizeMulNeg | 0 | Sin dependencias — fuera del problema. |

## La API mínima que (casi) todos usan — 18 helpers

Alcanzados desde ≥12 de los 25 entries. Tres grupos semánticos — exactamente
el «candidato + veredicto + rewrite» que el plan pedía:

- **Veredicto de equivalencia-para-cancelación**: `exprs_match_for_cancellation`
  (+`_leaf`, `_uncached`), `exprs_equal_up_to_add_term_order`,
  `exprs_equal_up_to_mul_factor_order_and_sign`,
  `exprs_equal_up_to_add_term_multiset_for_cancellation`,
  `exprs_equal_up_to_same_denominator`, `exprs_match_after_default_simplify`.
- **Candidato/colección**: `collect_add_terms`, `collect_signed_mul_factors`,
  `signed_term_expr`, `normalize_signed_add_term`, `term_has_matrix_product_factor`.
- **Rewrite/entorno**: `build_signed_sum_expr`, `build_scaled_expr`,
  `run_default_simplify`, `ambient_pipeline_value_domain` (el eje de dominio
  YA está en la API de facto — conexión directa con D4).

⚠️ `collect_add_terms` (16/25) es además el helper con 14 variantes
divergentes en el workspace (L13): al formalizar la API, esta copia es la
canónica del motor y las demás siguen el protocolo renombrar-o-fusionar.

## Mapa de migración D1c — orden por arrastre ascendente

`∩API` = helpers de la API de facto; `excl` = solo suyos (viajan con él);
`arrastre` = compartidos con otros entries SIN ser API — el trabajo real de
cada migración (subir a API, bajar a exclusivo, o quedar interno del núcleo).

| # | disparador | cierre | ∩API | excl | arrastre |
|---|---|---:|---:|---:|---:|
| 1 | SubtractExpandedSumDiffCubesQuotient | 3 | 2 | 0 | ~~1~~ **0** ✔ D1c-1 |
| 2 | CancelExactAdditivePairs | 18 | 15 | 1 | ~~2~~ **0** ✔ D1c-2 |
| 3 | ExpandTrigSineProductTripleAngle | 28 | 18 | 6 | **4** |
| 4 | ExpandOddHalfPower | 14 | 2 | 8 | **4** |
| 5 | ExpandHyperbolicPythagoreanFactor | 33 | 15 | 12 | **6** |
| 6 | ExpandHyperbolicAngleSumDiff | 24 | 15 | 1 | **8** |
| 7 | ExpandLogAbsMulDiv | 29 | 17 | 0 | **12** |
| 8 | ExpandLogProductPower | 30 | 17 | 0 | **13** |
| 9 | ExpandTrigSquareIdentity | 28 | 15 | 0 | **13** |
| 10 | ExpandTrigSumToProduct | 34 | 18 | 0 | **16** |
| 11 | CollapseExactZeroTrigDoubleAngleCosVariant | 39 | 18 | 0 | **21** |
| 12 | ExpandTrigPhaseShift | 119 | 18 | 9 | **92** |

Arrastre total (unión entre los 12): **148 helpers distintos**. El éxito de
D1 re-cuantificado con la métrica canónica: cada disparador migrado importa
SOLO la API (18) + sus exclusivos; el arrastre de la tabla baja a 0 por
peldaño, y `pct_aristas_intra_fichero` del directorio (baseline 36,4%,
`decoupling_metrics_baseline.json`) sube. PhaseShift (92) es el último a
propósito: la mitad de su arrastre son builders `build_general_phase_shift_*`
que probablemente pidan su propio submódulo del núcleo, no la API.

## Estado de ejecución

- **D1b (2026-08-02, `cfa6430ca`)**: los 18 de la API consolidados
  físicamente en `support.rs` con frontera declarada en su cabecera.
- **D1c-1 (2026-08-02)**: disparador #1 migrado — su único arrastre
  (`canonicalize_nested_integer_powers`) PROMOVIDO a la API (19 helpers: es
  el canonicalizador-para-comparar del grupo veredicto). Cierre de Cubes
  ⊆ API; arrastre global 148 → 147.
- **D1c-2 (2026-08-02)**: disparador #2 migrado — sus dos arrastres
  (`additive_term_is_nonfinite_or_undefined`, 6 entries;
  `combine_additive_numeric_constants_for_cancellation`, 7) PROMOVIDOS
  (API: 21). El umbral ≥12 descubrió la API; la semántica la cierra: ambos
  son primitivas del motor con 6-7 consumidores. Arrastre global 147 → 145
  (DoubleAngleCos bajó 21 → 20 gratis).
- **Matización del criterio operativo**: los `define_rule!` viven en el
  CATÁLOGO del padre (diseño del troceo P2) y resuelven por su glob — para
  ellos el invariante «solo importa la API» se verifica con el CIERRE medido
  (`--per-entry`: arrastre 0), no con grep de imports. El grep aplicará a
  disparadores con fichero propio si D1c los saca del catálogo.
- Pendiente D1c: los 11 restantes por arrastre ascendente (2, 4, 4, 6, 8,
  12, 13, 13, 16, 21, 92). Cada peldaño decide por helper: promover a API /
  bajar a exclusivo / dejar interno del núcleo.
