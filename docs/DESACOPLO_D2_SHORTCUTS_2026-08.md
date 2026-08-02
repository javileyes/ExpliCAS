# D2 — Justificación medible de los shortcuts del orquestador (2026-08-02)

Peldaño D2-2 del plan de desacoplo: «los shortcuts de regresión se agrupan
por el invariante que protegen; cada grupo con el contrato que lo justifica;
el que quede sin justificación medible → cuarentena→certificación». Este doc
es el INVENTARIO de esa justificación y el procedimiento para completarla.

## El censo (en `d9dea0a88`)

**127 fns `*shortcut*`** en el directorio del orquestador, ya agrupadas
físicamente por familia (el troceo P1): trig_angles 23, zero_detection 20,
trig 19, fractions 13, general 13, radicals_powers 11, pairing 10,
support 10, hyperbolic 6, logs_exp 2.

## Las tres fuentes de justificación, medidas

1. **Contrato directo (test que lo nombra)**: 53/127 aparecen nombrados en
   los tests del orquestador (711 tests del troceo). Justificados.
2. **Actividad bajo corpus (profiler)**: `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1`
   sobre `run_simplify_zero_mixed_corpus` emite la tabla
   Attempts/Hits/Misses POR SECCIÓN (labels jerárquicos
   `rule.<regla>.try.<detector>`, no nombres de fn): ~10 secciones con hits
   reales en ese corpus (p.ej. `zero_scope_exact_trig` 675 hits,
   `two_term_hyperbolic_di…` 658, `zero_scope_exact_hyper…` 322). La
   justificación por-fn exige el MAPEO label↔fn (pendiente, abajo).
3. **Pipeline de contratos CLI**: los shortcuts se ejercitan sin ser
   nombrados vía las suites de contrato (cli/semantics/integrate) — la
   ausencia en (1) y (2) NO certifica muerte. Doctrina de la campaña:
   cuarentena→certificación, jamás borrado por 0-refs.

## Pendiente (L18 en `SANEAMIENTO_LEDGER.md`)

Los **74 sin mención directa en tests**: mapear cada uno a su label de
sección (los `record_orchestrator_shortcut_*` del cuerpo dicen el label),
correr el profiler sobre los DOS corpus de examples + una pasada de la suite
de contratos con la var activa, y clasificar: activo-en-corpus /
protegido-por-contrato-indirecto / candidato a cuarentena (con la
certificación completa de la campaña: 0 refs + capacidad viva por otra ruta
+ ningún frente vivo + procedencia).

## Hallazgo de PERF anotado (no abierto aquí)

El corpus destapa detectores con **cientos de miles de intentos y 0 hits**
(`zero_scope_sinh_cubic` 583k/0, `expand_triple_angle` 511k/0, familias
`two_term_*` 450-520k/0) a 0,4-2,4 µs por intento. En este corpus (sesgado a
cancelación de ceros) suman ~5-8 s de detección sin producto. Es candidato
del frente de PERF (gates por familia de forma antes del detector fino — el
patrón visited-sets/memos ya validado), NO de D2: se anota y no se toca.
