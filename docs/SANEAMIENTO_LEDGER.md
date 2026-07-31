# Ledger de saneamiento

Sospechosos observados **mientras** se reestructura (protocolo de
`AUDITORIA_PARETO_ARQUITECTURA_2026-07-31.md`, §6): se anotan sin actuar, para
no abrir madrigueras a mitad de un movimiento. Las tandas de saneamiento
posteriores se alimentan de aquí, no de la memoria.

Estados: `abierto` · `en curso` · `cerrado` · `descartado`.

---

## Abiertos

### L1 — El repo no está limpio de `rustfmt` (preexistente)
- **Origen:** P5, troceo de `semantics_cli_contract_tests` (2026-07-31).
- **Qué:** dos hunks que `rustfmt --check` rechaza desde antes del troceo, en
  el test `bignum_available_mirrors_materialization_gates` y un vecino. Tras el
  troceo viven en `semantics_cli_contract_tests/misc_numeric.rs` y
  `misc_core.rs`, con el mismo contenido (verificado comparando el diff de
  rustfmt antes y después).
- **Por qué importa:** mientras `cargo fmt --check` esté rojo de base, no puede
  usarse como gate: cualquier suciedad nueva se camufla entre la vieja.
- **Acción propuesta:** un commit de formato aislado (`cargo fmt`) sobre esos
  hunks; después, `cargo fmt --check` sirve ya como gate binario.
- **Riesgo:** nulo (solo espacio en blanco), pero es cirugía → commit propio.

### L2 — Helpers de test duplicados por todo el workspace
- **Origen:** P5, barrido de `crates/*/tests` (2026-07-31).
- **Qué:** el mismo helper reescrito en muchos ficheros de test:
  `solve_display` ×22, `simplify_str` ×19, `solve` ×16, `label` ×9,
  `create_full_simplifier` ×9, `parse_wire` ×5, `run_cli` ×3.
- **Por qué importa:** es el mismo patrón de deriva silenciosa que
  `collect_add_terms` ×13 y `unary_builtin_arg` ×14 en `src`. Un arreglo en una
  copia no llega a las otras, y en helpers de test eso se traduce en suites que
  creen estar comprobando lo mismo y no lo comprueban.
- **Acción propuesta:** **primero diffear** las implementaciones (§4 del
  protocolo). Las idénticas → crate `cas_test_support` como `dev-dependency`.
  Las divergentes → cada divergencia es un hallazgo propio antes de fusionar.
- **Riesgo:** medio. No fusionar a ciegas.

### L3 — `metamorphic_simplification_tests.rs` es infraestructura, no casos
- **Origen:** P5 (2026-07-31), decidido NO trocear.
- **Qué:** 18.923 líneas con **139 tests y 316 helpers**. La proporción está
  invertida respecto a los otros monolitos: el contenido real es la maquinaria
  (carga de CSV, chequeo numérico, chequeo estructural, clasificadores).
- **Por qué importa:** el troceo mecánico por bloques `#[test]` aquí no aporta
  —dejaría un `main.rs` de 18k líneas—. Necesita mover maquinaria a módulos,
  que es cirugía.
- **Acción propuesta:** pasada propia que extraiga la infraestructura por
  responsabilidad, con tests verdes entre cada extracción.

### L4 — Rutas `--exact` obsoletas en la documentación
- **Origen:** P5 (2026-07-31).
- **Qué:** al pasar los tests a submódulos, sus rutas quedan cualificadas
  (`solving::test_eval_...`). Los repro con `--exact <nombre_desnudo>` de
  `SLOW_CI_TEST_LEDGER.md` ya no casan. Los `--test <binario>` sí siguen
  válidos: la convención `tests/<nombre>/main.rs` conserva el nombre del target.
- **Acción propuesta:** barrer los `--exact` de `docs/` y prefijarlos, o
  quitarles el `--exact` (el filtro por subcadena sigue funcionando).
- **Riesgo:** nulo, es documentación.

### L5 — La atribución de tiempo de CI puede estar inflada
- **Origen:** P5 (2026-07-31), medición incidental.
- **Qué:** `.claude/skills/auto-mejora/SKILL.md` da `cli_contract_tests` = 267 s
  y sostiene que cuatro suites se llevan 12 de los ~19 min. Medido aislado hoy:
  **57 s**. La cifra de 267 s salió de una corrida `--workspace` completa, es
  decir con contención de CPU entre binarios.
- **Por qué importa:** es exactamente el patrón ya registrado en memoria de
  «atribución heredada FALSA: medir, no heredar». Si el reparto real del tiempo
  es otro, el plan de acelerar CI está optimizando el objetivo equivocado.
- **Acción propuesta:** re-medir por binario en aislamiento antes de invertir en
  acelerar «las cuatro suites lentas».

---

## Cerrados

_(ninguno todavía)_

---

## Nota de método

Un aviso que costó un rato en P5: verificar un troceo comparando ficheros con
`f.endswith("main.rs")` **se salta `misc_domain.rs` en silencio** —
`"misc_domain.rs".endswith("main.rs")` es `True`—. El verificador daba 500 de
505 tests y parecía un fallo del troceo. Comparar el nombre completo
(`Path(f).name == "main.rs"`), no el sufijo. Es la misma familia de error que
los falsos verdes de `| tail` y `; echo OK` ya registrados: **cuando una
verificación acusa al código, sospecha primero de la verificación.**
