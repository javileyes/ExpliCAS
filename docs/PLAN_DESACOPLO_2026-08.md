# Plan de desacoplo real — 2026-08

Sucesor de `AUDITORIA_PARETO_ARQUITECTURA_2026-07-31.md` (campaña ejecutada:
troceo + cosecha). Aquella campaña compró navegabilidad y reparto de churn y
**demostró con medición que no compró desacoplamiento** (L6: el grafo del
orquestador es una bola; L8: arithmetic es un motor de cancelación cuyos
helpers no pueden viajar). Este plan va a por el desacoplo de verdad — que es
REDISEÑO — con tres objetivos simultáneos: calidad/mantenimiento,
direccionamiento a la universalidad, y caza de errores integrada.

## Principios (validados en la campaña; no re-litigar)

1. **Medir antes de mover.** Cada costura se elige con datos (grafo de
   llamadas, cierre transitivo, fan-in/out, imports reales). En 4 de 7 puntos
   de la campaña la medición corrigió al plan.
2. **Todo aquí es cirugía** ⟹ pasos PEQUEÑOS, cada uno con: suite completa,
   huella del scorecard, gate steps-on/off si roza simplificación,
   `cargo test --workspace --no-run` tras visibilidades, commit propio, sin
   push. Si un paso no puede demostrar equivalencia, se parte más pequeño o se
   difiere documentado.
3. **Caza de bugs por costura, no como fase aparte.** Al abrir cada costura:
   sondas adversariales (nombres internos `__*`, wrappers/formas canónicas
   duales, exponentes Neg, colisiones de temps), barrido diferencial, y el
   pipeline cuarentena→certificación→borrado para lo que aparezca muerto.
   P0 manda sobre el plan (doctrina vigente).
4. **La universalidad ordena la cola.** Ante dos costuras igual de sucias,
   primero la que bloquea escalar dominios/familias nuevas.

## Fases

### D0 — Baseline y arnés (1 sesión corta)
- Huella + scorecard baseline contra `2d4ad8c35` (pre-campaña) para anclar
  afirmaciones de perf; `cargo build --timings` como baseline de compilación.
- Formalizar las herramientas de medición de la campaña (callgraph, cierre
  transitivo, dup-diff, verify-move) como scripts versionados en `scripts/`
  con una línea de uso cada uno — hoy viven en scratchpad de sesión.
- Definir las métricas de desacoplo por costura: % aristas intra-módulo,
  fan-in de `support`, nº de `pub(crate)` transversales, imports cross-tema.

### D1 — El motor de cancelación se separa de sus disparadores
La costura más densa en deuda Y en bugs (la clase 7/3 vivía aquí al lado).
- Hoy: 25 reglas `*ToEnableCancellationRule` comparten ~500 helpers; trig
  necesita 151 y solo 15 son suyos.
- Objetivo: un núcleo `cancellation` con API estrecha (tipo «candidato a
  cancelación» + veredicto + rewrite) y disparadores por familia que SOLO
  detectan y delegan.
- Éxito medible: el cierre transitivo por tema pasa de «151 compartidos» a
  «solo la API»; los disparadores dejan de importar internals.
- Peldaños: (a) inventariar los 25 puntos de entrada y su intersección real
  de helpers; (b) extraer la API mínima que TODOS usan; (c) migrar
  disparador a disparador, cada uno con sus sondas.

### D2 — El orquestador: de bola a capas con interfaz
- `support/` (41 primitivas, emergió de la medición) se formaliza como API
  interna documentada; medir y estrechar la superficie de facto (548
  `pub(super)`).
- Los shortcuts de regresión (sedimento VIVO — 0 dead_code) se agrupan por el
  invariante que protegen; cada grupo con el contrato que lo justifica; el
  que quede sin justificación medible → cuarentena→certificación.
- Éxito medible: % de aristas intra-módulo sube de 34% por peldaños; ninguna
  pérdida en suite/huella.

### D3 — Partir CRATES (el desacoplo que compra compilación)
El troceo de ficheros no toca tiempos de cargo (la unidad es el crate);
cas_math sigue siendo 180k con cadena de 380k aguas abajo.
- Medir primero: imports reales entre familias de cas_math; `--timings`.
- Candidatos naturales: kernels de cálculo (integración+diferenciación+
  policies) y poly/multipoly como crates propios.
- Éxito medible: tocar integración deja de recompilar la cadena entera;
  segundos de `--timings` antes/después.
- Va DESPUÉS de D1/D2 a propósito: partir crates con las costuras sucias
  congela la suciedad en fronteras públicas.

### D4 — El eje de dominio como costura de primera clase (universalidad)
- Auditar las rutas de la cosecha (familia u-du completa) contra el eje
  complejo: gate `RealOnly→None` por defecto, validez declarada en UN sitio.
- Aplicar la lección «los PROBADORES internos heredan el eje» a los oráculos
  que las rutas nuevas usan (equivalencia, verificación por derivada).
- Esto alimenta directamente Fase 2+ (complejo) sin re-abrir cada ruta.

### D5 — Transversal: caza sistemática (no es fase, es modo)
En cada peldaño de D1-D4: sondas de la familia que destapó el 7/3 + el
barrido steps on/off + re-medición de la métrica de la costura. Hallazgos al
ledger (`SANEAMIENTO_LEDGER.md`); P0s se atienden antes de seguir.

## Orden y gobernanza

D0 → D1 → re-evaluar (D2 o D4 según lo que D1 enseñe) → D3 al final.
El proceso operativo es el de la skill `auto-mejora` (fuentes de verdad,
cadena de validación, lecciones); este doc solo aporta el QUÉ y el orden.
Melones explícitamente FUERA: C5 (oscilación expand↔factor — doctrina no
apresurar), series (decisión de usuario), y los 4 grandes supervivientes
(solve_outcome, calculus_residual_support, derive_command, derive/trig) salvo
que una costura de D1-D4 los toque.
