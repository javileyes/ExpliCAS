# Safe Feature Layering Pattern
**Capas opcionales, acotadas y auditables** para desarrollo seguro en el engine (CAS)

> Este documento formaliza el método de desarrollo que hemos usado con éxito en ExpliCAS (budgets anti‑explosión, `__hold`, unificación de JSON/errores, lints “hard fail” y utilidades canónicas).  
> El objetivo es **reducir riesgo** (loops, explosión combinatoria, regresiones semánticas) y **aumentar mantenibilidad** (single source of truth + contratos + pruebas).

---

## 1. Motivación

Un CAS/simplificador simbólico no es un programa “normal”: es un **sistema de reescritura**.  
Eso implica que:

- Un cambio local puede generar **efectos globales** (interacciones entre reglas).
- Es fácil introducir:
  - **explosión combinatoria** (expand/rationalize/poly),
  - **loops** (2‑cycles y oscilaciones),
  - **cambios de dominio** no deseados (p.ej. simplificar sin declarar supuestos),
  - **deuda invisible** (duplicación de helpers con semántica distinta),
  - **regresiones** sutiles (por orden de traversal, parent contexts, etc.).

Este método aborda todo eso con un patrón repetible.

---

## 2. Definiciones

### 2.1 Capa (Feature Layer)
Una “capa” es una funcionalidad nueva introducida con:

- **Opcionalidad**: se puede activar/desactivar de forma explícita.
- **Acotación**: vive en un perímetro claro con pocos puntos de entrada.
- **Auditabilidad**: tiene contrato, tests contractuales y enforcement (lint/CI).

### 2.2 Contrato
Texto normativo (POLICY/Docs) que define:

- Qué garantiza la capa.
- Qué *no* garantiza.
- Política conservadora (no “inventar” resultados).
- Cómo se expresa “no sé” (residual function, warning, `undefined`, error tipado).
- Stability: qué campos/outputs son estables.

### 2.3 Tests contractuales
Pruebas que validan **comportamiento estable** (interfaz) más que implementación interna:

- JSON schema estable.
- `kind`/`code` de errores estables.
- Invariantes de salida (p.ej. “no leak de `__hold`”).
- Presets y modos.

---

## 3. Los tres pilares del patrón

### A) Opcionalidad
La funcionalidad se añade de forma controlada:

- `--flag`, preset, modo, o `Options`.
- “Safe by default”: si el cambio altera semántica, debe ir tras un switch o ir acompañado de contratos/tests.

**Ejemplos reales:**
- Budgets: `preset_small/cli/unlimited`, `strict` vs `best-effort`.
- CLI: `--format`, `--budget`, alias ocultos legacy.
- Límites: residual `limit(expr, x, ±∞)` cuando no se puede probar.

---

### B) Acotación
La capa tiene un perímetro pequeño:

- **Un módulo canónico** (single source of truth).
- **Pocos puntos de entrada** (1–2) y flujo de datos simple.
- Evitar “doble parsing” o “lógica repartida”.

**Ejemplos reales:**
- `cas_ast::hold` (canónico) + transparencia en collectors.
- `cas_engine::budget` + `PassStats`.
- `cas_engine::json` canónico usado por CLI y FFI.
- `nary.rs` como canónico para flatten/add/mul.

---

### C) Auditabilidad
La capa es verificable y protegida contra regresión:

- Contrato (POLICY/docs).
- Tests contractuales (inputs/outputs, schema, invariantes).
- Lints (idealmente con escalado warning→hard fail).
- (Opcional) tracing/telemetría (p.ej. `PassStats`).

---

## 4. El template operativo (pasos del método)

Este es el “pipeline” recomendado para implementar cualquier mejora:

### 1) IDENTIFY (inventario + riesgo)
- Grep/auditoría para localizar duplicados, hotspots, semánticas divergentes.
- Clasificar riesgo: explosión, loop, dominio, UX, performance.

### 2) CANONICALIZE (una fuente de verdad)
- Crear o declarar el módulo canónico.
- Mover lógica ahí.
- Convertir copias a wrappers o eliminarlas.

### 3) CONTRACT (definir qué se garantiza)
- Política conservadora (“no inventar”).
- Qué se simplifica y qué no.
- Cómo se representa fallo/no determinable (residual, warning, error tipado).
- Versionado si es API externa (JSON schema).

### 4) GATE (pocos puntos de entrada)
- Introducir la capa a través de 1–2 funciones/entrypoints.
- Evitar “tocar 20 sitios” para activar algo.
- Preferir `Options`/`Preset` para configuración.

### 5) AUDIT (tests contractuales + lint/CI)
- Añadir tests contractuales.
- Añadir lints anti-regresión.
- Integrarlo en `make ci`.

### 6) MIGRATE (deuda a cero)
- Migración incremental (wrappers → eliminación).
- Lint como warning mientras migras.
- Cuando llegue a 0, convertir a HARD FAIL.

---

## 5. Herramientas “clave” que hacen que funcione

### 5.1 Presets y modos (DX + seguridad)
Los presets convierten configuraciones complejas en elecciones explícitas:

- Budgets: `small/cli/unlimited`.
- Limit/presimplify: `safe/off`.
- Output: `--format text/json`.

Esto hace el sistema reproducible y reduce “comportamiento sorpresa”.

### 5.2 Lints escalables
- Durante migración: warnings.
- Al completar migración: hard fail.
- Impiden reintroducir deuda.

Ejemplos ya implementados:
- no duplicar `strip_hold`, flatten/predicates/builders/traversal.
- no string-matching para step importance.

### 5.3 Contract tests como “garantía externa”
Especialmente importante cuando hay:
- JSON API, FFI, CLI.
- Semánticas sensibles (budgets, `undefined`, `infinity`).
- Invariantes (no `__hold` en salida).

---

## 6. Caso práctico: patrones ya demostrados en ExpliCAS

### 6.1 `__hold` (utilidad peligrosa pero útil)
**Problema:** wrapper útil para evitar re-traversal/explosión, pero generaba bugs y leaks.  
**Solución:** contrato + módulo canónico + collectors transparentes + tests.

- Canonical: `cas_ast::hold`
- Enforced: lint + eliminar duplicados
- Audit: contract (“no `__hold` en JSON output”, etc.)

### 6.2 Budgets anti-explosión
**Problema:** budgets fragmentados, agujeros, mediciones inconsistentes.  
**Solución:** sistema unificado con `Operation/Metric`, `PassStats`, presets, contract tests y lint enforcement.

- Layer A: `NodesCreated` (centralizado)
- Layer B: charges en hotspots
- Layer C: pre-estimación (fail fast)

### 6.3 JSON API + errores tipados
**Problema:** errores sin `kind/code`, difícil routing en UI, parse sin span.  
**Solución:** `CasError.kind/code`, spans canónicos, schema v1, contract tests y wiring CLI/FFI.

---

## 7. Checklist rápido (para PRs futuros)

**Antes de abrir PR:**
- [ ] ¿Hay módulo canónico?
- [ ] ¿Hay punto(s) de entrada bien definidos (≤2)?
- [ ] ¿Existe contrato escrito en docs/POLICY?
- [ ] ¿Se define política conservadora?
- [ ] ¿Hay tests contractuales (no solo unit tests)?
- [ ] ¿Hay lint/enforcement para evitar regresiones?
- [ ] ¿Hay modo/preset para activar/desactivar si procede?
- [ ] ¿Se define output estable (JSON) si aplica?

---

## 8. Hotspots futuros (lista priorizada)

> “Hotspot” = zona del engine donde es **fácil romper semántica** o donde el **riesgo de explosión/loop** es alto.  
> Para cada hotspot, se recomienda tratarlo como una capa: *contrato + módulo canónico + entrypoint + tests contractuales + lint*.

### H1) Sustitución avanzada por patrones (Pattern‑matching substitution)
**Motivo:** sustituciones tipo `target = x^2` en `x^4` requieren reconocer estructura algebraica, no textual.  
**Riesgos:**
- Cambios de dominio al reescribir potencias/raíces.
- Loops (expand ↔ factor ↔ power combine).
**Capa recomendada:**
- Módulo `substitute/` con modo `safe` vs `aggressive`.
- Contrato: cuándo se permite `x^4 -> (x^2)^2`, restricciones (exponentes enteros, sin raíces salvo reglas explícitas).
- Tests contractuales: `substitute(x^4 + x^2 + 1, x^2 -> y) = y^2 + y + 1`, más casos negativos.

### H2) Simplificación de potencias y reglas de “canonical form”
**Motivo:** oscilaciones del tipo `a^n*b^n ↔ (ab)^n`, `(-1)^n`, etc.  
**Riesgos:** ciclos y divergencias por reglas duales.
**Capa recomendada:**
- Contrato de orientación: “forma canónica preferida” (p.ej. num^n * sym^n).
- Rules declarativas con `importance` y guards.
- Tests contractuales de anti‑oscilación.

### H3) Racionalización generalizada (binomios, n‑ésimas raíces, productos)
**Motivo:** es una fuente clásica de explosión y errores de dominio.  
**Riesgos:** generar expresiones enormes, introducir `x^(-1/2)` indeseado, o suposiciones implícitas.
**Capa recomendada:**
- Reglas con pre‑estimación + budgets.
- Contrato: qué patrones se soportan (binomio surd, cubos, etc.) y cuándo devuelve residual.
- Tests contractuales de racionalización (incluyendo negativos y simetrías).

### H4) Canonicalización de resta global (NormalizeBinomialOrderRule)
**Motivo:** es útil, pero interactúa con trig/identidades y puede provocar loops/stack overflow.  
**Riesgos:** grandes loops con muchos estados intermedios.
**Capa recomendada:**
- Gate por fase (p.ej. “Canonicalization phase” separada, no en CORE).
- Detector de loop por “energía/medida monótona” o por hashes con ventana.
- Tests contractuales sobre trig suites (asin/acos/atan).

### H5) Evaluación de límites a puntos finitos (V2+)
**Motivo:** requiere manipulación de discontinuidades, signos, indeterminaciones.  
**Riesgos:** inventar resultados sin justificar; dominio.
**Capa recomendada:**
- Contrato “Never invent results”.
- Representación residual `limit(expr, x, a)` + warnings.
- Subcapas: clasificación de finitud, sign analysis (conservative).

### H6) Integrales impropias / convergencia (V3+)
**Motivo:** depende de límites y de criterios de convergencia.
**Riesgos:** muy fácil “pasarse de listo”.
**Capa recomendada:**
- Solo criterios conservadores y tipados (“Converges”, “Diverges”, “Unknown”).
- Tests contractuales con casos clásicos.

### H7) Dominio y supuestos (Assumptions framework)
**Motivo:** muchas simplificaciones solo son válidas con supuestos (x>0, x∈ℝ, etc.).
**Riesgos:** resultados incorrectos sin declarar asunciones.
**Capa recomendada:**
- “Domain assumptions” como primera clase (ya tenéis `domain_assumption` en Steps).
- Política: si se usa, debe aparecer como warning/step.
- Tests contractuales: outputs con/ sin supuestos.

### H8) Sistema de steps/substeps (UX/Tracing)
**Motivo:** depurar reglas requiere trazas fiables y no frágiles.
**Riesgos:** regressions por heurísticas de nombres o por pérdida de sub‑rewrites.
**Capa recomendada:**
- Steps con `importance` declarativo (ya iniciado).
- Substeps wiring opcional.
- Tests contractuales: schema y campos estables.

### H9) Iterative transform vs recursive transform (stack safety)
**Motivo:** evitar stack overflow y mejorar robustez.
**Riesgos:** order‑dependence, parent_ctx, requeue semantics.
**Capa recomendada:**
- Un único entrypoint con modo `recursive|iterative` (opcional), misma semántica.
- Contract tests de equivalencia en suites deterministas (rationalize/trig/abs barriers).
- Métrica: no overflow + budget/loop detection consistent.

### H10) Parsing + spans + highlighting para OCR UI
**Motivo:** Android/OCR necesita spans precisos.
**Riesgos:** spans incorrectos llevan a UX mala y debug difícil.
**Capa recomendada:**
- Span canónico (ya existe).
- Contrato del parser: span siempre que sea posible, o None.
- Tests contractuales de spans para inputs representativos.

---

## 9. Recomendación de priorización

1) **H1 Sustitución avanzada por patrones** (muy útil y muy peligrosa: requiere capa safe)  
2) **H2 Potencias/canonical form** (evita oscilaciones y loops futuros)  
3) **H4 Canonicalización global de resta** (solo con gating y detectores robustos)  
4) **H9 Iterative transform** (cuando queráis reactivar NormalizeBinomialOrderRule con seguridad)  
5) **H7 Assumptions framework** (para cálculo real y rigor)

---

## 10. Apéndice: plantilla de PR (copiable)

### PR Title
`[Layer] <Feature> — canonical module + contract + contract tests + enforcement`

### PR Checklist
- Canonical module: ✅
- Entry points (≤2): ✅
- Contract doc: ✅ (POLICY/docs)
- Contract tests: ✅
- Lint/enforcement: ✅
- Preset/flag (if needed): ✅
- Backward compatibility: ✅ (schema/versioning if API)
