# Informe de Mejoras - ExpliCAS Engine

**Fecha**: 2025-12-03  
**Análisis**: Arquitectura, Performance, Mantenibilidad, Depuración

---

## Resumen Ejecutivo

### Estado Actual
✅ **Logros Recientes**:
- Multi-pass orchestration implementado
- Fraction simplification con opposite/same denominators
- 100% tests passing (17/17 + suite completa)
- "El Puente Conjugado" resuelto
- **Context Mode** (2025-12): Auto-detection (integrate→IntegratePrep), Werner/Morrie rules, Solve-safe config
- **Domain Warning Deduplication** (2025-12): `DomainWarning` struct con rule_name source

⚠️ **Problemas Críticos Detectados**:
- **Performance regression**: Hasta **+99.9%** slowdown en sum_fractions_10
- Multi-pass overhead afectando casos simples
- Falta de herramientas de profiling/debugging
- Código duplicado en pattern matching
- Cache invalidation ineficiente

---

## 1. ANÁLISIS DE PERFORMANCE 🔴

### Benchmark Regressions

| Benchmark | Regression | Causa Probable |
|-----------|-----------|----------------|
| `sum_fractions_10` | **+99.9%** | Multi-pass loop ejecutando en casos que no lo necesitan |
| `integrate_trig_product` | +65.9% | Multi-pass + complejidad en trig |
| `solve_quadratic` | +50.7% | Solver llamando simplifier con overhead |
| `diff_nested_trig_exp` | +34.4% | Recursión profunda con multi-pass |
| `expand_binomial_power_10` | +15.7% | Cache invalidation |

### Root Cause Analysis

**Problema 1: Multi-Pass sin Early Exit Inteligente**

El loop siempre ejecuta `compare_expr` que es O(n) estructural, incluso cuando `simplified == current` (comparación de IDs que es O(1)).

**Solución**: Early exit con ID check primero.

**Problema 2: AddFractionsRule Always Evaluating**

Se llama para TODA Add, incluso `Add(Number, Number)`, ejecutando `get_num_den` antes de verificar si son fracciones.

**Solución**: Early rejection basado en tipos.

**Problema 3: Cache Invalidation en Multi-Pass**

Cada iteración crea nuevo LocalSimplificationTransformer con

 cache vacío, re-simplificando subexpresiones.

**Solución**: Cache persistente across passes.

---

## 2. OPTIMIZACIONES PROPUESTAS 🚀

### 2.1. Conditional Multi-Pass (HIGH IMPACT)

Solo ejecutar multi-pass cuando reglas específicas ("cascade triggers") se disparan:
- RationalizeDenominatorRule
- ExpandPolynomialRule
- FactorRule

**Expected Impact**: -80% regression en casos simples

### 2.2. Persistent Cache Across Passes

Mantener cache entre iteraciones del multi-pass loop.

**Expected Impact**: -30% en casos con multi-pass

### 2.3. Rule Priority System

Ordenar reglas por probabilidad de match usando hit counters.

**Expected Impact**: -10% en promedio

---

## 3. HERRAMIENTAS DE DEBUG Y VISUALIZACIÓN 🔍

### 3.1. Interactive Debugger

```bash
$ cas-cli debug
> break AddFractionsRule
> run simplify 1/(x-1) + 1/(1-x)
Breakpoint hit: AddFractionsRule
> step
> continue
```

### 3.2. AST Visualizer (Graphviz)

Generar visualizaciones SVG del árbol de expresiones.

### 3.3. Simplification Timeline (HTML)

Timeline interactivo mostrando cada paso con complejidades.

### 3.4. Profiling Integration

Reportes de tiempo por regla con `--feature profiling`.

---

## 4. REFACTORIZACIONES ARQUITECTÓNICAS 🏗️

### 4.1. Separar Orchestrator de Simplifier

Clarificar responsabilidades:
- Orchestrator: Estrategia alto nivel
- Simplifier: Aplicación de reglas
- RuleEngine: Match & Apply

### 4.2. Modularizar `are_denominators_opposite`

Separar 80 líneas en funciones específicas por patrón.

### 4.3. Type-Safe Rule Registration

Garantías compile-time sobre tipos de expresiones.

### 4.4. Error Handling con Result

Errores explícitos en lugar de silent failures (None).

---

## 5. MEJORAS DE ROBUSTEZ 🛡️

### 5.1. Infinite Loop Detection

Detectar ciclos de simplificación y abortar early.

### 5.2. Rule Consistency Validation

Tests automáticos verificando:
- Determinismo
- No incremento excesivo de complejidad
- Idempotencia

### 5.3. Memory Limits

Configuración de límites:
- Max expr size
- Max simplification time
- Max passes

---

## 6. MEJORAS DE MANTENIBILIDAD 📚

### 6.1. Documentación Auto-generada

Generar markdown con ejemplos de cada regla.

### 6.2. Integration Tests por Categoría

Tests organizados por dominio matemático.

### 6.3. Refactorizar `get_num_den`

Extraer a clase `FractionExtractor` con métodos específicos.

---

## 7. IMPLEMENTACIÓN PRIORITARIA 🎯

### Phase 1: Quick Wins (1 semana) - CRÍTICO
1. Conditional multi-pass
2. Early exit optimization
3. AddFractionsRule early rejection
4. Cycle detection

### Phase 2: Debug Tools (1 semana)
1. Basic profiler
2. AST visualizer
3. Timeline HTML

### Phase 3: Refactoring (2 semanas)
1. Modularizar opposite denominators
2. Extract FractionExtractor
3. Error handling

### Phase 4: Advanced (futuro)
1. Pattern compilation DSL
2. Interactive debugger
3. Type-safe registration

---

## 8. MÉTRICAS DE ÉXITO

**Performance Targets**:
- sum_fractions_10: < +10% (actual: +99%)
- integrate_trig_product: < +15% (actual: +65%)
- solve_quadratic: < +10% (actual: +50%)

**Code Quality**:
- Cyclomatic complexity < 10
- Test coverage > 80%
- Documentation > 90%

---

## CONCLUSIÓN

El sistema está **funcionalmente correcto** pero tiene **serios problemas de performance**. Las optimizaciones propuestas son **implementables en corto plazo** y recuperarán el performance.

**Prioridad #1**: Conditional multi-pass para recuperar performance.

**Beneficio a Largo Plazo**: Herramientas de debug mejorarán mantenibilidad.
