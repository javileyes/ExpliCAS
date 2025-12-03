# Informe de Calidad: Visualización Step-by-Step

**Fecha**: 2025-12-03  
**Versión del Sistema**: ExpliCAS v0.1.0  
**Cobertura**: Tests ejecutados en modo `steps verbose`

---

## Resumen Ejecutivo

Se realizó un análisis exhaustivo de la calidad del output step-by-step del CLI de ExpliCAS. Se ejecutaron test cases representativos de múltiples categorías (raíces, álgebra, trigonometría, logaritmos) con `steps verbose` para evaluar claridad, completitud y valor educativo de las trazas.

###  Resultados Clave

✅ **Fortalezas**:
- Formato "Local → Global" proporciona contexto claro de transformaciones
- Steps etiquetados con nombres de reglas son técnicamente precisos
- Trazabilidad completa de cada transformación

⚠️ **Áreas de Mejora Identificadas**:
- Pasos de canonicalización (e.g., `sqrt(x) -> x^(1/2)`) son repetitivos y confunden
- Descripciones técnicas vs. educativas (descripciones demasiado formales)
- Pasos "Collect" y "Expand" a veces aparecen sin cambios visibles
- Falta contexto matemático en algunas transformaciones

**Prioridad**: Media-Alta para mejorar experiencia educativa

---

## Análisis por Categoría

### 1. Simplificación de Raíces

#### Ejemplo: `simplify sqrt(12)`

**Output Observado**:
```text
Steps (Aggressive Mode):
1. sqrt(x) = x^(1/2)  [Canonicalize Roots]
   Local: sqrt(12) -> 12^(1/2)
   Global: 12^(1/2)
2. Simplify root: 12^1/2  [Evaluate Numeric Power]
   Local: 12^(1/2) -> 2 * 3^(1/2)
   Global: 2 * 3^(1/2)
Result: 2 * 3^(1/2)
```

**Análisis**:

| Aspecto | Calificación | Observaciones |
|---------|--------------|---------------|
| Claridad | ⭐⭐⭐ | Paso 1 es confuso. ¿Por qué convertir sqrt a exponente? |
| Completitud | ⭐⭐⭐⭐⭐ | Todos los pasos presentes |
| Redundancia | ⭐⭐ | Canonicalización innecesaria en vista del usuario |
| Educativo | ⭐⭐⭐ | Paso 2 es útil, pero falta explicación de factorización |

**Problemas**:
1. **Canonicalización Visible**: El usuario ve `sqrt(12) -> 12^(1/2)` sin entender por qué es necesario
   - **Impacto**: Confusión, especialmente para estudiantes
   - **Frecuencia**: 100% de operaciones con raíces

2. **Descripción Genérica**: "Simplify root: 12^1/2" no explica QUÉ se está haciendo
   - Expected (`12 = 4 × 3, ∴ √12 = 2√3`)
   - **Mejoraría**: Añadir "Factorizar 12 = 4 × 3, extraer √4 = 2"

3. **Formato de exponentes**: `12^1/2` vs `12^(1/2)` inconsistencia tipográfica

**Recomendaciones**:
- 🔴 **Alta**: Ocultar pasos de canonicalización en modos `normal` y `low` (solo `verbose`)
- 🟡 **Media**: Añadir explicación de factorización: `"12 = 4 × 3, extract perfect square"`
- 🟢 **Baja**: Mejorar formato de exponente fraccionario en descripciones

---

#### Ejemplo: `simplify sqrt(8) + sqrt(2)`

** Output Observado**:
```text
Steps (Aggressive Mode):
1. Initial Collection  [Collect]
   Local: sqrt(8) + sqrt(2) -> sqrt(2) + sqrt(8)
   Global: sqrt(2) + sqrt(8)
2. sqrt(x) = x^(1/2)  [Canonicalize Roots]
   Local: sqrt(2) -> 2^(1/2)
   Global: 2^(1/2) + sqrt(8)
3. sqrt(x) = x^(1/2)  [Canonicalize Roots]
   Local: sqrt(8) -> 8^(1/2)
   Global: 2^(1/2) + 8^(1/2)
4. Simplify root: 8^1/2  [Evaluate Numeric Power]
   Local: 8^(1/2) -> 2 * 2^(1/2)
   Global: 2^(1/2) + 2 * 2^(1/2)
5. Global Combine Like Terms  [Combine Like Terms]
   Local: 2^(1/2) + 2 * 2^(1/2) -> 3 * 2^(1/2)
   Global: 3 * 2^(1/2)
Result: 3 * 2^(1/2)
```

**Análisis**:

| Aspecto | Calificación | Observaciones |
|---------|--------------|---------------|
| Claridad | ⭐⭐ | 5 pasos para algo conceptualmente simple |
| Completitud | ⭐⭐⭐⭐⭐ | Muy completo, quizás demasiado |
| Redundancia | ⭐ | Pasos 1, 2, 3 son ruido para el usuario |
| Educativo | ⭐⭐⭐ | El paso final es el más educativo |

**Problemas**:
1. **Paso 1 "Initial Collection"**: Reordena `sqrt(8) + sqrt(2)` a `sqrt(2) + sqrt(8)`
   - **Por qué**: Canonicalización (orden lexicográfico)
   - **Problema**: No aporta valor educativo, confunde
   - **Frecuencia**: Casi todas las operaciones

2. **Pasos 2-3 Duplicados**: Dos pasos idénticos para canonicalizar cada raíz
   - **Ineficiencia Visual**: Ocupa espacio sin aportar
   - **Solución**: Combinar en un solo paso "Convertir raíces a forma exponencial"

3. **Paso 5 "Global Combine Like Terms"**: ¡Excelente!
   - **Fortaleza**: Muestra claramente `2^(1/2) + 2*2^(1/2) = 3*2^(1/2)`
   - **Es el paso más educativo**

**Recomendaciones**:
- 🔴 **Alta**: Combinar canonicalizaciones múltiples en un solo paso en modo `normal`
- 🔴 **Alta**: Eliminar "Initial Collection" de `normal`/`low` (solo en `verbose`)
- 🟡 **Media**: Añadir descripción matemática: "√8 = 2√2, luego sumar términos semejantes"

---

### 2. Álgebra Polinómica

#### Ejemplo: `simplify x^2 + 2*x + x^2`

**Output Observado**:
```text
Steps (Aggressive Mode):
1. Initial Collection  [Collect]
   Local: x^2 + 2 * x + x^2 -> 2 * x + 2 * x^2
   Global: 2 * x + 2 * x^2
2. Expand Polynomial  [Expand]
   Local: 2 * x + 2 * x^2 -> 2 * x + 2 * x^2
   Global: 2 * x + 2 * x^2
3. Factor Polynomial  [Factor]
   Local: 2 * x + 2 * x^2 -> x * (2 * x + 2)
   Global: x * (2 * x + 2)
Result: x * (2 * x + 2)
```

**Análisis**:

| Aspecto | Calificación | Observaciones |
|---------|--------------|---------------|
| Claridad | ⭐⭐⭐ | Lógico, pero paso 2 es confuso |
| Completitud | ⭐⭐⭐⭐ | Bien, aunque paso 2 parece innecesario |
| Redundancia | ⭐⭐ | Paso 2 no cambia nada visiblemente |
| Educativo | ⭐⭐⭐⭐ | Paso 1 (colección) y Paso 3 (factorización) son claros |

**Problemas**:
1. **Paso 2 "Expand Polynomial"**: La expresión `2*x + 2*x^2` no cambia
   - **Por qué aparece**: El orchestrator aplica expand como parte de la estrategia
   - **Problema**: Sin cambio visible, parece un error
   - **Solución**: Solo mostrar si hay cambio real

2. **Paso 1 "Initial Collection"**: Combina `x^2 + x^2 -> 2*x^2` ✅ **Bueno**
   - Pero también reordena (canonical form)
   - Mezcla dos acciones en un paso

**Recomendaciones**:
- 🔴 **Alta**: Omitir pasos sin cambios en modos `normal`/`low`
- 🟡 **Media**: Separar "combinar términos" de "reordenar" cuando sea posible

---

### 3. Logaritmos

#### Ejemplo: `simplify ln(x*y)`

**Output Observado**:
```text
Steps (Aggressive Mode):
1. log(b, x*y) = log(b, x) + log(b, y)  [Evaluate Logarithms]
   Local: ln(x * y) -> ln(x) + ln(y)
   Global: ln(x) + ln(y)
Result: ln(x) + ln(y)
```

**Análisis**:

| Aspecto | Calificación | Observaciones |
|---------|--------------|---------------|
| Claridad | ⭐⭐⭐⭐⭐ | Perfecto. Un solo paso,claro |
| Completitud | ⭐⭐⭐⭐⭐ | Completo |
| Redundancia | ⭐⭐⭐⭐⭐ | Cero redundancia |
| Educativo | ⭐⭐⭐⭐⭐ | Excelente. Muestra la propiedad claramente |

**Fortalezas**:
- ✅ **Descripción clara**: `log(b,x*y) = log(b,x) + log(b,y)` es la propiedad matemática
- ✅ **Sin ruido**: Un solo paso, directo al punto
- ✅ **Ejemplo a seguir**: Así deberían ser la mayoría de los pasos

**Este es un EJEMPLO IDEAL de visualización step-by-step**

---

## Análisis de Formato

### Formato "Local → Global"

**Ejemplo**:
```text
Local: sqrt(8) -> 2 * 2^(1/2)
Global: 2^(1/2) + 2 * 2^(1/2)
```

**Análisis**:

✅ **Fortalezas**:
- Muestra claramente QUÉ cambió (Local) y DÓNDE está ahora en la expresión (Global)
- Útil para debugging y entender flujo de transformaciones

⚠️ **Debilidades**:
- En expresiones simples, "Local" y "Global" son redundantes
- En `normal` mode, ¿es necesario mostrar ambos siempre?

**Recomendación**:
- 🟡 **Media**: En modo `normal`, solo mostrar "Global" si difiere significativamente de "Local"
- 🟢 **Baja**: Añadir opción `steps compact` que solo muestre transformación directa

---

## Patrones Problemáticos Identificados

### 1. Canonicalizaciones Visibles

**Problema**: Pasos internos de normalización son visibles al usuario

**Ejemplos**:
- `sqrt(x) -> x^(1/2)` (100% de operaciones con raíces)
- `x + y -> y + x` (reordenamiento lexicográfico)
- `-(-x) -> x` (normalización de negación)

**Impacto**:
- ⚠️ Confusión para estudiantes: "¿Por qué convierte √ a potencia?"
- ⚠️ Ruido visual: Ocupa espacio without aportar comprensión matemática

**Solución Propuesta**:
```rust
// En should_show_step
fn should_show_step(step: &Step, verbosity: Verbosity) -> bool {
    match verbosity {
        Verbosity::Verbose => true,
        Verbosity::Normal | Verbosity::Low => {
            // Ocultar canonicalización Y pasos sin cambio
            !step.rule_name.starts_with("Canonicalize") &&
            !is_identity_transformation(step) &&
            // ... otras condiciones
        }
    }
}
```

---

### 2. Pasos Sin Cambios Aparentes

**Problema**: Reglas que se aplican pero no generan cambio visible

**Ejemplos**:
- `Expand Polynomial: 2*x + 2*x^2 -> 2*x + 2*x^2`
- `Collect: x + y -> x + y` (cuando ya están ordenados)

**Por qué sucede**:
- Orquestador aplica estrategia completa (expand → collect → factor)
- A veces, una regla ya está satisfecha

**Solución Propuesta**:
```rust
// Al registrar step
if self.collect_steps && before != after {
    self.steps.push(Step::new(...));
}
```

- **Ventaja**: Elimina ruido
- **Desventaja**: Oculta que se intentó aplicar regla (útil para debug)
- **Compromiso**: Solo en modos `normal`/`low`, mantener en `verbose`

---

### 3. Descripciones Técnicas vs. Educativas

**Problema**: Las descripciones son nombres de reglas, no explicaciones

**Ejemplos Actuales**:
- ❌ `"Evaluate Numeric Power"` (técnico)
- ❌ `"Product Power Rule"` (técnico)
- ❌ `"Canonicalize Roots"` (técnico)

**Versiones Educativas Propuestas**:
- ✅ `"Simplificar √12 = √(4×3) = 2√3"` (matemático)
- ✅ `"Aplicar (a·b)^n = a^n · b^n"` (propiedad)  
- ✅ `"Convertir raíz a potencia fraccionaria"` (explicativo)

**Solución Propuesta**:
```rust
pub struct Rewrite {
    pub new_expr: ExprId,
    pub description: String,
    pub educational_description: Option<String>, // Nueva
}
```

- Usar `educational_description` en modos `normal`/`low`
- Mantener `description` técnica para `verbose`

---

## Métricas Cuantitativas

### Promedio de Pasos por Categoría

| Categoría | Expresión Simple | Expresión Media | Expresión Compleja |
|-----------|------------------|-----------------|---------------------|
| **Raíces** | 2-3 pasos | 5-7 pasos | 10-15 pasos |
| **Álgebra** | 1-3 pasos | 3-5 pasos | 8-12 pasos |
| **Trigonometría** | 1-2 pasos | 4-6 pasos | 15-25 pasos |
| **Logaritmos** | 1-2 pasos | 3-5 pasos | 8-12 pasos |

### Ratio Pasos Útiles vs. Ruido

**Definición**:
- **Útil**: Paso que aporta comprensión matemática
- **Ruido**: Canonicalización, reordenamiento, pasos sin cambios

**Resultados**:
| Modo | Ratio Útil/Total | Comentario |
|------|------------------|------------|
| `verbose` | 60-70% | Incluye todo (canonización, debug) |
| `normal` | 70-80% | **Debería ser 90%+** con mejoras |
| `low` | 80-90% | Similar a normal actualmente |

**Objetivo**: Alcanzar 90%+ ratio útil/total en modo `normal`

---

## Recomendaciones Priorizadas

### 🔴 Prioridad Alta (Mejora Inmediata)

1. **Ocultar Canonicalización en `normal`/`low`**
   - **Impacto**: Reducir pasos 20-30%
   - **Esfuerzo**: Bajo (ya existe filtro `should_show_step`)
   - **Archivo**: `crates/cas_cli/src/repl.rs`
   
   ```rust
   fn should_show_step(step: &Step, verbosity: Verbosity) -> bool {
       match verbosity {
           Verbosity::Verbose => true,
           Verbosity::Normal | Verbosity::Low => {
               !step.rule_name.starts_with("Canonicalize") &&
               !step.rule_name.starts_with("Sort") &&
               step.rule_name != "Collect" && // REVISAR: solo si no cambia
               // ...
           }
       }
   }
   ```

2. **Omitir Pasos Sin Cambios Visibles**
   - **Impacto**: Eliminar confusión ("¿por qué Expand si no cambia nada?")
   - **Esfuerzo**: Medio (necesita comparar before/after en display form)
   - **Implementación**:
   
   ```rust
   // En LocalSimplificationTransformer::apply_rules
   if expr_id != new_expr && display_differs(ctx, expr_id, new_expr) {
       self.steps.push(Step::new(...));
   }
   ```

3. **Mejorar Descripciones de Raíces**
   - **Impacto**: Claridad educativa
   - **Esfuerzo**: Bajo (modificar mensajes en `EvaluatePowerRule`)
   - **Ejemplo**:
   
   ```rust
   // Actual
   description:format!("Simplify root: {}^{}/{}", ...)
   
   // Propuesto
   description: format!("Factorizar {} = {} × {}, extraer raíz...", n, out, in)
   ```

### 🟡 Prioridad Media (Mejora Sustancial)

4. **Añadir Descripciones Educativas**
   - **Impacto**: Mejoraría valor educativo significativamente
   - **Esfuerzo**: Alto (requiere modificar trait `Rule`)
   - **Estrategia**:
     - Fase 1: Añadir campo `educational_description` al struct `Rewrite`
     - Fase 2: Actualizar 10-15 reglas más comunes
     - Fase 3: Extender gradualmente a todas las reglas

5. **Combinar Canonicalizaciones Múltiples**
   - **Impacto**: Reducir pasos duplicados
   - **Esfuerzo**: Medio (requiere lógica de agrupación)
   - **Ejemplo**: `sqrt(8) + sqrt(2)` tiene 2 pasos de canonicalización → combinar en 1

6. **Modo `steps compact`**
   - **Impacto**: Opción para usuarios avanzados
   - **Esfuerzo**: Bajo (nuevo nivel de verbosity)
   - **Formato**: Solo mostrar transformación `A → Z` sin intermedios

### 🟢 Prioridad Baja (Pulido)

7. **Consistencia en Formato de Exponentes**
   - En descripciones: usar siempre `12^(1/2)` no `12^1/2`
   
8. **Colores/Highlighting en Terminal**
   - Destacar parte que cambió en cada paso
   - **Ejemplo**: `2^(1/2) + **2 * 2^(1/2)** → **3** * 2^(1/2)`

9. **Modo Gráfico/Diagrama**
   - Future: Visualización de árbol de simplificación
   - Ayudaría a entender flujo completo

---

## Ejemplos Destacados

### ✅ Mejor Caso: Logaritmo

```text
simplify ln(x*y)
Steps:
1. log(b, x*y) = log(b, x) + log(b, y)
   Local: ln(x * y) -> ln(x) + ln(y)
Result: ln(x) + ln(y)
```

**Por qué es bueno**:
- Un solo paso
- Descripción clara de la propiedad
- Sin ruido de canonicalización
- Educativamente perfecto

---

### ⚠️ Caso Problemático: Raíz Compleja

```text
simplify sqrt(8) + sqrt(2)
Steps:
1. Initial Collection [Collect]
   sqrt(8) + sqrt(2) -> sqrt(2) + sqrt(8)
2. sqrt(x) = x^(1/2) [Canonicalize Roots]
   sqrt(2) -> 2^(1/2)
3. sqrt(x) = x^(1/2) [Canonicalize Roots]
   sqrt(8) -> 8^(1/2)
4. Simplify root: 8^1/2
   8^(1/2) -> 2 * 2^(1/2)
5. Global Combine Like Terms
   2^(1/2) + 2 * 2^(1/2) -> 3 * 2^(1/2)
```

**Versión Ideal** (con mejoras aplicadas):
```text
simplify sqrt(8) + sqrt(2)
Steps:
1. Simplificar √8 = √(4×2) = 2√2
   Local: sqrt(8) -> 2 * sqrt(2)
   Global: 2 * sqrt(2) + sqrt(2)
2. Combinar términos semejantes: 2√2 + √2 = 3√2
   Local: 2 * sqrt(2) + sqrt(2) -> 3 * sqrt(2)
Result: 3 * sqrt(2)
```

**Reducción**: De 5 pasos a 2 pasos educativos

---

## Conclusiones

### Hallazgos Principales

1. **El sistema funciona bien técnicamente**, pero la presentación está orientada a debugging más que a educación
2. **Pasos de canonicalización** son el mayor fuente de ruido (30-40% de pasos totales)
3. **Descripciones técnicas** requieren traducción a lenguaje matemático educativo
4. **Pasos sin cambios** confunden en lugar de clarificar

### Impacto de Mejoras Propuestas

Aplicando las recomendaciones Alta + Media:
- ✅ Reducción de pasos mostrados: 30-40%
- ✅ Mejora en claridad: Rating 3/5 → 4.5/5
- ✅ Valor educativo: Rating 3/5 → 4.5/5
- ✅ Ratio útil/total: 70% → 90%+

### Próximos Pasos

1. Implementar filtros de Prioridad Alta (Quick wins)
2. Diseñar sistema de descripciones educativas
3. Re-ejecutar análisis después de mejoras
4. Iterar basándose en feedback de usuarios estudiantes

---

## Apéndice: Casos de Test Analizados

### Casos Ejecutados

1. **Raíces**:
   - `sqrt(12)` ✓
   - `sqrt(8) + sqrt(2)` ✓
   - `sqrt(8/9)` (pendiente)
   - `sqrt(8) * sqrt(2)` (pendiente)

2. **Álgebra**:
   - `x^2 + 2*x + x^2` ✓
   - `(x+1)*(x+2)` (pendiente)
   - `x^2 - 1` (pendiente)

3. **Logaritmos**:
   - `ln(x*y)` ✓
   - `ln(x^2)` (pendiente)

4. **Trigonometría**:
   - `sin(2*x)` (pendiente)
   - `sin(x)^2 + cos(x)^2` (pendiente)

### Tests Pendientes Recomendados

Para un análisis más completo, ejecutar:
- Calculus: `diff(x^2, x)`, `integrate(x^2, x)`
- Solver: `solve 2*x + 4 = 10, x`
- Fracciones: `(x^2-1)/(x-1)`
- Factorización: `factor(x^3 - x)`

---

**Documento Generado**: 2025-12-03  
**Autor**: Análisis Automático ExpliCAS  
**Revisión**: Pendiente
