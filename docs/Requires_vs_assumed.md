# Requires vs Assumptions (y por qué esto hace especial a este engine)

Este motor no solo simplifica “porque sí”: **explica** qué hizo, qué **no** pudo hacer, y—lo más importante—distingue entre:

- **Requisitos (Requires)**: condiciones de dominio **ya implícitas** en la expresión original (no son “decisiones” del motor).
- **Suposiciones (Assumed / Assumptions)**: condiciones **adicionales** que el motor **elige aceptar** para poder aplicar una transformación (según el modo semántico).

Esta separación es clave para un uso **educativo** y también para integrar el motor en apps (FFI) sin perder honestidad matemática.

---

## Contrato Formal (TL;DR)

| Canal | Qué significa | Cuándo aparece |
|-------|---------------|----------------|
| **ℹ️ Requires** | Dominio **implícito** de la entrada | Siempre que se "consume" una restricción |
| **⚠ Assumed** | Hipótesis **extra** aceptada por política | `Generic` (≠0) o `Assume` (todo) |
| **Blocked** | Regla no aplicable en modo actual | Cuando modo lo impide + `hints on` |

> **Invariante**: Requires ≠ Assumed — nunca se mezclan.

---

## Referencia Rápida (verificable en REPL)

```bash
# 1. Requires: dominio implícito
> sqrt(x)^2
ℹ️ Requires: x ≥ 0
Result: x

# 2. Assumed: agujero algebraico
> x/x  # (en Generic)
⚠ Assumed x ≠ 0
Result: 1

# 3. Witness survival: no emite Requires si sqrt sobrevive
> (x-y)/(sqrt(x)-sqrt(y))
⚠ Assumed x - y ≠ 0
Result: √(x) + √(y)
# (NO hay x≥0, y≥0)
```

---

## 1) Glosario rápido

### Requires (requisitos de dominio)
Son condiciones necesarias para que la expresión **tenga sentido** (en ℝ / `RealOnly`) por su propia estructura.

Ejemplos típicos en `RealOnly`:

- `sqrt(t)` ⇒ **requiere** `t ≥ 0`
- `ln(t)` ⇒ **requiere** `t > 0`
- `1/t` ⇒ **requiere** `t ≠ 0`

Si el motor transforma una expresión y “se pierde” una de estas restricciones, el motor **no debe mentir**:
- o bien bloquea,
- o bien **propaga** ese dominio como `Requires` (y lo muestra explícitamente).

> Idea: “No estoy suponiendo nada: estoy diciendo el dominio de validez que ya estaba ahí.”

---

### Assumed (suposiciones)
Son condiciones que el motor **acepta** para poder aplicar reglas que no son universalmente válidas sin restricciones.

Ejemplos:
- Cancelar `(x-y)/(x-y)` ⇒ asumir `x-y ≠ 0`
- Expandir `ln(xy)` ⇒ asumir `x > 0` y `y > 0`
- Simplificar `exp(ln(x)) → x` ⇒ asumir `x > 0`

Estas suposiciones **dependen de la política de exploración** (modo semántico).

> Invariante: **No hay suposiciones sin registro** (timeline / resumen).

---

### Blocked (bloqueos)
Cuando una transformación requeriría suponer algo que el modo actual **no permite**, el motor no se calla:
- **no aplica** la regla
- muestra un hint del tipo “requires …; usa `domain assume`”.

---

## 2) Los modos: qué cambia y qué no

Lo importante: **Requires siempre existe** (cuando se puede inferir) y se propaga de forma estable.
Lo que cambia entre modos es **qué suposiciones** se permiten.

| Modo | Requisitos (Requires) | Suposiciones Definibility (≠0) | Suposiciones Analytic (>0, ≥0, rangos) |
|------|------------------------|---------------------------------|----------------------------------------|
| Strict | ✅ Sí (se muestran) | ❌ No | ❌ No |
| Generic | ✅ Sí (se muestran) | ✅ Sí | ❌ No |
| Assume | ✅ Sí (se muestran) | ✅ Sí | ✅ Sí |

**Lectura didáctica:**
- **Strict**: “no acepto hipótesis extra”
- **Generic**: “acepto agujeros algebraicos típicos (≠0)”
- **Assume**: “acepto hipótesis analíticas (positividad, rangos, etc.) y lo dejo por escrito”

---

## 3) Ejemplos clave (REPL)

### A) `sqrt(x)^2 → x` en Generic (Requires, no Assumed)

En ℝ, `sqrt(x)` ya implica `x ≥ 0`. Simplificar `sqrt(x)^2` a `x` **no inventa** el dominio: lo **hace explícito**.

```txt
> sqrt(x)^2
ℹ️ Requires:
  - x ≥ 0
Result: x

**Interpretación:**

* El motor no dice “asumí x≥0”.
* Dice: “para que esto tenga sentido en ℝ, se requiere x≥0”.
* Es una mejora enorme en UX educativa: el alumno avanza **sin cambiar a Assume**, pero el dominio no se pierde.

---

### B) Caso “dominio implícito + singularidad removible”

[
\frac{x-y}{\sqrt{x}-\sqrt{y}} = \sqrt{x}+\sqrt{y} \quad \text{(con cuidado en } x=y)
]

Aquí ocurre algo sutil y valioso:

* `sqrt(x)` y `sqrt(y)` ya traen `x≥0`, `y≥0` implícitos.
* Al simplificar, el único “agujero” real es el denominador `sqrt(x)-sqrt(y)` (equivale a `x-y`).

En `Generic` el motor puede simplificar y solo asume el agujero típico:

```txt
> (x - y) / (sqrt(x) - sqrt(y))
⚠ Assumed x - y ≠ 0
Result: √(x) + √(y)
```

Fíjate en lo importante:

* **NO** aparece `x ≥ 0, y ≥ 0` como suposición,
* porque el motor detecta que esas restricciones eran **implícitas** (y además el “testigo” `sqrt(...)` sigue presente: *witness survival*).

---

### C) `exp(ln(x))` en Generic: bloqueado por condición Analytic

En ℝ, `ln(x)` requiere `x>0`, pero aquí **no está garantizado** que el usuario “acepte” esa positividad como hipótesis extra para reescrituras.

```txt
> exp(ln(x))
Result: e^(ln(x))
ℹ️ Blocked simplifications:
  - requires x > 0 [Exponential-Log Inverse]
  tip: use `domain assume` to allow
```

La diferencia con `Requires` del caso anterior:

* Aquí la reescritura `exp(ln(x)) → x` **sí cambia** el significado fuera de `x>0`.
* En `Generic`, las condiciones Analytic no se asumen.

---

### D) El mismo ejemplo en Assume: se resuelve y queda registrado

```txt
> semantics set domain assume
> exp(ln(x))
Result: x
ℹ️ Assumptions used:
  - x > 0
```

Y en timeline, el paso queda como:

* `Assumed: x > 0`

---

### E) Cancelación algebraica típica: `x/x → 1`

* En **Strict**: se bloquea (no se asumen agujeros).
* En **Generic**: se permite con `Assumed x ≠ 0` (definability hole).

```txt
> x/x
Result: x/x
ℹ️ Blocked:
  - requires x ≠ 0 [Cancel Common Factors]
  tip: use `domain generic` to allow definability assumptions

> semantics set domain generic
> x/x
Result: 1
ℹ️ Assumptions used:
  - x ≠ 0
```

Esto es exactamente el “contrato complementario”:

* Generic simplifica fracciones y racionales de forma práctica
* Strict permanece “sin hipótesis extra”

---

## 4) Por qué esto es un multiplicador educativo

Sin esta separación, un motor suele caer en uno de los dos extremos:

1. **CAS clásico**: simplifica agresivo y el alumno no aprende qué condiciones se usaron.
2. **Sistema pedante**: bloquea muchísimo y el alumno no avanza.

Este engine busca una tercera vía:

* **Avanzar**, pero sin mentir:

  * si el dominio estaba implícito: **Requires**
  * si se aceptó una hipótesis extra: **Assumed**
  * si el modo actual no lo permite: **Blocked + tip**

El resultado es un “profesor automático”:

* muestra dominio cuando importa,
* registra suposiciones cuando se toman,
* y guía al usuario cuando está bloqueado.

---

## 4.1) Propagación en reutilización de expresiones

Un aspecto clave del sistema es que los **Requires se propagan automáticamente** cuando reutilizas expresiones de la sesión mediante `#id`.

### Comportamiento

Cuando evalúas `#1 + 4` (donde `#1` era `sqrt(x)`):

1. El motor **resuelve** `#1` → `sqrt(x)`
2. Construye la expresión compuesta: `sqrt(x) + 4`
3. **Re-infiere** el dominio estructural de la expresión resuelta
4. Muestra `ℹ️ Requires: x ≥ 0`

### Ejemplo en REPL

```bash
> sqrt(x)
#1: √(x)
ℹ️ Requires:
  - x ≥ 0

> #1 + 4
#2: 4 + √(x)
ℹ️ Requires:
  - x ≥ 0    ← SE PROPAGA

> show #2
Entry #2:
  Parsed:     4 + #1
  Resolved:   4 + sqrt(x)
  ℹ️ Requires:
    - x ≥ 0  ← VISIBLE EN SHOW
```

### Expresiones combinadas

Cuando combinas sub-expresiones con diferentes restricciones, **todas se acumulan**:

```bash
> sqrt(x) + ln(y)
ℹ️ Requires:
  - x ≥ 0    ← de sqrt(x)
  - y > 0    ← de ln(y)
```

### Por qué es importante

| Sin propagación | Con propagación |
|-----------------|-----------------|
| El alumno "pierde" el dominio al reutilizar | El dominio viaja con la expresión |
| Fácil cometer errores silenciosos | Transparencia total |
| El motor "olvida" contexto | La sesión es coherente |

> **Invariante**: Los Requires se infieren de la **estructura** de la expresión resuelta, no se "almacenan" con el ID. Esto significa que siempre reflejan el estado actual.

---

## 5) Qué puedes pedirle al motor (comandos útiles)

* Cambiar política:

  * `semantics set domain strict|generic|assume`
* Mostrar/ocultar hints:

  * `semantics set hints on|off`
* Resumen de suposiciones:

  * `semantics set assumptions off|summary`
* **Nivel de Requires** (nuevo en V1.3.8):

  * `semantics set requires essential|all`
  * `essential`: solo muestra Requires cuyo testigo fue consumido (ej: `sqrt(x)² → x`)
  * `all`: muestra todos los Requires, incluso si el testigo sobrevive
* Modo explicación:

  * `explain on|off`

---

## 6) Contratos e invariantes (para confianza)

* **No assumptions without record**:
  toda hipótesis aceptada aparece en timeline / resumen.
* **Requires no se mezcla con Assumed**:
  requiere = dominio implícito; assumed = política.
* **Rule-agnostic safety** (airbag):
  el motor detecta reescrituras que “pierden” restricciones implícitas y:

  * o las conserva mediante *witness survival*,
  * o las vuelve explícitas como `Requires`.

---

## 7) Ejercicios propuestos (para aula)

1. Simplifica y explica:

   * `sqrt(x)^2`
   * `1/ln(x)` (¿qué requiere?)
   * `(x^2-1)/(x-1)` (¿qué asume?)
2. Compara modos:

   * `ln(x*y)` en Strict/Generic/Assume
3. Detecta singularidades removibles:

   * `(x-y)/(sqrt(x)-sqrt(y))`
   * `(x^2-1)/(x-1)`

---

## Conclusión

La combinación de:

* **Step-by-step**
* **Requires vs Assumed**
* **Blocked hints guiados**
  hace que el motor sea útil *a la vez* para:
* aprender (dominio + condiciones),
* trabajar (simplificación potente en Generic),
* y mantener rigor en Generic y trazabilidad en Assume.
* modo Strict es el modo pendántico, que no permite suposiciones extra, ni tampoco presupone que las condiciones implícitas son ciertas.