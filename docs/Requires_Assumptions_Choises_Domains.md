# Manual de Semántica de Dominio en ExpliCAS

**Versión:** borrador para revisión matemática
**Propósito:** especificar cómo ExpliCAS gestiona condiciones de dominio durante simplificación paso a paso, distinguiendo entre restricciones necesarias (*requires*) y supuestos no deducibles (*assumptions*), incluyendo elecciones de rama (*branch choice*) y extensiones de dominio (*domain extension*).

## 1. Objetivo de la política de dominio

El motor debe producir transformaciones didácticas que sean:

1. **Correctas matemáticamente** (en el dominio que declara operar, por defecto ℝ).
2. **Transparentes**: el usuario debe ver qué condiciones son necesarias para la validez.
3. **No engañosas**: el motor no debe “introducir silenciosamente” restricciones o elecciones de rama sin avisar.
4. **Deterministas**: condiciones equivalentes deben mostrarse con una forma canónica y sin duplicados.

---

## 2. Terminología y definiciones

### 2.1 Requires (input)

Conjunto de condiciones **necesarias para que la expresión de entrada esté definida** en el dominio actual (por defecto ℝ). Se infieren directamente de la expresión inicial.

Ejemplos (en ℝ):

* `1/(x-1)` ⇒ requiere `x-1 ≠ 0`
* `sqrt(x)` ⇒ requiere `x ≥ 0`
* `log(x)` ⇒ requiere `x > 0`

**Propiedad:** las reglas de simplificación no deben “añadir” esto como *Assume*: es parte del dominio implícito del input.

---

### 2.2 Requires introduced (por un paso)

Condiciones **nuevas** que aparecen porque un paso transforma la expresión de manera que **para que el paso sea lógicamente válido (equivalencia) o la nueva expresión esté definida**, hay que añadir restricciones adicionales que **no estaban implicadas** por el dominio del input.

Se muestran como:
**ℹ️ Requires (introduced by Step k): …**

Ejemplos típicos:

* Transformación que introduce una división nueva por una expresión que no estaba previamente en denominador.
* Aplicación de una identidad condicionada que estrecha el dominio (p.ej., reglas de logaritmos en ℝ).

---

### 2.3 Assumption (HeuristicAssumption)

Hipótesis que el motor introduce para **elegir una forma “más simple” o aplicar una regla útil**, pero que **no es estrictamente necesaria** para la validez general o **no está justificada** por el dominio del input.

Se muestran como:
**⚠️ Assumes (heuristic): …**

Ejemplos:

* Simplificaciones agresivas no válidas en general sin condiciones adicionales (si el motor decide aplicarlas igualmente por modo “agresivo”).
* Reglas orientadas a “forma bonita” más que a equivalencia total.

---

### 2.4 Branch choice (elección de rama)

Se usa cuando el motor elige una rama/convención para una función multivaluada o una identidad que depende de rangos principales.

Se muestran como:
**🔀 Branch choice: …**

Ejemplos:

* `sqrt(x^2) → x` (en ℝ) elige la rama `x ≥ 0` (la identidad general es `sqrt(x^2)=|x|`).
* Simplificaciones de trigonometría inversa usando rangos principales:

  * `sin(arcsin(x)) → x` (válida si `arcsin` se entiende como inversa principal; requiere además `x∈[-1,1]` como dominio de `arcsin` en ℝ).

---

### 2.5 Domain extension (extensión de dominio)

Cuando una regla cambia el dominio operativo del motor (por ejemplo, de ℝ a ℂ) para continuar la simplificación.

Se muestra como:
**🧿 Domain extension: ℝ → ℂ (… motivo …)**

Ejemplo:

* `(-1)^(1/2) → i` requiere operar en ℂ.

---

## 3. Estructuras internas del motor

El motor mantiene:

* `required_conditions: Vec<ImplicitCondition>`
  Condiciones que el motor considera necesarias (input o introducidas).
* `assumption_events: Vec<AssumptionEvent>`
  Eventos de suposición/elección de rama/extensión.

### 3.1 Clasificación final de eventos (AssumptionKind)

Cada `AssumptionEvent` se clasifica en:

* `DerivedFromRequires` (redundante: deducible del input y/o de requires ya acumulados; **no se muestra**)
* `RequiresIntroduced` (**se muestra** como requires introducidos)
* `HeuristicAssumption` (**se muestra** como assume)
* `BranchChoice` (**se muestra**)
* `DomainExtension` (**se muestra**)

---

## 4. Algoritmo de inferencia y consolidación

### 4.1 Inferencia de Requires del input

Antes de cualquier paso:

1. `global_requires = infer_implicit_domain(input_expr)`
2. normalizar forma de cada condición (`normalize_condition`)
3. deduplicar por equivalencia (incluida equivalencia por signo en condiciones `E ≠ 0`).

Se muestran como:
**ℹ️ Requires (input): …**

---

### 4.2 Procesamiento de cada Step

Para cada paso `k`:

* `step.required_conditions` se normaliza.
* Se calcula:

  * `new_requires_k = step.required_conditions - (global_requires ∪ introduced_requires_so_far)`
* Si `new_requires_k` no está vacío:

  * se añade a `introduced_requires_so_far`
  * se muestra como **Requires (introduced)** si el paso realmente introduce restricciones nuevas.

Para `assumption_events`:

* Si el evento corresponde a una condición `C` y `C` está implicada por `(global_requires ∪ introduced_requires_so_far)`:

  * se reclasifica a `DerivedFromRequires` y no se muestra.
* Si no está implicada:

  * se mantiene según su kind (HeuristicAssumption / BranchChoice / DomainExtension)
  * o se reclasifica a RequiresIntroduced si el evento en realidad refleja una identidad condicional que estrecha dominio.

---

## 5. Principios matemáticos (contratos)

### 5.1 Contrato de equivalencia bajo Requires

Cuando el motor aplica un paso `E → E'` sin declarar “Assume” ni “BranchChoice”, el contrato es:

> Bajo `Requires_total = Requires_input ∪ Requires_introduced`,
> se garantiza que `E` y `E'` son equivalentes (mismo valor, donde estén definidas).

### 5.2 Contrato para Branch choice

Cuando se emite `BranchChoice`, el contrato pasa a ser:

> Bajo `Requires_total` y la condición de rama indicada, el resultado es válido.
> Sin esa condición, el resultado puede diferir de la identidad general.

### 5.3 Contrato para HeuristicAssumption

Cuando se emite `HeuristicAssumption`:

> El motor eligió una simplificación por utilidad/estética o por modo agresivo.
> La equivalencia general puede no sostenerse sin hipótesis adicionales.

### 5.4 Contrato para Domain extension

Cuando se emite `DomainExtension`:

> El motor cambia el dominio operativo (p.ej. ℝ→ℂ).
> Los requires y reglas posteriores se interpretan en el nuevo dominio.

---

## 6. Ejemplos canónicos

### 6.1 Divisiones y cancelación

**Entrada:** `(x^2 - 4) / (x - 2)`
**Requires (input):** `x-2 ≠ 0`

**Steps didácticos:**

1. Factor: `(x^2 - 4)/(x-2) → (x-2)(x+2)/(x-2)`
2. Cancel: `(x-2)(x+2)/(x-2) → x+2`

**No** se emite `Assume x-2 ≠ 0` en pasos si ya está en requires.

---

### 6.2 Caso signo: `P^n / (-P)`

**Entrada:** `(x-y)^4 / (y-x)`
**Requires (input):** `y-x ≠ 0` (equivalente a `x-y ≠ 0`)

**Steps:**

1. Reconocer: `y-x = -(x-y)` (si se muestra)
2. Cancel: `P^4/(-P) → -P^3`
   **Resultado:** `-(x-y)^3`

---

### 6.3 Racionalización con raíz

**Entrada:** `1/(sqrt(x)-1)`
**Requires (input):** `x ≥ 0`, `sqrt(x)-1 ≠ 0`

Paso de racionalización preserva equivalencia bajo esas condiciones; no introduce nuevas.

---

### 6.4 Rama: `sqrt(x^2)`

**Entrada:** `sqrt(x^2)` (en ℝ)

* Identidad general: `sqrt(x^2) = |x|`
* Si el motor produce `x`, entonces:

  * **🔀 Branch choice:** “asumir `x ≥ 0`”
* Alternativa sin branch: producir `|x|`.

---

### 6.5 Logaritmos: identidad condicional (estrecha dominio)

**Entrada:** `log(a*b)` (en ℝ)
**Requires (input):** `a*b > 0`

Si el motor aplica `log(a*b) → log(a)+log(b)`, entonces:

* **ℹ️ Requires (introduced):** `a>0`, `b>0`
  (no es branch choice: es restricción más fuerte que `ab>0`)

---

### 6.6 Trig inversa: rango principal

**Entrada:** `sin(arcsin(x))` (en ℝ)

* La regla usa inversa principal:

  * **🔀 Branch choice:** “arcsin devuelve valores en [-π/2, π/2]”
* Además, `arcsin(x)` requiere `x ∈ [-1,1]` (input requires si aparece `arcsin(x)`).

---

### 6.7 Extensión a complejos

**Entrada:** `(-1)^(1/2)`
En ℝ no está definido.

Si el motor devuelve `i`:

* **🧿 Domain extension:** ℝ→ℂ

---

## 7. Reglas para autores de nuevas reglas (checklist)

### 7.1 ¿Debo emitir required_conditions?

Sí, si el paso:

* introduce denominadores o cancelaciones donde se necesita `≠0` y no estaba en el input,
* aplica una identidad que solo vale bajo una condición (y quieres que sea formalmente correcta).

### 7.2 ¿Debo emitir assumption_events?

Sí, si el paso:

* elige una rama (sqrt/log/potencias fraccionarias, inversas trig),
* usa una heurística no garantizada,
* extiende dominio.

### 7.3 ¿Cuándo NO debo emitir assumption_events?

Cuando la condición ya está cubierta por:

* `infer_implicit_domain(input)`
* o por requires ya introducidos en pasos anteriores.

---

## 8. Presentación en UI (recomendación)

* Al inicio/final: **ℹ️ Requires (input)**
* En pasos: solo **Requires introduced** cuando aparezcan
* Separado: **⚠️ Assumes**, **🔀 Branch choice**, **🧿 Domain extension**
* Modo verbose: permitir ver también eventos `DerivedFromRequires` para debug.

---

## 9. Preguntas para revisión matemática

1. ¿Es correcta la clasificación `log(ab)→log(a)+log(b)` como “requires introduced” (no branch)?
2. ¿En ℝ, debería preferirse `sqrt(x^2)=|x|` por defecto para evitar branch choice?
3. En trig inversa, ¿cómo se debe documentar el uso de rango principal para identidades tipo `sin(arcsin(x))`?
4. ¿Cuándo es aceptable una `HeuristicAssumption` en un CAS didáctico (modo agresivo vs modo estricto)?
