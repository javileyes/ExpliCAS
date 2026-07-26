# C1.8 — Diseño de `SubStep::checked`

*Basado en el inventario de 410 emisores + medición propia sobre el árbol (`crates/cas_didactic/src`, tests excluidos). Todas las cifras son reproducibles con los greps citados; ninguna es estimación de despacho.*

---

## 0. Lo que la medición añade al inventario (y corrige del plan)

Conté los **puntos de emisión reales**, no las llamadas literales a `SubStep::*`:

| Vía | Sitios | Qué recibe |
|---|---|---|
| `concrete_expr_substep` (def. 8654) · `temp_ctx_substep` (7715) · `mixed_ctx_substep` (7726) | **95** | `(&Context, ExprId, ExprId)` — **nodo vivo** |
| `formula_substep` (def. 8642) + `number_theory_substep` (4555) | **88** | 5 cadenas, **nunca** un nodo |
| `SubStep::keyed` / `SubStep::new` directos | **238** | 2 cadenas |
| **Total en `cas_didactic`** | **≈422** | |

Provenance del lado `after` en los 329 puntos que no pasan por los tres helpers de nodo:

```
after = display_expr/human_expr inline .......... 131
after = variable ligada a un render de nodo ......  28   (migración mecánica)
after = format! ..................................  67
after = literal de cadena ("R·sin(u+φ)", "0", "") .  27
after = variable no resoluble (render_*_plain…) ..  76
```

**Tres hechos que cambian el diseño:**

1. **El contador `substep_unchecked_emitters = 126` del plan (§7.2) es `grep -c 'SubStep::new' focused_rule_substeps.rs`.** Está infra-anclado 3,3×. Si nace en 126 puede llegar a 0 con 300 emisores sin verificar. **Hay que re-anclarlo a 422 menos lo migrado.**
2. **39 de los 56 sitios de `concrete_expr_substep` pasan el par propio del `Step`** (`before, after`). Ahí `Equality` es cierta *porque la produjo el motor*: verificarla no caza nada. El que miente es el TÍTULO. Sin un contador que lo separe, C1.8 se auto-engaña exactamente igual que el frente E.
3. `SubStep` de `cas_didactic` (422 sitios) es un tipo **distinto** de `cas_solver_core::step_types::SubStep` (29 sitios, `derive_command.rs` y solve) y de `cas_engine::step::SubStep`. C1.8 toca **uno de los tres**; el plan ya lo dice en §579-581 y conviene no perderlo de vista al reportar cobertura.

**El prior de calibración** (auditoría, §710, prototipo de guard tipado sobre los 214 sub-pasos que el corpus emite de verdad): **45 OK · 80 MISMATCH (~51 de ellos relaciones legítimas no-igualdad) · 52 ABSTAIN · 36 NOEVAL · 1 TIMEOUT.** Es decir: si se activa un verificador de `Equality` sobre todo, **desaparece más de la mitad de la narración correcta**. Ese número manda sobre el alcance del ciclo.

---

## 1. El enum definitivo

Regla de admisión, y es la que da el diseño: **un brazo entra en el enum solo si su verificador entra en el MISMO commit.** Un brazo sin verificador es un hueco con nombre bonito — la clase que C1.8 existe para cerrar.

```rust
// crates/cas_didactic/src/didactic/types/substep/claim.rs  (fichero nuevo)

/// Lo que un sub-paso AFIRMA. El constructor lo decide antes de publicar.
pub enum Claim {
    /// before ≡ after como expresiones. 213 declaradas en el inventario.
    Equality,
    /// before − after es una CONSTANTE (dos primitivas difieren en C).
    /// Precedente vivo: focused_rule_substeps.rs:13258-13275.
    EqualityUpToConstant { var: String },
    /// after = d(before)/dvar.
    Derivative { var: String },
    /// d(after)/dvar = before. 27 sitios; 18 solo en 14500..19000.
    Antiderivative { var: String },
    /// after = F(upper) − F(lower) con F = before. EXIGE los límites como DATO.
    DefiniteEval { var: String, lower: ExprId, upper: ExprId },
    /// after = before|_{var := point}. Sustituye a `Substitution`.
    EvalAt { var: String, point: ExprId },
    /// after = op(before). op ∈ {Sqrt, Ln, Abs, Neg, Recip}.
    Applied { op: AppliedOp },

    /// ABSTENCIÓN HONESTA: el sub-paso no afirma una relación entre dos lados.
    /// Estructuralmente UNA SOLA LÍNEA (ver `Sides::Statement`).
    Statement,

    /// DEUDA DECLARADA: no se puede verificar todavía. Contada, con techo.
    Unchecked(UncheckedReason),
}

pub enum UncheckedReason {
    RenderedString,  // el after no existe como nodo (format!/render_*_plain)
    Schema,          // identidad-esquema con metavariables libres (u, A, B, f)
    Enumeration,     // lista con comas o serie con elipsis
    EqualityChain,   // uno o más '=' incrustados en el hueco
    LimitValue,      // subsistema de límites (falta el brazo `Limit`)
    DomainVerdict,   // after = undefined
    VectorOp,        // jacobiano/hessiano/divergencia/rotacional/potencial
    Focus,           // señalamiento / restricción a una subexpresión
}
```

### Qué se quita y por qué

- **`Substitution` — FUERA.** Los cinco inventariadores coinciden: *no aparece ni una vez* como «mismo valor bajo cambio de variable declarado». Los candidatos naturales (`usub.identify_u_du` 12441, «u = …» 11245/11298) **declaran** el cambio, no lo aplican → `Statement`. Los dos que sí evalúan (21104, 21249) son `EvalAt`. Un brazo con clientela cero es un imán de mal uso.

### Qué se añade y con qué respaldo numérico

- **`EqualityUpToConstant`** — el teorema real de toda la familia integral (7 sitios: 13307, 13368, 13615, 13680, 13789, 13851, 13927). El código **ya la implementa a mano** en 13258-13275 y su gemelo 13368 publica las mismas claves **sin gate**: generalizar el chequeo y borrar la copia artesanal es el ciclo entero en miniatura.
- **`EvalAt`** — 2 sitios, verificador de 6 líneas (`substitute_expr_by_id` + simplify).
- **`Applied`** — 5 sitios (10874, 11027, 11105, 11127, 11133). **Rescata una mentira P0 sin borrar narración**: «1 − x² ⇒ sqrt(1 − x²)» es falso como `Equality` y **cierto y verificable** como `Applied{Sqrt}`. Coste: ~15 líneas.

### Qué NO entra en C1.8, con su cuenta y su motivo

| Relación pedida por el inventario | Sitios | Por qué se aplaza |
|---|---|---|
| `Limit { var, point, at_infinity, side }` | 9 | Su verificador es el **oráculo de límites del propio motor**, que hoy no es invocable desde la capa didáctica sin reentrada. Ciclo propio. Mientras: `Unchecked(LimitValue)`. |
| `SchematicIdentity { plantilla, ligaduras }` | ~43 con ligadura declarada («con u = x»), ~30 sin ligar | Exige **parser de plantillas + matcher contra el par**. Es el segundo ciclo más grande de la campaña, no un rider. Mientras: `Unchecked(Schema)`. |
| `Gradient / VectorOp / Potential` | 5 | Sus `after` son `format!("[{join}]")`: **no hay nodo que verificar**. Migrar el render es prerequisito. `Unchecked(VectorOp)`. Excepción: `gradient.component` (12349) **es** `Derivative{var}` con la variable ya en `args[0]` → entra hoy. |
| `DomainVerdict` | 7 | Decidible (`calculus_domain_support::{positive_condition_is_impossible_over_reals, log_base_is_invalid_over_reals, bounded_inverse_real_domain_rejection_over_reals, nonfinite_or_undefined_constant}`), pero el emisor tendría que **devolver el testigo**, no solo llamar al predicado. Barato, pero es refactor de 7 emisores. `Unchecked(DomainVerdict)`. |
| `EqualityChain`, `Enumeration/Series`, `Bound`, `IntervalSplit`, `IndeterminateForm`, `LimitPreserving`, `DifferentialPair`, `LinearityOf` | ~30 | Cada uno pide maquinaria propia. Todos van a `Unchecked(<reason>)` **con su motivo separado**, para que cada tanda futura drene UNA razón y el techo baje de forma legible. |

### Dos ejes que NO son brazos del enum (y por qué el inventario se equivocaría al pedirlos así)

**`ConditionalEquality` → campo, no brazo.** Aparece en ≥14 sitios (3158, 2613, 2677, 10046, 10091, 10188, 10547, 10593, 10632, 10738, 10846, 11252, 11270, 14494). Su verificador es *el mismo* que el de `Equality`; lo que cambia es que la hipótesis debe **viajar como dato** y publicarse. Por eso:

```rust
pub struct SubStepClaim {
    pub relation: Claim,
    pub scope: Scope,                                    // Whole | Subexpr
    pub under: Vec<cas_solver_core::domain_condition::ImplicitCondition>,
}
```

En C1.8 `under` **solo se publica**, no relaja el verificador: una igualdad condicionada que no se prueba incondicionalmente **declina** (`acosh(cosh(u)) = u` debe declinar: es falsa, vale `|u|`). Es la respuesta conservadora y es la correcta.

**`Scope` → campo, no brazo.** 10242 (solo el denominador), 10298 (solo el numerador), 8200/8209 (numerador/denominador del cociente) publican `Equality` **cierta de subexpresión**; 10662 publica una **falsa** porque se traga el `+7`. La misma dimensión resuelve las dos cosas: `Scope::Subexpr` es legal y verificable, y hace que el invariante de cadena (C1.9) no marque como rotos los saltos de foco **declarados**. Sin `Scope`, o se rechaza narración correcta o no se caza 10662.

---

## 2. La firma, y cómo convive con `keyed`

El bug de l.4089/4464 (argumentos LaTeX intercambiados) **no es una relación falsa**: es que `formula_substep(desc, before, after, before_latex, after_latex)` acepta cuatro cadenas independientes. Se cierra con el TIPO, no con el verificador:

```rust
// crates/cas_didactic/src/didactic/types/substep/methods.rs

pub struct Rendered { pub plain: String, pub latex: Option<String> }

pub enum Sides<'a> {
    /// Camino sano: el constructor RENDERIZA los cuatro campos él mismo
    /// (display_expr/latex_expr, focused_rule_substeps.rs:22342/22352).
    /// El swap plano↔latex es INEXPRESABLE.
    Nodes {
        before_ctx: &'a Context, before: ExprId,
        after_ctx:  &'a Context, after:  ExprId,   // 14374 y 1426 emparejan ctx distintos
    },
    /// Solo construible con Claim::Unchecked. Cada lado viaja EMPAREJADO:
    /// plano y latex del MISMO lado en el mismo argumento.
    Rendered { before: Rendered, after: Rendered },
    /// Solo construible con Claim::Statement. UNA sola línea: dos huecos
    /// desparejados (4679: «φ(12)» vs «12 = 2^2·3») dejan de ser expresables.
    Statement { line: Rendered },
}

impl SubStep {
    /// Publica el sub-paso SOLO si la relación declarada queda PROBADA.
    /// `None` = declinado; la decisión queda registrada en el tally.
    pub fn checked(
        key: &'static str,
        args: Vec<String>,
        claim: SubStepClaim,
        sides: Sides<'_>,
    ) -> Option<SubStep>;

    /// Igual, para los ~122 emisores de título literal sin clave i18n todavía
    /// (77 de 84 no están en `description_en`: eso es C5.2, no C1.8).
    pub fn checked_titled(
        title: impl Into<String>,
        claim: SubStepClaim,
        sides: Sides<'_>,
    ) -> Option<SubStep>;
}
```

El acoplamiento `Claim` ↔ `Sides` se hace cumplir en el constructor (no en el tipo, para no explotar en genéricos): `Nodes` con `Unchecked` es legal (deuda que ya tiene nodo), `Rendered` con cualquier cosa que no sea `Unchecked` **es un panic en debug y una declinación en release**. Eso es lo que hace la clase imposible: *no se puede afirmar `Equality` sobre cadenas*.

### Invariantes de buena formación (se aplican a TODOS los brazos, incluido `Unchecked`)

Son gratis y cazan cuatro clases ya documentadas:

1. **No vacío** — mata l.1890 (`before=""`, `after=""`).
2. **`before_plain != after_plain` salvo `Statement`** — mata 12231 (`potential.reconstruct` publica `display_expr(after)` en los dos huecos) y cierra `E8_substep_noop` (5 → 0).
3. **Sin `=` en un hueco de expresión** → obliga a declarar `Unchecked(EqualityChain)`. Mata estructuralmente los 7 emisores de teoría de números (4631, 4657, 4679, 4684, 4706, 4727, 4751).
4. **Sin `,` de lista ni `…` ni prosa (`u = `, `opuesto = `)** → obliga a `Unchecked(Enumeration)` o `Statement`. Cubre 3502/3509, 3997, 4162, 10711, 10880, 11018.

### La pregunta directa: ¿`keyed` pasa a ser `checked(Statement, …)`?

**No, y es un punto de diseño importante.** `keyed`/`new` pasan a ser envoltorios de

```rust
checked(key, args, Claim::Unchecked(RenderedString), Sides::Rendered { … })
```

`Statement` es una **afirmación matemática** («este sub-paso no afirma una igualdad»). Mapear 300 emisores no migrados a `Statement` sería **blanquear deuda como honestidad**: el contador de deuda nacería en 0 y la campaña se declararía completa otra vez sobre su propia métrica — el fallo de método exacto que produjo esta auditoría.

### ¿`#[deprecated]`?

**No en C1.8.** 238 sitios directos generarían 238 warnings y la reacción sería un `#![allow(deprecated)]` que apaga la señal. En su lugar, el idioma que el repo ya usa: **un lint de presupuesto**, `scripts/lint_substep_unchecked_budget.sh` (molde literal de `scripts/lint_budget_enforcement.sh`), que cuenta los sitios `Claim::Unchecked(` + `SubStep::keyed(` + `SubStep::new(` y **falla si sube**. Un número que solo baja es mejor gobierno que un atributo que todo el mundo silencia. `#[deprecated]` se pone cuando el techo esté por debajo de ~40, y ahí sí duele lo justo.

---

## 3. El verificador, relación por relación

Todo con maquinaria que ya existe. Rutas exactas:

| Claim | Verificador | Maquinaria |
|---|---|---|
| `Equality` | escalera de 4 peldaños (abajo) | `focused_rule_substeps.rs:1186` `simplify_expr_in_context`, `:1196` `expr_is_zero_in_context`, `cas_math/src/poly_compare.rs:29` `poly_eq`, `cas_math/src/multipoly/conversion.rs:24` `multipoly_from_expr` |
| `EqualityUpToConstant{var}` | `simplify(before − after)` debe ser `Expr::Number` | copiado de `focused_rule_substeps.rs:13258-13275` (que además ya excluye `expr_contains_integrate_call`, `:15199`) |
| `Derivative{var}` | `d := differentiate_symbolic_expr(&mut scratch, before, var)?` → `Equality(d, after)` | `cas_math/src/symbolic_differentiation_support.rs:6392` |
| `Antiderivative{var}` | `d := differentiate_symbolic_expr(&mut scratch, after, var)?` → `Equality(d, before)`. Mata la constante de integración sola. Pre-gate: `expr_contains_integrate_call(after)` ⇒ Undecided | ídem + `:15199` |
| `DefiniteEval{var,l,u}` | `F(u) − F(l)` con `substitute_expr_by_id`, simplify, `Equality` contra `after`. **Y el constructor renderiza el `before` CON los límites** | `cas_ast/src/traversal.rs:201` |
| `EvalAt{var,point}` | sustituir + simplify + `Equality` | ídem |
| `Applied{op}` | estructural: `after == Function(op,[before])`, si no `Equality(after, op(before))` | `Context::add` |
| `Statement` | solo buena formación (un lado) | — |
| `Unchecked(_)` | solo buena formación | — |

### La escalera de `Equality` y **la regla que impide que el carril se auto-invalide**

```
1. before == after (mismo ExprId)                      → PROBADA (y marcada Trivial)
2. ambos Expr::Number                                  → PROBADA o REFUTADA
3. multipoly_from_expr(a) y (b) ambos Ok               → PROBADA o REFUTADA
4. expr_is_zero_in_context(scratch, Sub(a,b))          → PROBADA
5. cualquier otra cosa                                 → UNDECIDED
```

**`REFUTADA` exige una refutación POSITIVA, jamás la ausencia de prueba.** Dos trampas concretas que encontré leyendo el código:

- **`poly_eq` devuelve `false` tanto si difieren como si no convierten** (`poly_compare.rs:29`: `Err(_) => return false`). Usarlo como refutador convertiría `sin(x)` en «desigual». Por eso el peldaño 3 llama a `multipoly_from_expr` directamente y **solo refuta si las DOS conversiones fueron `Ok`**.
- **La atomización de funciones no es un refutador sano**: si se abstrae `sin(x)` y `cos(x)` como variables independientes, `sin²+cos²=1` sale «refutada». Descartado.

Consecuencia honesta y deliberada: **las tres mentiras trigonométricas P0 (7609, 7659 con el `4` hardcodeado) NO son refutables automáticamente** — caen en `UNDECIDED`. Y aun así **dejan de publicarse**, porque `UNDECIDED ⇒ declina`. La soundness la da la declinación, no la refutación. La refutación solo sirve para **señalar el bug con el dedo**, que es justo lo que debe ser una aserción dura.

Y el caso que valida la regla, documentado por el propio autor en l.21514-21520 (`limit.combine_over_common_denominator`): una igualdad **cierta por construcción** que el simplificador **no** pliega a 0. Con el peldaño 5 cae en `UNDECIDED` y declina. Correcto y conservador; se recupera cuando se cablee un normalizador de fracciones (`together`/`ratsimp`), no antes.

---

## 4. Cuando la verificación NO decide

**Política: declinar (no publicar) y contar.** Un `Equality` publicado es un `Equality` **probado**. Cuatro contadores, dos regímenes:

| Contador | Régimen | Definición |
|---|---|---|
| `substep_checked_failures` | **Aserción dura = 0** | Claims **REFUTADAS** (disproof positivo) que **no** están en `SUBSTEP_CLAIM_QUARANTINE`. Nombre ya reservado en el plan §7.1. |
| `substep_claim_undecided` | **Techo decreciente** | Claims que declinan por indecidibilidad. Nace **medido** en el shadow run, desglosado por `(desc_key, Claim)`. |
| `substep_unchecked_emitters` | **Techo monótono** | Conteo **estático** de sitios `Claim::Unchecked(`, por `UncheckedReason`. **Re-anclado de 126 a 422 − migrados.** |
| `substep_claim_verified` | **Suelo (mínimo)** | Claims probadas y **no** triviales. |
| `substep_claim_trivial` | Descriptiva | `Equality` cuyo par es el propio `(step.before, step.after)`: **39 sitios**. |

Los dos últimos son la lección de `min_expected` del gate de divergencia y de `generated_substeps_emitted` del plan (§492): **un verificador que declina todo pasa verde por vacío**, y un verificador que solo aprueba tautologías del motor pasa verde por trivialidad. Sin `substep_claim_verified >= N` y sin `substep_claim_trivial` a la vista, C1.8 se puede declarar completo sin haber probado nada — que es literalmente lo que le pasó al frente E.

**Cuarentena.** Molde literal de `crates/cas_cli/tests/steps_divergence_gate_tests.rs:98` (`QUARANTINE`) y su propiedad clave de `:107`: **auto-invalidante**. Una entrada que deja de refutar **rompe el gate** y obliga a borrarla. Nace con las refutaciones/declinaciones conocidas, cada una con dueño y ciclo:

```rust
const SUBSTEP_CLAIM_QUARANTINE: &[KnownClaimFailure] = &[
  // 7609/7659: coeficiente 4 hardcodeado (build_pythagorean_high_power_*). Dueño: tanda F-trig.
  // 10662: foco no declarado se traga el `+7`. Se cierra con Scope::Subexpr en C2.1.
  // 12231: potential.reconstruct pasa `after` en los dos huecos. Dueño: C1.8 (regla 2).
  // 13851: by_parts.integrate_remaining, ∫2e^x dx ≠ exp(x)·(x²+2−2x). Dueño: C1.6.
];
```

Y la propiedad que hace que la cuarentena **no sea una excusa**: **cuarentenado = declinado, nunca publicado.** La entrada existe para que el carril esté verde mientras el emisor se reescribe, no para seguir mintiendo con permiso. Ninguna de las 8 mentiras confirmadas sobrevive al ciclo, se arreglen o no sus emisores.

**Presupuesto de tiempo.** La auditoría ya vio **1 TIMEOUT** en su prototipo. Cada claim lleva un tope duro; agotarlo ⇒ `UNDECIDED`, jamás `REFUTED`. Y la memoria del repo es explícita: *el time-budget del gate no puede depender del perfil* (lección de la tanda-4 de EDOs).

---

## 5. EL PROBLEMA GORDO: el `after` que no existe como nodo

**Cifra dura: de 422 puntos de emisión, ~254 tienen un `after` que es render fiel de un `ExprId`; los otros ~170 no lo tienen** (67 `format!` explícitos + 27 literales + la mayoría de las 76 variables no resolubles, que son `*_plain`/`*_display` fabricados por constructores de cadenas). El inventario, contando distinto, llega al mismo sitio: **92 «necesita_contexto» + 133 «no_verificable» = 225 de 410 (55 %)**.

**Sí hay patrón, y es doble:**

1. **Un solo helper concentra la deuda**: `formula_substep` (def. 8642) — **81 sitios + 7 vía `number_theory_substep`**. Sus cinco parámetros son `&str`. Solo **dos** de sus 81 llamantes (7341 triple ángulo hiperbólico, 8416 extracción de cuadrado perfecto) renderizan `ExprId` reales y los tiran en la frontera.
2. **Los structs de traza guardan `String` en vez de `ExprId`**: `PolynomialDerivativeCofactorTrace` (17977), `SqrtChainCofactorTrace` (20376), `TrigLogPolynomialCofactorTrace`, `HyperbolicLogTableMatch`, `HyperbolicReciprocalTableMatch`, `LogPowerProductTableMatch`, `PolynomialBaseTableMatch`, `NestedInversePolynomialTableMatch`, `ArctanSqrtVarTableMatch`. **Para esos ~11 sub-pasos no hay que cambiar el constructor: hay que subir `ExprId` a los traces.** Caso más barato documentado: 20792, donde el plan calcula la derivada como `ExprId` en 20838-20843 y la convierte a `String` en la línea siguiente.

**Camino: triaje en tres, con contador propio por rama.** No hay respuesta única; forzarla sería deshonesto.

| Rama | Sitios (est.) | Qué se hace | Coste |
|---|---|---|---|
| **(a) MIGRAR** — la variable ya contiene un render de un nodo vivo | ~28 medidos + los 27 «necesita_contexto» que el inventario marca como mecánicos (530, 1622, 2362, 2369, 2613, 2620, 2677, 2684…) | pasar el `ExprId` en vez de la cadena | horas; cero riesgo de render |
| **(b) RECONSTRUIR** — el `format!` ensambla algo matemáticamente real: `(L)^2−(R)^2`, `f·(t₁)+f·(t₂)`, `F(b)−F(a)`, `{a}^3+{b}^3` | ~67 | construir el nodo con `ctx.add` y dejar que `checked` lo renderice | 1-2 ciclos. **Y aquí muere por construcción toda la clase de paréntesis**: 4106 («(3 − 1 · 1 + 7)/(3·7) = 16/21», que vale 9/21), 11499, 1806, 1844, 3151. **Cambia el render ⇒ churn en el fixture de sub-pasos: hay que declararlo, no descubrirlo.** |
| **(c) NUNCA MIGRAR** — no son expresiones | ~75 | quedan `Statement` o `Unchecked(Schema\|Enumeration\|EqualityChain)` **para siempre** en lo que respecta a `Equality` | 0. Su mejora honesta es **otro brazo** (`SchematicIdentity` con matcher obligatorio), no una migración |

**Respuesta directa a «¿se mide y baja por tandas?»: sí, pero por RAZÓN, no en bloque.** `substep_unchecked_emitters` desglosado por `UncheckedReason` es lo que hace que el techo sea legible: una tanda drena `RenderedString`, otra drena `Schema`, otra `LimitValue`. Un único número agregado permitiría el trueque silencioso «migro 10 fáciles, dejo 10 difíciles» — el mismo fallo que la regla del fixture del plan (§«la lista solo encoge») ya previene por expresión.

---

## 6. El primer paso: qué migrar en C1.8 para que el contador nazca en CERO

**Paso 0 — Shadow run, ANTES de activar nada** (media jornada). Test `--ignored` con el molde de `input_associativity_pairs_inventory` (`steps_divergence_gate_tests.rs:537`): correr el corpus guardrail con el verificador en modo **observador** y volcar `Verified / Trivial / Refuted / Undecided` por `(desc_key, Claim)`. **El prior dice que un `Equality` global borraría ~60 % de la narración de igualdad; el shadow run dice exactamente cuánto y dónde.** Sin esta medida, elegir el subconjunto es adivinar — y elegirlo *después* de ver qué sale verde sería la trampa que la campaña ya sufrió.

**Subconjunto a activar (enforcing) en C1.8 — ~113 puntos de emisión:**

| Familia | Sitios | Por qué entra |
|---|---|---|
| `Antiderivative{var}` | **27** (14762, 14917, 14987, 15086, 15255, 15514, 15630, 15760, 15820, 15878, 15962, 17011, 17398, 17860, 18301, 18476, 18735, 18861 + 13379, 13521, 14381 + 6 de 11000..14500) | La cosecha más limpia del fichero: `before`/`after` son nodos, la variable está en `args[1]`, el verificador es una derivación. Caza 13851. |
| `Derivative{var}` | **6 emisores / ~25 claves** (12349, 12403, 12424, 12960, 18306, 18738) | En 12424 y 12960 el `after` **lo produjo** `differentiate_symbolic_expr`: verificar es confirmar, indecidibilidad ~0. |
| `Equality` sobre los 3 helpers de nodo, **excluidos los 39 pares triviales** | **56** (17 de `concrete_expr_substep` + 33 `temp_ctx` + 6 `mixed_ctx`) | Es donde la verificación **caza algo**: 7609, 7659, 10662, 5894, 5900, 8175 viven aquí. |
| `EqualityUpToConstant{var}` | **2** (13307 ya gateado, **13368 su gemelo sin gate**) | Transferir el gate artesanal es el ROI más alto del ciclo por línea escrita. |
| `Applied{op}` | **5** (10874, 11027, 11105, 11127, 11133) | Rescata una P0 **sin borrar narración**. |
| `DefiniteEval` | **2** (14500, 14794) | 14500 publica «∫ \|2x−1\| dx ⇒ 5/2» sin límites: el tipo **exige** `lower`/`upper` y el constructor los renderiza. |
| `EvalAt` | **2** (21104, 21249) | Verificador trivial. |

**Los ~310 restantes** pasan a `Claim::Unchecked(<reason>)` de forma mecánica (los envoltorios `keyed`/`new` lo hacen solos; solo hay que etiquetar la razón en los ~88 de `formula_substep` y en las familias de límites/vectorial/dominio).

**Por qué `substep_checked_failures` puede nacer en 0** — las refutaciones/declinaciones conocidas se reparten en tres cubos, ninguno de los cuales publica una mentira:

- **Re-declaradas a un brazo que verifica**: 10874, 11027, 11105, 11127, 11133 → `Applied`. ✔ verde de verdad.
- **Arregladas en el ciclo por buena formación** (1 línea cada una): 1890 (huecos vacíos), 12231 (`before == after`), 14500 (límites en el render). ✔
- **En `SUBSTEP_CLAIM_QUARANTINE`, declinando**: 7609, 7659, 10662, 13851. ✔ no publican; el gate se auto-invalida cuando su dueño las arregle.
- Y 4106/4464 (paréntesis y swap LaTeX) **quedan fuera del subconjunto** —son `formula_substep`— pero el **tipo `Rendered`** ya hace inexpresable el swap en cuanto esos sitios se toquen. Honestamente: en C1.8 **4106 sigue publicando `16/21` para una cadena que vale `9/21`**. Se cierra en la rama (b) del §5. Decirlo es parte del entregable.

### Coste real, sin adornos

- **Código nuevo**: `claim.rs` + verificadores + tally + gate ≈ **500-700 LOC**. Es el trozo fácil.
- **Migración de ~113 call sites**: mecánica pero tediosa, **1-1,5 jornadas**. Efecto L del plan, confirmado.
- **El coste oculto, y es el riesgo nº 1**: los emisores tienen `&Context` (inmutable) y **todos los verificadores necesitan `&mut`** (`simplify_expr_in_context` mete el ctx en un `Simplifier` por `mem::swap`; `differentiate_symbolic_expr` toma `&mut Context`). La salida sin tocar 113 firmas es que `checked` **clone el contexto a un scratch internamente** — lo que ya hacen a mano decenas de emisores (`let work = ctx.clone()`). Pero eso es **un clon de `Context` y una pasada de simplificador por sub-paso verificado**, ~113 veces por evaluación con `--steps on` en vez de ~10. **La capa didáctica puede pasar a dominar el coste de `--steps on`.** Mitigación por fases: (1) medir con `make engine-scorecard` **en el mismo ciclo**, comparando huellas por contadores y slots, nunca por timing (lección `scorecard-huella-latency-noise`); (2) si duele, un `ClaimScratch` **por `Step`**, no por sub-paso — lo cual sí toca firmas y es un ciclo aparte. Presupuestarlo ahora es más barato que descubrirlo a mitad.
- **Un `thread_local!` nuevo** en `cas_didactic` para el tally. `scripts/lint_no_solver_tls.sh` **solo escanea `cas_engine/src/solve_*`**, así que es legal — lo digo explícitamente para que nadie lo descubra en revisión y lo lea como un salto de valla.

### Lo que C1.8 NO consigue, dicho por delante

- No toca `cas_solver_core::step_types::SubStep` (29 sitios de solve/derive): segunda instalación, ya anotada en el plan §579-581.
- No cierra el **invariante de vector** (`substep[i].after ≡ substep[i+1].before`, último `after ≡ step.after`), que es lo único que caza la cadena 7616→7622 —cada par es cierto y el destino es falso—. Eso es C1.9, y **depende de que `Scope` exista**, que es por lo que `Scope` entra hoy aunque su verificador sea trivial.
- No verifica el **TÍTULO**. En los 39 pares triviales, la `Equality` la garantiza el motor y quien miente es «Usar tan(u)·cot(u) = 1» emitido **incondicionalmente** (8087) o «Usar tan(u) = (1−cos 2u)/sin 2u» emitido **en la rama que no reconoció ninguna variante** (7474). Esa clase entera necesita `NamedIdentity{lhs, rhs}` con matcher. **C1.8 la deja abierta, y `substep_claim_trivial = 39` es el número que impide olvidarlo.**

---

## ADENDA 2026-07-26 — la familia `Equality` sobre los helpers de nodo NO se migra (medido)

El §6 asignaba **56 puntos de emisión** de `Equality` sobre los tres helpers de
nodo (`concrete_expr_substep`, `temp_ctx_substep`, `mixed_ctx_substep`),
excluyendo los 39 pares triviales, y era el bloque más grande del subconjunto a
activar. **Se midió antes de migrar y el resultado lo cancela.**

**Método.** Sonda temporal dentro de los tres helpers: verificar `Claim::Equality`
sobre cada par (saltando los ExprId idénticos, que son triviales por
hash-consing) y contar refutaciones. Barrido sobre **860 expresiones**: las 210
de `web/examples.csv` más 400 de `identity_pairs.csv` como `diff(...)` y 250 de
`derive_pairs.csv` como `integrate(...)` — es decir, siguiendo la regla de C5.1
de no medir solo sobre la vitrina.

**Resultado: 0 refutaciones.** Ni una en 860 expresiones, en ninguno de los tres
helpers.

**Por qué, y por qué era predecible:** los pares que llegan a estos helpers son
reescrituras PRODUCIDAS POR EL MOTOR, y el motor preserva equivalencia por
construcción (es lo que garantizan sus propias suites metamórficas y el fuzz de
equivalencia). Verificar en la capa de display una igualdad que el motor ya
garantiza es trabajo duplicado en tiempo de render. El diseño ya lo intuía para
los 39 triviales («`Equality` es cierta porque la produjo el motor: verificarla
no caza nada»); la medida extiende esa conclusión **a los 56**.

**Y el defecto real de esta familia sigue ahí, pero es otro:** lo que miente en
estos sub-pasos es el **TÍTULO**, no el par. «Usar tan(u)·cot(u) = 1» emitido
incondicionalmente, «Usar tan(u) = (1−cos 2u)/sin 2u» emitido en la rama que no
reconoció ninguna variante. `Claim::Equality` no puede ver eso por construcción:
comprueba los dos lados, no la frase. Esa clase necesita `NamedIdentity{lhs, rhs}`
con matcher de plantillas, que el propio §1 ya aplaza como el segundo ciclo más
grande de la campaña.

**Consecuencia para el plan:** el subconjunto a activar baja de ~113 a **~57**
puntos de emisión, y `substep_unchecked_emitters` no debe contar los 56 como
deuda pendiente sino como **fuera de alcance por medida**. Lo que queda de valor:
`Applied` (5, rescata un P0), `DefiniteEval` (2), `EvalAt` (2) y las familias de
límites/vectorial/dominio que siguen aplazadas con su razón.

---

## ADENDA 2026-07-26 (b) — `Applied`, `DefiniteEval` y `EvalAt` migradas: el subconjunto queda AGOTADO

Con los 56 de `Equality` fuera por medida (adenda anterior), el §6 dejaba tres
familias vivas. Están migradas. Lo que la migración corrigió del inventario:

| Familia | §6 decía | Medido | Sitios reales |
|---|---|---|---|
| `Applied` | 5 (10874, 11027, 11105, 11127, 11133) | **4** | 11027 y 11105 son `Sqrt` (catetos del triángulo de referencia); 11127 y 11133 son `Ln` (cambio de base). **10874 NO es `after = op(before)`**: es `sin(arcsin(u)) ⇒ u`, identidad esquemática con metavariable libre — pertenece a `NamedIdentity`, no aquí. |
| `DefiniteEval` | 2 (14500, 14794) | **1 + 1 de otra cosa** | 14794 (`integral.evaluate_antiderivative_at_bounds`) sí es `F ⇒ F(b) − F(a)`. 14500 (`abs_linear`) **no es esa relación**: es el P0 de los límites ausentes, y se arregla con render. |
| `EvalAt` | 2 (21104, 21249) | **2** | Cierre de factoriza-cancela y cierre de la iteración de L'Hôpital. |

### Los tres brazos no valen lo mismo (medido, no estimado)

**`Applied` y `DefiniteEval` tienen poder de refutación CERO por construcción**
en sus sitios vivos: el emisor fabrica el `after` aplicando la misma operación
que el verificador rehace, así que el hash-consing prueba la afirmación sin
tocar el simplificador. No es un defecto del brazo — es que ahí no había nada
que cazar. Lo que compran:

- `Applied`: la **declaración**. `1 − x² ⇒ sqrt(1 − x²)` es falso como igualdad;
  sin declarar la relación, el invariante de cadena de C1.9 lo leerá como
  eslabón roto y cualquier barrido de igualdad lo borrará como narración
  incorrecta. Es exactamente el rescate que el §1 prometía, y el §6 tenía razón
  al meterlo — pero por el motivo estructural, no por el de detección.
- `DefiniteEval`: los **límites como dato**. El tipo los exige, y no llevarlos
  es la causa exacta del P0 de 14500.

**`EvalAt` es el único cuyos dos lados vienen de productores distintos** (la
forma reconstruida en la capa didáctica contra el oráculo del motor), y por
tanto el único que puede discrepar. Sobre 1267 expresiones no discrepó ni una
vez; que tiene dientes se prueba con un test que le pasa un valor falso y
comprueba que la cadena entera declina
(`factor_cancel_declines_when_the_substitution_misses_the_engine_value`).

### El P0 de 14500 se cierra con RENDER

`∫ |2·x − 1| dx ⇒ 5/2` equiparaba una integral **indefinida** a un número. Ahora
publica la suma que su propio título describe (`1/4 + 9/4 ⇒ 5/2`), verificable
por aritmética racional pura, con el signo de la orientación incorporado
(`∫_2^0` da `−(1/4 + 9/4) ⇒ −5/2`). Dos lecciones de render que salieron de ahí
y que valen para toda la rama (b) del §5:

1. **El árbol no conserva el orden de una suma.** `|G(r) − G(a)| + |G(b) − G(r)|`
   se renderiza con los operandos de `Add` reordenados. Si el orden de los
   sumandos es parte del mensaje, hay que emitir valores, no estructura.
2. **Una resta con sustraendo negativo es una divergencia plano↔LaTeX**:
   `2 − −1/4` en texto contra `2 + \frac{1}{4}` en LaTeX. Plegarla a suma cuando
   el sustraendo es negativo la mata en origen.

### Estado del subconjunto

**Agotado.** Emisores declarados acumulados: **24**. Sigue sin migrar, con su
motivo explícito:

- la pareja **inversa** del cambio de base (`ln(x) ⇒ x`), que afirma
  `before == op(after)` y no tiene brazo — no se ensancha `Applied` para que
  trague (lección de `hessian.row`);
- la rama de **extremo infinito** de `DefiniteEval`, cuyo `after` es notación de
  límite sin nodo detrás: `Unchecked(LimitValue)`, no una relación afirmada
  sobre una cadena;
- las familias que el §1 ya aplazaba: `Limit`, `SchematicIdentity` /
  `NamedIdentity`, `VectorOp`, `DomainVerdict`.

---

## ADENDA 2026-07-26 (c) — el brazo `Limit`: el motivo del aplazamiento era falso, y el bloqueo real era otro

El §1 aplazaba `Limit { var, point, at_infinity, side }` (9 sitios) con este
motivo: *«Su verificador es el oráculo de límites del propio motor, que hoy no es
invocable desde la capa didáctica sin reentrada»*.

**No hay reentrada.** `cas_didactic` ya depende de `cas_math`, donde vive
`limits_support::eval_limit_at_infinity`; el oráculo no llama nunca de vuelta a la
capa didáctica (el grafo de cargo lo prohíbe); y la narración se genera DESPUÉS
de que el motor haya terminado, sobre los `Step` ya emitidos. Llamarlo son dos
líneas.

**El bloqueo real era que el APPROACH no existía como dato.** `StepMeta` solo
guardaba `limit_point`, que se rellena únicamente para `Approach::Finite`, y las
dos infinitudes comparten un mismo `rule_name` (`"Evaluar límite en infinito"`).
La capa didáctica venía leyendo la dirección de un `contains("infinito")` sobre
esa cadena — es decir, **no podía distinguir `+∞` de `−∞`**.

### Exigir el dato destapó dos mentiras vivas en `−∞`

| Entrada | Publicaba | Por qué es falso |
|---|---|---|
| `limit((1+1/x)^x, x, -infinity)` | «Aplicar el límite notable: lím(x**→∞**) (1 + 1/x)^x = e» | El valor es `e` en ambos lados, pero el sub-paso justifica el límite pedido citando un teorema sobre **otro** límite. |
| `limit((1+1/x)^x, x, -infinity)` | «La base tiende a 1 y el exponente a **∞**» | El exponente ES la variable: tiende a `−∞`. |
| `limit(x^3/(x^2+1), x, -infinity)` | «**Numerador y denominador → ∞**» | Con grado impar el numerador tiende a `−∞`. La forma sigue llamándose `∞/∞`; lo falso es la frase. |

Se cierran llevando `Approach` en `StepMeta::limit_approach`, que el emisor del
eval rellena siempre. La regla general: **un sub-paso que afirma algo sobre un
límite necesita saber DE QUÉ límite habla**, y no había forma de que lo supiera.

### De los 9 sitios: 8 encontrados, y solo 3 son verificables de verdad

El noveno del inventario es `generate_limit_residual_substeps`, cuyos dos lados
son la MISMA expresión (reafirma el residual y sugiere el método): es
`Statement`, no `Limit`.

De los 8 restantes, **5 pasan el par `(before, after)` propio del `Step`** — es
decir, preguntarle al oráculo es pedirle que confirme su propia respuesta. Es
exactamente la familia `Equality` de la adenda (a): cobertura sin hallazgos. **No
se migran**, y la razón se mide, no se estima.

Los **3 con lado izquierdo RECONSTRUIDO** por la capa didáctica sí enfrentan dos
productores distintos, y son los que se migran:

| Sitio | Lado izquierdo | Veredicto del oráculo (medido) |
|---|---|---|
| `generate_limit_squeeze_substeps` | `\|uᵏ\|` reconstruido | **decide** → comprobación real |
| `generate_limit_common_denom_substeps` | fracción combinada | **decide** → comprobación real |
| `generate_limit_conjugate_substeps` | cociente racionalizado | **DECLINA** → `Undecided`, publica |

El tercero es el hallazgo interesante y queda fijado en un test: el oráculo
**decide `√(x²+x) − x` en `+∞` y declina el cociente racionalizado que la
narración construye a partir de él**. La ruta didáctica y la ruta del motor no
son la misma ruta. Por eso `Undecided` no puede tratarse jamás como refutación:
lo contrario borraría esa narración, que es correcta.

---

## ADENDA 2026-07-26 (d) — `SchematicIdentity`: el discriminador es el SISTEMA DE TIPOS, y el censo se prueba UNA vez

El §1 aplazaba `SchematicIdentity { plantilla, ligaduras }` como «el segundo
ciclo más grande de la campaña» porque exigía «parser de plantillas + matcher
contra el par». La mitad tabla-y-verdad está CERRADA en este ciclo, y tres de
las premisas del plan no sobrevivieron a la ejecución:

### 1. Ninguna heurística textual distingue una plantilla de una instancia

El primer intento detectó esquemas por texto («ambos lados mencionan una letra
metavariable») y se ahogó dos veces: `x^2 + a ⇒ integrate(a, x) + …` es la
expresión del USUARIO con un símbolo llamado `a`, indistinguible por texto de un
esquema sobre `a`; y `formula_substep` — que el §5 daba como la familia entera —
mezcla plantillas con teoría de números e instancias construidas con `format!`
(el volcado dio 379 «plantillas» para 98 llamadas).

**La solución es el tipo.** Una plantilla es una CONSTANTE del fuente, así que
`schema_substep(lhs: &'static str, rhs: &'static str, …)` deja que el
COMPILADOR haga la partición: una instancia toma prestado de un render
por-emisión y no puede vivir `'static`. Migración medida: 100 sitios de
llamada → **35 plantillas** (17 literales directas + 18 ligadas vía tuplas
`match`/helpers) y **65 instancias**, con la frontera dictada por errores de
borrow-checker, no por opinión.

### 2. El censo: 78 pares, adjudicados con refutador POSITIVO

Resueltas las indirecciones (tuplas `match`, helpers, dispatchers a dos
niveles), las 35 plantillas publican **78 pares distintos**:

| Estado | Pares | Qué significa |
|---|---|---|
| `Proven` | **68** | `lhs ≡ rhs` con metavariables LIBRES; el simplificador pliega la diferencia a 0 |
| `OpenUnproven` | 2 | ciertas en muestreo numérico; el simplificador no pliega (ángulo cuádruple `sin(4u)`) |
| `DerivedMetavar` | 2 | FALSAS como identidad libre (contraejemplo numérico): `R·sin(u+φ) ⇔ a·sin(u)+b·cos(u)` exige `R,φ` derivadas |
| `FunctionMetavar` | 2 | `f(-u) = ±f(u)` cuantifica sobre `f` bajo hipótesis de paridad |
| `DisplayNotation` | 4 | cadenas `log_b(c)` con subíndices: el parser no las acepta |

**La regla de refutación es la misma que la de los verificadores**: solo refuta
un testigo positivo (muestreo numérico con contraejemplo claro), jamás «el
simplificador no plegó». Sin esa regla, las dos filas de ángulo cuádruple —
identidades CIERTAS — habrían salido refutadas. Y dos gotchas de medición que
casi contaminan la tabla: `sin²(u)` debe normalizarse a `sin(u)^2` (no
`sin^2(u)`, que el parser lee como `sin^(2u)`), y al sustituir muestras en
`4u` el lookbehind no puede excluir dígitos o deja la variable libre.

### 3. Verificar la plantilla en runtime probaría lo mismo 213 veces por corpus

La verdad de una plantilla es una propiedad ESTÁTICA. Por eso:

- **La prueba corre UNA vez**: `every_proven_schema_folds_to_zero` (68 pliegues
  en ~4 s de test), no una por emisión con `--steps on`.
- **El runtime es un LOOKUP** (`Claim::SchematicIdentity` → `schema_status`),
  más una aserción de debug que revienta —nombrando el par exacto— si un emisor
  declara un esquema que el censo nunca adjudicó. Las suites y el gate corren
  en debug: la deriva se caza en el siguiente `cargo test`.
- **El pin es auto-invalidante en las DOS direcciones**: una fila `Proven` que
  deja de plegar rompe; una excepción que empieza a plegar rompe y exige su
  promoción en el mismo commit. Y `OpenUnproven` no puede cobijar mentiras:
  sus filas se re-verifican numéricamente en el propio test.

### Lo que queda de la otra mitad (y ya tiene el prerrequisito)

El matcher instancia↔plantilla — el que cazaría «Usar tan(u)·cot(u) = 1»
emitido en la rama equivocada — sigue siendo su propio ciclo, pero su entrada
ya existe: los 78 pares están en una tabla de datos con status, y los 35
emisores ya declaran qué plantilla afirman. Las 65 instancias de
`formula_substep` pertenecen a la familia del §5 (render sin nodo), no a esta.

### Hallazgos colaterales del ciclo, con dueño propio

- **67 sub-pasos publican LaTeX crudo en el hueco de texto plano**
  (`\sqrt{y} - 1` llega al lector de la CLI). Chip task_580326b0.
- **Un numerador aritméticamente falso** en la narración de denominador común
  (`c + x - b + x` donde va `(c+x) − (b+x)`: error de 2x), con formateador y
  `Context::add` ya exonerados por prueba directa. Chip task_cbce7fb9.
