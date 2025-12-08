# Tests Avanzados Pendientes - Trigonometría Inversa

Este documento describe los tests de trigonometría inversa avanzados que actualmente **no simplifican completamente** y requieren implementación de reglas adicionales.

## Estado Actual

| Test | Estado | Simplifica | Requiere |
|------|--------|------------|----------|
| **Test 46** | ⚠️ Pasa* | No | Evaluación numérica de funciones inversas |
| **Test 47** | ✅ **PASA** | **Sí** | ✅ **Ya funciona** |
| **Test 48** | ⚠️ Pasa* | No | Regla `atan(x) + atan(1/x)` con manejo de signo |
| **Test 49** | ⚠️ Pasa* | No | Fórmula de adición de arctan |
| **Test 50** | ✅ **PASA** | **Sí** | ✅ **Ya funciona** (2025-12-08) |

\* "Pasa" = No crashea, pero las assertions están desactivadas porque no simplifica al resultado esperado

**Actualización 2025-12-08**: Test 50 ahora completamente funcional gracias a la implementación de `CosArcsinExpansionRule` y `SinArccosExpansionRule` ✅

---

## Test 46: Principal Values (Valor Principal)

### Input
```text
asin(sin(3*pi/2)) + acos(cos(3*pi))
```

### Resultado Actual
```text
acos(-1) + asin(sin(3/2 * pi))
```

### Resultado Esperado
```text
pi/2
```

### ¿Por Qué No Funciona?

El test requiere que el CAS:
1. **Evalúe numéricamente** las funciones trigonométricas con argumentos concretos:
   - `sin(3π/2) = -1`
   - `cos(3π) = -1`
   
2. **Evalúe las funciones inversas** con valores numéricos:
   - `asin(-1) = -π/2`
   - `acos(-1) = π`

3. **Reduzca al dominio principal**: `asin(sin(θ)) = θ` solo si `θ ∈ [-π/2, π/2]`

### Implementación Requerida

**Archivo**: `crates/cas_engine/src/rules/inverse_trig.rs`

Necesitamos agregar reglas de **evaluación numérica**:

```rust
// Regla: asin(numeric_value) → resultado
// Ejemplos:
//   asin(-1) → -π/2
//   asin(0)  → 0
//   asin(1)  → π/2
//   asin(1/2) → π/6

// Regla: acos(numeric_value) → resultado
// Ejemplos:
//   acos(-1) → π
//   acos(0)  → π/2
//   acos(1)  → 0
//   acos(1/2) → π/3

// Regla: atan(numeric_value) → resultado
// Ejemplos:
//   atan(0)  → 0
//   atan(1)  → π/4
//   atan(-1) → -π/4
```

**Complejidad**: Media  
**Prioridad**: Media  
**Estimación**: 2-3 horas

---

## Test 48: Atan Reciprocal con Signo

### Input
```text
atan(2) + atan(1/2) - pi/2
```

### Resultado Actual
```text
atan(1/2) + atan(2) - 1/2 * pi
```

### Resultado Esperado
```text
0
```

### ¿Por Qué No Funciona?

La regla actual `InverseTrigAtanRule` implementa `atan(x) + atan(1/x) = π/2`, pero **no funciona con constantes numéricas** como `2` y `1/2`.

El problema es que `are_reciprocals()` helper no reconoce que `2` y `1/2` son recíprocos.

### Implementación Requerida

**Archivo**: `crates/cas_engine/src/rules/inverse_trig.rs`

Mejorar el helper `are_reciprocals()` para:

```rust
fn are_reciprocals(ctx: &Context, expr1: ExprId, expr2: ExprId) -> bool {
    // Caso actual: detecta 1/x y x
    // ...código existente...
    
    // NUEVO: Detectar números recíprocos
    // Si expr1 es Number(a) y expr2 es Number(b), verificar si a * b = 1
    if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(expr1), ctx.get(expr2)) {
        if let (Some(f1), Some(f2)) = (n1.to_f64(), n2.to_f64()) {
            return (f1 * f2 - 1.0).abs() < 1e-10;
        }
    }
    
    // También manejar fracciones exactas: 2 y 1/2, etc.
    // ...
}
```

**Complejidad**: Baja-Media  
**Prioridad**: Alta (relativamente fácil)  
**Estimación**: 1-2 horas

---

## Test 49: Machin Formula (Fórmula de Machin)

### Input
```text
4*atan(1/5) - atan(1/239) - pi/4
```

### Resultado Actual
```text
-atan(1/239) - 1/4 * pi + 4 * atan(1/5)
```

### Resultado Esperado
```text
0
```

### ¿Por Qué No Funciona?

Esta es la famosa fórmula de Machin para calcular π:
```
π/4 = 4·arctan(1/5) - arctan(1/239)
```

Requiere la **fórmula de adición de arcotangente**:
```
atan(a) + atan(b) = atan((a+b)/(1-ab))   (si ab < 1)
```

### Implementación Requerida

**Archivo**: Nuevo `crates/cas_engine/src/rules/inverse_trig_advanced.rs`

Necesitamos implementar:

```rust
// Regla: Adición de arctan
// atan(a) + atan(b) → atan((a+b)/(1-ab))
//
// Pasos para Test 49:
// 1. 2·atan(1/5) = atan((1/5 + 1/5)/(1 - 1/25))
//                = atan((2/5)/(24/25))
//                = atan(5/12)
//
// 2. 4·atan(1/5) = 2·atan(5/12)
//                = atan((5/12 + 5/12)/(1 - 25/144))
//                = atan((10/12)/(119/144))
//                = atan(120/119)
//
// 3. atan(120/119) - atan(1/239)
//    = atan((120/119 - 1/239)/(1 + 120/(119·239)))
//    = atan(1)
//    = π/4
```

**Complejidad**: Alta  
**Prioridad**: Baja (caso muy específico, más académico)  
**Estimación**: 4-6 horas

**Nota**: Este test es principalmente para демостrar capacidades avanzadas. No es crítico para uso general.

---

## Test 50: Composiciones Trigonométricas (Triángulo Algebraico)

### Input
```text
tan(asin(x))^2 - x^2/(1-x^2)
```

### Resultado Actual
```text
-(x^2 / (1 - x^2)) + sin(asin(x))^2 / cos(asin(x))^2
```

### Resultado Esperado
```text
0
```

### ¿Por Qué No Funciona?

El sistema ya convierte `tan(u) → sin(u)/cos(u)`, y reconoce `sin(asin(x)) = x`, pero **falta**:

1. **`cos(asin(x)) → sqrt(1 - x²)`**
2. **`sin(acos(x)) → sqrt(1 - x²)`**

Estas son identidades fundamentales basadas en el **triángulo rectángulo**:
- Si `θ = asin(x)`, entonces `sin(θ) = x` (opuesto/hipotenusa)
- Por Pitágoras: `cos²(θ) = 1 - sin²(θ) = 1 - x²`
- Por tanto: `cos(θ) = sqrt(1 - x²)` (tomando raíz positiva)

### Implementación Requerida

**Archivo**: `crates/cas_engine/src/rules/trig_inverse_expansion.rs`

Ya existe este archivo con reglas similares. Necesitamos agregar:

```rust
// Regla: cos(asin(x)) → sqrt(1 - x²)
define_rule!(
    CosAsinRule,
    "cos(asin(x)) = sqrt(1-x²)",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if name == "cos" && args.len() == 1 {
                if let Expr::Function(inner_name, inner_args) = ctx.get(args[0]) {
                    if inner_name == "asin" && inner_args.len() == 1 {
                        let x = inner_args[0];
                        // sqrt(1 - x²)
                        let one = ctx.num(1);
                        let x_squared = ctx.add(Expr::Pow(x, ctx.num(2)));
                        let one_minus_x_sq = ctx.add(Expr::Sub(one, x_squared));
                        let result = ctx.add(Expr::Function(
                            "sqrt".to_string(),
                            vec![one_minus_x_sq]
                        ));
                        return Some(Rewrite {
                            new_expr: result,
                            description: "cos(asin(x)) = sqrt(1-x²)".to_string(),
                        });
                    }
                }
            }
        }
        None
    }
);

// Regla: sin(acos(x)) → sqrt(1 - x²)
// (Simétrica a la anterior)
```

**También agregar soporte para `acos` (sin "arc")**:
```rust
if (inner_name == "asin" || inner_name == "arcsin") && inner_args.len() == 1 {
    // ...
}
```

**Complejidad**: Media  
**Prioridad**: Alta (útil para muchos casos)  
**Estimación**: 2-3 horas

### Tests Adicionales Sugeridos

Una vez implementadas estas reglas, agregar tests para:
- `sin(acos(x))`
- `tan(acos(x))`
- `tan(asin(x))`
- `cot(asin(x))`, etc.

---

## Resumen de Implementaciones Pendientes

### Prioridad Alta (Utilidad general)
1. ✅ **Test 47** - Ya implementado
2. 🔵 **Test 50** - Composiciones `sin(acos)`, `cos(asin)` (2-3h)
3. 🔵 **Test 48** - Mejorar `are_reciprocals()` para números (1-2h)

### Prioridad Media
4. 🔵 **Test 46** - Evaluación numérica de inversas (2-3h)

### Prioridad Baja (Casos específicos/académicos)
5. 🔵 **Test 49** - Fórmula de adición de arctan (4-6h)

**Tiempo total estimado**: 10-16 horas para implementar todas las funcionalidades

---

## Cómo Contribuir

Si quieres implementar alguna de estas mejoras:

1. **Elige un test** de prioridad alta
2. **Lee la sección correspondiente** en este documento
3. **Implementa la regla** en el archivo indicado
4. **Activa la assertion** en el test correspondiente (archivo `inverse_trig_torture_tests.rs`)
5. **Ejecuta** `cargo test --test inverse_trig_torture_tests`
6. **Verifica** que el test ahora pasa completamente

### Estructura de Archivos

```
crates/cas_engine/src/rules/
├── inverse_trig.rs               ← Reglas básicas de inversas
├── trig_inverse_expansion.rs     ← Composiciones trig(inverse_trig)
└── inverse_trig_advanced.rs      ← (Crear) Reglas avanzadas (Machin, etc.)

crates/cas_cli/tests/
└── inverse_trig_torture_tests.rs ← Tests 46-50
```

---

## Referencias

- **Test 46**: Domain restrictions, principal values
- **Test 48**: [Arctan reciprocal identity](https://en.wikipedia.org/wiki/Inverse_trigonometric_functions#Arctangent_addition_formula)
- **Test 49**: [Machin's formula](https://en.wikipedia.org/wiki/Machin-like_formula)
- **Test 50**: [Pythagorean identity](https://en.wikipedia.org/wiki/Pythagorean_trigonometric_identity)

---

**Última actualización**: 2025-12-08  
**Estado**: 1 de 5 tests avanzados completamente funcional (Test 47) ✅
