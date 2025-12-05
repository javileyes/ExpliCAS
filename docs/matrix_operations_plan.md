# Plan de Implementación: Operaciones Matriciales

## Estado Actual

ExpliCAS ya tiene soporte básico para matrices:

✅ **Completado:**
- Representación en AST: `Expr::Matrix { rows, cols, data }`
- Parser: puede leer matrices `[[1, 2], [3, 4]]`
- Display: puede mostrar matrices en formato legible
- LaTeX rendering para matrices

❌ **Falta:**
- Operaciones matemáticas (suma, multiplicación, etc.)
- Funciones matriciales (determinante, inversa, transpuesta)
- Reglas de simplificación
- Comando CLI para operaciones matriciales

---

## Fase 1: Operaciones Básicas

### 1.1 Suma y Resta de Matrices

**Archivo:** `crates/cas_engine/src/rules/matrix.rs` (nuevo)

**Funcionalidad:**
```rust
// A + B (mismas dimensiones)
[[1, 2], [3, 4]] + [[5, 6], [7, 8]] → [[6, 8], [10, 12]]
```

**Validación:**
- Verificar dimensiones coinciden
- Suma elemento por elemento

**Regla:** `MatrixAddRule`

### 1.2 Multiplicación Escalar

**Funcionalidad:**
```rust
// k * Matrix
2 * [[1, 2], [3, 4]] → [[2, 4], [6, 8]]
```

**Regla:** `ScalarMultiplyRule`

### 1.3 Multiplicación de Matrices

**Funcionalidad:**
```rust
// A (m×n) * B (n×p) → C (m×p)
[[1, 2], [3, 4]] * [[5, 6], [7, 8]] → [[19, 22], [43, 50]]
```

**Validación:**
- cols(A) == rows(B)
- Producto punto fila × columna

**Regla:** `MatrixMultiplyRule`

---

## Fase 2: Funciones Matriciales

### 2.1 Transpuesta

**Función:** `transpose(M)`

```rust
transpose([[1, 2], [3, 4]]) → [[1, 3], [2, 4]]
```

**Implementación:**
- Intercambiar rows ↔ cols
- data[i][j] → data[j][i]

### 2.2 Determinante

**Función:** `det(M)` o `|M|`

**Algoritmos por tamaño:**
- 1×1: `det([[a]]) = a`
- 2×2: `det([[a,b],[c,d]]) = ad - bc`
- 3×3: Regla de Sarrus o expansión por cofactores
- n×n: Expansión por cofactores (recursivo) o eliminación gaussiana

**Ejemplo:**
```rust
det([[1, 2], [3, 4]]) → -2
det([[1, 2, 3], [0, 1, 4], [5, 6, 0]]) → 1
```

### 2.3 Inversa

**Función:** `inv(M)` o `M^(-1)`

**Requisitos:**
- Matriz cuadrada
- det(M) ≠ 0

**Métodos:**
- 2×2: Fórmula directa
- n×n: Gauss-Jordan o matriz adjunta

**Ejemplo:**
```rust
inv([[1, 2], [3, 4]]) → [[-2, 1], [3/2, -1/2]]
```

### 2.4 Traza

**Función:** `trace(M)`

```rust
trace([[1, 2], [3, 4]]) → 5  // 1 + 4
```

---

## Fase 3: Operaciones Avanzadas

### 3.1 Rango

**Función:** `rank(M)`

- Eliminación gaussiana
- Contar filas no nulas

### 3.2 Forma Escalonada

**Funciones:**
- `rref(M)`: Reduced Row Echelon Form (Gauss-Jordan)
- `ref(M)`: Row Echelon Form

**Uso educativo:** Resolver sistemas de ecuaciones

### 3.3 Valores y Vectores Propios

**Funciones:**
- `eigenvalues(M)`: Valores propios
- `eigenvectors(M)`: Vectores propios

**Complejidad:** Requiere resolver polinomio característico det(M - λI) = 0

---

## Fase 4: Integración CLI

### 4.1 Sintaxis de comandos

```text
> [[1, 2], [3, 4]] + [[5, 6], [7, 8]]
Result: [[6, 8], [10, 12]]

> det([[1, 2], [3, 4]])
Result: -2

> transpose([[1, 2, 3], [4, 5, 6]])
Result: [[1, 4], [2, 5], [3, 6]]

> [[1, 2], [3, 4]] * [[5, 6], [7, 8]]
Result: [[19, 22], [43, 50]]
```

### 4.2 Modo Educativo

```text
> explain det([[1, 2], [3, 4]])
Educational Steps:
────────────────────────────────────────
Calculando determinante de matriz 2×2
Usando fórmula: det = ad - bc
donde a=1, b=2, c=3, d=4
det = (1)(4) - (2)(3)
det = 4 - 6
det = -2
────────────────────────────────────────
Result: -2
```

---

## Estructura de Código

### Nuevo módulo: `matrix.rs`

```rust
// crates/cas_engine/src/matrix.rs

pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<ExprId>,
}

impl Matrix {
    // Constructores
    pub fn from_expr(ctx: &Context, id: ExprId) -> Option<Self>;
    pub fn to_expr(&self, ctx: &mut Context) -> ExprId;
    
    // Operaciones básicas
    pub fn add(&self, other: &Self, ctx: &mut Context) -> Option<Self>;
    pub fn mul(&self, other: &Self, ctx: &mut Context) -> Option<Self>;
    pub fn scalar_mul(&self, scalar: ExprId, ctx: &mut Context) -> Self;
    
    // Operaciones matriciales
    pub fn transpose(&self) -> Self;
    pub fn determinant(&self, ctx: &mut Context) -> Option<ExprId>;
    pub fn inverse(&self, ctx: &mut Context) -> Option<Self>;
    pub fn trace(&self, ctx: &mut Context) -> Option<ExprId>;
    
    // Avanzadas
    pub fn rref(&self, ctx: &mut Context) -> Self;
    pub fn rank(&self, ctx: &mut Context) -> usize;
}
```

### Reglas de simplificación

```rust
// crates/cas_engine/src/rules/matrix.rs

define_rule!(MatrixAddRule, "Matrix Addition", |ctx, expr| {
    // Detectar Add(Matrix, Matrix)
    // Verificar dimensiones
    // Sumar elemento a elemento
});

define_rule!(MatrixMultiplyRule, "Matrix Multiplication", |ctx, expr| {
    // Detectar Mul(Matrix, Matrix)
    // Verificar compatibilidad
    // Multiplicar matrices
});

define_rule!(ScalarMatrixRule, "Scalar Matrix Multiplication", |ctx, expr| {
    // Detectar Mul(Number, Matrix) o Mul(Matrix, Number)
    // Multiplicar cada elemento
});
```

---

## Plan de Implementación

### Sprint 1: Fundamentos
1. Crear módulo `matrix.rs`
2. Implementar `Matrix::add()` y `Matrix::scalar_mul()`
3. Crear reglas `MatrixAddRule` y `ScalarMatrixRule`
4. Tests básicos

### Sprint 2: Multiplicación
1. Implementar `Matrix::mul()`
2. Crear `MatrixMultiplyRule`
3. Tests de multiplicación (casos especiales: identidad, cero)

### Sprint 3: Funciones Matriciales
1. Implementar `transpose()`, `trace()`, `det()` (hasta 3×3)
2. Agregar funciones al parser
3. Tests exhaustivos

### Sprint 4: Determinante General
1. Implementar determinante n×n (expansión cofactores)
2. Optimización para matrices grandes
3. Modo educativo para determinantes

### Sprint 5: Inversa y RREF
1. Implementar `inverse()` usando Gauss-Jordan
2. Implementar `rref()`
3. Integración con solver de sistemas lineales

### Sprint 6: CLI y Educación
1. Comandos `explain det(...)`
2. Comandos `explain inv(...)`
3. Visualización de pasos intermedios
4. Documentación y ejemplos

---

## Priorización

**Prioridad Alta (MVP):**
- ✅ Suma de matrices
- ✅ Multiplicación escalar
- ✅ Multiplicación matricial
- ✅ Transpuesta
- ✅ Determinante 2×2 y 3×3
- ✅ Traza

**Prioridad Media:**
- 🔶 Determinante n×n
- 🔶 Inversa
- 🔶 RREF
- 🔶 Modo educativo

**Prioridad Baja (futuro):**
- ⬜ Valores propios
- ⬜ Descomposición LU
- ⬜ Descomposición QR
- ⬜ SVD

---

## Consideraciones Técnicas

### Rendimiento
- Matrices grandes: considerar algoritmos iterativos vs recursivos
- Determinante: eliminación gaussiana O(n³) vs expansión cofactores O(n!)
- Caché de resultados parciales

### Precisión
- Usar `BigRational` para exactitud
- Evitar errores de punto flotante
- Detectar división por cero

### Educación
- Steps claros en español
- Mostrar matrices intermedias
- Explicar algoritmos (ej: "Intercambiamos filas para crear pivote")

---

## Testing

### Test Cases

**Suma:**
```rust
[[1, 2], [3, 4]] + [[5, 6], [7, 8]] = [[6, 8], [10, 12]]
```

**Multiplicación:**
```rust
[[1, 2], [3, 4]] * [[1, 0], [0, 1]] = [[1, 2], [3, 4]]  // Identidad
[[1, 2], [3, 4]] * [[0, 0], [0, 0]] = [[0, 0], [0, 0]]  // Cero
```

**Determinante:**
```rust
det([[a]]) = a
det([[1, 2], [3, 4]]) = -2
det([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) = 1  // Identidad
```

**Inversa:**
```rust
M * inv(M) = I
inv([[1, 2], [3, 4]]) * [[1, 2], [3, 4]] = [[1, 0], [0, 1]]
```

---

## Documentación

Actualizar:
- `README.md`: Agregar ejemplos de matrices
- `ARCHITECTURE.md`: Documentar módulo `matrix.rs`
- `help matrix` en CLI
- Crear `docs/matrix_tutorial.md`

---

## Referencias

- Álgebra Lineal: Grossman, Strang
- Algoritmos: Numerical Recipes, Press et al.
- Implementaciones: NumPy, SymPy, Mathematica
