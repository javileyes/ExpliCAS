# Sistema de Debug con Tracing - Guía de Uso

## ✅ Estado: Implementado

El proyecto ahora usa `tracing` para debug logging profesional.

## Cómo Usar

### Sin debug (normal)
```bash
cargo build --release
./target/release/cas_cli
cargo bench  # ✓ Sin output de debug
cargo test   # ✓ Sin output de debug
```

### Con debug activado

#### Opción 1: Todo el módulo canonical_forms
```bash
RUST_LOG=cas_engine::canonical_forms=debug cargo test
RUST_LOG=cas_engine::canonical_forms=debug ./target/release/cas_cli
```

#### Opción 2: Todo el engine
```bash
RUST_LOG=cas_engine=debug cargo test
```

#### Opción 3: Múltiples módulos
```bash
RUST_LOG=cas_engine::canonical_forms=debug,cas_engine::engine=trace cargo test
```

#### Opción 4: Muy verbose (todos los módulos, nivel trace)
```bash
RUST_LOG=cas_engine=trace ./target/release/cas_cli
```

## Niveles de Log

- `error` - Sólo errores críticos
- `warn` - Advertencias
- `info` - Información general
- `debug` - ⭐ **Recomendado para development** - información de debugging
- `trace` - Muy verbose, todos los detalles

## Qué está loggeado actualmente

### `canonical_forms.rs`
- `is_canonical_form()` - Qué expresiones se están checkeando
- `is_conjugate()` - Pares de conjugados siendo verificados

### `engine.rs`
- Cuando se salta simplificación de Pow canónico

## Ejemplo de Salida con Debug

```bash
$ RUST_LOG=cas_engine::canonical_forms=debug echo "simplify ((x+1)*(x-1))^2" | ./target/release/cas_cli

# (Mostrará trazas de debug solo de canonical_forms)
```

## Para Desarrollo Futuro

### Agregar nuevo debug logging:

```rust
use tracing::{debug, trace, info};

pub fn my_function() {
    debug!("Checking something: {:?}", value);  // Para debugging general
    trace!("Detailed info: {}", detail);        // Para mucho detalle
    info!("Important milestone");               // Para hitos importantes
}
```

### Ventajas sobre `eprintln!`

✅ **Cero overhead** cuando desactivado (compilación optimizada elimina el código)  
✅ **Control granular** - activa solo los módulos que necesitas  
✅ **Estándar profesional** - compatible con herramientas de observabilidad  
✅ **No contamina** benchmarks ni tests  
✅ **Flexible** - cambia nivel sin recompilar

---

## 🔍 Debugging Pattern Detection System ★ (Added 2025-12)

El sistema de Pattern Detection puede ser debuggeado usando `eprintln!` temporal ya que no tiene integración con `tracing` aún.

### Quick Debug: ¿Se están marcando expresiones?

```bash
# Temporal: Agregar en orchestrator.rs después de scan_and_mark_patterns
eprintln!("[PATTERN] Marked {} expressions", pattern_marks.protected.len());

# Ejecutar
echo "sec(x)^2 - tan(x)^2" | cargo run -p cas_cli 2>&1 | grep PATTERN
```

### Debugging por Componente

#### 1. Pattern Scanner (`pattern_scanner.rs`)

**¿Qué detecta el scanner?**

```rust
// Temporal: Agregar en scan_and_mark_patterns()
pub fn scan_and_mark_patterns(ctx: &Context, expr_id: ExprId, marks: &mut PatternMarks) {
    eprintln!("[SCAN] Visiting: {:?}", ctx.get(expr_id));
    
    match ctx.get(expr_id) {
        Expr::Add(left, right) => {
            if is_pythagorean_difference(ctx, *left, *right) {
                eprintln!("[SCAN] ✓ Found Pythagorean pattern at {:?}", expr_id);
                // ...
            }
        }
        _ => {}
    }
}
```

**Ejecutar**:
```bash
echo "sec(x)^2 - tan(x)^2" | cargo run -p cas_cli 2>&1 | grep SCAN
```

**Output esperado**:
```
[SCAN] Visiting: Add(...)
[SCAN] ✓ Found Pythagorean pattern at ExprId(5)
```

#### 2. Pattern Marks Threading (`engine.rs`)

**¿Llegan los marks a las reglas?**

```rust
// Temporal: En LocalSimplificationTransformer::apply_rules
fn apply_rules(&mut self, expr: ExprId, parent_ctx: &ParentContext) -> ExprId {
    if let Some(marks) = parent_ctx.pattern_marks() {
        eprintln!("[MARKS] Rule has access to {} protected expressions", 
                  marks.protected.len());
    }
    // ... resto del código
}
```

#### 3. Guard Verification (`trigonometry.rs`)

**¿Está funcionando el guard?**

```rust
// Temporal: En TanToSinCosRule::apply
fn apply(&self, ctx: &mut Context, expr: ExprId, parent_ctx: &ParentContext) -> Option<Rewrite> {
    if let Some(marks) = parent_ctx.pattern_marks() {
        if marks.is_pythagorean_protected(expr) {
            eprintln!("[GUARD] ✓ Skipping tan→sin/cos for protected {:?}", expr);
            return None;
        } else {
            eprintln!("[GUARD] ✗ Not protected, converting {:?}", expr);
        }
    }
    // ... resto del código
}
```

#### 4. Direct Rule Application (`trigonometry.rs`)

**¿Se aplica la regla directa?**

```rust
// Temporal: En SecTanPythagoreanRule
define_rule!(
    SecTanPythagoreanRule,
    "Secant-Tangent Pythagorean Identity",
    |ctx, expr| {
        eprintln!("[RULE] SecTan checking: {:?}", ctx.get(expr));
        
        if let Expr::Add(left, right) = ctx.get(expr).clone() {
            eprintln!("[RULE] Found Add, checking for Neg pattern...");
            // ... resto
            
            if /* pattern matches */ {
                eprintln!("[RULE] ✓ Applying sec²-tan²=1");
                return Some(Rewrite { ... });
            }
        }
        None
    }
);
```

### Escenarios de Debug Comunes

#### Escenario 1: "sec²(x) - tan²(x) no simplifica a 1"

**Pasos de debugging**:

1. **Verificar detección**:
   ```bash
   # Agregar debug en scan_and_mark_patterns
   echo "sec(x)^2 - tan(x)^2" | cargo run -p cas_cli 2>&1 | grep "Found Pythagorean"
   ```
   
   - ✅ Si aparece: Scanner funciona
   - ❌ Si no aparece: Problema en pattern scanner

2. **Verificar AST structure**:
   ```rust
   // Temporal en SecTanPythagoreanRule
   eprintln!("[AST] Expression structure: {:#?}", ctx.get(expr));
   ```
   
   - Debe ser: `Add(sec²(x), Neg(tan²(x)))`
   - **NO**: `Sub(sec²(x), tan²(x))` ← esto nunca existe!

3. **Verificar marks**:
   ```bash
   # Agregar debug en orchestrator después del scan
   echo "sec(x)^2 - tan(x)^2" | cargo run -p cas_cli 2>&1 | grep "Marked"
   ```
   
   - Debe mostrar: `[PATTERN] Marked 2 expressions` (sec y tan)

4. **Verificar que regla está registrada**:
   ```rust
   // En trigonometry.rs::register, temporal:
   pub fn register(simplifier: &mut Simplifier) {
       eprintln!("[REGISTER] Adding SecTanPythagoreanRule");
       simplifier.add_rule(Box::new(SecTanPythagoreanRule));
       // ...
   }
   ```

#### Escenario 2: "Guard no previene conversión"

**Debug**:

```bash
# Agregar debug en TanToSinCosRule::apply
cargo test --test pythagorean_variants_test -- --nocapture 2>&1 | grep GUARD
```

**Output esperado**:
```
[GUARD] ✓ Skipping tan→sin/cos for protected ExprId(3)
```

**Si no aparece**: Verificar que `ParentContext` se extiende correctamente en recursión.

#### Escenario 3: "Tests pasan pero CLI no funciona"

**Posible causa**: `pattern_marks` no se pasan en CLI path.

**Debug**:
```rust
// Temporal en orchestrator.rs::simplify
pub fn simplify(...) -> ... {
    let pattern_marks = PatternScanner::scan_and_mark_patterns(&simplifier.context, expr);
    eprintln!("[ORCHESTRATOR] Created marks with {} protected", pattern_marks.protected.len());
    
    let (result, steps) = simplifier.apply_rules_loop(expr, &pattern_marks);
    // ...
}
```

### Debug Completo: Trace End-to-End

Para debuggear todo el flujo:

```rust
// 1. En orchestrator.rs
eprintln!("[1-ORCHESTRATOR] Starting simplify for {:?}", expr);
let pattern_marks = PatternScanner::scan_and_mark_patterns(...);
eprintln!("[2-ORCHESTRATOR] Scanned, found {} marks", pattern_marks.protected.len());

// 2. En pattern_scanner.rs
eprintln!("[3-SCANNER] Scanning expr {:?}", expr_id);
if is_pythagorean_difference(...) {
    eprintln!("[4-SCANNER] ✓ Marking Pythagorean pattern");
}

// 3. En engine.rs apply_rules_loop
eprintln!("[5-ENGINE] apply_rules_loop called with {} marks", pattern_marks.protected.len());
let initial_parent_ctx = ParentContext::with_marks(pattern_marks.clone());
eprintln!("[6-ENGINE] Created ParentContext");

// 4. En engine.rs apply_rules
eprintln!("[7-TRANSFORMER] Applying rules to {:?}", expr);
if let Some(marks) = parent_ctx.pattern_marks() {
    eprintln!("[8-TRANSFORMER] Has {} marks available", marks.protected.len());
}

// 5. En trigonometry.rs TanToSinCosRule
eprintln!("[9-RULE-GUARD] Checking protection for {:?}", expr);

// 6. En trigonometry.rs SecTanPythagoreanRule
eprintln!("[10-RULE-DIRECT] Checking pattern match for {:?}", expr);
```

**Ejecutar**:
```bash
echo "sec(x)^2 - tan(x)^2" | cargo run -p cas_cli 2>&1 | grep -E "\[(1|2|3|4|5|6|7|8|9|10)-"
```

**Output esperado** (orden):
```
[1-ORCHESTRATOR] Starting simplify...
[2-ORCHESTRATOR] Scanned, found 2 marks
[3-SCANNER] Scanning expr...
[4-SCANNER] ✓ Marking Pythagorean pattern
[5-ENGINE] apply_rules_loop called with 2 marks
[6-ENGINE] Created ParentContext
[7-TRANSFORMER] Applying rules...
[8-TRANSFORMER] Has 2 marks available
[9-RULE-GUARD] Checking protection...
[10-RULE-DIRECT] Checking pattern match...
```

### Testing con Debug Output

```bash
# Test específico con output
cargo test test_sec_tan_equals_one -- --nocapture 2>&1 | grep -E "(PATTERN|GUARD|RULE)"

# Todos los tests de pattern detection
cargo test pattern -- --nocapture

# Debug test especializado
cargo test --test debug_sec_tan -- --nocapture
```

### Herramientas Útiles

#### 1. Ver AST structure
```rust
// Temporal en cualquier lugar
eprintln!("{:#?}", ctx.get(expr));

// O más legible
use cas_format::format_expr;
eprintln!("Expression: {}", format_expr(ctx, expr));
```

#### 2. Verificar igualdad de expresiones
```rust
use crate::ordering::compare_expr;
eprintln!("Are equal? {:?}", compare_expr(ctx, expr1, expr2));
```

#### 3. Inspeccionar PatternMarks
```rust
// Temporal método helper
impl PatternMarks {
    pub fn debug_dump(&self) {
        eprintln!("[MARKS] {} protected expressions:", self.protected.len());
        for id in &self.protected {
            eprintln!("  - {:?}", id);
        }
    }
}

// Uso
pattern_marks.debug_dump();
```

### Removing Debug Code

**Antes de commit**:
```bash
# Buscar todos los eprintln! temporales
rg "eprintln.*PATTERN|GUARD|RULE|SCAN" crates/

# Remover los que agregaste
# Asegurarte que los únicos eprintln! son para errores reales
```

### Integration con Tracing (Futuro)

Para integrar con el sistema de tracing:

```rust
// En pattern_scanner.rs
use tracing::debug;

pub fn scan_and_mark_patterns(...) {
    debug!("Scanning expression: {:?}", expr_id);
    
    if is_pythagorean_difference(...) {
        debug!("Found Pythagorean pattern, marking bases");
    }
}
```

**Uso**:
```bash
RUST_LOG=cas_engine::pattern_scanner=debug cargo test
```

---

## Troubleshooting

**Problema**: No veo logs
- ✓ Verifica que `RUST_LOG` esté configurado
- ✓ Usa `=debug` no `=info` (debug es más verbose)

**Problema**: Demasiados logs
- ✓ Especifica módulos: `RUST_LOG=cas_engine::canonical_forms=debug` en vez de `RUST_LOG=debug`

**Problema**: Quiero logs en archivo
```bash
RUST_LOG=debug ./cas_cli 2> debug.log
```

**Problema**: Pattern detection no funciona (Added 2025-12)
- ✓ Sigue la guía de "Debug Completo: Trace End-to-End" arriba
- ✓ Verifica cada paso: Scanner → Marks → ParentContext → Guard/Rule
- ✓ **CRÍTICO**: Recuerda que `a - b` es `Add(a, Neg(b))`, NO `Sub(a, b)`
- ✓ Verifica que reglas están registradas: busca `register` en `trigonometry.rs`
- ✓ Corre tests: `cargo test pattern -- --nocapture`
