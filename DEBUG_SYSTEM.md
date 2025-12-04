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
