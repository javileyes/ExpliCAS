# Evaluación Sistemática del CAS Engine

## Objetivo
Analizar el código de `cas_engine` para identificar:
- Código ineficiente
- Código duplicado
- Código que necesita refactorización

## Tareas

### Análisis de Estructura
- [x] Explorar estructura del crate
- [x] Identificar archivos grandes (candidatos a refactorización)
- [x] Revisar archivos temporales (.tmp) - ELIMINADOS ✓

### Fase 1: Limpieza Inmediata ✅
- [x] Eliminar archivos [.tmp](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/trigonometry_new_rules.tmp)
- [x] Consolidar [is_one](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs#255-263), [is_zero](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/helpers.rs#307-315), [get_variant_name](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine.rs#883-898) en [helpers.rs](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/helpers.rs)
- [x] Actualizar 6 archivos de reglas para usar helpers

### Fase 2: Consolidación de Helpers ✅
- [x] Añadir [flatten_add_sub_chain](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/helpers.rs#127-139) a [helpers.rs](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/helpers.rs)
- [x] Añadir [flatten_mul_chain](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/helpers.rs#184-191) a [helpers.rs](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/helpers.rs)
- [x] Actualizar [collect.rs](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/collect.rs) y [grouping.rs](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/grouping.rs)
- [x] Documentar API de [helpers.rs](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/helpers.rs)
- [x] Añadir 12 tests para helpers

### Fase 3: División de Archivos Grandes
- [x] Analizar estructura de [algebra.rs](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs) (2649 líneas)
- [ ] ⚠️ División de [algebra.rs](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs) - POSPUESTA (alto acoplamiento)
- [ ] Refactorizar [trigonometry.rs](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/trigonometry.rs) con enfoque data-driven
- [ ] Separar templates HTML de [timeline.rs](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/timeline.rs)

> **Nota**: La división de [algebra.rs](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs) requiere más análisis debido al alto
> acoplamiento entre funciones helper y reglas. Se recomienda dividir en sesión
> dedicada con más tiempo.

### Fase 4: Optimización (opcional)
- [ ] Eliminar clones innecesarios
- [ ] Auditar y eliminar código muerto
- [ ] Implementar benchmarks para reglas críticas

## Resumen de Progreso

| Fase | Estado | Resultado |
|------|--------|-----------|
| Fase 1 | ✅ Completa | ~70 líneas eliminadas |
| Fase 2 | ✅ Completa | ~80 líneas + 12 tests |
| Fase 3 | ⏸️ Análisis completo | Plan documentado |
| Fase 4 | ⏳ Pendiente | - |

## Total Mejoras

- **~150 líneas** de código duplicado eliminadas
- **12 tests** nuevos para módulo `helpers`
- **2 archivos .tmp** eliminados
- **Documentación** añadida a [helpers.rs](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/helpers.rs)
