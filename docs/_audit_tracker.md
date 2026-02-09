# Documentation Audit Tracker (Feb 2026)

## Files Fixed

### README.md
1. ✅ Line 899: `timeline.rs` → `timeline/` (directory module) in Project Structure
2. ✅ Project Structure: Updated to reflect `helpers/`, `json/`, split REPL commands
3. ✅ Duplicated section numbers: 6→11, 6→12, 7→13

### ARCHITECTURE.md
1. ✅ Line 1756: `// En timeline.rs` → `// En timeline/ (módulo directorio)`
2. ✅ Lines 3650-3651: `AddView`/`MulChainView` → `nary::add_leaves`/`nary::mul_leaves`
3. ✅ Lines 3731-3734: `flatten_add`/`flatten_mul`/Predicates/Builders → ✅ Migrated

### ASSUMPTIONS_POLICY.md
1. ✅ Line 341: `timeline.rs` → `timeline/` module

### POLICY.md
1. ✅ Line 291: `helpers.rs` (file) → `helpers/` (directory module)

### MAINTENANCE.md
1. ✅ Line 92: `cas_ast::views` → `cas_engine::nary` for flatten canonical location
2. ✅ Lines 443-451: Removed duplicated troubleshooting checklist

### docs/task.md (archived)
1. ✅ Added "[!NOTE] Archived Plan" header noting stale file paths

### docs/implementation_plan.md (archived)
1. ✅ Added "[!NOTE] Archived Plan" header noting stale file paths

## Files Verified Accurate (no changes needed)

- JSON_CLI_API.md — all file paths verified existing
- CONST_FOLD_POLICY.md — `literal_integer_i64` confirmed in `const_eval.rs`
- Safe_Feature_Layering_Pattern.md — `flatten_add`/`nary.rs` references are conceptual examples, accurate
- SEMANTICS_POLICY.md, BUDGET_POLICY.md, DISPLAY_POLICY.md, LIMITS_POLICY.md
- SOLVER_SIMPLIFY_POLICY.md, SUBSTITUTE_POLICY.md, NORMAL_FORM_GOAL.md
- DIDACTIC_STEPS.md, RULES.md, DEBUG_SYSTEM.md, CORE_TESTS.md
- METAMORPHIC_TESTING.md, SOLVE_SYSTEM.md, builtin_guidelines.md
- POLY_GCD.md, ZIPPEL_GCD.md, POLY_REF_ARCHITECTURE.md
- POLY_EXPAND_PERFORMANCE.md, FAST_EXPAND.md, best_so_far.md
- Requires_vs_assumed.md, Requires_Assumptions_Choises_Domains.md
- ROBUSTNESS.md, n4_ast_variants.md, CHANGELOG.md, IMPROVEMENTS.md
- CONTRIBUTING.md, ROADMAP_V2_1_TO_V2_4.md, POLICY_TABLES.md
- docs/sp/POLICY_TABLES.md, matrix_operations_plan.md
- API_JSON.md, WIRE_SCHEMA.md, ANDROID_FFI_GUIDE.md, ANDROID_OUTPUT_ENVELOPE.md
- crates/cas_android_ffi/README.md, web/README.md

## Summary

- **50 docs audited** across 9 batches
- **7 files fixed** with targeted edits
- **43 files verified clean** (no stale references)
- **0 obsolete files** identified for deletion (archived plans preserved with headers)
