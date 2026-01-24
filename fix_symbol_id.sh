#!/bin/bash
# Script to fix SymbolId migration - ONLY cas_engine
# Run from the math directory

set -e

ENGINE_DIR="crates/cas_engine/src"

echo "=== Starting SymbolId migration fixes for cas_engine only ==="

echo "Phase 1: Fix 'name == string' patterns in Function matches..."
# Pattern: if name == "xyz" -> if ctx.sym_name(*name) == "xyz"
find "$ENGINE_DIR" -name "*.rs" -exec sed -i '' 's/if name == "\([^"]*\)"/if ctx.sym_name(*name) == "\1"/g' {} \;

echo "Phase 2: Fix 'fname == string' patterns..."
find "$ENGINE_DIR" -name "*.rs" -exec sed -i '' 's/if fname == "\([^"]*\)"/if ctx.sym_name(*fname) == "\1"/g' {} \;

echo "Phase 3: Fix match guard patterns with name ==..."
# Patterns like: Expr::Function(name, _) if name == "xyz"
find "$ENGINE_DIR" -name "*.rs" -exec sed -i '' 's/) if name == "\([^"]*\)"/) if ctx.sym_name(*name) == "\1"/g' {} \;

echo "Phase 4: Fix match guard patterns with fname ==..."
find "$ENGINE_DIR" -name "*.rs" -exec sed -i '' 's/) if fname == "\([^"]*\)"/) if ctx.sym_name(*fname) == "\1"/g' {} \;

echo "Phase 5: Fix fn_name == patterns..."
find "$ENGINE_DIR" -name "*.rs" -exec sed -i '' 's/fn_name == "\([^"]*\)"/ctx.sym_name(*fn_name) == "\1"/g' {} \;

echo "Phase 6: Fix fn_id patterns..."
find "$ENGINE_DIR" -name "*.rs" -exec sed -i '' 's/fn_id == "\([^"]*\)"/ctx.sym_name(*fn_id) == "\1"/g' {} \;

echo "Phase 7: Fix name.as_str() patterns..."
find "$ENGINE_DIR" -name "*.rs" -exec sed -i '' 's/name\.as_str()/ctx.sym_name(*name)/g' {} \;

echo "Phase 8: Fix fname.as_str() patterns..."
find "$ENGINE_DIR" -name "*.rs" -exec sed -i '' 's/fname\.as_str()/ctx.sym_name(*fname)/g' {} \;

echo "Phase 9: Fix name.clone() in Function construction..."
find "$ENGINE_DIR" -name "*.rs" -exec sed -i '' 's/Expr::Function(name\.clone(),/Expr::Function(*name,/g' {} \;

echo "Phase 10: Fix fn_id.clone() patterns..."
find "$ENGINE_DIR" -name "*.rs" -exec sed -i '' 's/Expr::Function(fn_id\.clone(),/Expr::Function(*fn_id,/g' {} \;

echo "Phase 11: Fix &*name dereference patterns..."
find "$ENGINE_DIR" -name "*.rs" -exec sed -i '' 's/\&\*name/ctx.sym_name(*name)/g' {} \;

echo "Phase 12: Fix double negation in ctx.sym_name..."
find "$ENGINE_DIR" -name "*.rs" -exec sed -i '' 's/ctx\.sym_name(\*\*name)/ctx.sym_name(*name)/g' {} \;

echo "=== Migration script completed ==="
echo "Run 'cargo check -p cas_engine' to verify"
