#!/bin/bash
# Rewrite Builder Migration Script V5 - Final aggressive version
# Handles both Rewrite { and crate::rule::Rewrite { patterns
# Usage: 
#   ./migrate_rewrite.sh --dry-run
#   ./migrate_rewrite.sh

set -e

DRY_RUN=false
[[ "$1" == "--dry-run" ]] && DRY_RUN=true && echo "=== DRY RUN ==="

RULES_DIR="crates/cas_engine/src/rules"

count_usages() {
    grep -r "Rewrite {" "$RULES_DIR" --include="*.rs" 2>/dev/null | wc -l | tr -d ' '
}

echo "Rewrite {} usages before: $(count_usages)"

process_file() {
    local file="$1"
    local temp_file="${file}.migrated"
    
    perl -0777 -pe '
        # ==== Pattern 1: return Some(Rewrite { / crate::rule::Rewrite { - simple string desc ====
        s/return\s+Some\s*\(\s*(?:crate::rule::)?Rewrite\s*\{\s*
            new_expr(?:\s*:\s*(\w+))?\s*,\s*
            description\s*:\s*"([^"]+)"\s*\.to_string\s*\(\s*\)\s*,\s*
            before_local\s*:\s*None\s*,\s*
            after_local\s*:\s*None\s*,\s*
            assumption_events\s*(?::\s*Default\s*::\s*default\s*\(\s*\)|:\s*events|)\s*,?\s*(?:\/\/[^\n]*)?\s*
            required_conditions\s*:\s*vec!\s*\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/return Some(Rewrite::new(${1}new_expr).desc("$2"))/gsx;
        
        # ==== Pattern 2: Some(Rewrite { - simple (without return) ====
        s/Some\s*\(\s*(?:crate::rule::)?Rewrite\s*\{\s*
            new_expr(?:\s*:\s*(\w+))?\s*,\s*
            description\s*:\s*"([^"]+)"\s*\.to_string\s*\(\s*\)\s*,\s*
            before_local\s*:\s*None\s*,\s*
            after_local\s*:\s*None\s*,\s*
            assumption_events\s*:\s*Default\s*::\s*default\s*\(\s*\)\s*,?\s*(?:\/\/[^\n]*)?\s*
            required_conditions\s*:\s*vec!\s*\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/Some(Rewrite::new(${1}new_expr).desc("$2"))/gsx;
        
        # ==== Pattern 3: With events variable (assumption_events: events) ====
        s/return\s+Some\s*\(\s*(?:crate::rule::)?Rewrite\s*\{\s*
            new_expr(?:\s*:\s*(\w+))?\s*,\s*
            description\s*:\s*"([^"]+)"\s*\.to_string\s*\(\s*\)\s*,\s*
            before_local\s*:\s*None\s*,\s*
            after_local\s*:\s*None\s*,\s*
            assumption_events\s*,?\s*
            required_conditions\s*:\s*vec!\s*\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/return Some(Rewrite::new(${1}new_expr).desc("$2").assume_all(assumption_events))/gsx;
        
        # ==== Pattern 4: With local() ====
        s/return\s+Some\s*\(\s*(?:crate::rule::)?Rewrite\s*\{\s*
            new_expr(?:\s*:\s*(\w+))?\s*,\s*
            description\s*:\s*"([^"]+)"\s*\.to_string\s*\(\s*\)\s*,\s*
            before_local\s*:\s*Some\s*\(\s*(\w+)\s*\)\s*,\s*
            after_local\s*:\s*Some\s*\(\s*(\w+)\s*\)\s*,\s*
            assumption_events\s*:\s*Default\s*::\s*default\s*\(\s*\)\s*,?\s*
            required_conditions\s*:\s*vec!\s*\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/return Some(Rewrite::new(${1}new_expr).desc("$2").local($3, $4))/gsx;
        
    ' "$file" > "$temp_file"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        diff -q "$file" "$temp_file" > /dev/null 2>&1 || { echo "=== $file ==="; diff -u "$file" "$temp_file" | head -50 || true; }
        rm "$temp_file"
    else
        diff -q "$file" "$temp_file" > /dev/null 2>&1 || { mv "$temp_file" "$file"; echo "Modified: $file"; } || rm -f "$temp_file"
    fi
}

echo ""
for file in $(grep -rl "Rewrite {" "$RULES_DIR" --include="*.rs" 2>/dev/null); do
    process_file "$file"
done

[[ "$DRY_RUN" == "false" ]] && echo -e "\nRewrite {} usages after: $(count_usages)\nRun 'cargo fmt && cargo build' to verify."
