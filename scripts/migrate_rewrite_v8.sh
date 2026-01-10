#!/bin/bash
# Rewrite Migration Script V8 - Handles format!() descriptions
# Conservative: only Default::default() for assumption_events

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
        # Pattern 1: format!("text", args) with shorthand new_expr
        s/return\s+Some\s*\(\s*Rewrite\s*\{\s*
            new_expr\s*,\s*
            description\s*:\s*(format!\([^)]+\))\s*,\s*
            before_local\s*:\s*None\s*,\s*
            after_local\s*:\s*None\s*,\s*
            assumption_events\s*:\s*Default::default\(\)\s*,\s*
            required_conditions\s*:\s*vec!\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/return Some(Rewrite::new(new_expr).desc($1))/gsx;
        
        # Pattern 2: format!("text", args) with explicit new_expr: var
        s/return\s+Some\s*\(\s*Rewrite\s*\{\s*
            new_expr\s*:\s*(\w+)\s*,\s*
            description\s*:\s*(format!\([^)]+\))\s*,\s*
            before_local\s*:\s*None\s*,\s*
            after_local\s*:\s*None\s*,\s*
            assumption_events\s*:\s*Default::default\(\)\s*,\s*
            required_conditions\s*:\s*vec!\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/return Some(Rewrite::new($1).desc($2))/gsx;
        
        # Pattern 3: Some(Rewrite { format! (without return), shorthand
        s/Some\s*\(\s*Rewrite\s*\{\s*
            new_expr\s*,\s*
            description\s*:\s*(format!\([^)]+\))\s*,\s*
            before_local\s*:\s*None\s*,\s*
            after_local\s*:\s*None\s*,\s*
            assumption_events\s*:\s*Default::default\(\)\s*,\s*
            required_conditions\s*:\s*vec!\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/Some(Rewrite::new(new_expr).desc($1))/gsx;
        
        # Pattern 4: Some(Rewrite { format! (without return), explicit
        s/Some\s*\(\s*Rewrite\s*\{\s*
            new_expr\s*:\s*(\w+)\s*,\s*
            description\s*:\s*(format!\([^)]+\))\s*,\s*
            before_local\s*:\s*None\s*,\s*
            after_local\s*:\s*None\s*,\s*
            assumption_events\s*:\s*Default::default\(\)\s*,\s*
            required_conditions\s*:\s*vec!\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/Some(Rewrite::new($1).desc($2))/gsx;
        
    ' "$file" > "$temp_file"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        if ! diff -q "$file" "$temp_file" > /dev/null 2>&1; then
            echo "=== Would modify: $file ==="
            diff -u "$file" "$temp_file" | head -30 || true
        fi
        rm "$temp_file"
    else
        if ! diff -q "$file" "$temp_file" > /dev/null 2>&1; then
            mv "$temp_file" "$file"
            echo "Modified: $file"
        else
            rm -f "$temp_file"
        fi
    fi
}

echo ""
for file in $(grep -rl "Rewrite {" "$RULES_DIR" --include="*.rs" 2>/dev/null); do
    process_file "$file"
done

[[ "$DRY_RUN" == "false" ]] && echo -e "\nRewrite {} usages after: $(count_usages)\nRun 'cargo fmt && cargo build' to verify."
