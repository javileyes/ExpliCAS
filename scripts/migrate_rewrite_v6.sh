#!/bin/bash
# Simple Rewrite Migration Script V6 - Handles ONLY simple patterns
# Avoids SmallVec type inference issues by only migrating patterns with:
# - assumption_events: Default::default()
# - No assumption_events variable

set -e

DRY_RUN=false
[[ "$1" == "--dry-run" ]] && DRY_RUN=true && echo "=== DRY RUN ==="

RULES_DIR="crates/cas_engine/src/rules"

count_usages() {
    grep -r "Rewrite {" "$RULES_DIR" --include="*.rs" 2>/dev/null | wc -l | tr -d ' '
}

echo "Rewrite {} usages before: $(count_usages)"

# Create a simple pattern replacement using perl for each common pattern
process_file() {
    local file="$1"
    local temp_file="${file}.migrated"
    
    # Pattern: Simple with string description, no locals, Default::default()
    perl -0777 -pe '
        # Pattern 1: return Some(Rewrite { new_expr, description: "...", no locals, Default::default() }
        s/return\s+Some\s*\(\s*Rewrite\s*\{\s*
            new_expr(?:\s*:\s*(\w+))?\s*,\s*
            description\s*:\s*"([^"]+)"\.to_string\(\)\s*,\s*
            before_local\s*:\s*None\s*,\s*
            after_local\s*:\s*None\s*,\s*
            assumption_events\s*:\s*Default::default\(\)\s*,\s*
            required_conditions\s*:\s*vec!\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/return Some(Rewrite::new($1new_expr).desc("$2"))/gsx;
        
        # Pattern 2: Some(Rewrite { ... } (without return)
        s/Some\s*\(\s*Rewrite\s*\{\s*
            new_expr(?:\s*:\s*(\w+))?\s*,\s*
            description\s*:\s*"([^"]+)"\.to_string\(\)\s*,\s*
            before_local\s*:\s*None\s*,\s*
            after_local\s*:\s*None\s*,\s*
            assumption_events\s*:\s*Default::default\(\)\s*,\s*
            required_conditions\s*:\s*vec!\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/Some(Rewrite::new($1new_expr).desc("$2"))/gsx;
        
        # Pattern 3: With local() - simple string desc, Default::default()
        s/return\s+Some\s*\(\s*Rewrite\s*\{\s*
            new_expr(?:\s*:\s*(\w+))?\s*,\s*
            description\s*:\s*"([^"]+)"\.to_string\(\)\s*,\s*
            before_local\s*:\s*Some\(\s*(\w+)\s*\)\s*,\s*
            after_local\s*:\s*Some\(\s*(\w+)\s*\)\s*,\s*
            assumption_events\s*:\s*Default::default\(\)\s*,\s*
            required_conditions\s*:\s*vec!\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/return Some(Rewrite::new($1new_expr).desc("$2").local($3, $4))/gsx;
        
        # Pattern 4: Some(Rewrite with local() (without return)
        s/Some\s*\(\s*Rewrite\s*\{\s*
            new_expr(?:\s*:\s*(\w+))?\s*,\s*
            description\s*:\s*"([^"]+)"\.to_string\(\)\s*,\s*
            before_local\s*:\s*Some\(\s*(\w+)\s*\)\s*,\s*
            after_local\s*:\s*Some\(\s*(\w+)\s*\)\s*,\s*
            assumption_events\s*:\s*Default::default\(\)\s*,\s*
            required_conditions\s*:\s*vec!\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/Some(Rewrite::new($1new_expr).desc("$2").local($3, $4))/gsx;
        
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
