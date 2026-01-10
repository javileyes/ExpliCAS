#!/bin/bash
# Rewrite Builder Migration Script V4
# Most aggressive - handles local() and more format! patterns
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
        # ========================================================
        # Pattern A: Simple - new_expr shorthand, string literal
        # ========================================================
        s/return\s+Some\s*\(\s*Rewrite\s*\{\s*
            new_expr\s*,\s*
            description\s*:\s*"([^"]+)"\s*\.to_string\s*\(\s*\)\s*,\s*
            before_local\s*:\s*None\s*,\s*
            after_local\s*:\s*None\s*,\s*
            assumption_events\s*:\s*Default\s*::\s*default\s*\(\s*\)\s*,?\s*(?:\/\/[^\n]*)?\s*
            required_conditions\s*:\s*vec!\s*\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/return Some(Rewrite::new(new_expr).desc("$1"))/gsx;
        
        # ========================================================
        # Pattern B: new_expr: identifier, string literal
        # ========================================================
        s/return\s+Some\s*\(\s*Rewrite\s*\{\s*
            new_expr\s*:\s*(\w+)\s*,\s*
            description\s*:\s*"([^"]+)"\s*\.to_string\s*\(\s*\)\s*,\s*
            before_local\s*:\s*None\s*,\s*
            after_local\s*:\s*None\s*,\s*
            assumption_events\s*:\s*Default\s*::\s*default\s*\(\s*\)\s*,?\s*(?:\/\/[^\n]*)?\s*
            required_conditions\s*:\s*vec!\s*\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/return Some(Rewrite::new($1).desc("$2"))/gsx;
        
        # ========================================================
        # Pattern C: With before_local/after_local Some(X)
        # Uses .local(before, after) builder method
        # ========================================================
        s/return\s+Some\s*\(\s*Rewrite\s*\{\s*
            new_expr\s*:\s*(\w+)\s*,\s*
            description\s*:\s*"([^"]+)"\s*\.to_string\s*\(\s*\)\s*,\s*
            before_local\s*:\s*Some\s*\(\s*(\w+)\s*\)\s*,\s*
            after_local\s*:\s*Some\s*\(\s*(\w+)\s*\)\s*,\s*
            assumption_events\s*:\s*Default\s*::\s*default\s*\(\s*\)\s*,?\s*(?:\/\/[^\n]*)?\s*
            required_conditions\s*:\s*vec!\s*\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/return Some(Rewrite::new($1).desc("$2").local($3, $4))/gsx;
        
        # ========================================================
        # Pattern D: format!() with simple args, no locals
        # ========================================================
        s/return\s+Some\s*\(\s*Rewrite\s*\{\s*
            new_expr\s*:\s*(\w+)\s*,\s*
            description\s*:\s*(format!\s*\("[^"]*",\s*[\w\s,\.&]+\))\s*,\s*
            before_local\s*:\s*None\s*,\s*
            after_local\s*:\s*None\s*,\s*
            assumption_events\s*:\s*Default\s*::\s*default\s*\(\s*\)\s*,?\s*(?:\/\/[^\n]*)?\s*
            required_conditions\s*:\s*vec!\s*\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/return Some(Rewrite::new($1).desc($2))/gsx;
        
        # ========================================================
        # Pattern E: format!() with locals
        # ========================================================
        s/return\s+Some\s*\(\s*Rewrite\s*\{\s*
            new_expr\s*:\s*(\w+)\s*,\s*
            description\s*:\s*(format!\s*\("[^"]*",\s*[\w\s,\.&]+\))\s*,\s*
            before_local\s*:\s*Some\s*\(\s*(\w+)\s*\)\s*,\s*
            after_local\s*:\s*Some\s*\(\s*(\w+)\s*\)\s*,\s*
            assumption_events\s*:\s*Default\s*::\s*default\s*\(\s*\)\s*,?\s*(?:\/\/[^\n]*)?\s*
            required_conditions\s*:\s*vec!\s*\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/return Some(Rewrite::new($1).desc($2).local($3, $4))/gsx;
        
        # ========================================================
        # Pattern F: description shorthand (variable name = field name)
        # ========================================================
        s/return\s+Some\s*\(\s*Rewrite\s*\{\s*
            new_expr\s*,\s*
            description\s*,\s*
            before_local\s*:\s*None\s*,\s*
            after_local\s*:\s*None\s*,\s*
            assumption_events\s*:\s*Default\s*::\s*default\s*\(\s*\)\s*,?\s*
            required_conditions\s*:\s*vec!\s*\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/return Some(Rewrite::new(new_expr).desc(description))/gsx;
        
        # ========================================================
        # Pattern G: new_expr shorthand with format!
        # ========================================================
        s/return\s+Some\s*\(\s*Rewrite\s*\{\s*
            new_expr\s*,\s*
            description\s*:\s*(format!\s*\([^)]+\))\s*,\s*
            before_local\s*:\s*None\s*,\s*
            after_local\s*:\s*None\s*,\s*
            assumption_events\s*:\s*Default\s*::\s*default\s*\(\s*\)\s*,?\s*
            required_conditions\s*:\s*vec!\s*\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/return Some(Rewrite::new(new_expr).desc($1))/gsx;
        
        # ========================================================
        # Pattern H: With named description variable
        # ========================================================
        s/return\s+Some\s*\(\s*Rewrite\s*\{\s*
            new_expr\s*:\s*(\w+)\s*,\s*
            description\s*:\s*(\w+)\s*,\s*
            before_local\s*:\s*None\s*,\s*
            after_local\s*:\s*None\s*,\s*
            assumption_events\s*:\s*Default\s*::\s*default\s*\(\s*\)\s*,?\s*
            required_conditions\s*:\s*vec!\s*\[\s*\]\s*,\s*
            poly_proof\s*:\s*None\s*,?\s*
        \}\s*\)/return Some(Rewrite::new($1).desc($2))/gsx;
        
    ' "$file" > "$temp_file"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        diff -q "$file" "$temp_file" > /dev/null 2>&1 || { echo "=== $file ==="; diff -u "$file" "$temp_file" | head -80 || true; }
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
