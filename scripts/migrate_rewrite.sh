#!/bin/bash
# Rewrite Builder Migration Script
# Converts verbose Rewrite {} structs to fluent builder pattern
# Usage: 
#   ./migrate_rewrite.sh --dry-run   # Preview changes
#   ./migrate_rewrite.sh             # Apply changes

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No files will be modified ==="
fi

# Target directory
RULES_DIR="crates/cas_engine/src/rules"

# Count before
count_before() {
    grep -r "Rewrite {" "$RULES_DIR" --include="*.rs" | wc -l | tr -d ' '
}

echo "Rewrite {} usages before: $(count_before)"

# Function to process a single file
process_file() {
    local file="$1"
    local temp_file="${file}.migrated"
    
    # Use Perl for multi-line regex replacement (sed is limited for multi-line)
    # Pattern 1: Simple case with Default::default() and vec![]
    perl -0777 -pe '
        # Pattern: Simple Rewrite with just new_expr and description
        # Matches the most common pattern:
        # return Some(Rewrite {
        #     new_expr: EXPR,
        #     description: DESC,
        #     before_local: None,
        #     after_local: None,
        #     assumption_events: Default::default(),
        #     required_conditions: vec![],
        #     poly_proof: None,
        # });
        
        s/return Some\(Rewrite \{\s*
            new_expr:\s*([^,]+),\s*
            description:\s*([^,]+?)\.to_string\(\),\s*
            before_local:\s*None,\s*
            after_local:\s*None,\s*
            assumption_events:\s*Default::default\(\),\s*
            required_conditions:\s*vec!\[\],\s*
            poly_proof:\s*None,\s*
        \}\)/return Some(Rewrite::new($1).desc($2))/gsx;
        
        # Pattern for format! descriptions
        s/return Some\(Rewrite \{\s*
            new_expr:\s*([^,]+),\s*
            description:\s*(format!\([^)]+\)),\s*
            before_local: None,\s*
            after_local: None,\s*
            assumption_events: Default::default\(\),\s*
            required_conditions: vec!\[\],\s*
            poly_proof: None,\s*
        \}\)/return Some(Rewrite::new($1).desc($2))/gsx;
        
        # Simplified pattern - less strict on whitespace
        s/return Some\(Rewrite \{\s*new_expr:\s*(\w+),\s*description:\s*"([^"]+)"\.to_string\(\),\s*before_local:\s*None,\s*after_local:\s*None,\s*assumption_events:\s*Default::default\(\),\s*required_conditions:\s*vec!\[\],\s*poly_proof:\s*None,?\s*\}\)/return Some(Rewrite::new($1).desc("$2"))/gs;
    ' "$file" > "$temp_file"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        # Show diff if any changes
        if ! diff -q "$file" "$temp_file" > /dev/null 2>&1; then
            echo "=== Changes in $file ==="
            diff -u "$file" "$temp_file" | head -50 || true
            echo ""
        fi
        rm "$temp_file"
    else
        # Apply changes
        if ! diff -q "$file" "$temp_file" > /dev/null 2>&1; then
            mv "$temp_file" "$file"
            echo "Modified: $file"
        else
            rm "$temp_file"
        fi
    fi
}

# Find all Rust files with Rewrite { and process them
echo ""
echo "Processing files..."
for file in $(grep -rl "Rewrite {" "$RULES_DIR" --include="*.rs"); do
    process_file "$file"
done

if [[ "$DRY_RUN" == "false" ]]; then
    echo ""
    echo "Rewrite {} usages after: $(count_before)"
    echo ""
    echo "Migration complete. Run 'cargo build' to verify."
fi
