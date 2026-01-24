#!/usr/bin/env python3
"""
Minimal SymbolId migration fixer v5.

ONLY handles the specific case where Expr::Function pattern match gives:
- From ctx.get(expr) -> &Expr -> captured fn_id is &SymbolId
- From ctx.get(expr).clone() -> Expr owned -> captured fn_id is SymbolId (usize)

This version is extremely conservative:
1. Only modifies ctx.sym_name(fn_id) -> ctx.sym_name(*fn_id) for CLEAR reference contexts
2. Only modifies Expr::Function(fn_id, ...) -> Expr::Function(*fn_id, ...) for CLEAR reference contexts in ctx.add()

Does NOT touch:
- Borrow issues (E0502)
- Comparisons with strings (E0277)
- Scope issues (E0425)
"""

import re
from pathlib import Path


def find_rust_files(directory):
    """Find all .rs files in directory recursively."""
    return list(Path(directory).rglob("*.rs"))


def is_definitely_owned(lines, line_idx, var_name):
    """
    Conservative check: is this definitely in an owned context?
    Returns True only for obvious cases.
    """
    start = max(0, line_idx - 10)
    context = '\n'.join(lines[start:line_idx + 1])
    
    # Pattern 1: if let Expr::Function(var, ...) = expr_data
    if re.search(rf'if let Expr::Function\({var_name},[^)]+\)\s*=\s*\w+_data', context):
        return True
    
    # Pattern 2: .clone() on same line as pattern
    if re.search(rf'if let Expr::Function\({var_name},[^)]+\)\s*=.*\.clone\(\)', context):
        return True
    
    # Pattern 3: let expr_data = ctx.get(...).clone(); ... if let Expr::Function(var, ...) = expr_data
    if '_data = ' in context and '.clone()' in context and f'Expr::Function({var_name},' in context:
        return True
    
    return False


def is_definitely_reference(lines, line_idx, var_name):
    """
    Conservative check: is this definitely in a reference context?
    Returns True only for obvious cases.
    """
    start = max(0, line_idx - 10)
    context = '\n'.join(lines[start:line_idx + 1])
    
    # Pattern: if let Expr::Function(var, ...) = ctx.get(...) { without .clone()
    # Check for ctx.get pattern WITHOUT _data and WITHOUT .clone()
    if f'Expr::Function({var_name},' in context:
        if 'ctx.get(' in context and '.clone()' not in context and '_data' not in context:
            return True
    
    return False


def fix_file(filepath):
    """Apply minimal fixes to a single file."""
    with open(filepath, 'r') as f:
        lines = f.read().split('\n')
    
    modified = False
    
    for i, line in enumerate(lines):
        original = line
        
        for var in ['fn_id', 'name', 'fn_name']:
            # Fix 1: ctx.sym_name(var) in reference context -> ctx.sym_name(*var)
            if f'ctx.sym_name({var})' in line and f'ctx.sym_name(*{var})' not in line:
                if is_definitely_reference(lines, i, var):
                    line = line.replace(f'ctx.sym_name({var})', f'ctx.sym_name(*{var})')
                    line = line.replace(f'self.ctx.sym_name({var})', f'self.ctx.sym_name(*{var})')
            
            # Fix 2: ctx.sym_name(*var) in owned context -> ctx.sym_name(var)
            if f'ctx.sym_name(*{var})' in line:
                if is_definitely_owned(lines, i, var):
                    line = line.replace(f'ctx.sym_name(*{var})', f'ctx.sym_name({var})')
                    line = line.replace(f'self.ctx.sym_name(*{var})', f'self.ctx.sym_name({var})')
            
            # Fix 3: Expr::Function(var, in ctx.add() for reference context
            if 'ctx.add(Expr::Function(' in line or 'self.ctx.add(Expr::Function(' in line:
                if f'Expr::Function({var},' in line and f'Expr::Function(*{var},' not in line:
                    if is_definitely_reference(lines, i, var):
                        line = line.replace(f'Expr::Function({var},', f'Expr::Function(*{var},')
            
            # Fix 4: Expr::Function(*var, in ctx.add() for owned context 
            if 'ctx.add(Expr::Function(' in line or 'self.ctx.add(Expr::Function(' in line:
                if f'Expr::Function(*{var},' in line:
                    if is_definitely_owned(lines, i, var):
                        line = line.replace(f'Expr::Function(*{var},', f'Expr::Function({var},')
        
        if line != original:
            lines[i] = line
            modified = True
    
    if modified:
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
    
    return modified


def main():
    """Main entry point."""
    base_dir = "/Users/javiergimenezmoya/developer/math/crates/cas_engine/src"
    
    print("SymbolId Migration Fixer v5 - Minimal")
    print("=" * 50)
    rust_files = find_rust_files(base_dir)
    print(f"Found {len(rust_files)} Rust files")
    print()
    
    fixed_files = []
    
    for filepath in sorted(rust_files):
        try:
            if fix_file(str(filepath)):
                fixed_files.append(filepath.name)
                print(f"  Fixed: {filepath.name}")
        except Exception as e:
            print(f"  Error: {filepath.name}: {e}")
    
    print()
    print(f"Total files modified: {len(fixed_files)}")


if __name__ == "__main__":
    main()
