#!/usr/bin/env python3
"""
domain_assumption Field Remover - Safe AST-lite approach

This script safely removes the `domain_assumption` field from:
1. Rewrite { ... } struct literals in rule files
2. Step { ... } struct literals in engine/orchestrator files
3. Struct field declarations in rule.rs and step.rs

SAFE because:
1. Uses brace counting to find complete struct blocks
2. Skips string literals and comments
3. Verifies block has expected Rewrite/Step fields before modifying
4. Reports all changes before applying (dry-run by default)

Usage:
    python remove_domain_assumption.py --dry-run     # Just report what would change
    python remove_domain_assumption.py --apply       # Apply changes
    python remove_domain_assumption.py --file X.rs   # Process single file
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class StructBlock:
    """Represents a found struct literal block"""
    file: str
    start_line: int
    end_line: int
    start_pos: int
    end_pos: int
    content: str
    struct_type: str  # "Rewrite" or "Step"
    domain_assumption_line: Optional[str]
    domain_assumption_pos: Optional[Tuple[int, int]]  # (start, end) in content

def find_matching_brace(content: str, start: int) -> Optional[int]:
    """
    Find the matching } for the { at position start.
    Returns None if not found or unbalanced.
    Skips string literals and comments.
    """
    if content[start] != '{':
        return None
    
    depth = 1
    i = start + 1
    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    
    while i < len(content) and depth > 0:
        c = content[i]
        prev = content[i-1] if i > 0 else ''
        next_c = content[i+1] if i + 1 < len(content) else ''
        
        # Handle comments
        if not in_string and not in_char:
            if c == '/' and next_c == '/' and not in_block_comment:
                in_line_comment = True
            elif c == '/' and next_c == '*' and not in_line_comment:
                in_block_comment = True
            elif c == '\n' and in_line_comment:
                in_line_comment = False
            elif c == '*' and next_c == '/' and in_block_comment:
                in_block_comment = False
                i += 1
        
        # Handle strings (when not in comment)
        if not in_line_comment and not in_block_comment and not in_char:
            if c == '"' and prev != '\\':
                in_string = not in_string
        
        # Handle char literals
        if not in_line_comment and not in_block_comment and not in_string:
            if c == "'" and prev != '\\':
                in_char = not in_char
        
        # Count braces (only when not in string/comment)
        if not in_string and not in_char and not in_line_comment and not in_block_comment:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
        
        i += 1
    
    return i - 1 if depth == 0 else None

def is_inside_string(content: str, pos: int) -> bool:
    """Check if position `pos` is inside a string literal."""
    in_string = False
    i = 0
    while i < pos:
        c = content[i]
        prev = content[i-1] if i > 0 else ''
        if c == '"' and prev != '\\':
            in_string = not in_string
        i += 1
    return in_string

def find_domain_assumption_in_block(block_content: str) -> Optional[Tuple[int, int, str]]:
    """
    Find the domain_assumption field line in a block.
    Returns (start_offset, end_offset, line_content) or None.
    
    Handles patterns:
    - domain_assumption: None,
    - domain_assumption: Some("..."),
    - domain_assumption,  (shorthand)
    - domain_assumption: decision.assumption,
    """
    # Pattern to match the full domain_assumption line including trailing comma and newline
    patterns = [
        # Standard: domain_assumption: None, or domain_assumption: Some(...),
        r'(\s*domain_assumption:\s*[^,\n]+,?\s*\n)',
        # With expression: domain_assumption: decision.assumption,
        r'(\s*domain_assumption:\s*\w+\.\w+,?\s*\n)',
        # Shorthand: domain_assumption,
        r'(\s*domain_assumption,\s*\n)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, block_content)
        if match:
            return (match.start(), match.end(), match.group(1).strip())
    
    return None

def find_struct_blocks(content: str, filename: str) -> List[StructBlock]:
    """Find all Rewrite and Step struct LITERALS in the content."""
    blocks = []
    
    # Pattern to find Rewrite { or Step { - may have path prefix
    pattern = re.compile(r'(\b\w+::)*(Rewrite|Step)\s*\{')
    
    for match in pattern.finditer(content):
        start_pos = match.start()
        brace_pos = match.end() - 1
        struct_type = match.group(2)  # "Rewrite" or "Step"
        
        # Check if inside a string literal
        if is_inside_string(content, start_pos):
            continue
        
        # Skip struct definitions (have 'pub struct' or 'struct' before)
        prefix_start = max(0, start_pos - 40)
        prefix = content[prefix_start:start_pos]
        if re.search(r'\b(pub\s+)?struct\s*$', prefix):
            continue
        
        # Find matching brace
        end_pos = find_matching_brace(content, brace_pos)
        if end_pos is None:
            print(f"WARNING: Unbalanced braces in {filename} at position {start_pos}")
            continue
        
        block_content = content[start_pos:end_pos+1]
        
        # Verify this is a real struct literal
        if struct_type == "Rewrite":
            # Rewrite must have new_expr and description
            if not re.search(r'\bnew_expr\s*[,:]', block_content):
                continue
            if not re.search(r'\bdescription\s*:', block_content):
                continue
        elif struct_type == "Step":
            # Step must have rule_name
            if not re.search(r'\brule_name\s*[,:]', block_content):
                continue
        
        # Find domain_assumption in block
        da_info = find_domain_assumption_in_block(block_content)
        
        start_line = content[:start_pos].count('\n') + 1
        end_line = content[:end_pos].count('\n') + 1
        
        blocks.append(StructBlock(
            file=filename,
            start_line=start_line,
            end_line=end_line,
            start_pos=start_pos,
            end_pos=end_pos,
            content=block_content,
            struct_type=struct_type,
            domain_assumption_line=da_info[2] if da_info else None,
            domain_assumption_pos=(da_info[0], da_info[1]) if da_info else None,
        ))
    
    return blocks

def remove_field_from_block(content: str, block: StructBlock) -> str:
    """Remove domain_assumption field from a single block."""
    if block.domain_assumption_pos is None:
        return content
    
    da_start, da_end = block.domain_assumption_pos
    
    # Calculate absolute positions in file content
    abs_start = block.start_pos + da_start
    abs_end = block.start_pos + da_end
    
    # Remove the line
    new_content = content[:abs_start] + content[abs_end:]
    
    return new_content

def remove_struct_field_declaration(content: str, filename: str) -> Tuple[str, List[str]]:
    """
    Remove the domain_assumption field declaration from struct definitions.
    Returns (modified_content, list_of_changes).
    """
    changes = []
    
    # Pattern for field declaration in Rewrite or Step struct
    # Matches: pub domain_assumption: Option<&'static str>,
    pattern = r'\n\s*pub\s+domain_assumption:\s*Option<[^>]+>,?\s*\n'
    
    match = re.search(pattern, content)
    if match:
        changes.append(f"  Removing struct field declaration at pos {match.start()}")
        content = content[:match.start()] + '\n' + content[match.end():]
    
    return content, changes

def process_file(filepath: Path, dry_run: bool = True) -> Tuple[int, int, List[str]]:
    """
    Process a single file.
    Returns (blocks_found, blocks_modified, messages)
    """
    messages = []
    
    try:
        content = filepath.read_text()
    except Exception as e:
        return 0, 0, [f"ERROR reading {filepath}: {e}"]
    
    original_content = content
    
    # Check for struct field declarations to remove
    is_struct_def_file = filepath.name in ['rule.rs', 'step.rs']
    struct_changes = []
    if is_struct_def_file:
        content, struct_changes = remove_struct_field_declaration(content, str(filepath))
    
    # Find struct literal blocks
    blocks = find_struct_blocks(content, str(filepath))
    
    # Filter to blocks that have domain_assumption
    needs_modification = [b for b in blocks if b.domain_assumption_pos is not None]
    
    if not needs_modification and not struct_changes:
        return 0, 0, []
    
    messages.append(f"\n📁 {filepath}")
    
    if struct_changes:
        messages.extend(struct_changes)
    
    if blocks:
        messages.append(f"   Found {len(blocks)} struct blocks, {len(needs_modification)} have domain_assumption")
    
    for block in needs_modification:
        messages.append(f"   Line {block.start_line}: {block.struct_type} → removing: {block.domain_assumption_line}")
    
    if dry_run:
        return len(blocks), len(needs_modification) + len(struct_changes), messages
    
    # Apply modifications (in reverse order to preserve positions)
    modified_content = content
    for block in sorted(needs_modification, key=lambda b: b.start_pos, reverse=True):
        modified_content = remove_field_from_block(modified_content, block)
    
    if modified_content != original_content:
        filepath.write_text(modified_content)
        messages.append(f"   ✅ Applied modifications")
    
    return len(blocks), len(needs_modification) + len(struct_changes), messages

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Remove domain_assumption field from Rewrite/Step structs')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Only report, do not modify (default)')
    parser.add_argument('--apply', action='store_true', help='Apply changes')
    parser.add_argument('--file', type=str, help='Process single file')
    parser.add_argument('--root', type=str, default='crates/cas_engine/src', help='Root directory to scan')
    args = parser.parse_args()
    
    dry_run = not args.apply
    
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
    else:
        print("⚠️  APPLY MODE - Files will be modified")
    
    if args.file:
        files = [Path(args.file)]
    else:
        root = Path(args.root)
        files = list(root.rglob('*.rs'))
    
    print(f"Scanning {len(files)} files...")
    
    total_blocks = 0
    total_needs_fix = 0
    all_messages = []
    
    for filepath in sorted(files):
        blocks, needs_fix, messages = process_file(filepath, dry_run)
        total_blocks += blocks
        total_needs_fix += needs_fix
        all_messages.extend(messages)
    
    # Print all messages
    for msg in all_messages:
        print(msg)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {total_blocks} struct blocks found, {total_needs_fix} domain_assumption fields to remove")
    
    if dry_run and total_needs_fix > 0:
        print(f"\nRun with --apply to modify files")

if __name__ == '__main__':
    main()
