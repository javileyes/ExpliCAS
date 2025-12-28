#!/usr/bin/env python3
"""
Rewrite Field Injector - Safe AST-lite approach

This script safely finds `Rewrite { ... }` struct literals in Rust code
by counting balanced braces, and injects the `assumption_events` field.

SAFE because:
1. Uses brace counting to find complete struct blocks
2. Checks we're not inside string literals
3. Verifies the block contains expected Rewrite fields
4. Reports all changes before applying (dry-run by default)

Usage:
    python inject_assumption_events.py --dry-run     # Just report what would change
    python inject_assumption_events.py --apply       # Apply changes
    python inject_assumption_events.py --file X.rs   # Process single file
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RewriteBlock:
    """Represents a found Rewrite { ... } block"""
    file: str
    start_line: int
    end_line: int
    start_pos: int
    end_pos: int
    content: str
    has_assumption_events: bool
    has_domain_assumption: bool

def is_inside_string(content: str, pos: int) -> bool:
    """
    Check if position `pos` is inside a string literal.
    Handles both "..." and r#"..."# raw strings.
    """
    # Simple heuristic: count unescaped quotes before position
    in_string = False
    in_raw_string = False
    i = 0
    while i < pos:
        c = content[i]
        
        # Check for raw string start: r#" or r##" etc
        if not in_string and not in_raw_string and c == 'r' and i + 1 < pos:
            # Count # symbols
            j = i + 1
            hash_count = 0
            while j < pos and content[j] == '#':
                hash_count += 1
                j += 1
            if j < pos and content[j] == '"':
                in_raw_string = True
                i = j + 1
                continue
        
        # Regular string
        if not in_raw_string:
            if c == '"' and (i == 0 or content[i-1] != '\\'):
                in_string = not in_string
        
        # Raw string end (simplified - doesn't handle hash counting)
        if in_raw_string and c == '"':
            in_raw_string = False
            
        i += 1
    
    return in_string or in_raw_string

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

def find_rewrite_blocks(content: str, filename: str) -> List[RewriteBlock]:
    """Find all Rewrite { ... } struct LITERALS (not definitions) in the content."""
    blocks = []
    
    # Pattern to find `Rewrite {` - may have path prefix like `crate::rule::Rewrite {`
    # We want literals, not struct definitions
    pattern = re.compile(r'(\b\w+::)*Rewrite\s*\{')
    
    for match in pattern.finditer(content):
        start_pos = match.start()
        brace_pos = match.end() - 1  # Position of the {
        
        # Check if inside a string literal
        if is_inside_string(content, start_pos):
            continue
        
        # IMPORTANT: Skip struct definitions (have 'pub struct' or 'struct' before)
        # Look at the ~30 chars before the match
        prefix_start = max(0, start_pos - 30)
        prefix = content[prefix_start:start_pos]
        if re.search(r'\b(pub\s+)?struct\s*$', prefix):
            continue
        
        # Find matching brace
        end_pos = find_matching_brace(content, brace_pos)
        if end_pos is None:
            print(f"WARNING: Unbalanced braces in {filename} at position {start_pos}")
            continue
        
        # Extract block content
        block_content = content[start_pos:end_pos+1]
        
        # Verify this looks like a struct literal (has field: value pattern)
        # Real Rewrite literals MUST have ALL of these required fields:
        # - new_expr: or new_expr, (the result expression)
        # - description: (the step description)  
        # - domain_assumption: (always present, even if None)
        # This triple-check prevents matching random blocks
        has_new_expr = re.search(r'\bnew_expr\s*[,:]', block_content)
        has_description = re.search(r'\bdescription\s*:', block_content)
        has_domain_assumption = re.search(r'\bdomain_assumption\s*:', block_content)
        
        if not has_new_expr or not has_description or not has_domain_assumption:
            # Not a real Rewrite literal - missing required fields
            continue
        
        # Check if it has the expected fields
        has_domain_assumption = 'domain_assumption:' in block_content
        has_assumption_events = 'assumption_events:' in block_content
        
        # Calculate line numbers
        start_line = content[:start_pos].count('\n') + 1
        end_line = content[:end_pos].count('\n') + 1
        
        blocks.append(RewriteBlock(
            file=filename,
            start_line=start_line,
            end_line=end_line,
            start_pos=start_pos,
            end_pos=end_pos,
            content=block_content,
            has_assumption_events=has_assumption_events,
            has_domain_assumption=has_domain_assumption,
        ))
    
    return blocks

def inject_field(block: RewriteBlock, content: str) -> str:
    """
    Inject assumption_events field into a Rewrite block.
    Returns the modified full file content.
    """
    # Find the position just before the closing }
    block_content = content[block.start_pos:block.end_pos+1]
    
    # Find the last field (look for the last comma before })
    # We'll insert after domain_assumption if it exists, otherwise before }
    
    # Find position of last } in block
    last_brace = block.end_pos
    
    # Determine indentation from the block
    # Find the line with domain_assumption or the last field
    lines = block_content.split('\n')
    indent = "            "  # Default 12 spaces
    
    for line in lines:
        if 'domain_assumption:' in line or 'after_local:' in line:
            # Extract indentation from this line
            stripped = line.lstrip()
            indent = line[:len(line) - len(stripped)]
            break
    
    # Build the new field line
    new_field = f"{indent}assumption_events: Default::default(),\n"
    
    # Insert before the closing brace
    # Find the position of the final } in the original content
    # We need to insert the new field before it
    
    # Get content before the block's closing brace
    before_brace = content[:block.end_pos]
    after_brace = content[block.end_pos:]
    
    # Check if we need to add a comma to the previous field
    # Find the last non-whitespace before the }
    pre_brace = before_brace.rstrip()
    needs_comma = not pre_brace.endswith(',') and not pre_brace.endswith('{')
    
    if needs_comma:
        # Add comma to last field
        new_content = pre_brace + ',\n' + new_field + indent.rstrip() + after_brace
    else:
        new_content = before_brace.rstrip() + '\n' + new_field + indent.rstrip() + after_brace
    
    return new_content

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
    
    blocks = find_rewrite_blocks(content, str(filepath))
    
    if not blocks:
        return 0, 0, []
    
    # Filter to blocks that need modification
    needs_modification = [b for b in blocks if not b.has_assumption_events]
    
    messages.append(f"\n📁 {filepath}")
    messages.append(f"   Found {len(blocks)} Rewrite blocks, {len(needs_modification)} need modification")
    
    for block in blocks:
        status = "✅" if block.has_assumption_events else "❌ NEEDS FIX"
        messages.append(f"   Line {block.start_line}-{block.end_line}: {status}")
        if not block.has_assumption_events:
            # Show first 100 chars of block for verification
            preview = block.content[:100].replace('\n', ' ')
            messages.append(f"      Preview: {preview}...")
    
    if dry_run:
        return len(blocks), len(needs_modification), messages
    
    # Apply modifications (in reverse order to preserve positions)
    if needs_modification:
        modified_content = content
        for block in sorted(needs_modification, key=lambda b: b.start_pos, reverse=True):
            modified_content = inject_field(block, modified_content)
        
        filepath.write_text(modified_content)
        messages.append(f"   ✅ Applied {len(needs_modification)} modifications")
    
    return len(blocks), len(needs_modification), messages

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Inject assumption_events field into Rewrite structs')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Only report, do not modify (default)')
    parser.add_argument('--apply', action='store_true', help='Apply changes')
    parser.add_argument('--file', type=str, help='Process single file')
    parser.add_argument('--root', type=str, default='crates', help='Root directory to scan')
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
    print(f"SUMMARY: {total_blocks} Rewrite blocks found, {total_needs_fix} need modification")
    
    if dry_run and total_needs_fix > 0:
        print(f"\nRun with --apply to modify files")

if __name__ == '__main__':
    main()
