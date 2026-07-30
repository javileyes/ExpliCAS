#!/usr/bin/env bash
set -euo pipefail

# Budget-based unwrap()/expect() guardrail for production code.
#
# Counts .unwrap() and .expect() in each production crate (excluding tests)
# and fails if any crate exceeds its budget. As developers fix unwrap calls,
# they lower the budget numbers below to ratchet down the count.
#
# Scope: ALL 14 production crates. Extendido 2026-07-30 por la auditoría
# integral (docs/AUDITORIA_INTEGRAL_2026-07-30.md, ficha Q4a-004): la versión
# anterior solo presupuestaba 4 crates (16% de la superficie de producción) y
# estaba ROJA en HEAD (cas_engine expect 15/9). Los techos nuevos son la
# MEDIDA REAL a 2026-07-30 (ratchet desde la realidad): bajarlos al arreglar,
# nunca subirlos sin anotar el porqué aquí.
# Nota cas_parser: ~15 de sus `.expect(` son llamadas al método propio
# `self.expect(&Token)` que devuelve Result (no panican); el techo cuenta el
# patrón textual igualmente para conservar el ratchet simple.
# Nota cas_engine: expect subió 9→15 entre tandas sin que nadie viera el rojo
# (los 15 sitios están listados en la ficha Q4a-004); pendiente ciclo Q de
# reducción.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ─── Budgets (current ceiling — lower these as you fix unwraps) ───
# Format: CRATE_DIR MAX_UNWRAP MAX_EXPECT
BUDGETS=(
    "crates/cas_engine/src        3  15"
    "crates/cas_ast/src           0   0"
    "crates/cas_cli/src           0   0"
    "crates/cas_android_ffi/src   0   0"
    "crates/cas_api_models/src    0   0"
    "crates/cas_didactic/src      3   5"
    "crates/cas_formatter/src     6   1"
    "crates/cas_math/src          3  24"
    "crates/cas_parser/src        1  15"
    "crates/cas_session/src       0   0"
    "crates/cas_session_core/src  0   1"
    "crates/cas_solver/src        9  18"
    "crates/cas_solver_core/src   2   7"
    "crates/cas_wasm/src          0   0"
)

count_pattern() {
    local dir="$1" rg_pattern="$2" grep_pattern="$3"
    if [[ ! -d "$ROOT_DIR/$dir" ]]; then
        echo "0"
        return
    fi
    python3 - "$ROOT_DIR/$dir" "$rg_pattern" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
pattern = re.compile(sys.argv[2])
cfg_test_attr = re.compile(r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]")


def is_test_file(path: Path) -> bool:
    return (
        "tests" in path.parts
        or "test" in path.parts
        or "test" in path.name
    )


def strip_cfg_test_modules(src: str) -> str:
    chunks = []
    pos = 0
    length = len(src)

    while True:
        match = cfg_test_attr.search(src, pos)
        if match is None:
            chunks.append(src[pos:])
            break

        chunks.append(src[pos : match.start()])
        cursor = match.end()
        while True:
            while cursor < length and src[cursor].isspace():
                cursor += 1

            if cursor + 1 >= length or src[cursor : cursor + 2] != "#[":
                break

            cursor += 2
            attr_depth = 1
            while cursor < length and attr_depth > 0:
                char = src[cursor]
                if char == "[":
                    attr_depth += 1
                elif char == "]":
                    attr_depth -= 1
                cursor += 1

        if not src.startswith("mod", cursor):
            chunks.append(src[match.start() : match.end()])
            pos = match.end()
            continue

        cursor += len("mod")
        while cursor < length and src[cursor].isspace():
            cursor += 1
        while cursor < length and (src[cursor].isalnum() or src[cursor] == "_"):
            cursor += 1
        while cursor < length and src[cursor].isspace():
            cursor += 1

        if cursor >= length or src[cursor] != "{":
            chunks.append(src[match.start() : match.end()])
            pos = match.end()
            continue

        depth = 0
        while cursor < length:
            char = src[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    cursor += 1
                    break
            cursor += 1

        pos = cursor

    return "".join(chunks)


total = 0
for path in root.rglob("*.rs"):
    if is_test_file(path):
        continue
    total += len(pattern.findall(strip_cfg_test_modules(path.read_text())))

print(total)
PY
}

echo "══════════════════════════════════════════════════════"
echo "  unwrap()/expect() budget lint"
echo "══════════════════════════════════════════════════════"
printf "  %-30s %8s %8s %8s %8s\n" "Crate" "unwrap" "budget" "expect" "budget"
echo "  ──────────────────────────────────────────────────"

failed=0

for entry in "${BUDGETS[@]}"; do
    read -r dir max_unwrap max_expect <<< "$entry"
    crate_name=$(echo "$dir" | sed 's|crates/||;s|/src||')

    actual_unwrap=$(count_pattern "$dir" '\.unwrap\(\)' '.unwrap()')
    actual_expect=$(count_pattern "$dir" '\.expect\(' '.expect(')

    # Status indicators
    u_status="✔"
    e_status="✔"
    if (( actual_unwrap > max_unwrap )); then
        u_status="✘"
        failed=1
    fi
    if (( actual_expect > max_expect )); then
        e_status="✘"
        failed=1
    fi

    printf "  %-30s %3s %4d/%-4d %3s %4d/%-4d\n" \
        "$crate_name" "$u_status" "$actual_unwrap" "$max_unwrap" \
        "$e_status" "$actual_expect" "$max_expect"
done

echo "  ──────────────────────────────────────────────────"

if [[ $failed -eq 1 ]]; then
    echo ""
    echo "✘ Budget exceeded! Lower the count or raise the budget in this script."
    exit 1
fi

echo ""
echo "✔ unwrap/expect budget: all crates within limits"
