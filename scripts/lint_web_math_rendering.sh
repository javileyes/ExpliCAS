#!/bin/bash
# =============================================================================
# Lint: the web must RENDER its LaTeX, never publish it as source
# =============================================================================
#
# Two defects put raw LaTeX in front of the reader, both found on
# `bignum(5^5^5)` (2026-08-01). Each is a pair of facts that only works
# together, and in both cases the halves live far apart — which is exactly how
# they drifted:
#
#   1. FRONTEND. A card whose result is a giant `<pre>` used to be typeset only
#      in its header, on the theory that "steps typeset themselves on expand".
#      They do not — `toggleSteps` never called MathJax — so every step of every
#      big-number/big-polynomial card showed `${\color{red}{\text{bignum}(…`.
#      The card is now typeset WHOLE and the `<pre>` opts out by class, so the
#      class and the `ignoreHtmlClass` that honours it must both survive.
#
#   2. ENGINE. `\text{…}` escapes that MathJax 3's text-mode parser does not
#      implement are echoed as source (`x\unicode{x5E}2`, `root\_sum`). The
#      table that knows which escapes are real is `cas_formatter::escape_for_
#      text_mode`; a second copy of it in cas_didactic drifted the same way, so
#      there must stay exactly one.
#
# =============================================================================

set -e

SCRIPT_DIR=$(dirname "$0")
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

INDEX="$ROOT_DIR/web/index.html"
CANONICAL_ESCAPE="crates/cas_formatter/src/escape.rs"

echo "==> Checking web math rendering invariants..."

ERRORS=0

# -----------------------------------------------------------------------------
# CHECK 1: the giant `<pre>` opts out of MathJax, and the config honours it.
# -----------------------------------------------------------------------------
echo "  [1/3] Checking the mathjax-ignore opt-out is wired at both ends..."

if ! grep -q "ignoreHtmlClass:.*mathjax-ignore" "$INDEX"; then
    echo -e "  ${RED}ERROR${NC}: web/index.html does not declare 'mathjax-ignore' in MathJax's ignoreHtmlClass"
    echo -e "         Fix: options: { ignoreHtmlClass: 'tex2jax_ignore|mathjax-ignore' }"
    ((ERRORS++))
fi

for line in $(grep -n 'class="poly-output' "$INDEX" | cut -d: -f1); do
    if ! sed -n "${line}p" "$INDEX" | grep -q "mathjax-ignore"; then
        echo -e "  ${RED}ERROR${NC}: web/index.html:$line renders .poly-output without mathjax-ignore"
        echo -e "         Fix: class=\"poly-output mathjax-ignore\" — MathJax must not scan the raw result"
        ((ERRORS++))
    fi
done

# -----------------------------------------------------------------------------
# CHECK 2: a card is typeset WHOLE — never a hand-picked sub-element of it.
# Typesetting `[header]` (or any other fragment) is the shape that starved the
# steps; the exclusion belongs in the class, not in the element picked here.
# -----------------------------------------------------------------------------
echo "  [2/3] Checking every card is typeset whole..."

if ! grep -q "MathJax.typesetPromise(\[card\])" "$INDEX"; then
    echo -e "  ${RED}ERROR${NC}: web/index.html never typesets a card whole"
    echo -e "         Fix: MathJax.typesetPromise([card]) in addExpressionCard"
    ((ERRORS++))
fi

if grep -nE "typesetPromise\(\[(header|.*querySelector)" "$INDEX" | grep -v "^\s*//" | grep -q .; then
    echo -e "  ${RED}ERROR${NC}: web/index.html typesets a FRAGMENT of a card:"
    grep -nE "typesetPromise\(\[(header|.*querySelector)" "$INDEX" | sed 's/^/           /'
    echo -e "         Fix: typeset the card and let the excluded subtree carry mathjax-ignore"
    ((ERRORS++))
fi

# -----------------------------------------------------------------------------
# CHECK 3: exactly one text-mode escape table.
# `\unicode{x5E}` / `\unicode{x7E}` (a caret or tilde spelled for math mode)
# only ever appear in one: they are meaningless anywhere else, and inside
# `\text{…}` they are the defect itself.
# -----------------------------------------------------------------------------
echo "  [3/3] Checking the text-mode escape table has exactly one home..."

for file in $(grep -rlE '\\\\unicode\{x(5E|7E)\}' "$ROOT_DIR/crates" --include="*.rs" 2>/dev/null || true); do
    rel="${file#"$ROOT_DIR"/}"
    case "$rel" in
        "$CANONICAL_ESCAPE") continue ;;
        */tests/*|*/tests.rs) continue ;;
    esac
    # A file may still NAME the defect in a doc comment or a test expectation;
    # what it must not do is build its own table.
    if grep -qE "'\^' *=>|'~' *=>|'_' *=>" "$file"; then
        echo -e "  ${RED}ERROR${NC}: $rel builds its own \\text{…} escape table"
        echo -e "         Fix: call cas_formatter::escape_for_text_mode — see $CANONICAL_ESCAPE"
        ((ERRORS++))
    fi
done

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
echo ""
if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}✗ Found $ERRORS error(s) - these MUST be fixed${NC}"
    exit 1
fi
echo -e "${GREEN}✔ Web math rendering invariants hold${NC}"
exit 0
