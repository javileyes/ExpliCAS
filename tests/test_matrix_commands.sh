#!/bin/bash
# Test script for matrix CLI commands

echo "Testing det command:"
echo "det [[1, 2], [3, 4]]" | cargo run -p cas_cli 2>/dev/null | grep -A2 "Result:"

echo ""
echo "Testing transpose command:"
echo "transpose [[1, 2, 3], [4, 5, 6]]" | cargo run -p cas_cli 2>/dev/null | grep -A2 "Result:"

echo ""
echo "Testing trace command:"
echo "trace [[1, 2], [3, 4]]" | cargo run -p cas_cli 2>/dev/null | grep -A2 "Result:"

echo ""
echo "Testing help det:"
echo "help det" | cargo run -p cas_cli 2>/dev/null | head -7

echo ""
echo "Testing general help (matrix section):"
echo "help" | cargo run -p cas_cli 2>/dev/null | grep -A4 "Matrix Operations:"
