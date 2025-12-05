#!/bin/bash
# Script para tests de SOLVER STRESS TESTS
# Enfoque: Tests comprehensivos del solver en 18 niveles de complejidad
# Desde ecuaciones lineales básicas hasta casos ultra-complejos con logaritmos
# Uso: ./run_cli_solver_tests.sh

OUTPUT_FILE="output_solver_stress_test.txt"
CLI_CMD="cargo run -p cas_cli --release"

echo "==========================================================" > "$OUTPUT_FILE"
echo "CLI Solver Stress Tests - 18 Niveles de Complejidad" >> "$OUTPUT_FILE"
echo "From Basic Linear to Ultra-Complex Mixed Operations" >> "$OUTPUT_FILE"
echo "Fecha: $(date)" >> "$OUTPUT_FILE"
echo "==========================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

COMMANDS_FILE="/tmp/solver_stress_commands.txt"
echo "steps normal" > "$COMMANDS_FILE"

echo "Preparando stress tests del solver (76 tests)..."

# ========================================
# LEVEL 1: Basic Linear Equations
# ========================================

echo "solve x + 5 = 10, x" >> "$COMMANDS_FILE"
echo "solve 3*x - 7 = 14, x" >> "$COMMANDS_FILE"
echo "solve 2*x + 3 = 5*x - 12, x" >> "$COMMANDS_FILE"

# ========================================
# LEVEL 2: Linear Inequalities
# ========================================

echo "solve x + 3 > 7, x" >> "$COMMANDS_FILE"
echo "solve -2*x < 10, x" >> "$COMMANDS_FILE"
echo "solve 3*x - 5 >= 2*x + 1, x" >> "$COMMANDS_FILE"

# ========================================
# LEVEL 3: Quadratic Equations
# ========================================

echo "solve x^2 = 16, x" >> "$COMMANDS_FILE"
echo "solve x^2 - 5*x + 6 = 0, x" >> "$COMMANDS_FILE"
echo "solve x^2 - 6*x + 9 = 0, x" >> "$COMMANDS_FILE"

# ========================================
# LEVEL 4: Quadratic Inequalities
# ========================================

echo "solve x^2 < 9, x" >> "$COMMANDS_FILE"
echo "solve x^2 > 4, x" >> "$COMMANDS_FILE"
echo "solve (x - 1)*(x - 3) > 0, x" >> "$COMMANDS_FILE"

# ========================================
# LEVEL 5: Rational Equations
# ========================================

echo "solve 1/x = 2, x" >> "$COMMANDS_FILE"
echo "solve 1/x = 2/3, x" >> "$COMMANDS_FILE"
echo "solve (x + 1)/(x - 2) = 3, x" >> "$COMMANDS_FILE"

# ========================================
# LEVEL 6: Rational Inequalities
# ========================================

echo "solve 1/x > 0, x" >> "$COMMANDS_FILE"
echo "solve 1/x < 0, x" >> "$COMMANDS_FILE"
echo "solve (x - 1)/(x + 2) > 0, x" >> "$COMMANDS_FILE"

# ========================================
# LEVEL 7: Absolute Value
# ========================================

echo "solve abs(x) = 5, x" >> "$COMMANDS_FILE"
echo "solve abs(x) < 3, x" >> "$COMMANDS_FILE"
echo "solve abs(2*x - 3) = 7, x" >> "$COMMANDS_FILE"

# ========================================
# LEVEL 8: Mixed Complex Cases
# ========================================

echo "solve x^2/(x - 1) = x + 2, x" >> "$COMMANDS_FILE"
echo "solve 1/(x + 1) = 1/(2*x - 3), x" >> "$COMMANDS_FILE"
echo "solve (x - 1)*(x - 2)*(x - 3) > 0, x" >> "$COMMANDS_FILE"

# ========================================
# LEVEL 9: Edge Cases
# ========================================

echo "solve x + 5 = x + 3, x" >> "$COMMANDS_FILE"
echo "solve x + 1 = x + 1, x" >> "$COMMANDS_FILE"
echo "solve 1000000*x + 500000 = 2500000, x" >> "$COMMANDS_FILE"
echo "solve (1/10000)*x = 5/10000, x" >> "$COMMANDS_FILE"

# ========================================
# LEVEL 10: Advanced Rational Inequalities
# ========================================

echo "solve 1/((x - 1)*(x - 3)) > 0, x" >> "$COMMANDS_FILE"
echo "solve (x - 1)/(x + 1) = (x + 1)/(x - 1), x" >> "$COMMANDS_FILE"

# ========================================
# LEVEL 11: Advanced Absolute & Radicals
# ========================================

echo "solve abs(abs(x) - 2) = 1, x" >> "$COMMANDS_FILE"
echo "solve (x + 5)^(1/2) = 3, x" >> "$COMMANDS_FILE"

# ========================================
# LEVEL 12: Pathological Edge Cases
# ========================================

echo "solve a*x + b = 0, x" >> "$COMMANDS_FILE"
echo "solve x^3 - 6*x^2 + 11*x - 6 = 0, x" >> "$COMMANDS_FILE"
echo "solve 1/(x - 1) = 1/(x - 1) + 1, x" >> "$COMMANDS_FILE"

# ========================================
# LEVEL 13: Complex Mixed Operations
# ========================================

echo "solve abs(x/(x - 1)) = 2, x" >> "$COMMANDS_FILE"
echo "solve (x^2 + 1)/(x^2 - 4) > 0, x" >> "$COMMANDS_FILE"
echo "solve 1/x + 1/(x - 1) = 1, x" >> "$COMMANDS_FILE"

# ========================================
# LEVEL 14: Deeply Nested Expressions
# ========================================

echo "solve abs(abs(abs(x) - 1) - 1) = 0, x" >> "$COMMANDS_FILE"
echo "solve 1/(1 + 1/x) = 2, x" >> "$COMMANDS_FILE"
echo "solve ((x^2)^2)^2 = 64, x" >> "$COMMANDS_FILE"

# ========================================
# LEVEL 15: Extreme Numerical Cases
# ========================================

echo "solve x^100 = 2^100, x" >> "$COMMANDS_FILE"
echo "solve x^(3/2) = 8, x" >> "$COMMANDS_FILE"
echo "solve x + 2*x + 3*x + 4*x + 5*x = 150, x" >> "$COMMANDS_FILE"
echo "solve x - 2*x + 3*x - 4*x + 5*x = 9, x" >> "$COMMANDS_FILE"

# ========================================
# LEVEL 16: Maximum Stress
# ========================================

echo "solve (x - 1)*(x - 2)*(x - 3)*(x - 4)*(x - 5) = 0, x" >> "$COMMANDS_FILE"
echo "solve (x + 1)/(x - 1) + (x - 1)/(x + 1) = 2, x" >> "$COMMANDS_FILE"
echo "solve abs(x/(x + 1)) < 1/2, x" >> "$COMMANDS_FILE"
echo "solve (x + 1)^2 - 2*(x + 1) + 1 = 0, x" >> "$COMMANDS_FILE"
echo "solve (x - 1)^(1/2) + (x + 1)^(1/2) = 2, x" >> "$COMMANDS_FILE"
echo "solve (x^2 - 1)/(x - 1) = x + 2, x" >> "$COMMANDS_FILE"
echo "solve 1/(1/(1/(1/x))) = 2, x" >> "$COMMANDS_FILE"

# ========================================
# LEVEL 17: Logarithmic & Exponential
# ========================================

echo "solve log(10, x) = 2, x" >> "$COMMANDS_FILE"
echo "solve ln(x) = 3, x" >> "$COMMANDS_FILE"
echo "solve exp(x) = exp(3), x" >> "$COMMANDS_FILE"
echo "solve log(2, 2*x + 4) = 3, x" >> "$COMMANDS_FILE"
echo "solve exp(x^2) = exp(4), x" >> "$COMMANDS_FILE"
echo "solve log(2, x) = log(2, 8), x" >> "$COMMANDS_FILE"
echo "solve ln(exp(x)) = 5, x" >> "$COMMANDS_FILE"
echo "solve log(10, x) + log(10, 2) = 2, x" >> "$COMMANDS_FILE"

# ========================================
# LEVEL 18: Ultra-Complex Mixed
# ========================================

echo "solve abs(log(10, x))/2 = 1, x" >> "$COMMANDS_FILE"
echo "solve x^(1/2)/(x - 1) = 2, x" >> "$COMMANDS_FILE"
echo "solve abs(x/abs(x - 1)) = 2, x" >> "$COMMANDS_FILE"
echo "solve x^2 + abs(x) - 6 < 0, x" >> "$COMMANDS_FILE"
echo "solve (x + 1)^(1/2)/(x - 1)^(1/2) = 2, x" >> "$COMMANDS_FILE"
echo "solve abs((x^2 - 4)/(x - 2)) = 3, x" >> "$COMMANDS_FILE"
echo "solve x^(1/2) + (x + 7)^(1/2) = 7, x" >> "$COMMANDS_FILE"
echo "solve abs(x)/(x^2 + 1) > 1/2, x" >> "$COMMANDS_FILE"
echo "solve (x^3 - 8)/(x - 2) = x^2 + 2*x + 4, x" >> "$COMMANDS_FILE"
echo "solve (x - 1)*abs(x + 1) = 0, x" >> "$COMMANDS_FILE"
echo "solve abs((abs(x - 1))^(1/2)) = 2, x" >> "$COMMANDS_FILE"
echo "solve (a*x + b)/(c*x + d) = 2, x" >> "$COMMANDS_FILE"

# Añadir comando de salida
echo "exit" >> "$COMMANDS_FILE"

echo "Ejecutando CLI con $(wc -l <$COMMANDS_FILE) comandos de solver stress tests..."

# Ejecutar el CLI
$CLI_CMD < "$COMMANDS_FILE" >> "$OUTPUT_FILE" 2>&1

# Cleanup
rm -f "$COMMANDS_FILE"

echo "" >> "$OUTPUT_FILE"
echo "==========================================================" >> "$OUTPUT_FILE"
echo "Solver Stress Tests completados: $(date)" >> "$OUTPUT_FILE"
echo "==========================================================" >> "$OUTPUT_FILE"

echo ""
echo "✅ Solver Stress Tests completados!"
echo "📄 Output guardado en: $OUTPUT_FILE"
echo "📊 Líneas totales: $(wc -l <$OUTPUT_FILE)"
echo "📝 Tests realizados por nivel:"
echo "   LEVEL 1 (Linear Equations): 3 tests"
echo "   LEVEL 2 (Linear Inequalities): 3 tests"
echo "   LEVEL 3 (Quadratic Equations): 3 tests"
echo "   LEVEL 4 (Quadratic Inequalities): 3 tests"
echo "   LEVEL 5 (Rational Equations): 3 tests"
echo "   LEVEL 6 (Rational Inequalities): 3 tests"
echo "   LEVEL 7 (Absolute Value): 3 tests"
echo "   LEVEL 8 (Mixed Complex): 3 tests"
echo "   LEVEL 9 (Edge Cases): 4 tests"
echo "   LEVEL 10 (Advanced Rational Inequalities): 2 tests"
echo "   LEVEL 11 (Advanced Absolute & Radicals): 2 tests"
echo "   LEVEL 12 (Pathological Edge Cases): 3 tests"
echo "   LEVEL 13 (Complex Mixed Operations): 3 tests"
echo "   LEVEL 14 (Deeply Nested): 3 tests"
echo "   LEVEL 15 (Extreme Numerical): 4 tests"
echo "   LEVEL 16 (Maximum Stress): 7 tests"
echo "   LEVEL 17 (Logarithmic & Exponential): 8 tests"
echo "   LEVEL 18 (Ultra-Complex Mixed): 12 tests"
echo "   ──────────────────────────────────────"
echo "   Total: 66 expresiones de solver"
echo ""
echo "Para ver el output:"
echo "  cat $OUTPUT_FILE"
echo "  less $OUTPUT_FILE"
echo "  grep 'Result:' $OUTPUT_FILE | wc -l  # Contar resultados"
echo "  grep 'solve' $OUTPUT_FILE | head -20  # Ver primeras 20 soluciones"
echo "  grep 'Empty Set' $OUTPUT_FILE  # Ver casos sin solución"
echo "  grep 'Union' $OUTPUT_FILE  # Ver casos con múltiples intervalos"
