#!/bin/bash
# Script para ejecutar tests del CLI y guardar output
# Uso: ./run_cli_tests.sh

OUTPUT_FILE="output_test.txt"
CLI_CMD="cargo run -p cas_cli --release"

echo "==================================================" > "$OUTPUT_FILE"
echo "CLI Tests Output - Modo NORMAL" >> "$OUTPUT_FILE"
echo "Fecha: $(date)" >> "$OUTPUT_FILE"
echo "==================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

COMMANDS_FILE="/tmp/all_cli_commands.txt"
echo "steps normal" > "$COMMANDS_FILE"

echo "Preparando tests exhaustivos del CLI..."

# ========================================
# RAÍCES (Root Simplification)
# ========================================
echo "simplify sqrt(12)" >> "$COMMANDS_FILE"
echo "simplify sqrt(72)" >> "$COMMANDS_FILE"
echo "simplify 16^(1/3)" >> "$COMMANDS_FILE"
echo "simplify 54^(1/3)" >> "$COMMANDS_FILE"
echo "simplify 32^(1/4)" >> "$COMMANDS_FILE"
echo "simplify 243^(1/4)" >> "$COMMANDS_FILE"
echo "simplify sqrt(8/9)" >> "$COMMANDS_FILE"
echo "simplify sqrt(12/25)" >> "$COMMANDS_FILE"
echo "simplify sqrt(12) + sqrt(27)" >> "$COMMANDS_FILE"
echo "simplify sqrt(8) + sqrt(2)" >> "$COMMANDS_FILE"
echo "simplify 16^(1/3) + 54^(1/3)" >> "$COMMANDS_FILE"
echo "simplify sqrt(8) / 16^(1/3)" >> "$COMMANDS_FILE"
echo "simplify sqrt(32) / sqrt(2)" >> "$COMMANDS_FILE"
echo "simplify (sqrt(12) + sqrt(27)) / sqrt(3)" >> "$COMMANDS_FILE"
echo "simplify sqrt(2) * 2^(1/3)" >> "$COMMANDS_FILE"
echo "simplify sqrt(8) * sqrt(2)" >> "$COMMANDS_FILE"
echo "simplify sqrt(3200)" >> "$COMMANDS_FILE"
echo "simplify 8000^(1/3)" >> "$COMMANDS_FILE"
echo "simplify sqrt(8 * x^2)" >> "$COMMANDS_FILE"
echo "simplify sqrt(12 * x^3)" >> "$COMMANDS_FILE"

# Root Denesting (Advanced)
echo "simplify sqrt(3 + 2*sqrt(2))" >> "$COMMANDS_FILE"
echo "simplify sqrt(5 + 2*sqrt(6))" >> "$COMMANDS_FILE"
echo "simplify sqrt(7 - 2*sqrt(10))" >> "$COMMANDS_FILE"

# ========================================
# ÁLGEBRA
# ========================================
echo "simplify 2*x + 3*x" >> "$COMMANDS_FILE"
echo "simplify x^2 + 2*x + x^2" >> "$COMMANDS_FILE"
echo "simplify 2*(x + 1)" >> "$COMMANDS_FILE"
echo "simplify x*(x + 2)" >> "$COMMANDS_FILE"
echo "simplify (x + 1)*(x + 2)" >> "$COMMANDS_FILE"
echo "simplify (x + 1)^2" >> "$COMMANDS_FILE"
echo "simplify (x + 1)^3" >> "$COMMANDS_FILE"
echo "simplify (x - 1)^2" >> "$COMMANDS_FILE"
echo "simplify x^2 - 1" >> "$COMMANDS_FILE"
echo "simplify x^2 - 4" >> "$COMMANDS_FILE"
echo "simplify x^2 + 2*x + 1" >> "$COMMANDS_FILE"
echo "simplify x^3 - x" >> "$COMMANDS_FILE"
echo "simplify x^3 - 8" >> "$COMMANDS_FILE"
echo "simplify x^4 - 1" >> "$COMMANDS_FILE"
echo "simplify (x^2 - 1)/(x - 1)" >> "$COMMANDS_FILE"
echo "simplify (x^2 - 4)/(x + 2)" >> "$COMMANDS_FILE"
echo "simplify (x^3 - 1)/(x - 1)" >> "$COMMANDS_FILE"
echo "simplify a*x + b*x" >> "$COMMANDS_FILE"
echo "simplify (a + b)*x" >> "$COMMANDS_FILE"
echo "simplify x*(a + b)" >> "$COMMANDS_FILE"

# Collect/Grouping
echo "simplify a*x + b*x + c" >> "$COMMANDS_FILE"
echo "simplify x*y + x*z" >> "$COMMANDS_FILE"
echo "simplify 2*x*y + 3*x*y" >> "$COMMANDS_FILE"

# Fractions
echo "simplify 1/2 + 1/3" >> "$COMMANDS_FILE"
echo "simplify 1/x + 1/y" >> "$COMMANDS_FILE"
echo "simplify (x+1)/x + 1/x" >> "$COMMANDS_FILE"
echo "simplify x/(x+1) + 1/(x+1)" >> "$COMMANDS_FILE"

# Nested operations
echo "simplify ((x+1)*(x-1))^2" >> "$COMMANDS_FILE"
echo "simplify (x^2 - 1)^2" >> "$COMMANDS_FILE"

# ========================================
# EXPONENTES Y POTENCIAS
# ========================================
echo "simplify 2^3 * 2^4" >> "$COMMANDS_FILE"
echo "simplify (2^3)^4" >> "$COMMANDS_FILE"
echo "simplify (x^2)^3" >> "$COMMANDS_FILE"
echo "simplify (x*y)^2" >> "$COMMANDS_FILE"
echo "simplify (x/y)^2" >> "$COMMANDS_FILE"
echo "simplify 2^(-3)" >> "$COMMANDS_FILE"
echo "simplify x^(-2)" >> "$COMMANDS_FILE"
echo "simplify x^0" >> "$COMMANDS_FILE"
echo "simplify x^1" >> "$COMMANDS_FILE"
echo "simplify (2*x)^3" >> "$COMMANDS_FILE"
echo "simplify x^2 * x^3" >> "$COMMANDS_FILE"
echo "simplify x^a * x^b" >> "$COMMANDS_FILE"

# Power identities
echo "simplify (a*b)^n" >> "$COMMANDS_FILE"
echo "simplify (a/b)^n" >> "$COMMANDS_FILE"

# ========================================
# LOGARITMOS
# ========================================
echo "simplify ln(x*y)" >> "$COMMANDS_FILE"
echo "simplify ln(x/y)" >> "$COMMANDS_FILE"
echo "simplify ln(x^2)" >> "$COMMANDS_FILE"
echo "simplify ln(x^n)" >> "$COMMANDS_FILE"
echo "simplify ln(x^2 * y) - 2*ln(x)" >> "$COMMANDS_FILE"
echo "simplify ln(x*y*z)" >> "$COMMANDS_FILE"
echo "simplify ln(x) + ln(y)" >> "$COMMANDS_FILE"
echo "simplify ln(x) - ln(y)" >> "$COMMANDS_FILE"
echo "simplify 2*ln(x)" >> "$COMMANDS_FILE"
echo "simplify log(10, 100)" >> "$COMMANDS_FILE"
echo "simplify log(10, 1000)" >> "$COMMANDS_FILE"
echo "simplify log(2, 8)" >> "$COMMANDS_FILE"
echo "simplify ln(e)" >> "$COMMANDS_FILE"
echo "simplify ln(1)" >> "$COMMANDS_FILE"
echo "simplify e^(ln(x))" >> "$COMMANDS_FILE"
echo "simplify ln(e^x)" >> "$COMMANDS_FILE"

# Log properties
echo "simplify log(b, b^x)" >> "$COMMANDS_FILE"
echo "simplify b^(log(b, x))" >> "$COMMANDS_FILE"

# ========================================
# TRIGONOMETRÍA
# ========================================

# Basic identities
echo "simplify sin(2*x)" >> "$COMMANDS_FILE"
echo "simplify cos(2*x)" >> "$COMMANDS_FILE"
echo "simplify sin(x)^2 + cos(x)^2" >> "$COMMANDS_FILE"
echo "simplify 1 - sin(x)^2" >> "$COMMANDS_FILE"
echo "simplify 1 - cos(x)^2" >> "$COMMANDS_FILE"
echo "simplify tan(x)" >> "$COMMANDS_FILE"
echo "simplify sin(x)/cos(x)" >> "$COMMANDS_FILE"

# Special values
echo "simplify sin(0)" >> "$COMMANDS_FILE"
echo "simplify cos(0)" >> "$COMMANDS_FILE"
echo "simplify sin(pi/2)" >> "$COMMANDS_FILE"
echo "simplify cos(pi/2)" >> "$COMMANDS_FILE"
echo "simplify sin(pi)" >> "$COMMANDS_FILE"
echo "simplify cos(pi)" >> "$COMMANDS_FILE"
echo "simplify sin(pi/6)" >> "$COMMANDS_FILE"
echo "simplify cos(pi/4)" >> "$COMMANDS_FILE"
echo "simplify sin(pi/3)" >> "$COMMANDS_FILE"
echo "simplify tan(pi/4)" >> "$COMMANDS_FILE"

# Angle sum/difference
echo "simplify sin(x+y)" >> "$COMMANDS_FILE"
echo "simplify cos(x+y)" >> "$COMMANDS_FILE"
echo "simplify sin(x-y)" >> "$COMMANDS_FILE"
echo "simplify cos(x-y)" >> "$COMMANDS_FILE"

# Triple angle
echo "simplify sin(3*x)" >> "$COMMANDS_FILE"

# Products
echo "simplify sin(x)*cos(x)" >> "$COMMANDS_FILE"
echo "simplify 2*sin(x)*cos(x)" >> "$COMMANDS_FILE"

# Powers
echo "simplify sin(x)^2" >> "$COMMANDS_FILE"
echo "simplify cos(x)^2" >> "$COMMANDS_FILE"
echo "simplify sin(x)^3" >> "$COMMANDS_FILE"

# ========================================
# CÁLCULO - DERIVADAS
# ========================================
# echo "simplify diff(x, x)" >> "$COMMANDS_FILE"
# echo "simplify diff(x^2, x)" >> "$COMMANDS_FILE"
# echo "simplify diff(x^3, x)" >> "$COMMANDS_FILE"
# echo "simplify diff(x^n, x)" >> "$COMMANDS_FILE"
# echo "simplify diff(sin(x), x)" >> "$COMMANDS_FILE"
# echo "simplify diff(cos(x), x)" >> "$COMMANDS_FILE"
# echo "simplify diff(tan(x), x)" >> "$COMMANDS_FILE"
# echo "simplify diff(e^x, x)" >> "$COMMANDS_FILE"
# echo "simplify diff(ln(x), x)" >> "$COMMANDS_FILE"
# echo "simplify diff(x*sin(x), x)" >> "$COMMANDS_FILE"
# echo "simplify diff(x^2*sin(x), x)" >> "$COMMANDS_FILE"
# echo "simplify diff(sin(x)*cos(x), x)" >> "$COMMANDS_FILE"
# echo "simplify diff(x/sin(x), x)" >> "$COMMANDS_FILE"
# echo "simplify diff(sin(x^2), x)" >> "$COMMANDS_FILE"
# echo "simplify diff(e^(x^2), x)" >> "$COMMANDS_FILE"
# echo "simplify diff((x+1)^2, x)" >> "$COMMANDS_FILE"

# ========================================
# CÁLCULO - INTEGRALES
# ========================================
# echo "simplify integrate(1, x)" >> "$COMMANDS_FILE"
# echo "simplify integrate(x, x)" >> "$COMMANDS_FILE"
# echo "simplify integrate(x^2, x)" >> "$COMMANDS_FILE"
# echo "simplify integrate(x^3, x)" >> "$COMMANDS_FILE"
# echo "simplify integrate(x^n, x)" >> "$COMMANDS_FILE"
# echo "simplify integrate(sin(x), x)" >> "$COMMANDS_FILE"
# echo "simplify integrate(cos(x), x)" >> "$COMMANDS_FILE"
# echo "simplify integrate(e^x, x)" >> "$COMMANDS_FILE"
# echo "simplify integrate(1/x, x)" >> "$COMMANDS_FILE"
# echo "simplify integrate(5, x)" >> "$COMMANDS_FILE"
# echo "simplify integrate(a*x, x)" >> "$COMMANDS_FILE"

# ========================================
# ARITMÉTICA BÁSICA
# ========================================
echo "simplify x + 0" >> "$COMMANDS_FILE"
echo "simplify 0 + x" >> "$COMMANDS_FILE"
echo "simplify x - 0" >> "$COMMANDS_FILE"
echo "simplify x * 1" >> "$COMMANDS_FILE"
echo "simplify 1 * x" >> "$COMMANDS_FILE"
echo "simplify x * 0" >> "$COMMANDS_FILE"
echo "simplify 0 * x" >> "$COMMANDS_FILE"
echo "simplify x / 1" >> "$COMMANDS_FILE"
echo "simplify 0 / x" >> "$COMMANDS_FILE"
echo "simplify 2 + 3" >> "$COMMANDS_FILE"
echo "simplify 5 - 2" >> "$COMMANDS_FILE"
echo "simplify 2 * 3" >> "$COMMANDS_FILE"
echo "simplify 6 / 2" >> "$COMMANDS_FILE"
echo "simplify 2 * 3 + 4" >> "$COMMANDS_FILE"
echo "simplify 2 + 3 * 4" >> "$COMMANDS_FILE"
echo "simplify (2 + 3) * 4" >> "$COMMANDS_FILE"
echo "simplify 2 * (3 + 4)" >> "$COMMANDS_FILE"

# Negation
echo "simplify -x" >> "$COMMANDS_FILE"
echo "simplify -(-x)" >> "$COMMANDS_FILE"
echo "simplify -(x + y)" >> "$COMMANDS_FILE"
echo "simplify -(x - y)" >> "$COMMANDS_FILE"

# ========================================
# NÚMEROS - TEORÍA DE NÚMEROS
# ========================================
echo "simplify gcd(12, 18)" >> "$COMMANDS_FILE"
echo "simplify gcd(24, 36)" >> "$COMMANDS_FILE"
echo "simplify lcm(4, 6)" >> "$COMMANDS_FILE"
echo "simplify lcm(12, 18)" >> "$COMMANDS_FILE"
echo "simplify 5!" >> "$COMMANDS_FILE"
echo "simplify 6!" >> "$COMMANDS_FILE"
echo "simplify fact(5)" >> "$COMMANDS_FILE"
echo "simplify factors(12)" >> "$COMMANDS_FILE"
echo "simplify factors(24)" >> "$COMMANDS_FILE"
echo "simplify factors(100)" >> "$COMMANDS_FILE"
echo "simplify choose(5, 2)" >> "$COMMANDS_FILE"
echo "simplify choose(10, 3)" >> "$COMMANDS_FILE"
echo "simplify perm(5, 2)" >> "$COMMANDS_FILE"
echo "simplify perm(10, 3)" >> "$COMMANDS_FILE"

# ========================================
# CASOS COMPLEJOS / EDGE CASES
# ========================================

# Nested fractions
echo "simplify (1/x) / (1/y)" >> "$COMMANDS_FILE"
echo "simplify (x/y) / (a/b)" >> "$COMMANDS_FILE"
echo "simplify 1 / (1/x)" >> "$COMMANDS_FILE"

# Mixed operations
echo "simplify (x + y) * (x - y)" >> "$COMMANDS_FILE"
echo "simplify (a + b)^2 - (a - b)^2" >> "$COMMANDS_FILE"
echo "simplify x^2 + 2*x*y + y^2" >> "$COMMANDS_FILE"
echo "simplify x^2 - 2*x*y + y^2" >> "$COMMANDS_FILE"

# Logarithmic equations
echo "simplify x^(1/ln(x))" >> "$COMMANDS_FILE"
echo "simplify ln(x^2) / ln(x)" >> "$COMMANDS_FILE"

# Trigonometric complex
echo "simplify sin(2*x) + 2*sin(x)*cos(x)" >> "$COMMANDS_FILE"
echo "simplify sin(3*x) - (3*sin(x) - 4*sin(x)^3)" >> "$COMMANDS_FILE"

# Constants
echo "simplify e^0" >> "$COMMANDS_FILE"
echo "simplify ln(e^2)" >> "$COMMANDS_FILE"
echo "simplify log(10, 10)" >> "$COMMANDS_FILE"
echo "simplify pi * 0" >> "$COMMANDS_FILE"
echo "simplify pi * 1" >> "$COMMANDS_FILE"

# Absolute value
echo "simplify abs(-5)" >> "$COMMANDS_FILE"
echo "simplify abs(x^2)" >> "$COMMANDS_FILE"
echo "simplify abs(-x)" >> "$COMMANDS_FILE"

# ========================================
# SOLVER TESTS (usando solve en lugar de simplify)
# ========================================
echo "solve x + 2 = 5, x" >> "$COMMANDS_FILE"
echo "solve 2*x = 10, x" >> "$COMMANDS_FILE"
echo "solve 2*x + 4 = 10, x" >> "$COMMANDS_FILE"
echo "solve x^2 = 4, x" >> "$COMMANDS_FILE"
echo "solve x^2 - 4 = 0, x" >> "$COMMANDS_FILE"
echo "solve 2*x - 5 = x + 3, x" >> "$COMMANDS_FILE"

# Añadir comando de salida
echo "exit" >> "$COMMANDS_FILE"

echo "Ejecutando CLI con $(wc -l < $COMMANDS_FILE) comandos..."

# Ejecutar el CLI
$CLI_CMD < "$COMMANDS_FILE" >> "$OUTPUT_FILE" 2>&1

# Cleanup
rm -f "$COMMANDS_FILE"

echo "" >> "$OUTPUT_FILE"
echo "==================================================" >> "$OUTPUT_FILE"
echo "Tests completados: $(date)" >> "$OUTPUT_FILE"
echo "==================================================" >> "$OUTPUT_FILE"

echo ""
echo "✅ Tests completados!"
echo "📄 Output guardado en: $OUTPUT_FILE"
echo "📊 Líneas totales: $(wc -l < $OUTPUT_FILE)"
echo "📝 Expresiones testeadas: ~220+"
echo ""
echo "Para ver el output:"
echo "  cat $OUTPUT_FILE"
echo "  less $OUTPUT_FILE"
echo "  grep 'Result:' $OUTPUT_FILE | wc -l  # Contar resultados"
