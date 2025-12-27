#!/bin/bash
# Script para tests de PRECÁLCULO
# Enfoque: Valores absolutos, inecuaciones, conjuntos de soluciones
# Preparación para teoría de límites
# Uso: ./run_cli_precalculus_tests.sh

OUTPUT_FILE="output_precalculus_test.txt"
CLI_CMD="cargo run -p cas_cli --release"

echo "==================================================

" > "$OUTPUT_FILE"
echo "CLI Precalculus Tests - Valores Absolutos, Inecuaciones, Conjuntos" >> "$OUTPUT_FILE"
echo "Fecha: $(date)" >> "$OUTPUT_FILE"
echo "==================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

COMMANDS_FILE="/tmp/precalculus_commands.txt"
echo "steps normal" > "$COMMANDS_FILE"

echo "Preparando tests de precálculo..."

# ========================================
# VALORES ABSOLUTOS - Propiedades y Simplificación
# ========================================

# Básicos
echo "simplify abs(-5)" >> "$COMMANDS_FILE"
echo "simplify abs(5)" >> "$COMMANDS_FILE"
echo "simplify abs(0)" >> "$COMMANDS_FILE"
echo "simplify abs(-x)" >> "$COMMANDS_FILE"
echo "simplify abs(x^2)" >> "$COMMANDS_FILE"
echo "simplify abs(-3*x)" >> "$COMMANDS_FILE"

# Propiedades
echo "simplify abs(x*y)" >> "$COMMANDS_FILE"
echo "simplify abs(x/y)" >> "$COMMANDS_FILE"
echo "simplify abs(x^2*y)" >> "$COMMANDS_FILE"
echo "simplify abs(-x^2)" >> "$COMMANDS_FILE"

# Con expresiones
echo "simplify abs(x + 3)" >> "$COMMANDS_FILE"
echo "simplify abs(2*x - 4)" >> "$COMMANDS_FILE"
echo "simplify abs(x^2 - 1)" >> "$COMMANDS_FILE"

# Composición
echo "simplify abs(abs(x))" >> "$COMMANDS_FILE"
echo "simplify abs(-abs(x))" >> "$COMMANDS_FILE"

# ========================================
# ECUACIONES CON VALOR ABSOLUTO
# ========================================

# Básicas
echo "solve abs(x) = 5, x" >> "$COMMANDS_FILE"
echo "solve abs(x) = 0, x" >> "$COMMANDS_FILE"
echo "solve abs(x) = -1, x" >> "$COMMANDS_FILE"
echo "solve abs(x - 2) = 3, x" >> "$COMMANDS_FILE"
echo "solve abs(x + 1) = 4, x" >> "$COMMANDS_FILE"
echo "solve abs(2*x) = 6, x" >> "$COMMANDS_FILE"
echo "solve abs(3*x - 1) = 5, x" >> "$COMMANDS_FILE"

# Con dos absolutos
echo "solve abs(x) = abs(3), x" >> "$COMMANDS_FILE"
echo "solve abs(x - 1) = abs(x + 1), x" >> "$COMMANDS_FILE"

# ========================================
# INECUACIONES CON VALOR ABSOLUTO
# ========================================

# Inecuaciones básicas abs(x) < a
echo "solve abs(x) < 5, x" >> "$COMMANDS_FILE"
echo "solve abs(x) <= 3, x" >> "$COMMANDS_FILE"
echo "solve abs(x) > 2, x" >> "$COMMANDS_FILE"
echo "solve abs(x) >= 4, x" >> "$COMMANDS_FILE"

# Inecuaciones con transformación
echo "solve abs(x - 3) < 2, x" >> "$COMMANDS_FILE"
echo "solve abs(x + 1) <= 5, x" >> "$COMMANDS_FILE"
echo "solve abs(x - 2) > 4, x" >> "$COMMANDS_FILE"
echo "solve abs(2*x) < 8, x" >> "$COMMANDS_FILE"
echo "solve abs(3*x + 1) <= 7, x" >> "$COMMANDS_FILE"

# Casos especiales
echo "solve abs(x) < -1, x" >> "$COMMANDS_FILE"
echo "solve abs(x) > 0, x" >> "$COMMANDS_FILE"

# ========================================
# ECUACIONES LINEALES (repaso para límites)
# ========================================

# Ecuaciones simples
echo "solve x + 5 = 0, x" >> "$COMMANDS_FILE"
echo "solve 3*x - 7 = 0, x" >> "$COMMANDS_FILE"
echo "solve 2*x + 1 = 5, x" >> "$COMMANDS_FILE"

# Con variables en ambos lados
echo "solve 3*x + 2 = x - 4, x" >> "$COMMANDS_FILE"
echo "solve 5*x - 3 = 2*x + 6, x" >> "$COMMANDS_FILE"
echo "solve x/2 = 3*x - 4, x" >> "$COMMANDS_FILE"

# Con fracciones
echo "solve x/2 + x/3 = 5, x" >> "$COMMANDS_FILE"
echo "solve 1/x = 2, x" >> "$COMMANDS_FILE"

# ========================================
# ECUACIONES CUADRÁTICAS (importantes para límites)
# ========================================

# Forma estándar
echo "solve x^2 = 9, x" >> "$COMMANDS_FILE"
echo "solve x^2 = 0, x" >> "$COMMANDS_FILE"
echo "solve x^2 + x = 0, x" >> "$COMMANDS_FILE"
echo "solve x^2 - x = 0, x" >> "$COMMANDS_FILE"
echo "solve x^2 - 5*x + 6 = 0, x" >> "$COMMANDS_FILE"
echo "solve x^2 + 4*x + 4 = 0, x" >> "$COMMANDS_FILE"

# Forma factorizada
echo "solve (x - 2)*(x + 3) = 0, x" >> "$COMMANDS_FILE"
echo "solve x*(x - 4) = 0, x" >> "$COMMANDS_FILE"

# ========================================
# INECUACIONES LINEALES
# ========================================

# Básicas
echo "solve x > 5, x" >> "$COMMANDS_FILE"
echo "solve x < -2, x" >> "$COMMANDS_FILE"
echo "solve x >= 0, x" >> "$COMMANDS_FILE"
echo "solve x <= 10, x" >> "$COMMANDS_FILE"

# Con coeficientes
echo "solve 2*x > 6, x" >> "$COMMANDS_FILE"
echo "solve -3*x < 9, x" >> "$COMMANDS_FILE"
echo "solve 5*x >= 15, x" >> "$COMMANDS_FILE"
echo "solve -2*x <= 8, x" >> "$COMMANDS_FILE"

# Con términos independientes
echo "solve x + 3 > 7, x" >> "$COMMANDS_FILE"
echo "solve 2*x - 1 < 5, x" >> "$COMMANDS_FILE"
echo "solve 3*x + 4 >= 10, x" >> "$COMMANDS_FILE"
echo "solve -x + 2 <= 6, x" >> "$COMMANDS_FILE"

# Con variables en ambos lados
echo "solve 2*x > x + 3, x" >> "$COMMANDS_FILE"
echo "solve 3*x + 1 < 2*x + 5, x" >> "$COMMANDS_FILE"
echo "solve 4*x - 2 >= x + 7, x" >> "$COMMANDS_FILE"

# Cambio de signo (importante)
echo "solve -2*x > 10, x" >> "$COMMANDS_FILE"
echo "solve -x < 5, x" >> "$COMMANDS_FILE"

# ========================================
# INECUACIONES CUADRÁTICAS (Base para límites)
# ========================================

# Parabola básica
echo "solve x^2 < 4, x" >> "$COMMANDS_FILE"
echo "solve x^2 > 9, x" >> "$COMMANDS_FILE"
echo "solve x^2 <= 1, x" >> "$COMMANDS_FILE"
echo "solve x^2 >= 16, x" >> "$COMMANDS_FILE"

# Forma factorizada
echo "solve (x - 1)*(x - 3) > 0, x" >> "$COMMANDS_FILE"
echo "solve (x + 2)*(x - 2) < 0, x" >> "$COMMANDS_FILE"
echo "solve x*(x - 4) >= 0, x" >> "$COMMANDS_FILE"

# Forma estándar
echo "solve x^2 - 5*x + 6 > 0, x" >> "$COMMANDS_FILE"
echo "solve x^2 + 2*x - 3 < 0, x" >> "$COMMANDS_FILE"
echo "solve x^2 - 4*x + 4 <= 0, x" >> "$COMMANDS_FILE"

# Parabola invertida
echo "solve -x^2 + 4 > 0, x" >> "$COMMANDS_FILE"
echo "solve -x^2 + 1 >= 0, x" >> "$COMMANDS_FILE"

# ========================================
# INECUACIONES RACIONALES (Preparación para límites)
# ========================================

# Básicas
echo "solve 1/x > 0, x" >> "$COMMANDS_FILE"
echo "solve 1/x < 0, x" >> "$COMMANDS_FILE"
echo "solve 1/x >= 1, x" >> "$COMMANDS_FILE"

# Con numeradores
echo "solve x/(x - 1) > 0, x" >> "$COMMANDS_FILE"
echo "solve (x + 2)/(x - 3) < 0, x" >> "$COMMANDS_FILE"
echo "solve (x - 1)/(x + 1) >= 0, x" >> "$COMMANDS_FILE"

# ========================================
# SISTEMAS DE INECUACIONES (para intervalos)
# ========================================

# Nota: Para múltiples inecuaciones, estas se resolverían por separado
# y luego se tomaría la intersección

# ========================================
# EXPRESIONES RACIONALES (importantes para límites)
# ========================================

# Simplificación
echo "simplify (x^2 - 1)/(x - 1)" >> "$COMMANDS_FILE"
echo "simplify (x^2 - 4)/(x + 2)" >> "$COMMANDS_FILE"
echo "simplify (x^2 + 3*x + 2)/(x + 1)" >> "$COMMANDS_FILE"
echo "simplify (x^3 - 1)/(x - 1)" >> "$COMMANDS_FILE"

# Con factorización
echo "simplify (x^2 + 5*x + 6)/(x + 2)" >> "$COMMANDS_FILE"
echo "simplify (2*x^2 - 8)/(x - 2)" >> "$COMMANDS_FILE"

# Operaciones
echo "simplify 1/x + 1/(x+1)" >> "$COMMANDS_FILE"
echo "simplify 1/(x-1) - 1/(x+1)" >> "$COMMANDS_FILE"
echo "simplify x/(x+1) + 1/(x+1)" >> "$COMMANDS_FILE"

# ========================================
# COMPOSICIÓN DE FUNCIONES (preparación)
# ========================================

# Básicas
echo "simplify abs(x - 1)" >> "$COMMANDS_FILE"
echo "simplify abs(x^2 - 4)" >> "$COMMANDS_FILE"
echo "simplify abs((x-2)*(x+2))" >> "$COMMANDS_FILE"

# Con raíces
echo "simplify sqrt(x^2)" >> "$COMMANDS_FILE"
echo "simplify sqrt((x-1)^2)" >> "$COMMANDS_FILE"

# ========================================
# DOMINIO DE FUNCIONES (conceptos para límites)
# ========================================

# Raíz cuadrada: x >= 0
echo "simplify sqrt(x)" >> "$COMMANDS_FILE"
echo "simplify sqrt(x - 2)" >> "$COMMANDS_FILE"
echo "simplify sqrt(4 - x^2)" >> "$COMMANDS_FILE"

# División: denominador != 0
echo "simplify 1/x" >> "$COMMANDS_FILE"
echo "simplify 1/(x - 3)" >> "$COMMANDS_FILE"
echo "simplify 1/(x^2 - 4)" >> "$COMMANDS_FILE"

# Logaritmos: argumento > 0
echo "simplify ln(x)" >> "$COMMANDS_FILE"
echo "simplify ln(x - 1)" >> "$COMMANDS_FILE"
echo "simplify ln(x^2 + 1)" >> "$COMMANDS_FILE"

# ========================================
# CASOS ESPECIALES PARA LÍMITES
# ========================================

# Indeterminaciones 0/0
echo "simplify (x^2 - 1)/(x - 1)" >> "$COMMANDS_FILE"
echo "simplify (x^2 + x)/(x)" >> "$COMMANDS_FILE"
echo "simplify (x^3 - 8)/(x - 2)" >> "$COMMANDS_FILE"

# Con raíces
echo "simplify (sqrt(x + 1) - 1)/x" >> "$COMMANDS_FILE"

# Racionalización
echo "simplify (x - 1)/(sqrt(x) - 1)" >> "$COMMANDS_FILE"

# Factorización útil
echo "simplify (x^2 - a^2)/(x - a)" >> "$COMMANDS_FILE"
echo "simplify (x^3 - a^3)/(x - a)" >> "$COMMANDS_FILE"

# Añadir comando de salida
echo "exit" >> "$COMMANDS_FILE"

echo "Ejecutando CLI con $(wc -l <$COMMANDS_FILE) comandos de precálculo..."

# Ejecutar el CLI
$CLI_CMD < "$COMMANDS_FILE" >> "$OUTPUT_FILE" 2>&1

# Cleanup
rm -f "$COMMANDS_FILE"

echo "" >> "$OUTPUT_FILE"
echo "==================================================" >> "$OUTPUT_FILE"
echo "Tests de precálculo completados: $(date)" >> "$OUTPUT_FILE"
echo "==================================================" >> "$OUTPUT_FILE"

echo ""
echo "✅ Tests de precálculo completados!"
echo "📄 Output guardado en: $OUTPUT_FILE"
echo "📊 Líneas totales: $(wc -l <$OUTPUT_FILE)"
echo "📝 Tests realizados:"
echo "   - Valores absolutos: ~25"
echo "   - Ecuaciones lineales: ~15"
echo "   - Ecuaciones cuadráticas: ~15"
echo "   - Inecuaciones lineales: ~20"
echo "   - Inecuaciones cuadráticas: ~15"
echo "   - Inecuaciones racionales: ~10"
echo "   - Expresiones racionales: ~15"
echo "   - Casos límite: ~10"
echo "   Total: ~125+ expresiones"
echo ""
echo "Para ver el output:"
echo "  cat $OUTPUT_FILE"
echo "  less $OUTPUT_FILE"
echo "  grep 'Result:' $OUTPUT_FILE | wc -l  # Contar resultados"
echo "  grep 'solve' $OUTPUT_FILE | head -20  # Ver primeras 20 soluciones"
