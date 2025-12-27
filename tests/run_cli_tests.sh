#!/usr/bin/env bash
set -euo pipefail

OUTPUT_FILE="${1:-output_test.txt}"

# Mejor que cargo run: compila una vez y ejecuta binario
CLI_BIN="./target/release/cas_cli"

# Toggles (cámbialos a gusto)
ENABLE_EXPLAIN="${ENABLE_EXPLAIN:-1}"   # 1=on, 0=off
STEPS_MODE="${STEPS_MODE:-normal}"      # normal | off | verbose (según vuestro REPL)
RUN_RELEASE_BUILD="${RUN_RELEASE_BUILD:-1}"

COMMANDS_FILE="$(mktemp /tmp/cas_cli_commands.XXXXXX)"

build_cli() {
  if [[ "$RUN_RELEASE_BUILD" == "1" ]]; then
    cargo build -p cas_cli --release
  fi
}

# Añade un "caso" aislado con profile: reset -> cmd -> profile -> health
add_case() {
  local label="$1"
  local cmd="$2"

  # Clear profiler, run command, then show profile stats and health table
  echo "profile clear" >> "$COMMANDS_FILE"
  echo "$cmd"          >> "$COMMANDS_FILE"
  echo "profile"       >> "$COMMANDS_FILE"
  echo "health"        >> "$COMMANDS_FILE"
  echo ""              >> "$COMMANDS_FILE"
}

# Cabecera del output
{
  echo "=================================================="
  echo "CLI Tests Output"
  echo "Fecha: $(date)"
  echo "Explain: $ENABLE_EXPLAIN | Steps: $STEPS_MODE"
  echo "=================================================="
  echo
} > "$OUTPUT_FILE"

# Preparar REPL session
{
  echo "steps $STEPS_MODE"
  echo "profile enable"
  echo "health on"
  if [[ "$ENABLE_EXPLAIN" == "1" ]]; then
    echo "set explain on"
  else
    echo "set explain off"
  fi
  echo ""
} > "$COMMANDS_FILE"

echo "Preparando suite exhaustiva del CLI..."

# ============================================================
# 1) RAÍCES / RADICALES (simplificación + denesting)
# ============================================================
add_case "roots" "simplify sqrt(12)"
add_case "roots" "simplify sqrt(72)"
add_case "roots" "simplify sqrt(8/9)"
add_case "roots" "simplify sqrt(12/25)"
add_case "roots" "simplify sqrt(12) + sqrt(27)"
add_case "roots" "simplify sqrt(8) + sqrt(2)"
add_case "roots" "simplify sqrt(32) / sqrt(2)"
add_case "roots" "simplify (sqrt(12) + sqrt(27)) / sqrt(3)"
add_case "roots" "simplify sqrt(8 * x^2)"
add_case "roots" "simplify sqrt(12 * x^3)"

# Denesting (si lo soporta)
add_case "denest" "simplify sqrt(3 + 2*sqrt(2))"
add_case "denest" "simplify sqrt(5 + 2*sqrt(6))"
add_case "denest" "simplify sqrt(7 - 2*sqrt(10))"

# ============================================================
# 2) RACIONALIZACIÓN (auto + comando)
# ============================================================
add_case "rationalize_auto" "simplify 1/sqrt(2)"
add_case "rationalize_auto" "simplify 1/(1 + sqrt(2))"
add_case "rationalize_auto" "simplify x/(1 + sqrt(2))"
add_case "rationalize_auto" "simplify x/(2*(1 + sqrt(2)))"
add_case "rationalize_auto" "simplify 1/(3 - 2*sqrt(5))"
add_case "rationalize_auto" "simplify x/(2*(3 - 2*sqrt(5)))"

# Multi-surd (manual) si existe el comando rationalize
add_case "rationalize_cmd" "rationalize 1/(1 + sqrt(2) + sqrt(3))"
add_case "rationalize_cmd" "rationalize 1/( (1+sqrt(2)) + sqrt(3) )"

# ============================================================
# 3) TELESCOPIA / FACTORIZACIÓN E IDENTIDADES
# ============================================================
add_case "telescoping" "simplify (x - 1)*(x + 1)"
add_case "telescoping" "simplify (x - 1)*(x + 1)*(x^2 + 1)"
add_case "telescoping" "simplify (x - 1)*(x + 1)*(x^2 + 1)*(x^4 + 1) - (x^8 - 1)"
add_case "telescoping" "simplify (a + b)^2 - (a - b)^2"
add_case "telescoping" "simplify (x + y)*(x - y)"

# ============================================================
# 4) BINOMIALES (expand vs simplify) + PRODUCTOS EN CADENA
# ============================================================
add_case "binomial" "simplify (1 + x)^3"
add_case "binomial" "simplify (1 + x)^5"
add_case "binomial" "simplify (x - 1)^2"
add_case "binomial" "simplify (x + 1)*(x + 2)"
add_case "binomial" "expand (x + 1)^6"
add_case "binomial" "expand (x - 1)*(x + 1)"
add_case "products" "simplify (x+1)*(x+2)*(x+3)"
add_case "products" "expand (x+1)*(x+2)*(x+3)"

# ============================================================
# 5) SUMAS / SUMATORIOS “EN CADENA” (stress de Add/Collect)
# ============================================================
add_case "sums" "simplify x + x + x + x"
add_case "sums" "simplify 2*x + 3*x + 4*x"
add_case "sums" "simplify x*y + x*z + x*w"
add_case "sums" "simplify (a+b+c+d) + (d+c+b+a)"
add_case "sums" "simplify (x+1) + (x+2) + (x+3) + (x+4)"

# ============================================================
# 6) FRACCIONES (sumas, simplificación, anidadas)
# ============================================================
add_case "fractions" "simplify 1/2 + 1/3"
add_case "fractions" "simplify x/2 + x/3"
add_case "fractions" "simplify 1/x + 1/y"
add_case "fractions" "simplify (x+1)/x + 1/x"
add_case "fractions" "simplify (1/x) / (1/y)"
add_case "fractions" "simplify 1 / (1/x)"

# ============================================================
# 7) EXPONENTES Y POTENCIAS
# ============================================================
add_case "powers" "simplify 2^3 * 2^4"
add_case "powers" "simplify (2^3)^4"
add_case "powers" "simplify (x^2)^3"
add_case "powers" "simplify x^2 * x^3"
add_case "powers" "simplify x^0"
add_case "powers" "simplify x^1"
add_case "powers" "simplify 2^(-3)"
add_case "powers" "simplify x^(-2)"

# ============================================================
# 8) LOGARITMOS
# ============================================================
add_case "logs" "simplify ln(x*y) - ln(x)"
add_case "logs" "simplify ln(x^2) / ln(x)"
add_case "logs" "simplify ln(e)"
add_case "logs" "simplify ln(1)"
add_case "logs" "simplify e^(ln(x))"
add_case "logs" "simplify ln(e^x)"

# ============================================================
# 9) TRIGONOMETRÍA (identidades + stress)
# ============================================================
add_case "trig" "simplify sin(x)^2 + cos(x)^2"
add_case "trig" "simplify sin(2*x) + 2*sin(x)*cos(x)"
add_case "trig" "simplify sin(3*x) - (3*sin(x) - 4*sin(x)^3)"

# ============================================================
# 10) SOLVER
# ============================================================
add_case "solver" "solve x + 2 = 5, x"
add_case "solver" "solve x^2 = 4, x"
add_case "solver" "solve x^2 - 4 = 0, x"

# Salida
echo "exit" >> "$COMMANDS_FILE"

# Build + run
build_cli

echo "Ejecutando CLI con $(grep -E '^(simplify|expand|rationalize|solve) ' "$COMMANDS_FILE" | wc -l | tr -d ' ') expresiones..."
"$CLI_BIN" < "$COMMANDS_FILE" >> "$OUTPUT_FILE" 2>&1

rm -f "$COMMANDS_FILE"

{
  echo
  echo "=================================================="
  echo "Tests completados: $(date)"
  echo "=================================================="
} >> "$OUTPUT_FILE"

echo ""
echo "✅ Tests completados!"
echo "📄 Output guardado en: $OUTPUT_FILE"
echo "📊 Líneas totales: $(wc -l < "$OUTPUT_FILE")"
echo ""
echo "Tips:"
echo "  grep -n \"Rule Health Report\" -n $OUTPUT_FILE | head"
echo "  grep -n \"──── Pipeline Diagnostics\" -n $OUTPUT_FILE | head"
echo "  grep -n \"Result:\" $OUTPUT_FILE | wc -l"
