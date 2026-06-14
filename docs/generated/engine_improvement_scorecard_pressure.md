# Engine Improvement Scorecard

- Generated: 2026-06-14T10:33:13.451040+00:00
- Git branch: main
- Git commit: `32b784ec9bafd99532576ceeb7ea956b652bc529`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=10
- By area: calculus / integration:4, calculus / runtime:3, calculus / differentiation:2, calculus / robustness:1
- Recent 1: `calculus / integration` - 2026-06-08 - Discovery observe-only: polynomial cosecant/cotangent source-return still emits depth pressure
- Recent 2: `calculus / differentiation` - 2026-06-06 - Observe-only discovery: exact-square atanh scaled-root runtime is not caused by the global empty-domain check
- Recent 3: `calculus / differentiation` - 2026-06-06 - Observe-only discovery: exact-square inverse-root diff runtime is not fixed by raw target preservation

## Calculus Support Matrix Signal

- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.
- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.
- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.
- `diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=263
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=780.89ms avg_case_ms=7.81 simplify=222.30ms avg_simplify_ms=2.22, sum total=200 failed=0 elapsed=683.36ms avg_case_ms=3.42 simplify=229.08ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=475.88ms avg_case_ms=4.76 simplify=136.89ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=322.87ms avg_case_ms=6.46 simplify=102.05ms avg_simplify_ms=2.04
- Engine hotspots: sum simplify=229.08ms avg_simplify_ms=1.15 wall=683.36ms, shifted_quotient simplify=222.30ms avg_simplify_ms=2.22 wall=780.89ms, product simplify=136.89ms avg_simplify_ms=1.37 wall=475.88ms, difference simplify=102.05ms avg_simplify_ms=2.04 wall=322.87ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=780.89ms avg_case_ms=7.81 avg_simplify_ms=2.22, sum@0+100 failed=0 elapsed=494.35ms avg_case_ms=4.94 avg_simplify_ms=1.59, product@0+100 failed=0 elapsed=475.88ms avg_case_ms=4.76 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=322.87ms avg_case_ms=6.46 avg_simplify_ms=2.04, sum@700+100 failed=0 elapsed=189.01ms avg_case_ms=1.89 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.71ms median_wire=12.78ms median_wall=48.47ms, product@0+100 #175 product runs=3 median_simplify=11.56ms median_wire=11.61ms median_wall=43.98ms, sum@0+100 #173 sum runs=3 median_simplify=11.38ms median_wire=11.43ms median_wall=43.53ms, difference@0+50 #174 difference runs=3 median_simplify=11.54ms median_wire=11.59ms median_wall=43.88ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.45ms median_wire=10.53ms median_wall=39.89ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.26s | passed=450 failed=0 total=450 avg_case=5.022ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.85s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
