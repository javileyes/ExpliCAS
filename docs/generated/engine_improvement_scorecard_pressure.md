# Engine Improvement Scorecard

- Generated: 2026-07-14T10:56:46.948554+00:00
- Git branch: main
- Git commit: `f90cc2b2834753aa665cd782f55943aed4761f6a`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=992.83ms avg_case_ms=9.93 simplify=277.65ms avg_simplify_ms=2.78, sum total=200 failed=0 elapsed=873.48ms avg_case_ms=4.37 simplify=281.04ms avg_simplify_ms=1.41, product total=100 failed=0 elapsed=606.48ms avg_case_ms=6.06 simplify=174.07ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=404.64ms avg_case_ms=8.09 simplify=123.26ms avg_simplify_ms=2.47
- Engine hotspots: sum simplify=281.04ms avg_simplify_ms=1.41 wall=873.48ms, shifted_quotient simplify=277.65ms avg_simplify_ms=2.78 wall=992.83ms, product simplify=174.07ms avg_simplify_ms=1.74 wall=606.48ms, difference simplify=123.26ms avg_simplify_ms=2.47 wall=404.64ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=992.83ms avg_case_ms=9.93 avg_simplify_ms=2.78, sum@0+100 failed=0 elapsed=642.34ms avg_case_ms=6.42 avg_simplify_ms=1.98, product@0+100 failed=0 elapsed=606.48ms avg_case_ms=6.06 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=404.64ms avg_case_ms=8.09 avg_simplify_ms=2.47, sum@700+100 failed=0 elapsed=231.14ms avg_case_ms=2.31 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.06ms median_wire=17.13ms median_wall=65.35ms, sum@0+100 #173 sum runs=3 median_simplify=15.43ms median_wire=15.48ms median_wall=58.45ms, difference@0+50 #174 difference runs=3 median_simplify=15.19ms median_wire=15.25ms median_wall=58.56ms, product@0+100 #175 product runs=3 median_simplify=16.59ms median_wire=16.65ms median_wall=60.12ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.75ms median_wire=12.82ms median_wall=48.53ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.88s | passed=450 failed=0 total=450 avg_case=6.400ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.72s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.94s | passed=1 failed=0 |
