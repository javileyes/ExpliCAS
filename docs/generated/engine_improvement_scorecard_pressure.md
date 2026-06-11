# Engine Improvement Scorecard

- Generated: 2026-06-11T23:10:44.217178+00:00
- Git branch: main
- Git commit: `15c8add95f1b31b6a44948f1b5b5c749a1381365`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=772.61ms avg_case_ms=7.73 simplify=218.79ms avg_simplify_ms=2.19, sum total=200 failed=0 elapsed=685.44ms avg_case_ms=3.43 simplify=229.28ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=473.73ms avg_case_ms=4.74 simplify=135.67ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=325.83ms avg_case_ms=6.52 simplify=103.77ms avg_simplify_ms=2.08
- Engine hotspots: sum simplify=229.28ms avg_simplify_ms=1.15 wall=685.44ms, shifted_quotient simplify=218.79ms avg_simplify_ms=2.19 wall=772.61ms, product simplify=135.67ms avg_simplify_ms=1.36 wall=473.73ms, difference simplify=103.77ms avg_simplify_ms=2.08 wall=325.83ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=772.61ms avg_case_ms=7.73 avg_simplify_ms=2.19, sum@0+100 failed=0 elapsed=496.88ms avg_case_ms=4.97 avg_simplify_ms=1.60, product@0+100 failed=0 elapsed=473.73ms avg_case_ms=4.74 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=325.83ms avg_case_ms=6.52 avg_simplify_ms=2.08, sum@700+100 failed=0 elapsed=188.56ms avg_case_ms=1.89 avg_simplify_ms=0.69
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.91ms median_wire=12.97ms median_wall=48.98ms, sum@0+100 #173 sum runs=3 median_simplify=11.39ms median_wire=11.44ms median_wall=43.34ms, product@0+100 #175 product runs=3 median_simplify=11.52ms median_wire=11.56ms median_wall=43.55ms, difference@0+50 #174 difference runs=3 median_simplify=11.44ms median_wire=11.49ms median_wall=43.35ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.47ms median_wire=10.53ms median_wall=39.90ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.26s | passed=450 failed=0 total=450 avg_case=5.022ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.82s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.08s | passed=1 failed=0 |
