# Engine Improvement Scorecard

- Generated: 2026-06-14T17:03:12.882403+00:00
- Git branch: main
- Git commit: `e643cf814f8964051e6291540912994d94bb893c`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=351

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=783.32ms avg_case_ms=7.83 simplify=223.43ms avg_simplify_ms=2.23, sum total=200 failed=0 elapsed=705.89ms avg_case_ms=3.53 simplify=236.46ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=469.23ms avg_case_ms=4.69 simplify=134.39ms avg_simplify_ms=1.34, difference total=50 failed=0 elapsed=323.03ms avg_case_ms=6.46 simplify=102.39ms avg_simplify_ms=2.05
- Engine hotspots: sum simplify=236.46ms avg_simplify_ms=1.18 wall=705.89ms, shifted_quotient simplify=223.43ms avg_simplify_ms=2.23 wall=783.32ms, product simplify=134.39ms avg_simplify_ms=1.34 wall=469.23ms, difference simplify=102.39ms avg_simplify_ms=2.05 wall=323.03ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=783.32ms avg_case_ms=7.83 avg_simplify_ms=2.23, sum@0+100 failed=0 elapsed=513.93ms avg_case_ms=5.14 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=469.23ms avg_case_ms=4.69 avg_simplify_ms=1.34, difference@0+50 failed=0 elapsed=323.03ms avg_case_ms=6.46 avg_simplify_ms=2.05, sum@700+100 failed=0 elapsed=191.96ms avg_case_ms=1.92 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.82ms median_wire=12.89ms median_wall=48.74ms, sum@0+100 #173 sum runs=3 median_simplify=11.60ms median_wire=11.65ms median_wall=44.05ms, difference@0+50 #174 difference runs=3 median_simplify=11.51ms median_wire=11.56ms median_wall=43.52ms, product@0+100 #175 product runs=3 median_simplify=11.41ms median_wire=11.46ms median_wall=43.72ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.61ms median_wire=10.69ms median_wall=40.09ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.28s | passed=450 failed=0 total=450 avg_case=5.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.80s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
