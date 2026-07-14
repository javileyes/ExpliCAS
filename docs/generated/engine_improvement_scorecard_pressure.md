# Engine Improvement Scorecard

- Generated: 2026-07-14T15:11:45.179869+00:00
- Git branch: main
- Git commit: `859d0b8ccc8c1dcb151276e837aacfb9e5c9d70d`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=359

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=995.16ms avg_case_ms=9.95 simplify=278.53ms avg_simplify_ms=2.79, sum total=200 failed=0 elapsed=893.07ms avg_case_ms=4.47 simplify=288.54ms avg_simplify_ms=1.44, product total=100 failed=0 elapsed=624.81ms avg_case_ms=6.25 simplify=180.97ms avg_simplify_ms=1.81, difference total=50 failed=0 elapsed=410.23ms avg_case_ms=8.20 simplify=125.90ms avg_simplify_ms=2.52
- Engine hotspots: sum simplify=288.54ms avg_simplify_ms=1.44 wall=893.07ms, shifted_quotient simplify=278.53ms avg_simplify_ms=2.79 wall=995.16ms, product simplify=180.97ms avg_simplify_ms=1.81 wall=624.81ms, difference simplify=125.90ms avg_simplify_ms=2.52 wall=410.23ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=995.16ms avg_case_ms=9.95 avg_simplify_ms=2.79, sum@0+100 failed=0 elapsed=658.67ms avg_case_ms=6.59 avg_simplify_ms=2.03, product@0+100 failed=0 elapsed=624.81ms avg_case_ms=6.25 avg_simplify_ms=1.81, difference@0+50 failed=0 elapsed=410.23ms avg_case_ms=8.20 avg_simplify_ms=2.52, sum@700+100 failed=0 elapsed=234.40ms avg_case_ms=2.34 avg_simplify_ms=0.85
- Steady-state engine reruns: product@0+100 #175 product runs=3 median_simplify=15.82ms median_wire=15.86ms median_wall=61.06ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.56ms median_wire=17.64ms median_wall=67.14ms, sum@0+100 #173 sum runs=3 median_simplify=15.93ms median_wire=15.98ms median_wall=61.60ms, difference@0+50 #174 difference runs=3 median_simplify=15.53ms median_wire=15.58ms median_wall=69.15ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.08ms median_wire=13.16ms median_wall=49.60ms
- Steady-state dominant expressions: product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.92s | passed=450 failed=0 total=450 avg_case=6.489ms |
| `calculus_diff_exhaustive_contract` | `pass` | 13.30s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.98s | passed=1 failed=0 |
