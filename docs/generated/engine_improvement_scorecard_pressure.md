# Engine Improvement Scorecard

- Generated: 2026-06-15T10:06:08.725177+00:00
- Git branch: main
- Git commit: `f2fa269b39b2557a8f3bf3b27dd26ce21802a0b7`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=352

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=789.30ms avg_case_ms=7.89 simplify=225.16ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=757.55ms avg_case_ms=3.79 simplify=255.32ms avg_simplify_ms=1.28, product total=100 failed=0 elapsed=492.24ms avg_case_ms=4.92 simplify=141.93ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=332.30ms avg_case_ms=6.65 simplify=105.89ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=255.32ms avg_simplify_ms=1.28 wall=757.55ms, shifted_quotient simplify=225.16ms avg_simplify_ms=2.25 wall=789.30ms, product simplify=141.93ms avg_simplify_ms=1.42 wall=492.24ms, difference simplify=105.89ms avg_simplify_ms=2.12 wall=332.30ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=789.30ms avg_case_ms=7.89 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=558.78ms avg_case_ms=5.59 avg_simplify_ms=1.81, product@0+100 failed=0 elapsed=492.24ms avg_case_ms=4.92 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=332.30ms avg_case_ms=6.65 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=198.77ms avg_case_ms=1.99 avg_simplify_ms=0.74
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=11.83ms median_wire=11.88ms median_wall=45.16ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.03ms median_wire=13.12ms median_wall=49.62ms, product@0+100 #175 product runs=3 median_simplify=11.57ms median_wire=11.62ms median_wall=44.16ms, difference@0+50 #174 difference runs=3 median_simplify=11.92ms median_wire=11.97ms median_wall=45.39ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.87ms median_wire=10.95ms median_wall=41.40ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.37s | passed=450 failed=0 total=450 avg_case=5.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.90s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.96s | passed=1 failed=0 |
