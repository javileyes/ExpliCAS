# Engine Improvement Scorecard

- Generated: 2026-07-03T23:20:03.300041+00:00
- Git branch: main
- Git commit: `5e998980d650327c9ab0c9491a450c14efb63a55`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=355

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=922.95ms avg_case_ms=9.23 simplify=254.77ms avg_simplify_ms=2.55, sum total=200 failed=0 elapsed=827.98ms avg_case_ms=4.14 simplify=267.52ms avg_simplify_ms=1.34, product total=100 failed=0 elapsed=578.68ms avg_case_ms=5.79 simplify=164.41ms avg_simplify_ms=1.64, difference total=50 failed=0 elapsed=384.67ms avg_case_ms=7.69 simplify=116.63ms avg_simplify_ms=2.33
- Engine hotspots: sum simplify=267.52ms avg_simplify_ms=1.34 wall=827.98ms, shifted_quotient simplify=254.77ms avg_simplify_ms=2.55 wall=922.95ms, product simplify=164.41ms avg_simplify_ms=1.64 wall=578.68ms, difference simplify=116.63ms avg_simplify_ms=2.33 wall=384.67ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=922.95ms avg_case_ms=9.23 avg_simplify_ms=2.55, sum@0+100 failed=0 elapsed=603.80ms avg_case_ms=6.04 avg_simplify_ms=1.88, product@0+100 failed=0 elapsed=578.68ms avg_case_ms=5.79 avg_simplify_ms=1.64, difference@0+50 failed=0 elapsed=384.67ms avg_case_ms=7.69 avg_simplify_ms=2.33, sum@700+100 failed=0 elapsed=224.17ms avg_case_ms=2.24 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.60ms median_wire=16.67ms median_wall=64.99ms, sum@0+100 #173 sum runs=3 median_simplify=14.72ms median_wire=14.76ms median_wall=56.63ms, product@0+100 #175 product runs=3 median_simplify=14.85ms median_wire=14.89ms median_wall=57.99ms, difference@0+50 #174 difference runs=3 median_simplify=15.07ms median_wire=15.12ms median_wall=57.72ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.16ms median_wire=12.23ms median_wall=46.91ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.71s | passed=450 failed=0 total=450 avg_case=6.022ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.97s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
