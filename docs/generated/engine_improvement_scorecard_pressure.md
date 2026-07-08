# Engine Improvement Scorecard

- Generated: 2026-07-08T11:47:57.109174+00:00
- Git branch: main
- Git commit: `e498e577891f95f90a1afbe8119ca7e4522082c0`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=932.37ms avg_case_ms=9.32 simplify=257.75ms avg_simplify_ms=2.58, sum total=200 failed=0 elapsed=817.93ms avg_case_ms=4.09 simplify=263.32ms avg_simplify_ms=1.32, product total=100 failed=0 elapsed=572.53ms avg_case_ms=5.73 simplify=162.57ms avg_simplify_ms=1.63, difference total=50 failed=0 elapsed=378.87ms avg_case_ms=7.58 simplify=114.78ms avg_simplify_ms=2.30
- Engine hotspots: sum simplify=263.32ms avg_simplify_ms=1.32 wall=817.93ms, shifted_quotient simplify=257.75ms avg_simplify_ms=2.58 wall=932.37ms, product simplify=162.57ms avg_simplify_ms=1.63 wall=572.53ms, difference simplify=114.78ms avg_simplify_ms=2.30 wall=378.87ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=932.37ms avg_case_ms=9.32 avg_simplify_ms=2.58, sum@0+100 failed=0 elapsed=595.65ms avg_case_ms=5.96 avg_simplify_ms=1.84, product@0+100 failed=0 elapsed=572.53ms avg_case_ms=5.73 avg_simplify_ms=1.63, difference@0+50 failed=0 elapsed=378.87ms avg_case_ms=7.58 avg_simplify_ms=2.30, sum@700+100 failed=0 elapsed=222.28ms avg_case_ms=2.22 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.39ms median_wire=16.46ms median_wall=63.36ms, sum@0+100 #173 sum runs=3 median_simplify=14.89ms median_wire=14.94ms median_wall=56.57ms, product@0+100 #175 product runs=3 median_simplify=14.67ms median_wire=14.72ms median_wall=55.72ms, difference@0+50 #174 difference runs=3 median_simplify=14.80ms median_wire=14.85ms median_wall=56.12ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.55ms median_wire=12.62ms median_wall=47.52ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.70s | passed=450 failed=0 total=450 avg_case=6.000ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.95s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.85s | passed=1 failed=0 |
