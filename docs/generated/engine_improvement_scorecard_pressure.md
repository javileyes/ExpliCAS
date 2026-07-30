# Engine Improvement Scorecard

- Generated: 2026-07-30T19:34:28.052069+00:00
- Git branch: main
- Git commit: `899ac46cb96d9bbe1289f802e8dd542060a06fbd`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=380

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=969.83ms avg_case_ms=9.70 simplify=270.88ms avg_simplify_ms=2.71, sum total=200 failed=0 elapsed=847.02ms avg_case_ms=4.24 simplify=271.81ms avg_simplify_ms=1.36, product total=100 failed=0 elapsed=603.55ms avg_case_ms=6.04 simplify=173.77ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=392.37ms avg_case_ms=7.85 simplify=115.42ms avg_simplify_ms=2.31
- Engine hotspots: sum simplify=271.81ms avg_simplify_ms=1.36 wall=847.02ms, shifted_quotient simplify=270.88ms avg_simplify_ms=2.71 wall=969.83ms, product simplify=173.77ms avg_simplify_ms=1.74 wall=603.55ms, difference simplify=115.42ms avg_simplify_ms=2.31 wall=392.37ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=969.83ms avg_case_ms=9.70 avg_simplify_ms=2.71, sum@0+100 failed=0 elapsed=613.59ms avg_case_ms=6.14 avg_simplify_ms=1.88, product@0+100 failed=0 elapsed=603.55ms avg_case_ms=6.04 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=392.37ms avg_case_ms=7.85 avg_simplify_ms=2.31, sum@700+100 failed=0 elapsed=233.43ms avg_case_ms=2.33 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.66ms median_wire=16.73ms median_wall=62.81ms, sum@0+100 #173 sum runs=3 median_simplify=15.01ms median_wire=15.05ms median_wall=57.43ms, difference@0+50 #174 difference runs=3 median_simplify=14.92ms median_wire=14.97ms median_wall=57.26ms, product@0+100 #175 product runs=3 median_simplify=14.83ms median_wire=14.88ms median_wall=56.47ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.70ms median_wire=12.77ms median_wall=48.22ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.81s | passed=450 failed=0 total=450 avg_case=6.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.12s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.73s | passed=1 failed=0 |
