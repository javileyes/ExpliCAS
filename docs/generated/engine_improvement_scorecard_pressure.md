# Engine Improvement Scorecard

- Generated: 2026-07-06T22:06:49.370402+00:00
- Git branch: main
- Git commit: `f281942946730ff630687233b56cdb278e4b293d`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=921.94ms avg_case_ms=9.22 simplify=255.32ms avg_simplify_ms=2.55, sum total=200 failed=0 elapsed=821.75ms avg_case_ms=4.11 simplify=266.57ms avg_simplify_ms=1.33, product total=100 failed=0 elapsed=582.58ms avg_case_ms=5.83 simplify=166.37ms avg_simplify_ms=1.66, difference total=50 failed=0 elapsed=380.22ms avg_case_ms=7.60 simplify=115.50ms avg_simplify_ms=2.31
- Engine hotspots: sum simplify=266.57ms avg_simplify_ms=1.33 wall=821.75ms, shifted_quotient simplify=255.32ms avg_simplify_ms=2.55 wall=921.94ms, product simplify=166.37ms avg_simplify_ms=1.66 wall=582.58ms, difference simplify=115.50ms avg_simplify_ms=2.31 wall=380.22ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=921.94ms avg_case_ms=9.22 avg_simplify_ms=2.55, sum@0+100 failed=0 elapsed=599.32ms avg_case_ms=5.99 avg_simplify_ms=1.87, product@0+100 failed=0 elapsed=582.58ms avg_case_ms=5.83 avg_simplify_ms=1.66, difference@0+50 failed=0 elapsed=380.22ms avg_case_ms=7.60 avg_simplify_ms=2.31, sum@700+100 failed=0 elapsed=222.43ms avg_case_ms=2.22 avg_simplify_ms=0.79
- Steady-state engine reruns: product@0+100 #175 product runs=3 median_simplify=14.48ms median_wire=14.52ms median_wall=55.71ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=15.80ms median_wire=15.87ms median_wall=61.98ms, sum@0+100 #173 sum runs=3 median_simplify=14.79ms median_wire=14.83ms median_wall=56.50ms, difference@0+50 #174 difference runs=3 median_simplify=14.88ms median_wire=14.93ms median_wall=56.65ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.34ms median_wire=12.40ms median_wall=46.97ms
- Steady-state dominant expressions: product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.71s | passed=450 failed=0 total=450 avg_case=6.022ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.97s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.86s | passed=1 failed=0 |
