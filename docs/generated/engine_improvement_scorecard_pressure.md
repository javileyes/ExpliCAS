# Engine Improvement Scorecard

- Generated: 2026-07-10T19:33:14.234396+00:00
- Git branch: main
- Git commit: `f1c011f41c1d00fc6b8f8c0149b60b7d102dcb1a`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=962.47ms avg_case_ms=9.62 simplify=267.37ms avg_simplify_ms=2.67, sum total=200 failed=0 elapsed=832.63ms avg_case_ms=4.16 simplify=268.96ms avg_simplify_ms=1.34, product total=100 failed=0 elapsed=574.94ms avg_case_ms=5.75 simplify=163.01ms avg_simplify_ms=1.63, difference total=50 failed=0 elapsed=390.30ms avg_case_ms=7.81 simplify=118.23ms avg_simplify_ms=2.36
- Engine hotspots: sum simplify=268.96ms avg_simplify_ms=1.34 wall=832.63ms, shifted_quotient simplify=267.37ms avg_simplify_ms=2.67 wall=962.47ms, product simplify=163.01ms avg_simplify_ms=1.63 wall=574.94ms, difference simplify=118.23ms avg_simplify_ms=2.36 wall=390.30ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=962.47ms avg_case_ms=9.62 avg_simplify_ms=2.67, sum@0+100 failed=0 elapsed=611.20ms avg_case_ms=6.11 avg_simplify_ms=1.90, product@0+100 failed=0 elapsed=574.94ms avg_case_ms=5.75 avg_simplify_ms=1.63, difference@0+50 failed=0 elapsed=390.30ms avg_case_ms=7.81 avg_simplify_ms=2.36, sum@700+100 failed=0 elapsed=221.43ms avg_case_ms=2.21 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.28ms median_wire=16.34ms median_wall=62.38ms, difference@0+50 #174 difference runs=3 median_simplify=14.59ms median_wire=14.64ms median_wall=55.72ms, product@0+100 #175 product runs=3 median_simplify=14.48ms median_wire=14.53ms median_wall=55.26ms, sum@0+100 #173 sum runs=3 median_simplify=14.58ms median_wire=14.63ms median_wall=55.66ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.28ms median_wire=12.35ms median_wall=46.95ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.76s | passed=450 failed=0 total=450 avg_case=6.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.00s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.88s | passed=1 failed=0 |
