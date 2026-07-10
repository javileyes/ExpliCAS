# Engine Improvement Scorecard

- Generated: 2026-07-10T08:33:57.582933+00:00
- Git branch: main
- Git commit: `74374a747b1483fa7f3e24b04f89e8678ff0ca2c`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=933.97ms avg_case_ms=9.34 simplify=258.68ms avg_simplify_ms=2.59, sum total=200 failed=0 elapsed=826.77ms avg_case_ms=4.13 simplify=266.29ms avg_simplify_ms=1.33, product total=100 failed=0 elapsed=570.41ms avg_case_ms=5.70 simplify=162.42ms avg_simplify_ms=1.62, difference total=50 failed=0 elapsed=386.36ms avg_case_ms=7.73 simplify=116.61ms avg_simplify_ms=2.33
- Engine hotspots: sum simplify=266.29ms avg_simplify_ms=1.33 wall=826.77ms, shifted_quotient simplify=258.68ms avg_simplify_ms=2.59 wall=933.97ms, product simplify=162.42ms avg_simplify_ms=1.62 wall=570.41ms, difference simplify=116.61ms avg_simplify_ms=2.33 wall=386.36ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=933.97ms avg_case_ms=9.34 avg_simplify_ms=2.59, sum@0+100 failed=0 elapsed=605.01ms avg_case_ms=6.05 avg_simplify_ms=1.87, product@0+100 failed=0 elapsed=570.41ms avg_case_ms=5.70 avg_simplify_ms=1.62, difference@0+50 failed=0 elapsed=386.36ms avg_case_ms=7.73 avg_simplify_ms=2.33, sum@700+100 failed=0 elapsed=221.76ms avg_case_ms=2.22 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.18ms median_wire=16.24ms median_wall=61.95ms, sum@0+100 #173 sum runs=3 median_simplify=14.72ms median_wire=14.76ms median_wall=55.93ms, difference@0+50 #174 difference runs=3 median_simplify=14.65ms median_wire=14.70ms median_wall=55.74ms, product@0+100 #175 product runs=3 median_simplify=14.87ms median_wire=14.92ms median_wall=56.52ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.55ms median_wire=12.62ms median_wall=47.88ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.72s | passed=450 failed=0 total=450 avg_case=6.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.99s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
