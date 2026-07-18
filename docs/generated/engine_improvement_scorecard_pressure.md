# Engine Improvement Scorecard

- Generated: 2026-07-18T23:41:27.557598+00:00
- Git branch: main
- Git commit: `50c3585dc7d06fc463cdd91047f59c26f95b359c`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=367

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=986.82ms avg_case_ms=9.87 simplify=276.13ms avg_simplify_ms=2.76, sum total=200 failed=0 elapsed=895.04ms avg_case_ms=4.48 simplify=290.42ms avg_simplify_ms=1.45, product total=100 failed=0 elapsed=606.29ms avg_case_ms=6.06 simplify=174.19ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=403.09ms avg_case_ms=8.06 simplify=122.88ms avg_simplify_ms=2.46
- Engine hotspots: sum simplify=290.42ms avg_simplify_ms=1.45 wall=895.04ms, shifted_quotient simplify=276.13ms avg_simplify_ms=2.76 wall=986.82ms, product simplify=174.19ms avg_simplify_ms=1.74 wall=606.29ms, difference simplify=122.88ms avg_simplify_ms=2.46 wall=403.09ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=986.82ms avg_case_ms=9.87 avg_simplify_ms=2.76, sum@0+100 failed=0 elapsed=662.48ms avg_case_ms=6.62 avg_simplify_ms=2.07, product@0+100 failed=0 elapsed=606.29ms avg_case_ms=6.06 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=403.09ms avg_case_ms=8.06 avg_simplify_ms=2.46, sum@700+100 failed=0 elapsed=232.55ms avg_case_ms=2.33 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.74ms median_wire=16.81ms median_wall=64.30ms, difference@0+50 #174 difference runs=3 median_simplify=15.12ms median_wire=15.17ms median_wall=58.46ms, product@0+100 #175 product runs=3 median_simplify=15.20ms median_wire=15.25ms median_wall=58.48ms, sum@0+100 #173 sum runs=3 median_simplify=16.31ms median_wire=16.36ms median_wall=59.80ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.85ms median_wire=12.92ms median_wall=49.42ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.89s | passed=450 failed=0 total=450 avg_case=6.422ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
