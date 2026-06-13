# Engine Improvement Scorecard

- Generated: 2026-06-13T23:51:26.132459+00:00
- Git branch: main
- Git commit: `4bfda5e08ffeb44a61c6d873e62815d2b16e474a`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=785.43ms avg_case_ms=7.85 simplify=224.41ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=687.84ms avg_case_ms=3.44 simplify=230.77ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=469.72ms avg_case_ms=4.70 simplify=134.79ms avg_simplify_ms=1.35, difference total=50 failed=0 elapsed=322.64ms avg_case_ms=6.45 simplify=102.63ms avg_simplify_ms=2.05
- Engine hotspots: sum simplify=230.77ms avg_simplify_ms=1.15 wall=687.84ms, shifted_quotient simplify=224.41ms avg_simplify_ms=2.24 wall=785.43ms, product simplify=134.79ms avg_simplify_ms=1.35 wall=469.72ms, difference simplify=102.63ms avg_simplify_ms=2.05 wall=322.64ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=785.43ms avg_case_ms=7.85 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=499.04ms avg_case_ms=4.99 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=469.72ms avg_case_ms=4.70 avg_simplify_ms=1.35, difference@0+50 failed=0 elapsed=322.64ms avg_case_ms=6.45 avg_simplify_ms=2.05, sum@700+100 failed=0 elapsed=188.80ms avg_case_ms=1.89 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.85ms median_wire=12.92ms median_wall=49.17ms, sum@0+100 #173 sum runs=3 median_simplify=11.47ms median_wire=11.51ms median_wall=43.98ms, difference@0+50 #174 difference runs=3 median_simplify=11.53ms median_wire=11.58ms median_wall=44.11ms, product@0+100 #175 product runs=3 median_simplify=11.70ms median_wire=11.75ms median_wall=44.38ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.56ms median_wire=10.63ms median_wall=40.08ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.27s | passed=450 failed=0 total=450 avg_case=5.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
