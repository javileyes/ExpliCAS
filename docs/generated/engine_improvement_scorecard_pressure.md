# Engine Improvement Scorecard

- Generated: 2026-07-24T09:38:25.789700+00:00
- Git branch: main
- Git commit: `797a93c214e2d362e0b2d13410269e92a670db57`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=981.26ms avg_case_ms=9.81 simplify=275.45ms avg_simplify_ms=2.75, sum total=200 failed=0 elapsed=886.15ms avg_case_ms=4.43 simplify=284.86ms avg_simplify_ms=1.42, product total=100 failed=0 elapsed=611.22ms avg_case_ms=6.11 simplify=176.76ms avg_simplify_ms=1.77, difference total=50 failed=0 elapsed=406.68ms avg_case_ms=8.13 simplify=124.21ms avg_simplify_ms=2.48
- Engine hotspots: sum simplify=284.86ms avg_simplify_ms=1.42 wall=886.15ms, shifted_quotient simplify=275.45ms avg_simplify_ms=2.75 wall=981.26ms, product simplify=176.76ms avg_simplify_ms=1.77 wall=611.22ms, difference simplify=124.21ms avg_simplify_ms=2.48 wall=406.68ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=981.26ms avg_case_ms=9.81 avg_simplify_ms=2.75, sum@0+100 failed=0 elapsed=648.16ms avg_case_ms=6.48 avg_simplify_ms=1.99, product@0+100 failed=0 elapsed=611.22ms avg_case_ms=6.11 avg_simplify_ms=1.77, difference@0+50 failed=0 elapsed=406.68ms avg_case_ms=8.13 avg_simplify_ms=2.48, sum@700+100 failed=0 elapsed=237.99ms avg_case_ms=2.38 avg_simplify_ms=0.86
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.77ms median_wire=16.84ms median_wall=64.81ms, product@0+100 #175 product runs=3 median_simplify=15.26ms median_wire=15.31ms median_wall=58.79ms, sum@0+100 #173 sum runs=3 median_simplify=15.41ms median_wire=15.46ms median_wall=58.99ms, difference@0+50 #174 difference runs=3 median_simplify=15.63ms median_wire=15.68ms median_wall=59.43ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.46ms median_wire=13.54ms median_wall=50.84ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.89s | passed=450 failed=0 total=450 avg_case=6.422ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.66s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
