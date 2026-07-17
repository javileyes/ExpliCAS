# Engine Improvement Scorecard

- Generated: 2026-07-17T08:20:12.853380+00:00
- Git branch: main
- Git commit: `4f9b1d9f2efa3770e8cf029a62f1b7f6be684893`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.00s avg_case_ms=10.01 simplify=279.19ms avg_simplify_ms=2.79, sum total=200 failed=0 elapsed=884.36ms avg_case_ms=4.42 simplify=283.25ms avg_simplify_ms=1.42, product total=100 failed=0 elapsed=602.93ms avg_case_ms=6.03 simplify=173.55ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=405.27ms avg_case_ms=8.11 simplify=122.74ms avg_simplify_ms=2.45
- Engine hotspots: sum simplify=283.25ms avg_simplify_ms=1.42 wall=884.36ms, shifted_quotient simplify=279.19ms avg_simplify_ms=2.79 wall=1.00s, product simplify=173.55ms avg_simplify_ms=1.74 wall=602.93ms, difference simplify=122.74ms avg_simplify_ms=2.45 wall=405.27ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.00s avg_case_ms=10.01 avg_simplify_ms=2.79, sum@0+100 failed=0 elapsed=651.41ms avg_case_ms=6.51 avg_simplify_ms=1.99, product@0+100 failed=0 elapsed=602.93ms avg_case_ms=6.03 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=405.27ms avg_case_ms=8.11 avg_simplify_ms=2.45, sum@700+100 failed=0 elapsed=232.95ms avg_case_ms=2.33 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.93ms median_wire=17.01ms median_wall=65.31ms, sum@0+100 #173 sum runs=3 median_simplify=15.36ms median_wire=15.42ms median_wall=58.81ms, difference@0+50 #174 difference runs=3 median_simplify=15.24ms median_wire=15.30ms median_wall=58.68ms, product@0+100 #175 product runs=3 median_simplify=15.31ms median_wire=15.36ms median_wall=58.80ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.29ms median_wire=13.36ms median_wall=49.80ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.89s | passed=450 failed=0 total=450 avg_case=6.422ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.52s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
