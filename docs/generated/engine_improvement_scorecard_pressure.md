# Engine Improvement Scorecard

- Generated: 2026-06-11T20:31:22.689175+00:00
- Git branch: main
- Git commit: `7b935b63d6041b6d80eb987d6e1b17ea6c5b90eb`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=769.19ms avg_case_ms=7.69 simplify=218.18ms avg_simplify_ms=2.18, sum total=200 failed=0 elapsed=680.33ms avg_case_ms=3.40 simplify=227.75ms avg_simplify_ms=1.14, product total=100 failed=0 elapsed=471.95ms avg_case_ms=4.72 simplify=135.41ms avg_simplify_ms=1.35, difference total=50 failed=0 elapsed=332.77ms avg_case_ms=6.66 simplify=105.38ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=227.75ms avg_simplify_ms=1.14 wall=680.33ms, shifted_quotient simplify=218.18ms avg_simplify_ms=2.18 wall=769.19ms, product simplify=135.41ms avg_simplify_ms=1.35 wall=471.95ms, difference simplify=105.38ms avg_simplify_ms=2.11 wall=332.77ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=769.19ms avg_case_ms=7.69 avg_simplify_ms=2.18, sum@0+100 failed=0 elapsed=490.77ms avg_case_ms=4.91 avg_simplify_ms=1.58, product@0+100 failed=0 elapsed=471.95ms avg_case_ms=4.72 avg_simplify_ms=1.35, difference@0+50 failed=0 elapsed=332.77ms avg_case_ms=6.66 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=189.55ms avg_case_ms=1.90 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.79ms median_wire=12.85ms median_wall=48.97ms, difference@0+50 #174 difference runs=3 median_simplify=11.43ms median_wire=11.48ms median_wall=43.68ms, sum@0+100 #173 sum runs=3 median_simplify=11.43ms median_wire=11.47ms median_wall=43.62ms, product@0+100 #175 product runs=3 median_simplify=11.41ms median_wire=11.45ms median_wall=43.60ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.19ms median_wire=10.25ms median_wall=38.97ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.25s | passed=450 failed=0 total=450 avg_case=5.000ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.79s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.12s | passed=1 failed=0 |
