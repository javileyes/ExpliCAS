# Engine Improvement Scorecard

- Generated: 2026-06-10T09:46:17.860060+00:00
- Git branch: main
- Git commit: `fd3c14e75850feeeac659f07577f4a9f217df44c`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=768.60ms avg_case_ms=7.69 simplify=217.34ms avg_simplify_ms=2.17, sum total=200 failed=0 elapsed=699.56ms avg_case_ms=3.50 simplify=234.08ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=472.22ms avg_case_ms=4.72 simplify=135.62ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=321.84ms avg_case_ms=6.44 simplify=101.92ms avg_simplify_ms=2.04
- Engine hotspots: sum simplify=234.08ms avg_simplify_ms=1.17 wall=699.56ms, shifted_quotient simplify=217.34ms avg_simplify_ms=2.17 wall=768.60ms, product simplify=135.62ms avg_simplify_ms=1.36 wall=472.22ms, difference simplify=101.92ms avg_simplify_ms=2.04 wall=321.84ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=768.60ms avg_case_ms=7.69 avg_simplify_ms=2.17, sum@0+100 failed=0 elapsed=508.78ms avg_case_ms=5.09 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=472.22ms avg_case_ms=4.72 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=321.84ms avg_case_ms=6.44 avg_simplify_ms=2.04, sum@700+100 failed=0 elapsed=190.79ms avg_case_ms=1.91 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.75ms median_wire=12.82ms median_wall=48.62ms, sum@0+100 #173 sum runs=3 median_simplify=11.63ms median_wire=11.68ms median_wall=44.02ms, product@0+100 #175 product runs=3 median_simplify=11.45ms median_wire=11.49ms median_wall=43.70ms, difference@0+50 #174 difference runs=3 median_simplify=11.36ms median_wire=11.41ms median_wall=43.44ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.84ms median_wire=10.92ms median_wall=40.80ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.26s | passed=450 failed=0 total=450 avg_case=5.022ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.05s | passed=1 failed=0 |
