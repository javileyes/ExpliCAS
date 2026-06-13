# Engine Improvement Scorecard

- Generated: 2026-06-13T07:23:55.480296+00:00
- Git branch: main
- Git commit: `3a32251d75cc4ad2851dce6a724af7387359c8f4`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=785.13ms avg_case_ms=7.85 simplify=224.12ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=686.41ms avg_case_ms=3.43 simplify=230.24ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=514.76ms avg_case_ms=5.15 simplify=147.77ms avg_simplify_ms=1.48, difference total=50 failed=0 elapsed=321.00ms avg_case_ms=6.42 simplify=101.53ms avg_simplify_ms=2.03
- Engine hotspots: sum simplify=230.24ms avg_simplify_ms=1.15 wall=686.41ms, shifted_quotient simplify=224.12ms avg_simplify_ms=2.24 wall=785.13ms, product simplify=147.77ms avg_simplify_ms=1.48 wall=514.76ms, difference simplify=101.53ms avg_simplify_ms=2.03 wall=321.00ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=785.13ms avg_case_ms=7.85 avg_simplify_ms=2.24, product@0+100 failed=0 elapsed=514.76ms avg_case_ms=5.15 avg_simplify_ms=1.48, sum@0+100 failed=0 elapsed=496.91ms avg_case_ms=4.97 avg_simplify_ms=1.60, difference@0+50 failed=0 elapsed=321.00ms avg_case_ms=6.42 avg_simplify_ms=2.03, sum@700+100 failed=0 elapsed=189.49ms avg_case_ms=1.89 avg_simplify_ms=0.70
- Steady-state engine reruns: product@0+100 #175 product runs=3 median_simplify=11.37ms median_wire=11.41ms median_wall=43.64ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.86ms median_wire=12.93ms median_wall=49.22ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.45ms median_wire=10.52ms median_wall=39.86ms, sum@0+100 #173 sum runs=3 median_simplify=11.62ms median_wire=11.66ms median_wall=44.10ms, difference@0+50 #174 difference runs=3 median_simplify=11.49ms median_wire=11.53ms median_wall=44.35ms
- Steady-state dominant expressions: product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), shifted_quotient@0+100 #4 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
