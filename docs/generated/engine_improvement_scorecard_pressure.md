# Engine Improvement Scorecard

- Generated: 2026-06-27T15:42:08.732960+00:00
- Git branch: main
- Git commit: `ec89cbfc920e3993c57c8530ce723aca7f36a4af`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=353

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=832.49ms avg_case_ms=8.32 simplify=242.06ms avg_simplify_ms=2.42, sum total=200 failed=0 elapsed=752.87ms avg_case_ms=3.76 simplify=264.38ms avg_simplify_ms=1.32, product total=100 failed=0 elapsed=512.73ms avg_case_ms=5.13 simplify=153.60ms avg_simplify_ms=1.54, difference total=50 failed=0 elapsed=359.51ms avg_case_ms=7.19 simplify=119.05ms avg_simplify_ms=2.38
- Engine hotspots: sum simplify=264.38ms avg_simplify_ms=1.32 wall=752.87ms, shifted_quotient simplify=242.06ms avg_simplify_ms=2.42 wall=832.49ms, product simplify=153.60ms avg_simplify_ms=1.54 wall=512.73ms, difference simplify=119.05ms avg_simplify_ms=2.38 wall=359.51ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=832.49ms avg_case_ms=8.32 avg_simplify_ms=2.42, sum@0+100 failed=0 elapsed=543.14ms avg_case_ms=5.43 avg_simplify_ms=1.83, product@0+100 failed=0 elapsed=512.73ms avg_case_ms=5.13 avg_simplify_ms=1.54, difference@0+50 failed=0 elapsed=359.51ms avg_case_ms=7.19 avg_simplify_ms=2.38, sum@700+100 failed=0 elapsed=209.73ms avg_case_ms=2.10 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.15ms median_wire=13.22ms median_wall=50.66ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.41ms median_wire=12.49ms median_wall=42.61ms, sum@0+100 #173 sum runs=3 median_simplify=11.93ms median_wire=11.98ms median_wall=45.72ms, product@0+100 #175 product runs=3 median_simplify=11.95ms median_wire=12.01ms median_wall=45.27ms, difference@0+50 #174 difference runs=3 median_simplify=12.49ms median_wire=12.54ms median_wall=50.26ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), shifted_quotient@0+100 #4 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.46s | passed=450 failed=0 total=450 avg_case=5.467ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.57s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.14s | passed=1 failed=0 |
