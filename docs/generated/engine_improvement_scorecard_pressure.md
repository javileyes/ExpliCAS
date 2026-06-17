# Engine Improvement Scorecard

- Generated: 2026-06-17T20:00:12.289354+00:00
- Git branch: main
- Git commit: `3f646d795c5b9e0ac7963c89f70418ab693f677a`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=800.03ms avg_case_ms=8.00 simplify=231.25ms avg_simplify_ms=2.31, sum total=200 failed=0 elapsed=755.52ms avg_case_ms=3.78 simplify=255.93ms avg_simplify_ms=1.28, product total=100 failed=0 elapsed=486.50ms avg_case_ms=4.86 simplify=140.48ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=333.88ms avg_case_ms=6.68 simplify=106.88ms avg_simplify_ms=2.14
- Engine hotspots: sum simplify=255.93ms avg_simplify_ms=1.28 wall=755.52ms, shifted_quotient simplify=231.25ms avg_simplify_ms=2.31 wall=800.03ms, product simplify=140.48ms avg_simplify_ms=1.40 wall=486.50ms, difference simplify=106.88ms avg_simplify_ms=2.14 wall=333.88ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=800.03ms avg_case_ms=8.00 avg_simplify_ms=2.31, sum@0+100 failed=0 elapsed=557.29ms avg_case_ms=5.57 avg_simplify_ms=1.81, product@0+100 failed=0 elapsed=486.50ms avg_case_ms=4.86 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=333.88ms avg_case_ms=6.68 avg_simplify_ms=2.14, sum@700+100 failed=0 elapsed=198.23ms avg_case_ms=1.98 avg_simplify_ms=0.75
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=11.65ms median_wire=11.70ms median_wall=44.47ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.09ms median_wire=13.16ms median_wall=49.52ms, difference@0+50 #174 difference runs=3 median_simplify=11.78ms median_wire=11.84ms median_wall=44.57ms, product@0+100 #175 product runs=3 median_simplify=11.54ms median_wire=11.60ms median_wall=44.18ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.56ms median_wire=10.63ms median_wall=40.05ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.38s | passed=450 failed=0 total=450 avg_case=5.289ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
