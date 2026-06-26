# Engine Improvement Scorecard

- Generated: 2026-06-26T07:41:06.744259+00:00
- Git branch: main
- Git commit: `94cee753e76316d34bf6e605e346a4cc1669cba4`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=803.85ms avg_case_ms=8.04 simplify=230.91ms avg_simplify_ms=2.31, sum total=200 failed=0 elapsed=716.40ms avg_case_ms=3.58 simplify=247.93ms avg_simplify_ms=1.24, product total=100 failed=0 elapsed=493.02ms avg_case_ms=4.93 simplify=145.41ms avg_simplify_ms=1.45, difference total=50 failed=0 elapsed=330.54ms avg_case_ms=6.61 simplify=106.61ms avg_simplify_ms=2.13
- Engine hotspots: sum simplify=247.93ms avg_simplify_ms=1.24 wall=716.40ms, shifted_quotient simplify=230.91ms avg_simplify_ms=2.31 wall=803.85ms, product simplify=145.41ms avg_simplify_ms=1.45 wall=493.02ms, difference simplify=106.61ms avg_simplify_ms=2.13 wall=330.54ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=803.85ms avg_case_ms=8.04 avg_simplify_ms=2.31, sum@0+100 failed=0 elapsed=517.21ms avg_case_ms=5.17 avg_simplify_ms=1.71, product@0+100 failed=0 elapsed=493.02ms avg_case_ms=4.93 avg_simplify_ms=1.45, difference@0+50 failed=0 elapsed=330.54ms avg_case_ms=6.61 avg_simplify_ms=2.13, sum@700+100 failed=0 elapsed=199.19ms avg_case_ms=1.99 avg_simplify_ms=0.77
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.07ms median_wire=13.14ms median_wall=49.70ms, sum@0+100 #173 sum runs=3 median_simplify=11.65ms median_wire=11.70ms median_wall=44.48ms, product@0+100 #175 product runs=3 median_simplify=11.59ms median_wire=11.64ms median_wall=44.27ms, difference@0+50 #174 difference runs=3 median_simplify=11.96ms median_wire=12.02ms median_wall=45.39ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.72ms median_wire=10.79ms median_wall=40.39ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.50s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.08s | passed=1 failed=0 |
