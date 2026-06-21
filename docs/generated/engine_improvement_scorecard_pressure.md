# Engine Improvement Scorecard

- Generated: 2026-06-21T17:47:37.797061+00:00
- Git branch: main
- Git commit: `200f3342c827612b089b30d6130bad72499b1782`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=788.99ms avg_case_ms=7.89 simplify=225.80ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=698.81ms avg_case_ms=3.49 simplify=236.89ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=479.09ms avg_case_ms=4.79 simplify=138.15ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=329.03ms avg_case_ms=6.58 simplify=105.58ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=236.89ms avg_simplify_ms=1.18 wall=698.81ms, shifted_quotient simplify=225.80ms avg_simplify_ms=2.26 wall=788.99ms, product simplify=138.15ms avg_simplify_ms=1.38 wall=479.09ms, difference simplify=105.58ms avg_simplify_ms=2.11 wall=329.03ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=788.99ms avg_case_ms=7.89 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=503.03ms avg_case_ms=5.03 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=479.09ms avg_case_ms=4.79 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=329.03ms avg_case_ms=6.58 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=195.78ms avg_case_ms=1.96 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.38ms median_wire=13.46ms median_wall=50.46ms, product@0+100 #175 product runs=3 median_simplify=11.69ms median_wire=11.74ms median_wall=44.72ms, sum@0+100 #173 sum runs=3 median_simplify=11.78ms median_wire=11.84ms median_wall=44.54ms, difference@0+50 #174 difference runs=3 median_simplify=11.58ms median_wire=11.63ms median_wall=44.14ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.50ms median_wire=10.57ms median_wall=39.87ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
