# Engine Improvement Scorecard

- Generated: 2026-07-14T09:52:18.129491+00:00
- Git branch: main
- Git commit: `377109652d5f86e21443877a9f90c5fe4bce117e`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=961.87ms avg_case_ms=9.62 simplify=269.03ms avg_simplify_ms=2.69, sum total=200 failed=0 elapsed=863.62ms avg_case_ms=4.32 simplify=281.42ms avg_simplify_ms=1.41, product total=100 failed=0 elapsed=589.47ms avg_case_ms=5.89 simplify=167.46ms avg_simplify_ms=1.67, difference total=50 failed=0 elapsed=391.90ms avg_case_ms=7.84 simplify=119.28ms avg_simplify_ms=2.39
- Engine hotspots: sum simplify=281.42ms avg_simplify_ms=1.41 wall=863.62ms, shifted_quotient simplify=269.03ms avg_simplify_ms=2.69 wall=961.87ms, product simplify=167.46ms avg_simplify_ms=1.67 wall=589.47ms, difference simplify=119.28ms avg_simplify_ms=2.39 wall=391.90ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=961.87ms avg_case_ms=9.62 avg_simplify_ms=2.69, sum@0+100 failed=0 elapsed=633.28ms avg_case_ms=6.33 avg_simplify_ms=2.00, product@0+100 failed=0 elapsed=589.47ms avg_case_ms=5.89 avg_simplify_ms=1.67, difference@0+50 failed=0 elapsed=391.90ms avg_case_ms=7.84 avg_simplify_ms=2.39, sum@700+100 failed=0 elapsed=230.35ms avg_case_ms=2.30 avg_simplify_ms=0.82
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=14.88ms median_wire=14.92ms median_wall=57.00ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.32ms median_wire=16.39ms median_wall=63.92ms, product@0+100 #175 product runs=3 median_simplify=14.97ms median_wire=15.02ms median_wall=64.39ms, difference@0+50 #174 difference runs=3 median_simplify=15.27ms median_wire=15.33ms median_wall=57.39ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.67ms median_wire=12.74ms median_wall=47.99ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.81s | passed=450 failed=0 total=450 avg_case=6.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
