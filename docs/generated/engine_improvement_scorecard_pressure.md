# Engine Improvement Scorecard

- Generated: 2026-07-28T23:53:03.172020+00:00
- Git branch: main
- Git commit: `d0b357af48b208d9bc97ef093ea2d863f56a94e0`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=380

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=964.04ms avg_case_ms=9.64 simplify=270.80ms avg_simplify_ms=2.71, sum total=200 failed=0 elapsed=857.33ms avg_case_ms=4.29 simplify=278.87ms avg_simplify_ms=1.39, product total=100 failed=0 elapsed=593.98ms avg_case_ms=5.94 simplify=170.16ms avg_simplify_ms=1.70, difference total=50 failed=0 elapsed=398.65ms avg_case_ms=7.97 simplify=121.91ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=278.87ms avg_simplify_ms=1.39 wall=857.33ms, shifted_quotient simplify=270.80ms avg_simplify_ms=2.71 wall=964.04ms, product simplify=170.16ms avg_simplify_ms=1.70 wall=593.98ms, difference simplify=121.91ms avg_simplify_ms=2.44 wall=398.65ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=964.04ms avg_case_ms=9.64 avg_simplify_ms=2.71, sum@0+100 failed=0 elapsed=627.01ms avg_case_ms=6.27 avg_simplify_ms=1.97, product@0+100 failed=0 elapsed=593.98ms avg_case_ms=5.94 avg_simplify_ms=1.70, difference@0+50 failed=0 elapsed=398.65ms avg_case_ms=7.97 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=230.32ms avg_case_ms=2.30 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.73ms median_wire=16.80ms median_wall=63.82ms, sum@0+100 #173 sum runs=3 median_simplify=15.29ms median_wire=15.34ms median_wall=57.86ms, product@0+100 #175 product runs=3 median_simplify=14.85ms median_wire=14.89ms median_wall=56.86ms, difference@0+50 #174 difference runs=3 median_simplify=15.02ms median_wire=15.07ms median_wall=57.57ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.82ms median_wire=12.89ms median_wall=48.99ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.81s | passed=450 failed=0 total=450 avg_case=6.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.27s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.84s | passed=1 failed=0 |
