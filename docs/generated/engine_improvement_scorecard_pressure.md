# Engine Improvement Scorecard

- Generated: 2026-06-25T22:31:56.808630+00:00
- Git branch: main
- Git commit: `1aeae9a95110e5ee8ed9996f4c2e8b9e22427db2`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=812.23ms avg_case_ms=8.12 simplify=232.64ms avg_simplify_ms=2.33, sum total=200 failed=0 elapsed=713.73ms avg_case_ms=3.57 simplify=248.46ms avg_simplify_ms=1.24, product total=100 failed=0 elapsed=502.86ms avg_case_ms=5.03 simplify=148.88ms avg_simplify_ms=1.49, difference total=50 failed=0 elapsed=335.23ms avg_case_ms=6.70 simplify=108.77ms avg_simplify_ms=2.18
- Engine hotspots: sum simplify=248.46ms avg_simplify_ms=1.24 wall=713.73ms, shifted_quotient simplify=232.64ms avg_simplify_ms=2.33 wall=812.23ms, product simplify=148.88ms avg_simplify_ms=1.49 wall=502.86ms, difference simplify=108.77ms avg_simplify_ms=2.18 wall=335.23ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=812.23ms avg_case_ms=8.12 avg_simplify_ms=2.33, sum@0+100 failed=0 elapsed=517.33ms avg_case_ms=5.17 avg_simplify_ms=1.73, product@0+100 failed=0 elapsed=502.86ms avg_case_ms=5.03 avg_simplify_ms=1.49, difference@0+50 failed=0 elapsed=335.23ms avg_case_ms=6.70 avg_simplify_ms=2.18, sum@700+100 failed=0 elapsed=196.39ms avg_case_ms=1.96 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.33ms median_wire=13.40ms median_wall=51.10ms, sum@0+100 #173 sum runs=3 median_simplify=12.11ms median_wire=12.17ms median_wall=45.82ms, difference@0+50 #174 difference runs=3 median_simplify=12.04ms median_wire=12.09ms median_wall=45.76ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.83ms median_wire=10.92ms median_wall=41.12ms, product@0+100 #175 product runs=3 median_simplify=12.04ms median_wire=12.09ms median_wall=45.71ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.36s | passed=450 failed=0 total=450 avg_case=5.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.47s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
