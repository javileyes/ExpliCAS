# Engine Improvement Scorecard

- Generated: 2026-06-27T09:49:38.383409+00:00
- Git branch: main
- Git commit: `9e7bb43f078d042476435cea3ea3bd933f60b361`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=794.71ms avg_case_ms=7.95 simplify=227.28ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=706.99ms avg_case_ms=3.53 simplify=243.82ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=486.00ms avg_case_ms=4.86 simplify=142.01ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=330.10ms avg_case_ms=6.60 simplify=106.24ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=243.82ms avg_simplify_ms=1.22 wall=706.99ms, shifted_quotient simplify=227.28ms avg_simplify_ms=2.27 wall=794.71ms, product simplify=142.01ms avg_simplify_ms=1.42 wall=486.00ms, difference simplify=106.24ms avg_simplify_ms=2.12 wall=330.10ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=794.71ms avg_case_ms=7.95 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=507.59ms avg_case_ms=5.08 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=486.00ms avg_case_ms=4.86 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=330.10ms avg_case_ms=6.60 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=199.40ms avg_case_ms=1.99 avg_simplify_ms=0.77
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.46ms median_wire=13.53ms median_wall=52.10ms, sum@0+100 #173 sum runs=3 median_simplify=12.15ms median_wire=12.20ms median_wall=45.77ms, difference@0+50 #174 difference runs=3 median_simplify=11.81ms median_wire=11.86ms median_wall=45.17ms, product@0+100 #175 product runs=3 median_simplify=11.60ms median_wire=11.66ms median_wall=44.62ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.83ms median_wire=10.91ms median_wall=41.06ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
