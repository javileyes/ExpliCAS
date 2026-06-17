# Engine Improvement Scorecard

- Generated: 2026-06-17T16:20:01.136985+00:00
- Git branch: main
- Git commit: `db6a53000c88c4d341ff2107bb2a87a02e1f612f`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=796.37ms avg_case_ms=7.96 simplify=228.19ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=705.68ms avg_case_ms=3.53 simplify=238.60ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=481.27ms avg_case_ms=4.81 simplify=138.93ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=330.19ms avg_case_ms=6.60 simplify=105.39ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=238.60ms avg_simplify_ms=1.19 wall=705.68ms, shifted_quotient simplify=228.19ms avg_simplify_ms=2.28 wall=796.37ms, product simplify=138.93ms avg_simplify_ms=1.39 wall=481.27ms, difference simplify=105.39ms avg_simplify_ms=2.11 wall=330.19ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=796.37ms avg_case_ms=7.96 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=509.23ms avg_case_ms=5.09 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=481.27ms avg_case_ms=4.81 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=330.19ms avg_case_ms=6.60 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=196.45ms avg_case_ms=1.96 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.19ms median_wire=13.26ms median_wall=50.43ms, product@0+100 #175 product runs=3 median_simplify=11.82ms median_wire=11.88ms median_wall=45.25ms, difference@0+50 #174 difference runs=3 median_simplify=11.78ms median_wire=11.83ms median_wall=44.96ms, sum@0+100 #173 sum runs=3 median_simplify=12.01ms median_wire=12.07ms median_wall=45.35ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.87ms median_wire=10.95ms median_wall=41.40ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.60s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
