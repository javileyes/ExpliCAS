# Engine Improvement Scorecard

- Generated: 2026-06-24T22:49:43.228532+00:00
- Git branch: main
- Git commit: `dd0dae8eab370bafca4fafee8227b90d96afd1f8`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=800.07ms avg_case_ms=8.00 simplify=230.17ms avg_simplify_ms=2.30, sum total=200 failed=0 elapsed=712.63ms avg_case_ms=3.56 simplify=243.88ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=482.53ms avg_case_ms=4.83 simplify=140.30ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=346.72ms avg_case_ms=6.93 simplify=113.02ms avg_simplify_ms=2.26
- Engine hotspots: sum simplify=243.88ms avg_simplify_ms=1.22 wall=712.63ms, shifted_quotient simplify=230.17ms avg_simplify_ms=2.30 wall=800.07ms, product simplify=140.30ms avg_simplify_ms=1.40 wall=482.53ms, difference simplify=113.02ms avg_simplify_ms=2.26 wall=346.72ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=800.07ms avg_case_ms=8.00 avg_simplify_ms=2.30, sum@0+100 failed=0 elapsed=515.62ms avg_case_ms=5.16 avg_simplify_ms=1.69, product@0+100 failed=0 elapsed=482.53ms avg_case_ms=4.83 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=346.72ms avg_case_ms=6.93 avg_simplify_ms=2.26, sum@700+100 failed=0 elapsed=197.01ms avg_case_ms=1.97 avg_simplify_ms=0.75
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.13ms median_wire=13.20ms median_wall=50.23ms, sum@0+100 #173 sum runs=3 median_simplify=12.06ms median_wire=12.11ms median_wall=45.44ms, product@0+100 #175 product runs=3 median_simplify=11.90ms median_wire=11.95ms median_wall=45.34ms, difference@0+50 #174 difference runs=3 median_simplify=11.94ms median_wire=11.99ms median_wall=45.39ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.82ms median_wire=10.90ms median_wall=40.95ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
