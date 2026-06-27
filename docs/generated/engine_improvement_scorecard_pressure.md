# Engine Improvement Scorecard

- Generated: 2026-06-27T07:31:39.623491+00:00
- Git branch: main
- Git commit: `f3988e95944ac8fba993c9d9bc1b6c8a9e29678d`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=802.26ms avg_case_ms=8.02 simplify=230.81ms avg_simplify_ms=2.31, sum total=200 failed=0 elapsed=713.54ms avg_case_ms=3.57 simplify=246.91ms avg_simplify_ms=1.23, product total=100 failed=0 elapsed=488.97ms avg_case_ms=4.89 simplify=143.71ms avg_simplify_ms=1.44, difference total=50 failed=0 elapsed=331.81ms avg_case_ms=6.64 simplify=107.40ms avg_simplify_ms=2.15
- Engine hotspots: sum simplify=246.91ms avg_simplify_ms=1.23 wall=713.54ms, shifted_quotient simplify=230.81ms avg_simplify_ms=2.31 wall=802.26ms, product simplify=143.71ms avg_simplify_ms=1.44 wall=488.97ms, difference simplify=107.40ms avg_simplify_ms=2.15 wall=331.81ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=802.26ms avg_case_ms=8.02 avg_simplify_ms=2.31, sum@0+100 failed=0 elapsed=514.86ms avg_case_ms=5.15 avg_simplify_ms=1.70, product@0+100 failed=0 elapsed=488.97ms avg_case_ms=4.89 avg_simplify_ms=1.44, difference@0+50 failed=0 elapsed=331.81ms avg_case_ms=6.64 avg_simplify_ms=2.15, sum@700+100 failed=0 elapsed=198.68ms avg_case_ms=1.99 avg_simplify_ms=0.77
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.30ms median_wire=13.37ms median_wall=49.76ms, difference@0+50 #174 difference runs=3 median_simplify=11.85ms median_wire=11.90ms median_wall=44.82ms, sum@0+100 #173 sum runs=3 median_simplify=11.81ms median_wire=11.87ms median_wall=44.53ms, product@0+100 #175 product runs=3 median_simplify=11.64ms median_wire=11.69ms median_wall=44.23ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.87ms median_wire=10.95ms median_wall=41.52ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.08s | passed=1 failed=0 |
