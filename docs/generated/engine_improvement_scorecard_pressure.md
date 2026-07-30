# Engine Improvement Scorecard

- Generated: 2026-07-30T22:24:30.625902+00:00
- Git branch: main
- Git commit: `488522043979b49429bcb8992892863984c8e9b5`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=966.60ms avg_case_ms=9.67 simplify=270.13ms avg_simplify_ms=2.70, sum total=200 failed=0 elapsed=854.24ms avg_case_ms=4.27 simplify=275.81ms avg_simplify_ms=1.38, product total=100 failed=0 elapsed=602.13ms avg_case_ms=6.02 simplify=173.20ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=396.29ms avg_case_ms=7.93 simplify=115.90ms avg_simplify_ms=2.32
- Engine hotspots: sum simplify=275.81ms avg_simplify_ms=1.38 wall=854.24ms, shifted_quotient simplify=270.13ms avg_simplify_ms=2.70 wall=966.60ms, product simplify=173.20ms avg_simplify_ms=1.73 wall=602.13ms, difference simplify=115.90ms avg_simplify_ms=2.32 wall=396.29ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=966.60ms avg_case_ms=9.67 avg_simplify_ms=2.70, sum@0+100 failed=0 elapsed=622.92ms avg_case_ms=6.23 avg_simplify_ms=1.91, product@0+100 failed=0 elapsed=602.13ms avg_case_ms=6.02 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=396.29ms avg_case_ms=7.93 avg_simplify_ms=2.32, sum@700+100 failed=0 elapsed=231.32ms avg_case_ms=2.31 avg_simplify_ms=0.85
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.05ms median_wire=17.12ms median_wall=64.48ms, difference@0+50 #174 difference runs=3 median_simplify=15.16ms median_wire=15.21ms median_wall=58.07ms, sum@0+100 #173 sum runs=3 median_simplify=15.42ms median_wire=15.46ms median_wall=58.69ms, product@0+100 #175 product runs=3 median_simplify=14.84ms median_wire=14.88ms median_wall=56.99ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.80ms median_wire=12.87ms median_wall=48.82ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.82s | passed=450 failed=0 total=450 avg_case=6.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.07s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.73s | passed=1 failed=0 |
