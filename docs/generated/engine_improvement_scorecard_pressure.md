# Engine Improvement Scorecard

- Generated: 2026-07-08T20:34:51.614876+00:00
- Git branch: main
- Git commit: `18c96a39bb19fea133c34cd6c0a4a4a6e0d9596f`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.01s avg_case_ms=10.08 simplify=281.21ms avg_simplify_ms=2.81, sum total=200 failed=0 elapsed=883.27ms avg_case_ms=4.42 simplify=286.95ms avg_simplify_ms=1.43, product total=100 failed=0 elapsed=615.04ms avg_case_ms=6.15 simplify=176.63ms avg_simplify_ms=1.77, difference total=50 failed=0 elapsed=410.91ms avg_case_ms=8.22 simplify=125.34ms avg_simplify_ms=2.51
- Engine hotspots: sum simplify=286.95ms avg_simplify_ms=1.43 wall=883.27ms, shifted_quotient simplify=281.21ms avg_simplify_ms=2.81 wall=1.01s, product simplify=176.63ms avg_simplify_ms=1.77 wall=615.04ms, difference simplify=125.34ms avg_simplify_ms=2.51 wall=410.91ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.01s avg_case_ms=10.08 avg_simplify_ms=2.81, sum@0+100 failed=0 elapsed=646.07ms avg_case_ms=6.46 avg_simplify_ms=2.01, product@0+100 failed=0 elapsed=615.04ms avg_case_ms=6.15 avg_simplify_ms=1.77, difference@0+50 failed=0 elapsed=410.91ms avg_case_ms=8.22 avg_simplify_ms=2.51, sum@700+100 failed=0 elapsed=237.19ms avg_case_ms=2.37 avg_simplify_ms=0.86
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.05ms median_wire=17.12ms median_wall=65.68ms, difference@0+50 #174 difference runs=3 median_simplify=15.47ms median_wire=15.53ms median_wall=58.92ms, product@0+100 #175 product runs=3 median_simplify=16.01ms median_wire=16.07ms median_wall=60.82ms, sum@0+100 #173 sum runs=3 median_simplify=15.45ms median_wire=15.51ms median_wall=59.94ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.78ms median_wire=12.86ms median_wall=49.54ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.92s | passed=450 failed=0 total=450 avg_case=6.489ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.09s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.90s | passed=1 failed=0 |
