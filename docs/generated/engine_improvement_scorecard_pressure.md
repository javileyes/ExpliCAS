# Engine Improvement Scorecard

- Generated: 2026-06-27T15:52:55.384310+00:00
- Git branch: main
- Git commit: `403b5d10176c66753111dd1b618db63d3bcccfb8`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=822.42ms avg_case_ms=8.22 simplify=236.61ms avg_simplify_ms=2.37, sum total=200 failed=0 elapsed=751.09ms avg_case_ms=3.76 simplify=262.65ms avg_simplify_ms=1.31, product total=100 failed=0 elapsed=501.97ms avg_case_ms=5.02 simplify=148.90ms avg_simplify_ms=1.49, difference total=50 failed=0 elapsed=341.37ms avg_case_ms=6.83 simplify=111.09ms avg_simplify_ms=2.22
- Engine hotspots: sum simplify=262.65ms avg_simplify_ms=1.31 wall=751.09ms, shifted_quotient simplify=236.61ms avg_simplify_ms=2.37 wall=822.42ms, product simplify=148.90ms avg_simplify_ms=1.49 wall=501.97ms, difference simplify=111.09ms avg_simplify_ms=2.22 wall=341.37ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=822.42ms avg_case_ms=8.22 avg_simplify_ms=2.37, sum@0+100 failed=0 elapsed=544.86ms avg_case_ms=5.45 avg_simplify_ms=1.82, product@0+100 failed=0 elapsed=501.97ms avg_case_ms=5.02 avg_simplify_ms=1.49, difference@0+50 failed=0 elapsed=341.37ms avg_case_ms=6.83 avg_simplify_ms=2.22, sum@700+100 failed=0 elapsed=206.22ms avg_case_ms=2.06 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.70ms median_wire=13.77ms median_wall=51.21ms, sum@0+100 #173 sum runs=3 median_simplify=11.96ms median_wire=12.02ms median_wall=45.01ms, product@0+100 #175 product runs=3 median_simplify=11.91ms median_wire=11.97ms median_wall=45.42ms, difference@0+50 #174 difference runs=3 median_simplify=11.96ms median_wire=12.01ms median_wall=45.12ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.73ms median_wire=10.81ms median_wall=40.93ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.42s | passed=450 failed=0 total=450 avg_case=5.378ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.52s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.07s | passed=1 failed=0 |
