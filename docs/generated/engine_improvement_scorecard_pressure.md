# Engine Improvement Scorecard

- Generated: 2026-06-22T21:52:03.022723+00:00
- Git branch: main
- Git commit: `dd704a772790c1758972988cfb9a3e49bd8a84a0`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=791.37ms avg_case_ms=7.91 simplify=226.70ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=700.13ms avg_case_ms=3.50 simplify=237.04ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=487.65ms avg_case_ms=4.88 simplify=140.38ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=328.52ms avg_case_ms=6.57 simplify=104.72ms avg_simplify_ms=2.09
- Engine hotspots: sum simplify=237.04ms avg_simplify_ms=1.19 wall=700.13ms, shifted_quotient simplify=226.70ms avg_simplify_ms=2.27 wall=791.37ms, product simplify=140.38ms avg_simplify_ms=1.40 wall=487.65ms, difference simplify=104.72ms avg_simplify_ms=2.09 wall=328.52ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=791.37ms avg_case_ms=7.91 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=507.20ms avg_case_ms=5.07 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=487.65ms avg_case_ms=4.88 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=328.52ms avg_case_ms=6.57 avg_simplify_ms=2.09, sum@700+100 failed=0 elapsed=192.92ms avg_case_ms=1.93 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.92ms median_wire=12.99ms median_wall=49.25ms, sum@0+100 #173 sum runs=3 median_simplify=11.78ms median_wire=11.83ms median_wall=45.01ms, difference@0+50 #174 difference runs=3 median_simplify=11.87ms median_wire=11.91ms median_wall=44.72ms, product@0+100 #175 product runs=3 median_simplify=11.88ms median_wire=11.94ms median_wall=44.81ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.57ms median_wire=10.64ms median_wall=40.44ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.42s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
