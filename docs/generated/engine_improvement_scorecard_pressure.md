# Engine Improvement Scorecard

- Generated: 2026-06-27T07:59:21.250803+00:00
- Git branch: main
- Git commit: `a3aa371a80e5fe3466dda0a4986c7a9c1b4b2aa1`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=797.28ms avg_case_ms=7.97 simplify=227.90ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=709.05ms avg_case_ms=3.55 simplify=244.82ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=489.72ms avg_case_ms=4.90 simplify=143.13ms avg_simplify_ms=1.43, difference total=50 failed=0 elapsed=334.36ms avg_case_ms=6.69 simplify=107.74ms avg_simplify_ms=2.15
- Engine hotspots: sum simplify=244.82ms avg_simplify_ms=1.22 wall=709.05ms, shifted_quotient simplify=227.90ms avg_simplify_ms=2.28 wall=797.28ms, product simplify=143.13ms avg_simplify_ms=1.43 wall=489.72ms, difference simplify=107.74ms avg_simplify_ms=2.15 wall=334.36ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=797.28ms avg_case_ms=7.97 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=511.36ms avg_case_ms=5.11 avg_simplify_ms=1.69, product@0+100 failed=0 elapsed=489.72ms avg_case_ms=4.90 avg_simplify_ms=1.43, difference@0+50 failed=0 elapsed=334.36ms avg_case_ms=6.69 avg_simplify_ms=2.15, sum@700+100 failed=0 elapsed=197.70ms avg_case_ms=1.98 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.03ms median_wire=13.10ms median_wall=50.30ms, product@0+100 #175 product runs=3 median_simplify=11.89ms median_wire=11.95ms median_wall=45.28ms, difference@0+50 #174 difference runs=3 median_simplify=12.00ms median_wire=12.05ms median_wall=45.67ms, sum@0+100 #173 sum runs=3 median_simplify=11.94ms median_wire=11.99ms median_wall=45.54ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.93ms median_wire=11.00ms median_wall=40.82ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.47s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
