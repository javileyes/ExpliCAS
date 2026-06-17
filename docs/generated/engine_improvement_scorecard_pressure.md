# Engine Improvement Scorecard

- Generated: 2026-06-17T19:25:07.040703+00:00
- Git branch: main
- Git commit: `ffed509de691012a994f1a054e970719b6118d47`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=803.60ms avg_case_ms=8.04 simplify=229.85ms avg_simplify_ms=2.30, sum total=200 failed=0 elapsed=703.94ms avg_case_ms=3.52 simplify=236.38ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=485.37ms avg_case_ms=4.85 simplify=142.10ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=333.67ms avg_case_ms=6.67 simplify=106.63ms avg_simplify_ms=2.13
- Engine hotspots: sum simplify=236.38ms avg_simplify_ms=1.18 wall=703.94ms, shifted_quotient simplify=229.85ms avg_simplify_ms=2.30 wall=803.60ms, product simplify=142.10ms avg_simplify_ms=1.42 wall=485.37ms, difference simplify=106.63ms avg_simplify_ms=2.13 wall=333.67ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=803.60ms avg_case_ms=8.04 avg_simplify_ms=2.30, sum@0+100 failed=0 elapsed=506.87ms avg_case_ms=5.07 avg_simplify_ms=1.63, product@0+100 failed=0 elapsed=485.37ms avg_case_ms=4.85 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=333.67ms avg_case_ms=6.67 avg_simplify_ms=2.13, sum@700+100 failed=0 elapsed=197.07ms avg_case_ms=1.97 avg_simplify_ms=0.73
- Steady-state engine reruns: product@0+100 #175 product runs=3 median_simplify=11.97ms median_wire=12.03ms median_wall=45.20ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.30ms median_wire=13.37ms median_wall=51.11ms, difference@0+50 #174 difference runs=3 median_simplify=12.06ms median_wire=12.11ms median_wall=45.47ms, sum@0+100 #173 sum runs=3 median_simplify=11.87ms median_wire=11.93ms median_wall=45.38ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.01ms median_wire=11.09ms median_wall=41.63ms
- Steady-state dominant expressions: product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
