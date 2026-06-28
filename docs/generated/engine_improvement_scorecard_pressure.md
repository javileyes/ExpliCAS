# Engine Improvement Scorecard

- Generated: 2026-06-28T13:24:39.653105+00:00
- Git branch: main
- Git commit: `42b9ea34c11b3b614fa39f0988485d09a07f7650`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=795.81ms avg_case_ms=7.96 simplify=228.40ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=717.60ms avg_case_ms=3.59 simplify=248.00ms avg_simplify_ms=1.24, product total=100 failed=0 elapsed=485.61ms avg_case_ms=4.86 simplify=142.74ms avg_simplify_ms=1.43, difference total=50 failed=0 elapsed=333.45ms avg_case_ms=6.67 simplify=107.81ms avg_simplify_ms=2.16
- Engine hotspots: sum simplify=248.00ms avg_simplify_ms=1.24 wall=717.60ms, shifted_quotient simplify=228.40ms avg_simplify_ms=2.28 wall=795.81ms, product simplify=142.74ms avg_simplify_ms=1.43 wall=485.61ms, difference simplify=107.81ms avg_simplify_ms=2.16 wall=333.45ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=795.81ms avg_case_ms=7.96 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=516.89ms avg_case_ms=5.17 avg_simplify_ms=1.71, product@0+100 failed=0 elapsed=485.61ms avg_case_ms=4.86 avg_simplify_ms=1.43, difference@0+50 failed=0 elapsed=333.45ms avg_case_ms=6.67 avg_simplify_ms=2.16, sum@700+100 failed=0 elapsed=200.71ms avg_case_ms=2.01 avg_simplify_ms=0.77
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.06ms median_wire=13.13ms median_wall=49.46ms, difference@0+50 #174 difference runs=3 median_simplify=11.78ms median_wire=11.83ms median_wall=44.57ms, sum@0+100 #173 sum runs=3 median_simplify=11.69ms median_wire=11.74ms median_wall=44.38ms, product@0+100 #175 product runs=3 median_simplify=11.63ms median_wire=11.68ms median_wall=44.58ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.82ms median_wire=10.89ms median_wall=41.14ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.08s | passed=1 failed=0 |
