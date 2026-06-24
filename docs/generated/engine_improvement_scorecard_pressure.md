# Engine Improvement Scorecard

- Generated: 2026-06-24T09:59:12.494298+00:00
- Git branch: main
- Git commit: `9a477d450c8c07197d92333774a002e6f0252515`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=837.87ms avg_case_ms=8.38 simplify=240.86ms avg_simplify_ms=2.41, sum total=200 failed=0 elapsed=701.78ms avg_case_ms=3.51 simplify=237.87ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=478.44ms avg_case_ms=4.78 simplify=137.45ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=331.30ms avg_case_ms=6.63 simplify=106.02ms avg_simplify_ms=2.12
- Engine hotspots: shifted_quotient simplify=240.86ms avg_simplify_ms=2.41 wall=837.87ms, sum simplify=237.87ms avg_simplify_ms=1.19 wall=701.78ms, product simplify=137.45ms avg_simplify_ms=1.37 wall=478.44ms, difference simplify=106.02ms avg_simplify_ms=2.12 wall=331.30ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=837.87ms avg_case_ms=8.38 avg_simplify_ms=2.41, sum@0+100 failed=0 elapsed=507.79ms avg_case_ms=5.08 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=478.44ms avg_case_ms=4.78 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=331.30ms avg_case_ms=6.63 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=194.00ms avg_case_ms=1.94 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.41ms median_wire=13.49ms median_wall=50.89ms, difference@0+50 #174 difference runs=3 median_simplify=12.06ms median_wire=12.11ms median_wall=45.48ms, sum@0+100 #173 sum runs=3 median_simplify=11.97ms median_wire=12.03ms median_wall=45.90ms, product@0+100 #175 product runs=3 median_simplify=11.75ms median_wire=11.81ms median_wall=45.27ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.85ms median_wire=10.93ms median_wall=40.78ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.35s | passed=450 failed=0 total=450 avg_case=5.222ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.51s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.07s | passed=1 failed=0 |
