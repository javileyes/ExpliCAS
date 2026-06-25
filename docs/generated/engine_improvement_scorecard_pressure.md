# Engine Improvement Scorecard

- Generated: 2026-06-25T23:11:07.473343+00:00
- Git branch: main
- Git commit: `0fde935caf0c86edf627e80f2e932aedd8de462a`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=805.81ms avg_case_ms=8.06 simplify=231.02ms avg_simplify_ms=2.31, sum total=200 failed=0 elapsed=705.49ms avg_case_ms=3.53 simplify=242.66ms avg_simplify_ms=1.21, product total=100 failed=0 elapsed=481.82ms avg_case_ms=4.82 simplify=141.11ms avg_simplify_ms=1.41, difference total=50 failed=0 elapsed=328.53ms avg_case_ms=6.57 simplify=105.76ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=242.66ms avg_simplify_ms=1.21 wall=705.49ms, shifted_quotient simplify=231.02ms avg_simplify_ms=2.31 wall=805.81ms, product simplify=141.11ms avg_simplify_ms=1.41 wall=481.82ms, difference simplify=105.76ms avg_simplify_ms=2.12 wall=328.53ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=805.81ms avg_case_ms=8.06 avg_simplify_ms=2.31, sum@0+100 failed=0 elapsed=508.69ms avg_case_ms=5.09 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=481.82ms avg_case_ms=4.82 avg_simplify_ms=1.41, difference@0+50 failed=0 elapsed=328.53ms avg_case_ms=6.57 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=196.80ms avg_case_ms=1.97 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.19ms median_wire=13.27ms median_wall=50.14ms, difference@0+50 #174 difference runs=3 median_simplify=11.97ms median_wire=12.02ms median_wall=45.00ms, sum@0+100 #173 sum runs=3 median_simplify=11.78ms median_wire=11.83ms median_wall=44.62ms, product@0+100 #175 product runs=3 median_simplify=11.60ms median_wire=11.65ms median_wall=44.04ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.67ms median_wire=10.75ms median_wall=40.39ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
