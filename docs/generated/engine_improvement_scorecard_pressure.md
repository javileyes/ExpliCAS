# Engine Improvement Scorecard

- Generated: 2026-06-27T08:25:28.461623+00:00
- Git branch: main
- Git commit: `caef06633d6a96b65a5dc347444bf607d1301845`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=789.24ms avg_case_ms=7.89 simplify=224.81ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=706.46ms avg_case_ms=3.53 simplify=243.48ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=482.95ms avg_case_ms=4.83 simplify=141.78ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=330.22ms avg_case_ms=6.60 simplify=107.06ms avg_simplify_ms=2.14
- Engine hotspots: sum simplify=243.48ms avg_simplify_ms=1.22 wall=706.46ms, shifted_quotient simplify=224.81ms avg_simplify_ms=2.25 wall=789.24ms, product simplify=141.78ms avg_simplify_ms=1.42 wall=482.95ms, difference simplify=107.06ms avg_simplify_ms=2.14 wall=330.22ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=789.24ms avg_case_ms=7.89 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=510.42ms avg_case_ms=5.10 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=482.95ms avg_case_ms=4.83 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=330.22ms avg_case_ms=6.60 avg_simplify_ms=2.14, sum@700+100 failed=0 elapsed=196.03ms avg_case_ms=1.96 avg_simplify_ms=0.75
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.08ms median_wire=13.14ms median_wall=50.27ms, difference@0+50 #174 difference runs=3 median_simplify=11.86ms median_wire=11.92ms median_wall=44.97ms, product@0+100 #175 product runs=3 median_simplify=11.60ms median_wire=11.65ms median_wall=44.10ms, sum@0+100 #173 sum runs=3 median_simplify=11.71ms median_wire=11.76ms median_wall=44.49ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.60ms median_wire=10.68ms median_wall=40.15ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
