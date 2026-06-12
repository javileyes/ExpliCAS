# Engine Improvement Scorecard

- Generated: 2026-06-12T22:27:21.431573+00:00
- Git branch: main
- Git commit: `af780ae82e5f141c1016da011db6c71caeecbd68`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=794.50ms avg_case_ms=7.94 simplify=224.54ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=690.92ms avg_case_ms=3.45 simplify=231.10ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=474.27ms avg_case_ms=4.74 simplify=134.70ms avg_simplify_ms=1.35, difference total=50 failed=0 elapsed=331.00ms avg_case_ms=6.62 simplify=105.37ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=231.10ms avg_simplify_ms=1.16 wall=690.92ms, shifted_quotient simplify=224.54ms avg_simplify_ms=2.25 wall=794.50ms, product simplify=134.70ms avg_simplify_ms=1.35 wall=474.27ms, difference simplify=105.37ms avg_simplify_ms=2.11 wall=331.00ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=794.50ms avg_case_ms=7.94 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=497.30ms avg_case_ms=4.97 avg_simplify_ms=1.59, product@0+100 failed=0 elapsed=474.27ms avg_case_ms=4.74 avg_simplify_ms=1.35, difference@0+50 failed=0 elapsed=331.00ms avg_case_ms=6.62 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=193.62ms avg_case_ms=1.94 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.88ms median_wire=12.95ms median_wall=49.74ms, difference@0+50 #174 difference runs=3 median_simplify=11.76ms median_wire=11.81ms median_wall=45.11ms, product@0+100 #175 product runs=3 median_simplify=11.77ms median_wire=11.82ms median_wall=44.79ms, sum@0+100 #173 sum runs=3 median_simplify=11.74ms median_wire=11.79ms median_wall=44.42ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.66ms median_wire=10.73ms median_wall=40.98ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.85s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.98s | passed=1 failed=0 |
