# Engine Improvement Scorecard

- Generated: 2026-06-21T07:15:36.926533+00:00
- Git branch: main
- Git commit: `cbeae14793a3504fe59c71cb87c37d6a66a07040`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=804.21ms avg_case_ms=8.04 simplify=230.08ms avg_simplify_ms=2.30, sum total=200 failed=0 elapsed=703.18ms avg_case_ms=3.52 simplify=237.56ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=492.02ms avg_case_ms=4.92 simplify=142.31ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=331.27ms avg_case_ms=6.63 simplify=105.32ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=237.56ms avg_simplify_ms=1.19 wall=703.18ms, shifted_quotient simplify=230.08ms avg_simplify_ms=2.30 wall=804.21ms, product simplify=142.31ms avg_simplify_ms=1.42 wall=492.02ms, difference simplify=105.32ms avg_simplify_ms=2.11 wall=331.27ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=804.21ms avg_case_ms=8.04 avg_simplify_ms=2.30, sum@0+100 failed=0 elapsed=508.48ms avg_case_ms=5.08 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=492.02ms avg_case_ms=4.92 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=331.27ms avg_case_ms=6.63 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=194.69ms avg_case_ms=1.95 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.00ms median_wire=13.07ms median_wall=49.57ms, difference@0+50 #174 difference runs=3 median_simplify=11.71ms median_wire=11.76ms median_wall=44.16ms, sum@0+100 #173 sum runs=3 median_simplify=11.68ms median_wire=11.73ms median_wall=44.30ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.74ms median_wire=10.81ms median_wall=40.64ms, product@0+100 #175 product runs=3 median_simplify=11.63ms median_wire=11.68ms median_wall=44.88ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.41s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
