# Engine Improvement Scorecard

- Generated: 2026-06-21T15:18:25.176918+00:00
- Git branch: main
- Git commit: `eda8e123efe365d4e2d1b470d159379a7781dedc`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=813.49ms avg_case_ms=8.13 simplify=233.78ms avg_simplify_ms=2.34, sum total=200 failed=0 elapsed=701.57ms avg_case_ms=3.51 simplify=238.31ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=483.74ms avg_case_ms=4.84 simplify=139.06ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=330.11ms avg_case_ms=6.60 simplify=105.56ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=238.31ms avg_simplify_ms=1.19 wall=701.57ms, shifted_quotient simplify=233.78ms avg_simplify_ms=2.34 wall=813.49ms, product simplify=139.06ms avg_simplify_ms=1.39 wall=483.74ms, difference simplify=105.56ms avg_simplify_ms=2.11 wall=330.11ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=813.49ms avg_case_ms=8.13 avg_simplify_ms=2.34, sum@0+100 failed=0 elapsed=507.87ms avg_case_ms=5.08 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=483.74ms avg_case_ms=4.84 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=330.11ms avg_case_ms=6.60 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=193.70ms avg_case_ms=1.94 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.86ms median_wire=13.93ms median_wall=52.24ms, product@0+100 #175 product runs=3 median_simplify=12.77ms median_wire=12.82ms median_wall=48.90ms, sum@0+100 #173 sum runs=3 median_simplify=11.95ms median_wire=12.00ms median_wall=45.23ms, difference@0+50 #174 difference runs=3 median_simplify=11.82ms median_wire=11.87ms median_wall=44.72ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.90ms median_wire=10.98ms median_wall=41.05ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
