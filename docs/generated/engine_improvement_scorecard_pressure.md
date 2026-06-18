# Engine Improvement Scorecard

- Generated: 2026-06-18T16:17:11.668243+00:00
- Git branch: main
- Git commit: `fe209eda205bba84418bdcf7fc35754e688131f9`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=799.66ms avg_case_ms=8.00 simplify=228.89ms avg_simplify_ms=2.29, sum total=200 failed=0 elapsed=705.76ms avg_case_ms=3.53 simplify=239.88ms avg_simplify_ms=1.20, product total=100 failed=0 elapsed=486.01ms avg_case_ms=4.86 simplify=140.01ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=348.43ms avg_case_ms=6.97 simplify=112.69ms avg_simplify_ms=2.25
- Engine hotspots: sum simplify=239.88ms avg_simplify_ms=1.20 wall=705.76ms, shifted_quotient simplify=228.89ms avg_simplify_ms=2.29 wall=799.66ms, product simplify=140.01ms avg_simplify_ms=1.40 wall=486.01ms, difference simplify=112.69ms avg_simplify_ms=2.25 wall=348.43ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=799.66ms avg_case_ms=8.00 avg_simplify_ms=2.29, sum@0+100 failed=0 elapsed=511.68ms avg_case_ms=5.12 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=486.01ms avg_case_ms=4.86 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=348.43ms avg_case_ms=6.97 avg_simplify_ms=2.25, sum@700+100 failed=0 elapsed=194.07ms avg_case_ms=1.94 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.33ms median_wire=13.41ms median_wall=50.35ms, difference@0+50 #174 difference runs=3 median_simplify=11.96ms median_wire=12.01ms median_wall=45.12ms, sum@0+100 #173 sum runs=3 median_simplify=11.82ms median_wire=11.88ms median_wall=44.72ms, product@0+100 #175 product runs=3 median_simplify=11.77ms median_wire=11.82ms median_wall=44.85ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.88ms median_wire=10.95ms median_wall=41.22ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.35s | passed=450 failed=0 total=450 avg_case=5.222ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.60s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.07s | passed=1 failed=0 |
