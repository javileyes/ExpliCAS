# Engine Improvement Scorecard

- Generated: 2026-07-30T14:10:25.386781+00:00
- Git branch: main
- Git commit: `371434a28a9842e386bfa498270bcccf3abc37bd`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=380

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=994.84ms avg_case_ms=9.95 simplify=280.27ms avg_simplify_ms=2.80, sum total=200 failed=0 elapsed=861.71ms avg_case_ms=4.31 simplify=278.20ms avg_simplify_ms=1.39, product total=100 failed=0 elapsed=616.10ms avg_case_ms=6.16 simplify=178.87ms avg_simplify_ms=1.79, difference total=50 failed=0 elapsed=404.05ms avg_case_ms=8.08 simplify=118.46ms avg_simplify_ms=2.37
- Engine hotspots: shifted_quotient simplify=280.27ms avg_simplify_ms=2.80 wall=994.84ms, sum simplify=278.20ms avg_simplify_ms=1.39 wall=861.71ms, product simplify=178.87ms avg_simplify_ms=1.79 wall=616.10ms, difference simplify=118.46ms avg_simplify_ms=2.37 wall=404.05ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=994.84ms avg_case_ms=9.95 avg_simplify_ms=2.80, sum@0+100 failed=0 elapsed=625.39ms avg_case_ms=6.25 avg_simplify_ms=1.93, product@0+100 failed=0 elapsed=616.10ms avg_case_ms=6.16 avg_simplify_ms=1.79, difference@0+50 failed=0 elapsed=404.05ms avg_case_ms=8.08 avg_simplify_ms=2.37, sum@700+100 failed=0 elapsed=236.32ms avg_case_ms=2.36 avg_simplify_ms=0.85
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.61ms median_wire=16.69ms median_wall=64.90ms, difference@0+50 #174 difference runs=3 median_simplify=15.32ms median_wire=15.37ms median_wall=58.07ms, product@0+100 #175 product runs=3 median_simplify=15.22ms median_wire=15.28ms median_wall=58.53ms, sum@0+100 #173 sum runs=3 median_simplify=15.22ms median_wire=15.28ms median_wall=58.62ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.95ms median_wire=13.03ms median_wall=49.96ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.88s | passed=450 failed=0 total=450 avg_case=6.400ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.39s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.76s | passed=1 failed=0 |
