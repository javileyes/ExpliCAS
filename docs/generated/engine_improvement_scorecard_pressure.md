# Engine Improvement Scorecard

- Generated: 2026-07-24T13:17:51.974069+00:00
- Git branch: main
- Git commit: `d1b722fa3606fb323566af986ed1d93e2866f8fe`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=367

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=977.28ms avg_case_ms=9.77 simplify=274.10ms avg_simplify_ms=2.74, sum total=200 failed=0 elapsed=889.95ms avg_case_ms=4.45 simplify=290.96ms avg_simplify_ms=1.45, product total=100 failed=0 elapsed=607.15ms avg_case_ms=6.07 simplify=174.67ms avg_simplify_ms=1.75, difference total=50 failed=0 elapsed=406.47ms avg_case_ms=8.13 simplify=125.68ms avg_simplify_ms=2.51
- Engine hotspots: sum simplify=290.96ms avg_simplify_ms=1.45 wall=889.95ms, shifted_quotient simplify=274.10ms avg_simplify_ms=2.74 wall=977.28ms, product simplify=174.67ms avg_simplify_ms=1.75 wall=607.15ms, difference simplify=125.68ms avg_simplify_ms=2.51 wall=406.47ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=977.28ms avg_case_ms=9.77 avg_simplify_ms=2.74, sum@0+100 failed=0 elapsed=654.59ms avg_case_ms=6.55 avg_simplify_ms=2.06, product@0+100 failed=0 elapsed=607.15ms avg_case_ms=6.07 avg_simplify_ms=1.75, difference@0+50 failed=0 elapsed=406.47ms avg_case_ms=8.13 avg_simplify_ms=2.51, sum@700+100 failed=0 elapsed=235.36ms avg_case_ms=2.35 avg_simplify_ms=0.85
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.91ms median_wire=16.98ms median_wall=65.13ms, difference@0+50 #174 difference runs=3 median_simplify=15.29ms median_wire=15.34ms median_wall=59.39ms, sum@0+100 #173 sum runs=3 median_simplify=15.34ms median_wire=15.39ms median_wall=59.07ms, product@0+100 #175 product runs=3 median_simplify=15.57ms median_wire=15.63ms median_wall=59.55ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.16ms median_wire=13.24ms median_wall=50.19ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.88s | passed=450 failed=0 total=450 avg_case=6.400ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.66s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
