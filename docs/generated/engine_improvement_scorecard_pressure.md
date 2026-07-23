# Engine Improvement Scorecard

- Generated: 2026-07-23T12:18:25.585643+00:00
- Git branch: main
- Git commit: `2b0b3dc6e4482ec44776c3db867342bf4240ae3e`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.02s avg_case_ms=10.17 simplify=286.94ms avg_simplify_ms=2.87, sum total=200 failed=0 elapsed=899.49ms avg_case_ms=4.50 simplify=289.19ms avg_simplify_ms=1.45, product total=100 failed=0 elapsed=610.17ms avg_case_ms=6.10 simplify=177.07ms avg_simplify_ms=1.77, difference total=50 failed=0 elapsed=404.04ms avg_case_ms=8.08 simplify=123.47ms avg_simplify_ms=2.47
- Engine hotspots: sum simplify=289.19ms avg_simplify_ms=1.45 wall=899.49ms, shifted_quotient simplify=286.94ms avg_simplify_ms=2.87 wall=1.02s, product simplify=177.07ms avg_simplify_ms=1.77 wall=610.17ms, difference simplify=123.47ms avg_simplify_ms=2.47 wall=404.04ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.02s avg_case_ms=10.17 avg_simplify_ms=2.87, sum@0+100 failed=0 elapsed=667.15ms avg_case_ms=6.67 avg_simplify_ms=2.05, product@0+100 failed=0 elapsed=610.17ms avg_case_ms=6.10 avg_simplify_ms=1.77, difference@0+50 failed=0 elapsed=404.04ms avg_case_ms=8.08 avg_simplify_ms=2.47, sum@700+100 failed=0 elapsed=232.34ms avg_case_ms=2.32 avg_simplify_ms=0.85
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.78ms median_wire=16.86ms median_wall=64.74ms, sum@0+100 #173 sum runs=3 median_simplify=15.27ms median_wire=15.32ms median_wall=58.10ms, product@0+100 #175 product runs=3 median_simplify=15.56ms median_wire=15.61ms median_wall=58.47ms, difference@0+50 #174 difference runs=3 median_simplify=15.94ms median_wire=15.99ms median_wall=59.63ms, shifted_quotient@0+100 #112 shifted_quotient runs=3 median_simplify=9.88ms median_wire=9.94ms median_wall=37.25ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.93s | passed=450 failed=0 total=450 avg_case=6.511ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.74s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.95s | passed=1 failed=0 |
