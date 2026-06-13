# Engine Improvement Scorecard

- Generated: 2026-06-13T09:06:40.317435+00:00
- Git branch: main
- Git commit: `e91621e728492cc797f4646087e467b82859b450`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=779.11ms avg_case_ms=7.79 simplify=221.06ms avg_simplify_ms=2.21, sum total=200 failed=0 elapsed=692.54ms avg_case_ms=3.46 simplify=232.52ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=477.24ms avg_case_ms=4.77 simplify=136.56ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=324.17ms avg_case_ms=6.48 simplify=102.70ms avg_simplify_ms=2.05
- Engine hotspots: sum simplify=232.52ms avg_simplify_ms=1.16 wall=692.54ms, shifted_quotient simplify=221.06ms avg_simplify_ms=2.21 wall=779.11ms, product simplify=136.56ms avg_simplify_ms=1.37 wall=477.24ms, difference simplify=102.70ms avg_simplify_ms=2.05 wall=324.17ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=779.11ms avg_case_ms=7.79 avg_simplify_ms=2.21, sum@0+100 failed=0 elapsed=499.10ms avg_case_ms=4.99 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=477.24ms avg_case_ms=4.77 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=324.17ms avg_case_ms=6.48 avg_simplify_ms=2.05, sum@700+100 failed=0 elapsed=193.44ms avg_case_ms=1.93 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.95ms median_wire=13.03ms median_wall=48.92ms, product@0+100 #175 product runs=3 median_simplify=11.56ms median_wire=11.61ms median_wall=44.64ms, sum@0+100 #173 sum runs=3 median_simplify=11.64ms median_wire=11.69ms median_wall=44.06ms, difference@0+50 #174 difference runs=3 median_simplify=11.60ms median_wire=11.66ms median_wall=43.72ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.50ms median_wire=10.57ms median_wall=39.64ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.27s | passed=450 failed=0 total=450 avg_case=5.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.80s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
