# Engine Improvement Scorecard

- Generated: 2026-07-17T01:02:28.462513+00:00
- Git branch: main
- Git commit: `ed3e8e61bbc644d7322625a24d086f3cff9b3dce`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=987.56ms avg_case_ms=9.88 simplify=273.27ms avg_simplify_ms=2.73, sum total=200 failed=0 elapsed=905.14ms avg_case_ms=4.53 simplify=288.37ms avg_simplify_ms=1.44, product total=100 failed=0 elapsed=605.56ms avg_case_ms=6.06 simplify=174.06ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=400.11ms avg_case_ms=8.00 simplify=121.71ms avg_simplify_ms=2.43
- Engine hotspots: sum simplify=288.37ms avg_simplify_ms=1.44 wall=905.14ms, shifted_quotient simplify=273.27ms avg_simplify_ms=2.73 wall=987.56ms, product simplify=174.06ms avg_simplify_ms=1.74 wall=605.56ms, difference simplify=121.71ms avg_simplify_ms=2.43 wall=400.11ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=987.56ms avg_case_ms=9.88 avg_simplify_ms=2.73, sum@0+100 failed=0 elapsed=671.96ms avg_case_ms=6.72 avg_simplify_ms=2.04, product@0+100 failed=0 elapsed=605.56ms avg_case_ms=6.06 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=400.11ms avg_case_ms=8.00 avg_simplify_ms=2.43, sum@700+100 failed=0 elapsed=233.18ms avg_case_ms=2.33 avg_simplify_ms=0.84
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.42ms median_wire=15.47ms median_wall=63.03ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.89ms median_wire=16.96ms median_wall=64.70ms, difference@0+50 #174 difference runs=3 median_simplify=15.23ms median_wire=15.28ms median_wall=58.34ms, product@0+100 #175 product runs=3 median_simplify=15.23ms median_wire=15.28ms median_wall=58.55ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.08ms median_wire=13.15ms median_wall=49.82ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.90s | passed=450 failed=0 total=450 avg_case=6.444ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.55s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
