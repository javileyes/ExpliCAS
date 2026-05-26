# Engine Improvement Scorecard

- Generated: 2026-05-26T07:08:27.793656+00:00
- Git branch: main
- Git commit: `2a9289ec6f1d0371567a03ac839ab158f97c4943`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=7
- By area: calculus / integration:2, calculus / limit:2, calculus / differentiation:1, calculus / post-calculus residual verification:1, orchestrator / exact-zero additive composition:1
- Recent 1: `calculus / post-calculus residual verification` - 2026-05-26 - Discovery observe-only: shifted sqrt reciprocal-trig residual verification times out
- Recent 2: `calculus / integration` - 2026-05-26 - Observe-only discovery: shifted sqrt-chain reciprocal trig external scale enters fragile simplification route
- Recent 3: `calculus / integration` - 2026-05-26 - Observe-only discovery: reciprocal trig derivative products do not yet share symbolic external-scale handling

## Calculus Support Matrix Signal

- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.
- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.
- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.
- `diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=261
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=340

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=644.57ms avg_case_ms=6.45 simplify=37.31ms avg_simplify_ms=0.37, sum total=200 failed=0 elapsed=537.85ms avg_case_ms=2.69 simplify=83.68ms avg_simplify_ms=0.42, product total=100 failed=0 elapsed=366.70ms avg_case_ms=3.67 simplify=25.89ms avg_simplify_ms=0.26, difference total=50 failed=0 elapsed=252.40ms avg_case_ms=5.05 simplify=31.13ms avg_simplify_ms=0.62
- Engine hotspots: sum simplify=83.68ms avg_simplify_ms=0.42 wall=537.85ms, shifted_quotient simplify=37.31ms avg_simplify_ms=0.37 wall=644.57ms, difference simplify=31.13ms avg_simplify_ms=0.62 wall=252.40ms, product simplify=25.89ms avg_simplify_ms=0.26 wall=366.70ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=644.57ms avg_case_ms=6.45 avg_simplify_ms=0.37, sum@0+100 failed=0 elapsed=385.40ms avg_case_ms=3.85 avg_simplify_ms=0.49, product@0+100 failed=0 elapsed=366.70ms avg_case_ms=3.67 avg_simplify_ms=0.26, difference@0+50 failed=0 elapsed=252.40ms avg_case_ms=5.05 avg_simplify_ms=0.62, sum@700+100 failed=0 elapsed=152.46ms avg_case_ms=1.52 avg_simplify_ms=0.34
- Steady-state engine reruns: sum@700+100 #2865 sum runs=3 median_simplify=5.93ms median_wire=5.97ms median_wall=6.43ms, sum@0+100 #53 sum runs=3 median_simplify=4.09ms median_wire=4.16ms median_wall=5.78ms, sum@700+100 #3021 sum runs=3 median_simplify=3.90ms median_wire=3.95ms median_wall=4.27ms, difference@0+50 #54 difference runs=3 median_simplify=4.19ms median_wire=4.24ms median_wall=5.69ms, sum@0+100 #25 sum runs=3 median_simplify=2.62ms median_wire=2.65ms median_wall=4.07ms
- Steady-state dominant expressions: sum@700+100 #2865 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #53 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@700+100 #3021 sum expr=(2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y))) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.80s | passed=450 failed=0 total=450 avg_case=4.000ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.69s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 18.35s | passed=1 failed=0 |
