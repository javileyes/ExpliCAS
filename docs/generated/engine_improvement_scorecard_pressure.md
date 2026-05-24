# Engine Improvement Scorecard

- Generated: 2026-05-24T16:32:45.187632+00:00
- Git branch: main
- Git commit: `58df39ec2e0b26038f155f6d4da6c0fbac215cae`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=1
- By area: orchestrator / exact-zero additive composition:1
- Recent 1: `orchestrator / exact-zero additive composition` - 2026-05-24 - Discovery observe-only: hyperbolic angle-sum plus telescoping residual remains slow

## Calculus Contract Signal

- Dimension: public calculus behavior, result simplification, domain conditions, and step noise.
- Interpretation: small executable calculus vertical slices; failures should be classified before broadening pre-calculus rules.
- `diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=257
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=337

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=844.33ms avg_case_ms=8.44 simplify=59.49ms avg_simplify_ms=0.59, sum total=200 failed=0 elapsed=719.85ms avg_case_ms=3.60 simplify=130.92ms avg_simplify_ms=0.65, product total=100 failed=0 elapsed=475.01ms avg_case_ms=4.75 simplify=37.78ms avg_simplify_ms=0.38, difference total=50 failed=0 elapsed=331.59ms avg_case_ms=6.63 simplify=46.28ms avg_simplify_ms=0.93
- Engine hotspots: sum simplify=130.92ms avg_simplify_ms=0.65 wall=719.85ms, shifted_quotient simplify=59.49ms avg_simplify_ms=0.59 wall=844.33ms, difference simplify=46.28ms avg_simplify_ms=0.93 wall=331.59ms, product simplify=37.78ms avg_simplify_ms=0.38 wall=475.01ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=844.33ms avg_case_ms=8.44 avg_simplify_ms=0.59, sum@0+100 failed=0 elapsed=510.92ms avg_case_ms=5.11 avg_simplify_ms=0.79, product@0+100 failed=0 elapsed=475.01ms avg_case_ms=4.75 avg_simplify_ms=0.38, difference@0+50 failed=0 elapsed=331.59ms avg_case_ms=6.63 avg_simplify_ms=0.93, sum@700+100 failed=0 elapsed=208.93ms avg_case_ms=2.09 avg_simplify_ms=0.52
- Steady-state engine reruns: sum@700+100 #2865 sum runs=3 median_simplify=15.08ms median_wire=15.20ms median_wall=15.65ms, sum@0+100 #25 sum runs=3 median_simplify=3.47ms median_wire=3.51ms median_wall=5.16ms, sum@0+100 #209 sum runs=3 median_simplify=4.55ms median_wire=4.63ms median_wall=8.51ms, difference@0+50 #54 difference runs=3 median_simplify=4.72ms median_wire=4.79ms median_wall=7.74ms, sum@700+100 #3021 sum runs=3 median_simplify=6.11ms median_wire=6.16ms median_wall=6.56ms
- Steady-state dominant expressions: sum@700+100 #2865 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #25 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (cosh(x) + sinh(x) - e^x), sum@0+100 #209 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.37s | passed=450 failed=0 total=450 avg_case=5.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 24.88s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 80.16s | passed=1 failed=0 |
