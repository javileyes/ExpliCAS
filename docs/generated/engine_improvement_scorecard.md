# Engine Improvement Scorecard

- Generated: 2026-05-08T13:09:58.074152+00:00
- Git branch: main
- Git commit: `a1ac8edc8c355044f24674540ac662e054f5eba9`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=181.58ms avg_case_ms=1.82 simplify=19.47ms avg_simplify_ms=0.19, sum total=200 failed=0 elapsed=174.71ms avg_case_ms=0.87 simplify=26.67ms avg_simplify_ms=0.13, product total=100 failed=0 elapsed=120.96ms avg_case_ms=1.21 simplify=14.62ms avg_simplify_ms=0.15, difference total=50 failed=0 elapsed=75.86ms avg_case_ms=1.52 simplify=6.29ms avg_simplify_ms=0.13
- Engine hotspots: sum simplify=26.67ms avg_simplify_ms=0.13 wall=174.71ms, shifted_quotient simplify=19.47ms avg_simplify_ms=0.19 wall=181.58ms, product simplify=14.62ms avg_simplify_ms=0.15 wall=120.96ms, difference simplify=6.29ms avg_simplify_ms=0.13 wall=75.86ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=181.58ms avg_case_ms=1.82 avg_simplify_ms=0.19, sum@0+100 failed=0 elapsed=124.38ms avg_case_ms=1.24 avg_simplify_ms=0.16, product@0+100 failed=0 elapsed=120.96ms avg_case_ms=1.21 avg_simplify_ms=0.15, difference@0+50 failed=0 elapsed=75.86ms avg_case_ms=1.52 avg_simplify_ms=0.13, sum@700+100 failed=0 elapsed=50.33ms avg_case_ms=0.50 avg_simplify_ms=0.11
- Steady-state engine reruns: product@0+100 #319 product runs=3 median_simplify=0.14ms median_wire=0.17ms median_wall=0.91ms, shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=1.37ms median_wire=1.45ms median_wall=3.02ms, shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=1.28ms median_wire=1.35ms median_wall=3.12ms, product@0+100 #355 product runs=3 median_simplify=0.08ms median_wire=0.10ms median_wall=0.12ms, shifted_quotient@0+100 #92 shifted_quotient runs=3 median_simplify=0.11ms median_wire=0.15ms median_wall=1.39ms
- Steady-state dominant expressions: product@0+100 #319 product expr=(tan(x) + cot(x) - sec(x)*csc(x)) * (1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)), shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 0.55s | passed=450 failed=0 total=450 avg_case=1.230ms |
