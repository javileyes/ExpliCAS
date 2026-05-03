# Engine Improvement Scorecard

- Generated: 2026-05-03T20:47:12.158849+00:00
- Git branch: main
- Git commit: `cd6bef5718153878ccde7ae9eaf8d41b392f3ce4`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=161.01ms avg_case_ms=1.61 simplify=15.66ms avg_simplify_ms=0.16, sum total=200 failed=0 elapsed=154.01ms avg_case_ms=0.77 simplify=25.62ms avg_simplify_ms=0.13, product total=100 failed=0 elapsed=108.81ms avg_case_ms=1.09 simplify=11.99ms avg_simplify_ms=0.12, difference total=50 failed=0 elapsed=71.51ms avg_case_ms=1.43 simplify=6.15ms avg_simplify_ms=0.12
- Engine hotspots: sum simplify=25.62ms avg_simplify_ms=0.13 wall=154.01ms, shifted_quotient simplify=15.66ms avg_simplify_ms=0.16 wall=161.01ms, product simplify=11.99ms avg_simplify_ms=0.12 wall=108.81ms, difference simplify=6.15ms avg_simplify_ms=0.12 wall=71.51ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=161.01ms avg_case_ms=1.61 avg_simplify_ms=0.16, sum@0+100 failed=0 elapsed=109.87ms avg_case_ms=1.10 avg_simplify_ms=0.14, product@0+100 failed=0 elapsed=108.81ms avg_case_ms=1.09 avg_simplify_ms=0.12, difference@0+50 failed=0 elapsed=71.51ms avg_case_ms=1.43 avg_simplify_ms=0.12, sum@700+100 failed=0 elapsed=44.14ms avg_case_ms=0.44 avg_simplify_ms=0.11
- Steady-state engine reruns: shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=0.80ms median_wire=0.89ms median_wall=1.75ms, shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=0.72ms median_wire=0.79ms median_wall=1.92ms, sum@0+100 #25 sum runs=3 median_simplify=0.13ms median_wire=0.15ms median_wall=0.79ms, shifted_quotient@0+100 #296 shifted_quotient runs=3 median_simplify=0.27ms median_wire=0.33ms median_wall=1.04ms, shifted_quotient@0+100 #140 shifted_quotient runs=3 median_simplify=0.23ms median_wire=0.27ms median_wall=1.39ms
- Steady-state dominant expressions: shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), sum@0+100 #25 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (cosh(x) + sinh(x) - e^x)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 0.50s | passed=450 failed=0 total=450 avg_case=1.102ms |
