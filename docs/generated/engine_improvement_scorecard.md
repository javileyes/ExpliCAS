# Engine Improvement Scorecard

- Generated: 2026-05-07T11:15:32.179678+00:00
- Git branch: main
- Git commit: `bde4833271a734db9e0eafb6a19dddb840c3634e`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=174.50ms avg_case_ms=1.74 simplify=15.72ms avg_simplify_ms=0.16, sum total=200 failed=0 elapsed=173.11ms avg_case_ms=0.87 simplify=25.94ms avg_simplify_ms=0.13, product total=100 failed=0 elapsed=122.54ms avg_case_ms=1.23 simplify=12.86ms avg_simplify_ms=0.13, difference total=50 failed=0 elapsed=77.40ms avg_case_ms=1.55 simplify=6.43ms avg_simplify_ms=0.13
- Engine hotspots: sum simplify=25.94ms avg_simplify_ms=0.13 wall=173.11ms, shifted_quotient simplify=15.72ms avg_simplify_ms=0.16 wall=174.50ms, product simplify=12.86ms avg_simplify_ms=0.13 wall=122.54ms, difference simplify=6.43ms avg_simplify_ms=0.13 wall=77.40ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=174.50ms avg_case_ms=1.74 avg_simplify_ms=0.16, product@0+100 failed=0 elapsed=122.54ms avg_case_ms=1.23 avg_simplify_ms=0.13, sum@0+100 failed=0 elapsed=121.66ms avg_case_ms=1.22 avg_simplify_ms=0.15, difference@0+50 failed=0 elapsed=77.40ms avg_case_ms=1.55 avg_simplify_ms=0.13, sum@700+100 failed=0 elapsed=51.45ms avg_case_ms=0.51 avg_simplify_ms=0.11
- Steady-state engine reruns: shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=0.80ms median_wire=0.89ms median_wall=2.56ms, shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=0.76ms median_wire=0.83ms median_wall=2.76ms, sum@0+100 #25 sum runs=3 median_simplify=0.14ms median_wire=0.17ms median_wall=0.86ms, shifted_quotient@0+100 #296 shifted_quotient runs=3 median_simplify=0.29ms median_wire=0.36ms median_wall=1.39ms, shifted_quotient@0+100 #140 shifted_quotient runs=3 median_simplify=0.25ms median_wire=0.30ms median_wall=1.61ms
- Steady-state dominant expressions: shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), sum@0+100 #25 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (cosh(x) + sinh(x) - e^x)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 0.55s | passed=450 failed=0 total=450 avg_case=1.218ms |
