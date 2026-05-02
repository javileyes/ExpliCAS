# Engine Improvement Scorecard

- Generated: 2026-05-02T09:13:08.077662+00:00
- Git branch: main
- Git commit: `9bbc87c97f6c018a1957b8f44e8cf87c9bfc135b`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=161.69ms avg_case_ms=1.62 simplify=15.02ms avg_simplify_ms=0.15, sum total=200 failed=0 elapsed=157.21ms avg_case_ms=0.79 simplify=25.44ms avg_simplify_ms=0.13, product total=100 failed=0 elapsed=109.02ms avg_case_ms=1.09 simplify=11.78ms avg_simplify_ms=0.12, difference total=50 failed=0 elapsed=72.67ms avg_case_ms=1.45 simplify=6.18ms avg_simplify_ms=0.12
- Engine hotspots: sum simplify=25.44ms avg_simplify_ms=0.13 wall=157.21ms, shifted_quotient simplify=15.02ms avg_simplify_ms=0.15 wall=161.69ms, product simplify=11.78ms avg_simplify_ms=0.12 wall=109.02ms, difference simplify=6.18ms avg_simplify_ms=0.12 wall=72.67ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=161.69ms avg_case_ms=1.62 avg_simplify_ms=0.15, sum@0+100 failed=0 elapsed=112.45ms avg_case_ms=1.12 avg_simplify_ms=0.14, product@0+100 failed=0 elapsed=109.02ms avg_case_ms=1.09 avg_simplify_ms=0.12, difference@0+50 failed=0 elapsed=72.67ms avg_case_ms=1.45 avg_simplify_ms=0.12, sum@700+100 failed=0 elapsed=44.76ms avg_case_ms=0.45 avg_simplify_ms=0.11
- Steady-state engine reruns: shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=0.79ms median_wire=0.87ms median_wall=1.70ms, shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=0.73ms median_wire=0.80ms median_wall=1.87ms, sum@0+100 #25 sum runs=3 median_simplify=0.12ms median_wire=0.15ms median_wall=0.73ms, shifted_quotient@0+100 #296 shifted_quotient runs=3 median_simplify=0.30ms median_wire=0.36ms median_wall=1.13ms, sum@0+100 #337 sum runs=3 median_simplify=0.15ms median_wire=0.17ms median_wall=0.19ms
- Steady-state dominant expressions: shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), sum@0+100 #25 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (cosh(x) + sinh(x) - e^x)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 0.50s | passed=450 failed=0 total=450 avg_case=1.113ms |
