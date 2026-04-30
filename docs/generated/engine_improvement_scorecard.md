# Engine Improvement Scorecard

- Generated: 2026-04-30T15:07:30.398098+00:00
- Git branch: main
- Git commit: `29692c0de295da6b016b1228389c8ccd70f14e9a`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: sum total=200 failed=0 elapsed=142.72ms avg_case_ms=0.71 simplify=23.09ms avg_simplify_ms=0.12, shifted_quotient total=100 failed=0 elapsed=141.10ms avg_case_ms=1.41 simplify=11.64ms avg_simplify_ms=0.12, product total=100 failed=0 elapsed=95.18ms avg_case_ms=0.95 simplify=9.43ms avg_simplify_ms=0.09, difference total=50 failed=0 elapsed=66.19ms avg_case_ms=1.32 simplify=5.08ms avg_simplify_ms=0.10
- Engine hotspots: sum simplify=23.09ms avg_simplify_ms=0.12 wall=142.72ms, shifted_quotient simplify=11.64ms avg_simplify_ms=0.12 wall=141.10ms, product simplify=9.43ms avg_simplify_ms=0.09 wall=95.18ms, difference simplify=5.08ms avg_simplify_ms=0.10 wall=66.19ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=141.10ms avg_case_ms=1.41 avg_simplify_ms=0.12, sum@0+100 failed=0 elapsed=104.22ms avg_case_ms=1.04 avg_simplify_ms=0.13, product@0+100 failed=0 elapsed=95.18ms avg_case_ms=0.95 avg_simplify_ms=0.09, difference@0+50 failed=0 elapsed=66.19ms avg_case_ms=1.32 avg_simplify_ms=0.10, sum@700+100 failed=0 elapsed=38.50ms avg_case_ms=0.38 avg_simplify_ms=0.10
- Steady-state engine reruns: shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=0.52ms median_wire=0.60ms median_wall=1.28ms, shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=0.47ms median_wire=0.53ms median_wall=1.54ms, sum@0+100 #25 sum runs=3 median_simplify=0.13ms median_wire=0.16ms median_wall=0.78ms, sum@0+100 #141 sum runs=3 median_simplify=0.12ms median_wire=0.14ms median_wall=0.69ms, sum@0+100 #9 sum runs=3 median_simplify=0.11ms median_wire=0.14ms median_wall=2.64ms
- Steady-state dominant expressions: shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), sum@0+100 #25 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (cosh(x) + sinh(x) - e^x)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 0.45s | passed=450 failed=0 total=450 avg_case=0.990ms |
