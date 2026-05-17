# Engine Improvement Scorecard

- Generated: 2026-05-17T17:41:13.845451+00:00
- Git branch: main
- Git commit: `796a81168cabad7a0d751b60454c028631a7645b`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=486.76ms avg_case_ms=4.87 simplify=31.82ms avg_simplify_ms=0.32, sum total=200 failed=0 elapsed=425.54ms avg_case_ms=2.13 simplify=51.77ms avg_simplify_ms=0.26, product total=100 failed=0 elapsed=295.82ms avg_case_ms=2.96 simplify=23.00ms avg_simplify_ms=0.23, difference total=50 failed=0 elapsed=191.60ms avg_case_ms=3.83 simplify=13.38ms avg_simplify_ms=0.27
- Engine hotspots: sum simplify=51.77ms avg_simplify_ms=0.26 wall=425.54ms, shifted_quotient simplify=31.82ms avg_simplify_ms=0.32 wall=486.76ms, product simplify=23.00ms avg_simplify_ms=0.23 wall=295.82ms, difference simplify=13.38ms avg_simplify_ms=0.27 wall=191.60ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=486.76ms avg_case_ms=4.87 avg_simplify_ms=0.32, sum@0+100 failed=0 elapsed=301.12ms avg_case_ms=3.01 avg_simplify_ms=0.28, product@0+100 failed=0 elapsed=295.82ms avg_case_ms=2.96 avg_simplify_ms=0.23, difference@0+50 failed=0 elapsed=191.60ms avg_case_ms=3.83 avg_simplify_ms=0.27, sum@700+100 failed=0 elapsed=124.41ms avg_case_ms=1.24 avg_simplify_ms=0.24
- Steady-state engine reruns: shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=1.53ms median_wire=1.63ms median_wall=9.83ms, shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=1.48ms median_wire=1.55ms median_wall=11.04ms, sum@700+100 #2829 sum runs=3 median_simplify=0.69ms median_wire=0.73ms median_wall=5.73ms, sum@0+100 #61 sum runs=3 median_simplify=0.28ms median_wire=0.33ms median_wall=2.97ms, product@0+100 #175 product runs=3 median_simplify=0.47ms median_wire=0.52ms median_wall=26.82ms
- Steady-state dominant expressions: shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), sum@700+100 #2829 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.40s | passed=450 failed=0 total=450 avg_case=3.111ms |
