# Engine Improvement Scorecard

- Generated: 2026-05-17T21:36:08.725591+00:00
- Git branch: main
- Git commit: `dd2d4ff4773accacc644966841d5d0302e1ff92a`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=463.61ms avg_case_ms=4.64 simplify=28.91ms avg_simplify_ms=0.29, sum total=200 failed=0 elapsed=405.87ms avg_case_ms=2.03 simplify=45.69ms avg_simplify_ms=0.23, product total=100 failed=0 elapsed=279.32ms avg_case_ms=2.79 simplify=20.17ms avg_simplify_ms=0.20, difference total=50 failed=0 elapsed=176.42ms avg_case_ms=3.53 simplify=11.36ms avg_simplify_ms=0.23
- Engine hotspots: sum simplify=45.69ms avg_simplify_ms=0.23 wall=405.87ms, shifted_quotient simplify=28.91ms avg_simplify_ms=0.29 wall=463.61ms, product simplify=20.17ms avg_simplify_ms=0.20 wall=279.32ms, difference simplify=11.36ms avg_simplify_ms=0.23 wall=176.42ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=463.61ms avg_case_ms=4.64 avg_simplify_ms=0.29, sum@0+100 failed=0 elapsed=291.34ms avg_case_ms=2.91 avg_simplify_ms=0.25, product@0+100 failed=0 elapsed=279.32ms avg_case_ms=2.79 avg_simplify_ms=0.20, difference@0+50 failed=0 elapsed=176.42ms avg_case_ms=3.53 avg_simplify_ms=0.23, sum@700+100 failed=0 elapsed=114.54ms avg_case_ms=1.15 avg_simplify_ms=0.21
- Steady-state engine reruns: shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=1.42ms median_wire=1.49ms median_wall=10.26ms, shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=1.48ms median_wire=1.56ms median_wall=9.28ms, sum@0+100 #17 sum runs=3 median_simplify=0.43ms median_wire=0.46ms median_wall=10.29ms, sum@0+100 #57 sum runs=3 median_simplify=0.21ms median_wire=0.24ms median_wall=1.30ms, sum@700+100 #2829 sum runs=3 median_simplify=0.60ms median_wire=0.64ms median_wall=5.56ms
- Steady-state dominant expressions: shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), sum@0+100 #17 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.33s | passed=450 failed=0 total=450 avg_case=2.956ms |
