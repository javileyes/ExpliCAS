# Engine Improvement Scorecard

- Generated: 2026-05-15T11:17:05.123441+00:00
- Git branch: main
- Git commit: `7cfb55b4d519c30977066f1569432bd065564184`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=316.94ms avg_case_ms=3.17 simplify=27.86ms avg_simplify_ms=0.28, sum total=200 failed=0 elapsed=285.73ms avg_case_ms=1.43 simplify=45.20ms avg_simplify_ms=0.23, product total=100 failed=0 elapsed=194.14ms avg_case_ms=1.94 simplify=20.62ms avg_simplify_ms=0.21, difference total=50 failed=0 elapsed=127.34ms avg_case_ms=2.55 simplify=12.35ms avg_simplify_ms=0.25
- Engine hotspots: sum simplify=45.20ms avg_simplify_ms=0.23 wall=285.73ms, shifted_quotient simplify=27.86ms avg_simplify_ms=0.28 wall=316.94ms, product simplify=20.62ms avg_simplify_ms=0.21 wall=194.14ms, difference simplify=12.35ms avg_simplify_ms=0.25 wall=127.34ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=316.94ms avg_case_ms=3.17 avg_simplify_ms=0.28, sum@0+100 failed=0 elapsed=199.72ms avg_case_ms=2.00 avg_simplify_ms=0.24, product@0+100 failed=0 elapsed=194.14ms avg_case_ms=1.94 avg_simplify_ms=0.21, difference@0+50 failed=0 elapsed=127.34ms avg_case_ms=2.55 avg_simplify_ms=0.25, sum@700+100 failed=0 elapsed=86.01ms avg_case_ms=0.86 avg_simplify_ms=0.21
- Steady-state engine reruns: shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=1.39ms median_wire=1.46ms median_wall=7.36ms, shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=1.44ms median_wire=1.52ms median_wall=6.63ms, sum@700+100 #2829 sum runs=3 median_simplify=0.54ms median_wire=0.57ms median_wall=3.91ms, difference@0+50 #66 difference runs=3 median_simplify=0.38ms median_wire=0.42ms median_wall=2.86ms, shifted_quotient@0+100 #20 shifted_quotient runs=3 median_simplify=0.55ms median_wire=0.61ms median_wall=9.40ms
- Steady-state dominant expressions: shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), sum@700+100 #2829 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 0.93s | passed=450 failed=0 total=450 avg_case=2.054ms |
