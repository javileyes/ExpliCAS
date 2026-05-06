# Engine Improvement Scorecard

- Generated: 2026-05-06T05:38:01.609997+00:00
- Git branch: main
- Git commit: `ad49581a26cbae51c35913bf1b2a430dff033eea`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=165.17ms avg_case_ms=1.65 simplify=14.69ms avg_simplify_ms=0.15, sum total=200 failed=0 elapsed=161.77ms avg_case_ms=0.81 simplify=24.54ms avg_simplify_ms=0.12, product total=100 failed=0 elapsed=112.69ms avg_case_ms=1.13 simplify=11.32ms avg_simplify_ms=0.11, difference total=50 failed=0 elapsed=74.38ms avg_case_ms=1.49 simplify=5.87ms avg_simplify_ms=0.12
- Engine hotspots: sum simplify=24.54ms avg_simplify_ms=0.12 wall=161.77ms, shifted_quotient simplify=14.69ms avg_simplify_ms=0.15 wall=165.17ms, product simplify=11.32ms avg_simplify_ms=0.11 wall=112.69ms, difference simplify=5.87ms avg_simplify_ms=0.12 wall=74.38ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=165.17ms avg_case_ms=1.65 avg_simplify_ms=0.15, sum@0+100 failed=0 elapsed=114.38ms avg_case_ms=1.14 avg_simplify_ms=0.13, product@0+100 failed=0 elapsed=112.69ms avg_case_ms=1.13 avg_simplify_ms=0.11, difference@0+50 failed=0 elapsed=74.38ms avg_case_ms=1.49 avg_simplify_ms=0.12, sum@700+100 failed=0 elapsed=47.39ms avg_case_ms=0.47 avg_simplify_ms=0.11
- Steady-state engine reruns: shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=0.79ms median_wire=0.86ms median_wall=2.34ms, shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=0.73ms median_wire=0.80ms median_wall=2.57ms, sum@0+100 #25 sum runs=3 median_simplify=0.13ms median_wire=0.15ms median_wall=0.83ms, shifted_quotient@0+100 #296 shifted_quotient runs=3 median_simplify=0.27ms median_wire=0.33ms median_wall=1.26ms, sum@700+100 #2909 sum runs=3 median_simplify=0.18ms median_wire=0.20ms median_wall=0.33ms
- Steady-state dominant expressions: shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), sum@0+100 #25 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (cosh(x) + sinh(x) - e^x)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 0.51s | passed=450 failed=0 total=450 avg_case=1.143ms |
