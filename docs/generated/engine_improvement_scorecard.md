# Engine Improvement Scorecard

- Generated: 2026-05-11T17:32:19.384259+00:00
- Git branch: main
- Git commit: `2e2017f3db1726f49c0046e043c50e074166d906`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=269.11ms avg_case_ms=2.69 simplify=19.16ms avg_simplify_ms=0.19, sum total=200 failed=0 elapsed=244.31ms avg_case_ms=1.22 simplify=27.94ms avg_simplify_ms=0.14, product total=100 failed=0 elapsed=161.37ms avg_case_ms=1.61 simplify=12.54ms avg_simplify_ms=0.13, difference total=50 failed=0 elapsed=109.71ms avg_case_ms=2.19 simplify=6.86ms avg_simplify_ms=0.14
- Engine hotspots: sum simplify=27.94ms avg_simplify_ms=0.14 wall=244.31ms, shifted_quotient simplify=19.16ms avg_simplify_ms=0.19 wall=269.11ms, product simplify=12.54ms avg_simplify_ms=0.13 wall=161.37ms, difference simplify=6.86ms avg_simplify_ms=0.14 wall=109.71ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=269.11ms avg_case_ms=2.69 avg_simplify_ms=0.19, sum@0+100 failed=0 elapsed=180.46ms avg_case_ms=1.80 avg_simplify_ms=0.17, product@0+100 failed=0 elapsed=161.37ms avg_case_ms=1.61 avg_simplify_ms=0.13, difference@0+50 failed=0 elapsed=109.71ms avg_case_ms=2.19 avg_simplify_ms=0.14, sum@700+100 failed=0 elapsed=63.86ms avg_case_ms=0.64 avg_simplify_ms=0.11
- Steady-state engine reruns: shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=1.36ms median_wire=1.44ms median_wall=6.18ms, shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=1.27ms median_wire=1.33ms median_wall=6.82ms, shifted_quotient@0+100 #296 shifted_quotient runs=3 median_simplify=0.43ms median_wire=0.49ms median_wall=2.80ms, sum@0+100 #25 sum runs=3 median_simplify=0.13ms median_wire=0.15ms median_wall=0.81ms, shifted_quotient@0+100 #140 shifted_quotient runs=3 median_simplify=0.35ms median_wire=0.40ms median_wall=3.32ms
- Steady-state dominant expressions: shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #296 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^2 + 2*x + 1 - (x+1)^2) + 1)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 0.79s | passed=450 failed=0 total=450 avg_case=1.744ms |
