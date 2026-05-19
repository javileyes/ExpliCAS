# Engine Improvement Scorecard

- Generated: 2026-05-19T07:40:20.992369+00:00
- Git branch: main
- Git commit: `a26ea4b555d9790a3b4e922fbd97535521945e5b`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=475.36ms avg_case_ms=4.75 simplify=31.71ms avg_simplify_ms=0.32, sum total=200 failed=0 elapsed=402.59ms avg_case_ms=2.01 simplify=46.49ms avg_simplify_ms=0.23, product total=100 failed=0 elapsed=283.72ms avg_case_ms=2.84 simplify=21.51ms avg_simplify_ms=0.22, difference total=50 failed=0 elapsed=191.53ms avg_case_ms=3.83 simplify=14.34ms avg_simplify_ms=0.29
- Engine hotspots: sum simplify=46.49ms avg_simplify_ms=0.23 wall=402.59ms, shifted_quotient simplify=31.71ms avg_simplify_ms=0.32 wall=475.36ms, product simplify=21.51ms avg_simplify_ms=0.22 wall=283.72ms, difference simplify=14.34ms avg_simplify_ms=0.29 wall=191.53ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=475.36ms avg_case_ms=4.75 avg_simplify_ms=0.32, sum@0+100 failed=0 elapsed=284.41ms avg_case_ms=2.84 avg_simplify_ms=0.24, product@0+100 failed=0 elapsed=283.72ms avg_case_ms=2.84 avg_simplify_ms=0.22, difference@0+50 failed=0 elapsed=191.53ms avg_case_ms=3.83 avg_simplify_ms=0.29, sum@700+100 failed=0 elapsed=118.18ms avg_case_ms=1.18 avg_simplify_ms=0.22
- Steady-state engine reruns: shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=1.50ms median_wire=1.57ms median_wall=9.33ms, shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=1.43ms median_wire=1.51ms median_wall=10.30ms, difference@0+50 #18 difference runs=3 median_simplify=0.54ms median_wire=0.58ms median_wall=10.27ms, shifted_quotient@0+100 #20 shifted_quotient runs=3 median_simplify=0.52ms median_wire=0.57ms median_wall=13.70ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=0.66ms median_wire=0.73ms median_wall=28.17ms
- Steady-state dominant expressions: shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), difference@0+50 #18 difference expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.35s | passed=450 failed=0 total=450 avg_case=3.000ms |
