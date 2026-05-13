# Engine Improvement Scorecard

- Generated: 2026-05-13T14:12:34.161370+00:00
- Git branch: main
- Git commit: `91e0e6385c66cb5627bfa2243ee51de46f1e2908`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=293.09ms avg_case_ms=2.93 simplify=20.13ms avg_simplify_ms=0.20, sum total=200 failed=0 elapsed=254.26ms avg_case_ms=1.27 simplify=29.02ms avg_simplify_ms=0.15, product total=100 failed=0 elapsed=174.73ms avg_case_ms=1.75 simplify=13.49ms avg_simplify_ms=0.13, difference total=50 failed=0 elapsed=115.55ms avg_case_ms=2.31 simplify=7.58ms avg_simplify_ms=0.15
- Engine hotspots: sum simplify=29.02ms avg_simplify_ms=0.15 wall=254.26ms, shifted_quotient simplify=20.13ms avg_simplify_ms=0.20 wall=293.09ms, product simplify=13.49ms avg_simplify_ms=0.13 wall=174.73ms, difference simplify=7.58ms avg_simplify_ms=0.15 wall=115.55ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=293.09ms avg_case_ms=2.93 avg_simplify_ms=0.20, sum@0+100 failed=0 elapsed=184.71ms avg_case_ms=1.85 avg_simplify_ms=0.17, product@0+100 failed=0 elapsed=174.73ms avg_case_ms=1.75 avg_simplify_ms=0.13, difference@0+50 failed=0 elapsed=115.55ms avg_case_ms=2.31 avg_simplify_ms=0.15, sum@700+100 failed=0 elapsed=69.55ms avg_case_ms=0.70 avg_simplify_ms=0.12
- Steady-state engine reruns: shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=1.44ms median_wire=1.52ms median_wall=6.60ms, shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=1.30ms median_wire=1.37ms median_wall=7.26ms, sum@0+100 #5 sum runs=3 median_simplify=0.08ms median_wire=0.11ms median_wall=0.77ms, shifted_quotient@0+100 #296 shifted_quotient runs=3 median_simplify=0.45ms median_wire=0.52ms median_wall=3.06ms, sum@0+100 #25 sum runs=3 median_simplify=0.14ms median_wire=0.16ms median_wall=0.84ms
- Steady-state dominant expressions: shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), sum@0+100 #5 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (tan(x) + cot(x) - sec(x)*csc(x))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 0.84s | passed=450 failed=0 total=450 avg_case=1.863ms |
