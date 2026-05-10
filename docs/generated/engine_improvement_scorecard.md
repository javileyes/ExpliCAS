# Engine Improvement Scorecard

- Generated: 2026-05-10T07:18:51.485672+00:00
- Git branch: main
- Git commit: `c060f3cd44e182078e2bb1724ba6cae1ca876ac3`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=282.29ms avg_case_ms=2.82 simplify=20.82ms avg_simplify_ms=0.21, sum total=200 failed=0 elapsed=244.61ms avg_case_ms=1.22 simplify=30.05ms avg_simplify_ms=0.15, product total=100 failed=0 elapsed=169.22ms avg_case_ms=1.69 simplify=14.12ms avg_simplify_ms=0.14, difference total=50 failed=0 elapsed=113.28ms avg_case_ms=2.27 simplify=7.53ms avg_simplify_ms=0.15
- Engine hotspots: sum simplify=30.05ms avg_simplify_ms=0.15 wall=244.61ms, shifted_quotient simplify=20.82ms avg_simplify_ms=0.21 wall=282.29ms, product simplify=14.12ms avg_simplify_ms=0.14 wall=169.22ms, difference simplify=7.53ms avg_simplify_ms=0.15 wall=113.28ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=282.29ms avg_case_ms=2.82 avg_simplify_ms=0.21, sum@0+100 failed=0 elapsed=177.96ms avg_case_ms=1.78 avg_simplify_ms=0.18, product@0+100 failed=0 elapsed=169.22ms avg_case_ms=1.69 avg_simplify_ms=0.14, difference@0+50 failed=0 elapsed=113.28ms avg_case_ms=2.27 avg_simplify_ms=0.15, sum@700+100 failed=0 elapsed=66.65ms avg_case_ms=0.67 avg_simplify_ms=0.12
- Steady-state engine reruns: shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=1.43ms median_wire=1.52ms median_wall=6.49ms, shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=1.36ms median_wire=1.43ms median_wall=7.33ms, sum@0+100 #25 sum runs=3 median_simplify=0.14ms median_wire=0.17ms median_wall=0.89ms, shifted_quotient@0+100 #296 shifted_quotient runs=3 median_simplify=0.46ms median_wire=0.52ms median_wall=2.95ms, sum@0+100 #9 sum runs=3 median_simplify=0.23ms median_wire=0.28ms median_wall=4.93ms
- Steady-state dominant expressions: shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), sum@0+100 #25 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (cosh(x) + sinh(x) - e^x)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 0.81s | passed=450 failed=0 total=450 avg_case=1.800ms |
