# Engine Improvement Scorecard

- Generated: 2026-05-01T19:08:34.617302+00:00
- Git branch: main
- Git commit: `d35e85f33ae22fa3397a805d5d2450355a4b5616`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=159.63ms avg_case_ms=1.60 simplify=14.12ms avg_simplify_ms=0.14, sum total=200 failed=0 elapsed=156.81ms avg_case_ms=0.78 simplify=23.77ms avg_simplify_ms=0.12, product total=100 failed=0 elapsed=105.46ms avg_case_ms=1.05 simplify=10.41ms avg_simplify_ms=0.10, difference total=50 failed=0 elapsed=69.98ms avg_case_ms=1.40 simplify=5.48ms avg_simplify_ms=0.11
- Engine hotspots: sum simplify=23.77ms avg_simplify_ms=0.12 wall=156.81ms, shifted_quotient simplify=14.12ms avg_simplify_ms=0.14 wall=159.63ms, product simplify=10.41ms avg_simplify_ms=0.10 wall=105.46ms, difference simplify=5.48ms avg_simplify_ms=0.11 wall=69.98ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=159.63ms avg_case_ms=1.60 avg_simplify_ms=0.14, sum@0+100 failed=0 elapsed=113.32ms avg_case_ms=1.13 avg_simplify_ms=0.13, product@0+100 failed=0 elapsed=105.46ms avg_case_ms=1.05 avg_simplify_ms=0.10, difference@0+50 failed=0 elapsed=69.98ms avg_case_ms=1.40 avg_simplify_ms=0.11, sum@700+100 failed=0 elapsed=43.49ms avg_case_ms=0.43 avg_simplify_ms=0.10
- Steady-state engine reruns: shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=0.74ms median_wire=0.81ms median_wall=1.59ms, shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=0.73ms median_wire=0.79ms median_wall=1.84ms, sum@0+100 #25 sum runs=3 median_simplify=0.12ms median_wire=0.14ms median_wall=0.71ms, shifted_quotient@0+100 #296 shifted_quotient runs=3 median_simplify=0.25ms median_wire=0.31ms median_wall=1.05ms, sum@0+100 #1 sum runs=3 median_simplify=0.14ms median_wire=0.18ms median_wall=3.39ms
- Steady-state dominant expressions: shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), sum@0+100 #25 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (cosh(x) + sinh(x) - e^x)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 0.49s | passed=450 failed=0 total=450 avg_case=1.094ms |
