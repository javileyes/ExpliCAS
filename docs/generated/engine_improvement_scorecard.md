# Engine Improvement Scorecard

- Generated: 2026-05-09T21:42:39.840964+00:00
- Git branch: main
- Git commit: `a7631cdd6898f62d9bb523edcae8b081f0e1ecf0`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=288.31ms avg_case_ms=2.88 simplify=20.38ms avg_simplify_ms=0.20, sum total=200 failed=0 elapsed=245.92ms avg_case_ms=1.23 simplify=28.90ms avg_simplify_ms=0.14, product total=100 failed=0 elapsed=172.13ms avg_case_ms=1.72 simplify=14.01ms avg_simplify_ms=0.14, difference total=50 failed=0 elapsed=112.86ms avg_case_ms=2.26 simplify=7.07ms avg_simplify_ms=0.14
- Engine hotspots: sum simplify=28.90ms avg_simplify_ms=0.14 wall=245.92ms, shifted_quotient simplify=20.38ms avg_simplify_ms=0.20 wall=288.31ms, product simplify=14.01ms avg_simplify_ms=0.14 wall=172.13ms, difference simplify=7.07ms avg_simplify_ms=0.14 wall=112.86ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=288.31ms avg_case_ms=2.88 avg_simplify_ms=0.20, sum@0+100 failed=0 elapsed=180.20ms avg_case_ms=1.80 avg_simplify_ms=0.17, product@0+100 failed=0 elapsed=172.13ms avg_case_ms=1.72 avg_simplify_ms=0.14, difference@0+50 failed=0 elapsed=112.86ms avg_case_ms=2.26 avg_simplify_ms=0.14, sum@700+100 failed=0 elapsed=65.72ms avg_case_ms=0.66 avg_simplify_ms=0.11
- Steady-state engine reruns: shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=1.42ms median_wire=1.50ms median_wall=6.51ms, shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=1.44ms median_wire=1.52ms median_wall=7.65ms, sum@0+100 #65 sum runs=3 median_simplify=0.18ms median_wire=0.22ms median_wall=2.49ms, shifted_quotient@0+100 #296 shifted_quotient runs=3 median_simplify=0.43ms median_wire=0.50ms median_wall=3.09ms, sum@0+100 #33 sum runs=3 median_simplify=0.10ms median_wire=0.14ms median_wall=0.86ms
- Steady-state dominant expressions: shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), sum@0+100 #65 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (sqrt(a^2 + 2*a*b + b^2) - abs(a+b))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 0.82s | passed=450 failed=0 total=450 avg_case=1.822ms |
