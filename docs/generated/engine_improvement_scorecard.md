# Engine Improvement Scorecard

- Generated: 2026-05-13T08:01:46.550964+00:00
- Git branch: main
- Git commit: `a047b19f464227ad3677455f57f7a485dd370e64`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=281.54ms avg_case_ms=2.82 simplify=18.87ms avg_simplify_ms=0.19, sum total=200 failed=0 elapsed=239.94ms avg_case_ms=1.20 simplify=26.13ms avg_simplify_ms=0.13, product total=100 failed=0 elapsed=168.11ms avg_case_ms=1.68 simplify=12.42ms avg_simplify_ms=0.12, difference total=50 failed=0 elapsed=111.14ms avg_case_ms=2.22 simplify=7.02ms avg_simplify_ms=0.14
- Engine hotspots: sum simplify=26.13ms avg_simplify_ms=0.13 wall=239.94ms, shifted_quotient simplify=18.87ms avg_simplify_ms=0.19 wall=281.54ms, product simplify=12.42ms avg_simplify_ms=0.12 wall=168.11ms, difference simplify=7.02ms avg_simplify_ms=0.14 wall=111.14ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=281.54ms avg_case_ms=2.82 avg_simplify_ms=0.19, sum@0+100 failed=0 elapsed=174.03ms avg_case_ms=1.74 avg_simplify_ms=0.15, product@0+100 failed=0 elapsed=168.11ms avg_case_ms=1.68 avg_simplify_ms=0.12, difference@0+50 failed=0 elapsed=111.14ms avg_case_ms=2.22 avg_simplify_ms=0.14, sum@700+100 failed=0 elapsed=65.92ms avg_case_ms=0.66 avg_simplify_ms=0.11
- Steady-state engine reruns: shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=1.34ms median_wire=1.43ms median_wall=6.34ms, shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=1.26ms median_wire=1.31ms median_wall=6.96ms, shifted_quotient@0+100 #296 shifted_quotient runs=3 median_simplify=0.42ms median_wire=0.48ms median_wall=2.94ms, sum@0+100 #25 sum runs=3 median_simplify=0.13ms median_wire=0.15ms median_wall=0.82ms, shifted_quotient@0+100 #140 shifted_quotient runs=3 median_simplify=0.33ms median_wire=0.38ms median_wall=3.27ms
- Steady-state dominant expressions: shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #296 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^2 + 2*x + 1 - (x+1)^2) + 1)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 0.80s | passed=450 failed=0 total=450 avg_case=1.780ms |
