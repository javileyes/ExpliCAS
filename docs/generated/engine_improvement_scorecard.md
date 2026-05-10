# Engine Improvement Scorecard

- Generated: 2026-05-10T17:40:20.244812+00:00
- Git branch: main
- Git commit: `88cc1309eeb8a3d247a32c15835d8d1f67bd819b`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=285.28ms avg_case_ms=2.85 simplify=22.06ms avg_simplify_ms=0.22, sum total=200 failed=0 elapsed=251.49ms avg_case_ms=1.26 simplify=32.74ms avg_simplify_ms=0.16, product total=100 failed=0 elapsed=174.06ms avg_case_ms=1.74 simplify=15.22ms avg_simplify_ms=0.15, difference total=50 failed=0 elapsed=115.05ms avg_case_ms=2.30 simplify=8.02ms avg_simplify_ms=0.16
- Engine hotspots: sum simplify=32.74ms avg_simplify_ms=0.16 wall=251.49ms, shifted_quotient simplify=22.06ms avg_simplify_ms=0.22 wall=285.28ms, product simplify=15.22ms avg_simplify_ms=0.15 wall=174.06ms, difference simplify=8.02ms avg_simplify_ms=0.16 wall=115.05ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=285.28ms avg_case_ms=2.85 avg_simplify_ms=0.22, sum@0+100 failed=0 elapsed=183.07ms avg_case_ms=1.83 avg_simplify_ms=0.20, product@0+100 failed=0 elapsed=174.06ms avg_case_ms=1.74 avg_simplify_ms=0.15, difference@0+50 failed=0 elapsed=115.05ms avg_case_ms=2.30 avg_simplify_ms=0.16, sum@700+100 failed=0 elapsed=68.42ms avg_case_ms=0.68 avg_simplify_ms=0.13
- Steady-state engine reruns: shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=1.41ms median_wire=1.50ms median_wall=6.52ms, shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=1.43ms median_wire=1.49ms median_wall=7.36ms, sum@0+100 #25 sum runs=3 median_simplify=0.13ms median_wire=0.15ms median_wall=0.87ms, sum@0+100 #337 sum runs=3 median_simplify=0.13ms median_wire=0.14ms median_wall=0.16ms, shifted_quotient@0+100 #296 shifted_quotient runs=3 median_simplify=0.45ms median_wire=0.51ms median_wall=3.02ms
- Steady-state dominant expressions: shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), sum@0+100 #25 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (cosh(x) + sinh(x) - e^x)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 0.83s | passed=450 failed=0 total=450 avg_case=1.836ms |
