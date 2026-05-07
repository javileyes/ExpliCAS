# Engine Improvement Scorecard

- Generated: 2026-05-07T15:19:05.849401+00:00
- Git branch: main
- Git commit: `6ff94a5a8f363f5a037ab7b2aafe8dde28399909`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=181.74ms avg_case_ms=1.82 simplify=19.55ms avg_simplify_ms=0.20, sum total=200 failed=0 elapsed=180.18ms avg_case_ms=0.90 simplify=28.52ms avg_simplify_ms=0.14, product total=100 failed=0 elapsed=120.84ms avg_case_ms=1.21 simplify=13.57ms avg_simplify_ms=0.14, difference total=50 failed=0 elapsed=80.28ms avg_case_ms=1.61 simplify=7.16ms avg_simplify_ms=0.14
- Engine hotspots: sum simplify=28.52ms avg_simplify_ms=0.14 wall=180.18ms, shifted_quotient simplify=19.55ms avg_simplify_ms=0.20 wall=181.74ms, product simplify=13.57ms avg_simplify_ms=0.14 wall=120.84ms, difference simplify=7.16ms avg_simplify_ms=0.14 wall=80.28ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=181.74ms avg_case_ms=1.82 avg_simplify_ms=0.20, sum@0+100 failed=0 elapsed=126.74ms avg_case_ms=1.27 avg_simplify_ms=0.17, product@0+100 failed=0 elapsed=120.84ms avg_case_ms=1.21 avg_simplify_ms=0.14, difference@0+50 failed=0 elapsed=80.28ms avg_case_ms=1.61 avg_simplify_ms=0.14, sum@700+100 failed=0 elapsed=53.45ms avg_case_ms=0.53 avg_simplify_ms=0.12
- Steady-state engine reruns: shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=1.48ms median_wire=1.57ms median_wall=3.23ms, shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=1.38ms median_wire=1.45ms median_wall=3.25ms, shifted_quotient@0+100 #296 shifted_quotient runs=3 median_simplify=0.41ms median_wire=0.48ms median_wall=1.53ms, sum@0+100 #25 sum runs=3 median_simplify=0.14ms median_wire=0.16ms median_wall=0.82ms, shifted_quotient@0+100 #140 shifted_quotient runs=3 median_simplify=0.35ms median_wire=0.40ms median_wall=1.62ms
- Steady-state dominant expressions: shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #296 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^2 + 2*x + 1 - (x+1)^2) + 1)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 0.56s | passed=450 failed=0 total=450 avg_case=1.254ms |
