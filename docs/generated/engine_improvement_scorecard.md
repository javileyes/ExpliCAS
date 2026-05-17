# Engine Improvement Scorecard

- Generated: 2026-05-16T20:50:38.210016+00:00
- Git branch: main
- Git commit: `958dfdb05fe6fff4158e7bff7f80cf0092350b6c`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=446.45ms avg_case_ms=4.46 simplify=26.57ms avg_simplify_ms=0.27, sum total=200 failed=0 elapsed=408.89ms avg_case_ms=2.04 simplify=45.51ms avg_simplify_ms=0.23, product total=100 failed=0 elapsed=268.09ms avg_case_ms=2.68 simplify=18.96ms avg_simplify_ms=0.19, difference total=50 failed=0 elapsed=167.57ms avg_case_ms=3.35 simplify=10.74ms avg_simplify_ms=0.21
- Engine hotspots: sum simplify=45.51ms avg_simplify_ms=0.23 wall=408.89ms, shifted_quotient simplify=26.57ms avg_simplify_ms=0.27 wall=446.45ms, product simplify=18.96ms avg_simplify_ms=0.19 wall=268.09ms, difference simplify=10.74ms avg_simplify_ms=0.21 wall=167.57ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=446.45ms avg_case_ms=4.46 avg_simplify_ms=0.27, sum@0+100 failed=0 elapsed=292.26ms avg_case_ms=2.92 avg_simplify_ms=0.24, product@0+100 failed=0 elapsed=268.09ms avg_case_ms=2.68 avg_simplify_ms=0.19, difference@0+50 failed=0 elapsed=167.57ms avg_case_ms=3.35 avg_simplify_ms=0.21, sum@700+100 failed=0 elapsed=116.64ms avg_case_ms=1.17 avg_simplify_ms=0.21
- Steady-state engine reruns: shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=1.39ms median_wire=1.47ms median_wall=8.82ms, shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=1.32ms median_wire=1.38ms median_wall=9.85ms, sum@700+100 #2829 sum runs=3 median_simplify=0.57ms median_wire=0.60ms median_wall=5.08ms, shifted_quotient@0+100 #20 shifted_quotient runs=3 median_simplify=0.50ms median_wire=0.55ms median_wall=12.93ms, sum@0+100 #65 sum runs=3 median_simplify=0.29ms median_wire=0.32ms median_wall=3.66ms
- Steady-state dominant expressions: shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), sum@700+100 #2829 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.29s | passed=450 failed=0 total=450 avg_case=2.867ms |
