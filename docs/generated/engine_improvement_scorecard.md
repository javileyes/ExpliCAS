# Engine Improvement Scorecard

- Generated: 2026-04-30T08:27:26.882939+00:00
- Git branch: main
- Git commit: `5bca778c5a132fd912ece96b44ad48f759f0b6ae`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=144.06ms avg_case_ms=1.44 simplify=12.13ms avg_simplify_ms=0.12, sum total=200 failed=0 elapsed=139.66ms avg_case_ms=0.70 simplify=23.08ms avg_simplify_ms=0.12, product total=100 failed=0 elapsed=96.84ms avg_case_ms=0.97 simplify=9.80ms avg_simplify_ms=0.10, difference total=50 failed=0 elapsed=66.60ms avg_case_ms=1.33 simplify=5.32ms avg_simplify_ms=0.11
- Engine hotspots: sum simplify=23.08ms avg_simplify_ms=0.12 wall=139.66ms, shifted_quotient simplify=12.13ms avg_simplify_ms=0.12 wall=144.06ms, product simplify=9.80ms avg_simplify_ms=0.10 wall=96.84ms, difference simplify=5.32ms avg_simplify_ms=0.11 wall=66.60ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=144.06ms avg_case_ms=1.44 avg_simplify_ms=0.12, sum@0+100 failed=0 elapsed=100.64ms avg_case_ms=1.01 avg_simplify_ms=0.13, product@0+100 failed=0 elapsed=96.84ms avg_case_ms=0.97 avg_simplify_ms=0.10, difference@0+50 failed=0 elapsed=66.60ms avg_case_ms=1.33 avg_simplify_ms=0.11, sum@700+100 failed=0 elapsed=39.02ms avg_case_ms=0.39 avg_simplify_ms=0.10
- Steady-state engine reruns: shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=0.52ms median_wire=0.60ms median_wall=1.29ms, shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=0.48ms median_wire=0.54ms median_wall=1.63ms, sum@0+100 #25 sum runs=3 median_simplify=0.12ms median_wire=0.14ms median_wall=0.72ms, sum@0+100 #13 sum runs=3 median_simplify=0.10ms median_wire=0.13ms median_wall=2.42ms, sum@0+100 #1 sum runs=3 median_simplify=0.14ms median_wire=0.18ms median_wall=3.45ms
- Steady-state dominant expressions: shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), sum@0+100 #25 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (cosh(x) + sinh(x) - e^x)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 0.45s | passed=450 failed=0 total=450 avg_case=0.995ms |
