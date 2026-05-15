# Engine Improvement Scorecard

- Generated: 2026-05-15T21:00:55.401264+00:00
- Git branch: main
- Git commit: `db174c422c93ebfcce0da01356ba5c01bbb37e69`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=339.97ms avg_case_ms=3.40 simplify=31.72ms avg_simplify_ms=0.32, sum total=200 failed=0 elapsed=299.19ms avg_case_ms=1.50 simplify=48.71ms avg_simplify_ms=0.24, product total=100 failed=0 elapsed=199.55ms avg_case_ms=2.00 simplify=21.43ms avg_simplify_ms=0.21, difference total=50 failed=0 elapsed=128.70ms avg_case_ms=2.57 simplify=12.52ms avg_simplify_ms=0.25
- Engine hotspots: sum simplify=48.71ms avg_simplify_ms=0.24 wall=299.19ms, shifted_quotient simplify=31.72ms avg_simplify_ms=0.32 wall=339.97ms, product simplify=21.43ms avg_simplify_ms=0.21 wall=199.55ms, difference simplify=12.52ms avg_simplify_ms=0.25 wall=128.70ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=339.97ms avg_case_ms=3.40 avg_simplify_ms=0.32, sum@0+100 failed=0 elapsed=210.44ms avg_case_ms=2.10 avg_simplify_ms=0.27, product@0+100 failed=0 elapsed=199.55ms avg_case_ms=2.00 avg_simplify_ms=0.21, difference@0+50 failed=0 elapsed=128.70ms avg_case_ms=2.57 avg_simplify_ms=0.25, sum@700+100 failed=0 elapsed=88.75ms avg_case_ms=0.89 avg_simplify_ms=0.22
- Steady-state engine reruns: shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=1.48ms median_wire=1.56ms median_wall=7.69ms, shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=1.57ms median_wire=1.70ms median_wall=7.06ms, sum@0+100 #17 sum runs=3 median_simplify=0.54ms median_wire=0.57ms median_wall=7.51ms, sum@0+100 #1 sum runs=3 median_simplify=0.40ms median_wire=0.46ms median_wall=12.24ms, sum@700+100 #2829 sum runs=3 median_simplify=0.60ms median_wire=0.65ms median_wall=3.96ms
- Steady-state dominant expressions: shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), sum@0+100 #17 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 0.97s | passed=450 failed=0 total=450 avg_case=2.151ms |
