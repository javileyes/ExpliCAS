# Engine Improvement Scorecard

- Generated: 2026-05-18T07:18:16.015311+00:00
- Git branch: main
- Git commit: `6090f324761b242f50ff32fa708d5c7acb16348d`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=484.55ms avg_case_ms=4.85 simplify=33.06ms avg_simplify_ms=0.33, sum total=200 failed=0 elapsed=431.64ms avg_case_ms=2.16 simplify=52.39ms avg_simplify_ms=0.26, product total=100 failed=0 elapsed=296.69ms avg_case_ms=2.97 simplify=22.67ms avg_simplify_ms=0.23, difference total=50 failed=0 elapsed=194.98ms avg_case_ms=3.90 simplify=14.12ms avg_simplify_ms=0.28
- Engine hotspots: sum simplify=52.39ms avg_simplify_ms=0.26 wall=431.64ms, shifted_quotient simplify=33.06ms avg_simplify_ms=0.33 wall=484.55ms, product simplify=22.67ms avg_simplify_ms=0.23 wall=296.69ms, difference simplify=14.12ms avg_simplify_ms=0.28 wall=194.98ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=484.55ms avg_case_ms=4.85 avg_simplify_ms=0.33, sum@0+100 failed=0 elapsed=304.51ms avg_case_ms=3.05 avg_simplify_ms=0.27, product@0+100 failed=0 elapsed=296.69ms avg_case_ms=2.97 avg_simplify_ms=0.23, difference@0+50 failed=0 elapsed=194.98ms avg_case_ms=3.90 avg_simplify_ms=0.28, sum@700+100 failed=0 elapsed=127.13ms avg_case_ms=1.27 avg_simplify_ms=0.25
- Steady-state engine reruns: shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=1.46ms median_wire=1.53ms median_wall=10.54ms, shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=1.56ms median_wire=1.65ms median_wall=9.94ms, sum@700+100 #2937 sum runs=3 median_simplify=0.24ms median_wire=0.28ms median_wall=0.56ms, sum@700+100 #2893 sum runs=3 median_simplify=0.30ms median_wire=0.33ms median_wall=0.60ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=0.69ms median_wire=0.77ms median_wall=29.95ms
- Steady-state dominant expressions: shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), sum@700+100 #2937 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.41s | passed=450 failed=0 total=450 avg_case=3.133ms |
