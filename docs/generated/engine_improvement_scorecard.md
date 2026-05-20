# Engine Improvement Scorecard

- Generated: 2026-05-20T09:13:06.303147+00:00
- Git branch: main
- Git commit: `0ca49274896dcc8e8d797170ebf10f3653caa606`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=593.50ms avg_case_ms=5.93 simplify=38.61ms avg_simplify_ms=0.39, sum total=200 failed=0 elapsed=484.95ms avg_case_ms=2.42 simplify=50.86ms avg_simplify_ms=0.25, product total=100 failed=0 elapsed=331.68ms avg_case_ms=3.32 simplify=24.76ms avg_simplify_ms=0.25, difference total=50 failed=0 elapsed=210.85ms avg_case_ms=4.22 simplify=13.06ms avg_simplify_ms=0.26
- Engine hotspots: sum simplify=50.86ms avg_simplify_ms=0.25 wall=484.95ms, shifted_quotient simplify=38.61ms avg_simplify_ms=0.39 wall=593.50ms, product simplify=24.76ms avg_simplify_ms=0.25 wall=331.68ms, difference simplify=13.06ms avg_simplify_ms=0.26 wall=210.85ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=593.50ms avg_case_ms=5.93 avg_simplify_ms=0.39, sum@0+100 failed=0 elapsed=355.82ms avg_case_ms=3.56 avg_simplify_ms=0.29, product@0+100 failed=0 elapsed=331.68ms avg_case_ms=3.32 avg_simplify_ms=0.25, difference@0+50 failed=0 elapsed=210.85ms avg_case_ms=4.22 avg_simplify_ms=0.26, sum@700+100 failed=0 elapsed=129.13ms avg_case_ms=1.29 avg_simplify_ms=0.22
- Steady-state engine reruns: shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=1.56ms median_wire=1.66ms median_wall=9.65ms, shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=1.46ms median_wire=1.54ms median_wall=12.03ms, product@0+100 #315 product runs=3 median_simplify=0.21ms median_wire=0.23ms median_wall=3.06ms, shifted_quotient@0+100 #296 shifted_quotient runs=3 median_simplify=0.59ms median_wire=0.66ms median_wall=5.07ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=0.73ms median_wire=0.81ms median_wall=38.27ms
- Steady-state dominant expressions: shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), product@0+100 #315 product expr=(tan(x) + cot(x) - sec(x)*csc(x)) * (ln(x^3) + ln(y^2) - ln(x^3 * y^2))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.62s | passed=450 failed=0 total=450 avg_case=3.600ms |
