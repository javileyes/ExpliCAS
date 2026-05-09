# Engine Improvement Scorecard

- Generated: 2026-05-09T03:05:29.056930+00:00
- Git branch: main
- Git commit: `1ef66ab15d17b32b8fd5c7c716e2fafd86493b87`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: sum total=200 failed=0 elapsed=186.45ms avg_case_ms=0.93 simplify=27.59ms avg_simplify_ms=0.14, shifted_quotient total=100 failed=0 elapsed=186.27ms avg_case_ms=1.86 simplify=20.16ms avg_simplify_ms=0.20, product total=100 failed=0 elapsed=125.71ms avg_case_ms=1.26 simplify=13.42ms avg_simplify_ms=0.13, difference total=50 failed=0 elapsed=74.59ms avg_case_ms=1.49 simplify=5.97ms avg_simplify_ms=0.12
- Engine hotspots: sum simplify=27.59ms avg_simplify_ms=0.14 wall=186.45ms, shifted_quotient simplify=20.16ms avg_simplify_ms=0.20 wall=186.27ms, product simplify=13.42ms avg_simplify_ms=0.13 wall=125.71ms, difference simplify=5.97ms avg_simplify_ms=0.12 wall=74.59ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=186.27ms avg_case_ms=1.86 avg_simplify_ms=0.20, sum@0+100 failed=0 elapsed=132.85ms avg_case_ms=1.33 avg_simplify_ms=0.16, product@0+100 failed=0 elapsed=125.71ms avg_case_ms=1.26 avg_simplify_ms=0.13, difference@0+50 failed=0 elapsed=74.59ms avg_case_ms=1.49 avg_simplify_ms=0.12, sum@700+100 failed=0 elapsed=53.60ms avg_case_ms=0.54 avg_simplify_ms=0.12
- Steady-state engine reruns: shifted_quotient@0+100 #132 shifted_quotient runs=3 median_simplify=1.31ms median_wire=1.37ms median_wall=3.13ms, shifted_quotient@0+100 #288 shifted_quotient runs=3 median_simplify=1.38ms median_wire=1.45ms median_wall=3.03ms, shifted_quotient@0+100 #140 shifted_quotient runs=3 median_simplify=0.34ms median_wire=0.39ms median_wall=1.59ms, shifted_quotient@0+100 #296 shifted_quotient runs=3 median_simplify=0.40ms median_wire=0.46ms median_wall=1.46ms, sum@0+100 #25 sum runs=3 median_simplify=0.13ms median_wire=0.15ms median_wall=0.82ms
- Steady-state dominant expressions: shifted_quotient@0+100 #132 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #288 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1), shifted_quotient@0+100 #140 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^2 + 2*x + 1 - (x+1)^2) + 1)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 0.57s | passed=450 failed=0 total=450 avg_case=1.274ms |
