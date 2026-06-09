# Engine Improvement Scorecard

- Generated: 2026-06-09T15:50:45.041187+00:00
- Git branch: main
- Git commit: `815dab9a86434caf270d0f386d64732c85ea5bce`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=15
- By area: calculus / integration:5, calculus / general integration backend:4, calculus / runtime:3, calculus / differentiation:2, calculus / robustness:1
- Recent 1: `calculus / general integration backend` - 2026-06-09 - Discovery observe-only: backend positive-quadratic arctan branch needs verifier policy
- Recent 2: `calculus / general integration backend` - 2026-06-09 - Discovery observe-only: backend mixed positive-quadratic numerator needs decomposition policy
- Recent 3: `calculus / general integration backend` - 2026-06-09 - Discovery observe-only: backend log-derivative verification misses unsimplified power decrement

## Calculus Support Matrix Signal

- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.
- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.
- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.
- `diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=263
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=765.36ms avg_case_ms=7.65 simplify=217.03ms avg_simplify_ms=2.17, sum total=200 failed=0 elapsed=711.69ms avg_case_ms=3.56 simplify=237.30ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=464.67ms avg_case_ms=4.65 simplify=133.62ms avg_simplify_ms=1.34, difference total=50 failed=0 elapsed=321.56ms avg_case_ms=6.43 simplify=102.15ms avg_simplify_ms=2.04
- Engine hotspots: sum simplify=237.30ms avg_simplify_ms=1.19 wall=711.69ms, shifted_quotient simplify=217.03ms avg_simplify_ms=2.17 wall=765.36ms, product simplify=133.62ms avg_simplify_ms=1.34 wall=464.67ms, difference simplify=102.15ms avg_simplify_ms=2.04 wall=321.56ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=765.36ms avg_case_ms=7.65 avg_simplify_ms=2.17, sum@0+100 failed=0 elapsed=525.90ms avg_case_ms=5.26 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=464.67ms avg_case_ms=4.65 avg_simplify_ms=1.34, difference@0+50 failed=0 elapsed=321.56ms avg_case_ms=6.43 avg_simplify_ms=2.04, sum@700+100 failed=0 elapsed=185.79ms avg_case_ms=1.86 avg_simplify_ms=0.69
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=11.28ms median_wire=11.32ms median_wall=42.85ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.28ms median_wire=13.36ms median_wall=52.13ms, difference@0+50 #174 difference runs=3 median_simplify=11.56ms median_wire=11.61ms median_wall=43.95ms, product@0+100 #175 product runs=3 median_simplify=11.51ms median_wire=11.56ms median_wall=44.39ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.89ms median_wire=10.97ms median_wall=40.67ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.26s | passed=450 failed=0 total=450 avg_case=5.022ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.11s | passed=1 failed=0 |
