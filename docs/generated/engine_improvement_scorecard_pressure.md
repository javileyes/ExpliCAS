# Engine Improvement Scorecard

- Generated: 2026-06-09T10:08:38.234399+00:00
- Git branch: main
- Git commit: `ec3b9a1a314b2fe941833b716b02f7c006892890`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=818.47ms avg_case_ms=8.18 simplify=234.72ms avg_simplify_ms=2.35, sum total=200 failed=0 elapsed=733.29ms avg_case_ms=3.67 simplify=245.24ms avg_simplify_ms=1.23, product total=100 failed=0 elapsed=493.68ms avg_case_ms=4.94 simplify=141.57ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=338.41ms avg_case_ms=6.77 simplify=107.92ms avg_simplify_ms=2.16
- Engine hotspots: sum simplify=245.24ms avg_simplify_ms=1.23 wall=733.29ms, shifted_quotient simplify=234.72ms avg_simplify_ms=2.35 wall=818.47ms, product simplify=141.57ms avg_simplify_ms=1.42 wall=493.68ms, difference simplify=107.92ms avg_simplify_ms=2.16 wall=338.41ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=818.47ms avg_case_ms=8.18 avg_simplify_ms=2.35, sum@0+100 failed=0 elapsed=534.82ms avg_case_ms=5.35 avg_simplify_ms=1.71, product@0+100 failed=0 elapsed=493.68ms avg_case_ms=4.94 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=338.41ms avg_case_ms=6.77 avg_simplify_ms=2.16, sum@700+100 failed=0 elapsed=198.47ms avg_case_ms=1.98 avg_simplify_ms=0.74
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.41ms median_wire=13.48ms median_wall=51.06ms, difference@0+50 #174 difference runs=3 median_simplify=11.90ms median_wire=11.96ms median_wall=45.52ms, product@0+100 #175 product runs=3 median_simplify=12.63ms median_wire=12.69ms median_wall=49.08ms, sum@0+100 #173 sum runs=3 median_simplify=12.88ms median_wire=12.93ms median_wall=51.05ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.11ms median_wire=11.19ms median_wall=42.13ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.38s | passed=450 failed=0 total=450 avg_case=5.289ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.91s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 18.26s | passed=1 failed=0 |
