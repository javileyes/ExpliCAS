# Engine Improvement Scorecard

- Generated: 2026-07-08T10:15:43.013771+00:00
- Git branch: main
- Git commit: `2215b0304a02b21c971e2396f88583d70f8dfeb2`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=10
- By area: calculus / integration:4, calculus / runtime:3, calculus / differentiation:2, calculus / robustness:1
- Recent 1: `calculus / integration` - 2026-06-08 - Discovery observe-only: polynomial cosecant/cotangent source-return still emits depth pressure
- Recent 2: `calculus / differentiation` - 2026-06-06 - Observe-only discovery: exact-square atanh scaled-root runtime is not caused by the global empty-domain check
- Recent 3: `calculus / differentiation` - 2026-06-06 - Observe-only discovery: exact-square inverse-root diff runtime is not fixed by raw target preservation

## Calculus Support Matrix Signal

- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.
- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.
- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.
- `diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=263
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=972.04ms avg_case_ms=9.72 simplify=270.16ms avg_simplify_ms=2.70, sum total=200 failed=0 elapsed=829.07ms avg_case_ms=4.15 simplify=267.61ms avg_simplify_ms=1.34, product total=100 failed=0 elapsed=605.74ms avg_case_ms=6.06 simplify=172.67ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=395.24ms avg_case_ms=7.90 simplify=120.64ms avg_simplify_ms=2.41
- Engine hotspots: shifted_quotient simplify=270.16ms avg_simplify_ms=2.70 wall=972.04ms, sum simplify=267.61ms avg_simplify_ms=1.34 wall=829.07ms, product simplify=172.67ms avg_simplify_ms=1.73 wall=605.74ms, difference simplify=120.64ms avg_simplify_ms=2.41 wall=395.24ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=972.04ms avg_case_ms=9.72 avg_simplify_ms=2.70, sum@0+100 failed=0 elapsed=607.78ms avg_case_ms=6.08 avg_simplify_ms=1.89, product@0+100 failed=0 elapsed=605.74ms avg_case_ms=6.06 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=395.24ms avg_case_ms=7.90 avg_simplify_ms=2.41, sum@700+100 failed=0 elapsed=221.29ms avg_case_ms=2.21 avg_simplify_ms=0.79
- Steady-state engine reruns: difference@0+50 #174 difference runs=3 median_simplify=15.43ms median_wire=15.47ms median_wall=58.69ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.02ms median_wire=17.09ms median_wall=65.66ms, product@0+100 #175 product runs=3 median_simplify=15.29ms median_wire=15.33ms median_wall=58.69ms, sum@0+100 #173 sum runs=3 median_simplify=15.07ms median_wire=15.11ms median_wall=56.81ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.13ms median_wire=12.21ms median_wall=46.96ms
- Steady-state dominant expressions: difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.80s | passed=450 failed=0 total=450 avg_case=6.222ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.99s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.85s | passed=1 failed=0 |
