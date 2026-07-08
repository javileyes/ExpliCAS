# Engine Improvement Scorecard

- Generated: 2026-07-08T10:53:57.857633+00:00
- Git branch: main
- Git commit: `0f87ac113f92d258a650d7c7d877081ff654b919`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=937.55ms avg_case_ms=9.38 simplify=261.06ms avg_simplify_ms=2.61, sum total=200 failed=0 elapsed=826.47ms avg_case_ms=4.13 simplify=266.39ms avg_simplify_ms=1.33, product total=100 failed=0 elapsed=583.04ms avg_case_ms=5.83 simplify=165.30ms avg_simplify_ms=1.65, difference total=50 failed=0 elapsed=387.21ms avg_case_ms=7.74 simplify=117.02ms avg_simplify_ms=2.34
- Engine hotspots: sum simplify=266.39ms avg_simplify_ms=1.33 wall=826.47ms, shifted_quotient simplify=261.06ms avg_simplify_ms=2.61 wall=937.55ms, product simplify=165.30ms avg_simplify_ms=1.65 wall=583.04ms, difference simplify=117.02ms avg_simplify_ms=2.34 wall=387.21ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=937.55ms avg_case_ms=9.38 avg_simplify_ms=2.61, sum@0+100 failed=0 elapsed=601.62ms avg_case_ms=6.02 avg_simplify_ms=1.86, product@0+100 failed=0 elapsed=583.04ms avg_case_ms=5.83 avg_simplify_ms=1.65, difference@0+50 failed=0 elapsed=387.21ms avg_case_ms=7.74 avg_simplify_ms=2.34, sum@700+100 failed=0 elapsed=224.85ms avg_case_ms=2.25 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.27ms median_wire=16.33ms median_wall=62.50ms, difference@0+50 #174 difference runs=3 median_simplify=15.10ms median_wire=15.14ms median_wall=57.85ms, product@0+100 #175 product runs=3 median_simplify=14.64ms median_wire=14.69ms median_wall=55.73ms, sum@0+100 #173 sum runs=3 median_simplify=14.77ms median_wire=14.82ms median_wall=56.10ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.39ms median_wire=12.46ms median_wall=46.95ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.73s | passed=450 failed=0 total=450 avg_case=6.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.96s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.85s | passed=1 failed=0 |
