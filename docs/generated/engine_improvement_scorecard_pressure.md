# Engine Improvement Scorecard

- Generated: 2026-07-10T09:40:07.779794+00:00
- Git branch: main
- Git commit: `d591eb8103905d17daabe3228375ccdcaccc5b85`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=931.98ms avg_case_ms=9.32 simplify=257.51ms avg_simplify_ms=2.58, sum total=200 failed=0 elapsed=834.57ms avg_case_ms=4.17 simplify=269.40ms avg_simplify_ms=1.35, product total=100 failed=0 elapsed=589.65ms avg_case_ms=5.90 simplify=168.69ms avg_simplify_ms=1.69, difference total=50 failed=0 elapsed=386.46ms avg_case_ms=7.73 simplify=116.89ms avg_simplify_ms=2.34
- Engine hotspots: sum simplify=269.40ms avg_simplify_ms=1.35 wall=834.57ms, shifted_quotient simplify=257.51ms avg_simplify_ms=2.58 wall=931.98ms, product simplify=168.69ms avg_simplify_ms=1.69 wall=589.65ms, difference simplify=116.89ms avg_simplify_ms=2.34 wall=386.46ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=931.98ms avg_case_ms=9.32 avg_simplify_ms=2.58, sum@0+100 failed=0 elapsed=610.82ms avg_case_ms=6.11 avg_simplify_ms=1.89, product@0+100 failed=0 elapsed=589.65ms avg_case_ms=5.90 avg_simplify_ms=1.69, difference@0+50 failed=0 elapsed=386.46ms avg_case_ms=7.73 avg_simplify_ms=2.34, sum@700+100 failed=0 elapsed=223.75ms avg_case_ms=2.24 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.21ms median_wire=17.27ms median_wall=64.55ms, product@0+100 #175 product runs=3 median_simplify=14.72ms median_wire=14.77ms median_wall=56.89ms, difference@0+50 #174 difference runs=3 median_simplify=14.51ms median_wire=14.55ms median_wall=55.62ms, sum@0+100 #173 sum runs=3 median_simplify=14.89ms median_wire=14.94ms median_wall=57.27ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.01ms median_wire=12.07ms median_wall=46.54ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.74s | passed=450 failed=0 total=450 avg_case=6.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.98s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.86s | passed=1 failed=0 |
