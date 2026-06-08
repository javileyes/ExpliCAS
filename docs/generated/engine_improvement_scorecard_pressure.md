# Engine Improvement Scorecard

- Generated: 2026-06-08T01:02:06.395427+00:00
- Git branch: main
- Git commit: `0bc3bc31c8b14c2b497382abbe16294639e7ef4a`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: sum total=200 failed=0 elapsed=841.81ms avg_case_ms=4.21 simplify=281.37ms avg_simplify_ms=1.41, shifted_quotient total=100 failed=0 elapsed=835.35ms avg_case_ms=8.35 simplify=238.54ms avg_simplify_ms=2.39, product total=100 failed=0 elapsed=502.39ms avg_case_ms=5.02 simplify=146.62ms avg_simplify_ms=1.47, difference total=50 failed=0 elapsed=351.96ms avg_case_ms=7.04 simplify=112.86ms avg_simplify_ms=2.26
- Engine hotspots: sum simplify=281.37ms avg_simplify_ms=1.41 wall=841.81ms, shifted_quotient simplify=238.54ms avg_simplify_ms=2.39 wall=835.35ms, product simplify=146.62ms avg_simplify_ms=1.47 wall=502.39ms, difference simplify=112.86ms avg_simplify_ms=2.26 wall=351.96ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=835.35ms avg_case_ms=8.35 avg_simplify_ms=2.39, sum@0+100 failed=0 elapsed=627.30ms avg_case_ms=6.27 avg_simplify_ms=1.97, product@0+100 failed=0 elapsed=502.39ms avg_case_ms=5.02 avg_simplify_ms=1.47, difference@0+50 failed=0 elapsed=351.96ms avg_case_ms=7.04 avg_simplify_ms=2.26, sum@700+100 failed=0 elapsed=214.51ms avg_case_ms=2.15 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.52ms median_wire=13.59ms median_wall=53.16ms, difference@0+50 #174 difference runs=3 median_simplify=12.06ms median_wire=12.11ms median_wall=46.55ms, product@0+100 #175 product runs=3 median_simplify=12.10ms median_wire=12.16ms median_wall=46.04ms, sum@0+100 #173 sum runs=3 median_simplify=12.06ms median_wire=12.11ms median_wall=45.17ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.18ms median_wire=12.26ms median_wall=49.57ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.53s | passed=450 failed=0 total=450 avg_case=5.622ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.10s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 18.59s | passed=1 failed=0 |
