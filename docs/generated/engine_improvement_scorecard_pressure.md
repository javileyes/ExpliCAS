# Engine Improvement Scorecard

- Generated: 2026-07-19T08:22:50.726613+00:00
- Git branch: main
- Git commit: `490c33b5fe311d8c9182357cb1bdf8914c7b1362`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=367

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.02s avg_case_ms=10.19 simplify=289.38ms avg_simplify_ms=2.89, sum total=200 failed=0 elapsed=915.52ms avg_case_ms=4.58 simplify=297.42ms avg_simplify_ms=1.49, product total=100 failed=0 elapsed=642.24ms avg_case_ms=6.42 simplify=185.88ms avg_simplify_ms=1.86, difference total=50 failed=0 elapsed=416.09ms avg_case_ms=8.32 simplify=128.53ms avg_simplify_ms=2.57
- Engine hotspots: sum simplify=297.42ms avg_simplify_ms=1.49 wall=915.52ms, shifted_quotient simplify=289.38ms avg_simplify_ms=2.89 wall=1.02s, product simplify=185.88ms avg_simplify_ms=1.86 wall=642.24ms, difference simplify=128.53ms avg_simplify_ms=2.57 wall=416.09ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.02s avg_case_ms=10.19 avg_simplify_ms=2.89, sum@0+100 failed=0 elapsed=673.35ms avg_case_ms=6.73 avg_simplify_ms=2.10, product@0+100 failed=0 elapsed=642.24ms avg_case_ms=6.42 avg_simplify_ms=1.86, difference@0+50 failed=0 elapsed=416.09ms avg_case_ms=8.32 avg_simplify_ms=2.57, sum@700+100 failed=0 elapsed=242.17ms avg_case_ms=2.42 avg_simplify_ms=0.88
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.91ms median_wire=17.98ms median_wall=67.31ms, sum@0+100 #173 sum runs=3 median_simplify=15.39ms median_wire=15.43ms median_wall=58.62ms, product@0+100 #175 product runs=3 median_simplify=16.07ms median_wire=16.12ms median_wall=61.45ms, difference@0+50 #174 difference runs=3 median_simplify=15.42ms median_wire=15.47ms median_wall=58.88ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.19ms median_wire=13.27ms median_wall=49.42ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.99s | passed=450 failed=0 total=450 avg_case=6.644ms |
| `calculus_diff_exhaustive_contract` | `pass` | 13.11s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.97s | passed=1 failed=0 |
