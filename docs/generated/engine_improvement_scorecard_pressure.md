# Engine Improvement Scorecard

- Generated: 2026-07-23T03:32:00.998252+00:00
- Git branch: main
- Git commit: `ce968c79388e9cdff31db3f049c9835e2e6fea14`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=972.98ms avg_case_ms=9.73 simplify=273.11ms avg_simplify_ms=2.73, sum total=200 failed=0 elapsed=874.08ms avg_case_ms=4.37 simplify=282.43ms avg_simplify_ms=1.41, product total=100 failed=0 elapsed=600.04ms avg_case_ms=6.00 simplify=173.26ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=398.25ms avg_case_ms=7.97 simplify=121.86ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=282.43ms avg_simplify_ms=1.41 wall=874.08ms, shifted_quotient simplify=273.11ms avg_simplify_ms=2.73 wall=972.98ms, product simplify=173.26ms avg_simplify_ms=1.73 wall=600.04ms, difference simplify=121.86ms avg_simplify_ms=2.44 wall=398.25ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=972.98ms avg_case_ms=9.73 avg_simplify_ms=2.73, sum@0+100 failed=0 elapsed=643.34ms avg_case_ms=6.43 avg_simplify_ms=1.99, product@0+100 failed=0 elapsed=600.04ms avg_case_ms=6.00 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=398.25ms avg_case_ms=7.97 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=230.74ms avg_case_ms=2.31 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.57ms median_wire=16.63ms median_wall=63.55ms, sum@0+100 #173 sum runs=3 median_simplify=15.02ms median_wire=15.07ms median_wall=58.24ms, difference@0+50 #174 difference runs=3 median_simplify=15.17ms median_wire=15.22ms median_wall=57.74ms, product@0+100 #175 product runs=3 median_simplify=17.29ms median_wire=17.35ms median_wall=60.95ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.80ms median_wire=12.87ms median_wall=48.82ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.85s | passed=450 failed=0 total=450 avg_case=6.333ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.56s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
