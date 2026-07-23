# Engine Improvement Scorecard

- Generated: 2026-07-23T16:01:48.803780+00:00
- Git branch: main
- Git commit: `3fea0df0ba5e1c1cf7f037d97efc0ebf8c3509d1`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.01s avg_case_ms=10.10 simplify=284.23ms avg_simplify_ms=2.84, sum total=200 failed=0 elapsed=897.41ms avg_case_ms=4.49 simplify=295.30ms avg_simplify_ms=1.48, product total=100 failed=0 elapsed=620.80ms avg_case_ms=6.21 simplify=178.64ms avg_simplify_ms=1.79, difference total=50 failed=0 elapsed=411.81ms avg_case_ms=8.24 simplify=126.24ms avg_simplify_ms=2.52
- Engine hotspots: sum simplify=295.30ms avg_simplify_ms=1.48 wall=897.41ms, shifted_quotient simplify=284.23ms avg_simplify_ms=2.84 wall=1.01s, product simplify=178.64ms avg_simplify_ms=1.79 wall=620.80ms, difference simplify=126.24ms avg_simplify_ms=2.52 wall=411.81ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.01s avg_case_ms=10.10 avg_simplify_ms=2.84, sum@0+100 failed=0 elapsed=660.69ms avg_case_ms=6.61 avg_simplify_ms=2.10, product@0+100 failed=0 elapsed=620.80ms avg_case_ms=6.21 avg_simplify_ms=1.79, difference@0+50 failed=0 elapsed=411.81ms avg_case_ms=8.24 avg_simplify_ms=2.52, sum@700+100 failed=0 elapsed=236.71ms avg_case_ms=2.37 avg_simplify_ms=0.85
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.12ms median_wire=17.19ms median_wall=65.12ms, sum@0+100 #173 sum runs=3 median_simplify=15.34ms median_wire=15.39ms median_wall=58.78ms, difference@0+50 #174 difference runs=3 median_simplify=15.47ms median_wire=15.52ms median_wall=58.96ms, product@0+100 #175 product runs=3 median_simplify=15.76ms median_wire=15.82ms median_wall=59.94ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.16ms median_wire=13.23ms median_wall=50.54ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.94s | passed=450 failed=0 total=450 avg_case=6.533ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.77s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.99s | passed=1 failed=0 |
