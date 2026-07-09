# Engine Improvement Scorecard

- Generated: 2026-07-09T09:21:09.694116+00:00
- Git branch: main
- Git commit: `4f25c56bfea1550e2cdfc33418c36edad1b4b06f`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=936.46ms avg_case_ms=9.36 simplify=260.42ms avg_simplify_ms=2.60, sum total=200 failed=0 elapsed=828.02ms avg_case_ms=4.14 simplify=267.84ms avg_simplify_ms=1.34, product total=100 failed=0 elapsed=600.59ms avg_case_ms=6.01 simplify=170.06ms avg_simplify_ms=1.70, difference total=50 failed=0 elapsed=401.55ms avg_case_ms=8.03 simplify=121.05ms avg_simplify_ms=2.42
- Engine hotspots: sum simplify=267.84ms avg_simplify_ms=1.34 wall=828.02ms, shifted_quotient simplify=260.42ms avg_simplify_ms=2.60 wall=936.46ms, product simplify=170.06ms avg_simplify_ms=1.70 wall=600.59ms, difference simplify=121.05ms avg_simplify_ms=2.42 wall=401.55ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=936.46ms avg_case_ms=9.36 avg_simplify_ms=2.60, sum@0+100 failed=0 elapsed=601.20ms avg_case_ms=6.01 avg_simplify_ms=1.87, product@0+100 failed=0 elapsed=600.59ms avg_case_ms=6.01 avg_simplify_ms=1.70, difference@0+50 failed=0 elapsed=401.55ms avg_case_ms=8.03 avg_simplify_ms=2.42, sum@700+100 failed=0 elapsed=226.82ms avg_case_ms=2.27 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.21ms median_wire=16.28ms median_wall=63.03ms, product@0+100 #175 product runs=3 median_simplify=14.98ms median_wire=15.03ms median_wall=57.69ms, difference@0+50 #174 difference runs=3 median_simplify=14.89ms median_wire=14.94ms median_wall=57.05ms, sum@0+100 #173 sum runs=3 median_simplify=14.78ms median_wire=14.82ms median_wall=56.62ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.49ms median_wire=12.55ms median_wall=48.08ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.77s | passed=450 failed=0 total=450 avg_case=6.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.00s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.86s | passed=1 failed=0 |
