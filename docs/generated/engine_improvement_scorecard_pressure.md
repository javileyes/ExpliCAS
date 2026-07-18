# Engine Improvement Scorecard

- Generated: 2026-07-18T20:10:21.410395+00:00
- Git branch: main
- Git commit: `e58ac62d415221ca600f149742028ec8464aa3e1`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=988.97ms avg_case_ms=9.89 simplify=276.22ms avg_simplify_ms=2.76, sum total=200 failed=0 elapsed=897.31ms avg_case_ms=4.49 simplify=286.81ms avg_simplify_ms=1.43, product total=100 failed=0 elapsed=604.97ms avg_case_ms=6.05 simplify=173.67ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=400.29ms avg_case_ms=8.01 simplify=121.86ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=286.81ms avg_simplify_ms=1.43 wall=897.31ms, shifted_quotient simplify=276.22ms avg_simplify_ms=2.76 wall=988.97ms, product simplify=173.67ms avg_simplify_ms=1.74 wall=604.97ms, difference simplify=121.86ms avg_simplify_ms=2.44 wall=400.29ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=988.97ms avg_case_ms=9.89 avg_simplify_ms=2.76, sum@0+100 failed=0 elapsed=664.14ms avg_case_ms=6.64 avg_simplify_ms=2.04, product@0+100 failed=0 elapsed=604.97ms avg_case_ms=6.05 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=400.29ms avg_case_ms=8.01 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=233.17ms avg_case_ms=2.33 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.09ms median_wire=17.16ms median_wall=65.47ms, sum@0+100 #173 sum runs=3 median_simplify=15.06ms median_wire=15.11ms median_wall=57.55ms, product@0+100 #175 product runs=3 median_simplify=15.15ms median_wire=15.20ms median_wall=60.74ms, difference@0+50 #174 difference runs=3 median_simplify=16.10ms median_wire=16.15ms median_wall=65.26ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.12ms median_wire=13.21ms median_wall=50.23ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.89s | passed=450 failed=0 total=450 avg_case=6.422ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
