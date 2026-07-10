# Engine Improvement Scorecard

- Generated: 2026-07-10T22:58:47.360570+00:00
- Git branch: main
- Git commit: `5d3bd21885ff0855e019ffd1e2248c1723aaf5c0`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=934.24ms avg_case_ms=9.34 simplify=258.49ms avg_simplify_ms=2.58, sum total=200 failed=0 elapsed=846.42ms avg_case_ms=4.23 simplify=272.69ms avg_simplify_ms=1.36, product total=100 failed=0 elapsed=590.66ms avg_case_ms=5.91 simplify=168.09ms avg_simplify_ms=1.68, difference total=50 failed=0 elapsed=386.43ms avg_case_ms=7.73 simplify=117.30ms avg_simplify_ms=2.35
- Engine hotspots: sum simplify=272.69ms avg_simplify_ms=1.36 wall=846.42ms, shifted_quotient simplify=258.49ms avg_simplify_ms=2.58 wall=934.24ms, product simplify=168.09ms avg_simplify_ms=1.68 wall=590.66ms, difference simplify=117.30ms avg_simplify_ms=2.35 wall=386.43ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=934.24ms avg_case_ms=9.34 avg_simplify_ms=2.58, sum@0+100 failed=0 elapsed=618.49ms avg_case_ms=6.18 avg_simplify_ms=1.92, product@0+100 failed=0 elapsed=590.66ms avg_case_ms=5.91 avg_simplify_ms=1.68, difference@0+50 failed=0 elapsed=386.43ms avg_case_ms=7.73 avg_simplify_ms=2.35, sum@700+100 failed=0 elapsed=227.93ms avg_case_ms=2.28 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.22ms median_wire=16.30ms median_wall=64.14ms, sum@0+100 #173 sum runs=3 median_simplify=15.05ms median_wire=15.11ms median_wall=57.63ms, product@0+100 #175 product runs=3 median_simplify=14.98ms median_wire=15.03ms median_wall=57.14ms, difference@0+50 #174 difference runs=3 median_simplify=14.79ms median_wire=14.84ms median_wall=56.89ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.78ms median_wire=12.86ms median_wall=48.58ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.76s | passed=450 failed=0 total=450 avg_case=6.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.02s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.90s | passed=1 failed=0 |
