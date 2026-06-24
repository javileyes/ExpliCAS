# Engine Improvement Scorecard

- Generated: 2026-06-24T10:26:11.014887+00:00
- Git branch: main
- Git commit: `f44b4e16f15e38b3925151a3bbc36568bb1897ab`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=353

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=787.49ms avg_case_ms=7.87 simplify=226.84ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=703.30ms avg_case_ms=3.52 simplify=238.45ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=480.05ms avg_case_ms=4.80 simplify=138.71ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=347.93ms avg_case_ms=6.96 simplify=112.62ms avg_simplify_ms=2.25
- Engine hotspots: sum simplify=238.45ms avg_simplify_ms=1.19 wall=703.30ms, shifted_quotient simplify=226.84ms avg_simplify_ms=2.27 wall=787.49ms, product simplify=138.71ms avg_simplify_ms=1.39 wall=480.05ms, difference simplify=112.62ms avg_simplify_ms=2.25 wall=347.93ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=787.49ms avg_case_ms=7.87 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=510.96ms avg_case_ms=5.11 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=480.05ms avg_case_ms=4.80 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=347.93ms avg_case_ms=6.96 avg_simplify_ms=2.25, sum@700+100 failed=0 elapsed=192.34ms avg_case_ms=1.92 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.80ms median_wire=12.87ms median_wall=48.87ms, sum@0+100 #173 sum runs=3 median_simplify=11.98ms median_wire=12.03ms median_wall=45.07ms, difference@0+50 #174 difference runs=3 median_simplify=11.89ms median_wire=11.94ms median_wall=45.01ms, product@0+100 #175 product runs=3 median_simplify=11.71ms median_wire=11.77ms median_wall=44.20ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.69ms median_wire=10.76ms median_wall=40.10ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.42s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
