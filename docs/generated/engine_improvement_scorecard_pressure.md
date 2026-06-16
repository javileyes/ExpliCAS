# Engine Improvement Scorecard

- Generated: 2026-06-16T08:57:35.169876+00:00
- Git branch: main
- Git commit: `1a25c39aaf763462f228b88d23799aa1b0474641`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=805.44ms avg_case_ms=8.05 simplify=230.41ms avg_simplify_ms=2.30, sum total=200 failed=0 elapsed=697.63ms avg_case_ms=3.49 simplify=235.46ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=483.21ms avg_case_ms=4.83 simplify=139.44ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=333.34ms avg_case_ms=6.67 simplify=106.20ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=235.46ms avg_simplify_ms=1.18 wall=697.63ms, shifted_quotient simplify=230.41ms avg_simplify_ms=2.30 wall=805.44ms, product simplify=139.44ms avg_simplify_ms=1.39 wall=483.21ms, difference simplify=106.20ms avg_simplify_ms=2.12 wall=333.34ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=805.44ms avg_case_ms=8.05 avg_simplify_ms=2.30, sum@0+100 failed=0 elapsed=504.08ms avg_case_ms=5.04 avg_simplify_ms=1.63, product@0+100 failed=0 elapsed=483.21ms avg_case_ms=4.83 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=333.34ms avg_case_ms=6.67 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=193.54ms avg_case_ms=1.94 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.21ms median_wire=13.29ms median_wall=50.38ms, sum@0+100 #173 sum runs=3 median_simplify=11.80ms median_wire=11.85ms median_wall=44.98ms, difference@0+50 #174 difference runs=3 median_simplify=12.96ms median_wire=13.02ms median_wall=48.84ms, product@0+100 #175 product runs=3 median_simplify=12.05ms median_wire=12.10ms median_wall=45.54ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.74ms median_wire=10.82ms median_wall=40.63ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.89s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.97s | passed=1 failed=0 |
