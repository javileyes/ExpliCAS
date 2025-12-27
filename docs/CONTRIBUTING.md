# Contributing to ExpliCAS

## Quick Start

1. Fork and clone the repository
2. Run `make ci` to verify your setup
3. Create a feature branch
4. Make changes following the guidelines below
5. Run `make ci` before committing
6. Open a PR

## Code Style

- Run `cargo fmt` before committing
- All clippy warnings must be addressed (see `POLICY.md` for allow exceptions)
- Follow existing patterns in the codebase

## Testing

- Add tests for new functionality
- Run `cargo test -p cas_engine` for engine tests
- Run `make ci` for full validation including lints

---

## Semantic-Axis Dependent Rules Checklist

When adding or modifying a rule that depends on any of these semantic axes:
- `DomainMode` (strict | generic | assume)
- `ValueDomain` (real | complex)
- `BranchPolicy` (principal)
- `InverseTrigPolicy` (strict | principal)
- `ConstFoldMode` (off | safe)

**You MUST complete this checklist:**

### 1. Propagation
- [ ] Rule reads the axis from `parent_ctx` (not hardcoded)
- [ ] If creating sub-context, uses `extend()` to preserve axes
- [ ] Test: child rule sees correct axis inside nested expression

### 2. Trazability
- [ ] `Step.domain_assumption` documents assumptions (if any)
- [ ] `EngineJsonWarning` emitted if result depends on non-universal convention
- [ ] Messages are user-readable: `"assuming x ≠ 0 for x/x → 1"`

### 3. Mode Contracts
- [ ] Test: `axis=ValueA → result X`
- [ ] Test: `axis=ValueB → result Y` (or residual)
- [ ] Test: `axis=Off/Strict → no transformation` (if applicable)

### 4. Isolation
- [ ] Rule is in the correct module:
  - `rules/*` → symbolic simplification
  - `const_eval` → neutral, always-on
  - `const_fold` → semantic, gated
- [ ] Lint enforcement updated (if touching `const_fold`)

### 5. Budget
- [ ] Rule charges to appropriate `Operation::*`
- [ ] No infinite loops with reasonable budget

---

## Documentation

- Update `docs/` if adding new features
- Reference `POLICY.md` for canonical utility patterns
- Reference `docs/SEMANTICS_POLICY.md` for semantic axis definitions

## Questions?

Open an issue or check existing policy documents in `docs/`.
