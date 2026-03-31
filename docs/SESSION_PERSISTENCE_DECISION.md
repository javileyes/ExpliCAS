# Session Persistence Decision

## Decision

Keep the current default snapshot persistence strategy:

- `write tmp`
- `flush`
- `rename`

Do **not** change the default session save path to either:

- direct overwrite / non-atomic write
- stronger durability with `sync_all`

Do **not** expose an alternate persistence mode/config at this time.

## Current Semantics

The current helper path is implemented in:

- [snapshot_io.rs](/Users/javiergimenezmoya/developer/math/crates/cas_session_core/src/snapshot_io.rs)

The retained default is:

- atomic replacement / visibility-oriented persistence
- not full crash-durable persistence

Specifically:

- the tmp file is written and flushed
- then renamed into place
- there is **no** `sync_all` on the tmp file
- there is **no** parent-directory sync

So the current behavior should be described as:

- `atomic replace`
- not `fully durable crash-consistent persistence`

## Why This Is The Decision

The persistence subtrack is already measured and the tradeoff is now clear.

### 1. Keeping the current default preserves the best current balance

Measured product-path overwrite saves land around the same band as the lower
level atomic overwrite path:

- frontend overwrite after mutation: about `~225-240 us`
- core atomic overwrite: about `~227-234 us`

That makes the current default cheap enough for normal session saves while
still keeping atomic replacement semantics.

### 2. A faster non-atomic path is not compelling enough yet

The practical lower bound for direct overwrite is roughly:

- `~123-128 us`

So trading away atomic replacement would save on the order of:

- `~100 us`

That is real, but not enough by itself to justify weakening default save
semantics without a product requirement that says latency matters more than
replacement safety.

### 3. A stronger durability mode is much more expensive

The measured lower bound for syncing the tmp file before rename is already:

- `~5.2-6.0 ms`

That is an order-of-magnitude jump from the current overwrite cost, before any
parent-directory sync is added.

So a stronger durability mode is not a “small safety upgrade”; it is a
different product tradeoff.

## Product Policy

Retained product policy:

1. Default session persistence remains `atomic replace`.
2. The current implementation must not be described as fully durable.
3. No alternate persistence mode is added until a concrete product use case
   requires it.

## Revisit Conditions

Reopen this decision only if one of these becomes true:

1. Session save latency becomes a proven user-facing bottleneck.
2. A product requirement demands stronger durability guarantees.
3. A new persistence design offers materially better latency/safety tradeoffs
   than the current `write + flush + rename` path.

## If Reopened Later

Any future alternate mode should be explicit and opt-in, not silent.

That future track should include:

1. naming the mode by semantics, not by implementation detail
2. explicit contract tests for the chosen behavior
3. benchmark baselines against the current default
4. a product-level reason for preferring that tradeoff
