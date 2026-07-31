// ExpliCAS — WASM engine worker (frente W · W3).
//
// Runs the SAME stateless JSON wire the CLI and server speak, fully inside
// the browser: `pkg/` is the wasm-pack (--target web) build of the
// `cas_wasm` crate. Heavy evaluations run here, off the UI thread; the page
// talks to this worker via {id, expr, opts} messages and receives
// {id, wire} (or {id, error}) back.
//
// Stop-recovery contract: BEFORE each evaluation the worker posts a
// {kind: 'snapshot'} with the full serialized session (context + #N store +
// := environment). If the user stops a long computation the page TERMINATES
// this worker (BigInt work cannot be interrupted cooperatively), spawns a
// fresh one and sends {kind: 'restore'} with the last snapshot — so #N
// references and := bindings survive the stop.
//
// This file is a module worker: `new Worker('wasm_worker.js', {type: 'module'})`.

import init, { WasmSession, engine_version } from './pkg/cas_wasm.js';

let session = null;
let readyPromise = init().then(() => {
    session = new WasmSession();
    postMessage({ kind: 'ready', version: engine_version() });
});

onmessage = async (event) => {
    const { id, kind, expr, opts, snapshot } = event.data;
    try {
        await readyPromise;
        if (kind === 'clear') {
            session.clear();
            postMessage({ kind: 'result', id, wire: '{"ok":true,"cleared":true}' });
            return;
        }
        if (kind === 'restore') {
            const ok = snapshot ? session.restore(snapshot) : false;
            postMessage({ kind: 'result', id, wire: JSON.stringify({ ok, restored: ok }) });
            return;
        }
        // Pre-eval snapshot: the page keeps the latest one as the recovery
        // point for a user stop. Transferred (not copied) — the worker has
        // no further use for it.
        try {
            const snap = session.snapshot();
            if (snap) {
                postMessage({ kind: 'snapshot', snapshot: snap }, [snap.buffer]);
            }
        } catch (error) {
            console.warn('ExpliCAS worker: pre-eval snapshot failed', error);
        }
        // Session-backed eval: #N references and := assignments persist
        // for the lifetime of this tab (W6).
        const wire = session.eval(expr, opts || '{}');
        postMessage({ kind: 'result', id, wire });
    } catch (error) {
        postMessage({ kind: 'result', id, error: String(error) });
    }
};
