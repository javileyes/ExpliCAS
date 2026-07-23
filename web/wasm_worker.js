// ExpliCAS — WASM engine worker (frente W · W3).
//
// Runs the SAME stateless JSON wire the CLI and server speak, fully inside
// the browser: `pkg/` is the wasm-pack (--target web) build of the
// `cas_wasm` crate. Heavy evaluations run here, off the UI thread; the page
// talks to this worker via {id, expr, opts} messages and receives
// {id, wire} (or {id, error}) back.
//
// This file is a module worker: `new Worker('wasm_worker.js', {type: 'module'})`.

import init, { WasmSession, engine_version } from './pkg/cas_wasm.js';

let session = null;
let readyPromise = init().then(() => {
    session = new WasmSession();
    postMessage({ kind: 'ready', version: engine_version() });
});

onmessage = async (event) => {
    const { id, kind, expr, opts } = event.data;
    try {
        await readyPromise;
        if (kind === 'clear') {
            session.clear();
            postMessage({ kind: 'result', id, wire: '{"ok":true,"cleared":true}' });
            return;
        }
        // Session-backed eval: #N references and := assignments persist
        // for the lifetime of this tab (W6).
        const wire = session.eval(expr, opts || '{}');
        postMessage({ kind: 'result', id, wire });
    } catch (error) {
        postMessage({ kind: 'result', id, error: String(error) });
    }
};
