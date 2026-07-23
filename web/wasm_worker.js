// ExpliCAS — WASM engine worker (frente W · W3).
//
// Runs the SAME stateless JSON wire the CLI and server speak, fully inside
// the browser: `pkg/` is the wasm-pack (--target web) build of the
// `cas_wasm` crate. Heavy evaluations run here, off the UI thread; the page
// talks to this worker via {id, expr, opts} messages and receives
// {id, wire} (or {id, error}) back.
//
// This file is a module worker: `new Worker('wasm_worker.js', {type: 'module'})`.

import init, { eval_str_to_wire, engine_version } from './pkg/cas_wasm.js';

let readyPromise = init().then(() => {
    postMessage({ kind: 'ready', version: engine_version() });
});

onmessage = async (event) => {
    const { id, expr, opts } = event.data;
    try {
        await readyPromise;
        const wire = eval_str_to_wire(expr, opts || '{}');
        postMessage({ kind: 'result', id, wire });
    } catch (error) {
        postMessage({ kind: 'result', id, error: String(error) });
    }
};
