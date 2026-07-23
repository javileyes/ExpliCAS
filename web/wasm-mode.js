// ExpliCAS — dual-mode engine shim (frente W · W3).
//
// Activated ONLY when the build config says so (the GitHub Pages build sets
// `engineMode: "wasm"`); the classic server deployment keeps `"server"` (or
// omits the key) and this file is inert — the server route is never lost.
//
// In wasm mode the page's callAPI routes evaluations through a module
// Worker running the `cas_wasm` engine build — the same JSON wire, zero
// network, all computation local to the user's browser.
//
// Honest limits of wasm mode today (named, not silent — see cas_wasm
// doc-comment): the stateless wire has no session state (#N references)
// and no per-request language/domain/value-domain/numeric-display flags
// yet; those selectors are disabled with an explanatory tooltip.

(function () {
    'use strict';

    const config = window.EXPLICAS_BUILD_CONFIG || {};
    if (config.engineMode !== 'wasm') {
        window.EXPLICAS_WASM = { enabled: false, ready: false };
        return;
    }

    const state = {
        enabled: true,
        ready: false,
        version: null,
        worker: null,
        pending: new Map(),
        nextId: 1,
    };
    window.EXPLICAS_WASM = state;

    try {
        state.worker = new Worker('wasm_worker.js', { type: 'module' });
    } catch (error) {
        console.error('ExpliCAS wasm mode: worker failed to start', error);
        state.enabled = false;
        return;
    }

    state.worker.onmessage = (event) => {
        const msg = event.data;
        if (msg.kind === 'ready') {
            state.ready = true;
            state.version = msg.version;
            document.dispatchEvent(new CustomEvent('explicas-wasm-ready', {
                detail: { version: msg.version },
            }));
            return;
        }
        if (msg.kind === 'result') {
            const entry = state.pending.get(msg.id);
            if (!entry) return;
            state.pending.delete(msg.id);
            if (msg.error) {
                entry.reject(new Error(msg.error));
            } else {
                entry.resolve(msg.wire);
            }
        }
    };

    state.worker.onerror = (event) => {
        console.error('ExpliCAS wasm worker error', event);
    };

    // Evaluate `expr` on the local engine; resolves to the wire JSON string
    // (EngineWireResponse schema v1 — same payload the server proxies).
    state.eval = function (expr, optsJson) {
        return new Promise((resolve, reject) => {
            const id = state.nextId++;
            state.pending.set(id, { resolve, reject });
            state.worker.postMessage({ id, expr, opts: optsJson || '{}' });
        });
    };
})();
